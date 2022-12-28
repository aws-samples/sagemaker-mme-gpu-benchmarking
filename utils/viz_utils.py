import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from collections import defaultdict
from pathlib import Path
from typing import Union


def get_summary_results(locust_result_path: Union[str, Path]) -> pd.DataFrame:

    locust_result_path = Path(locust_result_path)
    model_name = locust_result_path.name

    history_files = list(locust_result_path.glob("*_stats_history.csv"))

    load_test_summaries = []

    for file in history_files:
        df = pd.read_csv(file)
        df = df.query("Name != 'Aggregated'").copy()
        user_counts = list(range(20, 220, 20))

        df["Total Requests"] = df["Total Request Count"].diff(1)
        df["Failed Requests"] = df["Total Failure Count"].diff(1)
        df["Successful Requests"] = df["Total Requests"] - df["Failed Requests"]

        metric_cols = [
            "User Count",
            "Requests/s",
            "Total Median Response Time",
            "Total Average Response Time",
            "Total Min Response Time",
            "Total Max Response Time",
            "Successful Requests",
            "Failed Requests",
        ]
        aggregations = dict(
            (col, ["mean"])
            if col not in ("Successful Requests", "Failed Requests")
            else (col, ["sum"])
            for col in metric_cols
        )

        load_test_summary = (
            df.query("`User Count` in @user_counts")[metric_cols]
            .groupby("User Count", as_index=False)
            .agg(aggregations)
        )
        load_test_summary.columns = metric_cols

        instance_type_platform = file.name.split("_")[0]
        instance_type, platform, max_models_loaded = instance_type_platform.split("*")

        load_test_summary.insert(
            loc=0, column="max_models_loaded", value=int(max_models_loaded)
        )
        load_test_summary.insert(loc=0, column="platform", value=platform)
        load_test_summary.insert(loc=0, column="instance_type", value=instance_type)
        load_test_summary.insert(loc=0, column="model_name", value=model_name)

        load_test_summaries.append(load_test_summary)

        result = pd.concat(load_test_summaries)

        result.to_csv(locust_result_path / "summary_results.csv", index=False)

    return result


def generate_summary_plots(load_test_summary: pd.DataFrame):
    user_counts = load_test_summary["User Count"].astype(int).unique().tolist()
    model_names_instance_types = (
        load_test_summary[["model_name", "instance_type", "platform"]]
        .drop_duplicates()
        .values.tolist()
    )

    figures = defaultdict(dict)
    #

    for model_name, instance_type, platform in model_names_instance_types:
        fig, axs = plt.subplots(
            nrows=2, figsize=(10, 6), gridspec_kw={"height_ratios": [4, 1]}, sharex=True
        )
        ax = axs[0]

        df = load_test_summary.query(
            f"model_name == '{model_name}' & instance_type == '{instance_type}' & platform == '{platform}' "
        )

        sns.pointplot(
            x=df["User Count"].astype(int),
            y=df["Requests/s"],
            color="#8ecae6",
            ax=ax,
            markers="s",
            scale=0.7,
        )
        ax.set_xticks(user_counts)
        ax.set_ylim(df["Requests/s"].min() * 0.8, df["Requests/s"].max() * 1.1)

        ax2 = ax.twinx()
        sns.pointplot(
            x=df["User Count"].astype(int),
            y=df["Total Average Response Time"],
            color="#ffb703",
            markers="s",
            ax=ax2,
            scale=0.7,
        )

        ax.set_title(f"{model_name} {platform} platform on {instance_type}")

        [
            ax.text(
                x=p[0] - 0.25, y=p[1] + 1, s=f"{p[1]:.2f}", color="white"
            ).set_backgroundcolor("#8ecae6")
            for p in zip(ax.get_xticks(), df["Requests/s"])
        ]
        [
            ax2.text(
                x=p[0] - 0.25, y=p[1] - 25, s=f"{p[1]:.2f}", color="white"
            ).set_backgroundcolor("#ffb703")
            for p in zip(ax.get_xticks(), df["Total Average Response Time"])
        ]

        ax2.set_ylabel("Average Response Time (ms)")
        ax.legend(labels=["Requests/s"], loc=(0.55, 0.01), fontsize=10, frameon=False)
        ax2.legend(
            labels=["Average Response Time"],
            loc=(0.74, 0.01),
            fontsize=10,
            frameon=False,
        )

        sns.barplot(
            x=df["User Count"].astype(int),
            y=df["Successful Requests"],
            color="#43aa8b",
            ax=axs[1],
            label="Successful Requests",
        )
        sns.barplot(
            x=df["User Count"].astype(int),
            y=df["Failed Requests"],
            color="#f94144",
            ax=axs[1],
            label="Failed Requests",
        )
        axs[1].set_ylabel("# of Requests")
        axs[1].legend(loc=(0.55, 1), ncol=2, frameon=False)

        plt.tight_layout()

        figures[model_name][f"{instance_type}*{platform}"] = fig

    return figures


def generate_metrics_summary(load_test_summary: pd.DataFrame, instance_type: str):

    platform_summary = (
        load_test_summary.query(f"`Failed Requests` == 0  & instance_type == '{instance_type}'")[
            [
                "platform",
                "max_models_loaded",
                "Total Average Response Time",
                "Requests/s",
                "User Count",
            ]
        ]
        .groupby("platform")
        .agg(
            {
                "max_models_loaded": "max",
                "Total Average Response Time": "min",
                "Requests/s": "max",
                "User Count": "max",
            }
        )
    )

    model_name = load_test_summary["model_name"].iloc[0]
    scaled_summary = platform_summary / platform_summary.max()
    scaled_summary.columns = [
        "Max Models Loaded",
        "Latency",
        "Throughput",
        "Max Concurent Users",
    ]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax = scaled_summary.T.plot.bar(
        rot=0, ax=ax, width=0.7, color={"pt": "#ee4c2c", "trt": "#75b900"}
    )
    ax.set_title(f"{model_name} on {instance_type}")
    ax.set_ylim(top=1.1)
    ax.get_yaxis().set_ticks([])
    units = ["", " ms", " qps", ""]
    pt_labels = [
        f"{val:.0f}{units}" for val, units in zip(platform_summary.T["pt"], units)
    ]
    trt_labels = [
        f"{val:.0f}{units}" for val, units in zip(platform_summary.T["trt"], units)
    ]

    ax.bar_label(
        ax.containers[0],
        labels=pt_labels,
        color="white",
        label_type="center",
        fontweight="bold",
        fontsize=12,
    )
    ax.bar_label(
        ax.containers[1],
        labels=trt_labels,
        color="white",
        label_type="center",
        fontweight="bold",
        fontsize=12,
    )
    sns.despine(fig, ax, left=True)

    return fig