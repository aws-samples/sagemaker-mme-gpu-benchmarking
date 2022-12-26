import time
import json
import subprocess
import csv
from typing import Union
from pathlib import Path


def create_endpoint(
    sm_client,
    model_name: str,
    role: str,
    container: str,
    instance_type: str,
    engine: str,
):

    model_name = model_name.replace("_", "-")

    sm_model_name = f"{model_name}-{engine}-gpu-" + time.strftime(
        "%Y-%m-%d-%H-%M-%S", time.gmtime()
    )

    create_model_response = sm_client.create_model(
        ModelName=sm_model_name, ExecutionRoleArn=role, PrimaryContainer=container
    )

    print("Model Arn: " + create_model_response["ModelArn"])

    endpoint_config_name = f"{model_name}-{engine}-gpu-" + time.strftime(
        "%Y-%m-%d-%H-%M-%S", time.gmtime()
    )

    create_endpoint_config_response = sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "InstanceType": instance_type,
                "InitialVariantWeight": 1,
                "InitialInstanceCount": 1,
                "ModelName": sm_model_name,
                "VariantName": "AllTraffic",
            }
        ],
    )

    print(
        "Endpoint Config Arn: " + create_endpoint_config_response["EndpointConfigArn"]
    )

    endpoint_name = f"{model_name}-{engine}-gpu-" + time.strftime(
        "%Y-%m-%d-%H-%M-%S", time.gmtime()
    )

    create_endpoint_response = sm_client.create_endpoint(
        EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
    )

    print("Endpoint Arn: " + create_endpoint_response["EndpointArn"])

    resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
    status = resp["EndpointStatus"]
    print("Status: " + status)

    while status == "Creating":
        time.sleep(60)
        resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]
        print("Status: " + status)

    print("Arn: " + resp["EndpointArn"])
    print("Status: " + status)

    return sm_model_name, endpoint_config_name, endpoint_name


def delete_endpoint(sm_client, sm_model_name, endpoint_config_name, endpoint_name):
    sm_client.delete_endpoint(EndpointName=endpoint_name)
    sm_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    sm_client.delete_model(ModelName=sm_model_name)


def get_instance_utilization(runtime_sm_client, endpoint_name):

    payload = {
        "inputs": [
            {
                "name": "INPUT__0",
                "shape": [1],
                "datatype": "BYTES",
                "data": ["h"],
            }
        ]
    }

    response = runtime_sm_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/octet-stream",
        Body=json.dumps(payload),
        TargetModel=f"metrics.tar.gz",
    )

    output = json.loads(response["Body"].read().decode("utf8"))["outputs"]

    metrics = {}
    for metric in output:
        metrics[metric["name"]] = sum(metric["data"])

    metrics["gpu_memory_utilization"] = (
        metrics["gpu_used_memory"] / metrics["gpu_total_memory"]
    )

    return metrics

def run_load_test(
    endpoint_name: str,
    use_case: str,
    model_name: str,
    models_loaded: int,
    output_path: Union[str, Path],
    print_stdout: bool = False,
    n_procs=4,
    sample_payload:str = ""
):

    if print_stdout:
        stdout = None
    else:
        stdout = subprocess.DEVNULL

    output_path = Path(output_path)

    main_command = f"locust -f locust/locust_benchmark_sm.py --master --endpoint-name {endpoint_name} --use-case {use_case} --nlp-payload --model-name {model_name} --model-count {models_loaded} --headless --csv {output_path} --csv-full-history".split()
    worker_command = f"locust -f locust/locust_benchmark_sm.py --worker --endpoint-name {endpoint_name} --use-case {use_case} --nlp-payload --model-name {model_name} --model-count {models_loaded}".split()
    
    # insert payload into the shell command
    main_payload_idx = main_command.index("--nlp-payload")
    worker_payload_idx = worker_command.index("--nlp-payload")
    main_command.insert(main_payload_idx+1, sample_payload)
    worker_command.insert(worker_payload_idx+1, sample_payload)

    print("running load test")
    main_proc = subprocess.Popen(main_command, stdout=stdout, stderr=subprocess.STDOUT, close_fds=True)
    worker_procs = [
        subprocess.Popen(worker_command, stdout=stdout, stderr=subprocess.STDOUT, close_fds=True) for _ in range(n_procs)
    ]


    try:
        num_polls = 1
        while main_proc.poll() == None:
            if num_polls % 3 == 0:
                reader = csv.DictReader(Path(str(output_path) + "_stats.csv").open("r"))
                print(next(reader))
            time.sleep(10)
            num_polls += 1
        print(f"load test completed with exit code {main_proc.returncode}\n")  
        
    except:
        print("Load test interrupted or failed")
        print(main_proc.output)
    finally:
        print("cleaning up")
        main_proc.kill()
        [proc.kill() for proc in worker_procs]

    return output_path