import boto3
from botocore.config import Config
import os
import sys
import json
from locust import User, task, between, events, LoadTestShape

import numpy as np
from PIL import Image
import random
import time

# How to use
# 1. install locust & boto3
#   pip install locust boto3
# 2. run benchmark via cli
# with UI
# Since we are using a custom client for the request we need to define the "Host" as -.
#   ENDPOINT_NAME="distilbert-base-uncased-distilled-squad-6493832c-767d-4cdb-a9a" locust -f locust_benchmark_sm.py
#
# headless
# --users  Number of concurrent Locust users
# --spawn-rate  The rate per second in which users are spawned until num users
# --run-time duration of test
#   ENDPOINT_NAME="distilbert-base-uncased-distilled-squad-6493832c-767d-4cdb-a9a" locust -f locust_benchmark_sm.py \
#       --users 60 \
#       --spawn-rate 1 \
#       --run-time 360s \
#       --headless


# locust -f locust_benchmark_sm.py \
#       --users 60 \
#       --spawn-rate 1 \
#       --run-time 360s \
#       --headless

content_type = "application/octet-stream"

# nlp_payload = {
#     "inputs": [
#         {"name": "INPUT__0", "shape": [1, 128], "datatype": "INT32", "data": np.random.randint(1000, size=128).tolist()},
#         {"name": "INPUT__1", "shape": [1, 128], "datatype": "INT32", "data": np.zeros(128, dtype=int).tolist()},
#     ]
# }

cv_payload = {
    "inputs": [
        {
            "name": "INPUT__0",
            "shape": [1, 3, 224, 224],
            "datatype": "FP32",
            "data": np.random.rand(3, 224,224).tolist(),
        }
    ]
}

class SageMakerClient:
    _locust_environment = None

    def __init__(self):
        super().__init__()
        
        self.session = boto3.session.Session()
        
        self.client=self.session.client("sagemaker-runtime")
        # self.cv_payload = json.dumps(cv_payload)
        # self.nlp_payload = json.dumps(cv_payload)
        self.content_type = content_type

    def send(self, endpoint_name, use_case, model_name, model_count):
        
        request_meta = {
            "request_type": "InvokeEndpoint",
            "name": model_name,
            "start_time": time.time(),
            "num_models": model_count,
            "response_length": 0,
            "response": None,
            "context": {},
            "exception": None,
        }
        
        start_perf_counter = time.perf_counter()
        
        if use_case == 'cv':
            payload = json.dumps(cv_payload)
        elif use_case == 'nlp':
            payload = json.dumps(nlp_payload)
        else:
            payload = json.dumps(cv_payload)
            
        try:
            response = self.client.invoke_endpoint(
                EndpointName=endpoint_name,
                Body=payload,
                ContentType=self.content_type,
                TargetModel="{0}-v{1}.tar.gz".format(model_name, random.randint(0, model_count-1)),
            )
            # print(response)
        except Exception as e:
            request_meta['exception'] = e
        
        request_meta["response_time"] = (time.perf_counter() - start_perf_counter) * 1000

        events.request.fire(**request_meta)


class SageMakerUser(User):
    abstract = True
    
    @events.init_command_line_parser.add_listener
    def _(parser):
        parser.add_argument("--endpoint-name", type=str, default="mme-cv-benchmark-pt", help="sagemaker endpoint you want to invoke")
        parser.add_argument("--use-case", type=str, default="cv", help="CV or NLP")
        parser.add_argument("--model-name", type=str, default="vgg16", help="name of your model")
        parser.add_argument("--nlp-payload", type=str, default="", help="sample payload for nlp model benchmarking")
        parser.add_argument("--model-count", type=int, default=5, help="how many models you want to invoke")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
#         config = Config(max_pool_connections=300)
#         smr_client = boto3.client("sagemaker-runtime", config=config)
        
        self.client = SageMakerClient()
        self.client._locust_environment = self.environment


class SimpleSendRequest(SageMakerUser):
    wait_time = between(0.05, 0.5)

    @task
    def send_request(self):
        endpoint_name = self.environment.parsed_options.endpoint_name
        use_case = self.environment.parsed_options.use_case
        model_name = self.environment.parsed_options.model_name
        model_count = self.environment.parsed_options.model_count
        if self.environment.parsed_options.nlp_payload != "":
            globals()["nlp_payload"] = json.loads(self.environment.parsed_options.nlp_payload)
        
        self.client.send(endpoint_name, use_case, model_name, model_count)

class StagesShape(LoadTestShape):

    stages = [
        {"duration": 30, "users": 10, "spawn_rate": 5},
        {"duration": 60, "users": 20, "spawn_rate": 1},
        {"duration": 90, "users": 40, "spawn_rate": 2},
        {"duration": 120, "users": 60, "spawn_rate": 2},
        {"duration": 150, "users": 80, "spawn_rate": 2},
        {"duration": 180, "users": 100, "spawn_rate": 2},
        {"duration": 210, "users": 120, "spawn_rate": 2},
        {"duration": 240, "users": 140, "spawn_rate": 2},
        {"duration": 270, "users": 160, "spawn_rate": 2},
        {"duration": 300, "users": 180, "spawn_rate": 2},
        {"duration": 340, "users": 200, "spawn_rate": 2},
    ]

    def tick(self):
        run_time = self.get_run_time()

        for stage in self.stages:
            if run_time < stage["duration"]:
                tick_data = (stage["users"], stage["spawn_rate"])
                return tick_data

        return None