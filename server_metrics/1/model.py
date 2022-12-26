import json
import subprocess
import csv
import re
import os
import numpy as np
from io import StringIO
from itertools import cycle

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.model_config = json.loads(args['model_config'])

        print(self.model_config)

        self.gpu_util_command = "nvidia-smi --query-gpu=name,pci.bus_id,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv"
        self.header_names = [' utilization.gpu [%]', ' utilization.memory [%]', ' memory.total [MiB]', ' memory.free [MiB]', ' memory.used [MiB]']
        self.field_names = ["gpu_utilization", "gpu_memory_utilization", "gpu_total_memory", "gpu_free_memory", "gpu_used_memory", "cpu_utilization", "memory_utilization"]
        self.dtypes = list(zip(self.field_names, cycle(["f4"])))
      
    def get_cpu_util(self):
      cmd = r"""top -bn1 | grep 'Cpu(s)' | sed 's/.*, *\([0-9.]*\)%* id.*/\1/' | awk '{print 100 - $1}'"""
      p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)  
      out, err = p.communicate() 
      cpu_util = float(out.decode("utf-8"))
      
      return cpu_util
    
    def get_mem_util(self):
      tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
      return round(used_m/tot_m, 2)

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """


        cpu_util = self.get_cpu_util()
        memory_util = self.get_mem_util()
        nv_output = subprocess.check_output(self.gpu_util_command.split()).decode("utf8")
        buf = StringIO(nv_output)
        reader = csv.DictReader(buf)
        
        results = []
        for row in reader:
            rec = [float(re.findall(r"[0-9.]+", row[field])[0]) for field in self.header_names]
            rec += [cpu_util, memory_util]
            results.append(tuple(rec))


        result_tensors = np.array(results, dtype=self.dtypes)

        output_tensors = [pb_utils.Tensor(name, result_tensors[name]) for name in self.field_names]

        responses = [pb_utils.InferenceResponse(output_tensors=output_tensors)]

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')