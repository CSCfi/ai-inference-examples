# AI inference example scripts for supercomputers

## Starting a vLLM inference server

Scripts to run [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) (distilled versions, Qwen-32B or Llama-70B) vLLM using 4 GPUs on Puhti, Mahti, Roihu or LUMI. There is also a script to run on Roihu using two full nodes (8 GPUs). Finally, there is a script to run the full [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) model on two full LUMI nodes (16 GPUs).

- [`run-vllm-puhti4.sh`](run-vllm-puhti4.sh) ( deepseek-ai/DeepSeek-R1-Distill-Qwen-32B )
- [`run-vllm-mahti4.sh`](run-vllm-mahti4.sh) ( deepseek-ai/DeepSeek-R1-Distill-Qwen-32B )
- [`run-vllm-roihu4.sh`](run-vllm-roihu4.sh) ( deepseek-ai/DeepSeek-R1-Distill-Qwen-32B )
- [`run-vllm-roihu8.sh`](run-vllm-roihu8.sh) ( deepseek-ai/DeepSeek-R1-Distill-Qwen-32B )
- [`run-vllm-lumi4`](run-vllm-lumi4.sh) (deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)
- [`run-vllm-lumi16`](run-vllm-lumi16.sh) (deepseek-ai/DeepSeek-R1-0528)

**Note:** all script are Slurm batch job scripts and need to be submitted with `sbatch`, for example:

```bash
sbatch run-vllm-lumi4.sh
```

The LUMI scripts start the vLLM server listening on a Unix Domain Socket which is represented by a file on the filesystem (by default `vllm-<slurm_job_id>.sock`) rather than opening a network port on the node for security reasons. This also has the advantage that we cannot get into conflicts with other processes that might block the same port.

While the job is running, you can connect connect to the vLLM server with a process on the same node via that node.
For example, the following opens a terminal on the node running vLLM and sends a request via the cURL command line tool:

```bash
username@login-node$ srun --overlap --jobid <slurm-job-id> --pty bash

username@compute-node$ curl --unix-socket $TMPDIR/vllm-project_<project_id>.sock http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "prompt": "Running vLLM on a supercomputer is",
        "max_tokens": 100,
        "temperature": 0.5,
        "stream": false
    }'
```

You can also use e.g. the OpenAI client to programmatically interact with the vLLM server in Python:

```python
import httpx
import openai
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("socket_file", type=str)
args = parser.parse_args()

transport = httpx.HTTPTransport(uds=args.socket_file)
httpx_client = httpx.Client(transport=transport)
client = openai.OpenAI(
        api_key='',
        base_url='http://localhost/v1',
        http_client=httpx_client
)

prompt="Running vLLM on a supercomputer is "
print(prompt, end="")

for chunk in client.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        prompt=prompt,
        max_tokens=100,
        temperature=0.5,
        stream=True
):
        print(chunk.choices[0].text, end="")
```

The version of vLLM installed on Puhti and Mahti does not currently support UDS, so instead we configure it
to require authentication with an API key which we generate in the sbatch script. You can find the
key in the job log. The correponding cURL request is:

```bash 
username@compute-node$ curl http://localhost:8000/v1/completions \
    -H "Authorization: Bearer <api-key>"
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "prompt": "Running vLLM on a supercomputer is",
        "max_tokens": 100,
        "temperature": 0.5,
        "stream": false
    }'
```

You can run the script as follows.

On Puhti/Mahti/Roihu
```bash
username@login-node$ srun --overlap --jobid <slurm-job-id> --pty bash

username@compute-node$ module load pytorch
username@compute-node$ python script.py
```

On LUMI
```bash
username@login-node$ srun --overlap --jobid <slurm-job-id> --pty bash

username@compute-node$ singularity run -B /pfs,/scratch,/projappl /appl/local/laifs/containers/lumi-multitorch-latest.sif python python_client.py $TMPDIR/vllm-$SLURM_JOB_ACCOUNT.sock 
```

## Ollama examples

Scripts to run with Ollama:

- [`run-ollama-puhti4.sh`](run-ollama-puhti4.sh)
- [`run-ollama-lumi8.sh`](run-ollama-lumi8.sh)

**Note:** all script are Slurm batch job scripts and need to be submitted with `sbatch`, for example:

```bash
sbatch run-ollama-puhti4.sh
```

## TODO
- [ ] the Ollama scripts don't seem to use all GPUs, probably scripts are reserving too much
