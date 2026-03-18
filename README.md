# AI inference example scripts for supercomputers

## Starting a vLLM inference server

Scripts to run [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) (distilled versions, Qwen-32B or Llama-70B) vLLM using 4 GPUs on Puhti, Mahti or LUMI. There is also a script to run the full [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) model on two full LUMI nodes (16 GPUs).

- [`run-vllm-puhti4.sh`](run-vllm-puhti4.sh) ( deepseek-ai/DeepSeek-R1-Distill-Qwen-32B )
- [`run-vllm-mahti4.sh`](run-vllm-mahti4.sh) ( deepseek-ai/DeepSeek-R1-Distill-Qwen-32B )
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

On Puhti/Mahti
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

## Running Inference on LUMI with Python
We provide three Python scripts for running LLM inference on LUMI using the `lumi-multitorch` container.

### 1. Interactive Chat (Server-Client Mode)
Start a vLLM server and start a chat (with history) with the LLM. 

1.  **Start the vLLM server.** (Make sure to update your billing project first)
    ```bash
    sbatch run-vllm-lumi4.sh
    ```
2.  **Connect to the compute node's shell.** Find your job ID with `squeue --me`, then "overlap" into the allocated node as soon as the job is running:
    ```bash
    srun --overlap --jobid <slurm-job-id> --pty bash
    ```
3. **Wait ~20min for the model to load.** Monitor progress with `tail -f slurm-<job-id>.out`.
   The model has been loaded when you see line similar to `[0;36m(APIServer pid=8379)[0;0m INFO:     Application startup complete.`

5.  **Launch the chat script.**
    ```bash
    singularity run -B /pfs,/scratch,/projappl /appl/local/laifs/containers/lumi-multitorch-latest.sif \
    python chat_with_LLM.py "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    ```
    Type 'exit' to stop.

---

### 2. Batched Inference with the server
Send a large volume of prompts from `prompts.txt` to the vLLM server, 256 at a time. 

1.  **Start the server and connect to the node:** (follow steps 1-3 from the Chat mode above).
2.  **Run the batch script:**
    ```bash
    singularity run -B /pfs,/scratch,/projappl /appl/local/laifs/containers/lumi-multitorch-latest.sif \
    python offline_batched_inference_from_server.py "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    ```
    *The results will be saved to `results.json`.*

---

### 3. Python Batch Inference (single node)
Get resources with _salloc_ and run batched inference directly in Python. Use this method for high-throughput and simplicity. 

1.  **Request an interactive GPU allocation:**
    ```bash
    salloc -p dev-g --nodes=1 --gpus-per-node=8 --ntasks-per-node=1 --cpus-per-task=56 --time=2:00:00 --account=project_XXXXXXXXX
    ```
2.  **Enter the compute node:**
    ```bash
    srun --overlap --jobid <slurm-job-id> --pty bash
    ```
3.  **Set required environment variables:** (enter your project ID)
    ```bash
    export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES
    export TORCH_COMPILE_DISABLE=1
    export HF_HOME=/scratch/$SLURM_JOB_ACCOUNT/$USER/hf-cache
    ```

4.  **Run the script**:
    ```bash
    singularity run -B /pfs,/scratch,/projappl /appl/local/laifs/containers/lumi-multitorch-latest.sif \
    python offline_batched_inference_from_Python.py "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    ```

## TODO
- [ ] the Ollama scripts don't seem to use all GPUs, probably scripts are reserving too much
