#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=15
#SBATCH --gres=gpu:a100:4

module load pytorch/2.5

# Where to store the huge models
# For example Deepseek-R1-Distill-Llama-70B requires 132GB
export HF_HOME=/scratch/project_2001659/mvsjober/hf-cache

# Where to store the vLLM server log
VLLM_LOG=/scratch/project_2001659/mvsjober/vllm-logs/${SLURM_JOB_ID}.log
mkdir -p $(dirname $VLLM_LOG)

#MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

python -m vllm.entrypoints.openai.api_server --model=$MODEL \
       --tensor-parallel-size 4 \
       --max-model-len 32768 \
       --gpu_memory_utilization 0.98 \
       --enforce-eager > $VLLM_LOG &

VLLM_PID=$!

echo "Starting vLLM process $VLLM_PID - logs go to $VLLM_LOG"

# Wait until vLLM is running properly
sleep 20
while ! curl localhost:8000 >/dev/null 2>&1
do
    # catch if vllm has crashed
    if [ -z "$(ps --pid $VLLM_PID --no-headers)" ]; then
        exit
    fi
    sleep 10
done

curl localhost:8000/v1/completions -H "Content-Type: application/json" \
     -d "{\"prompt\": \"What would be like a hello world for LLMs?\", \"temperature\": 0, \"max_tokens\": 100, \"model\": \"$MODEL\"}" | json_pp

kill $VLLM_PID
