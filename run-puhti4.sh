#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=320G
#SBATCH --time=15
#SBATCH --gres=gpu:v100:4

module load pytorch/2.5

export HF_HOME=/scratch/project_2001659/mvsjober/hf-cache

# Deepseek-r1-70b
# MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

python -m vllm.entrypoints.openai.api_server --model=$MODEL \
       --tensor-parallel-size 4 \
       --max-model-len 32768 \
       --dtype half \
       --enforce-eager &

VLLM_PID=$!

# Wait until vLLM is running properly
sleep 20
while ! curl localhost:8000 >/dev/null 2>&1
do
    # catch if vllm has crashed
    if [ -z $(ps --pid $VLLM_PID --no-headers) ]; then
        exit
    fi
    sleep 10
done


curl localhost:8000/v1/completions -H "Content-Type: application/json" \
     -d "{\"prompt\": \"What would be like a hello world for LLMs?\", \"temperature\": 0, \"max_tokens\": 100, \"model\": \"$MODEL\"}" | json_pp

kill $VLLM_PID
