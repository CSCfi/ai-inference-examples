#!/bin/bash
#SBATCH --account=project_462000007
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=30

module use /appl/local/csc/modulefiles/
module load pytorch/2.5

export HF_HOME=/scratch/project_462000007/mvsjober/hf-cache

# Deepseek-r1-70b
MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

python -m vllm.entrypoints.openai.api_server --model=$MODEL \
       --tensor-parallel-size 4 \
       --max-model-len 32768 \
       --enforce-eager &

VLLM_PID=$!

# Wait until vLLM is running properly
sleep 20
while ! curl localhost:8000 >/dev/null 2>&1
do
    sleep 10
done


curl localhost:8000/v1/completions -H "Content-Type: application/json" \
     -d "{\"prompt\": \"What would be like a hello world for LLMs?\", \"temperature\": 0, \"max_tokens\": 100, \"model\": \"$MODEL\"}" | json_pp

kill $VLLM_PID
