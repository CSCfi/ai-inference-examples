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

# Where to store the huge models
# For example Deepseek-R1-Distill-Llama-70B requires 132GB
export HF_HOME=/scratch/project_462000007/mvsjober/hf-cache

# Where to store the vLLM server log
VLLM_LOG=/scratch/project_462000007/mvsjober/vllm-logs/${SLURM_JOB_ID}.log
mkdir -p $(dirname $VLLM_LOG)

MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
#MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

python -m vllm.entrypoints.openai.api_server --model=$MODEL \
       --tensor-parallel-size 4 \
       --pipeline-parallel-size 2 \
       --max-model-len 32768 \
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

# To stop job after we have run what we want kill it
kill $VLLM_PID

# ... if we want to leave it running instead, don't kill but wait
# wait

