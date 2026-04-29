#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=15
#SBATCH --gres=gpu:a100:4

module load pytorch/2.7

# Generate a random API key for the vLLM server and output it
export VLLM_API_KEY=$(mktemp -u XXXXXXXXXXXX)
echo "### THE API KEY IS ###"
echo $VLLM_API_KEY
echo "######################"

# Where to store the huge models. Point this to your project's scratch directory.
# For example Deepseek-R1-Distill-Llama-70B requires 132GB
export HF_HOME=/scratch/$SLURM_JOB_ACCOUNT/hf-cache

MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

srun vllm serve $MODEL \
       --port 8000 \
       --tensor-parallel-size 4 \
       --max-model-len 32768 \
       --gpu_memory_utilization 0.98 \
       --enforce-eager 
