#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=320G
#SBATCH --time=15
#SBATCH --gres=gpu:v100:4

module load pytorch/2.5

# We configure vLLM to use a Unix Domain Socket file (vllm.sock) to listen for requests using the --uds argument.
# This automatically restricts request to users that can access that file (i.e., members of our project), instead of being
# an open HTTP port anyone on the system could potentially access. The following code block ensures that this socket file
# gets deleted after the job finishes.
SOCKET_FILE=$TMPDIR/vllm-$SLURM_JOB_ACCOUNT.sock

# Where to store the huge models. Point this to your project's scratch directory.
# For example Deepseek-R1-Distill-Llama-70B requires 132GB
export HF_HOME=/scratch/$SLURM_JOB_ACCOUNT/hf-cache

# MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"  # too big for Puhti
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

python -m vllm.entrypoints.openai.api_server --model=$MODEL \
       --tensor-parallel-size 4 \
       --max-model-len 32768 \
       --dtype half \
       --enforce-eager \
       --uds $SOCKET_FILE
