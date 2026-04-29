#!/bin/bash
#SBATCH --argos=no
#SBATCH --account=project_2001659
#SBATCH --partition=gpumedium
#SBATCH --time=30
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:gh200:4
#SBATCH --nodes=2
#SBATCH --mem=480G
#SBATCH --cpus-per-task=288

module purge
module load csc-tools
module load python-vllm/0.19

MODEL=Qwen/Qwen3-32B

# We are putting the cache in the ramdisk, stored in
# memory. Alternatively store it to the project's scratch.

#export HF_HOME=/scratch/$SLURM_JOB_ACCOUNT/$USER/hf-cache/
export HF_HOME=/dev/shm/$USER/hf-cache
export TORCHINDUCTOR_CACHE_DIR=/dev/shm/$USER/
mkdir -p $HF_HOME

export MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}
export MASTER_PORT=${MASTER_PORT:-9999}

# We configure vLLM to use a Unix Domain Socket file (vllm.sock) to listen for requests using the --uds argument.
# This automatically restricts request to users that can access that file (i.e., members of our project), instead of being
# an open HTTP port anyone on the system could potentially access.
SOCKET_FILE=$TMPDIR/vllm-$SLURM_JOB_ACCOUNT.sock

srun apptainer exec --bind=$(csc-common-bind) $SIF ./run-vllm-process.sh $MODEL --tensor-parallel 4 --pipeline-parallel $SLURM_NNODES --all2all-backend deepep_low_latency --uds $SOCKET_FILE

