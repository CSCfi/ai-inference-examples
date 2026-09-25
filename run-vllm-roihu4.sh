#!/bin/bash
#SBATCH --account=project_2001659
#SBATCH --partition=gpumedium
#SBATCH --time=30
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:gh200:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=868344

module purge
module load python-vllm/0.29.0

MODEL=Qwen/Qwen3-32B

# We are putting the cache on the local NVME drive. 
# Alternatively store it to the project's scratch.

#export HF_HOME=/scratch/$SLURM_JOB_ACCOUNT/$USER/hf-cache/
export HF_HOME=$TMPDIR/hf-cache
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torch-cache

mkdir -p $HF_HOME

srun apptainer exec --bind=$(csc-common-bind) $SIF ./run-vllm-process.sh $MODEL

