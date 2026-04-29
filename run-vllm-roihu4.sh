#!/bin/bash
#SBATCH --argos=no
#SBATCH --account=project_2001659
#SBATCH --partition=gpumedium
#SBATCH --time=30
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:gh200:4
#SBATCH --nodes=1
#SBATCH --mem=240G
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

srun apptainer exec --bind=$(csc-common-bind) $SIF ./run-vllm-process.sh $MODEL

