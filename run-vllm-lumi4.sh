#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -p dev-g
#SBATCH --time 2:00:00
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task=28
#SBATCH --gpus-per-node 4
#SBATCH --nodes 1
#SBATCH --mem 460G

# We use the PyTorch container provided by the LUMI AI Factory Services, which contains vLLM.
export CONTAINER_IMAGE=/appl/local/laifs/containers/lumi-multitorch-latest.sif
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# Where to store the huge models. Point this to your project's scratch directory.
export HF_HOME=/scratch/$SLURM_JOB_ACCOUNT/hf-cache/

# Torch compilation currently fails in the container, so we disable it here.
export TORCH_COMPILE_DISABLE=1

# The default parallelisation options applied by the run_vllm_process script will apply 4-fold tensor parallelism,
# which is fine for this, so we don't need to provide any options here except for the model name.
srun singularity exec $CONTAINER_IMAGE ./run-vllm-process.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-32B

