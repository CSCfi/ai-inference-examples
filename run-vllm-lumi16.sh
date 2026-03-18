#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -p dev-g
#SBATCH --time 1:00:00
#SBATCH --tasks-per-node 1
#SBATCH --gpus-per-node 8
#SBATCH --nodes 2
#SBATCH --mem 460G

# We use the PyTorch container provided by the LUMI AI Factory Services, which contains vLLM.
export CONTAINER_IMAGE=/appl/local/laifs/containers/lumi-multitorch-latest.sif
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

## note: below are copied from large-scale training run, which was a known setup for multinode with torchrun, but they are not required to work. need to test if they improve performance.
# export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
# export NCCL_NET_GDR_LEVEL=PHB
# export FI_MR_CACHE_MONITOR=userfaultfd
# export FI_CXI_DEFAULT_CQ_SIZE=131072
# export HSA_ENABLE_SDMA=0 
# export RCCL_MSCCL_FORCE_ENABLE=1
# export OMP_NUM_THREADS=1
# export CUDA_DEVICE_MAX_CONNECTIONS=1

# Where to store the huge models. Point this to your project's scratch directory.
export HF_HOME=/scratch/$SLURM_JOB_ACCOUNT/hf-cache/

export TORCH_COMPILE_DISABLE=1

export MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}
export MASTER_PORT=${MASTER_PORT:-9999}

# We configure vLLM to use a Unix Domain Socket file (vllm.sock) to listen for requests using the --uds argument.
# This automatically restricts request to users that can access that file (i.e., members of our project), instead of being
# an open HTTP port anyone on the system could potentially access.
SOCKET_FILE=$TMPDIR/vllm-$SLURM_JOB_ACCOUNT.sock

# TODO: double check configuration for expert parallelism for better performance - data parallelism?
# https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/#example-2-node-deployment
srun singularity exec $CONTAINER_IMAGE ./run-vllm-process.sh deepseek-ai/DeepSeek-R1-0528 --tensor-parallel 8 --pipeline-parallel $SLURM_NNODES --enable-expert-parallel --all2all-backend deepep_low_latency --uds $SOCKET_FILE

