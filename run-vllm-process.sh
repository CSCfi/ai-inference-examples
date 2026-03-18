#!/bin/bash
#
# Runs a vLLM process using all allocated GPUs on a node. Will set up multi-node configuration for vLLM according to
# the number of reserved nodes. Any other arguments to this script will be passed on to vLLM. If there are no other
# arguments, this script by deafult configures vLLM with 
#   - tensor parallelism among the GPUs available on a node,
#   - pipeline parallelism between nodes,
#   - listening on unix domain socket file vllm-<USER>.sock.
#
# In that case, only power-of-two amounts of GPUs should be allocated (and ideally full nodes when using multiple nodes).

# Usage:
# ./run_vllm_process.sh <model_name> [<vllm args>]

MODEL_NAME=$1
shift 1

# By default we configure vLLM to use a Unix Domain Socket file (vllm.sock) to listen for requests using the --uds argument.
# This automatically restricts request to users that can access that file, instead of being an open HTTP port anyone 
# on the system could potentially access.
SOCKET_FILE=$TMPDIR/vllm-$USER.sock
VLLM_ARGS="${@:- --tensor-parallel-size $SLURM_GPUS_ON_NODE --pipeline-parallel-size $SLURM_NNODES --uds $SOCKET_FILE}"

if [[ -z "$MODEL_NAME" ]]; then
    echo "Usage: ./run_vllm_process.sh <model_name> "
    exit 1
fi

# note: vLLM will error if HIP_VISIBLE_DEVICES is not set but ROCR_VISIBLE_DEVICES is
export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES

# if more than one node is requested, we configure vllm multi-node command line arguments
MULTINODE_ARGS=""
if [[ "$SLURM_NNODES" -gt 1 ]]; then

    # if MASTER_ADDR is not set, exit with an error - inside the container we do not have access to slurm to look it up
    if [[ -z "$MASTER_ADDR" ]]; then
        echo "The MASTER_ADDR environment variable must be set to the hostname/IP of the node running the rank 0 process"
        exit 1
    fi

    MASTER_PORT=${MASTER_PORT:-9999}

    MULTINODE_ARGS="\
        --distributed-executor-backend mp \
        --nnodes $SLURM_NNODES \
        --node-rank $SLURM_PROCID \
        --master-addr $MASTER_ADDR \
        --master-port $MASTER_PORT "

    # all but the first vllm processes need the additional "--headless" argument
    if [[ "$SLURM_PROCID" -ne 0 ]]; then
        MULTINODE_ARGS="${MULTINODE_ARGS} --headless"
    fi
fi

CMD="vllm serve $MODEL_NAME $MULTINODE_ARGS $VLLM_ARGS"

echo $CMD
$CMD

