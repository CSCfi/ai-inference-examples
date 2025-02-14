#!/bin/bash
#SBATCH --account=project_462000007
#SBATCH --partition=dev-g
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=30

export NUMEXPR_MAX_THREADS=16
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

export VLLM_WORKER_MULTIPROC_METHOD=spawn

RAY="python3 -m ray.scripts.scripts"
RAY_PORT=6379
HEAD_NODE_ADDRESS=$(hostname -i)

# Needed on AMD at least, see https://github.com/vllm-project/vllm/issues/3818
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1

# Using the socket network, rather than infiniband et al, not ideal
# but works for now
#export NCCL_NET=Socket

# Load the modules
module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
#source venv/bin/activate

# Start Ray on the head node
echo "Initializing ray cluster on head node $HEAD_NODE_ADDRESS"
export VLLM_HOST_IP=${HEAD_NODE_ADDRESS}
$RAY start --head --port=${RAY_PORT} --disable-usage-stats

# Make sure head node has started properly
sleep 10
while ! ray status >/dev/null 2>&1
do
    sleep 5
done

WORKER_NNODES=$(( SLURM_NNODES - 1 ))
echo "Start the $WORKER_NNODES worker node(s)"
srun --ntasks=$WORKER_NNODES --nodes=$WORKER_NNODES --exclude=$(hostname) bash -c "export VLLM_HOST_IP=\$(hostname -i); $RAY start --block --address=${HEAD_NODE_ADDRESS}:${RAY_PORT}" &
#srun --ntasks=$WORKER_NNODES --nodes=$WORKER_NNODES --exclude=$(hostname) $RAY start --block --address=$(hostname):${RAY_PORT} &

# Wait until all worker nodes have checked in
sleep 10
while [ $(ray status 2>/dev/null | grep node_ | wc -l) -ne $SLURM_NNODES ]
do
    sleep 5
done
ray status

echo "Starting VLLM"
python -m vllm.entrypoints.openai.api_server \
            --distributed-executor-backend=ray \
            --model=meta-llama/Meta-Llama-3.1-8B-Instruct \
            --dtype=auto \
            --tensor-parallel-size=8 \
            --pipeline-parallel-size=2 \
            --gpu-memory-utilization=0.95 \
            --trust-remote-code &

# Wait until vLLM is running properly
sleep 20
while ! curl localhost:8000 >/dev/null 2>&1
do
    sleep 10
done

curl localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"prompt": "I finally got vLLM working on multiple nodes on LUMI", "temperature": 0, "max_tokens": 100, "model": "meta-llama/Meta-Llama-3.1-8B-Instruct"}' | json_pp

# If you want to keep vLLM running you need to add a "wait" here, otherwise the job will stop when the above line is done.
wait
