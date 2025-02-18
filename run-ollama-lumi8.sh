#!/bin/bash
#SBATCH --account=project_462000007
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=30

# Download ollama models to scratch rather than the home directory
OLLAMA_SCRATCH=/scratch/project_462000007/mvsjober/ollama
export OLLAMA_MODELS=${OLLAMA_SCRATCH}/models

# Add ollama installation dir to PATH
export PATH=/projappl/project_462000007/mvsjober/ollama/bin:$PATH

# Simple way to start ollama. All the server outputs will appear in
# the slurm log mixed with everything else.
#ollama serve &

# If you want to direct ollama server's outputs to a separate log file
# you can start it like this instead
mkdir -p ${OLLAMA_SCRATCH}/logs
ollama serve > ${OLLAMA_SCRATCH}/logs/${SLURM_JOB_ID}.log 2>&1 &

# Capture process id of ollama server
OLLAMA_PID=$!

# Wait to make sure Ollama has started properly
sleep 5

# After this you can use ollama normally in this session

MODEL=deepseek-r1:70b

# Example: use ollama commands
ollama pull $MODEL
ollama list
ollama run $MODEL "In the big picture, do LLMs really benefit humanity?"

# Example: Try REST API
# curl http://localhost:11434/api/generate -d '{
#   "model": "llama3.1:8b",
#   "prompt":"Why is the sky blue?"
# }'


# At the end of the job, stop the ollama server
kill $OLLAMA_PID
