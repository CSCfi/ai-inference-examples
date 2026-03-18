# salloc -p dev-g --nodes=1 --gpus-per-node=8 --ntasks-per-node=1 --cpus-per-task=56 --time=2:00:00 --account=project_462000131 
# srun --overlap --jobid <slurm-job-id> --pty bash
# export HF_HOME=/scratch/project_462000131/voitanta/LLM_Experiments/hf-cache
# export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES
# export TORCH_COMPILE_DISABLE=1
# singularity run -B /pfs,/scratch,/projappl /appl/local/laifs/containers/lumi-multitorch-latest.sif python offline_batched_inference_from_Python.py "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

import os
import sys
import json
import argparse
import torch
from vllm import LLM, SamplingParams

# 1. Define your system prompt here
SYSTEM_PROMPT = "You are a helpful and concise AI assistant."

def main():
    # 2. Parse the model name
    parser = argparse.ArgumentParser()
    parser.add_argument("MODEL", type=str)
    args = parser.parse_args()

    # 3. Open the file with prompts
    if not os.path.exists("prompts.txt"):
        print("Error: 'prompts.txt' not found.")
        sys.exit(1)
    with open("prompts.txt", "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    # 4.Format into the "Chat" structure (List of Lists of Dicts)
    # Every conversation in the batch gets the same system prompt
    conversations = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        for prompt in prompts
    ]

    # 5. Initialize the LLM
    llm = LLM(
        model=args.MODEL,
        tensor_parallel_size=torch.cuda.device_count()
        )
    sampling_params = SamplingParams(temperature=0.6, max_tokens=2000)

    # 6. Run batched inference
    outputs = llm.chat(conversations, sampling_params=sampling_params, use_tqdm=True)

    # 7. Prepare the data for JSON
    # We use zip to ensure each prompt matches its specific response
    results = []
    for original_prompt, output in zip(prompts, outputs):
        entry = {
            "prompt": original_prompt,
            "response": output.outputs[0].text.strip(),
        }
        results.append(entry)

    # 8. Save the results
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Done! Processed {len(results)} prompts.")

if __name__ == "__main__":
    main()