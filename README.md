# AI inference example scripts for supercomputers

Scripts to run [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) (distilled versions, Qwen-32B or Llama-70B) vLLM with a full node on Puhti, Mahti or LUMI:

- [`run-vllm-puhti4.sh`](run-vllm-puhti4.sh)
- [`run-vllm-mahti4.sh`](run-vllm-mahti4.sh)
- [`run-vllm-lumi8.sh`](run-vllm-lumi8.sh)

These scripts start the vLLM server in OpenAI-compatible API mode, runs a query (that you can replace with something more substantial), and then quits. You can modify the code to also keep it running for the duration of the job.

There's also a script to run vLLM with ray for two full nodes on LUMI: [run-vllm-ray.sh](run-vllm-ray.sh)

Scripts to run the same with Ollama:

- [`run-ollama-puhti4.sh`](run-ollama-puhti4.sh)
- [`run-ollama-lumi8.sh`](run-ollama-lumi8.sh)

**Note:** all script are Slurm batch job scripts and need to be submitted with `sbatch`, for example:

```bash
sbatch run-vllm-lumi8.sh
```

## TODO
- [ ] the Ollama scripts don't seem to use all GPUs, probably scripts are reserving too much
- [ ] run-vllm-ray.sh should use high-speed net and not need `NCCL_NET=Socket`
