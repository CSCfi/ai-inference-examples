# AI inference example scripts for supercomputers

Scripts to run [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) (distilled versions, Qwen-32B or Llama-70B) vLLM with a full node on Puhti, Mahti or LUMI:

- [`run-puhti4.sh`](run-puhti4.sh)
- [`run-mahti4.sh`](run-mahti4.sh)
- [`run-lumi8.sh`](run-lumi8.sh)

These scripts start the vLLM server in OpenAI-compatible API mode, runs a query (that you can replace with something more substantial), and then quits. You can modify the code to also keep it running for the duration of the job.

There's also a script to run vLLM with ray for two full nodes on LUMI: [run-ray.sh](run-ray.sh)

## TODO
- [ ] run-ray.sh should use high-speed net and not need `NCCL_NET=Socket`
