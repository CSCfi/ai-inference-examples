import httpx
import openai
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("socket_file", type=str)
args = parser.parse_args()

transport = httpx.HTTPTransport(uds=args.socket_file)
httpx_client = httpx.Client(transport=transport)
client = openai.OpenAI(
        api_key='',
        base_url='http://localhost/v1',
        http_client=httpx_client
)

prompt="Running vLLM on a supercomputer is "
print(prompt, end="")

for chunk in client.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        prompt=prompt,
        max_tokens=100,
        temperature=0.5,
        stream=True
):
        print(chunk.choices[0].text, end="")
