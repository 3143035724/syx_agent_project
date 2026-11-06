import os

from openai import OpenAI

api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("请先设置环境变量 DEEPSEEK_API_KEY 或 OPENAI_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

response = client.responses.create(
    model="deepseek-chat",
    instructions="You are a helpful assistant.",
    input="你好，请简单介绍一下你自己。",
)

print("消息内容:", response.output_text)
print("完成原因-status:", response.status)
print("usage:", response.usage)

for item in response.output:
    print("item type:", item.type)