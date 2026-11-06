
import os

from openai import OpenAI

api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("请先设置环境变量 DEEPSEEK_API_KEY 或 OPENAI_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好，请简单介绍一下你自己。"},
    ],
    max_tokens=512,
    extra_body={"thinking": {"type": "disabled"}},
)

choice = response.choices[0]
print("消息内容:", choice.message.content)
print("完成原因-finish_reason:", choice.finish_reason)
print("usage:", response.usage)
