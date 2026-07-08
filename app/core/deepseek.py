# app/core/deepseek.py
"""Shared DeepSeek client (OpenAI-compatible API)."""
import os
from openai import OpenAI

_client = None

def get_deepseek() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",   # points the OpenAI client at DeepSeek
        )
    return _client


def deepseek_generate(prompt: str, model: str = "deepseek-chat") -> str:
    resp = get_deepseek().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content