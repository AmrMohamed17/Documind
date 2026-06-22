# app/core/gemini.py
"""
The single Gemini client, shared by embedding and generation.
Created once, reused everywhere (the 'one client per service' pattern).
"""
import os
from google import genai

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _client