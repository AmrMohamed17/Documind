# app/core/embedder.py
"""
Turns text into 768-dim embedding vectors using gemini-embedding-001.
Embeds in batches with retry-on-rate-limit.
"""
import time
from google.genai import types
from google.genai import errors
from app.core.gemini import get_client

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
BATCH_SIZE = 50          # chunks per request (stays well under limits)
MAX_RETRIES = 5


def _normalize(vector: list[float]) -> list[float]:
    magnitude = sum(x * x for x in vector) ** 0.5
    if magnitude == 0:
        return vector
    return [x / magnitude for x in vector]


def _embed_batch(texts: list[str], task_type: str) -> list[list[float]]:
    """Embed a list of texts in ONE request, with retry on rate limits."""
    for attempt in range(MAX_RETRIES):
        try:
            response = get_client().models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=EMBED_DIM,
                ),
            )
            return [_normalize(e.values) for e in response.embeddings]
        except errors.ClientError as e:
            if e.code == 429 and attempt < MAX_RETRIES - 1:
                wait = 20 * (attempt + 1)   # 20s, 40s, 60s...
                print(f"  rate limited — waiting {wait}s then retrying...")
                time.sleep(wait)
            else:
                raise
    return []


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed many document chunks, in batches."""
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        out.extend(_embed_batch(batch, task_type="RETRIEVAL_DOCUMENT"))
    return out


def embed_query(text: str) -> list[float]:
    """Embed a single user question (for searching)."""
    return _embed_batch([text], task_type="RETRIEVAL_QUERY")[0]