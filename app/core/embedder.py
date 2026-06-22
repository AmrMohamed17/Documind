from google.genai import types
from app.core.gemini import get_client

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768


def _normalize(vector: list[float]) -> list[float]:
  magnitude = sum(x*x for x in vector) ** 0.5
  if magnitude == 0:
    return vector
  return [x/magnitude for x in vector]

def _embed(text: str, task_type: str) -> list[float]:
  response = get_client().models.embed_content(
    model=EMBED_MODEL,
    contents=text,
    config=types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=EMBED_DIM,
    ),
  )
  return _normalize(response.embeddings[0].values)


def embed_document(text: str) -> list[float]:
    """Embed a document chunk — used when storing/indexing."""
    return _embed(text, task_type="RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> list[float]:
    """Embed a user question — used when searching."""
    return _embed(text, task_type="RETRIEVAL_QUERY")
