from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

CHROMA_PATH = os.getenv("CHROMA_PATH", ".chroma")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

def get_embedding_function():
  return HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
  )


def get_vector_store(collection_name: str = "Documind") -> Chroma:
  return Chroma(
    collection_name=collection_name,
    embedding_function=get_embedding_function(),
    persist_directory=CHROMA_PATH
  )


def add_documents(docs: list, collection_name: str = "Documind"):
  store = get_vector_store(collection_name)
  store.add_documents(docs)
  return len(docs)

def similarity_search(query: str, k: int = 5, collection_name: str = "Documind"):
  store = get_vector_store(collection_name)
  return store.similarity_search(query, k)