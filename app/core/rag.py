# app/core/rag.py
"""
The RAG pipeline: load a document, chunk it, store it in pgvector,
and answer questions by retrieving relevant chunks and grounding Gemini on them.
"""
import os
import time
from google.genai import errors
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.core.embeddings import add_documents, similarity_search
from app.core.gemini import get_client

load_dotenv()

GEN_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))



def generate_answer(question: str, context: str) -> str:
    prompt = f"""You are a helpful assistant that answers questions based strictly on the provided context.
If the answer cannot be found in the context, say "I cannot find the answer in the provided documents."
Do not use any knowledge outside the provided context.

Context:
{context}

Question: {question}

Answer:"""
    for attempt in range(5):
        try:
            response = get_client().models.generate_content(
                model=GEN_MODEL,
                contents=prompt,
            )
            return response.text
        except errors.ClientError as e:
            if e.code == 429 and attempt < 4:
                wait = 25 * (attempt + 1)   # 25s, 50s, ...
                print(f"  generation rate limited — waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    return ""


def load_document(file_path: str) -> list[Document]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in (".txt", ".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load()


def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(documents)


def ingest_document(file_path: str) -> dict:
    documents = load_document(file_path)
    chunks = chunk_documents(documents)
    stored = add_documents(chunks)
    return {
        "file": os.path.basename(file_path),
        "pages_loaded": len(documents),
        "chunks_stored": stored,
    }


def build_context(results: list[dict]) -> str:
    return "\n\n".join(
        f"[Source: {r['source']} | Page: {r['page']}]\n{r['content']}"
        for r in results
    )


def generate_answer(question: str, context: str) -> str:
    prompt = f"""You are a helpful assistant that answers questions based strictly on the provided context.
If the answer cannot be found in the context, say "I cannot find the answer in the provided documents."
Do not use any knowledge outside the provided context.

Context:
{context}

Question: {question}

Answer:"""
    response = get_client().models.generate_content(
        model=GEN_MODEL,
        contents=prompt,
    )
    return response.text


def query_documents(question: str, k: int = 4) -> dict:
    results = similarity_search(question, k=k)

    if not results:
        return {"answer": "No relevant documents found.", "sources": [], "chunks_used": 0}

    context = build_context(results)
    sources = list({r["source"] for r in results})

    answer = generate_answer(question, context)

    return {"answer": answer, "sources": sources, "chunks_used": len(results)}