from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.core.embeddings import add_documents, similarity_search
import google.generativeai as genai
import mlflow
import time
import os

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
experiment = mlflow.get_experiment_by_name("documind-rag-queries")
if experiment is None:
    mlflow.create_experiment(
        "documind-rag-queries",
        artifact_location=os.getenv("MLFLOW_ARTIFACT_ROOT")
    )
mlflow.set_experiment("documind-rag-queries")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def load_document(file_path: str) -> list[Document]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    documents = loader.load()
    print(f"Loaded {len(documents)} page(s) from {file_path}")
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def ingest_document(file_path: str) -> dict:
    documents = load_document(file_path)
    chunks = chunk_documents(documents)
    add_documents(chunks)
    return {
        "file": os.path.basename(file_path),
        "pages_loaded": len(documents),
        "chunks_stored": len(chunks)
    }


def generate_answer(question: str, context: str) -> str:
    """
    Takes the retrieved context chunks and the user's question,
    sends them to Gemini with a structured prompt,
    returns a grounded, concise answer.

    This is the 'G' in RAG — Retrieval Augmented Generation.
    The prompt is designed to prevent hallucination by instructing
    the model to only use the provided context.
    """
    prompt = f"""You are a helpful assistant that answers questions based strictly on the provided context.
If the answer cannot be found in the context, say "I cannot find the answer in the provided documents."
Do not make up information or use knowledge outside the provided context.

Context:
{context}

Question: {question}

Answer:"""

    response = gemini_model.generate_content(prompt)
    return response.text


def query_documents(question: str, k: int = 4) -> dict:
    """
    Query pipeline wrapped with MLflow tracking.
    Every call logs parameters, metrics, and artifacts as a new run.
    """
    with mlflow.start_run() as run:

        mlflow.log_params({
            "question": question,
            "k": k,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "embedding_model": EMBEDDING_MODEL
        })

        # ── Retrieval ────────────────────────────────────────────────
        start_time = time.time()
        relevant_chunks = similarity_search(question, k=k)
        retrieval_latency = round(time.time() - start_time, 4)

        mlflow.log_metrics({
            "retrieval_latency_seconds": retrieval_latency,
            "chunks_retrieved": len(relevant_chunks),
            "chunks_requested": k,
            "retrieval_rate": len(relevant_chunks) / k if k > 0 else 0
        })

        if not relevant_chunks:
            mlflow.set_tag("result", "no_documents_found")
            return {
                "answer": "No relevant documents found.",
                "sources": [],
                "chunks_used": 0
            }

        # ── Build context ────────────────────────────────────────────
        context = "\n\n".join([
            f"[Source: {chunk.metadata.get('source', 'unknown')} | "
            f"Page: {chunk.metadata.get('page', 'N/A')}]\n{chunk.page_content}"
            for chunk in relevant_chunks
        ])

        sources = list(set([
            chunk.metadata.get("source", "unknown")
            for chunk in relevant_chunks
        ]))

        # ── Generation ───────────────────────────────────────────────
        start_generation = time.time()
        answer = generate_answer(question, context)
        generation_latency = round(time.time() - start_generation, 4)

        mlflow.log_metrics({
            "generation_latency_seconds": generation_latency,
            "total_latency_seconds": round(retrieval_latency + generation_latency, 4)
        })

        # ── Log artifact ─────────────────────────────────────────────
        artifact_content = f"""QUESTION:
{question}

PARAMETERS:
  k             : {k}
  chunk_size    : {CHUNK_SIZE}
  chunk_overlap : {CHUNK_OVERLAP}
  embedding     : {EMBEDDING_MODEL}

METRICS:
  retrieval_latency  : {retrieval_latency}s
  generation_latency : {generation_latency}s
  total_latency      : {round(retrieval_latency + generation_latency, 4)}s

SOURCES:
{chr(10).join(sources)}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}
"""
        artifact_path = "query_result.txt"
        with open(artifact_path, "w") as f:
            f.write(artifact_content)
        mlflow.log_artifact(artifact_path)
        os.remove(artifact_path)

        mlflow.set_tag("result", "success")

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(relevant_chunks)
    }