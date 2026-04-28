from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.core.embeddings import add_documents, similarity_search
import mlflow
import time
import os

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:///home/ubuntu/CS/projects/documind/mlruns"))


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


def query_documents(question: str, k: int = 4) -> dict:
    """
    Query pipeline wrapped with MLflow tracking.
    Every call logs parameters, metrics, and artifacts as a new run.
    """
    mlflow.set_experiment("documind-rag-queries")

    with mlflow.start_run():

        mlflow.log_params({
            "question": question,
            "k": k,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "embedding_model": EMBEDDING_MODEL
        })

        start_time = time.time()
        relevant_chunks = similarity_search(question, k=k)
        retrieval_latency = round(time.time() - start_time, 4)

        mlflow.log_metrics({
            "retrieval_latency_seconds": retrieval_latency,
            "chunks_retrieved": len(relevant_chunks),
            "chunks_requested": k,
            # if db has fewer docs than k, this will be < 1.0
            "retrieval_rate": len(relevant_chunks) / k if k > 0 else 0
        })

        if not relevant_chunks:
            mlflow.set_tag("result", "no_documents_found")
            return {
                "answer": "No relevant documents found.",
                "sources": [],
                "chunks_used": 0
            }

        # ── Build answer ─────────────────────────────────────────────
        context = "\n\n".join([
            f"[Source: {chunk.metadata.get('source', 'unknown')} | "
            f"Page: {chunk.metadata.get('page', 'N/A')}]\n{chunk.page_content}"
            for chunk in relevant_chunks
        ])

        sources = list(set([
            chunk.metadata.get("source", "unknown")
            for chunk in relevant_chunks
        ]))

        # ── Log artifact (save the full query+answer as a text file) ─
        artifact_content = f"""QUESTION:\n{question}

PARAMETERS:
  k             : {k}
  chunk_size    : {CHUNK_SIZE}
  chunk_overlap : {CHUNK_OVERLAP}
  embedding     : {EMBEDDING_MODEL}

METRICS:
  retrieval_latency : {retrieval_latency}s
  chunks_retrieved  : {len(relevant_chunks)}

SOURCES:
{chr(10).join(sources)}

RETRIEVED CONTEXT:
{context}
"""
        artifact_path = f"query_result.txt"
        with open(artifact_path, "w") as f:
            f.write(artifact_content)
        mlflow.log_artifact(artifact_path)
        os.remove(artifact_path)

        mlflow.set_tag("result", "success")

    return {
        "answer": context,
        "sources": sources,
        "chunks_used": len(relevant_chunks)
    }