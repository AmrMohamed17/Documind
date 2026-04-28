from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.core.embeddings import add_documents, similarity_search
import os

def load_document(file_path: str) -> list[Document]:
    """
    Loads a document from disk based on its file extension.
    Returns a list of Document objects — one per page for PDFs,
    one for the entire file for TXT.
    """
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
    """
    Splits documents into smaller chunks for embedding.

    chunk_size=500     — max characters per chunk
    chunk_overlap=50   — shared characters between consecutive chunks
                         prevents answers from being cut at boundaries

    RecursiveCharacterTextSplitter tries to split on paragraph breaks
    first (\n\n), then line breaks (\n), then sentences (.), then words.
    This keeps chunks semantically coherent.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def ingest_document(file_path: str) -> dict:
    """
    Full ingestion pipeline:
    1. Load document from disk
    2. Split into chunks
    3. Store chunks in ChromaDB

    Returns a summary dict for the API response.
    """
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
    Full query pipeline:
    1. Search ChromaDB for the k most relevant chunks
    2. Build a context string from retrieved chunks
    3. Return the answer with source references

    Note: we return raw retrieved context as the answer for now.
    In Phase 5 we'll wrap this with MLflow experiment tracking.
    In a full production system you'd pass this context to an LLM.
    """
    relevant_chunks = similarity_search(question, k=k)

    if not relevant_chunks:
        return {
            "answer": "No relevant documents found.",
            "sources": [],
            "chunks_used": 0
        }

    # Build readable context from retrieved chunks
    context = "\n\n".join([
        f"[Source: {chunk.metadata.get('source', 'unknown')} | "
        f"Page: {chunk.metadata.get('page', 'N/A')}]\n{chunk.page_content}"
        for chunk in relevant_chunks
    ])

    # Deduplicate sources for the response
    sources = list(set([
        chunk.metadata.get("source", "unknown")
        for chunk in relevant_chunks
    ]))

    return {
        "answer": context,
        "sources": sources,
        "chunks_used": len(relevant_chunks)
    }