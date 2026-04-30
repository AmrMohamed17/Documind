from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.core.rag import ingest_document, query_documents
import shutil
import os


router = APIRouter()

# Directory where uploaded files are temporarily saved
UPLOAD_DIR = "data/raw"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Request/Response models ─────────────────────────────────────────────
# Pydantic models define the shape of your API's inputs and outputs.
# FastAPI uses these for automatic validation and Swagger documentation.

class QueryRequest(BaseModel):
    question: str
    k: int = 4  # number of chunks to retrieve, default 4

class IngestResponse(BaseModel):
    message: str
    file: str
    pages_loaded: int
    chunks_stored: int

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    chunks_used: int


# ── Routes ──────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """
    Accepts a PDF or TXT file upload.
    Saves it to disk, runs the ingestion pipeline, returns a summary.

    async def — this endpoint is non-blocking. While the file is being
    saved and processed, the server can handle other requests concurrently.
    """
    # Validate file type
    allowed_extensions = {".pdf", ".txt"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed_extensions}"
        )

    # Save uploaded file to disk
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run ingestion pipeline
    try:
        result = ingest_document(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return IngestResponse(
        message="Document ingested successfully",
        file=result["file"],
        pages_loaded=result["pages_loaded"],
        chunks_stored=result["chunks_stored"]
    )


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Accepts a question, runs semantic search over ingested documents,
    returns the most relevant context with source references.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = query_documents(request.question, k=request.k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        chunks_used=result["chunks_used"]
    )


@router.get("/documents")
async def list_documents():
    """
    Returns a list of unique documents currently stored in ChromaDB.
    Called on page load to restore the document list in the UI.
    """
    from app.core.embeddings import get_vector_store
    store = get_vector_store()
    results = store.get()
    
    if not results or not results.get("metadatas"):
        return {"documents": []}

    # Deduplicate by source filename
    seen = {}
    for metadata in results["metadatas"]:
        source = metadata.get("source", "unknown")
        filename = source.split("/")[-1]
        if filename not in seen:
            seen[filename] = {"file": filename, "source": source}

    return {"documents": list(seen.values())}