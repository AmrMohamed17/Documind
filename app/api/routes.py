# app/api/routes.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os

from app.core.rag import ingest_document, query_documents
from app.core.embeddings import list_sources

router = APIRouter()

UPLOAD_DIR = "data/raw"
os.makedirs(UPLOAD_DIR, exist_ok=True)
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"


class QueryRequest(BaseModel):
    question: str
    k: int = 4


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


@router.post("/ingest", response_model=IngestResponse)
def ingest(file: UploadFile = File(...)):
    if DEMO_MODE:
        raise HTTPException(
            status_code=403,
            detail="Uploads are disabled in the public demo. This demo answers questions "
                   "over the FastAPI documentation corpus."
        )
    
    allowed = {".pdf", ".txt"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed}",
        )

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = ingest_document(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return IngestResponse(
        message="Document ingested successfully",
        file=result["file"],
        pages_loaded=result["pages_loaded"],
        chunks_stored=result["chunks_stored"],
    )


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
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
        chunks_used=result["chunks_used"],
    )


@router.get("/documents")
def list_documents():
    sources = list_sources()
    return {"documents": [{"file": s.split("/")[-1], "source": s} for s in sources]}