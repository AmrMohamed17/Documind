from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.routes import router
from app.core.embeddings import get_embedding_function
import os


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading embedding model into memory...")
    app.state.embeddings = get_embedding_function()
    print("Embedding model ready.")
    yield
    print("Shutting down DocuMind...")


app = FastAPI(
    title="DocuMind",
    description="Enterprise RAG Platform — semantic search over your documents",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    Used by Docker, CI/CD, and load balancers to verify the app is running.
    Always returns 200 if the server is up.
    """
    return {"status": "healthy", "service": "documind"}