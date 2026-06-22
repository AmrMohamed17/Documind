# app/main.py
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

load_dotenv()

from app.api.routes import router
from app.core.db import get_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_pool()        # open the connection pool on startup (fail fast if DB is down)
    yield
    get_pool().close()  # close it cleanly on shutdown


app = FastAPI(
    title="DocuMind",
    description="RAG over your documents — pgvector + Gemini",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/")
def serve_frontend():
    return FileResponse("app/static/index.html")


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "documind"}


app.include_router(router, prefix="/api/v1")
app.mount("/static", StaticFiles(directory="app/static"), name="static")