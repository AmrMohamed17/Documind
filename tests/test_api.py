import pytest
import io
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Health endpoint returns 200 with correct body"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["service"] == "documind"


def test_ingest_txt_file():
    """TXT file ingestion returns 200 with chunk count"""
    content = b"Artificial intelligence is transforming industries worldwide."
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("test.txt", io.BytesIO(content), "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Document ingested successfully"
    assert data["chunks_stored"] >= 1


def test_ingest_unsupported_file():
    """Unsupported file type returns 400"""
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("test.csv", io.BytesIO(b"a,b,c"), "text/csv")}
    )
    assert response.status_code == 400


def test_query_endpoint():
    """Query endpoint returns 200 with correct structure"""
    response = client.post(
        "/api/v1/query",
        json={"question": "What is artificial intelligence?", "k": 2}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "chunks_used" in data


def test_query_empty_question():
    """Empty question returns 400"""
    response = client.post(
        "/api/v1/query",
        json={"question": "", "k": 4}
    )
    assert response.status_code == 400


def test_query_default_k():
    """Query works without specifying k — uses default"""
    response = client.post(
        "/api/v1/query",
        json={"question": "What is AI?"}
    )
    assert response.status_code == 200