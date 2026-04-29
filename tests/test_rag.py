import pytest
import os
from app.core.rag import load_document, chunk_documents, ingest_document, query_documents


@pytest.fixture
def sample_txt_file(tmp_path):
    """Creates a temporary txt file for testing"""
    content = """Machine Learning Overview

Machine learning is a subset of artificial intelligence.
It enables systems to learn from data without explicit programming.

Deep learning uses neural networks with multiple layers.
It has revolutionized computer vision and natural language processing."""
    file = tmp_path / "sample.txt"
    file.write_text(content)
    return str(file)


def test_load_txt_document(sample_txt_file):
    """TXT files load correctly and return Document objects"""
    docs = load_document(sample_txt_file)
    assert len(docs) >= 1
    assert len(docs[0].page_content) > 0


def test_load_unsupported_format(tmp_path):
    """Unsupported file types raise ValueError"""
    bad_file = tmp_path / "test.csv"
    bad_file.write_text("a,b,c")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(str(bad_file))


def test_chunk_documents(sample_txt_file):
    """Documents are split into chunks correctly"""
    docs = load_document(sample_txt_file)
    chunks = chunk_documents(docs)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 600  # some tolerance over chunk_size


def test_chunk_overlap(sample_txt_file):
    """All chunks are non-empty"""
    docs = load_document(sample_txt_file)
    chunks = chunk_documents(docs)
    for chunk in chunks:
        assert len(chunk.page_content.strip()) > 0


def test_ingest_document(sample_txt_file):
    """Ingestion returns correct summary dict"""
    result = ingest_document(sample_txt_file)
    assert "file" in result
    assert "pages_loaded" in result
    assert "chunks_stored" in result
    assert result["pages_loaded"] >= 1
    assert result["chunks_stored"] >= 1


def test_query_returns_correct_structure(sample_txt_file):
    """Query response has correct keys and types"""
    ingest_document(sample_txt_file)
    result = query_documents("What is machine learning?", k=2)
    assert "answer" in result
    assert "sources" in result
    assert "chunks_used" in result
    assert isinstance(result["sources"], list)
    assert isinstance(result["chunks_used"], int)


def test_query_empty_db():
    """Query on empty collection returns no results gracefully"""
    result = query_documents("random question", k=4)
    # Should not crash — returns either answer or no results message
    assert "answer" in result