"""Tests for document loading, chunking, and context building."""
import pytest
from app.core.rag import load_document, chunk_documents, build_context


@pytest.fixture
def md_file(tmp_path):
    f = tmp_path / "sample.md"
    f.write_text("# Title\n\n" + ("Some documentation text. " * 60))
    return str(f)


def test_loads_markdown(md_file):
    docs = load_document(md_file)
    assert len(docs) >= 1
    assert "documentation" in docs[0].page_content


def test_rejects_unsupported_type(tmp_path):
    bad = tmp_path / "data.csv"
    bad.write_text("a,b,c")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(str(bad))


def test_chunking_splits_long_documents(md_file):
    chunks = chunk_documents(load_document(md_file))
    assert len(chunks) > 1
    assert all(c.page_content.strip() for c in chunks)


def test_build_context_includes_sources():
    """The LLM must see where each passage came from — that's what makes
    citations possible."""
    results = [
        {"content": "Use Query() for validation.", "source": "query-params.md", "page": 1},
        {"content": "Use Path() for path params.", "source": "path-params.md", "page": 2},
    ]
    ctx = build_context(results)
    assert "query-params.md" in ctx
    assert "path-params.md" in ctx
    assert "Use Query() for validation." in ctx