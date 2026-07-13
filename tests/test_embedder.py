"""Tests for embedding helpers. The Gemini API is mocked — tests must be
fast, free, and runnable with no API key."""
from unittest.mock import patch, MagicMock
from app.core.embedder import _normalize, embed_query, EMBED_DIM


def test_normalize_produces_unit_length():
    v = _normalize([3.0, 4.0])           # magnitude 5
    assert abs(sum(x * x for x in v) ** 0.5 - 1.0) < 1e-9
    assert abs(v[0] - 0.6) < 1e-9


def test_normalize_handles_zero_vector():
    """Must not divide by zero."""
    assert _normalize([0.0, 0.0]) == [0.0, 0.0]


def test_embed_query_returns_normalized_vector():
    fake = MagicMock()
    fake.embeddings = [MagicMock(values=[3.0, 4.0])]
    with patch("app.core.embedder.get_client") as client:
        client.return_value.models.embed_content.return_value = fake
        v = embed_query("what is a query parameter?")
    assert abs(sum(x * x for x in v) ** 0.5 - 1.0) < 1e-9


def test_embed_dim_is_768():
    """The chunks table column is vector(768) — these must not drift apart."""
    assert EMBED_DIM == 768