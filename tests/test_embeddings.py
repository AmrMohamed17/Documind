import pytest
from langchain_core.documents import Document
from app.core.embeddings import (
    get_embedding_function,
    get_vector_store,
    add_documents,
    similarity_search
)


def test_embedding_function_loads():
    """Embedding model loads and produces 384-dimensional vectors"""
    embeddings = get_embedding_function()
    vector = embeddings.embed_query("test query")
    assert len(vector) == 384
    assert all(isinstance(v, float) for v in vector)


def test_embedding_is_normalized():
    """Vectors should be normalized — magnitude close to 1.0"""
    embeddings = get_embedding_function()
    vector = embeddings.embed_query("test query")
    magnitude = sum(v**2 for v in vector) ** 0.5
    assert abs(magnitude - 1.0) < 0.01


def test_vector_store_initializes():
    """ChromaDB initializes with correct collection name"""
    store = get_vector_store(collection_name="test_collection")
    assert store._collection.name == "test_collection"


def test_add_and_search_documents():
    """Documents added to ChromaDB are retrievable via similarity search"""
    docs = [
        Document(
            page_content="Python is a high-level programming language.",
            metadata={"source": "test.txt", "page": 1}
        ),
        Document(
            page_content="The Eiffel Tower is located in Paris, France.",
            metadata={"source": "test.txt", "page": 2}
        ),
    ]
    add_documents(docs, collection_name="test_collection")
    results = similarity_search(
        "What programming language is high-level?",
        k=1,
        collection_name="test_collection"
    )
    assert len(results) == 1
    assert "Python" in results[0].page_content


def test_semantic_search_ranks_correctly():
    """Semantically relevant doc ranks above irrelevant one"""
    results = similarity_search(
        "Where is the Eiffel Tower?",
        k=2,
        collection_name="test_collection"
    )
    assert "Eiffel" in results[0].page_content