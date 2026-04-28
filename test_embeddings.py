from langchain_core.documents import Document
from app.core.embeddings import (
    get_embedding_function,
    get_vector_store,
    add_documents,
    similarity_search
)

# ── Test 1: Embedding function ──────────────────────────────────────────
print("=== TEST 1: Embedding Function ===")
embeddings = get_embedding_function()
vector = embeddings.embed_query("Hello world")
print(f"Embedding model loaded successfully")
print(f"Vector dimensions: {len(vector)}")  # should be 384
print(f"First 5 values: {vector[:5]}")      # should be floats

# ── Test 2: Vector store ────────────────────────────────────────────────
print("\n=== TEST 2: Vector Store ===")
store = get_vector_store()
print(f"Vector store created successfully")
print(f"Collection name: {store._collection.name}")  # should be "Documind"

# ── Test 3: Add documents ───────────────────────────────────────────────
print("\n=== TEST 3: Add Documents ===")
sample_docs = [
    Document(
        page_content="Enterprise clients get a full refund within 30 days.",
        metadata={"source": "refund_policy.txt", "page": 1}
    ),
    Document(
        page_content="Standard clients receive store credit only, no cash refunds.",
        metadata={"source": "refund_policy.txt", "page": 1}
    ),
    Document(
        page_content="The CEO founded the company in 2020 in San Francisco.",
        metadata={"source": "company_info.txt", "page": 1}
    ),
]
count = add_documents(sample_docs)
print(f"Added {count} documents successfully")

# ── Test 4: Similarity search ───────────────────────────────────────────
print("\n=== TEST 4: Similarity Search ===")
query = "What is the refund policy for enterprise clients?"
results = similarity_search(query, k=2)

print(f"Query: '{query}'")
print(f"Top {len(results)} results:\n")
for i, doc in enumerate(results):
    print(f"Result {i+1}:")
    print(f"  Content : {doc.page_content}")
    print(f"  Source  : {doc.metadata.get('source')}")
    print(f"  Page    : {doc.metadata.get('page')}")
    print()