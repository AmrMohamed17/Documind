from app.core.rag import load_document, chunk_documents, ingest_document, query_documents
import os

# ── Test 1: Load document ───────────────────────────────────────────────
print("=== TEST 1: Load Document ===")
os.makedirs("data/raw", exist_ok=True)
with open("data/raw/test_policy.txt", "w") as f:
    f.write("""Refund Policy

Enterprise clients are eligible for a full refund within 30 days of purchase.
Refund requests must be submitted via the enterprise support portal.

Standard clients receive store credit only. No cash refunds are available.
Store credit expires after 12 months.

Contact support@documind.com for all refund inquiries.""")

docs = load_document("data/raw/test_policy.txt")
print(f"Pages loaded: {len(docs)}")
print(f"Content preview: {docs[0].page_content[:80]}...")

# ── Test 2: Chunk documents ─────────────────────────────────────────────
print("\n=== TEST 2: Chunk Documents ===")
chunks = chunk_documents(docs)
print(f"Total chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"  Chunk {i+1} ({len(chunk.page_content)} chars): {chunk.page_content[:60]}...")

# ── Test 3: Ingest document ─────────────────────────────────────────────
print("\n=== TEST 3: Ingest Document ===")
result = ingest_document("data/raw/test_policy.txt")
print(f"Ingestion result: {result}")

# ── Test 4: Query documents ─────────────────────────────────────────────
print("\n=== TEST 4: Query Documents ===")
queries = [
    "What is the refund policy for enterprise clients?",
    "How long does store credit last?",
    "How do I contact support?"
]

for query in queries:
    print(f"\nQ: {query}")
    result = query_documents(query, k=2)
    print(f"Chunks used: {result['chunks_used']}")
    print(f"Sources: {result['sources']}")
    print(f"Answer preview: {result['answer'][:150]}...")