# ingest_corpus.py — bulk-ingest a folder of documents into pgvector
from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path
from app.core.rag import ingest_document

corpus_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "corpus")

files = sorted(
    p for p in corpus_dir.iterdir()
    if p.suffix.lower() in {".md", ".txt", ".pdf"}
)

print(f"Found {len(files)} files in {corpus_dir}\n")

total = 0
for i, path in enumerate(files, 1):
    result = ingest_document(str(path))
    total += result["chunks_stored"]
    print(f"[{i}/{len(files)}] {path.name}: {result['chunks_stored']} chunks")

print(f"\nDone — {len(files)} files, {total} chunks stored.")