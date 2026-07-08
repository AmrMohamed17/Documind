# eval/inspect_question.py
"""
Inspect one golden question: show the gold fragments and the top-10 retrieved
chunks, and DIAGNOSE each miss as either:
  - a RANKING problem  (the right chunk exists but ranked below k -> reranker territory)
  - a CHUNKING artifact (the gold text got split across chunks -> fix the snippet)

Run from repo root:  python -m eval.inspect_question q071
"""
import sys
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from app.core.embeddings import similarity_search
from app.core.db import get_pool
from eval.text_utils import normalize, split_fragments

GOLDEN_PATH = Path("eval/golden.jsonl")
MAX_K = 10


def load_entry(qid: str) -> dict:
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                if e["id"] == qid:
                    return e
    raise SystemExit(f"Question id '{qid}' not found in {GOLDEN_PATH}")


def chunks_for_sources(sources: list[str]) -> list[str]:
    """Every stored chunk (normalized) for the given source files."""
    out = []
    with get_pool().connection() as conn:
        for src in sources:
            rows = conn.execute(
                "SELECT content FROM chunks WHERE source = %s", (src,)
            ).fetchall()
            out.extend(normalize(r[0]) for r in rows)
    return out


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m eval.inspect_question <question_id>")
    e = load_entry(sys.argv[1])

    print(f"\n=== {e['id']} ({e['type']}) ===")
    print(f"Q: {e['question']}")
    print(f"gold_source: {e['gold_source']}")

    fragments = [normalize(f) for f in split_fragments(e["gold_snippet"])]
    print(f"\nGold fragments ({len(fragments)}):")
    for i, frag in enumerate(fragments, 1):
        print(f"  F{i}: {frag[:90]}")

    results = similarity_search(e["question"], k=MAX_K)
    chunks_norm = [normalize(r["content"]) for r in results]

    print(f"\nTop-{MAX_K} retrieved chunks:")
    for rank, r in enumerate(results, 1):
        tags = "".join(
            f"[F{i+1}]" for i, frag in enumerate(fragments)
            if frag in chunks_norm[rank - 1]
        )
        preview = " ".join(r["content"].split())[:100]
        print(f"  {rank:>2}. d={r['distance']:.3f}  {r['source']:<38} {tags}")
        print(f"      {preview}")

    print("\nFragment verdicts:")
    sources = [s.strip() for s in e["gold_source"].split(",")]
    db_chunks = None
    for i, frag in enumerate(fragments, 1):
        rank_found = next(
            (rank for rank, c in enumerate(chunks_norm, 1) if frag in c), None
        )
        if rank_found:
            print(f"  F{i}: FOUND at rank {rank_found}")
            continue
        if db_chunks is None:
            db_chunks = chunks_for_sources(sources)
        if any(frag in c for c in db_chunks):
            print(f"  F{i}: MISS — the chunk exists but ranked below {MAX_K} "
                  f"→ RANKING problem (a reranker should fix this)")
        else:
            print(f"  F{i}: MISS — not inside ANY single chunk "
                  f"→ CHUNKING artifact (gold text split across chunks; fix the snippet)")

    print("\nHit (all fragments present) at:")
    for k in (3, 5, 10):
        hit = all(any(frag in c for c in chunks_norm[:k]) for frag in fragments)
        print(f"  @{k}: {'HIT' if hit else 'miss'}")


if __name__ == "__main__":
    main()