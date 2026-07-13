# eval/latency.py
"""
Measure retrieval and end-to-end latency across the golden questions.
Reports p50/p95 — the numbers that actually describe user experience.

Run from repo root:  python -m eval.latency [--n 20] [--full]
  --full  also times generation (slow, uses API quota)
"""
import json
import sys
import time
from pathlib import Path
from statistics import median

from dotenv import load_dotenv
load_dotenv()

from app.core.embedder import embed_query
from app.core.embeddings import search_by_embedding
from app.core.reranker import rerank_with_rrf, RERANK_CANDIDATES
from app.core.rag import build_context, generate_answer

GOLDEN = Path("eval/golden.jsonl")
K = 4


def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    idx = min(int(len(s) * p), len(s) - 1)
    return s[idx]


def report(name: str, samples: list[float]) -> None:
    print(f"  {name:<24} p50 {median(samples)*1000:7.0f} ms   "
          f"p95 {percentile(samples, 0.95)*1000:7.0f} ms")


def main():
    full = "--full" in sys.argv
    n = 20
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])

    questions = [
        json.loads(l)["question"]
        for l in GOLDEN.read_text().splitlines() if l.strip()
    ][:n]

    embed_t, dense_t, rerank_t, retrieval_t, gen_t, total_t = [], [], [], [], [], []

    print(f"Timing {len(questions)} questions "
          f"({'end-to-end' if full else 'retrieval only'})...\n")

    for q in questions:
        t0 = time.perf_counter()
        vec = embed_query(q)
        t1 = time.perf_counter()
        candidates = search_by_embedding(vec, k=RERANK_CANDIDATES)
        t2 = time.perf_counter()
        results = rerank_with_rrf(q, candidates)[:K]
        t3 = time.perf_counter()

        embed_t.append(t1 - t0)
        dense_t.append(t2 - t1)
        rerank_t.append(t3 - t2)
        retrieval_t.append(t3 - t0)

        if full:
            context = build_context(results)
            t4 = time.perf_counter()
            generate_answer(q, context)
            t5 = time.perf_counter()
            gen_t.append(t5 - t4)
            total_t.append(t5 - t0)

    print("=== Latency ===")
    report("embed query (Gemini)", embed_t)
    report("dense search (pgvector)", dense_t)
    report("rerank + RRF (local)", rerank_t)
    print("  " + "-" * 52)
    report("RETRIEVAL total", retrieval_t)
    if full:
        report("generation (Gemini)", gen_t)
        report("END-TO-END", total_t)
    print(f"\n  n={len(questions)}")


if __name__ == "__main__":
    main()