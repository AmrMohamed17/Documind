# eval/recall.py
"""
recall@k = fraction of answerable questions where ALL gold snippet fragments
appear somewhere in the top-k retrieved chunks.
  single-hop : the one fragment must be found
  multi-hop  : every fragment must be found (across any of the top-k chunks)
  unanswerable: skipped here (no gold passage) — scored separately later

Run from repo root:  python -m eval.recall
"""
import json
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

from app.core.embedder import embed_queries
from app.core.reranker import rerank_with_rrf, RERANK_CANDIDATES
from app.core.embeddings import search_by_embedding
from eval.text_utils import normalize, split_fragments


GOLDEN_PATH = Path("eval/golden.jsonl")
K_VALUES = [3, 5, 10]
MAX_K = max(K_VALUES)


def load_golden() -> list[dict]:
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def all_fragments_found(fragments: list[str], chunks_norm: list[str]) -> bool:
    """True only if EVERY fragment appears in at least one of the chunks."""
    return all(
        any(frag in chunk for chunk in chunks_norm)
        for frag in fragments
    )


def compute_recall(rerank: bool = False) -> dict:
    answerable = [e for e in load_golden() if e.get("answerable")]
    query_embeddings = embed_queries([e["question"] for e in answerable])

    hits = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)

    fetch_k = RERANK_CANDIDATES if rerank else MAX_K
    for e, q_emb in zip(answerable, query_embeddings):
        qtype = e["type"]
        totals[qtype] += 1
        fragments = [normalize(f) for f in split_fragments(e["gold_snippet"])]

        results = search_by_embedding(q_emb, k=fetch_k)   # dense stage
        if rerank:
            results = rerank_with_rrf(e["question"], results)   # cross-encoder + RRF

        chunks_norm = [normalize(r["content"]) for r in results]
        for k in K_VALUES:
            if all_fragments_found(fragments, chunks_norm[:k]):
                hits[qtype][k] += 1

    n_all = totals["single"] + totals["multi"]
    scores = {"single": {}, "multi": {}, "overall": {}}
    for k in K_VALUES:
        scores["single"][k] = hits["single"][k] / totals["single"]
        scores["multi"][k] = hits["multi"][k] / totals["multi"]
        scores["overall"][k] = (hits["single"][k] + hits["multi"][k]) / n_all
    return {"scores": scores, "counts": dict(totals)}

def main():
    print("Measuring dense-only recall (baseline)...")
    dense = compute_recall(rerank=False)["scores"]
    print("Measuring reranked recall...")
    reranked = compute_recall(rerank=True)["scores"]

    print("\n=== Recall@k:  dense -> reranked ===")
    header = f"{'type':<9}" + "".join(f"{('@'+str(k)):>18}" for k in K_VALUES)
    print(header + "\n" + "-" * len(header))
    for qtype in ("single", "multi", "overall"):
        row = f"{qtype:<9}"
        for k in K_VALUES:
            d, r = dense[qtype][k], reranked[qtype][k]
            arrow = "↑" if r > d else ("↓" if r < d else "=")
            row += f"{d:.2f} -> {r:.2f} {arrow}".rjust(18)
        print(row)
        

if __name__ == "__main__":
    main()