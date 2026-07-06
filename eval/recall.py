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

from app.core.embeddings import similarity_search
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


def compute_recall() -> dict:
    """Run recall and return {type: {k: score}} plus counts. No printing."""
    answerable = [e for e in load_golden() if e.get("answerable")]
    hits = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)

    for e in answerable:
        qtype = e["type"]
        totals[qtype] += 1
        fragments = [normalize(f) for f in split_fragments(e["gold_snippet"])]
        results = similarity_search(e["question"], k=MAX_K)
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
    data = compute_recall()
    s, counts = data["scores"], data["counts"]
    print("\n=== Recall@k ===")
    head = "type        " + "".join(f" @{k:<5}" for k in K_VALUES) + " count"
    print(head + "\n" + "-" * len(head))
    for qtype in ("single", "multi", "overall"):
        n = counts.get(qtype, counts["single"] + counts["multi"])
        row = f"{qtype:<11}" + "".join(f" {s[qtype][k]:<5.2f}" for k in K_VALUES)
        print(row + f" {n}")
        

if __name__ == "__main__":
    main()