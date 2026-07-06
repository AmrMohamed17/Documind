# eval/faithfulness_ci.py
"""
Faithfulness SOFT report for CI: scores a small, fixed subset and prints the result.
Never fails the build — this is a visibility signal, not a gate.

Run from repo root:  python -m eval.faithfulness_ci
"""
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from app.core.embeddings import similarity_search
from app.core.rag import build_context
from app.core.deepseek import deepseek_generate
from eval.text_utils import is_refusal
from eval.faithfulness import FAITHFULNESS_ANSWER_PROMPT, judge_faithfulness

GOLDEN_PATH = Path("eval/golden.jsonl")
K = 4
SUBSET_SIZE = 12
BASELINE_MEAN = 4.82   # from the full local run, for context


def load_answerable() -> list[dict]:
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return [e for e in (json.loads(l) for l in f if l.strip()) if e.get("answerable")]


def main():
    answerable = load_answerable()
    # Deterministic spread: every Nth question, capped at SUBSET_SIZE.
    # Same questions every run -> the CI number is comparable across PRs.
    stride = max(1, len(answerable) // SUBSET_SIZE)
    subset = answerable[::stride][:SUBSET_SIZE]

    print(f"Faithfulness soft-report on {len(subset)} fixed questions "
          f"(non-blocking)...\n")

    scores, refused = [], 0
    for e in subset:
        results = similarity_search(e["question"], k=K)
        context = build_context(results)
        answer = deepseek_generate(
            FAITHFULNESS_ANSWER_PROMPT.format(context=context, question=e["question"]))

        if is_refusal(answer):
            refused += 1
            print(f"  {e['id']}: refused")
            continue

        v = judge_faithfulness(context, answer)
        if v["score"] is not None:
            scores.append(v["score"])
        print(f"  {e['id']}: score={v['score']}")

    print("\n=== Faithfulness (soft report — does NOT fail the build) ===")
    if scores:
        mean = sum(scores) / len(scores)
        good = sum(1 for s in scores if s >= 4)
        print(f"  Mean: {mean:.2f}   >=4: {good}/{len(scores)}   refused: {refused}")
        print(f"  (full-run baseline mean was {BASELINE_MEAN} — investigate if this "
              f"drops notably)")
    else:
        print("  No scored answers this run.")


if __name__ == "__main__":
    main()