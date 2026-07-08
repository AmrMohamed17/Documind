# eval/unanswerable.py
"""
Hallucination-guard score: across the answerable=false questions, how often does
the system correctly REFUSE ("I cannot find...") instead of inventing an answer?

Retrieval always returns its top-k chunks, so a refusal is the *model* judging that
those chunks don't answer the question. This measures that judgment.

Run from repo root:  python -m eval.unanswerable
"""
import json
import time  # add at the top with the other imports
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from app.core.rag import query_documents
from eval.text_utils import normalize, is_refusal

GOLDEN_PATH = Path("eval/golden.jsonl")
K = 4  # same k the app uses by default



def load_unanswerable() -> list[dict]:
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    return [e for e in entries if e.get("answerable") is False]



def main():
    questions = load_unanswerable()
    print(f"Checking {len(questions)} unanswerable questions (k={K})...\n")

    refused = 0
    failures = []  # questions where it did NOT refuse (a hallucination)

    for i, e in enumerate(questions, 1):
        result = query_documents(e["question"], k=K)
        answer = result["answer"]
        ok = is_refusal(answer)
        if ok:
            refused += 1
        else:
            failures.append((e, result))

        verdict = "REFUSED �" if ok else "ANSWERED ✗ (hallucination)"
        print(f"  [{i:>2}/{len(questions)}] {e['id']}: {verdict}")

        time.sleep(13)   # ~4-5 requests/min, stays under the free-tier ceiling


    n = len(questions)
    print(f"\n=== Hallucination-guard score ===")
    print(f"  Correctly refused: {refused}/{n}  ({refused/n:.2%})")

    if failures:
        print(f"\n=== Failures — inspect these ({len(failures)}) ===")
        for e, result in failures:
            print(f"\n  {e['id']}: {e['question']}")
            print(f"  Sources retrieved: {result['sources']}")
            print(f"  Model answered instead of refusing:")
            print(f"    {' '.join(result['answer'].split())[:300]}")
    else:
        print("\n  No hallucinations — the guard held on every unanswerable question.")


if __name__ == "__main__":
    main()