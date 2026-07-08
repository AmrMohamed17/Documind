# eval/check_gate.py
"""
CI gate: run recall, compare hard-gated metrics against baseline - tolerance.
Exit 0 (pass) or 1 (fail) so CI can act on it.

Run from repo root:  python -m eval.check_gate
"""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from eval.recall import compute_recall

BASELINE_PATH = Path("eval/baseline.json")


def main():
    baseline = json.loads(BASELINE_PATH.read_text())
    tol = baseline["tolerance"]
    hard = set(baseline["hard_gate"])
    gate_k = set(baseline["gate_k"])
    base = baseline["recall"]

    print("Running recall for the gate...\n")
    current = compute_recall(rerank=True)["scores"]

    print(f"{'metric':<16}{'baseline':>9}{'current':>9}{'floor':>8}{'':>4}status")
    print("-" * 52)

    failed = []
    for qtype in ("single", "multi", "overall"):
        for k_str, base_score in base[qtype].items():
            cur = current[qtype][int(k_str)]
            floor = base_score - tol
            gated = (qtype in hard) and (int(k_str) in gate_k)

            if not gated:
                status = "report"          # multi-hop: shown, never fails
            elif cur >= floor:
                status = "PASS"
            else:
                status = "FAIL"
                failed.append(f"{qtype}@{k_str}: {cur:.2f} < {floor:.2f}")

            label = f"{qtype}@{k_str}"
            print(f"{label:<16}{base_score:>9.2f}{cur:>9.2f}"
                  f"{floor:>8.2f}{'':>4}{status}")

    print("-" * 52)
    if failed:
        print("\n❌ GATE FAILED — retrieval regressed:")
        for f in failed:
            print(f"   {f}")
        sys.exit(1)
    print("\n✅ GATE PASSED — no regression beyond tolerance.")
    sys.exit(0)


if __name__ == "__main__":
    main()