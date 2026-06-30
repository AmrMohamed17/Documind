# eval/validate_golden.py
"""
Validates the golden dataset against the source corpus AND its own schema.

Two jobs:
  1. Structure — type/source/answerable consistency, unique ids, REAL counts.
  2. Snippets  — every gold_snippet fragment actually appears in its source file,
                 so the recall metric can never wrongly fail a question the
                 retriever actually got right.

Run from the repo root:  python eval/validate_golden.py
"""
import json
import re
import sys
from pathlib import Path
from collections import Counter

GOLDEN_PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "eval/golden.jsonl")


def normalize(text: str) -> str:
    """Make text comparable the SAME way the recall metric will:
    unify quotes/dashes, strip invisible Markdown markup, collapse whitespace, lowercase."""
    # unify fancy quotes / dashes
    text = (text.replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2019", "'").replace("\u2018", "'")
                .replace("\u2014", "-").replace("\u2013", "-"))
    # flatten Markdown links  [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # strip inline-code backticks and emphasis markers (invisible when rendered)
    text = text.replace("`", "").replace("*", "")
    # collapse all whitespace to single spaces, lowercase
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def split_fragments(snippet: str) -> list[str]:
    """A snippet may join several spans with '...'. Split into the real spans."""
    return [p for p in re.split(r"\s*\.\.\.\s*", snippet) if p.strip()]


# ── Load dataset ────────────────────────────────────────────────────
entries = []
with open(GOLDEN_PATH, encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"FATAL: line {line_no} is not valid JSON: {e}")
            sys.exit(1)
print(f"Loaded {len(entries)} entries from {GOLDEN_PATH}\n")

# Read & normalize each source file once.
_cache: dict[str, str | None] = {}
def load_source(path: str) -> str | None:
    if path not in _cache:
        p = Path(path)
        _cache[path] = normalize(p.read_text(encoding="utf-8")) if p.exists() else None
    return _cache[path]


structural, elision, missing = [], [], []
counts, ids_seen = Counter(), Counter()

for e in entries:
    qid = e.get("id", "<no id>")
    ids_seen[qid] += 1
    answerable = e.get("answerable")
    qtype = e.get("type")
    source = e.get("gold_source", "")
    snippet = e.get("gold_snippet", "")

    counts["unanswerable" if answerable is False else f"{qtype} (answerable)"] += 1

    if answerable is False:
        if source != "N/A":
            structural.append((qid, f"unanswerable but gold_source='{source}' (should be N/A)"))
        if snippet != "N/A":
            structural.append((qid, "unanswerable but gold_snippet is not N/A"))
        continue

    if source == "N/A":
        structural.append((qid, "answerable but gold_source is N/A"))
        continue

    sources = [s.strip() for s in source.split(",")]
    if qtype == "single" and len(sources) != 1:
        structural.append((qid, f"type=single but has {len(sources)} sources"))
    if qtype == "multi" and len(sources) < 2:
        structural.append((qid, f"type=multi but has {len(sources)} source(s) (needs >=2)"))

    norm_sources = {}
    for s in sources:
        content = load_source(s)
        if content is None:
            structural.append((qid, f"source file not found: {s}"))
        else:
            norm_sources[s] = content

    fragments = split_fragments(snippet)
    if qtype == "single" and len(fragments) > 1:
        elision.append(qid)

    for frag in fragments:
        nfrag = normalize(frag)
        if not any(nfrag in content for content in norm_sources.values()):
            missing.append((qid, source, frag[:70]))


# ── Report ──────────────────────────────────────────────────────────
print("=== Real counts ===")
for k, v in counts.items():
    print(f"  {k:24} {v}")
print(f"  {'total':24} {len(entries)}\n")

if dupes := [i for i, n in ids_seen.items() if n > 1]:
    print(f"=== Duplicate ids ===\n  {dupes}\n")

print(f"=== Structural problems ({len(structural)}) ===")
for qid, msg in structural:
    print(f"  [{qid}] {msg}")

print(f"\n=== Single-hop snippets containing '...' ({len(elision)}) — should be one span ===")
print(f"  {elision}")

print(f"\n=== Snippet fragments NOT found in source ({len(missing)}) ===")
for qid, src, frag in missing:
    print(f'  [{qid}] not in {src}: "{frag}..."')

clean = not (structural or missing or dupes)
print("\nRESULT:", "PASS" if clean else "ISSUES FOUND — fix the entries above")
if elision and clean:
    print("       (but clean up the single-hop '...' snippets too)")