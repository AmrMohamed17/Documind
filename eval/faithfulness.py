# eval/faithfulness.py
"""
Faithfulness: for each answerable question, is the generated answer grounded in the
retrieved context? Scored 1-5 by an LLM-as-judge (one judge call per question).

Refusals on answerable questions are recorded separately (coverage gap, not judged).
Resumable: appends each result to eval/faithfulness_results.jsonl and skips done ids
on restart (free-tier generation is 5/min, so this runs slow).

Run from repo root:  python -m eval.faithfulness
"""
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from app.core.embeddings import similarity_search
from app.core.rag import build_context, generate_answer
from app.core.gemini import get_client
from google.genai import errors
from eval.text_utils import is_refusal
from app.core.deepseek import deepseek_generate   # ← DeepSeek for the judge

GOLDEN_PATH = Path("eval/golden.jsonl")
RESULTS_PATH = Path("eval/faithfulness_results.jsonl")
JUDGE_MODEL = "gemini-2.5-flash"
K = 4

JUDGE_PROMPT = """You evaluate whether an ANSWER is faithful to (supported by) a CONTEXT.

Judge ONLY grounding: is every claim in the answer supported by the context? An answer can be true in general but still UNFAITHFUL if the context doesn't support it. Ignore writing quality.

Scale:
5 = every claim directly supported by the context
4 = core claims supported; a minor detail unsupported
3 = mix of supported and unsupported claims
2 = mostly unsupported
1 = contradicts the context or entirely unsupported

CONTEXT:
{context}

ANSWER:
{answer}

Respond with ONLY a JSON object and nothing else:
{{"score": <1-5>, "reason": "<one short sentence>"}}"""


FAITHFULNESS_ANSWER_PROMPT = """You are a helpful assistant that answers questions based strictly on the provided context.
If the answer cannot be found in the context, say "I cannot find the answer in the provided documents."
Do not use any knowledge outside the provided context.

Context:
{context}

Question: {question}

Answer:"""


def judge_faithfulness(context: str, answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(context=context, answer=answer)
    raw = ""
    try:
        raw = (deepseek_generate(prompt) or "").strip()
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start:end + 1])
        return {"score": int(data["score"]), "reason": str(data.get("reason", ""))}
    except (json.JSONDecodeError, KeyError, ValueError):
        return {"score": None, "reason": f"unparseable: {raw[:120]}"}
    

def load_answerable() -> list[dict]:
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return [e for e in (json.loads(l) for l in f if l.strip()) if e.get("answerable")]


def load_done() -> set:
    if not RESULTS_PATH.exists():
        return set()
    with open(RESULTS_PATH, encoding="utf-8") as f:
        return {json.loads(l)["id"] for l in f if l.strip()}


def main():
    questions = load_answerable()
    done = load_done()
    todo = [e for e in questions if e["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go (of {len(questions)} answerable)\n")

    with open(RESULTS_PATH, "a", encoding="utf-8") as out:
        for i, e in enumerate(todo, 1):
            results = similarity_search(e["question"], k=K)
            context = build_context(results)
            answer = deepseek_generate(FAITHFULNESS_ANSWER_PROMPT.format(
                            context=context, question=e["question"]))            

            if is_refusal(answer):
                rec = {"id": e["id"], "type": e["type"], "score": None,
                       "refused": True, "reason": "refused on answerable"}
                print(f"  [{i}/{len(todo)}] {e['id']}: REFUSED (coverage gap)")
            else:
                v = judge_faithfulness(context, answer)         # call 2
                rec = {"id": e["id"], "type": e["type"], "score": v["score"],
                       "refused": False, "reason": v["reason"]}
                print(f"  [{i}/{len(todo)}] {e['id']}: score={v['score']}")

            out.write(json.dumps(rec) + "\n")
            out.flush()   # persist immediately so a crash keeps progress

    report()


def report():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    scored = [r for r in rows if r["score"] is not None]
    refused = [r for r in rows if r.get("refused")]
    bad = [r for r in rows if not r.get("refused") and r["score"] is None]

    print("\n=== Faithfulness ===")
    if scored:
        mean = sum(r["score"] for r in scored) / len(scored)
        faithful = sum(1 for r in scored if r["score"] >= 4)
        print(f"  Mean score (1-5):      {mean:.2f}  over {len(scored)} answered")
        print(f"  Faithful (>=4):        {faithful}/{len(scored)} ({faithful/len(scored):.0%})")
    print(f"  Refused on answerable: {len(refused)}  (coverage gap, not judged)")
    if bad:
        print(f"  Unparseable: {len(bad)} — review {[r['id'] for r in bad]}")


if __name__ == "__main__":
    main()