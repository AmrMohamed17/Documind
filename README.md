# DocuMind

**Evaluation-first RAG.** Every quality claim below is measured — and a CI gate blocks any pull request that makes retrieval worse.

**Live demo → [documind.amr-mohammed.com](https://documind.amr-mohammed.com)** · [Case study](https://amr-mohammed.com/documind)

---

## Measured results

Retrieval is evaluated against a hand-built, programmatically-validated 100-question golden dataset over the FastAPI documentation (40 files → 667 chunks).

**Retrieval — recall@k** *(dense baseline → two-stage: rerank + RRF)*

| question type | k=3 | k=5 | k=10 |
|---|---|---|---|
| single-hop (71) | 0.93 → 0.90 | 0.93 → 0.92 | **0.97 → 0.97** |
| multi-hop (14) | 0.14 → 0.21 | 0.29 → 0.36 | **0.36 → 0.57** |

Multi-hop recall@10 improved **58%** after adding the reranker, while single-hop held flat at 0.97.

**Generation**

| metric | result |
|---|---|
| Hallucination guard | **15 / 15** unanswerable questions correctly refused |
| Answer faithfulness | **4.82 / 5** mean · 95% (74/78) scored ≥ 4 |

No LLM is involved in the recall metric — it's pure text-matching arithmetic, exact and reproducible. Faithfulness uses a single-call LLM-as-judge (spot-checked, not formally calibrated, so it's used directionally).

**Latency** *(retrieval pipeline, n=20, local)*

| stage | p50 | p95 |
|---|---|---|
| Embed query (Gemini API) | 457 ms | 826 ms |
| Dense search (pgvector) | **11 ms** | 50 ms |
| Rerank + RRF (FlashRank, local) | 343 ms | 566 ms |
| **Retrieval total** | **809 ms** | 1441 ms |

Vector search itself is essentially free — 11 ms over 667 chunks, no index needed at this
scale. The dominant cost is the embedding API round-trip, not compute. Reranking adds
~340 ms, which is the price of the multi-hop gain (0.36 → 0.57 recall@10).

End-to-end latency is retrieval + generation; generation (Gemini) dominates and is not
included here.

---

## What it does

DocuMind answers questions over a document corpus using retrieval-augmented generation. It grounds every answer in retrieved passages, cites its sources, and **refuses when the answer isn't in the corpus** instead of fabricating one.

```
Q: "How do you declare a required query parameter?"

A: "Declare the parameter in the function with no default value.
    Without a default, FastAPI treats the query parameter as required."

    Sources: query-params.md   Chunks used: 4
```

```
Q: "How do I connect FastAPI to MongoDB?"

A: "I cannot find the answer in the provided documents."
```

That second one is the point. A model answering from its own training knowledge would happily explain MongoDB. The refusal is what proves the answers are actually coming from the retrieved documents.

---

## Architecture

```
                     Browser (Caddy → HTTPS)
                              │
                     ┌────────▼─────────┐
                     │     FastAPI      │
                     │  /query  /health │
                     └────────┬─────────┘
                              │
        ┌─────────────────────▼──────────────────────┐
        │            Two-stage retrieval             │
        │                                            │
        │  1. embed question   → Gemini (768d)       │
        │  2. dense search     → pgvector (top 12)   │
        │  3. rerank           → FlashRank           │
        │     cross-encoder      (local ONNX)        │
        │  4. fuse rankings    → Reciprocal Rank     │
        │                        Fusion → top k      │
        └─────────────────────┬──────────────────────┘
                              │
                     ┌────────▼─────────┐
                     │  Gemini 2.5 Flash │
                     │  grounding prompt │
                     │  + citations      │
                     └───────────────────┘

  ┌──────────────────────────────────────────────────┐
  │  Evaluation (eval/)                              │
  │    golden.jsonl  → 100 validated questions       │
  │    recall.py     → recall@k (no LLM, exact)      │
  │    unanswerable  → hallucination guard           │
  │    faithfulness  → LLM-as-judge                  │
  │    check_gate.py → fails CI on regression        │
  └──────────────────────────────────────────────────┘
```

**Why a reranker.** Dense search embeds the query and each chunk *separately*, which is fast but coarse. A cross-encoder reads the query and a candidate chunk *together*, so it judges relevance far more precisely — too slowly to run over 667 chunks, but perfect for re-scoring 12 candidates. Reciprocal Rank Fusion then merges the dense and reranked orders so a chunk ranked highly by *either* signal survives.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI (sync endpoints) | The I/O is blocking; `async def` with blocking calls freezes the event loop. Sync handlers run in FastAPI's threadpool. |
| Vector store | PostgreSQL + **pgvector** | One durable system instead of two. Vector search is SQL-native (`<=>`), so filtering and joins come free. |
| DB access | psycopg 3 + connection pool | Raw parameterized SQL — transparent, injection-safe, no ORM overhead for a one-table schema. |
| Embeddings | `gemini-embedding-001` @ **768d** | Matryoshka-truncated from 3072 and re-normalized. Task-type asymmetry: `RETRIEVAL_DOCUMENT` for chunks, `RETRIEVAL_QUERY` for questions. |
| Reranker | **FlashRank** (ONNX, local) | Lightweight cross-encoder — no `torch`, runs on CPU, keeps the deployed image small. |
| Generation | Gemini 2.5 Flash | Strict grounding prompt with an explicit refusal path. |
| Eval judge | DeepSeek | Free-tier Gemini's daily generation cap made full eval runs impractical. |
| Deploy | Docker Compose + Caddy on AWS EC2 | Caddy handles automatic HTTPS (Let's Encrypt, auto-renewing). |
| CI | GitHub Actions | Unit tests → recall gate → faithfulness report, on every PR. |

---

## The CI quality gate

The centerpiece. On every pull request, GitHub Actions:

1. Runs the unit test suite (fast, mocked, no API keys needed)
2. Spins up a pgvector database and **loads pre-computed corpus embeddings** from a committed SQL fixture — no re-embedding, so runs are cheap and deterministic
3. Runs the recall suite and **fails the build if retrieval regresses** below the committed baseline
4. Reports faithfulness as a non-blocking signal

**A change that quietly makes retrieval worse cannot be merged.**

Only the deterministic metrics are hard-gated. Multi-hop recall is reported but not gated (n=14 — one question swings it 7 points, too noisy for a pass/fail). Faithfulness runs through an LLM judge, so it's reported, never blocking. *A gate you can't trust is worse than no gate.*

---

## Evaluation

The golden dataset is the foundation — and it's validated by code, not trust:

```bash
python -m eval.validate_golden   # every gold answer must exist in its source file
python -m eval.recall            # recall@k, dense vs reranked
python -m eval.unanswerable      # hallucination guard
python -m eval.faithfulness      # LLM-as-judge (resumable)
python -m eval.check_gate        # the CI gate, locally
python -m eval.inspect_question q071   # trace why one question failed
```

`validate_golden` checks that every gold snippet actually appears in the file it claims to come from. Its first run flagged 88 of 100 — almost all false failures from a Markdown-markup mismatch. Fixing the text normalization dropped it to 6 genuinely wrong entries, corrected by hand, to a final 0. The same normalization is shared by the validator and every metric, so *"found"* means the same thing everywhere.

---

## Run it locally

**Requirements:** Docker, a [Gemini API key](https://aistudio.google.com/app/apikey) (free tier is fine).

```bash
git clone https://github.com/AmrMohamed17/Documind.git
cd Documind

cp .env.example .env.docker      # add your GEMINI_API_KEY
docker compose up -d --build
```

Load the corpus (pre-computed embeddings — no API calls, instant):

```bash
docker compose exec -T db psql -U documind -d documind < db/fixtures/chunks_seed.sql
```

Open **http://localhost:8000**.

To ingest your own documents instead, drop `.md`, `.txt`, or `.pdf` files into `corpus/` and run:

```bash
python ingest_corpus.py corpus
```

**Tests:**

```bash
pytest -v      # fast, mocked, no database or API key required
```

---

## API

| Endpoint | Description |
|---|---|
| `POST /api/v1/query` | Ask a question. Returns a grounded answer, sources, and chunk count. |
| `POST /api/v1/ingest` | Ingest a document. *(Disabled in the public demo via `DEMO_MODE`.)* |
| `GET /api/v1/documents` | List the documents in the corpus. |
| `GET /health` | Health check. |

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does response_model do?", "k": 4}'
```

---

## Decisions

**Postgres + pgvector over a dedicated vector DB.** One system, durable persistence, SQL-native filtering. A purpose-built vector database earns its keep at very large scale or query volume — not at 667 chunks.

**Hand-rolled metrics over RAGAS.** Recall is exact arithmetic; a framework would add a dependency and hide the logic. RAGAS's multi-call faithfulness judging was impractical under free-tier rate limits, so the judge is a single call I wrote and can explain. RAGAS is the right tool with paid infrastructure or a fuller metric suite.

**Strict multi-hop scoring.** A multi-hop question counts as a hit only if *every* required passage is retrieved — partial credit would reward retrieval that couldn't actually produce a correct answer.

**Frozen corpus embeddings.** CI loads a committed SQL fixture instead of re-embedding. Cheaper, faster, and it makes recall deterministic run-to-run — which removed the embedding jitter that had caused a false-alarm gate failure at the tightest cutoff.

**Rank fusion, not rank replacement.** Naive reranking helped multi-hop but *hurt* single-hop, because it overwrote a first stage that was already getting it right. RRF keeps both signals.

---

## Known limitations

- **Multi-hop retrieval is still the weak point** (0.57 @ k=10). The reranker lifted it substantially, but hybrid search (BM25 + dense) is the natural next step.
- **The faithfulness judge is not formally calibrated** against human labels — it's spot-checked and used directionally, not as precise ground truth.
- **The hallucination guard is measured on 15 questions.** It held on all of them; that's not the same as "never hallucinates."
- **Two of 85 answerable questions were over-refused** by the eval pipeline despite sufficient retrieved context — a model-dependent effect, documented rather than patched (tuning the prompt against two eval cases would risk overfitting).
- **Uploads are disabled in the public demo** (`DEMO_MODE=true`) — a public, unauthenticated ingest endpoint would burn API quota and mix strangers' documents into one shared corpus. Session-scoped uploads are the next enhancement.

---

## Project structure

```
app/
  api/routes.py        FastAPI endpoints + Pydantic models
  core/
    db.py              pgvector connection pool
    embedder.py        Gemini embeddings (batched, normalized)
    embeddings.py      store + two-stage retrieval
    reranker.py        FlashRank cross-encoder + RRF fusion
    rag.py             chunking, context building, grounded generation
    gemini.py          shared Gemini client
    deepseek.py        eval-only judge client
eval/
  golden.jsonl         100 validated questions
  validate_golden.py   dataset ↔ corpus validator
  recall.py            recall@k (dense vs reranked)
  unanswerable.py      hallucination guard
  faithfulness.py      LLM-as-judge (resumable)
  check_gate.py        the CI gate
  inspect_question.py  per-question failure tracer
  baseline.json        the numbers the gate defends
db/
  init.sql             schema (pgvector extension + chunks table)
  fixtures/            frozen corpus embeddings
corpus/                the FastAPI documentation
tests/                 unit tests (mocked, fast)
```

---

## License

MIT
