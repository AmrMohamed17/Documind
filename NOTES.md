# DocuMind v2 — Build Notes & Decisions

Running log of architectural decisions, measured baselines, and interview talking points.
Kept alongside the code so the "why" is never lost. Updated as the build progresses.

---

## Positioning

Evaluation-first, production-grade RAG. The differentiator is not "it answers questions
about documents" — it's the measurement, CI gating, and observability wrapped around it.

---

## Phase 0 — Foundations (DONE)

**What changed and why:**

- **Embeddings: local `all-MiniLM-L6-v2` → Gemini `gemini-embedding-001` API.**
  - Reason: dropped `torch` + the entire CUDA stack (gigabytes), making the app light
    enough to run on a small AWS EC2 instance (~1 GB RAM would OOM with a local model).
  - Dimensions: **768** via `output_dimensionality` (model default is 3072). Chosen for
    the storage/latency vs. quality sweet spot; the model supports truncation via
    Matryoshka representation learning.
  - Gotcha handled: gemini-embedding-001 does **not** normalize truncated vectors, so we
    normalize to unit length ourselves. `output_dimensionality` must be passed on the
    embed *call*, not the client constructor.
  - Task types: `RETRIEVAL_DOCUMENT` for chunks, `RETRIEVAL_QUERY` for questions
    (asymmetric embedding improves retrieval).

- **Vector store: ChromaDB → PostgreSQL + pgvector.**
  - Reason: one system instead of two; durable persistence (Chroma's local dir gets wiped
    on ephemeral hosts); and the "vector search in Postgres" story signals engineering
    maturity (extend the boring battle-tested DB rather than bolt on a trendy one).
  - Distance: cosine (`<=>`). Vectors normalized, so cosine is clean.
  - Index: **none yet** (brute-force exact search). Fine at 667 chunks. Add an HNSW index
    (`CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)`) when the corpus grows;
    that becomes a measured before/after (latency down, recall ~unchanged).

- **DB access: raw SQL via `psycopg` (v3) + parameterized queries + connection pool.**
  - Reason: transparent, explainable, SQL-native for pgvector; ORM is overkill for one
    table. Parameterized queries (`%s`) prevent SQL injection.
  - Pool (`psycopg_pool`) so concurrent requests don't fight over one connection;
    `configure=register_vector` teaches every pooled connection the `vector` type.

- **API: sync `def` endpoints (not `async def`).**
  - Reason: all I/O here is blocking (psycopg + Gemini SDK are synchronous). `async def`
    with blocking calls and no `await` freezes the event loop. Plain `def` lets FastAPI
    run handlers in a threadpool → real concurrency without async-native libraries.

- **Removed MLflow.** It logged latency/chunk-count (operational), not quality; and its
  artifact store pointed at a now-dead GCS bucket. Real observability returns later as
  Langfuse tracing; real *quality* numbers come from the eval harness.

- **Deploy target: GCP → AWS EC2** (GCP free trial expired; AWS trial + $200 credits).
  Also makes the CV's "AWS (EC2)" claim real. Guardrail: keep the demo footprint tiny
  (small EC2 + docker-compose'd pgvector); reserve most credits for burst compute
  (e.g. a GPU box for the QLoRA sequel), not a 24/7 web server.

- **Runs from a clean clone via `docker compose up`** (app + pgvector). `db/init.sql`
  auto-creates the `vector` extension and `chunks` table on first boot. Inside the compose
  network the DB host is the service name `db`, not `localhost`.

**Stack now:** Python (typed), FastAPI, PostgreSQL + pgvector, psycopg3 + pool,
google-genai (Gemini), Docker + docker-compose, GitHub Actions (CI to be reworked).

---

## Phase 2 — Golden Dataset (DONE)

- **`eval/golden.jsonl`** — 100 questions over the FastAPI tutorial docs corpus
  (40 `.md` files → 667 chunks, chunk_size 500 / overlap 50).
- **Counts (validator-verified):** 71 single-hop, 14 multi-hop, 15 unanswerable.
  (Started as "15 multi"; one entry — q083 — was a mislabeled single-source question, so
  the honest count is 14. A truthful 14 beats an inflated 15.)
- **Schema per entry:** `id`, `question`, `reference_answer`, `gold_source`,
  `gold_snippet`, `type` (single/multi), `answerable` (bool).
- **Unanswerable questions** are cleanly out-of-corpus (Django ORM, MongoDB/Motor,
  WebSockets, Auth0, Sentry, etc.), `gold_source: N/A`. They test the "I don't know" path,
  not recall.

**The validator (`eval/validate_golden.py`) — key story:**
- Checks (a) structural consistency (type ↔ source count, answerable ↔ N/A) and
  (b) that every `gold_snippet` fragment actually appears in its `gold_source` file.
- **88 → 6 → 0:** first run failed 88/100. Almost all were false failures — snippets were
  clean prose but the source files are Markdown full of invisible backticks/emphasis, so
  the matcher missed. Fixed by normalizing markup on both sides → dropped to 6 → those 6
  were genuinely wrong entries (AI-written paraphrases posing as verbatim) → fixed by hand
  against the source → 0.
- **Lesson (interview-quotable):** when a check fails on ~100% of cases, the check is
  broken, not the world. And never trust an eval set you haven't verified with code.
- Snippet-matching handles `...`-joined multi-fragment snippets by splitting and checking
  each fragment.

**Shared definition of "found":** `eval/text_utils.py` holds `normalize` +
`split_fragments`, imported by BOTH the validator and the recall metric — so "found" means
exactly the same thing in validation and in scoring (they can't drift).

---

## Phase 3 — Eval Harness (DONE)

Three metrics, all reusing one shared definition of "found" (`eval/text_utils.normalize`):

| Metric              | Result (baseline, dense retrieval, reranker)      |
|---------------------|---------------------------------------------------|
| Recall@k (single)   | 0.93 / 0.93 / 0.97  (@3 / @5 / @10)               |
| Recall@k (multi)    | 0.14 / 0.29 / 0.36 — all misses = ranking problems|
| Hallucination guard | 15/15 unanswerable questions all correctly refused|
| Faithfulness        | 4.82/5 mean, 95% >=4 (LLM-as-judge, DeepSeek)     |

Headline: strong single-hop retrieval and strong groundedness; multi-hop recall is limited
purely by ranking (diagnosed, not assumed) — the evidence-backed justification for the
Phase 5 reranker, with these numbers as the "before" to beat.

### recall@k (DONE)

**Definition:** fraction of answerable questions where all gold snippet fragments appear
(substring, after `normalize`) in the top-k retrieved chunks.
- single-hop: the one fragment must be found.
- multi-hop: **every** fragment must be found across the top-k (strict).
  Chosen because a multi-hop question is only answerable if all its pieces are retrieved;
  partial credit would reward retrieval that couldn't produce a correct answer.
- unanswerable: excluded (no gold passage) — scored separately.
- **No LLM in the loop** — pure text-matching arithmetic (exact, cheap, reproducible).
  Gemini is used only to embed the question for the vector search.

**Baseline (plain dense retrieval, no reranker):**

| type    | @3   | @5   | @10  | count |
|---------|------|------|------|-------|
| single  | 0.93 | 0.93 | 0.97 | 71    |
| multi   | 0.14 | 0.29 | 0.36 | 14    |
| overall | 0.80 | 0.82 | 0.87 | 85    |

**Diagnosis of the low multi-hop number (fully checked, all misses):**
- Built `eval/inspect_question.py` to trace each miss back to the DB and classify it as a
  **ranking problem** (the correct chunk exists but ranked below k) vs a **chunking
  artifact** (gold text split across chunks; can never match).
- **Every multi-hop miss is a ranking problem.** Zero chunking artifacts, zero bad labels.
- **Meaning:** the right passages ARE retrieved; dense search just ranks the 2nd one too
  low. This is exactly what a reranker fixes → evidence-backed justification for Phase 5,
  with 0.14/0.29/0.36 as the before-number to beat.

**Interview paragraph (recall):**
> Single-hop recall was strong (0.93–0.97). Multi-hop recall@3 was only 0.14 — but instead
> of assuming the retriever was broken, I built an inspector that traced every miss back to
> the database and confirmed all of them were ranking problems, not retrieval or data
> failures: the right passages were retrieved, just ranked below the cutoff. That gave me a
> measured, diagnosed reason to add a reranker, and a baseline to beat.


### Unanswerable check / hallucination guard (DONE)

**Definition:** across the 15 answerable=false questions (out-of-corpus: Django, MongoDB,
WebSockets, Auth0, Kafka, etc.), how often does the system correctly REFUSE instead of
fabricating an answer? Detected via flexible string match on refusal markers (no LLM judge).

**Key insight:** retrieval ALWAYS returns its top-k chunks (never empty) — so a refusal is
the *model* judging that the retrieved chunks don't answer the question, not the DB coming
up empty. This measures that judgment under pressure.

**Result: 15/15 (100%) correctly refused. Zero hallucinations.**
- Notably, no failures on topically-adjacent retrieval (e.g. a MongoDB question pulling the
  SQL-databases chunk and being seduced into fabricating from it) — the guard held.
- Honest caveat: 15 is a small sample; claim is "held on all 15 eval cases," not "never
  hallucinates." Hardening = more unanswerable questions (post-launch).

**Bottleneck note:** free-tier Gemini generation is 5 req/min. Added retry-with-backoff to
`generate_answer` + a ~13s pace delay between questions. Generation can't be batched (unlike
embeddings), so faithfulness (runs generation on all 85 answerable Qs) will be slow — build
it resumable.


### Faithfulness (DONE)

**Definition:** for each answerable question, is the generated answer grounded in the
retrieved context? Scored 1-5 by an LLM-as-judge (one judge call per question). Judges
ONLY grounding — an answer can be factually true in general but still UNFAITHFUL if the
retrieved context doesn't support it (catches the model answering from training memory
rather than from what retrieval actually returned).

**Design decisions:**
- **Hand-rolled single-call judge, not RAGAS.** RAGAS makes multiple LLM calls per question
  (claim decomposition + per-claim verification), impractical on free-tier rate limits.
  A single-call judge is cheaper, explainable, and calibratable. (RAGAS is the right tool
  with paid infra or when you need the full metric suite — a deliberate tradeoff, not
  framework-avoidance.)
- **Refusals on answerable questions are excluded from the score, counted separately.**
  A refusal makes no claims, so a judge would score it a vacuous 5 — leaving them in would
  inflate faithfulness and hide a real coverage problem. Pulling them out keeps the score
  honest (groundedness of genuine answers only).
- **Resumable runner** (`eval/faithfulness_results.jsonl`, append + flush per question,
  skip done ids on restart) because the run is long and generation-heavy.

**Provider note:** free-tier Gemini has a ~20 generation-requests-PER-DAY cap that made the
full run impractical (stalled for ~30 min per few questions). Switched BOTH answer
generation and judging to DeepSeek (OpenAI-compatible API) for the eval — cost a few cents.
Embeddings stayed on Gemini. Discarded the 22 partial Gemini-judged results and re-ran all
85 on DeepSeek so the whole set is judged by ONE consistent instrument (a metric is only
comparable to itself if the judge stays constant).
Honest framing: eval answers were generated by DeepSeek; the live app generates with Gemini.
Fine for measuring retrieval groundedness (question is "given this context, is the answer
faithful," independent of which model wrote it).

**Result (DeepSeek judge, k=4):**

| metric                | value            |
|-----------------------|------------------|
| Mean score (1-5)      | 4.82 over 78     |
| Faithful (>=4)        | 74/78 (95%)      |
| Refused on answerable | 7 (coverage gap) |

Strong groundedness — when the system answers, it stays on its retrieved context. This is
the quantified version of the anti-hallucination grounding prompt working.

**Coverage gaps (the 7 refusals on answerable questions):**
- 5 of 7: gold chunk ranked below k=4 (retrieval ranking problem) → reranker candidates,
  same root cause as the multi-hop recall misses. No action; expect Phase 5 to recover some.
- 2 of 7: gold chunk WAS in the top-k (sufficient context) but the DeepSeek eval pipeline
  refused anyway. A Gemini spot-check on the same context answered correctly → model-
  dependent over-refusal on borderline context, NOT a retrieval or dataset problem.
  Decision: documented, not fixed — 2/85 is noise-level, the effect is likely DeepSeek-
  specific (production model is Gemini), and tuning the prompt against 2 eval cases would
  risk overfitting the prompt to the eval set. Reinforces why eval model-consistency matters.

**Interview paragraph (faithfulness):**
I scored answer groundedness with a hand-rolled LLM-as-judge — one call per question, rating 1-5 whether each answer is supported by its retrieved context, deliberately judging grounding rather than general correctness so I'd catch the model answering from memory. I chose a single-call judge over RAGAS because RAGAS's multi-call approach was impractical on my rate limits, and hand-rolling let me explain and calibrate the metric. Mean faithfulness was 4.82/5 with 95% of answers scoring >=4. Seven answerable questions were refused — I kept those out of the faithfulness average (a refusal is vacuously faithful) and analyzed them separately: five were retrieval ranking problems, two were model-dependent over-refusals I chose to document rather than overfit my prompt to.

**Calibration (deferred — post-launch):** the judge's scores were not formally validated against human labels (planned: hand-score ~15, measure judge-human agreement %). Informal check only — spot-read the judge's low scores and they matched my own reading. Formal calibration is the documented next step before over-trusting the absolute number; the result is used directionally (strong groundedness) rather than as a precise ground truth.

---

## On the horizon (not started)

- **Phase 4 — CI eval gate:** run the eval on PRs; fail if scores drop below committed
  thresholds. THE single most interview-quotable feature. Demonstrate by opening a PR that
  degrades retrieval and showing CI go red, then green after the fix.
- **Phase 5 — Reranker:** over-retrieve (top-N) → rerank → keep top-k. Headline metric:
  multi-hop recall before/after (predicted to jump, per the diagnosis above).
- **Phase 6 — Deploy to AWS EC2 + light Langfuse tracing.** Live demo link for the CV.
- **Phase 7 — Metrics-first README + architecture diagram + 2-min video → START APPLYING.**

**Explicitly post-launch (do NOT let these delay applying):** hybrid search + RRF,
HNSW-vs-IVFFlat benchmark, bilingual Arabic, QLoRA sequel, multi-agent project.

---

## Standing principles

- Evidence is the deliverable, not the code. Every claim ships with a measured number.
- Never publish an unmeasured metric.
- Report single-hop and multi-hop recall separately — the gap is the insight, a blended
  number hides it.
- Diagnose before fixing: when a check fails on nearly everything, suspect the check.