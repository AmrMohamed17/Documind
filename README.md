# DocuMind
### Enterprise RAG Platform with MLOps

A production-ready Retrieval Augmented Generation system that enables semantic search and AI-powered question answering over unstructured corporate documents. Fully automated from experiment tracking to cloud deployment — built from scratch with no shortcuts.

**Live demo:** `http://34.133.130.98:8000`  
**API docs:** `http://34.133.130.98:8000/docs`

---

## What it does

Upload any PDF or TXT document and ask questions about it in plain English. DocuMind chunks and embeds your documents into a vector database, retrieves the most semantically relevant passages, and uses Gemini to generate a grounded, accurate answer — no hallucinations, no keyword matching.

```
User:     "What is a binary search tree?"

DocuMind: "The property that makes a binary tree into a binary search tree is
           that for every node X in the tree, the values of all items in its
           left subtree are smaller than the item in X, and the values of all
           items in its right subtree are larger than the item in X."

           Sources: DataStructure.pdf  |  Chunks used: 4  |  Latency: 1.2s
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Browser UI                             │
│              (Served directly from FastAPI at /)              │
└──────────────────────────┬───────────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼───────────────────────────────────┐
│                  FastAPI (Async REST API)                      │
│  POST /api/v1/ingest  POST /api/v1/query  GET /api/v1/documents│
└───────┬──────────────────────┬───────────────────────────────┘
        │                      │
┌───────▼──────────┐  ┌────────▼────────────────────────────┐
│   Ingestion      │  │         Query Pipeline               │
│                  │  │                                      │
│ 1. Load document │  │ 1. Embed question                    │
│ 2. Chunk text    │  │ 2. Semantic search → ChromaDB        │
│ 3. Embed chunks  │  │ 3. Build context from top-k chunks   │
│ 4. Store vectors │  │ 4. Generate answer via Gemini        │
│    → ChromaDB    │  │ 5. Log run → MLflow                  │
└──────────────────┘  └────────────────┬─────────────────────┘
                                       │
              ┌────────────────────────┼──────────────────────┐
              │                        │                      │
   ┌──────────▼──────┐    ┌────────────▼──────┐   ┌──────────▼──────┐
   │   ChromaDB      │    │  Gemini 2.5 Flash  │   │     MLflow      │
   │ (Vector Store)  │    │ (Answer Generation)│   │   Tracking      │
   │ Persisted to    │    │  Grounded prompts  │   │ SQLite backend  │
   │ .chroma/ dir    │    │  Anti-hallucination│   │ GCS artifacts   │
   └─────────────────┘    └───────────────────┘   └─────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **API Framework** | FastAPI + Uvicorn | Async REST API, auto Swagger docs, lifespan events |
| **RAG Orchestration** | LangChain | Document loaders, text splitters, vector store wrappers |
| **Vector Database** | ChromaDB | Local persistent vector store for semantic search |
| **Embedding Model** | `all-MiniLM-L6-v2` | Local sentence embeddings — 384 dimensions, runs on GPU |
| **LLM** | Gemini 2.5 Flash | Grounded answer generation via Google AI |
| **Experiment Tracking** | MLflow | Parameters, metrics, artifacts logged per query run |
| **Data Versioning** | DVC + GCS | Large file versioning with Google Cloud Storage remote |
| **Containerization** | Docker | Single-file reproducible environment |
| **CI/CD** | GitHub Actions | Automated test → build → push → deploy pipeline |
| **Image Registry** | GCP Artifact Registry | Versioned Docker images (tagged by commit SHA) |
| **Cloud VM** | GCP Compute Engine | Ubuntu 22.04, e2-medium, always-on deployment |
| **Object Storage** | GCP Cloud Storage | DVC data remote + MLflow artifact storage |
| **Testing** | Pytest + pytest-asyncio + httpx | Unit and integration tests with mocking |

---

## Project Structure

```
documind/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py              # All API route definitions + Pydantic models
│   ├── core/
│   │   ├── __init__.py
│   │   ├── embeddings.py          # Embedding model + ChromaDB interface
│   │   └── rag.py                 # Full RAG pipeline + MLflow tracking + Gemini
│   ├── static/
│   │   └── index.html             # Frontend UI (served at /)
│   ├── __init__.py
│   └── main.py                    # FastAPI app, lifespan, router registration
├── tests/
│   ├── __init__.py
│   ├── test_api.py                # Integration tests for all API endpoints
│   ├── test_embeddings.py         # Unit tests for embedding + ChromaDB functions
│   └── test_rag.py                # Unit tests for RAG pipeline functions
├── .github/
│   └── workflows/
│       └── deploy.yml             # 3-job CI/CD pipeline
├── credentials/                   # GCP service account key (gitignored)
├── data/                          # Documents managed by DVC (gitignored)
│   ├── raw/                       # Original uploaded files
│   └── processed/                 # Chunked/processed output
├── Dockerfile                     # Container definition
├── pytest.ini                     # Pytest configuration
├── requirements.txt               # Pinned Python dependencies
├── data.dvc                       # DVC pointer file (tracked by Git)
├── .dvc/
│   └── config                     # DVC remote config (GCS bucket)
├── .env                           # Local env vars (gitignored)
├── .env.docker                    # Docker env vars (gitignored)
├── .env.example                   # Template for required env vars
└── .gitignore
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker Desktop
- Google Cloud account
- Gemini API key (free at [Google AI Studio](https://aistudio.google.com/app/apikey))
- `gcloud` CLI installed

### 1. Clone the repository

```bash
git clone https://github.com/AmrMohamed17/Documind.git
cd Documind
```

### 2. Set up virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set up GCP credentials

```bash
# Create a service account and download the key
gcloud iam service-accounts create documind-sa --display-name="DocuMind SA"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:documind-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

gcloud iam service-accounts keys create credentials/gcp-key.json \
  --iam-account=documind-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

### 4. Configure environment variables

```bash
cp .env.example .env
nano .env  # Fill in your values
```

| Variable | Description | Example |
|---|---|---|
| `GEMINI_API_KEY` | Gemini API key | `AIza...` |
| `GEMINI_MODEL` | Gemini model name | `gemini-2.5-flash` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Absolute path to GCP key | `/home/user/documind/credentials/gcp-key.json` |
| `MLFLOW_TRACKING_URI` | SQLite DB path (4 slashes for absolute) | `sqlite:////home/user/documind/mlflow.db` |
| `MLFLOW_ARTIFACT_ROOT` | GCS artifact bucket | `gs://your-bucket/mlflow-artifacts` |
| `CHUNK_SIZE` | Characters per chunk | `500` |
| `CHUNK_OVERLAP` | Overlap between chunks | `50` |
| `EMBEDDING_MODEL` | HuggingFace model name | `all-MiniLM-L6-v2` |

### 5. Pull data with DVC (optional)

```bash
dvc pull  # Restores data/ from GCS remote
```

### 6. Run locally

```bash
uvicorn app.main:app --reload --port 8000
```

- **UI:** `http://localhost:8000`
- **API Docs:** `http://localhost:8000/docs`
- **Health:** `http://localhost:8000/health`

### 7. Run with Docker

Create `.env.docker` with container-appropriate paths:

```bash
cp .env.example .env.docker
# Set GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/gcp-key.json
# Set MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db
```

Then build and run:

```bash
docker build -t documind:latest .

docker run -d \
  --name documind \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/credentials:/app/credentials \
  -v $(pwd)/.chroma:/app/.chroma \
  -v $(pwd)/mlflow.db:/app/mlflow.db \
  --env-file .env.docker \
  documind:latest
```

---

## API Reference

### `GET /health`
Health check — used by CI/CD and load balancers.

```json
{"status": "healthy", "service": "documind"}
```

### `GET /api/v1/documents`
Returns all documents currently stored in ChromaDB. Called on page load to restore the document list in the UI.

```json
{
  "documents": [
    {"file": "report.pdf", "source": "data/raw/report.pdf"}
  ]
}
```

### `POST /api/v1/ingest`
Upload and ingest a PDF or TXT file. Chunks the document, generates embeddings, stores in ChromaDB.

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@/path/to/document.pdf"
```

```json
{
  "message": "Document ingested successfully",
  "file": "document.pdf",
  "pages_loaded": 54,
  "chunks_stored": 209
}
```

### `POST /api/v1/query`
Ask a question over all ingested documents. Retrieves top-k chunks and generates a grounded answer via Gemini.

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is supervised learning?", "k": 4}'
```

```json
{
  "question": "What is supervised learning?",
  "answer": "Supervised learning is a type of machine learning where the model is trained on labeled data...",
  "sources": ["ml_intro.pdf"],
  "chunks_used": 4
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `question` | string | required | The question to answer |
| `k` | int | 4 | Number of chunks to retrieve (1–10) |

---

## MLOps: Experiment Tracking

Every query creates a new MLflow run logging:

**Parameters (inputs — what you configured):**
- `question` — the query asked
- `k` — number of chunks retrieved
- `chunk_size`, `chunk_overlap` — chunking configuration
- `embedding_model` — model used for embeddings

**Metrics (outputs — what you measured):**
- `retrieval_latency_seconds` — time to search ChromaDB
- `generation_latency_seconds` — time for Gemini to respond
- `total_latency_seconds` — end-to-end query time
- `chunks_retrieved` — actual chunks returned
- `retrieval_rate` — chunks retrieved / chunks requested

**Artifacts (files — stored in GCS):**
- `query_result.txt` — full record of the query, parameters, metrics, retrieved context, and generated answer

To launch the MLflow UI:

```bash
mlflow ui \
  --backend-store-uri sqlite:////absolute/path/to/mlflow.db \
  --port 5000
```

Open `http://localhost:5000` to compare runs across different chunk sizes, k values, and embedding models.

---

## Data Versioning with DVC

Data files are tracked with DVC and stored in GCS — not in Git.

```bash
# Pull latest data from GCS
dvc pull

# After adding new documents to data/
dvc add data/
dvc push

# Commit the updated pointer file
git add data.dvc
git commit -m "data: add new documents"
git push origin main
```

**Why DVC instead of Git?**
Git stores the full file content on every commit — a 1GB dataset committed 10 times = 10GB in repo history. DVC stores only a tiny pointer file in Git (containing the file's MD5 hash) and the actual data in GCS. The repo stays lightweight and data is always reproducible from any commit.

---

## CI/CD Pipeline

Every push to `main` triggers three sequential GitHub Actions jobs:

```
git push origin main
        │
        ▼
┌───────────────────┐
│  1. Run Tests     │  18 pytest tests across API, RAG, embeddings
│     (Pytest)      │  Gemini calls are mocked — fast, free, deterministic
└────────┬──────────┘
         │ all pass
         ▼
┌───────────────────┐
│  2. Build & Push  │  Docker image built and pushed to GCP Artifact Registry
│  Docker Image     │  Tagged with both :latest and :<git-commit-sha>
└────────┬──────────┘  (SHA tag enables rollback to any specific commit)
         │ success
         ▼
┌───────────────────┐
│  3. Deploy to     │  SSH into GCP VM
│  GCP VM           │  Pull new image from Artifact Registry
│                   │  Stop old container, start new one
└───────────────────┘  --restart unless-stopped (survives VM reboots)
```

If any job fails, subsequent jobs are skipped — nothing broken ever reaches production.

**GitHub Secrets required:**

| Secret | Description |
|---|---|
| `GCP_PROJECT_ID` | Your GCP project ID |
| `GCP_SA_KEY` | Full contents of `credentials/gcp-key.json` |
| `VM_HOST` | GCP VM external IP address |
| `VM_USER` | VM SSH username (`ubuntu`) |
| `VM_SSH_KEY` | Private SSH key for VM access |
| `GEMINI_API_KEY` | Gemini API key for CI test environment |

---

## Running Tests

```bash
# Clean state first
rm -rf .chroma/

# Run all tests
pytest tests/ -v
```

Expected output:

```
tests/test_embeddings.py::test_embedding_function_loads        PASSED
tests/test_embeddings.py::test_embedding_is_normalized         PASSED
tests/test_embeddings.py::test_vector_store_initializes        PASSED
tests/test_embeddings.py::test_add_and_search_documents        PASSED
tests/test_embeddings.py::test_semantic_search_ranks_correctly PASSED
tests/test_rag.py::test_load_txt_document                      PASSED
tests/test_rag.py::test_load_unsupported_format                PASSED
tests/test_rag.py::test_chunk_documents                        PASSED
tests/test_rag.py::test_chunk_overlap                          PASSED
tests/test_rag.py::test_ingest_document                        PASSED
tests/test_rag.py::test_query_returns_correct_structure        PASSED
tests/test_rag.py::test_query_empty_db                         PASSED
tests/test_api.py::test_health_check                           PASSED
tests/test_api.py::test_ingest_txt_file                        PASSED
tests/test_api.py::test_ingest_unsupported_file                PASSED
tests/test_api.py::test_query_endpoint                         PASSED
tests/test_api.py::test_query_empty_question                   PASSED
tests/test_api.py::test_query_default_k                        PASSED

18 passed
```

**Test breakdown:**
- `test_embeddings.py` — unit tests: embedding dimensions, vector normalization, ChromaDB initialization, semantic ranking correctness
- `test_rag.py` — unit tests: document loading, chunking behavior, ingestion pipeline, query response structure
- `test_api.py` — integration tests: all endpoints, file type validation, empty input handling, default parameter behavior

Gemini API calls are mocked using `unittest.mock.patch` — tests run without hitting the real API, making them fast, free, and deterministic in CI.

---

## GCP Infrastructure

| Resource | Type | Spec |
|---|---|---|
| `documind-vm` | Compute Engine | e2-medium, Ubuntu 22.04, us-central1-a |
| `documind-amr-data` | Cloud Storage | us-central1, stores DVC data + MLflow artifacts |
| `documind-repo` | Artifact Registry | Docker format, us-central1 |
| `documind-sa` | Service Account | roles: storage.admin, compute.admin, artifactregistry.writer |

**GCS bucket structure:**
```
documind-amr-data/
├── dvc/              ← DVC content-addressable cache (data files by MD5 hash)
└── mlflow-artifacts/ ← MLflow run artifacts (query_result.txt per run)
```

---

## Environment Variables

Two env files are used — both gitignored, never committed:

- `.env` — local development, uses absolute machine paths
- `.env.docker` — Docker container, uses `/app/...` paths

See `.env.example` for the full template with all required variables.

---

## License

MIT