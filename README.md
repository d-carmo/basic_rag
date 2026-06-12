# RAG Pipeline

A production-grade Retrieval-Augmented Generation pipeline built from scratch in Python — no LangChain, no LlamaIndex. Every stage is independently swappable, fully async, and covered by tests.

## What it does

The pipeline ingests documents (PDF, DOCX, HTML, Markdown, JSON, CSV), chunks and enriches them, embeds them into a Qdrant vector store, and answers natural-language queries with grounded, cited responses via a FastAPI service.

```
Documents → Loader → Chunker → Enricher → Embedder → Qdrant
                                                         ↓
Query → Retriever (dense + sparse + reranker) → Assembler → LLM → Answer + Citations
```

## Features

- **Document loaders** — PDF (with OCR fallback), DOCX, HTML, Markdown, JSON/JSONL, CSV
- **Chunking strategies** — recursive, sentence-aware, section/header, semantic, parent-child
- **Enrichment** — language detection, NER (spaCy), hypothetical questions, title extraction
- **Embeddings** — local (BGE-M3 via sentence-transformers), OpenAI, Cohere, BM25 sparse; Redis/SQLite cache
- **Retrieval** — dense, hybrid (dense + sparse), HyDE, multi-query, step-back; cross-encoder and Cohere reranking; RRF merging
- **Context assembly** — near-duplicate filtering, lost-in-the-middle reordering, token budget trimming, citation maps
- **LLM generation** — Anthropic Claude (default), OpenAI, or any local model via OpenAI-compatible API (Ollama, LM Studio, vLLM); prompt injection guard; faithfulness scoring
- **REST API** — FastAPI with streaming SSE, API-key auth, per-key rate limiting, CORS
- **Evaluation** — token-F1 answer similarity, context recall/precision, faithfulness; golden dataset + baseline regression
- **Observability** — structlog (JSON/pretty), Prometheus metrics, OpenTelemetry tracing, Grafana dashboard

---

## Quick start (local dev)

**Prerequisites:** Python 3.12+, Docker, `uv` or `pip`.

### 1. Clone and install

```bash
git clone <repo-url> && cd RAG_pipeline
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Start services

```bash
docker compose up -d
```

This starts Qdrant (`:6333`), Redis (`:6379`), Jaeger (`:16686`), Prometheus (`:9090`), and Grafana (`:3000`). The FastAPI app is **not** included in the compose file during development — run it locally (step 4).

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env — set LLM_BACKEND and the relevant API key or local server URL
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `API_KEY` | `dev-key` | Comma-separated valid API keys for `X-API-Key` header |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `LLM_BACKEND` | `anthropic` | `anthropic` / `openai` / `local` |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_BACKEND=anthropic` |
| `LLM_BASE_URL` | — | Required when `LLM_BACKEND=local` (e.g. `http://localhost:11434/v1`) |
| `LLM_MODEL` | `claude-opus-4-8` | Model name passed to the LLM backend |
| `LLM_MAX_TOKENS` | `2048` | Maximum tokens in LLM response |
| `LLM_TIMEOUT` | `600` | HTTP timeout in seconds for LLM calls — increase for slow local models |
| `EMBEDDER_BACKEND` | `sentence_transformer` | `sentence_transformer` / `openai` / `cohere` |
| `EMBEDDER_MODEL` | `BAAI/bge-m3` | Embedding model name |
| `COLLECTION_NAME` | `rag` | Qdrant collection name |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` |

### 4. Run the API

```bash
# Bind to 0.0.0.0 so Prometheus (running in Docker) can scrape /metrics
uvicorn rag.api.app:app --host 0.0.0.0 --port 8000 --reload
```

The API is now at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### 5. Run tests

```bash
pytest
```

Integration tests (require a live Qdrant) are marked `@pytest.mark.integration` and skipped by default. To run only unit tests: `pytest -m "not integration"`.

---

## API usage

All endpoints (except `/health` and `/ready`) require `X-API-Key: <your-key>`.

### Ingest raw text

```bash
curl -X POST http://localhost:8000/v1/ingest \
  -H "X-API-Key: changeme" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Retrieval-Augmented Generation combines a retrieval system with a language model...",
    "source_id": "doc-rag-intro",
    "doc_type": "text",
    "title": "Introduction to RAG"
  }'
```

```json
{"source_id": "doc-rag-intro", "status": "ok", "chunk_count": 1}
```

### Ingest a file (PDF or plain text)

Upload a file directly — the server extracts the text using the appropriate loader.

```bash
# Plain text or Markdown
curl -X POST http://localhost:8000/v1/ingest/file \
  -H "X-API-Key: changeme" \
  -F "file=@/path/to/document.txt" \
  -F "source_id=my-doc-001" \
  -F "title=My Document"

# PDF (one chunk per page)
curl -X POST http://localhost:8000/v1/ingest/file \
  -H "X-API-Key: changeme" \
  -F "file=@/path/to/document.pdf" \
  -F "source_id=my-doc-001" \
  -F "title=My Document"
```

Supported extensions: `.pdf`, `.txt`, `.text`, `.rst`, `.md`, `.markdown`.
`source_id` and `title` are optional — `source_id` defaults to the filename.

### Batch ingest

```bash
curl -X POST http://localhost:8000/v1/ingest/batch \
  -H "X-API-Key: changeme" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"text": "Dense retrieval embeds queries and documents...", "source_id": "doc-dense"},
      {"text": "Sparse retrieval uses term weights like BM25...", "source_id": "doc-sparse"}
    ]
  }'
```

### Query

```bash
curl -X POST http://localhost:8000/v1/query \
  -H "X-API-Key: changeme" \
  -H "Content-Type: application/json" \
  -d '{"query": "How does dense retrieval work?", "top_k": 5}'
```

```json
{
  "answer": "Dense retrieval embeds both the query and documents into a shared vector space...",
  "sources": [
    {"citation_number": 1, "source_id": "doc-dense", "title": null, "score": 0.92}
  ],
  "trace_id": "3f2a...",
  "faithfulness_score": null,
  "truncated": false
}
```

### Streaming query (SSE)

```bash
curl -X POST http://localhost:8000/v1/query \
  -H "X-API-Key: changeme" \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain RAG", "stream": true}'
```

Each line is a Server-Sent Event:
```
data: {"chunk": "Retrieval-Augmented"}
data: {"chunk": " Generation combines"}
...
data: {"done": true, "trace_id": "3f2a..."}
```

### List sources

```bash
curl http://localhost:8000/v1/sources -H "X-API-Key: dev-key"
```

### Delete a source

```bash
curl -X DELETE http://localhost:8000/v1/source/doc-rag-intro -H "X-API-Key: dev-key"
```

### Health checks

```bash
curl http://localhost:8000/health   # {"status": "ok"}
curl http://localhost:8000/ready    # {"status": "ok", "checks": {"qdrant": true}}
```

### Prometheus metrics

```bash
curl http://localhost:8000/metrics  # Prometheus text format — no auth required
```

---

## Using a local LLM

Point the pipeline at any OpenAI-compatible server — no API key needed:

Set these in `.env` (or inline):

```bash
LLM_BACKEND=local
LLM_BASE_URL=http://localhost:11434/v1   # Ollama
# LLM_BASE_URL=http://localhost:1234/v1  # LM Studio
# LLM_BASE_URL=http://localhost:8000/v1  # vLLM
LLM_MODEL=llama3
LLM_TIMEOUT=600   # increase if your model is slow to respond
```

Then start the server:

```bash
uvicorn rag.api.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## Running evaluation

```bash
python -m rag.eval run \
  --dataset eval/golden.jsonl \
  --report eval/results/report.json \
  --baseline eval/baseline.json
```

Output:
```
Evaluation complete: eval/results/report.json
  answer_similarity: 0.0000
  context_recall:    0.0000
  context_precision: 0.0000
  faithfulness:      0.0000

Comparison vs baseline:
  answer_similarity: 0.0000 (+0.0000)
  ...
```

The golden dataset lives in [eval/golden.jsonl](eval/golden.jsonl). Add Q&A triples there and commit `eval/baseline.json` after a successful run to set a regression baseline.

---

## Observability

### Prometheus

Prometheus scrapes the app's `/metrics` endpoint and is available at `http://localhost:9090` after `docker compose up`.

> **Note:** When running uvicorn locally (outside Docker), start it with `--host 0.0.0.0` so the Prometheus container can reach it via `host.docker.internal`. The scrape target is pre-configured in `docker/prometheus/prometheus.yml`.

Useful queries to get started:

| What | PromQL |
|---|---|
| Query rate (req/min) | `rate(http_requests_total{path="/v1/query"}[1m]) * 60` |
| p99 query latency | `histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{path="/v1/query"}[5m]))` |
| Total chunks ingested | `sum(chunks_ingested_total)` |
| All requests by path | `rate(http_requests_total[1m]) * 60` |

Open the expression browser at `http://localhost:9090/graph`, type a query, and click **Execute**.

To check which targets are being scraped: `http://localhost:9090/targets`

### Grafana

Grafana is available at `http://localhost:3000`.

**Default credentials:** `admin` / `admin` (you will be prompted to change the password on first login).

The **RAG Pipeline** dashboard is pre-provisioned and appears under **Dashboards → RAG** automatically. It includes:

- Query rate (requests per minute)
- p99 query latency (stat)
- Total chunks ingested (stat)
- Query latency p50 / p95 / p99 over time
- HTTP request rate by path and status

To find it manually: **Dashboards** (left sidebar) → search for `RAG Pipeline`.

The Prometheus datasource is also pre-provisioned — no manual connection setup is required. If you want to build your own panels, create a new dashboard and select **Prometheus** as the datasource.

To change the default admin password, set it in `.env` before starting:

```bash
GRAFANA_ADMIN_PASSWORD=mysecretpassword
```

---

## Production deployment

For production, uncomment and configure the `app` service in `docker-compose.yml` (currently commented out for local development), then:

```bash
docker compose up -d
```

At minimum: set `API_KEY`, your LLM credentials, and `GRAFANA_ADMIN_PASSWORD` in `.env`. Add resource limits and `restart: always` to the compose services as needed.

---

## Project structure

```
src/rag/
├── api/            # FastAPI app, routes, middleware, deps
├── assembler/      # Context assembly: dedup, reorder, budget, citations
├── chunker/        # Chunking strategies + factory
├── config.py       # Central RagConfig (from_env)
├── embedder/       # Dense + sparse embedders, cache, factory
├── enricher/       # Language, NER, hypothetical questions, title
├── eval/           # Evaluation: dataset, metrics, runner, CLI
├── generation/     # LLM backends, prompts, guard, faithfulness, pipeline
├── ingestion/      # IngestionLog (SQLite)
├── loaders/        # Document loaders (PDF, DOCX, HTML, …)
├── observability/  # Logging, metrics, tracing
├── retriever/      # Retrievers, transforms, rerankers, pipeline
└── vector_store/   # Qdrant store, schema, filters, snapshots
tests/              # 438 unit tests mirroring src/rag/
eval/               # golden.jsonl, baseline.json
docker/             # Dockerfile, Grafana provisioning, Prometheus config
```

---

## Tech stack

| Layer | Library |
|---|---|
| Language | Python 3.12 |
| API | FastAPI + uvicorn |
| Vector store | Qdrant |
| Embeddings | sentence-transformers, OpenAI, Cohere |
| LLM | Anthropic Claude (default), OpenAI-compatible |
| Cache | Redis, SQLite |
| Observability | structlog, Prometheus, OpenTelemetry, Grafana |
| Testing | pytest, pytest-asyncio, httpx |
| Linting | ruff, mypy (strict) |
