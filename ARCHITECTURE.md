# RAG Pipeline — Architecture

## Overview

A sample Retrieval-Augmented Generation (RAG) pipeline that ingests arbitrary documents, indexes them in a vector store, and answers natural-language queries with grounded, cited responses. The design prioritizes modularity (each stage is independently swappable), observability (every hop is traced), and correctness (evaluation gates prevent silent quality regressions).

This is not intended to be production ready or full feature complete - this is my personal learning/investigation project.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          INGESTION PIPELINE                          │
│                                                                      │
│  Raw Sources ──► Document Loader ──► Chunker ──► Enricher            │
│  (PDF, HTML,       (unified I/O        (sliding      (metadata,      │
│   DOCX, DB,         abstraction)        window,       titles,        │
│   API, S3)                              recursive,    summaries,     │
│                                         semantic)     entities)      │
│                                              │                       │
└──────────────────────────────────────────────┼───────────────────────┘
                                               │ Chunks + Metadata
                                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          EMBEDDING LAYER                             │
│                                                                      │
│  Chunks ──► Embedding Model ──► Embedding Cache ──► Batch Writer     │
│              (dense: local         (Redis /            (Qdrant       │
│               BGE/E5, or           disk lru)            upsert)      │
│               OpenAI/Cohere;                                         │
│               sparse: BM25/SPLADE)                                   │
│                                                                      │
└──────────────────────────────────────────────┬───────────────────────┘
                                               │ Vectors + Payloads
                                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          VECTOR STORE (Qdrant)                       │
│                                                                      │
│  Collections: one per corpus / schema version                        │
│  Named vectors: dense  (1536d or 768d)                               │
│                 sparse (SPLADE / BM25 term weights)                  │
│  Payload index: source_id, doc_type, created_at, language, tags      │
│                                                                      │
└──────────────────────────────────────────────┬───────────────────────┘
                                               │
                           ┌───────────────────┘
                           │  Query Path
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          RETRIEVAL ENGINE                            │
│                                                                      │
│  Query ──► Query Transform ──► Hybrid Search ──► Reranker            │
│              (HyDE,               (dense +          (cross-encoder   │
│               multi-query,         sparse,            or Cohere      │
│               step-back)           RRF fusion)        Rerank)        │
│                    │                                      │          │
│                    └─────────────────────────────────────►│          │
│                                                           │          │
│                                              Top-K Chunks │          │
└──────────────────────────────────────────────────────────┼───────────┘
                                                           │
                                                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          GENERATION LAYER                            │
│                                                                      │
│  Chunks ──► Context Assembler ──► Prompt Builder ──► LLM             │
│              (dedup, trim,          (system +          (Claude        │
│               re-order by           user template,     claude-        │
│               relevance,            citation           opus-4-8      │
│               fit to window)        format)            or local)     │
│                                                           │          │
│                                                  Response + Sources  │
└──────────────────────────────────────────────────────────┼───────────┘
                                                           │
                                                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          API LAYER (FastAPI)                         │
│                                                                      │
│  POST /v1/query        — answer a question (JSON or SSE stream)      │
│  POST /v1/ingest       — ingest raw text                             │
│  POST /v1/ingest/batch — ingest multiple documents                   │
│  POST /v1/ingest/file  — upload a file (PDF, txt, md, rst)          │
│  GET  /v1/sources      — list indexed sources (scrolls Qdrant)       │
│  DELETE /v1/source/:id — remove a source                             │
│  GET  /health          — liveness check                              │
│  GET  /ready           — readiness check (Qdrant reachable)          │
│  GET  /metrics         — Prometheus metrics (no auth required)       │
│                                                                      │
│  Auth: X-API-Key header (comma-separated keys in API_KEY env var)   │
│  Rate limiting: per-key token bucket (RATE_LIMIT_RPM env var)        │
│  Metrics: MetricsMiddleware — http_requests_total, latency histogram │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Document Loader

Provides a unified `Document` dataclass output regardless of source format.

**Supported sources (initial):** PDF, DOCX, plain text, Markdown, HTML (via crawl), JSON/JSONL, CSV, PostgreSQL tables, S3/GCS blobs.

**Key concerns:**
- OCR fallback for scanned PDFs (Tesseract / AWS Textract)
- Incremental ingestion: hash-based deduplication to skip unchanged documents
- Preserve source URL / file path / page number in metadata for citations

### 2. Chunker

Splits documents into retrieval-sized chunks. Strategy is configurable per document type.

| Strategy | When to use |
|---|---|
| Recursive character splitting | General prose |
| Sentence-aware splitting | Q&A, dialogues |
| Semantic splitting | Long-form articles (split on embedding similarity drops) |
| Section/header splitting | Markdown, HTML, structured docs |
| Fixed-token | Code, tables |

**Parameters:** `chunk_size` (tokens), `overlap` (tokens), `min_chunk_size`.

Parent-child chunking: store large parent chunks in payload; retrieve small child chunks, return parent for context.

### 3. Enricher

Runs lightweight enrichment passes before embedding:
- **Title extraction** — first heading or LLM-generated title per chunk
- **Hypothetical questions** — generate 3–5 questions the chunk answers (improves retrieval recall via HyDE)
- **Entity tagging** — named entities for payload filtering
- **Language detection** — for multilingual routing

### 4. Embedding Model

Abstract `Embedder` interface with pluggable backends:

| Backend | Model | Dims | Notes |
|---|---|---|---|
| Local (sentence-transformers) | `BAAI/bge-m3` | 1024 | Multilingual, free, strong |
| Local | `BAAI/bge-large-en-v1.5` | 1024 | English-only, fast |
| OpenAI | `text-embedding-3-large` | 3072 (reducible) | Best quality |
| Cohere | `embed-v3` | 1024 | Multilingual |

Sparse embeddings via SPLADE or BM25 term weights (stored in Qdrant sparse vectors).

**Caching:** SHA-256 hash of text → embedding, backed by Redis or an on-disk SQLite cache. Avoid re-embedding unchanged chunks on re-ingestion.

### 5. Vector Store (Qdrant)

Single Qdrant instance (Docker locally, Qdrant Cloud in production).

**Collection schema:**
```
Collection: {corpus_name}_v{schema_version}
Named vectors:
  dense:  { size: 1024, distance: Cosine }
  sparse: { type: Sparse }
Payload fields (all indexed):
  chunk_id:    uuid
  source_id:   string   (document identifier)
  source_url:  string
  page:        int
  created_at:  datetime
  doc_type:    enum
  language:    string
  tags:        string[]
  parent_id:   uuid     (for parent-child chunking)
  raw_text:    string   (stored, not indexed)
```

**Operations:**
- `upsert_chunks(chunks, dense_vectors, sparse_vectors?)` — bulk upsert with exponential-backoff retry
- `query_points(query, using, filter, limit)` — dense-only, sparse-only, or hybrid RRF via Qdrant's `query_points` API
- `delete_by_source(source_id)` — clean removal for re-ingestion
- `get_by_ids(ids)` — fetch payloads by point ID (used by parent-chunk fetcher)

### 6. Retrieval Engine

#### Query Transformation (pre-retrieval)
- **HyDE (Hypothetical Document Embedding):** Generate a hypothetical answer, embed it, use that vector for retrieval. Bridges the query-document vocabulary gap.
- **Multi-query expansion:** Rephrase the query N ways, retrieve for each, union results. Improves recall.
- **Step-back prompting:** Derive a more general question; retrieve for both specific and general.

#### Search
- Hybrid search: Qdrant's built-in dense + sparse with Reciprocal Rank Fusion (RRF).
- Pre-filter by payload (doc_type, date range, tags) before vector search.
- `top_k` = 20 candidates before reranking.

#### Reranker (post-retrieval)
- Local cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Remote: Cohere Rerank v3 (higher quality)
- Returns top 5–8 reranked chunks passed to generation.

### 7. Context Assembler

Takes reranked chunks and prepares the generation context:
- Deduplication (remove near-identical chunks by cosine sim)
- Re-order: most relevant first, then "lost in the middle" mitigation (put second-most-relevant last)
- Truncate to fit LLM context window minus prompt overhead
- Build citation map: chunk_index → source metadata

### 8. LLM / Generation

Abstract `LLM` interface. Default: Claude (`claude-opus-4-8`). Supports Anthropic, OpenAI, and any OpenAI-compatible local server (Ollama, LM Studio, vLLM) via `LLM_BACKEND` / `LLM_BASE_URL` env vars.

Prompt structure:
```
[SYSTEM]
You are a precise assistant. Answer using only the provided context.
Cite sources as [1], [2], etc. If the context doesn't contain the answer, say so.

[USER]
Context:
[1] {chunk_1_text}  (source: {url}, page {page})
[2] {chunk_2_text}  ...

Question: {user_query}
```

**Streaming:** SSE stream from LLM forwarded directly to API response.

**Guardrails:**
- Prompt injection detection (check user query for instruction overrides)
- Response length cap
- Hallucination mitigation: faithfulness score checked async; flag low-confidence responses

### 9. Evaluation

Offline eval suite using RAGAS or a custom harness:

| Metric | What it measures |
|---|---|
| Context Recall | Are the ground-truth chunks retrieved? |
| Context Precision | Are retrieved chunks relevant? |
| Faithfulness | Does the answer stay within the context? |
| Answer Relevance | Does the answer address the question? |
| End-to-end correctness | Against golden Q&A dataset |

Evaluation runs as a CI step on each pipeline change using a fixed golden dataset.

### 10. Observability

- **Structured logging:** JSON logs via `structlog`, log level configurable per env
- **Tracing:** OpenTelemetry spans for each pipeline stage; export to Jaeger (local) or Honeycomb/Datadog (prod)
- **Metrics:** Prometheus metrics — query latency p50/p95/p99, retrieval top-k hit rate, embedding cache hit rate, tokens consumed
- **LLM tracing:** LangSmith (optional) for prompt/response logging and eval dashboards

---

## Data Flow — Query Path

```
User Query
    │
    ▼
1. API receives POST /query {query, corpus, filters, top_k}
    │
    ▼
2. Query Transform (optional HyDE / multi-query)
    │
    ▼
3. Embed query → dense vector + sparse vector
    │
    ▼
4. Qdrant hybrid search (RRF fusion, payload pre-filter)
    │  → 20 candidate chunks
    ▼
5. Reranker → top 8 chunks
    │
    ▼
6. Context Assembler → ordered, deduplicated, truncated context
    │
    ▼
7. Prompt Builder → final prompt
    │
    ▼
8. LLM (Claude) → streaming response
    │
    ▼
9. API streams response + citation metadata to client
    │
    ▼
10. Async: log trace, compute faithfulness score, emit metrics
```

---

## Data Flow — Ingestion Path

```
Document Source
    │
    ▼
1. Document Loader → Document(text, metadata)
    │
    ▼
2. Hash check against ingestion log → skip if unchanged
    │
    ▼
3. Chunker → List[Chunk]
    │
    ▼
4. Enricher → chunks + hypothetical_questions + entities
    │
    ▼
5. Embedder → dense + sparse vectors (cache-hit aware)
    │
    ▼
6. Qdrant upsert (batch, with retry + idempotency)
    │
    ▼
7. Update ingestion log (source_id, hash, timestamp, chunk_count)
```

---

## Deployment

### Local Development

```
docker compose up -d
  ├── qdrant          (port 6333)
  ├── redis           (port 6379, embedding cache)
  ├── jaeger          (port 16686, tracing UI)
  ├── prometheus      (port 9090, metrics)
  └── grafana         (port 3000, dashboards)

uvicorn rag.api.app:app --host 0.0.0.0 --port 8000 --reload
  └── FastAPI app     (port 8000, hot-reload)
```

The FastAPI app runs locally (outside Docker) so that hot-reload works without rebuilding an image. Binding to `0.0.0.0` is required so the Prometheus container can scrape `/metrics` via `host.docker.internal`.

All pipeline components (store, embedder, retriever, assembler, LLM) are initialized at startup by the FastAPI lifespan from settings loaded via `pydantic-settings` (reads `.env` automatically). Copy `.env.example` to `.env` and configure at minimum `API_KEY` and either `ANTHROPIC_API_KEY` or `LLM_BACKEND=local` + `LLM_BASE_URL`.

### Production (single-server or small cloud)
- Docker Compose or Kubernetes (k3s)
- Qdrant with persistent volume; snapshot to S3 daily
- Redis with AOF persistence
- FastAPI behind Nginx (TLS termination, rate limiting)
- Systemd or k8s liveness/readiness probes on `/health`
- Secrets via environment variables or Vault

### Scaling Considerations
- Ingestion: run as async background workers (Celery + Redis or `asyncio` task queue)
- Embedding: GPU node for local models; horizontal scale with batch queue
- Qdrant: single node handles ~10M vectors on a 16GB machine; shard when needed
- API: stateless FastAPI workers, scale horizontally behind load balancer

---

## Technology Stack

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.12 | ML ecosystem depth; no Rust equivalent for LLM tooling yet |
| Package manager | `uv` | 10–100× faster than pip, lock file support |
| API framework | FastAPI + Uvicorn | Async-native, auto OpenAPI docs, type-safe |
| Vector DB | Qdrant v1.13 | Rust-native speed, hybrid search, Query API, Docker-ready, good Python SDK |
| Embedding (local) | `sentence-transformers` + BGE-M3 | Free, strong multilingual, runs on CPU/GPU |
| Embedding cache | Redis | Fast k/v, TTL support |
| LLM | Anthropic Claude (`claude-opus-4-8`) or any OpenAI-compatible server | Configurable via `LLM_BACKEND` / `LLM_BASE_URL`; Ollama, LM Studio, vLLM supported |
| Reranker | `cross-encoder` (local) / Cohere Rerank | Cross-encoder for offline, Cohere for production quality |
| Evaluation | RAGAS | Standard RAG metrics, LLM-as-judge |
| Tracing | OpenTelemetry + Jaeger | Vendor-neutral, production-ready |
| Metrics | Prometheus + Grafana | Standard stack |
| Containerization | Docker + Docker Compose | Reproducible local + prod environments |
| CI | GitHub Actions | Lint, type-check, eval suite on each push |

---

## Key Design Decisions

**Why Qdrant over alternatives?**
- Chroma: great for prototyping, weak filtering and production ops story
- Pinecone: managed but expensive and vendor-locked; no self-hosted option
- Milvus: over-engineered for most use cases, heavy Kubernetes dependency
- pgvector: excellent if already on Postgres; weaker approximate search performance at scale
- Qdrant: self-hosted or cloud, Rust performance, hybrid search built-in, named vectors (multiple embedding models per document), strong filtering

**Why not LangChain/LlamaIndex?**
These frameworks are useful scaffolding but introduce hidden complexity and version-churn overhead. The pipeline is built with clean Python abstractions (`Protocol`-based interfaces) that can wrap or replace framework components as needed. LlamaIndex can be used selectively (e.g., for document loaders or evaluation) without committing to its full stack.

**Chunking strategy matters more than embedding model choice.** Most RAG quality failures trace to bad chunking — too large (low precision), too small (missing context), or crossing semantic boundaries. The pipeline exposes chunking as a first-class configurable concern.

**Parent-child chunking for context quality.** Retrieve small precise chunks; return their larger parent chunk to the LLM for richer context. Balances retrieval precision with generation context quality.
