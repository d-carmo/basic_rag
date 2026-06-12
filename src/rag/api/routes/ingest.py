from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Request

from rag.api.deps import get_embedder, get_store
from rag.api.models import (
    BatchIngestRequest,
    BatchIngestResponse,
    IngestRequest,
    IngestResponse,
)
from rag.chunker.chunk import Chunk, ChunkMetadata
from rag.loaders.base import DocType

router = APIRouter()

from rag.api.api_metrics import chunks_ingested_counter as _chunks_ingested


def _make_chunk(req: IngestRequest) -> Chunk:
    try:
        dt = DocType(req.doc_type)
    except ValueError:
        dt = DocType.TEXT
    return Chunk(
        text=req.text,
        token_count=len(req.text.split()),
        metadata=ChunkMetadata(
            source_url=req.source_id,
            doc_type=dt,
            language=req.language,
            extra={"title": req.title, **req.metadata} if req.title else req.metadata,
        ),
    )


async def _ingest_one(req: IngestRequest, store, embedder) -> IngestResponse:
    chunk = _make_chunk(req)
    vectors = await embedder.embed([chunk.text])
    await store.upsert_chunks([chunk], dense_vectors=vectors)
    _chunks_ingested.labels(route="ingest").inc()
    return IngestResponse(source_id=req.source_id, status="ok", chunk_count=1)


@router.post("/v1/ingest", response_model=IngestResponse)
async def ingest(
    body: IngestRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    store = get_store(request)
    embedder = get_embedder(request)
    return await _ingest_one(body, store, embedder)


@router.post("/v1/ingest/batch", response_model=BatchIngestResponse)
async def ingest_batch(body: BatchIngestRequest, request: Request) -> BatchIngestResponse:
    store = get_store(request)
    embedder = get_embedder(request)
    ingested: list[IngestResponse] = []
    failed: list[dict[str, str]] = []
    for doc in body.documents:
        try:
            resp = await _ingest_one(doc, store, embedder)
            ingested.append(resp)
        except Exception as exc:
            failed.append({"source_id": doc.source_id, "error": str(exc)})
    return BatchIngestResponse(ingested=ingested, failed=failed)
