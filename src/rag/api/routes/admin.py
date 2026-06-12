from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from rag.api.models import (
    DeleteResponse,
    HealthResponse,
    ReadyResponse,
    SourceInfo,
    SourcesResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
async def ready(request: Request) -> ReadyResponse:
    checks: dict[str, bool] = {}

    store = getattr(request.app.state, "store", None)
    if store is not None:
        try:
            await store.collection_exists()
            checks["qdrant"] = True
        except Exception:
            checks["qdrant"] = False
    else:
        checks["qdrant"] = False

    status = "ok" if all(checks.values()) else "degraded"
    return ReadyResponse(status=status, checks=checks)


@router.get("/metrics")
async def metrics() -> Response:
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/v1/sources", response_model=SourcesResponse)
async def list_sources(request: Request) -> SourcesResponse:
    store = getattr(request.app.state, "store", None)
    if store is None:
        return SourcesResponse(sources=[], total=0)
    rows = await store.list_sources()
    sources = [
        SourceInfo(source_id=r["source_id"], chunk_count=r["chunk_count"])
        for r in rows
    ]
    return SourcesResponse(sources=sources, total=len(sources))


@router.delete("/v1/source/{source_id}", response_model=DeleteResponse)
async def delete_source(source_id: str, request: Request) -> DeleteResponse:
    store = getattr(request.app.state, "store", None)
    if store is not None:
        await store.delete_by_source(source_id)

    log = getattr(request.app.state, "ingestion_log", None)
    if log is not None:
        log.remove(source_id)

    return DeleteResponse(source_id=source_id, deleted=True)
