from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request

from rag.api.config import Settings


def get_settings(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


def _require(request: Request, name: str, label: str) -> Any:
    obj = getattr(request.app.state, name, None)
    if obj is None:
        raise HTTPException(status_code=503, detail=f"{label} not configured")
    return obj


def get_store(request: Request) -> Any:
    return _require(request, "store", "Vector store")


def get_embedder(request: Request) -> Any:
    return _require(request, "embedder", "Embedder")


def get_retriever(request: Request) -> Any:
    return _require(request, "retriever", "Retrieval pipeline")


def get_assembler(request: Request) -> Any:
    return _require(request, "assembler", "Context assembler")


def get_generation_pipeline(request: Request) -> Any:
    return _require(request, "generation_pipeline", "Generation pipeline")
