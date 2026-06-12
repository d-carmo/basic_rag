from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=8, ge=1, le=100)
    stream: bool = False
    doc_type: str | None = None
    filters: dict[str, Any] | None = None


class CitationSource(BaseModel):
    citation_number: int
    source_id: str
    title: str | None = None
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[CitationSource]
    trace_id: str
    faithfulness_score: float | None = None
    truncated: bool = False


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source_id: str = Field(..., min_length=1)
    doc_type: str = "text"
    title: str | None = None
    language: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchIngestRequest(BaseModel):
    documents: list[IngestRequest] = Field(..., min_length=1)


class IngestResponse(BaseModel):
    source_id: str
    status: str
    chunk_count: int


class BatchIngestResponse(BaseModel):
    ingested: list[IngestResponse]
    failed: list[dict[str, str]] = Field(default_factory=list)


class SourceInfo(BaseModel):
    source_id: str
    chunk_count: int
    ingested_at: str | None = None


class SourcesResponse(BaseModel):
    sources: list[SourceInfo]
    total: int


class DeleteResponse(BaseModel):
    source_id: str
    deleted: bool


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    checks: dict[str, bool]
