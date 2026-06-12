from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from rag.chunker.chunk import Chunk
from rag.enricher.base import EnrichedChunk


class ChunkPayload(BaseModel):
    """Qdrant point payload for a chunk."""

    text: str
    source_id: str
    doc_type: str
    created_at: str          # ISO 8601 — compatible with Qdrant DATETIME index
    language: str | None = None
    chunk_index: int = 0
    parent_id: str | None = None
    token_count: int = 0
    title: str | None = None
    hypothetical_questions: list[str] = Field(default_factory=list)
    entities: list[dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> ChunkPayload:
        language: str | None = chunk.metadata.language
        title: str | None = None
        hypothetical_questions: list[str] = []
        entities: list[dict[str, Any]] = []

        if isinstance(chunk, EnrichedChunk):
            language = chunk.language or language
            title = chunk.title
            hypothetical_questions = chunk.hypothetical_questions
            entities = [e.model_dump() for e in chunk.entities]

        return cls(
            text=chunk.text,
            source_id=chunk.metadata.source_url,
            doc_type=chunk.metadata.doc_type.value,
            created_at=chunk.metadata.created_at.isoformat(),
            language=language,
            chunk_index=chunk.metadata.chunk_index,
            parent_id=chunk.metadata.parent_id,
            token_count=chunk.token_count,
            title=title,
            hypothetical_questions=hypothetical_questions,
            entities=entities,
        )
