from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import Field

from rag.loaders.base import Document, DocumentMetadata


class ChunkMetadata(DocumentMetadata):
    chunk_index: int = 0
    parent_id: str | None = None


class Chunk(Document):
    metadata: ChunkMetadata  # type: ignore[assignment]
    token_count: int = Field(default=0, ge=0)


class BaseChunker(ABC):
    """Abstract base for all chunking strategies."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        ...
