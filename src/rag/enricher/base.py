from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from rag.chunker.chunk import Chunk


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int


class EnrichedChunk(Chunk):
    hypothetical_questions: list[str] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    language: str | None = None
    title: str | None = None


class BaseEnricher(ABC):
    expensive: bool = False

    @abstractmethod
    async def enrich(self, chunk: EnrichedChunk) -> EnrichedChunk:
        ...
