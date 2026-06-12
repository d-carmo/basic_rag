from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from rag.chunker.chunk import Chunk
from rag.enricher.base import BaseEnricher, EnrichedChunk


@dataclass
class EnricherConfig:
    fast_mode: bool = False
    batch_concurrency: int = 8
    enrichers: list[BaseEnricher] = field(default_factory=list)


class EnricherPipeline:
    def __init__(
        self,
        enrichers: list[BaseEnricher],
        config: EnricherConfig | None = None,
    ) -> None:
        self._enrichers = enrichers
        self._config = config or EnricherConfig()

    def _active_enrichers(self) -> list[BaseEnricher]:
        if self._config.fast_mode:
            return [e for e in self._enrichers if not e.expensive]
        return self._enrichers

    async def enrich(self, chunk: Chunk) -> EnrichedChunk:
        enriched = EnrichedChunk(**chunk.model_dump())
        for enricher in self._active_enrichers():
            enriched = await enricher.enrich(enriched)
        return enriched

    async def enrich_batch(self, chunks: list[Chunk]) -> list[EnrichedChunk]:
        sem = asyncio.Semaphore(self._config.batch_concurrency)

        async def _bounded(c: Chunk) -> EnrichedChunk:
            async with sem:
                return await self.enrich(c)

        return list(await asyncio.gather(*[_bounded(c) for c in chunks]))
