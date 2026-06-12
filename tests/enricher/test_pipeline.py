"""Tests for EnricherPipeline and EnricherConfig."""

import pytest

from rag.chunker.chunk import Chunk, ChunkMetadata
from rag.enricher.base import BaseEnricher, EnrichedChunk
from rag.enricher.pipeline import EnricherConfig, EnricherPipeline


def _chunk(text: str = "test text") -> Chunk:
    return Chunk(text=text, metadata=ChunkMetadata(chunk_index=0))


class _LanguageStub(BaseEnricher):
    expensive = False

    async def enrich(self, chunk: EnrichedChunk) -> EnrichedChunk:
        return chunk.model_copy(update={"language": "en"})


class _TitleStub(BaseEnricher):
    expensive = True

    async def enrich(self, chunk: EnrichedChunk) -> EnrichedChunk:
        return chunk.model_copy(update={"title": "stub title"})


class _CountingEnricher(BaseEnricher):
    expensive = False

    def __init__(self) -> None:
        self.call_count = 0

    async def enrich(self, chunk: EnrichedChunk) -> EnrichedChunk:
        self.call_count += 1
        return chunk


# ── Single chunk enrichment ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_promotes_chunk_to_enriched_chunk() -> None:
    pipeline = EnricherPipeline(enrichers=[])
    result = await pipeline.enrich(_chunk())
    assert isinstance(result, EnrichedChunk)


@pytest.mark.asyncio
async def test_pipeline_runs_enrichers_in_order() -> None:
    pipeline = EnricherPipeline(enrichers=[_LanguageStub(), _TitleStub()])
    result = await pipeline.enrich(_chunk())
    assert result.language == "en"
    assert result.title == "stub title"


@pytest.mark.asyncio
async def test_pipeline_fast_mode_skips_expensive() -> None:
    config = EnricherConfig(fast_mode=True)
    pipeline = EnricherPipeline(enrichers=[_LanguageStub(), _TitleStub()], config=config)
    result = await pipeline.enrich(_chunk())
    assert result.language == "en"
    assert result.title is None  # TitleStub is expensive → skipped


@pytest.mark.asyncio
async def test_pipeline_fast_mode_false_runs_all() -> None:
    config = EnricherConfig(fast_mode=False)
    pipeline = EnricherPipeline(enrichers=[_LanguageStub(), _TitleStub()], config=config)
    result = await pipeline.enrich(_chunk())
    assert result.language == "en"
    assert result.title == "stub title"


# ── Batch enrichment ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_enrich_batch_returns_all_results() -> None:
    pipeline = EnricherPipeline(enrichers=[_LanguageStub()])
    chunks = [_chunk(f"text {i}") for i in range(5)]
    results = await pipeline.enrich_batch(chunks)
    assert len(results) == 5
    assert all(isinstance(r, EnrichedChunk) for r in results)
    assert all(r.language == "en" for r in results)


@pytest.mark.asyncio
async def test_enrich_batch_empty_input() -> None:
    pipeline = EnricherPipeline(enrichers=[_LanguageStub()])
    results = await pipeline.enrich_batch([])
    assert results == []


@pytest.mark.asyncio
async def test_enrich_batch_concurrency_respected() -> None:
    counter = _CountingEnricher()
    config = EnricherConfig(batch_concurrency=2)
    pipeline = EnricherPipeline(enrichers=[counter], config=config)
    chunks = [_chunk(f"t{i}") for i in range(10)]
    results = await pipeline.enrich_batch(chunks)
    assert len(results) == 10
    assert counter.call_count == 10
