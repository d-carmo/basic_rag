"""Tests for LanguageEnricher."""

import pytest

from rag.chunker.chunk import ChunkMetadata
from rag.enricher.base import EnrichedChunk
from rag.enricher.language import LanguageEnricher


def _chunk(text: str) -> EnrichedChunk:
    return EnrichedChunk(text=text, metadata=ChunkMetadata(chunk_index=0))


@pytest.mark.asyncio
async def test_language_enricher_uses_injected_detector() -> None:
    detector = lambda text: "en"  # noqa: E731
    enricher = LanguageEnricher(detector=detector)
    result = await enricher.enrich(_chunk("anything"))
    assert result.language == "en"


@pytest.mark.asyncio
async def test_language_enricher_returns_none_on_detector_error() -> None:
    def bad_detector(text: str) -> str:
        raise RuntimeError("fail")

    enricher = LanguageEnricher(detector=bad_detector)
    result = await enricher.enrich(_chunk("anything"))
    assert result.language is None


@pytest.mark.asyncio
async def test_language_enricher_preserves_other_fields() -> None:
    enricher = LanguageEnricher(detector=lambda t: "fr")
    chunk = _chunk("bonjour")
    chunk = chunk.model_copy(update={"title": "existing title"})
    result = await enricher.enrich(chunk)
    assert result.language == "fr"
    assert result.title == "existing title"


def test_language_enricher_not_expensive() -> None:
    assert LanguageEnricher.expensive is False
