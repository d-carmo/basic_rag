"""Tests for NEREnricher."""

import pytest

from rag.chunker.chunk import ChunkMetadata
from rag.enricher.base import EnrichedChunk, Entity
from rag.enricher.ner import NEREnricher


def _chunk(text: str = "Apple is based in Cupertino.") -> EnrichedChunk:
    return EnrichedChunk(text=text, metadata=ChunkMetadata(chunk_index=0))


class _FakeEnt:
    def __init__(self, text: str, label_: str, start_char: int, end_char: int) -> None:
        self.text = text
        self.label_ = label_
        self.start_char = start_char
        self.end_char = end_char


class _FakeDoc:
    def __init__(self, ents: list[_FakeEnt]) -> None:
        self.ents = ents


class _FakeNLP:
    def __call__(self, text: str) -> _FakeDoc:
        return _FakeDoc([_FakeEnt("Apple", "ORG", 0, 5)])


@pytest.mark.asyncio
async def test_ner_enricher_with_fake_nlp() -> None:
    enricher = NEREnricher(nlp=_FakeNLP())
    result = await enricher.enrich(_chunk())
    assert len(result.entities) == 1
    assert result.entities[0] == Entity(text="Apple", label="ORG", start=0, end=5)


@pytest.mark.asyncio
async def test_ner_enricher_empty_entities_on_nlp_error() -> None:
    class _BrokenNLP:
        def __call__(self, text: str) -> _FakeDoc:
            raise RuntimeError("nlp broke")

    enricher = NEREnricher(nlp=_BrokenNLP())
    result = await enricher.enrich(_chunk())
    assert result.entities == []


@pytest.mark.asyncio
async def test_ner_enricher_preserves_other_fields() -> None:
    enricher = NEREnricher(nlp=_FakeNLP())
    chunk = _chunk().model_copy(update={"language": "en"})
    result = await enricher.enrich(chunk)
    assert result.language == "en"
    assert len(result.entities) == 1


def test_ner_enricher_not_expensive() -> None:
    assert NEREnricher.expensive is False
