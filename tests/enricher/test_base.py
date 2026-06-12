"""Tests for EnrichedChunk model and Entity."""

from rag.chunker.chunk import ChunkMetadata
from rag.enricher.base import EnrichedChunk, Entity
from rag.loaders.base import DocumentMetadata


def _chunk(text: str = "hello world") -> EnrichedChunk:
    return EnrichedChunk(
        text=text,
        metadata=ChunkMetadata(chunk_index=0),
    )


def test_enriched_chunk_defaults() -> None:
    c = _chunk()
    assert c.hypothetical_questions == []
    assert c.entities == []
    assert c.language is None
    assert c.title is None


def test_enriched_chunk_inherits_token_count() -> None:
    c = _chunk()
    assert c.token_count == 0


def test_model_copy_update_preserves_other_fields() -> None:
    c = _chunk("some text")
    updated = c.model_copy(update={"language": "en"})
    assert updated.language == "en"
    assert updated.text == "some text"
    assert updated.entities == []


def test_entity_fields() -> None:
    e = Entity(text="London", label="GPE", start=0, end=6)
    assert e.text == "London"
    assert e.label == "GPE"
    assert e.start == 0
    assert e.end == 6
