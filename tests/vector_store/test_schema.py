"""Tests for ChunkPayload — pure Pydantic, no qdrant_client needed."""

from datetime import datetime, timezone

import pytest

from rag.chunker.chunk import Chunk, ChunkMetadata
from rag.enricher.base import EnrichedChunk, Entity
from rag.loaders.base import DocType, DocumentMetadata
from rag.vector_store.schema import ChunkPayload


def _meta(**kw: object) -> ChunkMetadata:
    defaults: dict[str, object] = dict(
        source_url="http://example.com/doc.pdf",
        doc_type=DocType.PDF,
        created_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        chunk_index=0,
    )
    defaults.update(kw)
    return ChunkMetadata(**defaults)  # type: ignore[arg-type]


def _chunk(text: str = "Hello world", **kw: object) -> Chunk:
    return Chunk(text=text, metadata=_meta(**kw))


def _enriched(text: str = "Hello world", **kw: object) -> EnrichedChunk:
    return EnrichedChunk(text=text, metadata=_meta(**kw))


# ── from plain Chunk ───────────────────────────────────────────────────────────

def test_payload_from_chunk_text() -> None:
    p = ChunkPayload.from_chunk(_chunk("some text"))
    assert p.text == "some text"


def test_payload_from_chunk_source_id() -> None:
    p = ChunkPayload.from_chunk(_chunk(source_url="http://foo.com/bar"))
    assert p.source_id == "http://foo.com/bar"


def test_payload_from_chunk_doc_type() -> None:
    p = ChunkPayload.from_chunk(_chunk())
    assert p.doc_type == "pdf"


def test_payload_from_chunk_created_at_is_iso() -> None:
    p = ChunkPayload.from_chunk(_chunk())
    # Should be a valid ISO 8601 string
    assert "2026-01-01" in p.created_at
    assert "T" in p.created_at


def test_payload_from_chunk_chunk_index() -> None:
    p = ChunkPayload.from_chunk(_chunk(chunk_index=7))
    assert p.chunk_index == 7


def test_payload_from_chunk_parent_id() -> None:
    p = ChunkPayload.from_chunk(_chunk(parent_id="abc-123"))
    assert p.parent_id == "abc-123"


def test_payload_from_chunk_token_count() -> None:
    c = _chunk()
    c = c.model_copy(update={"token_count": 42})
    p = ChunkPayload.from_chunk(c)
    assert p.token_count == 42


def test_payload_from_plain_chunk_has_empty_enriched_fields() -> None:
    p = ChunkPayload.from_chunk(_chunk())
    assert p.title is None
    assert p.hypothetical_questions == []
    assert p.entities == []


# ── from EnrichedChunk ─────────────────────────────────────────────────────────

def test_payload_from_enriched_chunk_title() -> None:
    c = _enriched()
    c = c.model_copy(update={"title": "My Title"})
    p = ChunkPayload.from_chunk(c)
    assert p.title == "My Title"


def test_payload_from_enriched_chunk_language() -> None:
    c = _enriched()
    c = c.model_copy(update={"language": "fr"})
    p = ChunkPayload.from_chunk(c)
    assert p.language == "fr"


def test_payload_from_enriched_chunk_hypothetical_questions() -> None:
    c = _enriched()
    c = c.model_copy(update={"hypothetical_questions": ["Q1?", "Q2?"]})
    p = ChunkPayload.from_chunk(c)
    assert p.hypothetical_questions == ["Q1?", "Q2?"]


def test_payload_from_enriched_chunk_entities() -> None:
    c = _enriched()
    c = c.model_copy(update={"entities": [Entity(text="Paris", label="GPE", start=0, end=5)]})
    p = ChunkPayload.from_chunk(c)
    assert len(p.entities) == 1
    assert p.entities[0]["text"] == "Paris"
    assert p.entities[0]["label"] == "GPE"


def test_payload_model_dump_is_json_serialisable() -> None:
    import json

    p = ChunkPayload.from_chunk(_enriched("test"))
    data = p.model_dump()
    # Should not raise
    json.dumps(data)
