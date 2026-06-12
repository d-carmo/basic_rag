"""Tests for ContextChunk and AssembledContext — no external deps."""

from rag.assembler.base import AssembledContext, ContextChunk


def _chunk(**kwargs) -> ContextChunk:
    defaults = dict(text="hello", source_id="src", score=0.9, rank=0, payload={})
    return ContextChunk(**{**defaults, **kwargs})


# ── ContextChunk ───────────────────────────────────────────────────────────────

def test_context_chunk_fields() -> None:
    c = _chunk(text="test text", source_id="doc/1", score=0.8, rank=2)
    assert c.text == "test text"
    assert c.source_id == "doc/1"
    assert c.score == 0.8
    assert c.rank == 2


def test_context_chunk_title_from_payload() -> None:
    c = _chunk(payload={"title": "My Document"})
    assert c.title == "My Document"


def test_context_chunk_title_missing() -> None:
    c = _chunk(payload={})
    assert c.title is None


def test_context_chunk_page_from_payload() -> None:
    c = _chunk(payload={"page": 3})
    assert c.page == 3


def test_context_chunk_page_missing() -> None:
    c = _chunk(payload={})
    assert c.page is None


def test_context_chunk_page_coerces_to_int() -> None:
    c = _chunk(payload={"page": "7"})
    assert c.page == 7


# ── AssembledContext ───────────────────────────────────────────────────────────

def test_assembled_context_fields() -> None:
    chunks = [_chunk(text="a"), _chunk(text="b")]
    ctx = AssembledContext(
        chunks=chunks,
        citation_map={1: {"source_id": "s"}, 2: {"source_id": "t"}},
        total_tokens=10,
        truncated=False,
    )
    assert len(ctx.chunks) == 2
    assert ctx.total_tokens == 10
    assert ctx.truncated is False


def test_assembled_context_truncated_flag() -> None:
    ctx = AssembledContext(chunks=[], citation_map={}, total_tokens=0, truncated=True)
    assert ctx.truncated is True
