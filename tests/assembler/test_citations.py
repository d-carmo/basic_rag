"""Tests for CitationMapBuilder."""

from rag.assembler.base import ContextChunk
from rag.assembler.citations import CitationMapBuilder


def _chunk(source_id: str = "src", title: str | None = None, page: int | None = None, score: float = 0.9) -> ContextChunk:
    payload: dict = {}
    if title is not None:
        payload["title"] = title
    if page is not None:
        payload["page"] = page
    return ContextChunk(text="text", source_id=source_id, score=score, rank=0, payload=payload)


builder = CitationMapBuilder()


def test_empty_chunks_returns_empty_map() -> None:
    assert builder.build([]) == {}


def test_citation_map_is_one_indexed() -> None:
    chunks = [_chunk("a"), _chunk("b")]
    cmap = builder.build(chunks)
    assert 1 in cmap
    assert 2 in cmap
    assert 0 not in cmap


def test_citation_map_source_id() -> None:
    cmap = builder.build([_chunk("http://example.com/doc")])
    assert cmap[1]["source_id"] == "http://example.com/doc"


def test_citation_map_title_present() -> None:
    cmap = builder.build([_chunk(title="My Paper")])
    assert cmap[1]["title"] == "My Paper"


def test_citation_map_title_none() -> None:
    cmap = builder.build([_chunk()])
    assert cmap[1]["title"] is None


def test_citation_map_page_present() -> None:
    cmap = builder.build([_chunk(page=7)])
    assert cmap[1]["page"] == 7


def test_citation_map_score_included() -> None:
    cmap = builder.build([_chunk(score=0.77)])
    assert cmap[1]["score"] == 0.77


def test_citation_map_length_matches_chunks() -> None:
    chunks = [_chunk(f"src/{i}") for i in range(5)]
    cmap = builder.build(chunks)
    assert len(cmap) == 5
    assert set(cmap.keys()) == {1, 2, 3, 4, 5}
