"""Tests for ParentChunkFetcher."""

from __future__ import annotations

import pytest

from rag.assembler.base import ContextChunk
from rag.assembler.parent import ParentChunkFetcher


def _chunk(text: str, parent_id: str | None = None) -> ContextChunk:
    payload: dict = {}
    if parent_id is not None:
        payload["parent_id"] = parent_id
    return ContextChunk(text=text, source_id="src", score=0.9, rank=0, payload=payload)


async def _fetch(ids: list[str]) -> dict[str, dict]:
    db = {
        "parent-1": {"text": "Parent text for doc 1", "title": "Doc 1"},
        "parent-2": {"text": "Parent text for doc 2"},
    }
    return {id_: db[id_] for id_ in ids if id_ in db}


# ── basic behaviour ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_no_parent_id_chunks_unchanged() -> None:
    chunks = [_chunk("child text")]
    fetcher = ParentChunkFetcher(fetch_fn=_fetch)
    result = await fetcher.fetch(chunks)
    assert len(result) == 1
    assert result[0].text == "child text"


@pytest.mark.asyncio
async def test_parent_text_replaces_child_text() -> None:
    chunks = [_chunk("child text", parent_id="parent-1")]
    fetcher = ParentChunkFetcher(fetch_fn=_fetch)
    result = await fetcher.fetch(chunks)
    assert result[0].text == "Parent text for doc 1"


@pytest.mark.asyncio
async def test_parent_payload_merged() -> None:
    chunks = [_chunk("child", parent_id="parent-1")]
    fetcher = ParentChunkFetcher(fetch_fn=_fetch)
    result = await fetcher.fetch(chunks)
    assert result[0].payload.get("title") == "Doc 1"
    assert result[0].payload.get("parent_id") == "parent-1"


@pytest.mark.asyncio
async def test_parent_id_not_in_fetch_result_leaves_chunk_unchanged() -> None:
    chunks = [_chunk("child text", parent_id="missing-parent")]
    fetcher = ParentChunkFetcher(fetch_fn=_fetch)
    result = await fetcher.fetch(chunks)
    assert result[0].text == "child text"


@pytest.mark.asyncio
async def test_mixed_chunks_with_and_without_parent() -> None:
    chunks = [
        _chunk("no parent"),
        _chunk("child 1", parent_id="parent-1"),
        _chunk("child 2", parent_id="parent-2"),
    ]
    fetcher = ParentChunkFetcher(fetch_fn=_fetch)
    result = await fetcher.fetch(chunks)
    assert result[0].text == "no parent"
    assert result[1].text == "Parent text for doc 1"
    assert result[2].text == "Parent text for doc 2"


@pytest.mark.asyncio
async def test_deduplicates_fetch_calls() -> None:
    """Parent IDs that appear multiple times should only be fetched once."""
    call_log: list[list[str]] = []

    async def _logging_fetch(ids: list[str]) -> dict[str, dict]:
        call_log.append(sorted(ids))
        return {"parent-1": {"text": "parent"}}

    chunks = [
        _chunk("c1", parent_id="parent-1"),
        _chunk("c2", parent_id="parent-1"),
    ]
    fetcher = ParentChunkFetcher(fetch_fn=_logging_fetch)
    await fetcher.fetch(chunks)
    assert len(call_log) == 1
    assert call_log[0] == ["parent-1"]
