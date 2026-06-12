"""Tests for ContextAssembler pipeline."""

from __future__ import annotations

import pytest

from rag.assembler.base import ContextChunk
from rag.assembler.budget import TokenBudgetManager
from rag.assembler.dedup import NearDuplicateFilter
from rag.assembler.parent import ParentChunkFetcher
from rag.assembler.pipeline import AssemblerConfig, ContextAssembler
from rag.assembler.reorder import LostInMiddleReorder
from rag.retriever.base import RetrievalResult


def _result(id_: str, text: str, score: float = 0.9, rank: int = 0) -> RetrievalResult:
    return RetrievalResult(id=id_, score=score, rank=rank, payload={"text": text, "source_id": id_})


def _count(text: str) -> int:
    return len(text)


# ── basic assembly ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_assemble_returns_assembled_context() -> None:
    assembler = ContextAssembler(config=AssemblerConfig(enable_lost_in_middle=False))
    results = [_result("a", "hello world", rank=0), _result("b", "goodbye world", rank=1)]
    ctx = await assembler.assemble(results)
    assert len(ctx.chunks) == 2
    assert isinstance(ctx.citation_map, dict)
    assert ctx.total_tokens > 0


@pytest.mark.asyncio
async def test_empty_results_returns_empty_context() -> None:
    assembler = ContextAssembler()
    ctx = await assembler.assemble([])
    assert ctx.chunks == []
    assert ctx.citation_map == {}
    assert ctx.total_tokens == 0
    assert ctx.truncated is False


@pytest.mark.asyncio
async def test_results_converted_to_context_chunks() -> None:
    results = [_result("doc-1", "some text", score=0.77, rank=0)]
    assembler = ContextAssembler(config=AssemblerConfig(enable_lost_in_middle=False))
    ctx = await assembler.assemble(results)
    assert ctx.chunks[0].text == "some text"
    assert ctx.chunks[0].source_id == "doc-1"
    assert ctx.chunks[0].score == 0.77


# ── dedup ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dedup_removes_exact_duplicates() -> None:
    text = "identical content that repeats exactly here"
    results = [_result("a", text, rank=0), _result("b", text, rank=1)]
    assembler = ContextAssembler(config=AssemblerConfig(enable_lost_in_middle=False))
    ctx = await assembler.assemble(results)
    assert len(ctx.chunks) == 1


# ── lost-in-middle reorder ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_lost_in_middle_places_best_first() -> None:
    results = [_result(str(i), f"chunk {i}", score=float(10 - i), rank=i) for i in range(4)]
    assembler = ContextAssembler(
        config=AssemblerConfig(enable_lost_in_middle=True),
        dedup_filter=NearDuplicateFilter(threshold=2.0),  # threshold > 1.0 → never removes
    )
    ctx = await assembler.assemble(results)
    assert ctx.chunks[0].rank == 0


@pytest.mark.asyncio
async def test_lost_in_middle_disabled_preserves_input_order() -> None:
    results = [_result(str(i), f"unique text doc {i}", score=float(10 - i), rank=i) for i in range(3)]
    assembler = ContextAssembler(
        config=AssemblerConfig(enable_lost_in_middle=False),
        dedup_filter=NearDuplicateFilter(threshold=2.0),
    )
    ctx = await assembler.assemble(results)
    assert [c.rank for c in ctx.chunks] == [0, 1, 2]


# ── token budget ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_truncation_sets_truncated_flag() -> None:
    results = [_result("a", "a" * 100, rank=0), _result("b", "b" * 100, rank=1)]
    budget = TokenBudgetManager(max_tokens=50, token_counter=_count)
    assembler = ContextAssembler(
        config=AssemblerConfig(enable_lost_in_middle=False),
        dedup_filter=NearDuplicateFilter(threshold=0.01),
        budget_manager=budget,
    )
    ctx = await assembler.assemble(results)
    assert ctx.truncated is True


@pytest.mark.asyncio
async def test_no_truncation_when_fits() -> None:
    results = [_result("a", "hi", rank=0)]
    budget = TokenBudgetManager(max_tokens=1000, token_counter=_count)
    assembler = ContextAssembler(
        config=AssemblerConfig(enable_lost_in_middle=False),
        budget_manager=budget,
    )
    ctx = await assembler.assemble(results)
    assert ctx.truncated is False


# ── citations ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_citation_map_one_indexed() -> None:
    results = [_result("a", "text a", rank=0), _result("b", "text b", rank=1)]
    assembler = ContextAssembler(config=AssemblerConfig(enable_lost_in_middle=False))
    ctx = await assembler.assemble(results)
    assert 1 in ctx.citation_map
    assert 0 not in ctx.citation_map


# ── parent fetch ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_parent_fetch_enabled_replaces_text() -> None:
    async def _fetch(ids: list[str]) -> dict[str, dict]:
        return {"p1": {"text": "parent content"}}

    results = [
        RetrievalResult(
            id="child-1",
            score=0.9,
            rank=0,
            payload={"text": "child text", "source_id": "child-1", "parent_id": "p1"},
        )
    ]
    assembler = ContextAssembler(
        config=AssemblerConfig(enable_parent_fetch=True, enable_lost_in_middle=False),
        dedup_filter=NearDuplicateFilter(threshold=0.01),
        parent_fetcher=ParentChunkFetcher(fetch_fn=_fetch),
    )
    ctx = await assembler.assemble(results)
    assert ctx.chunks[0].text == "parent content"


@pytest.mark.asyncio
async def test_parent_fetch_disabled_skips_fetcher() -> None:
    called = []

    async def _fetch(ids: list[str]) -> dict[str, dict]:
        called.append(ids)
        return {}

    results = [
        RetrievalResult(
            id="c",
            score=0.9,
            rank=0,
            payload={"text": "original", "source_id": "c", "parent_id": "p1"},
        )
    ]
    assembler = ContextAssembler(
        config=AssemblerConfig(enable_parent_fetch=False, enable_lost_in_middle=False),
        parent_fetcher=ParentChunkFetcher(fetch_fn=_fetch),
    )
    ctx = await assembler.assemble(results)
    assert called == []
    assert ctx.chunks[0].text == "original"
