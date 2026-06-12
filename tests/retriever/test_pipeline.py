"""Tests for RetrievalPipeline."""

from __future__ import annotations

from typing import Any

import pytest

from rag.retriever.base import (
    BaseQueryTransform,
    BaseReranker,
    BaseRetriever,
    RetrievalResult,
)
from rag.retriever.pipeline import RetrieverConfig, RetrievalPipeline


# ── Stubs ─────────────────────────────────────────────────────────────────────

def _result(id_: str, score: float = 0.5, rank: int = 0) -> RetrievalResult:
    return RetrievalResult(id=id_, score=score, rank=rank, payload={"text": id_})


class _StubRetriever(BaseRetriever):
    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results
        self.calls: list[str] = []

    async def retrieve(self, query: str, filter_: Any = None, top_k: int = 20) -> list[RetrievalResult]:
        self.calls.append(query)
        return self._results


class _ConstantTransform(BaseQueryTransform):
    def __init__(self, variants: list[str]) -> None:
        self._variants = variants

    async def transform(self, query: str) -> list[str]:
        return self._variants


class _ScoreBoostReranker(BaseReranker):
    """Returns results with score=99.0 to distinguish reranked from raw."""

    async def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int = 8
    ) -> list[RetrievalResult]:
        return [
            RetrievalResult(id=r.id, score=99.0, rank=i, payload=r.payload)
            for i, r in enumerate(results[:top_k])
        ]


# ── No transforms, no reranker ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_returns_top_k_final() -> None:
    results = [_result(str(i)) for i in range(10)]
    retriever = _StubRetriever(results)
    config = RetrieverConfig(top_k_candidates=10, top_k_final=3)
    pipeline = RetrievalPipeline(retriever=retriever, config=config)
    final = await pipeline.retrieve("q")
    assert len(final) == 3


@pytest.mark.asyncio
async def test_pipeline_normalises_ranks() -> None:
    results = [_result(str(i), rank=i + 10) for i in range(5)]
    pipeline = RetrievalPipeline(
        retriever=_StubRetriever(results),
        config=RetrieverConfig(top_k_final=5),
    )
    final = await pipeline.retrieve("q")
    assert [r.rank for r in final] == list(range(len(final)))


@pytest.mark.asyncio
async def test_pipeline_always_searches_original_query() -> None:
    retriever = _StubRetriever([])
    pipeline = RetrievalPipeline(retriever=retriever)
    await pipeline.retrieve("original query")
    assert "original query" in retriever.calls


# ── With transforms ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_searches_all_transform_variants() -> None:
    retriever = _StubRetriever([])
    transform = _ConstantTransform(["variant A", "variant B"])
    pipeline = RetrievalPipeline(retriever=retriever, transforms=[transform])
    await pipeline.retrieve("original")
    assert set(retriever.calls) == {"original", "variant A", "variant B"}


@pytest.mark.asyncio
async def test_pipeline_empty_transform_output_still_searches_original() -> None:
    retriever = _StubRetriever([_result("x")])
    pipeline = RetrievalPipeline(
        retriever=retriever,
        transforms=[_ConstantTransform([])],
        config=RetrieverConfig(top_k_final=5),
    )
    final = await pipeline.retrieve("q")
    assert len(final) == 1
    assert "q" in retriever.calls


@pytest.mark.asyncio
async def test_pipeline_rrf_deduplicates_results() -> None:
    shared = _result("shared", score=0.9)
    only_in_transform = _result("extra", score=0.5)

    class _TransformRetriever(BaseRetriever):
        def __init__(self) -> None:
            self._call = 0

        async def retrieve(self, query: str, filter_: Any = None, top_k: int = 20) -> list[RetrievalResult]:
            self._call += 1
            if self._call == 1:
                return [shared]
            return [shared, only_in_transform]

    retriever = _TransformRetriever()
    transform = _ConstantTransform(["variant"])
    config = RetrieverConfig(top_k_final=10)
    pipeline = RetrievalPipeline(retriever=retriever, transforms=[transform], config=config)
    final = await pipeline.retrieve("q")
    # "shared" must appear exactly once
    assert sum(1 for r in final if r.id == "shared") == 1


# ── With reranker ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_applies_reranker() -> None:
    results = [_result(str(i)) for i in range(5)]
    pipeline = RetrievalPipeline(
        retriever=_StubRetriever(results),
        reranker=_ScoreBoostReranker(),
        config=RetrieverConfig(top_k_final=3),
    )
    final = await pipeline.retrieve("q")
    assert all(r.score == 99.0 for r in final)


@pytest.mark.asyncio
async def test_pipeline_reranker_limits_to_top_k_final() -> None:
    results = [_result(str(i)) for i in range(10)]
    pipeline = RetrievalPipeline(
        retriever=_StubRetriever(results),
        reranker=_ScoreBoostReranker(),
        config=RetrieverConfig(top_k_candidates=10, top_k_final=4),
    )
    final = await pipeline.retrieve("q")
    assert len(final) == 4
