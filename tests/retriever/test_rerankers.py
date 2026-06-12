"""Tests for CrossEncoderReranker, CohereReranker, RerankerFactory."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from rag.retriever.base import RetrievalResult
from rag.retriever.rerankers import (
    CohereReranker,
    CrossEncoderReranker,
    RerankerConfig,
    RerankerFactory,
)


def _result(id_: str, rank: int = 0) -> RetrievalResult:
    return RetrievalResult(id=id_, score=0.5, rank=rank, payload={"text": f"text for {id_}"})


# ── CrossEncoderReranker ───────────────────────────────────────────────────────

class _FakeEncoder:
    """Score = 10 - index, so first document gets highest score by default."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(10 - i) for i in range(len(pairs))]


class _ReverseEncoder:
    """Gives last document the highest score → forces reordering."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(i) for i in range(len(pairs))]


@pytest.mark.asyncio
async def test_cross_encoder_reranker_returns_top_k() -> None:
    results = [_result(str(i)) for i in range(5)]
    reranker = CrossEncoderReranker(model=_FakeEncoder())
    reranked = await reranker.rerank("query", results, top_k=3)
    assert len(reranked) == 3


@pytest.mark.asyncio
async def test_cross_encoder_reranker_reorders_by_score() -> None:
    results = [_result("low"), _result("high")]
    reranker = CrossEncoderReranker(model=_ReverseEncoder())
    reranked = await reranker.rerank("query", results, top_k=2)
    # _ReverseEncoder gives score=1 to "high" (index 1) and score=0 to "low" (index 0)
    assert reranked[0].id == "high"


@pytest.mark.asyncio
async def test_cross_encoder_reranker_reassigns_ranks() -> None:
    results = [_result(str(i)) for i in range(3)]
    reranker = CrossEncoderReranker(model=_FakeEncoder())
    reranked = await reranker.rerank("q", results, top_k=3)
    assert [r.rank for r in reranked] == [0, 1, 2]


@pytest.mark.asyncio
async def test_cross_encoder_reranker_empty_input() -> None:
    reranker = CrossEncoderReranker(model=_FakeEncoder())
    assert await reranker.rerank("q", [], top_k=5) == []


# ── CohereReranker ─────────────────────────────────────────────────────────────

def _cohere_response(n: int) -> Any:
    class _Resp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return {
                "results": [
                    {"index": n - 1 - i, "relevance_score": float(i + 1)}
                    for i in range(n)
                ]
            }

    client = AsyncMock()
    client.post = AsyncMock(return_value=_Resp())
    return client


@pytest.mark.asyncio
async def test_cohere_reranker_returns_reranked_results() -> None:
    results = [_result("a"), _result("b"), _result("c")]
    reranker = CohereReranker(client=_cohere_response(3), api_key="fake")
    reranked = await reranker.rerank("query", results, top_k=3)
    assert len(reranked) == 3
    # Response reverses order: last index gets highest score
    assert reranked[0].id == "c"


@pytest.mark.asyncio
async def test_cohere_reranker_empty_input() -> None:
    reranker = CohereReranker(client=AsyncMock(), api_key="fake")
    assert await reranker.rerank("q", [], top_k=5) == []


@pytest.mark.asyncio
async def test_cohere_reranker_reassigns_ranks() -> None:
    results = [_result("x"), _result("y")]
    reranker = CohereReranker(client=_cohere_response(2), api_key="fake")
    reranked = await reranker.rerank("q", results, top_k=2)
    assert [r.rank for r in reranked] == [0, 1]


# ── RerankerFactory ────────────────────────────────────────────────────────────

def test_factory_creates_cross_encoder() -> None:
    config = RerankerConfig(backend="cross_encoder")
    reranker = RerankerFactory.create(config)
    assert isinstance(reranker, CrossEncoderReranker)


def test_factory_creates_cohere_reranker() -> None:
    config = RerankerConfig(backend="cohere", api_key="k")
    reranker = RerankerFactory.create(config)
    assert isinstance(reranker, CohereReranker)


def test_factory_unknown_backend_raises() -> None:
    config = RerankerConfig(backend="unknown")
    with pytest.raises(ValueError, match="unknown"):
        RerankerFactory.create(config)


def test_factory_passes_model_name() -> None:
    config = RerankerConfig(backend="cross_encoder", model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2")
    reranker = RerankerFactory.create(config)
    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker._model_name == "cross-encoder/ms-marco-TinyBERT-L-2-v2"
