"""Tests for DenseRetriever and HybridRetriever using mock store and embedder."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from rag.embedder.base import BaseEmbedder, SparseVector
from rag.retriever.base import RetrievalResult
from rag.retriever.dense import DenseRetriever
from rag.retriever.hybrid import HybridRetriever
from rag.vector_store.store import SearchHit


# ── Fakes ─────────────────────────────────────────────────────────────────────

class _FakeEmbedder(BaseEmbedder):
    def __init__(self, vector: list[float], sparse: SparseVector | None = None) -> None:
        self._vector = vector
        self._sparse = sparse or SparseVector(indices=[0], values=[1.0])

    @property
    def model_id(self) -> str:
        return "fake"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector for _ in texts]

    async def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        return [self._sparse for _ in texts]


def _fake_store(hits: list[SearchHit]) -> Any:
    store = AsyncMock()
    store.search = AsyncMock(return_value=hits)
    return store


def _hit(id_: str, score: float, text: str = "doc") -> SearchHit:
    return SearchHit(id=id_, score=score, payload={"text": text, "source_id": "src"})


# ── DenseRetriever ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dense_retriever_returns_results() -> None:
    store = _fake_store([_hit("a", 0.9), _hit("b", 0.7)])
    retriever = DenseRetriever(store=store, embedder=_FakeEmbedder([0.1, 0.2]))
    results = await retriever.retrieve("what is photosynthesis?")
    assert len(results) == 2
    assert all(isinstance(r, RetrievalResult) for r in results)


@pytest.mark.asyncio
async def test_dense_retriever_calls_embed_then_search() -> None:
    store = _fake_store([_hit("x", 0.5)])
    embedder = _FakeEmbedder([1.0, 0.0])
    retriever = DenseRetriever(store=store, embedder=embedder)
    await retriever.retrieve("query", top_k=5)
    store.search.assert_called_once()
    call_kwargs = store.search.call_args.kwargs
    assert call_kwargs["dense_query"] == [1.0, 0.0]
    assert call_kwargs["top_k"] == 5


@pytest.mark.asyncio
async def test_dense_retriever_assigns_ranks() -> None:
    store = _fake_store([_hit("a", 0.9), _hit("b", 0.7), _hit("c", 0.5)])
    retriever = DenseRetriever(store=store, embedder=_FakeEmbedder([0.5]))
    results = await retriever.retrieve("q")
    assert [r.rank for r in results] == [0, 1, 2]


@pytest.mark.asyncio
async def test_dense_retriever_passes_filter() -> None:
    store = _fake_store([])
    retriever = DenseRetriever(store=store, embedder=_FakeEmbedder([1.0]))
    await retriever.retrieve("q", filter_="my_filter")
    assert store.search.call_args.kwargs["filter_"] == "my_filter"


@pytest.mark.asyncio
async def test_dense_retriever_empty_results() -> None:
    store = _fake_store([])
    retriever = DenseRetriever(store=store, embedder=_FakeEmbedder([0.1]))
    results = await retriever.retrieve("q")
    assert results == []


# ── HybridRetriever ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_hybrid_retriever_returns_results() -> None:
    sv = SparseVector(indices=[0, 1], values=[0.5, 0.3])
    store = _fake_store([_hit("a", 0.8)])
    retriever = HybridRetriever(
        store=store,
        dense_embedder=_FakeEmbedder([0.5, 0.5]),
        sparse_embedder=_FakeEmbedder([], sparse=sv),
    )
    results = await retriever.retrieve("query")
    assert len(results) == 1


@pytest.mark.asyncio
async def test_hybrid_retriever_passes_both_vectors() -> None:
    sv = SparseVector(indices=[2], values=[0.9])
    store = _fake_store([])
    retriever = HybridRetriever(
        store=store,
        dense_embedder=_FakeEmbedder([1.0, 0.0]),
        sparse_embedder=_FakeEmbedder([], sparse=sv),
    )
    await retriever.retrieve("q")
    call_kwargs = store.search.call_args.kwargs
    assert call_kwargs["dense_query"] == [1.0, 0.0]
    assert call_kwargs["sparse_query"] == sv
