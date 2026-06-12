"""Tests for SparseVector and BaseEmbedder contract."""

import pytest

from rag.embedder.base import BaseEmbedder, SparseVector


class _DummyEmbedder(BaseEmbedder):
    @property
    def model_id(self) -> str:
        return "dummy"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]


def test_sparse_vector_fields() -> None:
    sv = SparseVector(indices=[0, 5, 10], values=[0.5, 1.2, 0.8])
    assert sv.indices == [0, 5, 10]
    assert sv.values == [0.5, 1.2, 0.8]


@pytest.mark.asyncio
async def test_embed_sparse_raises_not_implemented_by_default() -> None:
    embedder = _DummyEmbedder()
    with pytest.raises(NotImplementedError):
        await embedder.embed_sparse(["hello"])


@pytest.mark.asyncio
async def test_embed_returns_one_vector_per_text() -> None:
    embedder = _DummyEmbedder()
    result = await embedder.embed(["a", "b", "c"])
    assert len(result) == 3
    assert all(len(v) == 2 for v in result)
