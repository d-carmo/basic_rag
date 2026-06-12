"""Tests for BM25SparseEmbedder."""

import pytest

from rag.embedder.bm25 import BM25SparseEmbedder
from rag.embedder.base import SparseVector


def _fitted(corpus: list[str] | None = None) -> BM25SparseEmbedder:
    embedder = BM25SparseEmbedder()
    embedder.fit(corpus or ["the cat sat on the mat", "the dog barked at the cat"])
    return embedder


@pytest.mark.asyncio
async def test_embed_sparse_returns_one_vector_per_text() -> None:
    embedder = _fitted()
    result = await embedder.embed_sparse(["cat sat", "dog barked"])
    assert len(result) == 2
    assert all(isinstance(v, SparseVector) for v in result)


@pytest.mark.asyncio
async def test_embed_sparse_empty_returns_empty() -> None:
    embedder = _fitted()
    result = await embedder.embed_sparse([])
    assert result == []


@pytest.mark.asyncio
async def test_in_vocab_tokens_produce_nonzero_weights() -> None:
    embedder = _fitted(["apple banana cherry"])
    (sv,) = await embedder.embed_sparse(["apple"])
    assert len(sv.indices) > 0
    assert all(v > 0 for v in sv.values)


@pytest.mark.asyncio
async def test_out_of_vocab_tokens_ignored() -> None:
    embedder = _fitted(["hello world"])
    (sv,) = await embedder.embed_sparse(["xyz_unknown_token"])
    assert sv.indices == []
    assert sv.values == []


@pytest.mark.asyncio
async def test_embed_raises_not_implemented() -> None:
    embedder = _fitted()
    with pytest.raises(NotImplementedError):
        await embedder.embed(["hello"])


def test_embed_sparse_raises_before_fit() -> None:
    embedder = BM25SparseEmbedder()
    with pytest.raises(RuntimeError, match="fit"):
        embedder._vectorize("hello world")


def test_model_id() -> None:
    assert BM25SparseEmbedder().model_id == "bm25"


@pytest.mark.asyncio
async def test_repeated_terms_have_higher_weight() -> None:
    corpus = ["apple apple apple mango", "mango mango mango apple"]
    embedder = BM25SparseEmbedder()
    embedder.fit(corpus)
    sv_a, sv_b = await embedder.embed_sparse(["apple apple apple", "apple"])
    weight_a = dict(zip(sv_a.indices, sv_a.values))
    weight_b = dict(zip(sv_b.indices, sv_b.values))
    apple_idx_a = next(
        embedder._vocab[t] for t in embedder._vocab if t == "apple"
    )
    assert weight_a[apple_idx_a] > weight_b[apple_idx_a]
