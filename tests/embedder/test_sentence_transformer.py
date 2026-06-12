"""Tests for SentenceTransformerEmbedder."""

import pytest

from rag.embedder.sentence_transformer import SentenceTransformerEmbedder


class _FakeArray:
    """Minimal numpy-array stand-in with a .tolist() method."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


class _FakeModel:
    """Minimal sentence-transformers mock."""

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> list[_FakeArray]:
        return [_FakeArray([float(i), float(i + 1)]) for i in range(len(texts))]


@pytest.mark.asyncio
async def test_embed_returns_correct_shape() -> None:
    embedder = SentenceTransformerEmbedder(model=_FakeModel())
    result = await embedder.embed(["hello", "world", "foo"])
    assert len(result) == 3
    assert all(len(v) == 2 for v in result)


@pytest.mark.asyncio
async def test_embed_empty_returns_empty() -> None:
    embedder = SentenceTransformerEmbedder(model=_FakeModel())
    result = await embedder.embed([])
    assert result == []


@pytest.mark.asyncio
async def test_embed_returns_float_lists() -> None:
    embedder = SentenceTransformerEmbedder(model=_FakeModel())
    result = await embedder.embed(["test"])
    assert isinstance(result[0], list)
    assert all(isinstance(v, float) for v in result[0])


def test_model_id_is_model_name() -> None:
    embedder = SentenceTransformerEmbedder(model_name="BAAI/bge-m3", model=_FakeModel())
    assert embedder.model_id == "BAAI/bge-m3"
