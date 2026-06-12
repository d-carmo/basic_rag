"""Tests for OpenAIEmbedder."""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from rag.embedder.openai_embedder import OpenAIEmbedder


class _FakeResponse:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._data


def _fake_client(embeddings: list[list[float]]) -> Any:
    data = {"data": [{"embedding": emb, "index": i} for i, emb in enumerate(embeddings)]}
    client = AsyncMock()
    client.post = AsyncMock(return_value=_FakeResponse(data))
    return client


@pytest.mark.asyncio
async def test_embed_returns_correct_number_of_vectors() -> None:
    client = _fake_client([[0.1, 0.2], [0.3, 0.4]])
    embedder = OpenAIEmbedder(client=client, api_key="fake")
    result = await embedder.embed(["hello", "world"])
    assert len(result) == 2
    assert result[0] == [0.1, 0.2]


@pytest.mark.asyncio
async def test_embed_empty_returns_empty() -> None:
    embedder = OpenAIEmbedder(api_key="fake")
    result = await embedder.embed([])
    assert result == []


@pytest.mark.asyncio
async def test_embed_batches_large_input() -> None:
    n_texts = 5
    vecs = [[float(i)] for i in range(n_texts)]
    # batch_size=2 → 3 API calls for 5 texts
    call_count = 0

    async def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        nonlocal call_count
        texts = kwargs["json"]["input"]
        data = {
            "data": [
                {"embedding": [float(i)], "index": i} for i, _ in enumerate(texts)
            ]
        }
        call_count += 1
        return _FakeResponse(data)

    client = AsyncMock()
    client.post = AsyncMock(side_effect=fake_post)
    embedder = OpenAIEmbedder(batch_size=2, client=client, api_key="fake")
    result = await embedder.embed([f"text{i}" for i in range(n_texts)])
    assert len(result) == n_texts
    assert call_count == 3


@pytest.mark.asyncio
async def test_dimensions_included_in_payload() -> None:
    captured: dict[str, Any] = {}

    async def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        captured.update(kwargs["json"])
        return _FakeResponse({"data": [{"embedding": [0.1], "index": 0}]})

    client = AsyncMock()
    client.post = AsyncMock(side_effect=fake_post)
    embedder = OpenAIEmbedder(dimensions=256, client=client, api_key="fake")
    await embedder.embed(["test"])
    assert captured.get("dimensions") == 256


def test_model_id_without_dimensions() -> None:
    embedder = OpenAIEmbedder(model_name="text-embedding-3-large", api_key="k")
    assert embedder.model_id == "text-embedding-3-large"


def test_model_id_with_dimensions() -> None:
    embedder = OpenAIEmbedder(model_name="text-embedding-3-large", dimensions=256, api_key="k")
    assert embedder.model_id == "text-embedding-3-large-256d"
