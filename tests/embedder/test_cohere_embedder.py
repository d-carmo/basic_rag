"""Tests for CohereEmbedder."""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from rag.embedder.cohere_embedder import CohereEmbedder


class _FakeResponse:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self._embeddings = embeddings

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return {"embeddings": self._embeddings}


def _fake_client(embeddings: list[list[float]]) -> Any:
    client = AsyncMock()
    client.post = AsyncMock(return_value=_FakeResponse(embeddings))
    return client


@pytest.mark.asyncio
async def test_embed_returns_correct_vectors() -> None:
    client = _fake_client([[0.1, 0.2], [0.3, 0.4]])
    embedder = CohereEmbedder(client=client, api_key="fake")
    result = await embedder.embed(["hello", "world"])
    assert len(result) == 2
    assert result[0] == [0.1, 0.2]


@pytest.mark.asyncio
async def test_embed_empty_returns_empty() -> None:
    embedder = CohereEmbedder(api_key="fake")
    result = await embedder.embed([])
    assert result == []


@pytest.mark.asyncio
async def test_batching_splits_calls() -> None:
    call_count = 0

    async def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        nonlocal call_count
        n = len(kwargs["json"]["texts"])
        call_count += 1
        return _FakeResponse([[float(i)] for i in range(n)])

    client = AsyncMock()
    client.post = AsyncMock(side_effect=fake_post)
    embedder = CohereEmbedder(batch_size=3, client=client, api_key="fake")
    result = await embedder.embed(["t"] * 7)
    assert len(result) == 7
    assert call_count == 3


@pytest.mark.asyncio
async def test_input_type_in_payload() -> None:
    captured: dict[str, Any] = {}

    async def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        captured.update(kwargs["json"])
        return _FakeResponse([[0.1]])

    client = AsyncMock()
    client.post = AsyncMock(side_effect=fake_post)
    embedder = CohereEmbedder(input_type="search_query", client=client, api_key="fake")
    await embedder.embed(["test"])
    assert captured.get("input_type") == "search_query"


def test_model_id_includes_input_type() -> None:
    embedder = CohereEmbedder(
        model_name="embed-english-v3.0",
        input_type="search_document",
        api_key="k",
    )
    assert embedder.model_id == "embed-english-v3.0/search_document"
