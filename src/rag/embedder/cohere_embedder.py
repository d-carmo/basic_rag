from __future__ import annotations

import os
from typing import Literal

import httpx

from rag.embedder.base import BaseEmbedder

_API_URL = "https://api.cohere.ai/v1/embed"

InputType = Literal["search_document", "search_query", "classification", "clustering"]


class CohereEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str = "embed-english-v3.0",
        input_type: InputType = "search_document",
        api_key: str | None = None,
        batch_size: int = 96,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model_name = model_name
        self._input_type = input_type
        self._api_key = api_key or os.environ.get("COHERE_API_KEY", "")
        self._batch_size = batch_size
        self._client: httpx.AsyncClient | None = client

    @property
    def model_id(self) -> str:
        return f"{self._model_name}/{self._input_type}"

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = await self._get_client().post(
            _API_URL,
            json={
                "texts": texts,
                "model": self._model_name,
                "input_type": self._input_type,
            },
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        return response.json()["embeddings"]  # type: ignore[no-any-return]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            results.extend(await self._embed_batch(texts[i : i + self._batch_size]))
        return results
