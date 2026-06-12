from __future__ import annotations

import os
from typing import Any

import httpx

from rag.embedder.base import BaseEmbedder

_API_URL = "https://api.openai.com/v1/embeddings"


class OpenAIEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        dimensions: int | None = None,
        api_key: str | None = None,
        batch_size: int = 100,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model_name = model_name
        self._dimensions = dimensions
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._batch_size = batch_size
        self._client: httpx.AsyncClient | None = client

    @property
    def model_id(self) -> str:
        suffix = f"-{self._dimensions}d" if self._dimensions else ""
        return f"{self._model_name}{suffix}"

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {"model": self._model_name, "input": texts}
        if self._dimensions is not None:
            payload["dimensions"] = self._dimensions
        response = await self._get_client().post(
            _API_URL,
            json=payload,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        response.raise_for_status()
        data = response.json()
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            results.extend(await self._embed_batch(texts[i : i + self._batch_size]))
        return results
