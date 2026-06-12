from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import httpx

from rag.retriever.base import BaseReranker, RetrievalResult

_COHERE_RERANK_URL = "https://api.cohere.ai/v1/rerank"


class CrossEncoderReranker(BaseReranker):
    """Local cross-encoder reranker via sentence-transformers."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        model: Any = None,
    ) -> None:
        self._model_name = model_name
        self._model: Any = model

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]

            self._model = CrossEncoder(self._model_name)
        return self._model

    def _score(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = self._get_model().predict(pairs)
        # predict returns numpy array or list; normalise to list[float]
        return [float(s) for s in scores]

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 8,
    ) -> list[RetrievalResult]:
        if not results:
            return []
        pairs = [(query, r.text) for r in results]
        scores = await asyncio.to_thread(self._score, pairs)
        ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        return [
            RetrievalResult(id=r.id, score=float(s), rank=i, payload=r.payload)
            for i, (r, s) in enumerate(ranked[:top_k])
        ]


class CohereReranker(BaseReranker):
    """Cohere Rerank v3 via httpx."""

    def __init__(
        self,
        model_name: str = "rerank-english-v3.0",
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model_name = model_name
        self._api_key = api_key or os.environ.get("COHERE_API_KEY", "")
        self._client: httpx.AsyncClient | None = client

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 8,
    ) -> list[RetrievalResult]:
        if not results:
            return []
        response = await self._get_client().post(
            _COHERE_RERANK_URL,
            json={
                "query": query,
                "documents": [r.text for r in results],
                "model": self._model_name,
                "top_n": top_k,
            },
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        response.raise_for_status()
        data = response.json()
        return [
            RetrievalResult(
                id=results[item["index"]].id,
                score=float(item["relevance_score"]),
                rank=i,
                payload=results[item["index"]].payload,
            )
            for i, item in enumerate(data["results"])
        ]


@dataclass
class RerankerConfig:
    backend: str = "cross_encoder"
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    api_key: str | None = None


class RerankerFactory:
    @staticmethod
    def create(config: RerankerConfig) -> BaseReranker:
        if config.backend == "cross_encoder":
            return CrossEncoderReranker(model_name=config.model_name)
        if config.backend == "cohere":
            return CohereReranker(model_name=config.model_name, api_key=config.api_key)
        raise ValueError(f"Unknown reranker backend: {config.backend!r}")
