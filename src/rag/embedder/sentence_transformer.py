from __future__ import annotations

import asyncio
from typing import Any

from rag.embedder.base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 32,
        model: Any = None,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: Any = model

    @property
    def model_id(self) -> str:
        return self._model_name

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [emb.tolist() for emb in embeddings]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, texts)
