from __future__ import annotations

from typing import Any

from rag.embedder.base import BaseEmbedder
from rag.retriever.base import BaseRetriever, RetrievalResult
from rag.vector_store.store import QdrantStore


class DenseRetriever(BaseRetriever):
    def __init__(self, store: QdrantStore, embedder: BaseEmbedder) -> None:
        self._store = store
        self._embedder = embedder

    async def retrieve(
        self,
        query: str,
        filter_: Any = None,
        top_k: int = 20,
    ) -> list[RetrievalResult]:
        vectors = await self._embedder.embed([query])
        hits = await self._store.search(
            dense_query=vectors[0],
            filter_=filter_,
            top_k=top_k,
        )
        return [
            RetrievalResult(id=h.id, score=h.score, rank=i, payload=h.payload)
            for i, h in enumerate(hits)
        ]
