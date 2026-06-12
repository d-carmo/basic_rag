from __future__ import annotations

import asyncio
from typing import Any

from rag.embedder.base import BaseEmbedder
from rag.retriever.base import BaseRetriever, RetrievalResult
from rag.vector_store.store import QdrantStore


class HybridRetriever(BaseRetriever):
    """Dense + sparse retrieval with Qdrant's native RRF fusion."""

    def __init__(
        self,
        store: QdrantStore,
        dense_embedder: BaseEmbedder,
        sparse_embedder: BaseEmbedder,
    ) -> None:
        self._store = store
        self._dense = dense_embedder
        self._sparse = sparse_embedder

    async def retrieve(
        self,
        query: str,
        filter_: Any = None,
        top_k: int = 20,
    ) -> list[RetrievalResult]:
        dense_vecs, sparse_vecs = await asyncio.gather(
            self._dense.embed([query]),
            self._sparse.embed_sparse([query]),
        )
        hits = await self._store.search(
            dense_query=dense_vecs[0],
            sparse_query=sparse_vecs[0],
            filter_=filter_,
            top_k=top_k,
        )
        return [
            RetrievalResult(id=h.id, score=h.score, rank=i, payload=h.payload)
            for i, h in enumerate(hits)
        ]
