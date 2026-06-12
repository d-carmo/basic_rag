from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from rag.retriever.base import (
    BaseQueryTransform,
    BaseReranker,
    BaseRetriever,
    RetrievalResult,
    rrf_merge,
)


@dataclass
class RetrieverConfig:
    top_k_candidates: int = 20
    top_k_final: int = 8
    rrf_k: int = 60


class RetrievalPipeline:
    """
    Orchestrates: query transforms → retrieval → RRF merge → reranking.

    Flow:
      1. The original query is always searched first.
      2. Each transform produces additional query variants; all are retrieved in parallel.
      3. Multiple result lists are fused via RRF.
      4. An optional reranker further re-orders the top candidates.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        transforms: list[BaseQueryTransform] | None = None,
        reranker: BaseReranker | None = None,
        config: RetrieverConfig | None = None,
    ) -> None:
        self._retriever = retriever
        self._transforms = transforms or []
        self._reranker = reranker
        self._config = config or RetrieverConfig()

    async def retrieve(
        self,
        query: str,
        filter_: Any = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        cfg = self._config
        top_k_final = top_k if top_k is not None else cfg.top_k_final

        # 1. Build query variants
        variant_lists = await asyncio.gather(
            *[t.transform(query) for t in self._transforms]
        )
        queries = [query] + [q for variants in variant_lists for q in variants]

        # 2. Retrieve for every variant in parallel
        results_per_query = list(
            await asyncio.gather(
                *[
                    self._retriever.retrieve(q, filter_=filter_, top_k=cfg.top_k_candidates)
                    for q in queries
                ]
            )
        )

        # 3. Merge
        if len(results_per_query) == 1:
            merged: list[RetrievalResult] = results_per_query[0]
        else:
            merged = rrf_merge(results_per_query, k=cfg.rrf_k)
        merged = merged[: cfg.top_k_candidates]

        # 4. Rerank
        if self._reranker is not None:
            merged = await self._reranker.rerank(query, merged, top_k=top_k_final)
        else:
            merged = merged[:top_k_final]

        # 5. Normalise ranks
        return [
            RetrievalResult(id=r.id, score=r.score, rank=i, payload=r.payload)
            for i, r in enumerate(merged)
        ]
