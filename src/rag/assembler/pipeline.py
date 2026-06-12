from __future__ import annotations

from dataclasses import dataclass

from rag.assembler.base import AssembledContext, ContextChunk
from rag.assembler.budget import TokenBudgetManager
from rag.assembler.citations import CitationMapBuilder
from rag.assembler.dedup import NearDuplicateFilter
from rag.assembler.parent import ParentChunkFetcher
from rag.assembler.reorder import LostInMiddleReorder
from rag.retriever.base import RetrievalResult


@dataclass
class AssemblerConfig:
    max_context_tokens: int = 4096
    dedup_threshold: float = 0.85
    dedup_method: str = "jaccard"
    enable_lost_in_middle: bool = True
    enable_parent_fetch: bool = False


class ContextAssembler:
    """
    Orchestrates: parent fetch → dedup → reorder → token budget → citations.

    All steps are optional/injectable. Defaults produce a sensible out-of-the-box
    assembler with Jaccard dedup, lost-in-the-middle reordering, and a 4096-token budget.
    """

    def __init__(
        self,
        config: AssemblerConfig | None = None,
        dedup_filter: NearDuplicateFilter | None = None,
        budget_manager: TokenBudgetManager | None = None,
        citation_builder: CitationMapBuilder | None = None,
        parent_fetcher: ParentChunkFetcher | None = None,
        reorder: LostInMiddleReorder | None = None,
    ) -> None:
        cfg = config or AssemblerConfig()
        self._config = cfg
        self._dedup = dedup_filter or NearDuplicateFilter(
            threshold=cfg.dedup_threshold, method=cfg.dedup_method
        )
        self._budget = budget_manager or TokenBudgetManager(max_tokens=cfg.max_context_tokens)
        self._citations = citation_builder or CitationMapBuilder()
        self._parent_fetcher = parent_fetcher
        self._reorder = reorder or LostInMiddleReorder()

    async def assemble(
        self,
        results: list[RetrievalResult],
        vectors: list[list[float]] | None = None,
    ) -> AssembledContext:
        chunks: list[ContextChunk] = [
            ContextChunk(
                text=r.text,
                source_id=r.source_id,
                score=r.score,
                rank=r.rank,
                payload=r.payload,
            )
            for r in results
        ]

        if self._config.enable_parent_fetch and self._parent_fetcher is not None:
            chunks = await self._parent_fetcher.fetch(chunks)

        chunks = self._dedup.filter(chunks, vectors=vectors)

        if self._config.enable_lost_in_middle:
            chunks = self._reorder.reorder(chunks)

        chunks, truncated = self._budget.fit(chunks)

        citation_map = self._citations.build(chunks)

        total_tokens = sum(self._budget._count(c.text) for c in chunks)

        return AssembledContext(
            chunks=chunks,
            citation_map=citation_map,
            total_tokens=total_tokens,
            truncated=truncated,
        )
