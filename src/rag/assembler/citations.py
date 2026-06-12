from __future__ import annotations

from typing import Any

from rag.assembler.base import ContextChunk


class CitationMapBuilder:
    """Build a 1-indexed citation map from the final assembled chunks."""

    def build(self, chunks: list[ContextChunk]) -> dict[int, dict[str, Any]]:
        return {
            i + 1: {
                "source_id": chunk.source_id,
                "title": chunk.title,
                "page": chunk.page,
                "score": chunk.score,
            }
            for i, chunk in enumerate(chunks)
        }
