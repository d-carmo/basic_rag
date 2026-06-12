from __future__ import annotations

from rag.assembler.base import ContextChunk


class LostInMiddleReorder:
    """
    Reorder chunks so the most relevant appear at the edges, least relevant in the middle.

    LLMs attend more to context at the beginning and end of the prompt. This reordering
    places rank-0 (best) at position 0, rank-1 (second best) at the last position, and
    interleaves the rest inward from both ends.
    """

    def reorder(self, chunks: list[ContextChunk]) -> list[ContextChunk]:
        if len(chunks) <= 1:
            return list(chunks)

        result: list[ContextChunk | None] = [None] * len(chunks)
        front, back = 0, len(chunks) - 1
        for i, chunk in enumerate(chunks):
            if i % 2 == 0:
                result[front] = chunk
                front += 1
            else:
                result[back] = chunk
                back -= 1
        return [c for c in result if c is not None]
