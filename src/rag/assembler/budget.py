from __future__ import annotations

import dataclasses
from typing import Callable

from rag.assembler.base import ContextChunk


def _approx_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


class TokenBudgetManager:
    """
    Fit chunks within a maximum token budget.

    When a chunk would exceed the budget it is trimmed at word boundaries so
    the context is maximally filled. Sets `truncated=True` whenever any text
    is dropped or trimmed.
    """

    def __init__(
        self,
        max_tokens: int,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self._max_tokens = max_tokens
        self._count = token_counter or _approx_tokens

    def fit(self, chunks: list[ContextChunk]) -> tuple[list[ContextChunk], bool]:
        fitted: list[ContextChunk] = []
        used = 0
        truncated = False

        for chunk in chunks:
            cost = self._count(chunk.text)
            if used + cost <= self._max_tokens:
                fitted.append(chunk)
                used += cost
            else:
                remaining = self._max_tokens - used
                if remaining <= 0:
                    truncated = True
                    break
                # Binary-search for how many words fit within the remaining budget
                words = chunk.text.split()
                lo, hi = 0, len(words)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if self._count(" ".join(words[:mid])) <= remaining:
                        lo = mid
                    else:
                        hi = mid - 1
                if lo > 0:
                    fitted.append(dataclasses.replace(chunk, text=" ".join(words[:lo])))
                truncated = True
                break

        return fitted, truncated
