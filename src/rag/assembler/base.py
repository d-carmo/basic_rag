from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContextChunk:
    text: str
    source_id: str
    score: float
    rank: int
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def title(self) -> str | None:
        return self.payload.get("title")  # type: ignore[return-value]

    @property
    def page(self) -> int | None:
        v = self.payload.get("page")
        return int(v) if v is not None else None


@dataclass
class AssembledContext:
    chunks: list[ContextChunk]
    citation_map: dict[int, dict[str, Any]]
    total_tokens: int
    truncated: bool
