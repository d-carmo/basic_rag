from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalResult:
    id: str
    score: float
    rank: int
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return str(self.payload.get("text", ""))

    @property
    def source_id(self) -> str:
        return str(self.payload.get("source_id", ""))


def rrf_merge(results_list: list[list[RetrievalResult]], k: int = 60) -> list[RetrievalResult]:
    """Reciprocal Rank Fusion: merge multiple ranked lists into a single ranked list."""
    scores: dict[str, float] = {}
    payloads: dict[str, dict[str, Any]] = {}

    for results in results_list:
        for rank, result in enumerate(results):
            scores[result.id] = scores.get(result.id, 0.0) + 1.0 / (k + rank + 1)
            payloads[result.id] = result.payload

    return [
        RetrievalResult(id=id_, score=scores[id_], rank=i, payload=payloads[id_])
        for i, id_ in enumerate(sorted(scores, key=lambda x: scores[x], reverse=True))
    ]


class BaseRetriever(ABC):
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        filter_: Any = None,
        top_k: int = 20,
    ) -> list[RetrievalResult]:
        ...


class BaseQueryTransform(ABC):
    @abstractmethod
    async def transform(self, query: str) -> list[str]:
        """Return additional query variants. The pipeline always searches the original first."""
        ...


class BaseReranker(ABC):
    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 8,
    ) -> list[RetrievalResult]:
        ...
