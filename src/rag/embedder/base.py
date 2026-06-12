from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class SparseVector(BaseModel):
    indices: list[int]
    values: list[float]


class BaseEmbedder(ABC):
    @property
    @abstractmethod
    def model_id(self) -> str:
        ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    async def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        raise NotImplementedError(
            f"{type(self).__name__} does not support sparse embeddings"
        )
