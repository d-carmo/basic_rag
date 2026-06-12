from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from rag.embedder.base import BaseEmbedder, SparseVector


def _key(model_id: str, text: str) -> str:
    return "emb:" + hashlib.sha256(f"{model_id}:{text}".encode()).hexdigest()


class EmbeddingCache(ABC):
    @abstractmethod
    async def get(self, model_id: str, text: str) -> list[float] | None:
        ...

    @abstractmethod
    async def set(self, model_id: str, text: str, vector: list[float]) -> None:
        ...


class RedisEmbeddingCache(EmbeddingCache):
    def __init__(self, url: str = "redis://localhost:6379", ttl: int = 86400) -> None:
        self._url = url
        self._ttl = ttl
        self._redis: Any = None

    def _get_redis(self) -> Any:
        if self._redis is None:
            import redis.asyncio as aioredis  # type: ignore[import-untyped]

            self._redis = aioredis.from_url(self._url, decode_responses=True)
        return self._redis

    async def get(self, model_id: str, text: str) -> list[float] | None:
        raw: str | None = await self._get_redis().get(_key(model_id, text))
        return json.loads(raw) if raw is not None else None

    async def set(self, model_id: str, text: str, vector: list[float]) -> None:
        await self._get_redis().setex(_key(model_id, text), self._ttl, json.dumps(vector))


class SQLiteEmbeddingCache(EmbeddingCache):
    def __init__(self, db_path: str | Path = ".embeddings_cache.db", ttl: int = 86400) -> None:
        self._db_path = str(db_path)
        self._ttl = ttl
        self._init()

    def _init(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS embeddings "
                "(key TEXT PRIMARY KEY, vector TEXT NOT NULL, expires_at INTEGER NOT NULL)"
            )

    def _get_sync(self, key: str) -> list[float] | None:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT vector FROM embeddings WHERE key = ? AND expires_at > ?",
                (key, int(time.time())),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def _set_sync(self, key: str, vector: list[float]) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (key, vector, expires_at) VALUES (?, ?, ?)",
                (key, json.dumps(vector), int(time.time()) + self._ttl),
            )

    async def get(self, model_id: str, text: str) -> list[float] | None:
        return await asyncio.to_thread(self._get_sync, _key(model_id, text))

    async def set(self, model_id: str, text: str, vector: list[float]) -> None:
        await asyncio.to_thread(self._set_sync, _key(model_id, text), vector)


class CachedEmbedder(BaseEmbedder):
    """Wraps a BaseEmbedder, serving hits from cache and batching misses to the underlying embedder."""

    def __init__(self, embedder: BaseEmbedder, cache: EmbeddingCache) -> None:
        self._embedder = embedder
        self._cache = cache

    @property
    def model_id(self) -> str:
        return self._embedder.model_id

    async def embed(self, texts: list[str]) -> list[list[float]]:
        model_id = self.model_id
        hits: dict[int, list[float]] = {}

        for i, text in enumerate(texts):
            cached = await self._cache.get(model_id, text)
            if cached is not None:
                hits[i] = cached

        miss_indices = [i for i in range(len(texts)) if i not in hits]
        if miss_indices:
            miss_vectors = await self._embedder.embed([texts[i] for i in miss_indices])
            for i, vector in zip(miss_indices, miss_vectors):
                hits[i] = vector
                await self._cache.set(model_id, texts[i], vector)

        return [hits[i] for i in range(len(texts))]

    async def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        return await self._embedder.embed_sparse(texts)
