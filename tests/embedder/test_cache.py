"""Tests for SQLiteEmbeddingCache and CachedEmbedder."""

import pytest

from rag.embedder.base import BaseEmbedder
from rag.embedder.cache import CachedEmbedder, EmbeddingCache, SQLiteEmbeddingCache


# ── In-memory cache for CachedEmbedder tests ──────────────────────────────────

class _MemoryCache(EmbeddingCache):
    def __init__(self) -> None:
        self._store: dict[str, list[float]] = {}

    async def get(self, model_id: str, text: str) -> list[float] | None:
        return self._store.get(f"{model_id}:{text}")

    async def set(self, model_id: str, text: str, vector: list[float]) -> None:
        self._store[f"{model_id}:{text}"] = vector


# ── Fake embedder ──────────────────────────────────────────────────────────────

class _CountingEmbedder(BaseEmbedder):
    def __init__(self) -> None:
        self.call_count = 0

    @property
    def model_id(self) -> str:
        return "counter"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        return [[float(i), float(i + 1)] for i in range(len(texts))]


# ── SQLiteEmbeddingCache ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sqlite_cache_miss_returns_none(tmp_path: pytest.TempPathFactory) -> None:
    cache = SQLiteEmbeddingCache(db_path=tmp_path / "test.db")  # type: ignore[operator]
    result = await cache.get("model", "hello")
    assert result is None


@pytest.mark.asyncio
async def test_sqlite_cache_set_then_get(tmp_path: pytest.TempPathFactory) -> None:
    cache = SQLiteEmbeddingCache(db_path=tmp_path / "test.db")  # type: ignore[operator]
    vector = [0.1, 0.2, 0.3]
    await cache.set("model", "hello", vector)
    result = await cache.get("model", "hello")
    assert result == vector


@pytest.mark.asyncio
async def test_sqlite_cache_different_texts_isolated(tmp_path: pytest.TempPathFactory) -> None:
    cache = SQLiteEmbeddingCache(db_path=tmp_path / "test.db")  # type: ignore[operator]
    await cache.set("m", "text1", [1.0])
    await cache.set("m", "text2", [2.0])
    assert await cache.get("m", "text1") == [1.0]
    assert await cache.get("m", "text2") == [2.0]


@pytest.mark.asyncio
async def test_sqlite_cache_expired_returns_none(tmp_path: pytest.TempPathFactory) -> None:
    cache = SQLiteEmbeddingCache(db_path=tmp_path / "test.db", ttl=-1)  # type: ignore[operator]
    await cache.set("m", "hello", [0.5])
    result = await cache.get("m", "hello")
    assert result is None


# ── CachedEmbedder ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cached_embedder_calls_underlying_on_miss() -> None:
    inner = _CountingEmbedder()
    cached = CachedEmbedder(inner, _MemoryCache())
    result = await cached.embed(["a", "b"])
    assert len(result) == 2
    assert inner.call_count == 1


@pytest.mark.asyncio
async def test_cached_embedder_skips_call_on_hit() -> None:
    inner = _CountingEmbedder()
    mem = _MemoryCache()
    cached = CachedEmbedder(inner, mem)

    await cached.embed(["a"])          # miss → call embedder
    await cached.embed(["a"])          # hit → no embedder call
    assert inner.call_count == 1


@pytest.mark.asyncio
async def test_cached_embedder_partial_hit() -> None:
    inner = _CountingEmbedder()
    mem = _MemoryCache()
    cached = CachedEmbedder(inner, mem)

    await cached.embed(["a"])          # caches "a"
    result = await cached.embed(["a", "b"])  # "a" hits, "b" misses
    assert len(result) == 2
    assert inner.call_count == 2       # once for ["a"], once for ["b"]


@pytest.mark.asyncio
async def test_cached_embedder_model_id_delegates() -> None:
    cached = CachedEmbedder(_CountingEmbedder(), _MemoryCache())
    assert cached.model_id == "counter"
