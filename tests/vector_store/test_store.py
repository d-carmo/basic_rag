"""Integration tests for QdrantStore — require qdrant_client installed."""

from __future__ import annotations

import pytest

pytest.importorskip("qdrant_client", reason="qdrant_client not installed")

from rag.chunker.chunk import ChunkMetadata
from rag.embedder.base import SparseVector
from rag.enricher.base import EnrichedChunk
from rag.vector_store.store import QdrantStore, SearchHit, StoreConfig


def _chunk(text: str = "hello world", source: str = "http://example.com") -> EnrichedChunk:
    from rag.loaders.base import DocType
    return EnrichedChunk(
        text=text,
        metadata=ChunkMetadata(source_url=source, doc_type=DocType.TEXT, chunk_index=0),
    )


# ── collection management ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_collection_exists_after_create(qdrant_store: QdrantStore) -> None:
    assert await qdrant_store.collection_exists()


@pytest.mark.asyncio
async def test_get_or_create_is_idempotent(qdrant_store: QdrantStore) -> None:
    await qdrant_store.get_or_create_collection()  # should not raise even if exists
    assert await qdrant_store.collection_exists()


def test_collection_name_includes_version() -> None:
    store = QdrantStore(config=StoreConfig(collection_name="myrag", schema_version=3))
    assert store.collection_name == "myrag_v3"


def test_collection_name_for_version() -> None:
    store = QdrantStore(config=StoreConfig(collection_name="myrag"))
    assert store.collection_name_for_version(5) == "myrag_v5"


# ── upsert ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upsert_chunks_dense_only(qdrant_store: QdrantStore) -> None:
    chunks = [_chunk("apple"), _chunk("banana")]
    vecs = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    await qdrant_store.upsert_chunks(chunks, dense_vectors=vecs)


@pytest.mark.asyncio
async def test_upsert_chunks_with_sparse(qdrant_store: QdrantStore) -> None:
    chunks = [_chunk("cat sat")]
    dense = [[0.5, 0.5, 0.0, 0.0]]
    sparse = [SparseVector(indices=[0, 3], values=[0.8, 0.4])]
    await qdrant_store.upsert_chunks(chunks, dense_vectors=dense, sparse_vectors=sparse)


@pytest.mark.asyncio
async def test_upsert_is_idempotent(qdrant_store: QdrantStore) -> None:
    chunk = _chunk("hello")
    vec = [[1.0, 0.0, 0.0, 0.0]]
    await qdrant_store.upsert_chunks([chunk], dense_vectors=vec)
    await qdrant_store.upsert_chunks([chunk], dense_vectors=vec)  # same id → replace


# ── dense search ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_dense_returns_results(qdrant_store: QdrantStore) -> None:
    chunk = _chunk("the quick brown fox")
    await qdrant_store.upsert_chunks([chunk], dense_vectors=[[1.0, 0.0, 0.0, 0.0]])
    results = await qdrant_store.search(dense_query=[1.0, 0.0, 0.0, 0.0], top_k=5)
    assert len(results) >= 1
    assert isinstance(results[0], SearchHit)
    assert results[0].payload["text"] == "the quick brown fox"


@pytest.mark.asyncio
async def test_search_dense_score_between_0_and_1(qdrant_store: QdrantStore) -> None:
    chunk = _chunk("hello")
    await qdrant_store.upsert_chunks([chunk], dense_vectors=[[1.0, 0.0, 0.0, 0.0]])
    results = await qdrant_store.search(dense_query=[1.0, 0.0, 0.0, 0.0])
    assert 0.0 <= results[0].score <= 1.0


@pytest.mark.asyncio
async def test_search_empty_collection_returns_empty(qdrant_store: QdrantStore) -> None:
    results = await qdrant_store.search(dense_query=[1.0, 0.0, 0.0, 0.0])
    assert results == []


@pytest.mark.asyncio
async def test_search_raises_without_query(qdrant_store: QdrantStore) -> None:
    with pytest.raises(ValueError):
        await qdrant_store.search()


# ── delete ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_by_source_removes_points(qdrant_store: QdrantStore) -> None:
    source = "http://source-to-delete.com"
    chunks = [_chunk("doc text", source=source)]
    await qdrant_store.upsert_chunks(chunks, dense_vectors=[[1.0, 0.0, 0.0, 0.0]])

    before = await qdrant_store.search(dense_query=[1.0, 0.0, 0.0, 0.0])
    assert any(h.payload.get("source_id") == source for h in before)

    await qdrant_store.delete_by_source(source)

    after = await qdrant_store.search(dense_query=[1.0, 0.0, 0.0, 0.0])
    assert not any(h.payload.get("source_id") == source for h in after)


@pytest.mark.asyncio
async def test_delete_only_removes_matching_source(qdrant_store: QdrantStore) -> None:
    c1 = _chunk("keep this", source="http://keep.com")
    c2 = _chunk("delete this", source="http://delete.com")
    await qdrant_store.upsert_chunks([c1, c2], dense_vectors=[
        [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]
    ])

    await qdrant_store.delete_by_source("http://delete.com")

    results = await qdrant_store.search(dense_query=[1.0, 0.0, 0.0, 0.0], top_k=10)
    sources = {h.payload.get("source_id") for h in results}
    assert "http://keep.com" in sources
    assert "http://delete.com" not in sources
