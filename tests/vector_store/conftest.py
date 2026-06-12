"""Shared fixtures for vector_store integration tests."""

from __future__ import annotations

import pytest


@pytest.fixture
async def qdrant_store():  # type: ignore[return]
    """Return a QdrantStore backed by an in-memory Qdrant instance.

    Skips the test if qdrant_client is not installed.
    """
    qdrant_client_mod = pytest.importorskip("qdrant_client", reason="qdrant_client not installed")
    AsyncQdrantClient = qdrant_client_mod.AsyncQdrantClient

    from rag.vector_store.store import QdrantStore, StoreConfig

    client = AsyncQdrantClient(location=":memory:")
    config = StoreConfig(collection_name="test", dense_dims=4, schema_version=1)
    store = QdrantStore(config=config, client=client)
    await store.get_or_create_collection()
    yield store
    await client.close()
