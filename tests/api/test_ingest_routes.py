from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from rag.api.app import create_app


class _FakeEmbedder:
    async def embed(self, texts):
        return [[0.1] * 4 for _ in texts]


class _FakeStore:
    def __init__(self):
        self.upserted = []

    async def upsert_chunks(self, chunks, dense_vectors=None, sparse_vectors=None):
        self.upserted.extend(chunks)


@pytest.mark.asyncio
async def test_ingest_no_store(app, auth_headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/v1/ingest",
            json={"text": "hello", "source_id": "doc1"},
            headers=auth_headers,
        )
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_ingest_ok(settings, auth_headers):
    store = _FakeStore()
    app = create_app(settings=settings, store=store, embedder=_FakeEmbedder())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/v1/ingest",
            json={"text": "hello world", "source_id": "doc1"},
            headers=auth_headers,
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source_id"] == "doc1"
    assert body["status"] == "ok"
    assert body["chunk_count"] == 1
    assert len(store.upserted) == 1


@pytest.mark.asyncio
async def test_batch_ingest_ok(settings, auth_headers):
    store = _FakeStore()
    app = create_app(settings=settings, store=store, embedder=_FakeEmbedder())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/v1/ingest/batch",
            json={
                "documents": [
                    {"text": "doc one", "source_id": "d1"},
                    {"text": "doc two", "source_id": "d2"},
                ]
            },
            headers=auth_headers,
        )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["ingested"]) == 2
    assert body["failed"] == []


@pytest.mark.asyncio
async def test_ingest_with_title(settings, auth_headers):
    store = _FakeStore()
    app = create_app(settings=settings, store=store, embedder=_FakeEmbedder())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/v1/ingest",
            json={"text": "content here", "source_id": "doc2", "title": "My Doc", "doc_type": "markdown"},
            headers=auth_headers,
        )
    assert resp.status_code == 200
    chunk = store.upserted[0]
    assert chunk.metadata.extra.get("title") == "My Doc"
