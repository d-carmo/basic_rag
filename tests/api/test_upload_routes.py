from __future__ import annotations

import io

import pytest
from httpx import ASGITransport, AsyncClient

from rag.api.app import create_app


class _FakeEmbedder:
    async def embed(self, texts):
        return [[0.1] * 4 for _ in texts]


class _FakeStore:
    def __init__(self):
        self.upserted: list = []

    async def upsert_chunks(self, chunks, dense_vectors=None, sparse_vectors=None):
        self.upserted.extend(chunks)


_TEXT_CONTENT = b"This is a plain text document about retrieval-augmented generation."
_MD_CONTENT = b"# Introduction\n\nRAG combines retrieval with generation."


@pytest.mark.asyncio
async def test_upload_no_store(app, auth_headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/v1/ingest/file",
            headers=auth_headers,
            files={"file": ("doc.txt", io.BytesIO(_TEXT_CONTENT), "text/plain")},
        )
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_upload_txt(settings, auth_headers):
    store = _FakeStore()
    app = create_app(settings=settings, store=store, embedder=_FakeEmbedder())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/v1/ingest/file",
            headers=auth_headers,
            files={"file": ("readme.txt", io.BytesIO(_TEXT_CONTENT), "text/plain")},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source_id"] == "readme.txt"
    assert body["status"] == "ok"
    assert body["chunk_count"] == 1
    assert store.upserted[0].metadata.source_url == "readme.txt"


@pytest.mark.asyncio
async def test_upload_txt_custom_source_id(settings, auth_headers):
    store = _FakeStore()
    app = create_app(settings=settings, store=store, embedder=_FakeEmbedder())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/v1/ingest/file",
            headers=auth_headers,
            files={"file": ("doc.txt", io.BytesIO(_TEXT_CONTENT), "text/plain")},
            data={"source_id": "my-source", "title": "My Doc"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source_id"] == "my-source"
    chunk = store.upserted[0]
    assert chunk.metadata.source_url == "my-source"
    assert chunk.metadata.extra.get("title") == "My Doc"


@pytest.mark.asyncio
async def test_upload_markdown(settings, auth_headers):
    store = _FakeStore()
    app = create_app(settings=settings, store=store, embedder=_FakeEmbedder())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/v1/ingest/file",
            headers=auth_headers,
            files={"file": ("guide.md", io.BytesIO(_MD_CONTENT), "text/markdown")},
        )
    assert resp.status_code == 200
    assert resp.json()["chunk_count"] == 1
    from rag.loaders.base import DocType
    assert store.upserted[0].metadata.doc_type == DocType.MARKDOWN


@pytest.mark.asyncio
async def test_upload_unsupported_type(settings, auth_headers):
    store = _FakeStore()
    app = create_app(settings=settings, store=store, embedder=_FakeEmbedder())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/v1/ingest/file",
            headers=auth_headers,
            files={"file": ("data.csv", io.BytesIO(b"a,b,c"), "text/csv")},
        )
    assert resp.status_code == 415


@pytest.mark.asyncio
async def test_upload_empty_file(settings, auth_headers):
    store = _FakeStore()
    app = create_app(settings=settings, store=store, embedder=_FakeEmbedder())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/v1/ingest/file",
            headers=auth_headers,
            files={"file": ("empty.txt", io.BytesIO(b"   "), "text/plain")},
        )
    assert resp.status_code == 422
