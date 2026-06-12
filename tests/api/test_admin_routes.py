from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from rag.api.app import create_app


class _FakeLog:
    def get_all(self):
        return [
            {"source_id": "s1", "chunk_count": 3, "ingested_at": "2026-01-01T00:00:00Z"},
        ]

    def remove(self, _: str) -> None:
        pass


class _FakeStore:
    async def list_sources(self):
        return [{"source_id": "s1", "chunk_count": 3}]

    async def delete_by_source(self, source_id: str) -> None:
        pass

    async def collection_exists(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_health(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_ready_no_store(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["checks"]["qdrant"] is False


@pytest.mark.asyncio
async def test_auth_required(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/v1/sources")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_sources_empty(app, auth_headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/v1/sources", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["sources"] == []
    assert body["total"] == 0


@pytest.mark.asyncio
async def test_sources_with_store(settings, auth_headers):
    app = create_app(settings=settings, store=_FakeStore())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/v1/sources", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["sources"][0]["source_id"] == "s1"
    assert body["sources"][0]["chunk_count"] == 3


@pytest.mark.asyncio
async def test_delete_source_no_store(settings, auth_headers):
    app = create_app(settings=settings, ingestion_log=_FakeLog())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.delete("/v1/source/s1", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True


@pytest.mark.asyncio
async def test_wrong_api_key(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/v1/sources", headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401
