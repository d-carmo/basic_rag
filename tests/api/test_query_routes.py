from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from rag.api.app import create_app
from rag.assembler.base import AssembledContext, ContextChunk
from rag.generation.pipeline import GenerationResult


class _FakeRetriever:
    async def retrieve(self, query, filter_=None, top_k=8):
        return []


class _FakeAssembler:
    async def assemble(self, results, vectors=None):
        return AssembledContext(
            chunks=[
                ContextChunk(text="RAG context text", source_id="src1", score=0.9, rank=0)
            ],
            citation_map={1: {"source_id": "src1", "title": None, "page": None, "score": 0.9}},
            total_tokens=4,
            truncated=False,
        )


class _FakeGen:
    async def generate(self, query, context):
        return GenerationResult(answer="The answer is 42.", faithfulness_score=0.8)

    async def stream(self, query, context):
        for token in ["Hello", " world"]:
            yield token


@pytest.mark.asyncio
async def test_query_no_retriever(app, auth_headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/v1/query", json={"query": "hi"}, headers=auth_headers)
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_query_returns_answer(settings, auth_headers):
    app = create_app(
        settings=settings,
        retriever=_FakeRetriever(),
        assembler=_FakeAssembler(),
        generation_pipeline=_FakeGen(),
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/v1/query", json={"query": "what is 42?"}, headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "The answer is 42."
    assert body["faithfulness_score"] == pytest.approx(0.8)
    assert "X-Trace-ID" in resp.headers


@pytest.mark.asyncio
async def test_query_stream(settings, auth_headers):
    app = create_app(
        settings=settings,
        retriever=_FakeRetriever(),
        assembler=_FakeAssembler(),
        generation_pipeline=_FakeGen(),
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/v1/query",
            json={"query": "stream me", "stream": True},
            headers=auth_headers,
        )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert "data:" in resp.text


@pytest.mark.asyncio
async def test_query_empty_query(app, auth_headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/v1/query", json={"query": ""}, headers=auth_headers)
    assert resp.status_code == 422
