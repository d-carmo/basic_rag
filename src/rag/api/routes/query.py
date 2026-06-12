from __future__ import annotations

import json
from collections.abc import AsyncIterator
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from rag.api.deps import get_assembler, get_generation_pipeline, get_retriever
from rag.api.models import CitationSource, QueryRequest, QueryResponse

router = APIRouter()


async def _sse(gen: AsyncIterator[str], trace_id: str):
    async for chunk in gen:
        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
    yield f"data: {json.dumps({'done': True, 'trace_id': trace_id})}\n\n"


@router.post("/v1/query")
async def query(body: QueryRequest, request: Request):
    retriever = get_retriever(request)
    assembler = get_assembler(request)
    gen_pipeline = get_generation_pipeline(request)

    trace_id = str(uuid4())

    filter_ = None
    if body.doc_type:
        from rag.vector_store.filters import FilterBuilder
        filter_ = FilterBuilder.by_doc_type(body.doc_type)

    results = await retriever.retrieve(body.query, filter_=filter_, top_k=body.top_k * 3)
    context = await assembler.assemble(results)

    if body.stream:
        return StreamingResponse(
            _sse(gen_pipeline.stream(body.query, context), trace_id),
            media_type="text/event-stream",
            headers={"X-Trace-ID": trace_id},
        )

    result = await gen_pipeline.generate(body.query, context)
    sources = [
        CitationSource(
            citation_number=i + 1,
            source_id=c.source_id,
            title=c.title,
            score=c.score,
        )
        for i, c in enumerate(context.chunks)
    ]
    resp = QueryResponse(
        answer=result.answer,
        sources=sources,
        trace_id=trace_id,
        faithfulness_score=result.faithfulness_score,
        truncated=context.truncated,
    )
    return JSONResponse(content=resp.model_dump(), headers={"X-Trace-ID": trace_id})
