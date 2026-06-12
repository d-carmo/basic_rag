from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile

from rag.api.deps import get_embedder, get_store
from rag.api.models import IngestResponse
from rag.chunker.chunk import Chunk, ChunkMetadata

router = APIRouter()

from rag.api.api_metrics import chunks_ingested_counter as _chunks_ingested

_SUPPORTED = {
    ".pdf": "pdf",
    ".txt": "text",
    ".text": "text",
    ".rst": "text",
    ".md": "markdown",
    ".markdown": "markdown",
}


def _load_file(path: Path) -> list[tuple[str, str, int | None]]:
    """Return list of (text, doc_type, page) tuples from a file path."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from rag.loaders.pdf import PdfLoader
        docs = PdfLoader().load(path)
    elif suffix in {".txt", ".text", ".rst", ".md", ".markdown"}:
        from rag.loaders.text import TextLoader
        docs = TextLoader().load(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix!r}")
    return [(d.text, d.metadata.doc_type.value, d.metadata.page) for d in docs]


@router.post("/v1/ingest/file", response_model=IngestResponse)
async def ingest_file(
    request: Request,
    file: UploadFile,
    source_id: str = Form(default=None),
    title: str | None = Form(default=None),
) -> IngestResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _SUPPORTED:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type {suffix!r}. Supported: {', '.join(sorted(_SUPPORTED))}",
        )

    sid = source_id or file.filename or "upload"
    store = get_store(request)
    embedder = get_embedder(request)

    content = await file.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        pages = await asyncio.to_thread(_load_file, tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if not pages:
        raise HTTPException(status_code=422, detail="No text could be extracted from the file.")

    from rag.loaders.base import DocType

    chunks: list[Chunk] = []
    for idx, (text, doc_type_val, page) in enumerate(pages):
        try:
            dt = DocType(doc_type_val)
        except ValueError:
            dt = DocType.UNKNOWN
        chunks.append(
            Chunk(
                text=text,
                token_count=len(text.split()),
                metadata=ChunkMetadata(
                    source_url=sid,
                    doc_type=dt,
                    chunk_index=idx,
                    extra={"title": title} if title else {},
                    page=page,
                ),
            )
        )

    texts = [c.text for c in chunks]
    vectors = await embedder.embed(texts)
    await store.upsert_chunks(chunks, dense_vectors=vectors)
    _chunks_ingested.labels(route="upload").inc(len(chunks))

    return IngestResponse(source_id=sid, status="ok", chunk_count=len(chunks))
