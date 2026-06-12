from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rag.chunker.chunk import Chunk
from rag.embedder.base import SparseVector
from rag.vector_store.schema import ChunkPayload


@dataclass
class StoreConfig:
    url: str = "http://localhost:6333"
    api_key: str | None = None
    collection_name: str = "rag"
    schema_version: int = 1
    dense_dims: int = 1024
    upsert_batch_size: int = 100
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"


@dataclass
class SearchHit:
    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


class QdrantStore:
    def __init__(
        self,
        config: StoreConfig | None = None,
        client: Any = None,
    ) -> None:
        self._config = config or StoreConfig()
        self._client: Any = client

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    def collection_name(self) -> str:
        return f"{self._config.collection_name}_v{self._config.schema_version}"

    def collection_name_for_version(self, version: int) -> str:
        return f"{self._config.collection_name}_v{version}"

    # ── client lifecycle ──────────────────────────────────────────────────────

    def _get_client(self) -> Any:
        if self._client is None:
            from qdrant_client import AsyncQdrantClient  # type: ignore[import-not-found]

            self._client = AsyncQdrantClient(
                url=self._config.url,
                api_key=self._config.api_key,
            )
        return self._client

    # ── collection management ─────────────────────────────────────────────────

    async def collection_exists(self) -> bool:
        return bool(await self._get_client().collection_exists(self.collection_name))

    async def create_collection(self) -> None:
        from qdrant_client.models import Distance, SparseVectorParams, VectorParams  # type: ignore[import-not-found]

        await self._get_client().create_collection(
            collection_name=self.collection_name,
            vectors_config={
                self._config.dense_vector_name: VectorParams(
                    size=self._config.dense_dims,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                self._config.sparse_vector_name: SparseVectorParams()
            },
        )

    async def get_or_create_collection(self) -> None:
        if not await self.collection_exists():
            await self.create_collection()

    async def create_payload_indexes(self) -> None:
        from qdrant_client.models import PayloadSchemaType  # type: ignore[import-not-found]

        client = self._get_client()
        name = self.collection_name
        for field_name in ("source_id", "doc_type", "language"):
            await client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        await client.create_payload_index(
            collection_name=name,
            field_name="chunk_index",
            field_schema=PayloadSchemaType.INTEGER,
        )
        await client.create_payload_index(
            collection_name=name,
            field_name="created_at",
            field_schema=PayloadSchemaType.DATETIME,
        )

    # ── write ─────────────────────────────────────────────────────────────────

    async def _upsert_batch(self, points: list[Any]) -> None:
        """Upsert with exponential-backoff retry (3 attempts)."""
        for attempt in range(3):
            try:
                await self._get_client().upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,
                )
                return
            except Exception:
                if attempt == 2:
                    raise
                await asyncio.sleep(2.0 ** attempt)

    async def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        dense_vectors: list[list[float]],
        sparse_vectors: list[SparseVector] | None = None,
    ) -> None:
        from qdrant_client.models import PointStruct  # type: ignore[import-not-found]
        from qdrant_client.models import SparseVector as QSparse  # type: ignore[import-not-found]

        points: list[Any] = []
        for i, chunk in enumerate(chunks):
            payload = ChunkPayload.from_chunk(chunk)
            vector_dict: dict[str, Any] = {
                self._config.dense_vector_name: dense_vectors[i],
            }
            if sparse_vectors is not None:
                sv = sparse_vectors[i]
                vector_dict[self._config.sparse_vector_name] = QSparse(
                    indices=sv.indices, values=sv.values
                )
            points.append(
                PointStruct(id=chunk.id, vector=vector_dict, payload=payload.model_dump())
            )

        batch = self._config.upsert_batch_size
        for i in range(0, len(points), batch):
            await self._upsert_batch(points[i : i + batch])

    async def delete_by_source(self, source_id: str) -> None:
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue  # type: ignore[import-not-found]

        await self._get_client().delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))]
                )
            ),
        )

    # ── read ──────────────────────────────────────────────────────────────────

    async def search(
        self,
        dense_query: list[float] | None = None,
        sparse_query: SparseVector | None = None,
        filter_: Any | None = None,
        top_k: int = 8,
        candidates: int = 20,
    ) -> list[SearchHit]:
        if dense_query is None and sparse_query is None:
            raise ValueError("At least one of dense_query or sparse_query must be provided")

        client = self._get_client()
        name = self.collection_name

        if dense_query is not None and sparse_query is not None:
            from qdrant_client.models import Fusion, FusionQuery, Prefetch  # type: ignore[import-not-found]
            from qdrant_client.models import SparseVector as QSparse  # type: ignore[import-not-found]

            response = await client.query_points(
                collection_name=name,
                prefetch=[
                    Prefetch(
                        query=dense_query,
                        using=self._config.dense_vector_name,
                        limit=candidates,
                    ),
                    Prefetch(
                        query=QSparse(indices=sparse_query.indices, values=sparse_query.values),
                        using=self._config.sparse_vector_name,
                        limit=candidates,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_payload=True,
                query_filter=filter_,
            )
            scored = response.points

        elif dense_query is not None:
            response = await client.query_points(
                collection_name=name,
                query=dense_query,
                using=self._config.dense_vector_name,
                limit=top_k,
                with_payload=True,
                query_filter=filter_,
            )
            scored = response.points

        else:  # sparse only
            from qdrant_client.models import SparseVector as QSparse  # type: ignore[import-not-found]

            response = await client.query_points(
                collection_name=name,
                query=QSparse(
                    indices=sparse_query.indices,  # type: ignore[union-attr]
                    values=sparse_query.values,
                ),
                using=self._config.sparse_vector_name,
                limit=top_k,
                with_payload=True,
                query_filter=filter_,
            )
            scored = response.points

        return [
            SearchHit(id=str(p.id), score=p.score, payload=dict(p.payload or {}))
            for p in scored
        ]

    async def list_sources(self) -> list[dict[str, Any]]:
        """Scroll the collection and return unique source_ids with chunk counts."""
        client = self._get_client()
        counts: dict[str, int] = {}
        offset = None
        while True:
            records, next_offset = await client.scroll(
                collection_name=self.collection_name,
                with_payload=["source_id"],
                limit=1000,
                offset=offset,
            )
            for point in records:
                sid = (point.payload or {}).get("source_id", "")
                if sid:
                    counts[sid] = counts.get(sid, 0) + 1
            if next_offset is None:
                break
            offset = next_offset
        return [{"source_id": sid, "chunk_count": n} for sid, n in sorted(counts.items())]

    async def get_by_ids(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        """Retrieve payload dicts keyed by point ID; missing IDs are omitted."""
        client = self._get_client()
        records = await client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
            with_payload=True,
        )
        return {str(r.id): dict(r.payload or {}) for r in records}

    # ── snapshot ──────────────────────────────────────────────────────────────

    async def snapshot(self) -> str:
        result = await self._get_client().create_snapshot(
            collection_name=self.collection_name
        )
        return getattr(result, "name", str(result))
