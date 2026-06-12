from __future__ import annotations

import dataclasses
from typing import Any, Awaitable, Callable

from rag.assembler.base import ContextChunk


class ParentChunkFetcher:
    """
    Replace child chunks with their parent text when a `parent_id` is present.

    `fetch_fn` is an async callable that maps a list of point IDs to a dict of
    {id: payload_dict}. IDs missing from the result are left unchanged.
    """

    def __init__(
        self,
        fetch_fn: Callable[[list[str]], Awaitable[dict[str, dict[str, Any]]]],
    ) -> None:
        self._fetch = fetch_fn

    async def fetch(self, chunks: list[ContextChunk]) -> list[ContextChunk]:
        parent_ids = list(
            {c.payload["parent_id"] for c in chunks if c.payload.get("parent_id")}
        )
        if not parent_ids:
            return list(chunks)

        parent_payloads = await self._fetch(parent_ids)

        result: list[ContextChunk] = []
        for chunk in chunks:
            pid = chunk.payload.get("parent_id")
            if pid and pid in parent_payloads:
                pp = parent_payloads[pid]
                result.append(
                    dataclasses.replace(
                        chunk,
                        text=str(pp.get("text", chunk.text)),
                        payload={**chunk.payload, **pp},
                    )
                )
            else:
                result.append(chunk)
        return result
