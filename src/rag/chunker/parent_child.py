from __future__ import annotations

from rag.chunker.chunk import BaseChunker, Chunk, ChunkMetadata
from rag.loaders.base import Document


class ParentChildChunker(BaseChunker):
    """
    Two-pass chunker that produces large parent chunks and small child chunks.

    Returns both in a flat list.  Children carry ``parent_id`` pointing to their
    parent's ``id``, allowing the vector store to retrieve parent context after
    finding a relevant child via similarity search.

    Callers can separate parents from children by checking ``chunk.metadata.parent_id``:
    - ``None``  → parent chunk (store in payload, skip indexing or index separately)
    - non-None  → child chunk (index for retrieval)
    """

    def __init__(
        self,
        parent_chunker: BaseChunker,
        child_chunker: BaseChunker,
    ) -> None:
        self._parent_chunker = parent_chunker
        self._child_chunker = child_chunker

    def chunk(self, document: Document) -> list[Chunk]:
        parents = self._parent_chunker.chunk(document)
        result: list[Chunk] = []

        for parent in parents:
            result.append(parent)

            parent_doc = Document(text=parent.text, metadata=parent.metadata)
            children = self._child_chunker.chunk(parent_doc)

            for child in children:
                child_meta_data = child.metadata.model_dump()
                child_meta_data["parent_id"] = parent.id
                result.append(
                    Chunk(
                        id=child.id,
                        text=child.text,
                        token_count=child.token_count,
                        metadata=ChunkMetadata(**child_meta_data),
                    )
                )

        return result
