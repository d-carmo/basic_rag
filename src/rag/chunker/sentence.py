from __future__ import annotations

import re

from rag.chunker.chunk import BaseChunker, Chunk, ChunkMetadata
from rag.chunker.token_counter import Tokenizer, word_count_tokenizer
from rag.loaders.base import Document

# Splits on sentence-ending punctuation followed by whitespace, but skips
# common abbreviations like "Mr." or "U.S." via negative lookbehinds.
_SENT_BOUNDARY = re.compile(
    r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.!?])\s+"
)


def _split_sentences(text: str) -> list[str]:
    parts = _SENT_BOUNDARY.split(text)
    return [p.strip() for p in parts if p.strip()]


class SentenceChunker(BaseChunker):
    """
    Group sentences into chunks of at most ``chunk_size`` tokens.

    ``overlap_sentences`` trailing sentences from the previous chunk are
    prepended to the next one to provide context continuity.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap_sentences: int = 1,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self._chunk_size = chunk_size
        self._overlap_sentences = overlap_sentences
        self._tokenizer = tokenizer or word_count_tokenizer

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = _split_sentences(document.text)
        if not sentences:
            return []

        base_meta = document.metadata.model_dump(exclude={"chunk_index", "parent_id"})
        chunks: list[Chunk] = []
        buf: list[str] = []
        buf_tokens = 0
        chunk_index = 0

        for sent in sentences:
            sent_tokens = self._tokenizer(sent)
            if buf_tokens + sent_tokens > self._chunk_size and buf:
                text = " ".join(buf)
                chunks.append(
                    Chunk(
                        text=text,
                        token_count=self._tokenizer(text),
                        metadata=ChunkMetadata(**base_meta, chunk_index=chunk_index),
                    )
                )
                chunk_index += 1
                buf = buf[-self._overlap_sentences :] if self._overlap_sentences else []
                buf_tokens = sum(self._tokenizer(s) for s in buf)

            buf.append(sent)
            buf_tokens += sent_tokens

        if buf:
            text = " ".join(buf)
            chunks.append(
                Chunk(
                    text=text,
                    token_count=self._tokenizer(text),
                    metadata=ChunkMetadata(**base_meta, chunk_index=chunk_index),
                )
            )

        return chunks
