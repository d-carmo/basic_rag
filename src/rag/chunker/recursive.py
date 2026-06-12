from __future__ import annotations

from rag.chunker.chunk import BaseChunker, Chunk, ChunkMetadata
from rag.chunker.token_counter import Tokenizer, word_count_tokenizer
from rag.loaders.base import Document

_DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]


class RecursiveCharacterChunker(BaseChunker):
    """
    Split text recursively on a priority-ordered list of separators.

    Tries paragraph breaks first, then line breaks, sentence endings, words,
    and finally individual characters as a last resort. Produces chunks of at
    most ``chunk_size`` tokens with ``overlap`` token overlap between neighbours.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        separators: list[str] | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._separators = separators if separators is not None else _DEFAULT_SEPARATORS
        self._tokenizer = tokenizer or word_count_tokenizer

    def chunk(self, document: Document) -> list[Chunk]:
        pieces = self._split_text(document.text, self._separators)
        # Exclude ChunkMetadata-specific fields so they don't conflict when the
        # document was itself produced by an earlier chunking pass.
        base_meta = document.metadata.model_dump(exclude={"chunk_index", "parent_id"})
        return [
            Chunk(
                text=piece,
                token_count=self._tokenizer(piece),
                metadata=ChunkMetadata(**base_meta, chunk_index=i),
            )
            for i, piece in enumerate(pieces)
        ]

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        if not text.strip():
            return []
        if self._tokenizer(text) <= self._chunk_size:
            return [text]

        sep, *rest = separators

        if not sep:
            return self._split_by_chars(text)

        raw = text.split(sep)
        # Re-attach the separator to preserve the original text when pieces are
        # merged back together (except the final piece which has no trailing sep).
        pieces = [p + sep for p in raw[:-1]]
        if raw[-1]:
            pieces.append(raw[-1])

        good: list[str] = []
        for piece in pieces:
            if not piece.strip():
                continue
            if self._tokenizer(piece) <= self._chunk_size:
                good.append(piece)
            elif rest:
                good.extend(self._split_text(piece, rest))
            else:
                good.append(piece)  # can't split further; keep as-is

        return self._merge_with_overlap(good)

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        """Merge small pieces into chunks, keeping ``overlap`` tokens of context."""
        chunks: list[str] = []
        buf: list[str] = []
        buf_tokens = 0

        for piece in pieces:
            piece_tokens = self._tokenizer(piece)
            if buf_tokens + piece_tokens > self._chunk_size and buf:
                chunks.append("".join(buf))
                # Trim the buffer front until we're within the overlap budget.
                while buf and buf_tokens > self._overlap:
                    removed = buf.pop(0)
                    buf_tokens -= self._tokenizer(removed)
            buf.append(piece)
            buf_tokens += piece_tokens

        if buf:
            chunks.append("".join(buf))

        return [c for c in chunks if c.strip()]

    def _split_by_chars(self, text: str) -> list[str]:
        """Character-level split when no separator successfully reduces a piece."""
        n_tokens = self._tokenizer(text)
        chars_per_token = max(1, len(text) // n_tokens)
        size = self._chunk_size * chars_per_token
        overlap_chars = self._overlap * chars_per_token

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            if end == len(text):
                break
            start = end - overlap_chars

        return chunks
