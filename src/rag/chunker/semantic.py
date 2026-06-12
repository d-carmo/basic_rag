from __future__ import annotations

import math
import re
from typing import Any

from rag.chunker.chunk import BaseChunker, Chunk, ChunkMetadata
from rag.chunker.token_counter import Tokenizer, word_count_tokenizer
from rag.loaders.base import Document

_SENT_BOUNDARY = re.compile(
    r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.!?])\s+"
)


def _split_sentences(text: str) -> list[str]:
    parts = _SENT_BOUNDARY.split(text)
    return [p.strip() for p in parts if p.strip()]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticChunker(BaseChunker):
    """
    Split text at points where consecutive sentence embeddings diverge sharply.

    Uses a sentence-transformer model to embed each sentence, then inserts a
    chunk boundary wherever the cosine similarity between neighbours drops below
    ``breakpoint_threshold``.  Sections smaller than ``min_chunk_tokens`` are
    merged into the next chunk rather than emitted standalone.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        breakpoint_threshold: float = 0.75,
        min_chunk_tokens: int = 32,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self._model_name = model_name
        self._threshold = breakpoint_threshold
        self._min_tokens = min_chunk_tokens
        self._tokenizer = tokenizer or word_count_tokenizer
        self._model: Any = None

    # ------------------------------------------------------------------ #

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        vectors = model.encode(texts, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    # ------------------------------------------------------------------ #

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = _split_sentences(document.text)
        if not sentences:
            return []

        if len(sentences) == 1:
            text = sentences[0]
            return [
                Chunk(
                    text=text,
                    token_count=self._tokenizer(text),
                    metadata=ChunkMetadata(**document.metadata.model_dump(), chunk_index=0),
                )
            ]

        embeddings = self._embed(sentences)
        similarities = [
            _cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]

        # A breakpoint sits *before* sentence[i+1] when similarity[i] < threshold.
        breakpoints: set[int] = {
            i + 1 for i, sim in enumerate(similarities) if sim < self._threshold
        }

        base_meta = document.metadata.model_dump(exclude={"chunk_index", "parent_id"})
        chunks: list[Chunk] = []
        chunk_index = 0
        current: list[str] = []

        for i, sent in enumerate(sentences):
            if i in breakpoints and current:
                candidate = " ".join(current)
                if self._tokenizer(candidate) >= self._min_tokens:
                    chunks.append(
                        Chunk(
                            text=candidate,
                            token_count=self._tokenizer(candidate),
                            metadata=ChunkMetadata(**base_meta, chunk_index=chunk_index),
                        )
                    )
                    chunk_index += 1
                    current = []
            current.append(sent)

        if current:
            text = " ".join(current)
            chunks.append(
                Chunk(
                    text=text,
                    token_count=self._tokenizer(text),
                    metadata=ChunkMetadata(**base_meta, chunk_index=chunk_index),
                )
            )

        return chunks
