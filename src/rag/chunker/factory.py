from __future__ import annotations

from dataclasses import dataclass, field as dc_field

from rag.chunker.chunk import BaseChunker
from rag.chunker.recursive import RecursiveCharacterChunker
from rag.chunker.section import SectionChunker
from rag.chunker.token_counter import Tokenizer
from rag.loaders.base import DocType

# Default strategy per document type.  Can be overridden via ChunkerConfig.
_DOC_TYPE_STRATEGY: dict[DocType, str] = {
    DocType.MARKDOWN: "section",
    DocType.HTML: "section",
    DocType.PDF: "recursive",
    DocType.DOCX: "recursive",
    DocType.TEXT: "recursive",
    DocType.JSON: "recursive",
    DocType.CSV: "recursive",
    DocType.UNKNOWN: "recursive",
}


@dataclass
class ChunkerConfig:
    # Override the default strategy for the given doc_type.
    strategy: str = "auto"
    # Shared
    chunk_size: int = 512
    overlap: int = 64
    tokenizer: Tokenizer | None = None
    # SectionChunker
    section_chunk_size: int = 1024
    # SentenceChunker
    overlap_sentences: int = 1
    # SemanticChunker
    semantic_model: str = "BAAI/bge-small-en-v1.5"
    semantic_threshold: float = 0.75
    semantic_min_tokens: int = 32


def get_chunker(
    doc_type: DocType = DocType.UNKNOWN,
    config: ChunkerConfig | None = None,
) -> BaseChunker:
    """Return the appropriate chunker for *doc_type* according to *config*."""
    cfg = config or ChunkerConfig()

    strategy = (
        _DOC_TYPE_STRATEGY.get(doc_type, "recursive")
        if cfg.strategy == "auto"
        else cfg.strategy
    )

    if strategy == "section":
        return SectionChunker(
            chunk_size=cfg.section_chunk_size,
            overlap=cfg.overlap,
            tokenizer=cfg.tokenizer,
        )

    if strategy == "sentence":
        from rag.chunker.sentence import SentenceChunker

        return SentenceChunker(
            chunk_size=cfg.chunk_size,
            overlap_sentences=cfg.overlap_sentences,
            tokenizer=cfg.tokenizer,
        )

    if strategy == "semantic":
        from rag.chunker.semantic import SemanticChunker

        return SemanticChunker(
            model_name=cfg.semantic_model,
            breakpoint_threshold=cfg.semantic_threshold,
            min_chunk_tokens=cfg.semantic_min_tokens,
            tokenizer=cfg.tokenizer,
        )

    # Default / "recursive"
    return RecursiveCharacterChunker(
        chunk_size=cfg.chunk_size,
        overlap=cfg.overlap,
        tokenizer=cfg.tokenizer,
    )
