from __future__ import annotations

import re

from rag.chunker.chunk import BaseChunker, Chunk, ChunkMetadata
from rag.chunker.token_counter import Tokenizer, word_count_tokenizer
from rag.loaders.base import DocType, Document

# Split just before a Markdown heading line (# / ## / ### / ####).
_MD_HEADER_RE = re.compile(r"(?=^#{1,4}\s)", re.MULTILINE)

_HTML_HEADING_TAG = re.compile(r"^h[1-4]$", re.IGNORECASE)


def _split_markdown(text: str) -> list[str]:
    sections = _MD_HEADER_RE.split(text)
    return [s.strip() for s in sections if s.strip()]


def _split_html(text: str) -> list[str]:
    try:
        from bs4 import BeautifulSoup  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("beautifulsoup4 is required for HTML section splitting") from exc

    soup = BeautifulSoup(text, "html.parser")
    headings = soup.find_all(_HTML_HEADING_TAG)

    if not headings:
        body = soup.get_text(separator=" ", strip=True)
        return [body] if body else []

    sections: list[str] = []
    for heading in headings:
        buf: list[str] = [heading.get_text(strip=True)]
        for sibling in heading.next_siblings:
            if hasattr(sibling, "name") and _HTML_HEADING_TAG.match(sibling.name or ""):
                break
            if hasattr(sibling, "get_text"):
                part = sibling.get_text(separator=" ", strip=True)  # type: ignore[union-attr]
            else:
                part = str(sibling).strip()
            if part:
                buf.append(part)
        section_text = " ".join(buf).strip()
        if section_text:
            sections.append(section_text)

    return sections if sections else [soup.get_text(separator=" ", strip=True)]


class SectionChunker(BaseChunker):
    """
    Split on document structure: Markdown ``#`` headers or HTML ``<h1>``–``<h4>`` tags.

    Sections that still exceed ``chunk_size`` tokens are further split by a
    :class:`~rag.chunker.recursive.RecursiveCharacterChunker` fallback.
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        overlap: int = 64,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self._chunk_size = chunk_size
        self._overflow_chunk_size = chunk_size
        self._overflow_overlap = overlap
        self._tokenizer = tokenizer or word_count_tokenizer
        # Deferred to avoid circular import at module level.
        self._overflow_chunker: BaseChunker | None = None

    def _get_overflow_chunker(self) -> BaseChunker:
        if self._overflow_chunker is None:
            from rag.chunker.recursive import RecursiveCharacterChunker

            self._overflow_chunker = RecursiveCharacterChunker(
                chunk_size=self._overflow_chunk_size,
                overlap=self._overflow_overlap,
                tokenizer=self._tokenizer,
            )
        return self._overflow_chunker

    def chunk(self, document: Document) -> list[Chunk]:
        doc_type = document.metadata.doc_type
        if doc_type == DocType.MARKDOWN:
            raw_sections = _split_markdown(document.text)
        elif doc_type == DocType.HTML:
            raw_sections = _split_html(document.text)
        else:
            raw_sections = [document.text]

        base_meta = document.metadata.model_dump(exclude={"chunk_index", "parent_id"})
        chunks: list[Chunk] = []
        chunk_index = 0

        for section in raw_sections:
            if not section.strip():
                continue
            section_tokens = self._tokenizer(section)
            if section_tokens <= self._chunk_size:
                chunks.append(
                    Chunk(
                        text=section,
                        token_count=section_tokens,
                        metadata=ChunkMetadata(**base_meta, chunk_index=chunk_index),
                    )
                )
                chunk_index += 1
            else:
                section_doc = Document(text=section, metadata=document.metadata)
                for sub in self._get_overflow_chunker().chunk(section_doc):
                    chunks.append(
                        Chunk(
                            text=sub.text,
                            token_count=sub.token_count,
                            metadata=ChunkMetadata(**base_meta, chunk_index=chunk_index),
                        )
                    )
                    chunk_index += 1

        return chunks
