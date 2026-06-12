import pytest

from rag.chunker.chunk import ChunkMetadata
from rag.chunker.parent_child import ParentChildChunker
from rag.chunker.recursive import RecursiveCharacterChunker
from rag.loaders.base import Document, DocumentMetadata


def _doc(text: str) -> Document:
    return Document(text=text, metadata=DocumentMetadata())


def _make_chunker(parent_size: int = 200, child_size: int = 50) -> ParentChildChunker:
    return ParentChildChunker(
        parent_chunker=RecursiveCharacterChunker(chunk_size=parent_size, overlap=0),
        child_chunker=RecursiveCharacterChunker(chunk_size=child_size, overlap=0),
    )


def test_returns_parents_and_children() -> None:
    text = "\n\n".join([f"Paragraph {i}. " * 10 for i in range(6)])
    chunks = _make_chunker().chunk(_doc(text))
    parents = [c for c in chunks if c.metadata.parent_id is None]
    children = [c for c in chunks if c.metadata.parent_id is not None]
    assert len(parents) >= 1
    assert len(children) >= 1


def test_children_reference_valid_parent_id() -> None:
    text = "\n\n".join([f"Paragraph {i}. " * 12 for i in range(8)])
    chunks = _make_chunker().chunk(_doc(text))
    parent_ids = {c.id for c in chunks if c.metadata.parent_id is None}
    for child in [c for c in chunks if c.metadata.parent_id is not None]:
        assert child.metadata.parent_id in parent_ids


def test_parent_appears_before_its_children() -> None:
    text = "\n\n".join([f"Para {i}. " * 15 for i in range(6)])
    chunks = _make_chunker().chunk(_doc(text))
    # Build index map: id → position
    id_to_pos = {c.id: i for i, c in enumerate(chunks)}
    for child in [c for c in chunks if c.metadata.parent_id is not None]:
        parent_pos = id_to_pos[child.metadata.parent_id]  # type: ignore[index]
        child_pos = id_to_pos[child.id]
        assert parent_pos < child_pos


def test_empty_document_returns_empty() -> None:
    assert _make_chunker().chunk(_doc("")) == []


def test_child_text_is_substring_of_parent() -> None:
    text = "\n\n".join([f"Sentence {i} has several words and more content." for i in range(20)])
    chunks = _make_chunker(parent_size=100, child_size=30).chunk(_doc(text))
    parent_map = {c.id: c.text for c in chunks if c.metadata.parent_id is None}
    for child in [c for c in chunks if c.metadata.parent_id is not None]:
        parent_text = parent_map[child.metadata.parent_id]  # type: ignore[index]
        # Child text must be contained within (or equal to) its parent's text
        assert child.text in parent_text or child.text.strip() in parent_text.strip()
