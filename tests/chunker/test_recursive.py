import pytest

from rag.chunker.chunk import ChunkMetadata
from rag.chunker.recursive import RecursiveCharacterChunker
from rag.chunker.token_counter import word_count_tokenizer
from rag.loaders.base import Document, DocumentMetadata


def _doc(text: str) -> Document:
    return Document(text=text, metadata=DocumentMetadata())


def _chunker(**kw: object) -> RecursiveCharacterChunker:
    return RecursiveCharacterChunker(**kw)  # type: ignore[arg-type]


# ── Basic behaviour ────────────────────────────────────────────────────────────

def test_short_text_returns_single_chunk() -> None:
    doc = _doc("Hello world.")
    chunks = _chunker(chunk_size=100).chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."


def test_empty_text_returns_empty() -> None:
    chunks = _chunker().chunk(_doc(""))
    assert chunks == []


def test_whitespace_only_returns_empty() -> None:
    chunks = _chunker().chunk(_doc("   \n\n   "))
    assert chunks == []


def test_long_text_splits_into_multiple_chunks() -> None:
    # Build text with clear paragraph breaks
    paragraphs = [f"Paragraph {i}. " * 20 for i in range(10)]
    text = "\n\n".join(paragraphs)
    chunks = _chunker(chunk_size=50, overlap=5).chunk(_doc(text))
    assert len(chunks) > 1


def test_chunk_token_count_set() -> None:
    doc = _doc("This is a test sentence with several words in it.")
    chunks = _chunker(chunk_size=200).chunk(doc)
    for chunk in chunks:
        assert chunk.token_count > 0


def test_chunk_indices_are_sequential() -> None:
    paragraphs = "\n\n".join(["word " * 30 for _ in range(6)])
    chunks = _chunker(chunk_size=20, overlap=2).chunk(_doc(paragraphs))
    for i, chunk in enumerate(chunks):
        assert chunk.metadata.chunk_index == i


def test_metadata_propagated() -> None:
    meta = DocumentMetadata(source_url="file.txt")
    doc = Document(text="hello world", metadata=meta)
    chunks = _chunker(chunk_size=100).chunk(doc)
    assert chunks[0].metadata.source_url == "file.txt"


# ── Overlap ────────────────────────────────────────────────────────────────────

def test_overlap_shares_content_between_adjacent_chunks() -> None:
    # Create text that will produce at least 2 chunks with overlap
    sentences = ". ".join([f"Sentence number {i} has some words" for i in range(30)])
    chunks = _chunker(chunk_size=30, overlap=10).chunk(_doc(sentences))
    if len(chunks) < 2:
        pytest.skip("Text too short to produce multiple chunks with these settings")
    # Last token(s) of chunk[0] should appear at the start of chunk[1]
    tail = chunks[0].text[-20:]
    assert tail in chunks[1].text or chunks[1].text[:20] in chunks[0].text


# ── Custom separators ──────────────────────────────────────────────────────────

def test_custom_separator() -> None:
    # Use a char-count tokenizer so each "AAA|" piece (4 chars) exceeds chunk_size=3.
    char_tok = len
    text = "AAA|BBB|CCC|DDD|EEE"
    chunks = _chunker(chunk_size=3, overlap=0, separators=["|", ""], tokenizer=char_tok).chunk(
        _doc(text)
    )
    assert len(chunks) > 1


# ── No chunk exceeds chunk_size by more than one piece ────────────────────────

def test_chunks_roughly_within_size() -> None:
    text = " ".join([f"word{i}" for i in range(500)])
    chunk_size = 40
    chunks = _chunker(chunk_size=chunk_size, overlap=5).chunk(_doc(text))
    for chunk in chunks:
        # Allow a small margin because the last merge step may carry one extra piece
        assert chunk.token_count <= chunk_size * 2
