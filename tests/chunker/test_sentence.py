import pytest

from rag.chunker.sentence import SentenceChunker, _split_sentences
from rag.loaders.base import Document, DocumentMetadata


def _doc(text: str) -> Document:
    return Document(text=text, metadata=DocumentMetadata())


# ── Sentence splitter ──────────────────────────────────────────────────────────

def test_split_sentences_basic() -> None:
    sents = _split_sentences("Hello world. How are you? Fine!")
    assert len(sents) == 3


def test_split_sentences_empty() -> None:
    assert _split_sentences("") == []


def test_split_sentences_single() -> None:
    assert _split_sentences("Just one sentence") == ["Just one sentence"]


# ── SentenceChunker ────────────────────────────────────────────────────────────

def test_short_text_single_chunk() -> None:
    doc = _doc("Hello world. How are you?")
    chunks = SentenceChunker(chunk_size=200).chunk(doc)
    assert len(chunks) == 1


def test_empty_returns_empty() -> None:
    assert SentenceChunker().chunk(_doc("")) == []


def test_multiple_chunks_produced() -> None:
    # 30 sentences, small chunk size → must split
    text = " ".join([f"This is sentence number {i}." for i in range(30)])
    chunks = SentenceChunker(chunk_size=20, overlap_sentences=1).chunk(_doc(text))
    assert len(chunks) > 1


def test_chunk_indices_sequential() -> None:
    text = " ".join([f"Sentence {i}." for i in range(40)])
    chunks = SentenceChunker(chunk_size=15, overlap_sentences=0).chunk(_doc(text))
    for i, chunk in enumerate(chunks):
        assert chunk.metadata.chunk_index == i


def test_overlap_sentence_appears_in_next_chunk() -> None:
    sentences = [f"Sentence number {i} has multiple words." for i in range(20)]
    text = " ".join(sentences)
    chunks = SentenceChunker(chunk_size=25, overlap_sentences=1).chunk(_doc(text))
    if len(chunks) < 2:
        pytest.skip("Not enough chunks produced")
    # The last sentence of chunk[0] should appear in chunk[1]
    last_sent_of_first = chunks[0].text.split(".")[-2].strip() + "."
    assert last_sent_of_first in chunks[1].text


def test_no_overlap_no_repeat() -> None:
    sentences = [f"Sentence {i}." for i in range(20)]
    text = " ".join(sentences)
    chunks = SentenceChunker(chunk_size=15, overlap_sentences=0).chunk(_doc(text))
    if len(chunks) < 2:
        pytest.skip("Not enough chunks")
    # Concatenating all chunks should not be longer than text * 1.1 (no overlap inflation)
    total = sum(len(c.text) for c in chunks)
    assert total <= len(text) * 1.1


def test_token_count_populated() -> None:
    doc = _doc("Hello world. Second sentence here.")
    chunks = SentenceChunker(chunk_size=200).chunk(doc)
    assert all(c.token_count > 0 for c in chunks)
