import pytest

from rag.chunker.section import SectionChunker, _split_markdown
from rag.loaders.base import DocType, Document, DocumentMetadata


def _doc(text: str, doc_type: DocType = DocType.TEXT) -> Document:
    return Document(text=text, metadata=DocumentMetadata(doc_type=doc_type))


# ── Markdown splitting ─────────────────────────────────────────────────────────

def test_split_markdown_on_headers() -> None:
    text = "# Title\n\nIntro text.\n\n## Section One\n\nContent one.\n\n## Section Two\n\nContent two."
    sections = _split_markdown(text)
    assert len(sections) == 3
    assert sections[0].startswith("# Title")
    assert sections[1].startswith("## Section One")
    assert sections[2].startswith("## Section Two")


def test_split_markdown_no_headers_returns_whole() -> None:
    text = "Just a plain paragraph with no headers."
    sections = _split_markdown(text)
    assert len(sections) == 1
    assert sections[0] == text


def test_markdown_chunker_produces_one_chunk_per_section() -> None:
    text = "# H1\n\nBody one.\n\n## H2\n\nBody two.\n\n### H3\n\nBody three."
    doc = _doc(text, DocType.MARKDOWN)
    chunks = SectionChunker(chunk_size=500).chunk(doc)
    assert len(chunks) == 3


def test_markdown_chunk_indices_sequential() -> None:
    text = "# A\n\ntext a.\n\n## B\n\ntext b.\n\n### C\n\ntext c."
    doc = _doc(text, DocType.MARKDOWN)
    chunks = SectionChunker(chunk_size=500).chunk(doc)
    for i, chunk in enumerate(chunks):
        assert chunk.metadata.chunk_index == i


# ── HTML splitting ─────────────────────────────────────────────────────────────

def test_html_splits_on_headings() -> None:
    html = "<h1>Title</h1><p>Intro.</p><h2>Sub</h2><p>Details.</p>"
    doc = _doc(html, DocType.HTML)
    chunks = SectionChunker(chunk_size=500).chunk(doc)
    assert len(chunks) == 2
    assert "Title" in chunks[0].text
    assert "Sub" in chunks[1].text


def test_html_no_headings_returns_single_chunk() -> None:
    html = "<p>Just a paragraph.</p><p>Another one.</p>"
    doc = _doc(html, DocType.HTML)
    chunks = SectionChunker(chunk_size=500).chunk(doc)
    assert len(chunks) == 1


# ── Overflow ───────────────────────────────────────────────────────────────────

def test_large_section_is_split_further() -> None:
    # A single section with 500 words should be split when chunk_size is small
    long_body = "word " * 500
    text = f"# Big Section\n\n{long_body}"
    doc = _doc(text, DocType.MARKDOWN)
    chunks = SectionChunker(chunk_size=50, overlap=5).chunk(doc)
    assert len(chunks) > 1


# ── Non-structural doc_type ────────────────────────────────────────────────────

def test_plain_text_returns_as_single_section() -> None:
    doc = _doc("Plain text without any headers.", DocType.TEXT)
    chunks = SectionChunker(chunk_size=500).chunk(doc)
    assert len(chunks) == 1
