import pytest

from rag.chunker.factory import ChunkerConfig, get_chunker
from rag.chunker.recursive import RecursiveCharacterChunker
from rag.chunker.section import SectionChunker
from rag.chunker.sentence import SentenceChunker
from rag.loaders.base import DocType


def test_markdown_defaults_to_section_chunker() -> None:
    chunker = get_chunker(DocType.MARKDOWN)
    assert isinstance(chunker, SectionChunker)


def test_html_defaults_to_section_chunker() -> None:
    chunker = get_chunker(DocType.HTML)
    assert isinstance(chunker, SectionChunker)


def test_text_defaults_to_recursive_chunker() -> None:
    chunker = get_chunker(DocType.TEXT)
    assert isinstance(chunker, RecursiveCharacterChunker)


def test_pdf_defaults_to_recursive_chunker() -> None:
    chunker = get_chunker(DocType.PDF)
    assert isinstance(chunker, RecursiveCharacterChunker)


def test_unknown_defaults_to_recursive_chunker() -> None:
    chunker = get_chunker(DocType.UNKNOWN)
    assert isinstance(chunker, RecursiveCharacterChunker)


def test_no_args_returns_recursive_chunker() -> None:
    chunker = get_chunker()
    assert isinstance(chunker, RecursiveCharacterChunker)


def test_config_strategy_override_sentence() -> None:
    cfg = ChunkerConfig(strategy="sentence")
    chunker = get_chunker(DocType.MARKDOWN, config=cfg)
    assert isinstance(chunker, SentenceChunker)


def test_config_strategy_recursive_overrides_markdown_default() -> None:
    cfg = ChunkerConfig(strategy="recursive")
    chunker = get_chunker(DocType.MARKDOWN, config=cfg)
    assert isinstance(chunker, RecursiveCharacterChunker)


def test_config_chunk_size_passed_through() -> None:
    cfg = ChunkerConfig(chunk_size=128, overlap=16)
    chunker = get_chunker(DocType.TEXT, config=cfg)
    assert isinstance(chunker, RecursiveCharacterChunker)
    assert chunker._chunk_size == 128
    assert chunker._overlap == 16


def test_config_section_chunk_size_passed_through() -> None:
    cfg = ChunkerConfig(section_chunk_size=2048)
    chunker = get_chunker(DocType.MARKDOWN, config=cfg)
    assert isinstance(chunker, SectionChunker)
    assert chunker._chunk_size == 2048
