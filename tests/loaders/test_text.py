from pathlib import Path

from rag.loaders.base import DocType
from rag.loaders.text import TextLoader


def test_load_txt(tmp_path: Path) -> None:
    f = tmp_path / "doc.txt"
    f.write_text("Hello world.\nSecond line.")
    docs = TextLoader().load(f)
    assert len(docs) == 1
    assert "Hello world" in docs[0].text
    assert docs[0].metadata.doc_type == DocType.TEXT


def test_load_markdown(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text("# Title\n\nSome content.")
    docs = TextLoader().load(f)
    assert len(docs) == 1
    assert docs[0].metadata.doc_type == DocType.MARKDOWN


def test_load_markdown_alt_extension(tmp_path: Path) -> None:
    f = tmp_path / "doc.markdown"
    f.write_text("content")
    docs = TextLoader().load(f)
    assert docs[0].metadata.doc_type == DocType.MARKDOWN


def test_empty_file_returns_empty(tmp_path: Path) -> None:
    f = tmp_path / "empty.txt"
    f.write_text("   \n  ")
    assert TextLoader().load(f) == []


def test_source_url_is_absolute(tmp_path: Path) -> None:
    f = tmp_path / "doc.txt"
    f.write_text("content")
    docs = TextLoader().load(f)
    assert docs[0].metadata.source_url == str(f.resolve())


def test_can_handle(tmp_path: Path) -> None:
    loader = TextLoader()
    assert loader.can_handle("file.txt")
    assert loader.can_handle("file.md")
    assert loader.can_handle("file.rst")
    assert not loader.can_handle("file.pdf")
    assert not loader.can_handle("file.csv")
