from datetime import datetime

from rag.loaders.base import DocType, Document, DocumentMetadata


def test_document_ids_are_unique() -> None:
    doc1 = Document(text="a", metadata=DocumentMetadata())
    doc2 = Document(text="b", metadata=DocumentMetadata())
    assert doc1.id != doc2.id


def test_document_metadata_defaults() -> None:
    meta = DocumentMetadata()
    assert meta.doc_type == DocType.UNKNOWN
    assert meta.source_url == ""
    assert meta.page is None
    assert meta.language is None
    assert meta.extra == {}
    assert isinstance(meta.created_at, datetime)


def test_doc_type_string_values() -> None:
    assert DocType.PDF == "pdf"
    assert DocType.MARKDOWN == "markdown"
    assert DocType.HTML == "html"


def test_document_text_stored() -> None:
    doc = Document(text="hello world", metadata=DocumentMetadata())
    assert doc.text == "hello world"


def test_document_metadata_page() -> None:
    meta = DocumentMetadata(source_url="file.pdf", page=3, doc_type=DocType.PDF)
    assert meta.page == 3
    assert meta.source_url == "file.pdf"
