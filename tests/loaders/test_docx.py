from pathlib import Path
from unittest.mock import MagicMock, patch

from rag.loaders.base import DocType
from rag.loaders.docx import DocxLoader


def _para(text: str) -> MagicMock:
    p: MagicMock = MagicMock()
    p.text = text
    return p


def _cell(text: str) -> MagicMock:
    c: MagicMock = MagicMock()
    c.text = text
    return c


@patch("rag.loaders.docx.python_docx")
def test_paragraphs_joined(mock_docx: MagicMock, tmp_path: Path) -> None:
    (tmp_path / "doc.docx").touch()
    mock_doc = MagicMock()
    mock_doc.paragraphs = [_para("First paragraph"), _para("Second paragraph")]
    mock_doc.tables = []
    mock_docx.Document.return_value = mock_doc

    docs = DocxLoader().load(tmp_path / "doc.docx")
    assert len(docs) == 1
    assert "First paragraph" in docs[0].text
    assert "Second paragraph" in docs[0].text
    assert docs[0].metadata.doc_type == DocType.DOCX


@patch("rag.loaders.docx.python_docx")
def test_table_cells_appended(mock_docx: MagicMock, tmp_path: Path) -> None:
    (tmp_path / "doc.docx").touch()
    row = MagicMock()
    row.cells = [_cell("Col1"), _cell("Col2")]
    table = MagicMock()
    table.rows = [row]
    mock_doc = MagicMock()
    mock_doc.paragraphs = [_para("Text")]
    mock_doc.tables = [table]
    mock_docx.Document.return_value = mock_doc

    docs = DocxLoader().load(tmp_path / "doc.docx")
    assert "Col1 | Col2" in docs[0].text


@patch("rag.loaders.docx.python_docx")
def test_empty_docx_returns_empty(mock_docx: MagicMock, tmp_path: Path) -> None:
    (tmp_path / "doc.docx").touch()
    mock_doc = MagicMock()
    mock_doc.paragraphs = [_para(""), _para("  ")]
    mock_doc.tables = []
    mock_docx.Document.return_value = mock_doc

    assert DocxLoader().load(tmp_path / "doc.docx") == []
