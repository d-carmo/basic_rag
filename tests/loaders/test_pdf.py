from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from rag.loaders.base import DocType
from rag.loaders.pdf import PdfLoader


def _mock_page(text: str) -> MagicMock:
    page: MagicMock = MagicMock()
    page.extract_text.return_value = text
    return page


def _mock_pdf_ctx(pages: list[Any]) -> MagicMock:
    pdf = MagicMock()
    pdf.__enter__ = MagicMock(return_value=pdf)
    pdf.__exit__ = MagicMock(return_value=False)
    pdf.pages = pages
    return pdf


@patch("rag.loaders.pdf.pdfplumber")
def test_one_doc_per_page(mock_pdfplumber: MagicMock, tmp_path: Path) -> None:
    (tmp_path / "doc.pdf").touch()
    mock_pdfplumber.open.return_value = _mock_pdf_ctx(
        [_mock_page("Page one"), _mock_page("Page two")]
    )
    docs = PdfLoader().load(tmp_path / "doc.pdf")
    assert len(docs) == 2
    assert docs[0].text == "Page one"
    assert docs[0].metadata.page == 1
    assert docs[1].metadata.page == 2
    assert docs[0].metadata.doc_type == DocType.PDF


@patch("rag.loaders.pdf.pdfplumber")
def test_empty_pages_are_skipped(mock_pdfplumber: MagicMock, tmp_path: Path) -> None:
    (tmp_path / "doc.pdf").touch()
    mock_pdfplumber.open.return_value = _mock_pdf_ctx(
        [_mock_page(""), _mock_page("  "), _mock_page("Real content")]
    )
    docs = PdfLoader(ocr_fallback=False).load(tmp_path / "doc.pdf")
    assert len(docs) == 1
    assert docs[0].text == "Real content"


@patch("rag.loaders.pdf.pdfplumber")
def test_all_empty_returns_empty_list(mock_pdfplumber: MagicMock, tmp_path: Path) -> None:
    (tmp_path / "doc.pdf").touch()
    mock_pdfplumber.open.return_value = _mock_pdf_ctx(
        [_mock_page(""), _mock_page("")]
    )
    docs = PdfLoader(ocr_fallback=False).load(tmp_path / "doc.pdf")
    assert docs == []
