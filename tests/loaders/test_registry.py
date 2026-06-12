import pytest

from rag.loaders.csv_loader import CsvLoader
from rag.loaders.docx import DocxLoader
from rag.loaders.html import HtmlLoader
from rag.loaders.json_loader import JsonLoader
from rag.loaders.pdf import PdfLoader
from rag.loaders.registry import get_loader
from rag.loaders.text import TextLoader


@pytest.mark.parametrize(
    "filename,expected_type",
    [
        ("doc.pdf", PdfLoader),
        ("doc.docx", DocxLoader),
        ("doc.doc", DocxLoader),
        ("doc.txt", TextLoader),
        ("doc.md", TextLoader),
        ("doc.rst", TextLoader),
        ("page.html", HtmlLoader),
        ("page.htm", HtmlLoader),
        ("data.json", JsonLoader),
        ("data.jsonl", JsonLoader),
        ("data.ndjson", JsonLoader),
        ("data.csv", CsvLoader),
        ("data.tsv", CsvLoader),
    ],
)
def test_correct_loader_returned(filename: str, expected_type: type) -> None:
    assert isinstance(get_loader(filename), expected_type)


def test_unknown_extension_raises() -> None:
    with pytest.raises(ValueError, match="No loader registered"):
        get_loader("file.xyz")


def test_case_insensitive_extension() -> None:
    assert isinstance(get_loader("DOC.PDF"), PdfLoader)
    assert isinstance(get_loader("page.HTML"), HtmlLoader)
