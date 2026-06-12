from pathlib import Path

from rag.loaders.base import DocType
from rag.loaders.csv_loader import CsvLoader


def test_load_csv_all_columns(tmp_path: Path) -> None:
    f = tmp_path / "data.csv"
    f.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\n")
    docs = CsvLoader().load(f)
    assert len(docs) == 2
    assert "Alice" in docs[0].text
    assert docs[0].metadata.doc_type == DocType.CSV


def test_load_csv_text_column(tmp_path: Path) -> None:
    f = tmp_path / "data.csv"
    f.write_text("id,content\n1,First document\n2,Second document\n")
    docs = CsvLoader(text_column="content").load(f)
    assert docs[0].text == "First document"
    assert docs[1].text == "Second document"


def test_load_tsv_auto_delimiter(tmp_path: Path) -> None:
    f = tmp_path / "data.tsv"
    f.write_text("col1\tcol2\nval1\tval2\n")
    docs = CsvLoader().load(f)
    assert len(docs) == 1
    assert "val1" in docs[0].text


def test_explicit_delimiter(tmp_path: Path) -> None:
    f = tmp_path / "data.txt"
    f.write_text("col1|col2\nval1|val2\n")
    docs = CsvLoader(delimiter="|").load(f)
    assert len(docs) == 1


def test_skip_empty_rows(tmp_path: Path) -> None:
    f = tmp_path / "data.csv"
    f.write_text("text\nhello\n\nworld\n")
    docs = CsvLoader(text_column="text").load(f)
    assert len(docs) == 2


def test_row_number_in_extra(tmp_path: Path) -> None:
    f = tmp_path / "data.csv"
    f.write_text("text\nfirst\nsecond\n")
    docs = CsvLoader(text_column="text").load(f)
    assert docs[0].metadata.extra["row"] == 2
    assert docs[1].metadata.extra["row"] == 3


def test_metadata_columns(tmp_path: Path) -> None:
    f = tmp_path / "data.csv"
    f.write_text("text,source\nhello,web\nworld,pdf\n")
    docs = CsvLoader(text_column="text", metadata_columns=["source"]).load(f)
    assert docs[0].metadata.extra["source"] == "web"
    assert docs[1].metadata.extra["source"] == "pdf"
