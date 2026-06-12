import json
from pathlib import Path

from rag.loaders.base import DocType
from rag.loaders.json_loader import JsonLoader


def test_load_json_object_with_text_field(tmp_path: Path) -> None:
    f = tmp_path / "doc.json"
    f.write_text(json.dumps({"title": "Test", "body": "Content here"}))
    docs = JsonLoader(text_field="body").load(f)
    assert len(docs) == 1
    assert docs[0].text == "Content here"
    assert docs[0].metadata.doc_type == DocType.JSON


def test_load_json_array(tmp_path: Path) -> None:
    f = tmp_path / "docs.json"
    f.write_text(json.dumps([{"text": "First"}, {"text": "Second"}]))
    docs = JsonLoader(text_field="text").load(f)
    assert len(docs) == 2
    assert docs[0].text == "First"
    assert docs[1].text == "Second"


def test_no_text_field_serialises_whole_object(tmp_path: Path) -> None:
    f = tmp_path / "doc.json"
    f.write_text(json.dumps({"a": 1, "b": "two"}))
    docs = JsonLoader().load(f)
    assert len(docs) == 1
    assert '"a"' in docs[0].text


def test_load_jsonl(tmp_path: Path) -> None:
    f = tmp_path / "docs.jsonl"
    lines = [json.dumps({"text": f"Line {i}"}) for i in range(3)]
    f.write_text("\n".join(lines))
    docs = JsonLoader(text_field="text").load(f)
    assert len(docs) == 3
    assert docs[0].metadata.extra["line"] == 1
    assert docs[2].metadata.extra["line"] == 3


def test_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    f = tmp_path / "docs.jsonl"
    f.write_text('{"text":"a"}\n\n{"text":"b"}\n')
    docs = JsonLoader(text_field="text").load(f)
    assert len(docs) == 2


def test_metadata_fields_extracted(tmp_path: Path) -> None:
    f = tmp_path / "doc.json"
    f.write_text(json.dumps({"text": "hello", "author": "Alice", "year": 2024}))
    docs = JsonLoader(text_field="text", metadata_fields=["author", "year"]).load(f)
    assert docs[0].metadata.extra["author"] == "Alice"
    assert docs[0].metadata.extra["year"] == 2024


def test_ndjson_extension(tmp_path: Path) -> None:
    f = tmp_path / "docs.ndjson"
    f.write_text(json.dumps({"text": "hello"}))
    docs = JsonLoader(text_field="text").load(f)
    assert len(docs) == 1
