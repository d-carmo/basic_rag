import hashlib
from pathlib import Path

from rag.ingestion.log import IngestionLog


def test_record_and_unchanged(tmp_path: Path) -> None:
    log = IngestionLog(db_path=tmp_path / "log.sqlite")
    log.record("doc1", "abc123", chunk_count=5)
    assert not log.is_changed("doc1", "abc123")


def test_different_hash_is_changed(tmp_path: Path) -> None:
    log = IngestionLog(db_path=tmp_path / "log.sqlite")
    log.record("doc1", "abc123")
    assert log.is_changed("doc1", "different")


def test_unknown_source_is_changed(tmp_path: Path) -> None:
    log = IngestionLog(db_path=tmp_path / "log.sqlite")
    assert log.is_changed("never_seen", "anyhash")


def test_remove_makes_it_changed_again(tmp_path: Path) -> None:
    log = IngestionLog(db_path=tmp_path / "log.sqlite")
    log.record("doc1", "abc123")
    log.remove("doc1")
    assert log.is_changed("doc1", "abc123")


def test_upsert_updates_existing(tmp_path: Path) -> None:
    log = IngestionLog(db_path=tmp_path / "log.sqlite")
    log.record("doc1", "oldhash", chunk_count=5)
    log.record("doc1", "newhash", chunk_count=8)
    assert not log.is_changed("doc1", "newhash")
    assert log.is_changed("doc1", "oldhash")


def test_compute_hash_matches_sha256(tmp_path: Path) -> None:
    log = IngestionLog(db_path=tmp_path / "log.sqlite")
    f = tmp_path / "file.txt"
    f.write_bytes(b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert log.compute_hash(f) == expected


def test_get_all(tmp_path: Path) -> None:
    log = IngestionLog(db_path=tmp_path / "log.sqlite")
    log.record("doc1", "hash1", chunk_count=3)
    log.record("doc2", "hash2", chunk_count=7)
    entries = log.get_all()
    assert len(entries) == 2
    ids = {e["source_id"] for e in entries}
    assert ids == {"doc1", "doc2"}


def test_chunk_count_stored(tmp_path: Path) -> None:
    log = IngestionLog(db_path=tmp_path / "log.sqlite")
    log.record("doc1", "h", chunk_count=42)
    entries = log.get_all()
    assert entries[0]["chunk_count"] == 42
