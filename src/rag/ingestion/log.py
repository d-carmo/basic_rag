import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class IngestionLog:
    """Tracks ingested documents by source_id + SHA-256 hash to enable skip-on-unchanged."""

    def __init__(self, db_path: str | Path = "ingestion_log.sqlite") -> None:
        self._db_path = str(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_log (
                    source_id   TEXT PRIMARY KEY,
                    hash        TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    ingested_at TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def compute_hash(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65_536), b""):
                h.update(chunk)
        return h.hexdigest()

    def is_changed(self, source_id: str, current_hash: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT hash FROM ingestion_log WHERE source_id = ?", (source_id,)
            ).fetchone()
        return row is None or row[0] != current_hash

    def record(self, source_id: str, file_hash: str, chunk_count: int = 0) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_log (source_id, hash, chunk_count, ingested_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    hash        = excluded.hash,
                    chunk_count = excluded.chunk_count,
                    ingested_at = excluded.ingested_at
                """,
                (source_id, file_hash, chunk_count, datetime.now(timezone.utc).isoformat()),
            )

    def remove(self, source_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM ingestion_log WHERE source_id = ?", (source_id,)
            )

    def get_all(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT source_id, hash, chunk_count, ingested_at FROM ingestion_log"
            ).fetchall()
        return [
            {
                "source_id": row[0],
                "hash": row[1],
                "chunk_count": row[2],
                "ingested_at": row[3],
            }
            for row in rows
        ]
