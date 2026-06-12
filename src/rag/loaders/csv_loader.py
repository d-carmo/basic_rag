import csv
from pathlib import Path
from typing import Any

from rag.loaders.base import BaseLoader, DocType, Document, DocumentMetadata


class CsvLoader(BaseLoader):
    def __init__(
        self,
        text_column: str | None = None,
        delimiter: str | None = None,
        metadata_columns: list[str] | None = None,
    ) -> None:
        self._text_column = text_column
        self._delimiter = delimiter
        self._metadata_columns = metadata_columns or []

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".csv", ".tsv"})

    def load(self, source: Path | str) -> list[Document]:
        path = Path(source)
        delimiter = self._delimiter or ("\t" if path.suffix.lower() == ".tsv" else ",")
        source_url = str(path.resolve())

        docs: list[Document] = []
        with path.open(encoding="utf-8", newline="", errors="replace") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row_num, row in enumerate(reader, start=2):  # row 1 is the header
                text = self._row_to_text(row)
                if not text.strip():
                    continue

                extra: dict[str, Any] = {"row": row_num}
                extra.update({c: row[c] for c in self._metadata_columns if c in row})

                docs.append(
                    Document(
                        text=text,
                        metadata=DocumentMetadata(
                            source_url=source_url,
                            doc_type=DocType.CSV,
                            extra=extra,
                        ),
                    )
                )

        return docs

    def _row_to_text(self, row: dict[str, str]) -> str:
        if self._text_column and self._text_column in row:
            return row[self._text_column]
        return " | ".join(f"{k}: {v}" for k, v in row.items() if v.strip())
