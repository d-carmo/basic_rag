import json
from pathlib import Path
from typing import Any

from rag.loaders.base import BaseLoader, DocType, Document, DocumentMetadata

_JSONL_EXTENSIONS = frozenset({".jsonl", ".ndjson"})


class JsonLoader(BaseLoader):
    def __init__(
        self,
        text_field: str | None = None,
        metadata_fields: list[str] | None = None,
    ) -> None:
        self._text_field = text_field
        self._metadata_fields = metadata_fields or []

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".json"}) | _JSONL_EXTENSIONS

    def load(self, source: Path | str) -> list[Document]:
        path = Path(source)
        if path.suffix.lower() in _JSONL_EXTENSIONS:
            return self._load_jsonl(path)
        return self._load_json(path)

    def _load_json(self, path: Path) -> list[Document]:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
        source_url = str(path.resolve())
        if isinstance(data, list):
            return [self._record_to_doc(r, source_url) for r in data]
        return [self._record_to_doc(data, source_url)]

    def _load_jsonl(self, path: Path) -> list[Document]:
        docs: list[Document] = []
        source_url = str(path.resolve())
        for line_num, raw in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            raw = raw.strip()
            if not raw:
                continue
            record: dict[str, Any] = json.loads(raw)
            docs.append(self._record_to_doc(record, source_url, extra={"line": line_num}))
        return docs

    def _record_to_doc(
        self,
        record: dict[str, Any],
        source_url: str,
        extra: dict[str, Any] | None = None,
    ) -> Document:
        if self._text_field and self._text_field in record:
            text = str(record[self._text_field])
        else:
            text = json.dumps(record, ensure_ascii=False)

        combined: dict[str, Any] = dict(extra) if extra else {}
        combined.update({f: record[f] for f in self._metadata_fields if f in record})

        return Document(
            text=text,
            metadata=DocumentMetadata(
                source_url=source_url,
                doc_type=DocType.JSON,
                extra=combined,
            ),
        )
