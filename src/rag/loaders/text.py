from pathlib import Path

from rag.loaders.base import BaseLoader, DocType, Document, DocumentMetadata


class TextLoader(BaseLoader):
    _MARKDOWN = frozenset({".md", ".markdown"})
    _TEXT = frozenset({".txt", ".text", ".rst"})

    @property
    def supported_extensions(self) -> frozenset[str]:
        return self._MARKDOWN | self._TEXT

    def load(self, source: Path | str) -> list[Document]:
        path = Path(source)
        text = path.read_text(encoding="utf-8", errors="replace")

        if not text.strip():
            return []

        doc_type = DocType.MARKDOWN if path.suffix.lower() in self._MARKDOWN else DocType.TEXT

        return [
            Document(
                text=text,
                metadata=DocumentMetadata(
                    source_url=str(path.resolve()),
                    doc_type=doc_type,
                ),
            )
        ]
