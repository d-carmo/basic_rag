import logging
from pathlib import Path
from typing import Any

import pdfplumber

from rag.loaders.base import BaseLoader, DocType, Document, DocumentMetadata

logger = logging.getLogger(__name__)


class PdfLoader(BaseLoader):
    def __init__(self, ocr_fallback: bool = True) -> None:
        self._ocr_fallback = ocr_fallback

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".pdf"})

    def load(self, source: Path | str) -> list[Document]:
        path = Path(source)
        docs: list[Document] = []

        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text: str = page.extract_text() or ""

                if not text.strip() and self._ocr_fallback:
                    text = self._ocr_page(page)

                if not text.strip():
                    continue

                docs.append(
                    Document(
                        text=text,
                        metadata=DocumentMetadata(
                            source_url=str(path.resolve()),
                            page=page_num,
                            doc_type=DocType.PDF,
                        ),
                    )
                )

        return docs

    def _ocr_page(self, page: Any) -> str:
        try:
            import pytesseract  # type: ignore[import-untyped]

            img = page.to_image(resolution=300).original
            return str(pytesseract.image_to_string(img))
        except ImportError:
            logger.warning("pytesseract not installed; skipping OCR fallback for empty page")
            return ""
