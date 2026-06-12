import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Comment

from rag.loaders.base import BaseLoader, DocType, Document, DocumentMetadata

_BOILERPLATE = ["nav", "header", "footer", "aside", "script", "style", "noscript"]
_MULTI_BLANK = re.compile(r"\n{3,}")


def _is_comment(text: Any) -> bool:
    return isinstance(text, Comment)


class HtmlLoader(BaseLoader):
    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".html", ".htm"})

    def load(self, source: Path | str) -> list[Document]:
        path = Path(source)
        html = path.read_text(encoding="utf-8", errors="replace")
        return self._parse(html, source_url=str(path.resolve()))

    def load_from_string(self, html: str, source_url: str = "") -> list[Document]:
        return self._parse(html, source_url=source_url)

    def _parse(self, html: str, source_url: str) -> list[Document]:
        soup = BeautifulSoup(html, "html.parser")

        for tag in list(soup.find_all(_BOILERPLATE)):
            tag.decompose()

        for comment in list(soup.find_all(string=_is_comment)):
            comment.extract()

        text = soup.get_text(separator="\n", strip=True)
        text = _MULTI_BLANK.sub("\n\n", text)

        if not text.strip():
            return []

        return [
            Document(
                text=text,
                metadata=DocumentMetadata(
                    source_url=source_url,
                    doc_type=DocType.HTML,
                ),
            )
        ]
