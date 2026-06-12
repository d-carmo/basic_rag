from pathlib import Path
from typing import TYPE_CHECKING

from rag.loaders.base import BaseLoader

if TYPE_CHECKING:
    from rag.loaders.csv_loader import CsvLoader
    from rag.loaders.docx import DocxLoader
    from rag.loaders.html import HtmlLoader
    from rag.loaders.json_loader import JsonLoader
    from rag.loaders.pdf import PdfLoader
    from rag.loaders.text import TextLoader

# Maps extension → (module path, class name) to defer imports until first use.
# This prevents a missing optional library (pdfplumber, python-docx, etc.)
# from blocking loaders that don't need it.
_EXTENSION_MAP: dict[str, tuple[str, str]] = {
    ".pdf":      ("rag.loaders.pdf",         "PdfLoader"),
    ".docx":     ("rag.loaders.docx",        "DocxLoader"),
    ".doc":      ("rag.loaders.docx",        "DocxLoader"),
    ".txt":      ("rag.loaders.text",        "TextLoader"),
    ".text":     ("rag.loaders.text",        "TextLoader"),
    ".rst":      ("rag.loaders.text",        "TextLoader"),
    ".md":       ("rag.loaders.text",        "TextLoader"),
    ".markdown": ("rag.loaders.text",        "TextLoader"),
    ".html":     ("rag.loaders.html",        "HtmlLoader"),
    ".htm":      ("rag.loaders.html",        "HtmlLoader"),
    ".json":     ("rag.loaders.json_loader", "JsonLoader"),
    ".jsonl":    ("rag.loaders.json_loader", "JsonLoader"),
    ".ndjson":   ("rag.loaders.json_loader", "JsonLoader"),
    ".csv":      ("rag.loaders.csv_loader",  "CsvLoader"),
    ".tsv":      ("rag.loaders.csv_loader",  "CsvLoader"),
}


def get_loader(source: Path | str) -> BaseLoader:
    ext = Path(source).suffix.lower()
    entry = _EXTENSION_MAP.get(ext)
    if entry is None:
        raise ValueError(f"No loader registered for extension '{ext}': {source}")
    module_path, class_name = entry
    import importlib
    module = importlib.import_module(module_path)
    loader_cls: type[BaseLoader] = getattr(module, class_name)
    return loader_cls()
