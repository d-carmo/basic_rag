from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
import uuid

from pydantic import BaseModel, Field


class DocType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    UNKNOWN = "unknown"


class DocumentMetadata(BaseModel):
    source_url: str = ""
    page: int | None = None
    doc_type: DocType = DocType.UNKNOWN
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    language: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    metadata: DocumentMetadata


class BaseLoader(ABC):
    """Abstract base for all document loaders."""

    @abstractmethod
    def load(self, source: Path | str) -> list[Document]:
        ...

    @property
    @abstractmethod
    def supported_extensions(self) -> frozenset[str]:
        ...

    def can_handle(self, source: Path | str) -> bool:
        return Path(source).suffix.lower() in self.supported_extensions
