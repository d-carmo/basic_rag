from __future__ import annotations

from typing import Any


class FilterBuilder:
    """Builds qdrant_client Filter objects. All imports are lazy so this module
    can be imported without qdrant_client installed."""

    @staticmethod
    def by_source(source_id: str) -> Any:
        from qdrant_client.models import FieldCondition, Filter, MatchValue  # type: ignore[import-not-found]

        return Filter(must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))])

    @staticmethod
    def by_doc_type(doc_type: str) -> Any:
        from qdrant_client.models import FieldCondition, Filter, MatchValue  # type: ignore[import-not-found]

        return Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value=doc_type))])

    @staticmethod
    def by_language(language: str) -> Any:
        from qdrant_client.models import FieldCondition, Filter, MatchValue  # type: ignore[import-not-found]

        return Filter(must=[FieldCondition(key="language", match=MatchValue(value=language))])

    @staticmethod
    def combine(*filters: Any) -> Any:
        from qdrant_client.models import Filter  # type: ignore[import-not-found]

        must: list[Any] = []
        for f in filters:
            if getattr(f, "must", None):
                must.extend(f.must)
        return Filter(must=must)
