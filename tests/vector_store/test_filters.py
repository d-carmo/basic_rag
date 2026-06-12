"""Integration tests for FilterBuilder — require qdrant_client installed."""

from __future__ import annotations

import pytest

pytest.importorskip("qdrant_client", reason="qdrant_client not installed")

from rag.vector_store.filters import FilterBuilder


def test_by_source_produces_filter() -> None:
    f = FilterBuilder.by_source("http://example.com")
    assert f.must is not None
    assert len(f.must) == 1
    assert f.must[0].key == "source_id"


def test_by_doc_type_produces_filter() -> None:
    f = FilterBuilder.by_doc_type("pdf")
    assert f.must[0].key == "doc_type"
    assert f.must[0].match.value == "pdf"


def test_by_language_produces_filter() -> None:
    f = FilterBuilder.by_language("en")
    assert f.must[0].key == "language"
    assert f.must[0].match.value == "en"


def test_combine_merges_must_clauses() -> None:
    f1 = FilterBuilder.by_source("http://example.com")
    f2 = FilterBuilder.by_language("en")
    combined = FilterBuilder.combine(f1, f2)
    assert combined.must is not None
    keys = [c.key for c in combined.must]
    assert "source_id" in keys
    assert "language" in keys


def test_combine_single_filter_unchanged() -> None:
    f = FilterBuilder.by_doc_type("pdf")
    combined = FilterBuilder.combine(f)
    assert len(combined.must) == 1
