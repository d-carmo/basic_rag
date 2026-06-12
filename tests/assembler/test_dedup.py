"""Tests for NearDuplicateFilter."""

import pytest

from rag.assembler.base import ContextChunk
from rag.assembler.dedup import NearDuplicateFilter, _jaccard, _cosine, _shingles


def _chunk(text: str, rank: int = 0) -> ContextChunk:
    return ContextChunk(text=text, source_id="src", score=0.9, rank=rank)


# ── internal helpers ──────────────────────────────────────────────────────────

def test_shingles_produces_frozenset() -> None:
    sh = _shingles("hello world")
    assert isinstance(sh, frozenset)
    assert len(sh) > 0


def test_jaccard_identical() -> None:
    sh = _shingles("hello world")
    assert _jaccard(sh, sh) == 1.0


def test_jaccard_disjoint() -> None:
    a = _shingles("aaaaa")
    b = _shingles("bbbbb")
    assert _jaccard(a, b) == 0.0


def test_cosine_identical() -> None:
    v = [1.0, 0.0, 0.5]
    assert abs(_cosine(v, v) - 1.0) < 1e-6


def test_cosine_orthogonal() -> None:
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_zero_vector() -> None:
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


# ── NearDuplicateFilter: Jaccard ──────────────────────────────────────────────

def test_jaccard_keeps_distinct_chunks() -> None:
    chunks = [_chunk("The cat sat on the mat"), _chunk("Quantum entanglement in physics")]
    result = NearDuplicateFilter(threshold=0.8).filter(chunks)
    assert len(result) == 2


def test_jaccard_removes_exact_duplicate() -> None:
    text = "The quick brown fox jumps over the lazy dog"
    chunks = [_chunk(text), _chunk(text)]
    result = NearDuplicateFilter(threshold=0.85).filter(chunks)
    assert len(result) == 1
    assert result[0].text == text


def test_jaccard_first_occurrence_kept() -> None:
    chunks = [_chunk("identical text here", rank=0), _chunk("identical text here", rank=1)]
    result = NearDuplicateFilter(threshold=0.9).filter(chunks)
    assert result[0].rank == 0


def test_jaccard_empty_input() -> None:
    assert NearDuplicateFilter().filter([]) == []


def test_jaccard_single_chunk() -> None:
    chunks = [_chunk("only one")]
    assert NearDuplicateFilter().filter(chunks) == chunks


# ── NearDuplicateFilter: cosine ───────────────────────────────────────────────

def test_cosine_removes_near_duplicate_vectors() -> None:
    chunks = [_chunk("doc a", rank=0), _chunk("doc b", rank=1)]
    # nearly identical vectors → should be deduplicated at threshold=0.95
    vecs = [[1.0, 0.001], [1.0, 0.002]]
    result = NearDuplicateFilter(threshold=0.95, method="cosine").filter(chunks, vectors=vecs)
    assert len(result) == 1


def test_cosine_keeps_orthogonal_vectors() -> None:
    chunks = [_chunk("doc a"), _chunk("doc b")]
    vecs = [[1.0, 0.0], [0.0, 1.0]]
    result = NearDuplicateFilter(threshold=0.95, method="cosine").filter(chunks, vectors=vecs)
    assert len(result) == 2


def test_cosine_falls_back_to_jaccard_when_no_vectors() -> None:
    text = "repeated repeated repeated repeated repeated"
    chunks = [_chunk(text), _chunk(text)]
    result = NearDuplicateFilter(threshold=0.9, method="cosine").filter(chunks, vectors=None)
    assert len(result) == 1


# ── validation ─────────────────────────────────────────────────────────────────

def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="unknown"):
        NearDuplicateFilter(method="unknown")
