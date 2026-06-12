"""Tests for RetrievalResult, rrf_merge — no external deps."""

from rag.retriever.base import RetrievalResult, rrf_merge


def _r(id_: str, score: float, rank: int = 0) -> RetrievalResult:
    return RetrievalResult(id=id_, score=score, rank=rank, payload={"text": id_, "source_id": f"src/{id_}"})


# ── RetrievalResult ────────────────────────────────────────────────────────────

def test_retrieval_result_fields() -> None:
    r = _r("abc", 0.9, rank=1)
    assert r.id == "abc"
    assert r.score == 0.9
    assert r.rank == 1


def test_retrieval_result_text_property() -> None:
    r = RetrievalResult(id="x", score=1.0, rank=0, payload={"text": "hello world"})
    assert r.text == "hello world"


def test_retrieval_result_text_missing() -> None:
    r = RetrievalResult(id="x", score=1.0, rank=0)
    assert r.text == ""


def test_retrieval_result_source_id_property() -> None:
    r = RetrievalResult(id="x", score=1.0, rank=0, payload={"source_id": "http://example.com"})
    assert r.source_id == "http://example.com"


# ── rrf_merge ─────────────────────────────────────────────────────────────────

def test_rrf_merge_single_list_preserves_order() -> None:
    results = [_r("a", 1.0, 0), _r("b", 0.8, 1), _r("c", 0.5, 2)]
    merged = rrf_merge([results])
    ids = [r.id for r in merged]
    assert ids == ["a", "b", "c"]


def test_rrf_merge_two_lists_boosts_overlapping() -> None:
    # "b" appears first in both lists → should score highest
    list1 = [_r("a", 1.0, 0), _r("b", 0.8, 1)]
    list2 = [_r("b", 1.0, 0), _r("c", 0.8, 1)]
    merged = rrf_merge([list1, list2])
    assert merged[0].id == "b"


def test_rrf_merge_assigns_ascending_ranks() -> None:
    merged = rrf_merge([[_r("x", 1.0, 0), _r("y", 0.5, 1)]])
    for i, r in enumerate(merged):
        assert r.rank == i


def test_rrf_merge_empty_lists() -> None:
    assert rrf_merge([[], []]) == []


def test_rrf_merge_scores_are_positive() -> None:
    results = [_r("a", 1.0, 0), _r("b", 0.5, 1)]
    merged = rrf_merge([results])
    assert all(r.score > 0 for r in merged)


def test_rrf_merge_deduplicates_ids() -> None:
    # Same id in both lists → appears only once in output
    r = _r("dup", 0.9, 0)
    merged = rrf_merge([[r], [r]])
    assert sum(1 for x in merged if x.id == "dup") == 1


def test_rrf_merge_payload_preserved() -> None:
    results = [RetrievalResult(id="a", score=1.0, rank=0, payload={"text": "hello"})]
    merged = rrf_merge([results])
    assert merged[0].payload == {"text": "hello"}
