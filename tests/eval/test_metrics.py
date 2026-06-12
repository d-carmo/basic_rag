from __future__ import annotations

import pytest

from rag.eval import metrics as m


def test_answer_similarity_identical():
    s = m.answer_similarity("the cat sat", "the cat sat")
    assert s == pytest.approx(1.0)


def test_answer_similarity_disjoint():
    s = m.answer_similarity("cat", "dog")
    assert s == pytest.approx(0.0)


def test_answer_similarity_partial():
    s = m.answer_similarity("cat sat mat", "cat sat")
    assert 0.0 < s < 1.0


def test_context_recall_all_found():
    assert m.context_recall(["a", "b", "c"], ["a", "b"]) == pytest.approx(1.0)


def test_context_recall_none_found():
    assert m.context_recall(["x"], ["a", "b"]) == pytest.approx(0.0)


def test_context_recall_empty_relevant():
    assert m.context_recall([], []) == pytest.approx(1.0)


def test_context_precision_all_relevant():
    assert m.context_precision(["a", "b"], ["a", "b", "c"]) == pytest.approx(1.0)


def test_context_precision_none_relevant():
    assert m.context_precision(["x", "y"], ["a"]) == pytest.approx(0.0)


def test_context_precision_empty_retrieved():
    assert m.context_precision([], ["a"]) == pytest.approx(0.0)


def test_faithfulness_empty_answer():
    assert m.faithfulness("", ["context"]) == pytest.approx(0.0)


def test_faithfulness_grounded():
    score = m.faithfulness("cat sat mat", ["cat sat mat"])
    assert score > 0.0


def test_faithfulness_ungrounded():
    score = m.faithfulness("xyz qwerty", ["cat sat mat"])
    assert score == pytest.approx(0.0)
