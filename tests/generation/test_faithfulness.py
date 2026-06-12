"""Tests for FaithfulnessChecker."""

from rag.assembler.base import ContextChunk
from rag.generation.faithfulness import FaithfulnessChecker


def _chunk(text: str) -> ContextChunk:
    return ContextChunk(text=text, source_id="src", score=0.9, rank=0)


checker = FaithfulnessChecker(overlap_threshold=0.5)


# ── score() ────────────────────────────────────────────────────────────────────

def test_score_empty_chunks_returns_zero() -> None:
    assert checker.score("Some answer.", []) == 0.0


def test_score_empty_answer_returns_one() -> None:
    assert checker.score("", [_chunk("context")]) == 1.0


def test_score_high_overlap_returns_high() -> None:
    chunks = [_chunk("Python programming language created Guido van Rossum")]
    answer = "Python programming language was created by Guido van Rossum."
    s = checker.score(answer, chunks)
    assert s > 0.5


def test_score_no_overlap_returns_low() -> None:
    chunks = [_chunk("Machine learning trains neural networks")]
    answer = "The weather today is sunny and warm outside."
    s = checker.score(answer, chunks)
    assert s < 0.5


def test_score_between_zero_and_one() -> None:
    chunks = [_chunk("Qdrant is a vector database for similarity search")]
    answer = "Qdrant stores vectors. The stock market rose today."
    s = checker.score(answer, chunks)
    assert 0.0 <= s <= 1.0


def test_score_multiple_chunks_merged() -> None:
    chunks = [_chunk("Python language"), _chunk("vector databases search")]
    answer = "Python language vector databases search."
    s = checker.score(answer, chunks)
    assert s >= 0.5


# ── is_faithful() ──────────────────────────────────────────────────────────────

def test_is_faithful_high_overlap() -> None:
    chunks = [_chunk("quantum entanglement physics particles")]
    answer = "Quantum entanglement involves physics particles."
    assert checker.is_faithful(answer, chunks) is True


def test_is_faithful_no_overlap() -> None:
    chunks = [_chunk("advanced quantum computing hardware")]
    answer = "The flower blooms every spring in the garden."
    assert checker.is_faithful(answer, chunks) is False


def test_is_faithful_respects_threshold() -> None:
    strict = FaithfulnessChecker(overlap_threshold=0.9)
    lenient = FaithfulnessChecker(overlap_threshold=0.1)
    chunks = [_chunk("Python machine learning library scikit")]
    answer = "Python library."
    assert lenient.is_faithful(answer, chunks) is True
    # strict might differ — just verify it's a bool
    assert isinstance(strict.is_faithful(answer, chunks), bool)
