"""Tests for TokenBudgetManager."""

from rag.assembler.base import ContextChunk
from rag.assembler.budget import TokenBudgetManager


def _count(text: str) -> int:
    """Simple token counter: 1 token per character (deterministic for tests)."""
    return len(text)


def _chunk(text: str, rank: int = 0) -> ContextChunk:
    return ContextChunk(text=text, source_id="src", score=0.9, rank=rank)


def _manager(max_tokens: int) -> TokenBudgetManager:
    return TokenBudgetManager(max_tokens=max_tokens, token_counter=_count)


# ── all chunks fit ─────────────────────────────────────────────────────────────

def test_all_chunks_fit() -> None:
    chunks = [_chunk("hello"), _chunk("world")]  # 5 + 5 = 10 chars
    fitted, truncated = _manager(20).fit(chunks)
    assert fitted == chunks
    assert truncated is False


def test_exact_budget_fit() -> None:
    chunks = [_chunk("abcde")]  # exactly 5 chars
    fitted, truncated = _manager(5).fit(chunks)
    assert fitted == chunks
    assert truncated is False


def test_empty_input() -> None:
    fitted, truncated = _manager(100).fit([])
    assert fitted == []
    assert truncated is False


def test_single_chunk_fits() -> None:
    fitted, truncated = _manager(100).fit([_chunk("hi")])
    assert len(fitted) == 1
    assert truncated is False


# ── truncation ─────────────────────────────────────────────────────────────────

def test_second_chunk_dropped_when_over_budget() -> None:
    chunks = [_chunk("hello"), _chunk("world")]  # 5 + 5
    fitted, truncated = _manager(5).fit(chunks)
    assert len(fitted) == 1
    assert fitted[0].text == "hello"
    assert truncated is True


def test_zero_budget_drops_all() -> None:
    fitted, truncated = _manager(0).fit([_chunk("hi")])
    assert fitted == []
    assert truncated is True


def test_partial_trim_at_word_boundary() -> None:
    # "ab cd ef" is 8 chars; budget=5 → "ab cd" (5 chars) fits
    chunks = [_chunk("ab cd ef")]
    fitted, truncated = _manager(5).fit(chunks)
    assert len(fitted) == 1
    assert fitted[0].text == "ab cd"
    assert truncated is True


def test_trim_preserves_other_chunk_fields() -> None:
    chunk = ContextChunk(text="aa bb cc dd", source_id="s", score=0.7, rank=2, payload={"x": 1})
    # "aa bb" = 5 chars, "aa bb c" = 7 → budget 5
    fitted, _ = _manager(5).fit([chunk])
    assert fitted[0].source_id == "s"
    assert fitted[0].score == 0.7
    assert fitted[0].rank == 2
    assert fitted[0].payload == {"x": 1}


def test_multiple_chunks_partial_last() -> None:
    # "hello"=5 chars → used=5; remaining=1; "w x"→words=["w","x"]; "w"=1 char fits
    chunks = [_chunk("hello"), _chunk("w x")]
    fitted, truncated = _manager(6).fit(chunks)
    assert fitted[0].text == "hello"
    assert fitted[1].text == "w"
    assert truncated is True
