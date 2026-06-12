"""Tests for SystemPromptBuilder and UserPromptBuilder."""

from rag.assembler.base import ContextChunk
from rag.generation.prompts import SystemPromptBuilder, UserPromptBuilder


def _chunk(text: str) -> ContextChunk:
    return ContextChunk(text=text, source_id="src", score=0.9, rank=0)


# ── SystemPromptBuilder ────────────────────────────────────────────────────────

def test_system_prompt_default_template() -> None:
    s = SystemPromptBuilder().build()
    assert len(s) > 0
    assert "context" in s.lower()


def test_system_prompt_custom_template() -> None:
    s = SystemPromptBuilder(template="Custom template.").build()
    assert s == "Custom template."


def test_system_prompt_extra_appended() -> None:
    s = SystemPromptBuilder().build(extra="Extra instruction.")
    assert "Extra instruction." in s


def test_system_prompt_no_extra() -> None:
    b = SystemPromptBuilder(template="Base.")
    assert b.build() == "Base."
    assert b.build(extra=None) == "Base."


# ── UserPromptBuilder ──────────────────────────────────────────────────────────

def test_user_prompt_includes_query() -> None:
    p = UserPromptBuilder().build("What is Python?", [])
    assert "What is Python?" in p


def test_user_prompt_includes_chunk_text() -> None:
    chunks = [_chunk("Python is a programming language.")]
    p = UserPromptBuilder().build("Question?", chunks)
    assert "Python is a programming language." in p


def test_user_prompt_citation_notation() -> None:
    chunks = [_chunk("First chunk."), _chunk("Second chunk.")]
    p = UserPromptBuilder().build("Q?", chunks)
    assert "[1]" in p
    assert "[2]" in p


def test_user_prompt_empty_chunks() -> None:
    p = UserPromptBuilder().build("Why?", [])
    assert "Why?" in p
    assert "[1]" not in p


def test_user_prompt_multiple_chunks_ordered() -> None:
    chunks = [_chunk("Alpha content."), _chunk("Beta content."), _chunk("Gamma content.")]
    p = UserPromptBuilder().build("Q?", chunks)
    pos1 = p.index("[1]")
    pos2 = p.index("[2]")
    pos3 = p.index("[3]")
    assert pos1 < pos2 < pos3
