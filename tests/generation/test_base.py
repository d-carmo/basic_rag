"""Tests for Message dataclass and BaseLLM interface."""

import pytest

from rag.generation.base import BaseLLM, Message


def test_message_fields() -> None:
    m = Message(role="user", content="hello")
    assert m.role == "user"
    assert m.content == "hello"


def test_message_system_role() -> None:
    m = Message(role="system", content="You are helpful.")
    assert m.role == "system"


def test_base_llm_stream_raises_not_implemented() -> None:
    class _MinimalLLM(BaseLLM):
        async def complete(self, messages):
            return "ok"

    with pytest.raises(NotImplementedError):
        _MinimalLLM().stream([])
