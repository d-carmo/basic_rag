"""Tests for AnthropicLLM and OpenAILLM using injected fake clients."""

from __future__ import annotations

import json
from typing import Any

import pytest

from rag.generation.anthropic_llm import AnthropicLLM
from rag.generation.base import Message
from rag.generation.openai_llm import OpenAILLM


def _msg(content: str = "What is Python?") -> list[Message]:
    return [Message(role="user", content=content)]


# ── Anthropic fakes ────────────────────────────────────────────────────────────

def _anthropic_client(text: str, stream_chunks: list[str] | None = None) -> Any:
    chunks = stream_chunks or [text]

    class _Block:
        type = "text"

        def __init__(self, t: str) -> None:
            self.text = t

    class _Msg:
        def __init__(self, t: str) -> None:
            self.content = [_Block(t)]

    class _TextStream:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            for c in chunks:
                yield c

    class _StreamCtx:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        @property
        def text_stream(self):
            return _TextStream()

    class _Messages:
        async def create(self, **kwargs):
            return _Msg(text)

        def stream(self, **kwargs):
            return _StreamCtx()

    class _Client:
        messages = _Messages()

    return _Client()


# ── AnthropicLLM ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_anthropic_complete_returns_text() -> None:
    llm = AnthropicLLM(client=_anthropic_client("Python is a language."))
    result = await llm.complete(_msg())
    assert result == "Python is a language."


@pytest.mark.asyncio
async def test_anthropic_complete_uses_model() -> None:
    llm = AnthropicLLM(model="claude-opus-4-8", client=_anthropic_client("ok"))
    assert llm._model == "claude-opus-4-8"


@pytest.mark.asyncio
async def test_anthropic_stream_yields_chunks() -> None:
    llm = AnthropicLLM(client=_anthropic_client("", stream_chunks=["Hello", " ", "world"]))
    chunks = []
    async for chunk in llm.stream(_msg()):
        chunks.append(chunk)
    assert "".join(chunks) == "Hello world"


@pytest.mark.asyncio
async def test_anthropic_system_param_included() -> None:
    llm = AnthropicLLM(system="Be helpful.", client=_anthropic_client("ok"))
    params = llm._params(_msg())
    assert params["system"] == "Be helpful."


@pytest.mark.asyncio
async def test_anthropic_no_system_omits_key() -> None:
    llm = AnthropicLLM(client=_anthropic_client("ok"))
    params = llm._params(_msg())
    assert "system" not in params


# ── OpenAI fakes ───────────────────────────────────────────────────────────────

def _openai_client(text: str, stream_chunks: list[str] | None = None) -> Any:
    chunks = stream_chunks or []

    class _Resp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"choices": [{"message": {"content": text}}]}

    class _StreamLine:
        def raise_for_status(self) -> None:
            pass

        async def aiter_lines(self):
            for c in chunks:
                yield f'data: {json.dumps({"choices": [{"delta": {"content": c}}]})}'
            yield "data: [DONE]"

    class _StreamCtx:
        async def __aenter__(self):
            return _StreamLine()

        async def __aexit__(self, *a):
            pass

    class _Client:
        async def post(self, url, **kwargs):
            return _Resp()

        def stream(self, method, url, **kwargs):
            return _StreamCtx()

    return _Client()


# ── OpenAILLM ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_openai_complete_returns_text() -> None:
    llm = OpenAILLM(client=_openai_client("OpenAI answer."))
    result = await llm.complete(_msg())
    assert result == "OpenAI answer."


@pytest.mark.asyncio
async def test_openai_stream_yields_chunks() -> None:
    llm = OpenAILLM(client=_openai_client("", stream_chunks=["chunk1", "chunk2"]))
    chunks = []
    async for c in llm.stream(_msg()):
        chunks.append(c)
    assert chunks == ["chunk1", "chunk2"]


@pytest.mark.asyncio
async def test_openai_custom_base_url() -> None:
    llm = OpenAILLM(base_url="http://localhost:11434/v1", client=_openai_client("ok"))
    assert llm._base_url == "http://localhost:11434/v1"


@pytest.mark.asyncio
async def test_openai_local_no_api_key() -> None:
    llm = OpenAILLM(base_url="http://localhost:1234/v1", api_key="", client=_openai_client("ok"))
    headers = llm._headers()
    assert "Authorization" not in headers


@pytest.mark.asyncio
async def test_openai_trailing_slash_stripped() -> None:
    llm = OpenAILLM(base_url="http://localhost:8000/v1/", client=_openai_client("ok"))
    assert not llm._base_url.endswith("/")


@pytest.mark.asyncio
async def test_openai_stream_skips_done_sentinel() -> None:
    llm = OpenAILLM(client=_openai_client("", stream_chunks=["hello"]))
    chunks = []
    async for c in llm.stream(_msg()):
        chunks.append(c)
    assert "[DONE]" not in chunks
