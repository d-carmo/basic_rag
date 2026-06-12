"""Tests for HypotheticalQuestionsEnricher."""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from rag.chunker.chunk import ChunkMetadata
from rag.enricher.base import EnrichedChunk
from rag.enricher.hypothetical import HypotheticalQuestionsEnricher


def _chunk(text: str = "Paris is the capital of France.") -> EnrichedChunk:
    return EnrichedChunk(text=text, metadata=ChunkMetadata(chunk_index=0))


def _fake_client(response_text: str) -> Any:
    """Build a minimal async mock that mimics the Anthropic messages.create response."""

    class _Block:
        type = "text"
        text = response_text

    class _Message:
        content = [_Block()]

    messages_mock = AsyncMock()
    messages_mock.create = AsyncMock(return_value=_Message())

    client = AsyncMock()
    client.messages = messages_mock
    return client


@pytest.mark.asyncio
async def test_hypothetical_parses_questions() -> None:
    raw = "What is the capital of France?\nWhere is Paris located?\nIs Paris in Europe?"
    enricher = HypotheticalQuestionsEnricher(n_questions=3, client=_fake_client(raw))
    result = await enricher.enrich(_chunk())
    assert len(result.hypothetical_questions) == 3
    assert all("?" in q for q in result.hypothetical_questions)


@pytest.mark.asyncio
async def test_hypothetical_caps_at_n_questions() -> None:
    raw = "\n".join(f"Question {i}?" for i in range(10))
    enricher = HypotheticalQuestionsEnricher(n_questions=3, client=_fake_client(raw))
    result = await enricher.enrich(_chunk())
    assert len(result.hypothetical_questions) <= 3


@pytest.mark.asyncio
async def test_hypothetical_filters_non_questions() -> None:
    raw = "This is not a question.\nBut this is a question?\nAnother non-question."
    enricher = HypotheticalQuestionsEnricher(n_questions=5, client=_fake_client(raw))
    result = await enricher.enrich(_chunk())
    assert result.hypothetical_questions == ["But this is a question?"]


@pytest.mark.asyncio
async def test_hypothetical_returns_empty_on_client_error() -> None:
    client = AsyncMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("API down"))
    enricher = HypotheticalQuestionsEnricher(client=client)
    result = await enricher.enrich(_chunk())
    assert result.hypothetical_questions == []


def test_hypothetical_is_expensive() -> None:
    assert HypotheticalQuestionsEnricher.expensive is True
