"""Tests for TitleEnricher."""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from rag.chunker.chunk import ChunkMetadata
from rag.enricher.base import EnrichedChunk
from rag.enricher.title import TitleEnricher


def _chunk(text: str) -> EnrichedChunk:
    return EnrichedChunk(text=text, metadata=ChunkMetadata(chunk_index=0))


def _fake_client(title_text: str) -> Any:
    class _Block:
        type = "text"
        text = title_text

    class _Message:
        content = [_Block()]

    messages_mock = AsyncMock()
    messages_mock.create = AsyncMock(return_value=_Message())
    client = AsyncMock()
    client.messages = messages_mock
    return client


# ── Heading extraction ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_extracts_markdown_h1() -> None:
    enricher = TitleEnricher()
    result = await enricher.enrich(_chunk("# My Section\n\nSome content here."))
    assert result.title == "My Section"


@pytest.mark.asyncio
async def test_extracts_markdown_h3() -> None:
    enricher = TitleEnricher()
    result = await enricher.enrich(_chunk("### Deep Heading\n\nContent."))
    assert result.title == "Deep Heading"


@pytest.mark.asyncio
async def test_extracts_html_heading() -> None:
    enricher = TitleEnricher()
    result = await enricher.enrich(_chunk("<h2>HTML Title</h2>\n<p>Body.</p>"))
    assert result.title == "HTML Title"


@pytest.mark.asyncio
async def test_returns_none_for_plain_text_no_llm() -> None:
    enricher = TitleEnricher(use_llm_fallback=False)
    result = await enricher.enrich(_chunk("Just some plain text with no heading."))
    assert result.title is None


# ── LLM fallback ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_llm_fallback_used_when_no_heading() -> None:
    enricher = TitleEnricher(use_llm_fallback=True, client=_fake_client("A Generated Title"))
    result = await enricher.enrich(_chunk("Some plain text without a heading."))
    assert result.title == "A Generated Title"


@pytest.mark.asyncio
async def test_llm_fallback_not_called_when_heading_found() -> None:
    client = _fake_client("Should Not Appear")
    enricher = TitleEnricher(use_llm_fallback=True, client=client)
    result = await enricher.enrich(_chunk("# Existing Heading\n\nContent."))
    assert result.title == "Existing Heading"
    client.messages.create.assert_not_called()


@pytest.mark.asyncio
async def test_llm_fallback_returns_none_on_error() -> None:
    client = AsyncMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("fail"))
    enricher = TitleEnricher(use_llm_fallback=True, client=client)
    result = await enricher.enrich(_chunk("plain text"))
    assert result.title is None


# ── expensive flag ─────────────────────────────────────────────────────────────

def test_title_enricher_not_expensive_by_default() -> None:
    e = TitleEnricher()
    assert e.expensive is False


def test_title_enricher_expensive_with_llm_fallback() -> None:
    e = TitleEnricher(use_llm_fallback=True)
    assert e.expensive is True
