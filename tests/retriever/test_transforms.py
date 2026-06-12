"""Tests for HyDETransform, MultiQueryTransform, StepBackTransform."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from rag.retriever.transforms import HyDETransform, MultiQueryTransform, StepBackTransform


def _fake_client(text: str) -> Any:
    class _Block:
        type = "text"

        def __init__(self, t: str) -> None:
            self.text = t

    class _Msg:
        def __init__(self, t: str) -> None:
            self.content = [_Block(t)]

    client = AsyncMock()
    client.messages.create = AsyncMock(return_value=_Msg(text))
    return client


def _broken_client() -> Any:
    client = AsyncMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("API down"))
    return client


# ── HyDETransform ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_hyde_returns_hypothetical_doc() -> None:
    client = _fake_client("Photosynthesis is the process by which plants convert light to energy.")
    transform = HyDETransform(client=client)
    variants = await transform.transform("What is photosynthesis?")
    assert len(variants) == 1
    assert "Photosynthesis" in variants[0]


@pytest.mark.asyncio
async def test_hyde_returns_empty_on_error() -> None:
    transform = HyDETransform(client=_broken_client())
    variants = await transform.transform("something")
    assert variants == []


@pytest.mark.asyncio
async def test_hyde_strips_whitespace() -> None:
    client = _fake_client("  Some answer.  ")
    transform = HyDETransform(client=client)
    variants = await transform.transform("q")
    assert variants == ["Some answer."]


@pytest.mark.asyncio
async def test_hyde_empty_response_returns_empty() -> None:
    client = _fake_client("")
    transform = HyDETransform(client=client)
    variants = await transform.transform("q")
    assert variants == []


# ── MultiQueryTransform ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_multi_query_returns_n_queries() -> None:
    raw = "How do plants make food?\nWhat is the role of chlorophyll?\nHow does sunlight power life?"
    client = _fake_client(raw)
    transform = MultiQueryTransform(n=3, client=client)
    variants = await transform.transform("What is photosynthesis?")
    assert len(variants) == 3


@pytest.mark.asyncio
async def test_multi_query_caps_at_n() -> None:
    raw = "\n".join(f"Query {i}" for i in range(10))
    client = _fake_client(raw)
    transform = MultiQueryTransform(n=3, client=client)
    variants = await transform.transform("q")
    assert len(variants) == 3


@pytest.mark.asyncio
async def test_multi_query_returns_empty_on_error() -> None:
    transform = MultiQueryTransform(client=_broken_client())
    variants = await transform.transform("q")
    assert variants == []


@pytest.mark.asyncio
async def test_multi_query_filters_blank_lines() -> None:
    raw = "Real query\n\n  \nAnother query"
    client = _fake_client(raw)
    transform = MultiQueryTransform(n=5, client=client)
    variants = await transform.transform("q")
    assert len(variants) == 2
    assert "" not in variants


# ── StepBackTransform ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_step_back_returns_one_broader_query() -> None:
    client = _fake_client("What are the fundamental biological processes in plants?")
    transform = StepBackTransform(client=client)
    variants = await transform.transform("What is photosynthesis?")
    assert len(variants) == 1
    assert len(variants[0]) > 0


@pytest.mark.asyncio
async def test_step_back_returns_only_first_line() -> None:
    client = _fake_client("Broader question?\nIgnored second line.")
    transform = StepBackTransform(client=client)
    variants = await transform.transform("q")
    assert len(variants) == 1
    assert variants[0] == "Broader question?"


@pytest.mark.asyncio
async def test_step_back_returns_empty_on_error() -> None:
    transform = StepBackTransform(client=_broken_client())
    variants = await transform.transform("q")
    assert variants == []
