from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from rag.generation.base import BaseLLM, Message

_OPENAI_DEFAULT_URL = "https://api.openai.com/v1"


class OpenAILLM(BaseLLM):
    """
    OpenAI-compatible backend via httpx.

    Works with OpenAI and any local LLM server that exposes an OpenAI-compatible
    chat completions endpoint:
      - Ollama:    base_url="http://localhost:11434/v1"
      - LM Studio: base_url="http://localhost:1234/v1"
      - vLLM:      base_url="http://localhost:8000/v1"

    Local servers typically do not require an API key; set api_key="" or omit it.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 2048,
        timeout: float = 600.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL", _OPENAI_DEFAULT_URL)
        ).rstrip("/")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = client

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _payload(self, messages: list[Message], *, stream: bool = False) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": self._max_tokens,
            "stream": stream,
        }

    async def complete(self, messages: list[Message]) -> str:
        resp = await self._get_client().post(
            f"{self._base_url}/chat/completions",
            json=self._payload(messages),
            headers=self._headers(),
        )
        resp.raise_for_status()
        return str(resp.json()["choices"][0]["message"]["content"])

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:  # type: ignore[override]
        async with self._get_client().stream(
            "POST",
            f"{self._base_url}/chat/completions",
            json=self._payload(messages, stream=True),
            headers=self._headers(),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    content = chunk["choices"][0]["delta"].get("content") or ""
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError):
                    continue
