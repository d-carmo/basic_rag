from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from rag.generation.base import BaseLLM, Message


class AnthropicLLM(BaseLLM):
    """Claude backend via the anthropic SDK (lazy-imported)."""

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        max_tokens: int = 2048,
        system: str | None = None,
        client: Any = None,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._system = system
        self._client: Any = client
        self._api_key = api_key

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic  # type: ignore[import-not-found]

            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    def _params(self, messages: list[Message]) -> dict[str, Any]:
        p: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if self._system:
            p["system"] = self._system
        return p

    async def complete(self, messages: list[Message]) -> str:
        msg = await self._get_client().messages.create(**self._params(messages))
        return next((b.text for b in msg.content if b.type == "text"), "")

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:  # type: ignore[override]
        async with self._get_client().messages.stream(**self._params(messages)) as s:
            async for text in s.text_stream:
                yield text
