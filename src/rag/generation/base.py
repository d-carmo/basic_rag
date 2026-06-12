from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass
class Message:
    role: str    # "system" | "user" | "assistant"
    content: str


class BaseLLM(ABC):
    @abstractmethod
    async def complete(self, messages: list[Message]) -> str:
        """Non-streaming completion. Returns the full answer as a string."""
        ...

    def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """Async iterator yielding text chunks. Override as an async generator method."""
        raise NotImplementedError(f"{type(self).__name__} does not support streaming.")
