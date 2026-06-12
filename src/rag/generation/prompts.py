from __future__ import annotations

from rag.assembler.base import ContextChunk
from rag.generation.base import Message

_SYSTEM_TEMPLATE = """\
You are a helpful AI assistant. Answer questions based ONLY on the provided context blocks.
If the answer cannot be found in the context, say you don't have enough information — do not invent facts.
Cite sources using [n] notation, where n is the context block number.\
"""


class SystemPromptBuilder:
    def __init__(self, template: str | None = None) -> None:
        self._template = template or _SYSTEM_TEMPLATE

    def build(self, extra: str | None = None) -> str:
        if extra:
            return f"{self._template}\n\n{extra}"
        return self._template


class UserPromptBuilder:
    def build(
        self,
        query: str,
        chunks: list[ContextChunk],
        citation_map: dict[int, dict] | None = None,
    ) -> str:
        parts: list[str] = ["Context:"]
        for i, chunk in enumerate(chunks):
            parts.append(f"[{i + 1}] {chunk.text}")
        parts.append(f"\nQuestion: {query}")
        return "\n".join(parts)
