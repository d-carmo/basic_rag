from __future__ import annotations

from typing import Any

from rag.retriever.base import BaseQueryTransform

_HYDE_PROMPT = """\
Write a short passage (3-5 sentences) that directly answers the following question. \
Write as if excerpted from a reference document. Output only the passage, nothing else.

Question: {query}"""

_MULTI_QUERY_PROMPT = """\
Generate {n} distinct search queries that would help retrieve information to answer \
the following question. Output one query per line with no numbering or extra text.

Question: {query}"""

_STEP_BACK_PROMPT = """\
Given the following question, write a more general background question whose answer \
would help in answering the original. Output only the background question, nothing else.

Question: {query}"""


def _parse_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


class HyDETransform(BaseQueryTransform):
    """Hypothetical Document Embeddings: generate a hypothetical answer and embed it."""

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        max_tokens: int = 512,
        client: Any = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client: Any = client

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic  # type: ignore[import-not-found]

            self._client = anthropic.AsyncAnthropic()
        return self._client

    async def transform(self, query: str) -> list[str]:
        try:
            msg = await self._get_client().messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": _HYDE_PROMPT.format(query=query)}],
            )
            text = next((b.text for b in msg.content if b.type == "text"), "").strip()
            return [text] if text else []
        except Exception:
            return []


class MultiQueryTransform(BaseQueryTransform):
    """Generate N rephrased query variants for multi-query retrieval."""

    def __init__(
        self,
        n: int = 3,
        model: str = "claude-opus-4-8",
        max_tokens: int = 256,
        client: Any = None,
    ) -> None:
        self._n = n
        self._model = model
        self._max_tokens = max_tokens
        self._client: Any = client

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic  # type: ignore[import-not-found]

            self._client = anthropic.AsyncAnthropic()
        return self._client

    async def transform(self, query: str) -> list[str]:
        try:
            msg = await self._get_client().messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": _MULTI_QUERY_PROMPT.format(n=self._n, query=query),
                    }
                ],
            )
            raw = next((b.text for b in msg.content if b.type == "text"), "")
            return _parse_lines(raw)[: self._n]
        except Exception:
            return []


class StepBackTransform(BaseQueryTransform):
    """Generate a broader background question to improve context retrieval."""

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        max_tokens: int = 128,
        client: Any = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client: Any = client

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic  # type: ignore[import-not-found]

            self._client = anthropic.AsyncAnthropic()
        return self._client

    async def transform(self, query: str) -> list[str]:
        try:
            msg = await self._get_client().messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": _STEP_BACK_PROMPT.format(query=query)}],
            )
            raw = next((b.text for b in msg.content if b.type == "text"), "").strip()
            lines = _parse_lines(raw)
            return [lines[0]] if lines else []
        except Exception:
            return []
