from __future__ import annotations

import re
from typing import Any

from rag.enricher.base import BaseEnricher, EnrichedChunk

_MD_HEADING = re.compile(r"^#{1,4}\s+(.+)$", re.MULTILINE)
_HTML_HEADING = re.compile(r"<h[1-4][^>]*>(.*?)</h[1-4]>", re.IGNORECASE | re.DOTALL)
_INNER_TAG = re.compile(r"<[^>]+>")


class TitleEnricher(BaseEnricher):
    def __init__(
        self,
        use_llm_fallback: bool = False,
        model: str = "claude-opus-4-8",
        max_tokens: int = 64,
        client: Any = None,
    ) -> None:
        self._use_llm = use_llm_fallback
        self._model = model
        self._max_tokens = max_tokens
        self._client: Any = client
        self.expensive = use_llm_fallback

    def _extract_heading(self, text: str) -> str | None:
        m = _MD_HEADING.search(text)
        if m:
            return m.group(1).strip() or None
        m = _HTML_HEADING.search(text)
        if m:
            return _INNER_TAG.sub("", m.group(1)).strip() or None
        return None

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic()
        return self._client

    async def _llm_title(self, text: str) -> str | None:
        try:
            client = self._get_client()
            prompt = (
                "Provide a short, descriptive title (5–10 words) for the following passage. "
                "Output only the title, nothing else.\n\nPassage:\n" + text[:1500]
            )
            msg = await client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            title = next(
                (b.text for b in msg.content if b.type == "text"),
                "",
            ).strip()
            return title or None
        except Exception:
            return None

    async def enrich(self, chunk: EnrichedChunk) -> EnrichedChunk:
        title = self._extract_heading(chunk.text)
        if title is None and self._use_llm:
            title = await self._llm_title(chunk.text)
        return chunk.model_copy(update={"title": title})
