from __future__ import annotations

from typing import Callable

from rag.enricher.base import BaseEnricher, EnrichedChunk

Detector = Callable[[str], str]


class LanguageEnricher(BaseEnricher):
    expensive = False

    def __init__(self, detector: Detector | None = None) -> None:
        self._detector = detector

    def _detect(self, text: str) -> str | None:
        if self._detector is not None:
            try:
                return self._detector(text)
            except Exception:
                return None
        try:
            from langdetect import detect  # type: ignore[import-not-found]

            return detect(text)
        except Exception:
            return None

    async def enrich(self, chunk: EnrichedChunk) -> EnrichedChunk:
        language = self._detect(chunk.text)
        return chunk.model_copy(update={"language": language})
