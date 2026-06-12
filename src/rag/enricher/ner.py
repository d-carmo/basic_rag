from __future__ import annotations

from typing import Any

from rag.enricher.base import BaseEnricher, Entity, EnrichedChunk


class NEREnricher(BaseEnricher):
    expensive = False

    def __init__(self, model: str = "en_core_web_sm", nlp: Any = None) -> None:
        self._model_name = model
        self._nlp: Any = nlp

    def _get_nlp(self) -> Any:
        if self._nlp is None:
            import spacy  # type: ignore[import-not-found]

            self._nlp = spacy.load(self._model_name)
        return self._nlp

    async def enrich(self, chunk: EnrichedChunk) -> EnrichedChunk:
        try:
            nlp = self._get_nlp()
            doc = nlp(chunk.text)
            entities = [
                Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                )
                for ent in doc.ents
            ]
        except Exception:
            entities = []
        return chunk.model_copy(update={"entities": entities})
