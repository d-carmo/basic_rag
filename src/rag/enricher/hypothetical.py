from __future__ import annotations

from typing import Any

from rag.enricher.base import BaseEnricher, EnrichedChunk

_PROMPT = """\
Given the following text passage, generate {n} concise questions that the passage could answer.
Output only the questions, one per line, with no numbering or bullet points.

Passage:
{text}
"""


class HypotheticalQuestionsEnricher(BaseEnricher):
    expensive = True

    def __init__(
        self,
        n_questions: int = 5,
        model: str = "claude-opus-4-8",
        max_tokens: int = 256,
        client: Any = None,
    ) -> None:
        self._n = n_questions
        self._model = model
        self._max_tokens = max_tokens
        self._client: Any = client

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic()
        return self._client

    async def enrich(self, chunk: EnrichedChunk) -> EnrichedChunk:
        try:
            client = self._get_client()
            prompt = _PROMPT.format(n=self._n, text=chunk.text[:2000])
            message = await client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = next(
                (b.text for b in message.content if b.type == "text"),
                "",
            )
            questions = [q.strip() for q in raw.splitlines() if q.strip() and "?" in q]
            questions = questions[: self._n]
        except Exception:
            questions = []
        return chunk.model_copy(update={"hypothetical_questions": questions})
