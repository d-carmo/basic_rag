from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

from rag.assembler.base import AssembledContext
from rag.generation.base import BaseLLM, Message
from rag.generation.faithfulness import FaithfulnessChecker
from rag.generation.guard import PromptInjectionGuard
from rag.generation.prompts import SystemPromptBuilder, UserPromptBuilder

_INJECTION_RESPONSE = (
    "Your query was flagged for potential prompt injection and cannot be processed."
)


@dataclass
class GenerationConfig:
    enable_guard: bool = True
    enable_faithfulness_check: bool = False
    faithfulness_threshold: float = 0.5


@dataclass
class GenerationResult:
    answer: str
    faithfulness_score: float | None = None
    was_guarded: bool = False


class GenerationPipeline:
    """Orchestrates: guard → prompt build → LLM complete/stream → faithfulness check."""

    def __init__(
        self,
        llm: BaseLLM,
        system_builder: SystemPromptBuilder | None = None,
        user_builder: UserPromptBuilder | None = None,
        guard: PromptInjectionGuard | None = None,
        faithfulness_checker: FaithfulnessChecker | None = None,
        config: GenerationConfig | None = None,
    ) -> None:
        self._llm = llm
        self._system = system_builder or SystemPromptBuilder()
        self._user = user_builder or UserPromptBuilder()
        self._guard = guard or PromptInjectionGuard()
        self._faithfulness = faithfulness_checker or FaithfulnessChecker()
        self._config = config or GenerationConfig()

    def _build_messages(self, query: str, context: AssembledContext) -> list[Message]:
        return [
            Message(role="system", content=self._system.build()),
            Message(role="user", content=self._user.build(query, context.chunks, context.citation_map)),
        ]

    async def generate(self, query: str, context: AssembledContext) -> GenerationResult:
        if self._config.enable_guard:
            try:
                query = self._guard.check(query)
            except ValueError:
                return GenerationResult(answer=_INJECTION_RESPONSE, was_guarded=True)

        answer = await self._llm.complete(self._build_messages(query, context))

        faithfulness_score: float | None = None
        if self._config.enable_faithfulness_check and context.chunks:
            faithfulness_score = self._faithfulness.score(answer, context.chunks)

        return GenerationResult(answer=answer, faithfulness_score=faithfulness_score)

    async def stream(self, query: str, context: AssembledContext) -> AsyncIterator[str]:  # type: ignore[return]
        if self._config.enable_guard:
            try:
                query = self._guard.check(query)
            except ValueError:
                yield _INJECTION_RESPONSE
                return

        async for chunk in self._llm.stream(self._build_messages(query, context)):
            yield chunk
