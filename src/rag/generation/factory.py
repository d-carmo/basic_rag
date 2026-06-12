from __future__ import annotations

from dataclasses import dataclass

from rag.generation.anthropic_llm import AnthropicLLM
from rag.generation.base import BaseLLM
from rag.generation.openai_llm import OpenAILLM, _OPENAI_DEFAULT_URL


@dataclass
class LLMConfig:
    backend: str = "anthropic"     # "anthropic" | "openai" | "local"
    model: str = "claude-opus-4-8"
    api_key: str | None = None
    base_url: str | None = None    # required for "local"; optional override for "openai"
    max_tokens: int = 2048
    timeout: float = 600.0


class LLMFactory:
    @staticmethod
    def create(config: LLMConfig) -> BaseLLM:
        if config.backend == "anthropic":
            return AnthropicLLM(model=config.model, max_tokens=config.max_tokens, api_key=config.api_key)
        if config.backend in ("openai", "local"):
            return OpenAILLM(
                model=config.model,
                base_url=config.base_url or _OPENAI_DEFAULT_URL,
                api_key=config.api_key or "",
                max_tokens=config.max_tokens,
                timeout=config.timeout,
            )
        raise ValueError(
            f"Unknown LLM backend: {config.backend!r}. Use 'anthropic', 'openai', or 'local'."
        )
