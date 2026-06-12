"""Tests for LLMFactory."""

import pytest

from rag.generation.anthropic_llm import AnthropicLLM
from rag.generation.factory import LLMConfig, LLMFactory
from rag.generation.openai_llm import OpenAILLM, _OPENAI_DEFAULT_URL


def test_factory_creates_anthropic() -> None:
    cfg = LLMConfig(backend="anthropic", model="claude-opus-4-8")
    llm = LLMFactory.create(cfg)
    assert isinstance(llm, AnthropicLLM)
    assert llm._model == "claude-opus-4-8"


def test_factory_creates_openai() -> None:
    cfg = LLMConfig(backend="openai", model="gpt-4o", api_key="sk-test")
    llm = LLMFactory.create(cfg)
    assert isinstance(llm, OpenAILLM)
    assert llm._model == "gpt-4o"


def test_factory_openai_default_base_url() -> None:
    cfg = LLMConfig(backend="openai")
    llm = LLMFactory.create(cfg)
    assert isinstance(llm, OpenAILLM)
    assert llm._base_url == _OPENAI_DEFAULT_URL


def test_factory_local_uses_custom_base_url() -> None:
    cfg = LLMConfig(backend="local", model="llama3", base_url="http://localhost:11434/v1")
    llm = LLMFactory.create(cfg)
    assert isinstance(llm, OpenAILLM)
    assert llm._base_url == "http://localhost:11434/v1"
    assert llm._model == "llama3"


def test_factory_local_lm_studio_url() -> None:
    cfg = LLMConfig(backend="local", model="mistral", base_url="http://localhost:1234/v1")
    llm = LLMFactory.create(cfg)
    assert isinstance(llm, OpenAILLM)
    assert llm._base_url == "http://localhost:1234/v1"


def test_factory_local_no_api_key_allowed() -> None:
    cfg = LLMConfig(backend="local", base_url="http://localhost:11434/v1", api_key=None)
    llm = LLMFactory.create(cfg)
    assert isinstance(llm, OpenAILLM)


def test_factory_max_tokens_passed() -> None:
    cfg = LLMConfig(backend="anthropic", max_tokens=4096)
    llm = LLMFactory.create(cfg)
    assert isinstance(llm, AnthropicLLM)
    assert llm._max_tokens == 4096


def test_factory_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="unknown"):
        LLMFactory.create(LLMConfig(backend="unknown"))
