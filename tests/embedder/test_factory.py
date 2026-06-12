"""Tests for EmbedderFactory."""

import pytest

from rag.embedder.cache import CachedEmbedder
from rag.embedder.cohere_embedder import CohereEmbedder
from rag.embedder.factory import EmbedderConfig, EmbedderFactory
from rag.embedder.openai_embedder import OpenAIEmbedder
from rag.embedder.sentence_transformer import SentenceTransformerEmbedder


def test_factory_creates_sentence_transformer() -> None:
    config = EmbedderConfig(backend="sentence_transformer")
    embedder = EmbedderFactory.create(config)
    assert isinstance(embedder, SentenceTransformerEmbedder)


def test_factory_creates_openai_embedder() -> None:
    config = EmbedderConfig(backend="openai", api_key="fake")
    embedder = EmbedderFactory.create(config)
    assert isinstance(embedder, OpenAIEmbedder)


def test_factory_creates_cohere_embedder() -> None:
    config = EmbedderConfig(backend="cohere", api_key="fake")
    embedder = EmbedderFactory.create(config)
    assert isinstance(embedder, CohereEmbedder)


def test_factory_unknown_backend_raises() -> None:
    config = EmbedderConfig(backend="unknown_backend")
    with pytest.raises(ValueError, match="unknown_backend"):
        EmbedderFactory.create(config)


def test_factory_sqlite_cache_wraps_embedder(tmp_path: pytest.TempPathFactory) -> None:
    config = EmbedderConfig(
        backend="sentence_transformer",
        cache_backend="sqlite",
        cache_db_path=str(tmp_path / "emb.db"),  # type: ignore[operator]
    )
    embedder = EmbedderFactory.create(config)
    assert isinstance(embedder, CachedEmbedder)
    assert isinstance(embedder._embedder, SentenceTransformerEmbedder)


def test_factory_no_cache_returns_raw_embedder() -> None:
    config = EmbedderConfig(backend="sentence_transformer", cache_backend=None)
    embedder = EmbedderFactory.create(config)
    assert isinstance(embedder, SentenceTransformerEmbedder)


def test_factory_cohere_invalid_input_type_defaults_to_search_document() -> None:
    config = EmbedderConfig(backend="cohere", api_key="k", input_type="invalid_type")
    embedder = EmbedderFactory.create(config)
    assert isinstance(embedder, CohereEmbedder)
    assert embedder._input_type == "search_document"


def test_factory_openai_passes_dimensions() -> None:
    config = EmbedderConfig(backend="openai", api_key="k", dimensions=512)
    embedder = EmbedderFactory.create(config)
    assert isinstance(embedder, OpenAIEmbedder)
    assert embedder._dimensions == 512
