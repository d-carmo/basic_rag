from __future__ import annotations

from dataclasses import dataclass

from rag.embedder.base import BaseEmbedder


@dataclass
class EmbedderConfig:
    backend: str = "sentence_transformer"
    model_name: str = "BAAI/bge-m3"
    dimensions: int | None = None
    batch_size: int = 32
    input_type: str = "search_document"
    api_key: str | None = None
    cache_backend: str | None = None
    cache_url: str = "redis://localhost:6379"
    cache_db_path: str = ".embeddings_cache.db"
    cache_ttl: int = 86400


class EmbedderFactory:
    @staticmethod
    def create(config: EmbedderConfig) -> BaseEmbedder:
        embedder: BaseEmbedder

        if config.backend == "sentence_transformer":
            from rag.embedder.sentence_transformer import SentenceTransformerEmbedder

            embedder = SentenceTransformerEmbedder(
                model_name=config.model_name,
                batch_size=config.batch_size,
            )
        elif config.backend == "openai":
            from rag.embedder.openai_embedder import OpenAIEmbedder

            embedder = OpenAIEmbedder(
                model_name=config.model_name,
                dimensions=config.dimensions,
                api_key=config.api_key,
                batch_size=config.batch_size,
            )
        elif config.backend == "cohere":
            from rag.embedder.cohere_embedder import CohereEmbedder, InputType
            from typing import get_args

            valid = get_args(InputType)
            it: InputType = config.input_type if config.input_type in valid else "search_document"  # type: ignore[assignment]
            embedder = CohereEmbedder(
                model_name=config.model_name,
                input_type=it,
                api_key=config.api_key,
                batch_size=config.batch_size,
            )
        else:
            raise ValueError(f"Unknown embedder backend: {config.backend!r}")

        if config.cache_backend == "redis":
            from rag.embedder.cache import CachedEmbedder, RedisEmbeddingCache

            return CachedEmbedder(embedder, RedisEmbeddingCache(config.cache_url, config.cache_ttl))
        if config.cache_backend == "sqlite":
            from rag.embedder.cache import CachedEmbedder, SQLiteEmbeddingCache

            return CachedEmbedder(
                embedder, SQLiteEmbeddingCache(config.cache_db_path, config.cache_ttl)
            )

        return embedder
