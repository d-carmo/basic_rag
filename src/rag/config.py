from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class RagConfig:
    """Central config loaded from environment variables."""

    # Vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection_name: str = "rag"
    schema_version: int = 1

    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # LLM
    llm_backend: str = "anthropic"
    llm_model: str = "claude-opus-4-8"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_max_tokens: int = 2048

    # Retrieval
    retrieval_top_k: int = 8
    sparse_weight: float = 0.3
    dense_weight: float = 0.7

    # Assembler
    max_context_tokens: int = 4096

    # API
    api_keys: frozenset[str] = field(default_factory=lambda: frozenset({"dev-key"}))
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    rate_limit_rpm: int = 60
    log_level: str = "INFO"
    environment: str = "development"

    # Observability
    otlp_endpoint: str | None = None
    metrics_port: int = 9090

    @classmethod
    def from_env(cls) -> RagConfig:
        raw_keys = os.environ.get("API_KEYS", "dev-key")
        keys = frozenset(k.strip() for k in raw_keys.split(",") if k.strip())
        raw_origins = os.environ.get("CORS_ORIGINS", "*")
        origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
        return cls(
            qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
            collection_name=os.environ.get("COLLECTION_NAME", "rag"),
            schema_version=int(os.environ.get("SCHEMA_VERSION", "1")),
            embedding_model=os.environ.get(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            embedding_dim=int(os.environ.get("EMBEDDING_DIM", "384")),
            llm_backend=os.environ.get("LLM_BACKEND", "anthropic"),
            llm_model=os.environ.get("LLM_MODEL", "claude-opus-4-8"),
            llm_api_key=os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or None,
            llm_base_url=os.environ.get("LLM_BASE_URL") or None,
            llm_max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "2048")),
            retrieval_top_k=int(os.environ.get("RETRIEVAL_TOP_K", "8")),
            sparse_weight=float(os.environ.get("SPARSE_WEIGHT", "0.3")),
            dense_weight=float(os.environ.get("DENSE_WEIGHT", "0.7")),
            max_context_tokens=int(os.environ.get("MAX_CONTEXT_TOKENS", "4096")),
            api_keys=keys,
            cors_origins=origins,
            rate_limit_rpm=int(os.environ.get("RATE_LIMIT_RPM", "60")),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            environment=os.environ.get("ENVIRONMENT", "development"),
            otlp_endpoint=os.environ.get("OTLP_ENDPOINT") or None,
            metrics_port=int(os.environ.get("METRICS_PORT", "9090")),
        )
