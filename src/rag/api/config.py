from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: str = "dev-key"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    redis_url: str | None = None
    collection_name: str = "rag"
    schema_version: int = 1
    cors_origins: str = "*"
    rate_limit_rpm: int = 60
    log_level: str = "INFO"
    environment: str = "development"
    embedder_backend: str = "sentence_transformer"
    embedder_model: str = "BAAI/bge-m3"
    anthropic_api_key: str | None = None
    llm_backend: str = "anthropic"
    llm_model: str = "claude-opus-4-8"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_max_tokens: int = 2048
    llm_timeout: int = 600

    @property
    def api_keys(self) -> frozenset[str]:
        return frozenset(k.strip() for k in self.api_key.split(",") if k.strip())

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @classmethod
    def from_env(cls) -> Settings:
        return cls()
