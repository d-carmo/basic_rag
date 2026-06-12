from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag.api.config import Settings
from rag.api.middleware import AuthMiddleware, MetricsMiddleware
from rag.api.routes.admin import router as admin_router
from rag.api.routes.ingest import router as ingest_router
from rag.api.routes.query import router as query_router
from rag.api.routes.upload import router as upload_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg: Settings = app.state.settings

    if app.state.store is None:
        from rag.vector_store.store import QdrantStore, StoreConfig

        store = QdrantStore(
            StoreConfig(
                url=cfg.qdrant_url,
                api_key=cfg.qdrant_api_key,
                collection_name=cfg.collection_name,
                schema_version=cfg.schema_version,
            )
        )
        await store.get_or_create_collection()
        app.state.store = store

    if app.state.embedder is None:
        from rag.embedder.factory import EmbedderConfig, EmbedderFactory

        backend = cfg.embedder_backend
        if backend == "local":
            backend = "sentence_transformer"
        app.state.embedder = EmbedderFactory.create(
            EmbedderConfig(backend=backend, model_name=cfg.embedder_model)
        )

    if app.state.retriever is None:
        from rag.retriever.dense import DenseRetriever
        from rag.retriever.pipeline import RetrievalPipeline

        app.state.retriever = RetrievalPipeline(
            retriever=DenseRetriever(
                store=app.state.store, embedder=app.state.embedder
            )
        )

    if app.state.assembler is None:
        from rag.assembler.pipeline import ContextAssembler

        app.state.assembler = ContextAssembler()

    if app.state.generation_pipeline is None:
        from rag.generation.factory import LLMConfig, LLMFactory
        from rag.generation.pipeline import GenerationPipeline

        api_key = (
            cfg.anthropic_api_key
            if cfg.llm_backend == "anthropic"
            else cfg.llm_api_key
        )
        llm = LLMFactory.create(
            LLMConfig(
                backend=cfg.llm_backend,
                model=cfg.llm_model,
                api_key=api_key,
                base_url=cfg.llm_base_url,
                max_tokens=cfg.llm_max_tokens,
                timeout=float(cfg.llm_timeout),
            )
        )
        app.state.generation_pipeline = GenerationPipeline(llm=llm)

    yield


def create_app(
    settings: Settings | None = None,
    store: Any = None,
    embedder: Any = None,
    retriever: Any = None,
    assembler: Any = None,
    generation_pipeline: Any = None,
    ingestion_log: Any = None,
) -> FastAPI:
    cfg = settings or Settings.from_env()

    app = FastAPI(
        title="RAG Pipeline API",
        version="1.0.0",
        description="Production-grade Retrieval-Augmented Generation API",
        lifespan=lifespan,
    )

    app.state.settings = cfg
    app.state.store = store
    app.state.embedder = embedder
    app.state.retriever = retriever
    app.state.assembler = assembler
    app.state.generation_pipeline = generation_pipeline
    app.state.ingestion_log = ingestion_log

    app.add_middleware(MetricsMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(
        AuthMiddleware,
        api_keys=cfg.api_keys,
        rate_limit_rpm=cfg.rate_limit_rpm,
    )

    app.include_router(admin_router)
    app.include_router(query_router)
    app.include_router(ingest_router)
    app.include_router(upload_router)

    return app


# Module-level app for `uvicorn rag.api.app:app`
app = create_app(settings=Settings.from_env())
