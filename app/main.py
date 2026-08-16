from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import documents, health, query
from app.config import AppConfig, get_config
from app.services.rag import RagService


def create_app(service: RagService | None = None, config: AppConfig | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.config = config or get_config()
        app.state.rag_service = service
        yield

    app = FastAPI(
        title="Production RAG Engineering System",
        version="0.1.0",
        description="Phase 1: dense retrieval with grounded local generation and citations.",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(query.router)
    return app


app = create_app()
