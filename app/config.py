from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkingConfig(BaseModel):
    chunk_size: int = Field(default=700, ge=50)
    chunk_overlap: int = Field(default=100, ge=0)


class RetrievalConfig(BaseModel):
    dense_top_k: int = Field(default=5, ge=1, le=100)
    minimum_score: float = Field(default=0.25, ge=-1, le=1)


class ProviderConfig(BaseModel):
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen3:4b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


class PromptConfig(BaseModel):
    answer_file: Path = Path("prompts/answer_v1.yaml")


class LoggingConfig(BaseModel):
    level: str = "INFO"


class AppConfig(BaseModel):
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    providers: ProviderConfig = ProviderConfig()
    prompt: PromptConfig = PromptConfig()
    logging: LoggingConfig = LoggingConfig()


class EnvironmentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    config_file: Path = Path("config/default.yaml")
    qdrant_url: str | None = None
    qdrant_collection: str | None = None
    ollama_base_url: str | None = None
    llm_model: str | None = None
    embedding_model: str | None = None
    log_level: str | None = None


def load_config(settings: EnvironmentSettings | None = None) -> AppConfig:
    env = settings or EnvironmentSettings()
    raw: dict[str, Any] = {}
    if env.config_file.exists():
        raw = yaml.safe_load(env.config_file.read_text(encoding="utf-8")) or {}
    config = AppConfig.model_validate(raw)
    overrides = {
        "qdrant_url": env.qdrant_url,
        "qdrant_collection": env.qdrant_collection,
        "ollama_base_url": env.ollama_base_url,
        "llm_model": env.llm_model,
        "embedding_model": env.embedding_model,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(config.providers, name, value)
    if env.log_level is not None:
        config.logging.level = env.log_level
    if config.chunking.chunk_overlap >= config.chunking.chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    return config


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return load_config()

