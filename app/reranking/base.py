from __future__ import annotations

from typing import Protocol

from app.models.schemas import ScoredChunk


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]: ...


class PassthroughReranker:
    """Preserve the hybrid order for injected/test configurations without a model."""

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        return candidates[:top_k]
