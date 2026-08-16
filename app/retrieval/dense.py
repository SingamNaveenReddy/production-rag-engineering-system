from __future__ import annotations

from app.embeddings.base import Embedder
from app.models.schemas import ScoredChunk
from app.vectorstore.base import VectorStore


class DenseRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        if not query.strip():
            raise ValueError("Query must not be empty")
        return self._store.search(self._embedder.embed_query(query), limit=top_k)

