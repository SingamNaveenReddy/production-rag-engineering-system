from __future__ import annotations

from collections import defaultdict

from app.models.schemas import DocumentChunk, ScoredChunk
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import SparseRetriever


class HybridRetriever:
    """Combine dense and sparse ranks using normalized reciprocal rank fusion."""

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        *,
        rrf_k: int = 60,
        dense_minimum_score: float = 0.0,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._rrf_k = rrf_k
        self._dense_minimum_score = dense_minimum_score

    def retrieve(
        self,
        query: str,
        *,
        dense_top_k: int,
        sparse_top_k: int,
        limit: int,
    ) -> list[ScoredChunk]:
        dense_results = [
            item
            for item in self._dense.retrieve(query, dense_top_k)
            if item.score >= self._dense_minimum_score
        ]
        sparse_results = self._sparse.retrieve(query, sparse_top_k)
        scores: defaultdict[str, float] = defaultdict(float)
        chunks: dict[str, DocumentChunk] = {}
        for ranking in (dense_results, sparse_results):
            for rank, item in enumerate(ranking, start=1):
                chunk_id = item.chunk.metadata.chunk_id
                chunks[chunk_id] = item.chunk
                scores[chunk_id] += 1 / (self._rrf_k + rank)

        maximum_score = 2 / (self._rrf_k + 1)
        fused = [
            ScoredChunk(chunk=chunks[chunk_id], score=score / maximum_score)
            for chunk_id, score in scores.items()
        ]
        return sorted(
            fused,
            key=lambda item: (-item.score, item.chunk.metadata.chunk_id),
        )[:limit]

