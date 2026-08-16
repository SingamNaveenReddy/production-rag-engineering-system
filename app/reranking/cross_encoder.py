from __future__ import annotations

from sentence_transformers import CrossEncoder

from app.models.schemas import ScoredChunk


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not query.strip():
            raise ValueError("Query must not be empty")
        if not candidates:
            return []
        pairs = [(query, candidate.chunk.text) for candidate in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)
        rescored = [
            ScoredChunk(chunk=candidate.chunk, score=float(score))
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        return sorted(
            rescored,
            key=lambda item: (-item.score, item.chunk.metadata.chunk_id),
        )[:top_k]

