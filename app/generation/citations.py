from __future__ import annotations

from app.models.schemas import Citation, ScoredChunk


def citations_from_retrieval(results: list[ScoredChunk]) -> list[Citation]:
    return [
        Citation(
            document=item.chunk.metadata.filename,
            page=item.chunk.metadata.page,
            chunk_id=item.chunk.metadata.chunk_id,
            supporting_text=item.chunk.text[:400],
        )
        for item in results
    ]

