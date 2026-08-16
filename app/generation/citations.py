from __future__ import annotations

from app.models.schemas import Citation, GeneratedAnswer, ScoredChunk


class CitationValidationError(ValueError):
    code = "citation_validation_failed"


class FabricatedCitationError(CitationValidationError):
    code = "fabricated_citation"


class MissingCitationError(CitationValidationError):
    code = "missing_citation"


class InconsistentAnswerabilityError(CitationValidationError):
    code = "inconsistent_answerability"


def validate_citations(
    generated: GeneratedAnswer, retrieved: list[ScoredChunk]
) -> list[Citation]:
    if not generated.answerable:
        if generated.supporting_chunk_ids:
            raise InconsistentAnswerabilityError(
                "An unanswerable response must not contain supporting chunk IDs"
            )
        return []
    if not generated.supporting_chunk_ids:
        raise MissingCitationError("An answerable response requires supporting chunk IDs")

    retrieved_by_id = {item.chunk.metadata.chunk_id: item.chunk for item in retrieved}
    requested_ids = list(dict.fromkeys(generated.supporting_chunk_ids))
    fabricated = [chunk_id for chunk_id in requested_ids if chunk_id not in retrieved_by_id]
    if fabricated:
        raise FabricatedCitationError(
            f"Generated response cited chunks that were not retrieved: {fabricated}"
        )
    return [
        Citation(
            document=retrieved_by_id[chunk_id].metadata.filename,
            page=retrieved_by_id[chunk_id].metadata.page,
            chunk_id=chunk_id,
            supporting_text=retrieved_by_id[chunk_id].text[:400],
        )
        for chunk_id in requested_ids
    ]
