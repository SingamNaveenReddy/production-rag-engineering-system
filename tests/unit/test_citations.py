import pytest

from app.generation.citations import (
    FabricatedCitationError,
    InconsistentAnswerabilityError,
    MissingCitationError,
    validate_citations,
)
from app.models.schemas import (
    DocumentChunk,
    GeneratedAnswer,
    ScoredChunk,
    SourceMetadata,
)


def _retrieved_chunk() -> ScoredChunk:
    return ScoredChunk(
        chunk=DocumentChunk(
            text="Authentication requires a FIDO2 hardware key.",
            metadata=SourceMetadata(
                document_id="doc-security",
                filename="security.pdf",
                page=17,
                section="Authentication",
                chunk_id="doc-security-p17-c0001",
                original_source="security.pdf",
            ),
        ),
        score=8.5,
    )


def test_citation_metadata_is_derived_from_retrieved_chunk() -> None:
    retrieved = _retrieved_chunk()
    generated = GeneratedAnswer(
        answer="A FIDO2 hardware key is required.",
        answerable=True,
        supporting_chunk_ids=[retrieved.chunk.metadata.chunk_id],
    )

    citations = validate_citations(generated, [retrieved])

    assert citations[0].document == "security.pdf"
    assert citations[0].page == 17
    assert citations[0].supporting_text == retrieved.chunk.text


def test_fabricated_chunk_id_is_rejected() -> None:
    generated = GeneratedAnswer(
        answer="Invented answer.",
        answerable=True,
        supporting_chunk_ids=["invented-p99-c9999"],
    )
    with pytest.raises(FabricatedCitationError):
        validate_citations(generated, [_retrieved_chunk()])


def test_answerable_response_requires_at_least_one_citation() -> None:
    generated = GeneratedAnswer(
        answer="Unsupported answer.", answerable=True, supporting_chunk_ids=[]
    )
    with pytest.raises(MissingCitationError):
        validate_citations(generated, [_retrieved_chunk()])


def test_unanswerable_response_cannot_claim_support() -> None:
    generated = GeneratedAnswer(
        answer="Not enough evidence.",
        answerable=False,
        supporting_chunk_ids=["doc-security-p17-c0001"],
    )
    with pytest.raises(InconsistentAnswerabilityError):
        validate_citations(generated, [_retrieved_chunk()])
