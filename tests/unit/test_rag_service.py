from pathlib import Path

import pytest

from app.models.schemas import DocumentChunk, GeneratedAnswer
from app.services.rag import DuplicateDocumentError, RagService
from tests.fakes import FakeEmbedder, FakeGenerator, FakeReranker, MemoryVectorStore


def make_service(minimum_score: float = 0.2) -> RagService:
    return RagService(
        embedder=FakeEmbedder(),
        store=MemoryVectorStore(),
        generator=FakeGenerator(),
        chunk_size=20,
        chunk_overlap=5,
        dense_top_k=5,
        minimum_score=minimum_score,
    )


class FabricatingGenerator:
    def generate(self, question: str, context: list[DocumentChunk]) -> GeneratedAnswer:
        return GeneratedAnswer(
            answer="A fabricated answer.",
            answerable=True,
            supporting_chunk_ids=["invented-document-p999-c9999"],
        )


class RefusingGenerator:
    def generate(self, question: str, context: list[DocumentChunk]) -> GeneratedAnswer:
        return GeneratedAnswer(
            answer="There is not enough evidence.",
            answerable=False,
            supporting_chunk_ids=[],
        )


def test_query_returns_grounded_answer_and_programmatic_citation(tmp_path: Path) -> None:
    source = tmp_path / "security.txt"
    source.write_text("Authentication requires a hardware security key.", encoding="utf-8")
    service = make_service()
    summary = service.ingest(source)
    result = service.query("What authentication is required?")
    assert result.answerable is True
    assert result.citations[0].document == "security.txt"
    assert result.citations[0].chunk_id.startswith(summary.document_id)
    assert result.retrieval_metadata["strategy"] == "hybrid_rrf_cross_encoder"
    assert result.retrieval_metadata["citation_validation"] == {
        "status": "passed",
        "validated_count": 1,
    }
    assert result.retrieval_metadata["retrieved_sources"][0]["document"] == "security.txt"


def test_unsupported_question_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "security.txt"
    source.write_text("Authentication requires a hardware security key.", encoding="utf-8")
    service = make_service()
    service.ingest(source)
    result = service.query("Tell me about cats")
    assert result.answerable is False
    assert result.citations == []


def test_duplicate_document_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "security.txt"
    source.write_text("Authentication policy.", encoding="utf-8")
    service = make_service()
    service.ingest(source)
    with pytest.raises(DuplicateDocumentError):
        service.ingest(source)


def test_fabricated_generator_citation_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "security.txt"
    source.write_text("Authentication requires a hardware key.", encoding="utf-8")
    service = RagService(
        embedder=FakeEmbedder(),
        store=MemoryVectorStore(),
        generator=FabricatingGenerator(),
        chunk_size=20,
        chunk_overlap=5,
        dense_top_k=5,
        minimum_score=0.2,
    )
    service.ingest(source)

    result = service.query("What authentication is required?")

    assert result.answerable is False
    assert result.citations == []
    assert result.retrieval_metadata["citation_validation"]["reason"] == (
        "fabricated_citation"
    )


def test_generator_can_explicitly_refuse_supported_candidates(tmp_path: Path) -> None:
    source = tmp_path / "security.txt"
    source.write_text("Authentication requires a hardware key.", encoding="utf-8")
    service = RagService(
        embedder=FakeEmbedder(),
        store=MemoryVectorStore(),
        generator=RefusingGenerator(),
        chunk_size=20,
        chunk_overlap=5,
        dense_top_k=5,
        minimum_score=0.2,
    )
    service.ingest(source)

    result = service.query("What authentication is required?")

    assert result.answerable is False
    assert result.citations == []
    assert result.retrieval_metadata["answerability"]["status"] == "refused_by_generator"


def test_reranking_changes_order_and_records_latency(tmp_path: Path) -> None:
    store = MemoryVectorStore()
    service = RagService(
        embedder=FakeEmbedder(),
        store=store,
        generator=FakeGenerator(),
        reranker=FakeReranker(),
        chunk_size=20,
        chunk_overlap=5,
        dense_top_k=5,
        sparse_top_k=5,
        hybrid_candidate_count=5,
        reranker_top_k=1,
        minimum_score=0.2,
    )
    distractor = tmp_path / "glossary.txt"
    distractor.write_text(
        "Authentication control mandatory policy glossary.", encoding="utf-8"
    )
    relevant = tmp_path / "security.txt"
    relevant.write_text(
        "Authentication requires a FIDO2 hardware key for employee login.",
        encoding="utf-8",
    )
    service.ingest(distractor)
    service.ingest(relevant)

    result = service.query("Which authentication control is mandatory?", top_k=1)

    assert result.citations[0].document == "security.txt"
    assert result.retrieval_metadata["returned_count"] == 1
    assert set(result.retrieval_metadata["timing_ms"]) == {
        "retrieval",
        "reranking",
        "generation",
    }
    assert all(value >= 0 for value in result.retrieval_metadata["timing_ms"].values())
