from pathlib import Path

import pytest

from app.services.rag import DuplicateDocumentError, RagService
from tests.fakes import FakeEmbedder, FakeGenerator, MemoryVectorStore


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


def test_query_returns_grounded_answer_and_programmatic_citation(tmp_path: Path) -> None:
    source = tmp_path / "security.txt"
    source.write_text("Authentication requires a hardware security key.", encoding="utf-8")
    service = make_service()
    summary = service.ingest(source)
    result = service.query("What authentication is required?")
    assert result.answerable is True
    assert result.citations[0].document == "security.txt"
    assert result.citations[0].chunk_id.startswith(summary.document_id)
    assert result.retrieval_metadata["strategy"] == "hybrid_rrf"


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
