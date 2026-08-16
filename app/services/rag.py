from __future__ import annotations

from pathlib import Path
from time import perf_counter

from app.embeddings.base import Embedder
from app.generation.base import AnswerGenerator
from app.generation.citations import citations_from_retrieval
from app.ingestion.chunker import chunk_pages
from app.ingestion.loaders import load_document
from app.ingestion.metadata import document_id_for
from app.models.schemas import DocumentSummary, QueryResult
from app.retrieval.dense import DenseRetriever
from app.vectorstore.base import VectorStore

REFUSAL = "I don't have enough evidence in the provided documents to answer that question."


class DuplicateDocumentError(ValueError):
    pass


class RagService:
    def __init__(
        self,
        *,
        embedder: Embedder,
        store: VectorStore,
        generator: AnswerGenerator,
        chunk_size: int,
        chunk_overlap: int,
        dense_top_k: int,
        minimum_score: float,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._generator = generator
        self._retriever = DenseRetriever(embedder, store)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._dense_top_k = dense_top_k
        self._minimum_score = minimum_score

    def ingest(self, path: Path) -> DocumentSummary:
        document_id = document_id_for(path)
        if self._store.contains_document(document_id):
            raise DuplicateDocumentError(f"Document already ingested: {path.name}")
        pages = load_document(path)
        chunks = chunk_pages(
            pages,
            document_id=document_id,
            path=path,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        vectors = self._embedder.embed_documents([chunk.text for chunk in chunks])
        self._store.upsert(chunks, vectors)
        return DocumentSummary(
            document_id=document_id, filename=path.name, chunk_count=len(chunks)
        )

    def query(self, question: str, top_k: int | None = None) -> QueryResult:
        started = perf_counter()
        results = self._retriever.retrieve(question, top_k or self._dense_top_k)
        supported = [result for result in results if result.score >= self._minimum_score]
        if not supported:
            return QueryResult(
                answer=REFUSAL,
                answerable=False,
                citations=[],
                retrieval_metadata={"strategy": "dense", "candidate_count": len(results)},
                latency_ms=int((perf_counter() - started) * 1000),
            )
        answer = self._generator.generate(question, [item.chunk for item in supported])
        return QueryResult(
            answer=answer,
            answerable=True,
            citations=citations_from_retrieval(supported),
            retrieval_metadata={
                "strategy": "dense",
                "candidate_count": len(results),
                "scores": [item.score for item in supported],
            },
            latency_ms=int((perf_counter() - started) * 1000),
        )

    def list_documents(self) -> list[DocumentSummary]:
        return self._store.list_documents()

    def delete_document(self, document_id: str) -> bool:
        return self._store.delete_document(document_id)

