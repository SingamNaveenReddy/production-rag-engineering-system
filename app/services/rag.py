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
from app.reranking.base import PassthroughReranker, Reranker
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.sparse import SparseRetriever
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
        sparse_top_k: int = 5,
        hybrid_candidate_count: int = 30,
        rrf_k: int = 60,
        reranker: Reranker | None = None,
        reranker_top_k: int = 5,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._generator = generator
        dense_retriever = DenseRetriever(embedder, store)
        self._retriever = HybridRetriever(
            dense_retriever,
            SparseRetriever(store),
            rrf_k=rrf_k,
            dense_minimum_score=minimum_score,
        )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._dense_top_k = dense_top_k
        self._sparse_top_k = sparse_top_k
        self._hybrid_candidate_count = hybrid_candidate_count
        self._minimum_score = minimum_score
        self._reranker = reranker or PassthroughReranker()
        self._reranker_top_k = reranker_top_k

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
        requested_limit = top_k or self._reranker_top_k
        hybrid_results = self._retriever.retrieve(
            question,
            dense_top_k=max(self._dense_top_k, self._hybrid_candidate_count),
            sparse_top_k=max(self._sparse_top_k, self._hybrid_candidate_count),
            limit=self._hybrid_candidate_count,
        )
        retrieval_finished = perf_counter()
        supported_candidates = [
            result for result in hybrid_results if result.score >= self._minimum_score
        ]
        if not supported_candidates:
            return QueryResult(
                answer=REFUSAL,
                answerable=False,
                citations=[],
                retrieval_metadata={
                    "strategy": "hybrid_rrf_cross_encoder",
                    "candidate_count": len(hybrid_results),
                    "returned_count": 0,
                    "timing_ms": {
                        "retrieval": _milliseconds(started, retrieval_finished),
                        "reranking": 0,
                        "generation": 0,
                    },
                },
                latency_ms=int((perf_counter() - started) * 1000),
            )
        reranked = self._reranker.rerank(question, supported_candidates, requested_limit)
        reranking_finished = perf_counter()
        answer = self._generator.generate(question, [item.chunk for item in reranked])
        generation_finished = perf_counter()
        return QueryResult(
            answer=answer,
            answerable=True,
            citations=citations_from_retrieval(reranked),
            retrieval_metadata={
                "strategy": "hybrid_rrf_cross_encoder",
                "candidate_count": len(hybrid_results),
                "returned_count": len(reranked),
                "hybrid_scores": [item.score for item in supported_candidates],
                "reranker_scores": [item.score for item in reranked],
                "timing_ms": {
                    "retrieval": _milliseconds(started, retrieval_finished),
                    "reranking": _milliseconds(retrieval_finished, reranking_finished),
                    "generation": _milliseconds(reranking_finished, generation_finished),
                },
            },
            latency_ms=_milliseconds(started, generation_finished),
        )

    def list_documents(self) -> list[DocumentSummary]:
        return self._store.list_documents()

    def delete_document(self, document_id: str) -> bool:
        return self._store.delete_document(document_id)


def _milliseconds(started: float, finished: float) -> int:
    return max(0, int((finished - started) * 1000))
