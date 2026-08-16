from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from app.ingestion.chunker import chunk_pages
from app.ingestion.loaders import DocumentPage
from app.models.schemas import DocumentChunk, DocumentSummary, ScoredChunk
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.sparse import SparseRetriever


class IdentifierBlindEmbedder:
    """Deterministic dense baseline that intentionally has no identifier vocabulary."""

    @property
    def dimension(self) -> int:
        return 1

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0]


class BenchmarkStore:
    def __init__(self) -> None:
        self.entries: list[tuple[DocumentChunk, list[float]]] = []

    def upsert(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        self.entries.extend(zip(chunks, vectors, strict=True))

    def search(self, vector: list[float], limit: int) -> list[ScoredChunk]:
        scored = [
            ScoredChunk(
                chunk=chunk,
                score=sum(a * b for a, b in zip(vector, stored, strict=True)),
            )
            for chunk, stored in self.entries
        ]
        return scored[:limit]

    def list_chunks(self) -> list[DocumentChunk]:
        return [chunk for chunk, _ in self.entries]

    def list_documents(self) -> list[DocumentSummary]:
        counts = Counter((c.metadata.document_id, c.metadata.filename) for c, _ in self.entries)
        return [
            DocumentSummary(document_id=key[0], filename=key[1], chunk_count=count)
            for key, count in counts.items()
        ]

    def delete_document(self, document_id: str) -> bool:
        before = len(self.entries)
        self.entries = [
            entry for entry in self.entries if entry[0].metadata.document_id != document_id
        ]
        return len(self.entries) != before

    def contains_document(self, document_id: str) -> bool:
        return any(chunk.metadata.document_id == document_id for chunk, _ in self.entries)


def build_corpus() -> tuple[BenchmarkStore, IdentifierBlindEmbedder, list[dict[str, str]]]:
    store = BenchmarkStore()
    embedder = IdentifierBlindEmbedder()
    examples = [
        ("alpha.txt", "Deployment identifier ALPHA-ZZ-101 targets the gateway."),
        ("cve.txt", "Patch guidance for CVE-2026-1234 applies immediately."),
        ("model.txt", "GPT-5.6 configuration details appear in section 14.3."),
    ]
    cases: list[dict[str, str]] = []
    for index, (filename, text) in enumerate(examples, start=1):
        chunks = chunk_pages(
            [DocumentPage(text=text, page=None)],
            document_id=f"benchmark-{index}",
            path=Path(filename),
            chunk_size=50,
            chunk_overlap=5,
        )
        store.upsert(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))
        query = ("ALPHA-ZZ-101", "CVE-2026-1234", "GPT-5.6")[index - 1]
        cases.append({"query": query, "expected_chunk_id": chunks[0].metadata.chunk_id})
    return store, embedder, cases


def main() -> None:
    store, embedder, cases = build_corpus()
    dense = DenseRetriever(embedder, store)
    hybrid = HybridRetriever(
        dense,
        SparseRetriever(store),
        dense_minimum_score=0.01,
    )
    details: list[dict[str, object]] = []
    dense_hits = 0
    hybrid_hits = 0
    for case in cases:
        dense_id = dense.retrieve(case["query"], top_k=1)[0].chunk.metadata.chunk_id
        hybrid_id = hybrid.retrieve(
            case["query"], dense_top_k=3, sparse_top_k=3, limit=1
        )[0].chunk.metadata.chunk_id
        dense_hit = dense_id == case["expected_chunk_id"]
        hybrid_hit = hybrid_id == case["expected_chunk_id"]
        dense_hits += dense_hit
        hybrid_hits += hybrid_hit
        details.append(
            {
                **case,
                "dense_chunk_id": dense_id,
                "hybrid_chunk_id": hybrid_id,
                "dense_hit": dense_hit,
                "hybrid_hit": hybrid_hit,
            }
        )

    count = len(cases)
    result = {
        "benchmark": "phase2-exact-identifier-retrieval",
        "query_count": count,
        "metric": "recall@1",
        "dense_recall_at_1": dense_hits / count,
        "hybrid_recall_at_1": hybrid_hits / count,
        "absolute_improvement": (hybrid_hits - dense_hits) / count,
        "cases": details,
    }
    output_directory = Path("evaluation/results")
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "phase2_retrieval.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    markdown = (
        "# Phase 2 exact-identifier retrieval benchmark\n\n"
        f"Queries: {count}\n\n"
        f"- Dense recall@1: {result['dense_recall_at_1']:.3f}\n"
        f"- Hybrid recall@1: {result['hybrid_recall_at_1']:.3f}\n"
        f"- Absolute improvement: {result['absolute_improvement']:.3f}\n\n"
        "This deterministic benchmark isolates identifiers that the dense baseline does not "
        "represent. It is a regression fixture, not a production-quality corpus benchmark.\n"
    )
    (output_directory / "phase2_retrieval.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
