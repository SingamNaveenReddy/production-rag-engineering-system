from __future__ import annotations

import re
from collections import Counter
from math import sqrt

from app.models.schemas import DocumentChunk, DocumentSummary, GeneratedAnswer, ScoredChunk


class FakeEmbedder:
    @property
    def dimension(self) -> int:
        return 3

    def _embed(self, text: str) -> list[float]:
        tokens = re.findall(r"[a-z]+", text.lower())
        values = [float(tokens.count(word)) for word in ("authentication", "logging", "cat")]
        norm = sqrt(sum(value * value for value in values)) or 1
        return [value / norm for value in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class MemoryVectorStore:
    def __init__(self) -> None:
        self.entries: list[tuple[DocumentChunk, list[float]]] = []

    def upsert(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        self.entries.extend(zip(chunks, vectors, strict=True))

    def search(self, vector: list[float], limit: int) -> list[ScoredChunk]:
        scored = [
            ScoredChunk(chunk=chunk, score=sum(a * b for a, b in zip(vector, stored, strict=True)))
            for chunk, stored in self.entries
        ]
        return sorted(scored, key=lambda item: item.score, reverse=True)[:limit]

    def list_documents(self) -> list[DocumentSummary]:
        counts = Counter((c.metadata.document_id, c.metadata.filename) for c, _ in self.entries)
        return [
            DocumentSummary(document_id=key[0], filename=key[1], chunk_count=count)
            for key, count in counts.items()
        ]

    def list_chunks(self) -> list[DocumentChunk]:
        return [chunk for chunk, _ in self.entries]

    def delete_document(self, document_id: str) -> bool:
        before = len(self.entries)
        self.entries = [
            entry for entry in self.entries if entry[0].metadata.document_id != document_id
        ]
        return len(self.entries) != before

    def contains_document(self, document_id: str) -> bool:
        return any(chunk.metadata.document_id == document_id for chunk, _ in self.entries)


class FakeGenerator:
    def generate(self, question: str, context: list[DocumentChunk]) -> GeneratedAnswer:
        return GeneratedAnswer(
            answer=f"Grounded answer: {context[0].text}",
            answerable=True,
            supporting_chunk_ids=[context[0].metadata.chunk_id],
        )


class FakeReranker:
    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        rescored = [
            ScoredChunk(
                chunk=candidate.chunk,
                score=1.0 if "fido2" in candidate.chunk.text.lower() else 0.0,
            )
            for candidate in candidates
        ]
        return sorted(rescored, key=lambda item: item.score, reverse=True)[:top_k]
