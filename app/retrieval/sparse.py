from __future__ import annotations

import math
import re
from collections import Counter

from app.models.schemas import ScoredChunk
from app.vectorstore.base import VectorStore

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


class SparseRetriever:
    """BM25 lexical retrieval over the canonical chunks stored in Qdrant."""

    def __init__(self, store: VectorStore, *, k1: float = 1.5, b: float = 0.75) -> None:
        self._store = store
        self._k1 = k1
        self._b = b

    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        query_terms = Counter(tokenize(query))
        if not query_terms:
            raise ValueError("Query must not be empty")
        chunks = self._store.list_chunks()
        if not chunks:
            return []

        documents = [tokenize(chunk.text) for chunk in chunks]
        document_frequencies = Counter(
            term for document in documents for term in set(document) if term in query_terms
        )
        average_length = sum(map(len, documents)) / len(documents)
        scored: list[ScoredChunk] = []
        for chunk, document in zip(chunks, documents, strict=True):
            frequencies = Counter(document)
            score = sum(
                self._term_score(
                    frequency=frequencies[term],
                    document_frequency=document_frequencies[term],
                    document_length=len(document),
                    average_length=average_length,
                    document_count=len(documents),
                )
                * query_frequency
                for term, query_frequency in query_terms.items()
                if frequencies[term]
            )
            if score > 0:
                scored.append(ScoredChunk(chunk=chunk, score=score))
        return sorted(
            scored,
            key=lambda item: (-item.score, item.chunk.metadata.chunk_id),
        )[:top_k]

    def _term_score(
        self,
        *,
        frequency: int,
        document_frequency: int,
        document_length: int,
        average_length: float,
        document_count: int,
    ) -> float:
        inverse_document_frequency = math.log(
            1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        normalization = frequency + self._k1 * (
            1 - self._b + self._b * document_length / average_length
        )
        return inverse_document_frequency * frequency * (self._k1 + 1) / normalization
