from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

from app.config import AppConfig
from app.container import build_rag_service
from app.generation.base import AnswerGenerator
from app.models.schemas import DocumentChunk, GeneratedAnswer
from app.services.rag import REFUSAL, RagService
from app.vectorstore.qdrant_store import QdrantVectorStore
from evaluation.metrics import content_tokens

SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


class HashingEmbedder:
    """Stable local embedding profile for deterministic evaluation and CI."""

    def __init__(self, dimension: int = 256) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in content_tokens(text):
            digest = hashlib.sha256(token.encode()).digest()
            vector[int.from_bytes(digest[:4]) % self._dimension] += 1
        norm = math.sqrt(sum(value * value for value in vector)) or 1
        return [value / norm for value in vector]


class ExtractiveGenerator(AnswerGenerator):
    """Select the best-supported sentence without an external generation service."""

    def generate(self, question: str, context: list[DocumentChunk]) -> GeneratedAnswer:
        query_tokens = set(content_tokens(question))
        best: tuple[int, str, str] | None = None
        for chunk in context:
            for sentence in SENTENCE_PATTERN.split(chunk.text.replace("\n", " ")):
                score = len(query_tokens & set(content_tokens(sentence)))
                candidate = (score, sentence.strip(), chunk.metadata.chunk_id)
                if best is None or candidate[0] > best[0]:
                    best = candidate
        if best is None or best[0] == 0:
            return GeneratedAnswer(
                answer=REFUSAL, answerable=False, supporting_chunk_ids=[]
            )
        return GeneratedAnswer(
            answer=best[1], answerable=True, supporting_chunk_ids=[best[2]]
        )


def build_deterministic_engine(sample_directory: Path) -> RagService:
    embedder = HashingEmbedder()
    store = QdrantVectorStore(":memory:", "golden-evaluation", embedder.dimension)
    service = RagService(
        embedder=embedder,
        store=store,
        generator=ExtractiveGenerator(),
        chunk_size=50,
        chunk_overlap=10,
        dense_top_k=5,
        sparse_top_k=5,
        hybrid_candidate_count=10,
        reranker_top_k=3,
        minimum_score=0.05,
    )
    for path in sorted(sample_directory.glob("*.md")):
        service.ingest(path)
    return service


def build_production_engine(config: AppConfig) -> RagService:
    return build_rag_service(config)

