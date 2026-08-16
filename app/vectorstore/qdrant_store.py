from __future__ import annotations

from collections import Counter
from hashlib import md5
from uuid import UUID

from qdrant_client import QdrantClient, models

from app.models.schemas import DocumentChunk, DocumentSummary, ScoredChunk


def _point_id(chunk_id: str) -> str:
    digest = md5(chunk_id.encode(), usedforsecurity=False).digest()
    return str(UUID(bytes=digest))


class QdrantVectorStore:
    def __init__(self, url: str, collection: str, vector_size: int) -> None:
        self._client = (
            QdrantClient(location=":memory:") if url == ":memory:" else QdrantClient(url=url)
        )
        self._collection = collection
        self._ensure_collection(vector_size)

    def _ensure_collection(self, vector_size: int) -> None:
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )

    def upsert(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one embedding")
        points = [
            models.PointStruct(
                id=_point_id(chunk.metadata.chunk_id),
                vector=vector,
                payload=chunk.model_dump(mode="json"),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self._collection, points=points, wait=True)

    def search(self, vector: list[float], limit: int) -> list[ScoredChunk]:
        result = self._client.query_points(
            collection_name=self._collection, query=vector, limit=limit, with_payload=True
        ).points
        return [
            ScoredChunk(chunk=DocumentChunk.model_validate(point.payload), score=point.score)
            for point in result
            if point.payload is not None
        ]

    def list_documents(self) -> list[DocumentSummary]:
        pairs = [
            (chunk.metadata.document_id, chunk.metadata.filename) for chunk in self.list_chunks()
        ]
        counts = Counter(pairs)
        return [
            DocumentSummary(document_id=doc_id, filename=filename, chunk_count=count)
            for (doc_id, filename), count in sorted(counts.items())
        ]

    def list_chunks(self) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        offset = None
        while True:
            records, offset = self._client.scroll(
                collection_name=self._collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            chunks.extend(
                DocumentChunk.model_validate(record.payload)
                for record in records
                if record.payload is not None
            )
            if offset is None:
                return chunks

    def delete_document(self, document_id: str) -> bool:
        existed = self.contains_document(document_id)
        if existed:
            self._client.delete(
                collection_name=self._collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.document_id",
                                match=models.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
                wait=True,
            )
        return existed

    def contains_document(self, document_id: str) -> bool:
        records, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.document_id", match=models.MatchValue(value=document_id)
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return bool(records)
