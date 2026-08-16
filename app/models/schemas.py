from __future__ import annotations

from pydantic import BaseModel, Field


class SourceMetadata(BaseModel):
    document_id: str
    filename: str
    page: int | None = None
    section: str | None = None
    chunk_id: str
    original_source: str


class DocumentChunk(BaseModel):
    text: str = Field(min_length=1)
    metadata: SourceMetadata


class ScoredChunk(BaseModel):
    chunk: DocumentChunk
    score: float


class GeneratedAnswer(BaseModel):
    answer: str = Field(min_length=1)
    answerable: bool
    supporting_chunk_ids: list[str]


class Citation(BaseModel):
    document: str
    page: int | None = None
    chunk_id: str
    supporting_text: str


class QueryResult(BaseModel):
    answer: str
    answerable: bool
    citations: list[Citation]
    retrieval_metadata: dict[str, object] = Field(default_factory=dict)
    latency_ms: int = Field(ge=0)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=100)


class IngestPathRequest(BaseModel):
    path: str


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    chunk_count: int


class HealthResponse(BaseModel):
    status: str
    phase: str
