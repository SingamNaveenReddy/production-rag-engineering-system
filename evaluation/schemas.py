from __future__ import annotations

from pydantic import BaseModel, Field


class ExpectedSource(BaseModel):
    document: str
    page: int | None = None
    chunk_id: str | None = None


class GoldenCase(BaseModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_answer: str | None
    expected_sources: list[ExpectedSource]
    answerable: bool


class CaseMetrics(BaseModel):
    case_id: str
    answerable_expected: bool
    answerable_actual: bool
    retrieval_recall_at_k: float | None
    faithfulness: float | None
    citation_accuracy: float | None
    refusal_correct: bool
    latency_ms: int


class LatencyMetrics(BaseModel):
    mean_ms: float
    median_ms: float
    p95_ms: int


class AggregateMetrics(BaseModel):
    retrieval_recall_at_k: float
    faithfulness: float
    citation_accuracy: float
    refusal_accuracy: float
    latency: LatencyMetrics


class ThresholdResult(BaseModel):
    passed: bool
    failures: list[str]


class EvaluationReport(BaseModel):
    profile: str
    dataset: str
    questions_evaluated: int
    retrieval_k: int
    faithfulness_method: str
    metrics: AggregateMetrics
    quality_gate: ThresholdResult
    cases: list[CaseMetrics]

