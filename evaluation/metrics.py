from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Iterable
from statistics import mean, median
from typing import Protocol

from app.config import EvaluationThresholds
from app.models.schemas import Citation, QueryResult
from evaluation.schemas import (
    AggregateMetrics,
    CaseMetrics,
    ExpectedSource,
    GoldenCase,
    LatencyMetrics,
    ThresholdResult,
)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", re.IGNORECASE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "for",
    "how",
    "is",
    "must",
    "of",
    "the",
    "to",
    "what",
    "when",
    "which",
    "with",
}


class FaithfulnessScorer(Protocol):
    name: str

    def score(self, result: QueryResult) -> float: ...


class CitationTokenCoverageScorer:
    """Deterministic local proxy; production runs can inject an LLM-based scorer."""

    name = "lexical_citation_token_coverage"

    def score(self, result: QueryResult) -> float:
        answer_tokens = set(content_tokens(result.answer))
        if not answer_tokens:
            return 0.0
        context_tokens = set(
            content_tokens(" ".join(citation.supporting_text for citation in result.citations))
        )
        return len(answer_tokens & context_tokens) / len(answer_tokens)


class AsyncFaithfulnessMetric(Protocol):
    async def ascore(self, *, response: str, retrieved_contexts: list[str]) -> object: ...


class RagasFaithfulnessScorer:
    """Adapter for the Ragas collections Faithfulness metric."""

    name = "ragas_faithfulness"

    def __init__(self, metric: AsyncFaithfulnessMetric) -> None:
        self._metric = metric

    def score(self, result: QueryResult) -> float:
        scored = asyncio.run(
            self._metric.ascore(
                response=result.answer,
                retrieved_contexts=[
                    citation.supporting_text for citation in result.citations
                ],
            )
        )
        value = getattr(scored, "value", scored)
        return float(value)


def evaluate_case(
    case: GoldenCase,
    result: QueryResult,
    *,
    retrieval_k: int,
    faithfulness_scorer: FaithfulnessScorer,
) -> CaseMetrics:
    retrieved_sources = result.retrieval_metadata.get("retrieved_sources", [])
    recall = None
    faithfulness = None
    citation_accuracy = None
    if case.answerable:
        recall = _retrieval_recall(case.expected_sources, retrieved_sources[:retrieval_k])
        faithfulness = faithfulness_scorer.score(result) if result.answerable else 0.0
        citation_accuracy = _citation_accuracy(case.expected_sources, result.citations)
    return CaseMetrics(
        case_id=case.id,
        answerable_expected=case.answerable,
        answerable_actual=result.answerable,
        retrieval_recall_at_k=recall,
        faithfulness=faithfulness,
        citation_accuracy=citation_accuracy,
        refusal_correct=result.answerable == case.answerable,
        latency_ms=result.latency_ms,
    )


def aggregate_metrics(cases: list[CaseMetrics]) -> AggregateMetrics:
    latencies = sorted(case.latency_ms for case in cases)
    return AggregateMetrics(
        retrieval_recall_at_k=_average(
            case.retrieval_recall_at_k for case in cases if case.retrieval_recall_at_k is not None
        ),
        faithfulness=_average(
            case.faithfulness for case in cases if case.faithfulness is not None
        ),
        citation_accuracy=_average(
            case.citation_accuracy for case in cases if case.citation_accuracy is not None
        ),
        refusal_accuracy=_average(1.0 if case.refusal_correct else 0.0 for case in cases),
        latency=LatencyMetrics(
            mean_ms=mean(latencies),
            median_ms=median(latencies),
            p95_ms=latencies[max(0, math.ceil(0.95 * len(latencies)) - 1)],
        ),
    )


def evaluate_thresholds(
    metrics: AggregateMetrics, thresholds: EvaluationThresholds
) -> ThresholdResult:
    failures: list[str] = []
    minimums = {
        "retrieval_recall_at_k": thresholds.retrieval_recall_at_k,
        "faithfulness": thresholds.faithfulness,
        "citation_accuracy": thresholds.citation_accuracy,
        "refusal_accuracy": thresholds.refusal_accuracy,
    }
    for name, minimum in minimums.items():
        actual = float(getattr(metrics, name))
        if actual < minimum:
            failures.append(f"{name}={actual:.3f} is below {minimum:.3f}")
    maximum_latency = thresholds.max_p95_latency_ms
    if maximum_latency is not None and metrics.latency.p95_ms > maximum_latency:
        failures.append(
            f"p95_latency_ms={metrics.latency.p95_ms} exceeds {maximum_latency}"
        )
    return ThresholdResult(passed=not failures, failures=failures)


def _retrieval_recall(
    expected_sources: list[ExpectedSource], retrieved_sources: list[object]
) -> float:
    matched = sum(
        any(_source_matches(expected, actual) for actual in retrieved_sources)
        for expected in expected_sources
    )
    return matched / len(expected_sources)


def _citation_accuracy(
    expected_sources: list[ExpectedSource], citations: list[Citation]
) -> float:
    if not citations:
        return 0.0
    correct = sum(
        any(_citation_matches(expected, citation) for expected in expected_sources)
        for citation in citations
    )
    return correct / len(citations)


def _source_matches(expected: ExpectedSource, actual: object) -> bool:
    if not isinstance(actual, dict) or actual.get("document") != expected.document:
        return False
    if expected.page is not None and actual.get("page") != expected.page:
        return False
    return expected.chunk_id is None or actual.get("chunk_id") == expected.chunk_id


def _citation_matches(expected: ExpectedSource, citation: Citation) -> bool:
    if citation.document != expected.document:
        return False
    if expected.page is not None and citation.page != expected.page:
        return False
    return expected.chunk_id is None or citation.chunk_id == expected.chunk_id


def content_tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
        if token.lower() not in STOPWORDS
    ]


def _average(values: Iterable[float]) -> float:
    items = list(values)
    return mean(items) if items else 0.0
