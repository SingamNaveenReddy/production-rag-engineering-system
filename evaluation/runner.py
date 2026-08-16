from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.config import EvaluationThresholds
from app.models.schemas import QueryResult
from evaluation.metrics import (
    CitationTokenCoverageScorer,
    FaithfulnessScorer,
    aggregate_metrics,
    evaluate_case,
    evaluate_thresholds,
)
from evaluation.schemas import EvaluationReport, GoldenCase


class QueryEngine(Protocol):
    def query(self, question: str, top_k: int | None = None) -> QueryResult: ...


def run_evaluation(
    engine: QueryEngine,
    cases: list[GoldenCase],
    *,
    dataset_path: Path,
    profile: str,
    retrieval_k: int,
    thresholds: EvaluationThresholds,
    faithfulness_scorer: FaithfulnessScorer | None = None,
) -> EvaluationReport:
    scorer = faithfulness_scorer or CitationTokenCoverageScorer()
    case_metrics = [
        evaluate_case(
            case,
            engine.query(case.question),
            retrieval_k=retrieval_k,
            faithfulness_scorer=scorer,
        )
        for case in cases
    ]
    metrics = aggregate_metrics(case_metrics)
    return EvaluationReport(
        profile=profile,
        dataset=str(dataset_path),
        questions_evaluated=len(cases),
        retrieval_k=retrieval_k,
        faithfulness_method=scorer.name,
        metrics=metrics,
        quality_gate=evaluate_thresholds(metrics, thresholds),
        cases=case_metrics,
    )

