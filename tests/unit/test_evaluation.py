from pathlib import Path

import pytest

from app.config import EvaluationThresholds
from app.models.schemas import Citation, QueryResult
from evaluation.dataset import load_golden_dataset
from evaluation.metrics import RagasFaithfulnessScorer, aggregate_metrics, evaluate_thresholds
from evaluation.runner import run_evaluation
from evaluation.schemas import CaseMetrics, ExpectedSource, GoldenCase


class StubEngine:
    def query(self, question: str, top_k: int | None = None) -> QueryResult:
        return QueryResult(
            answer="A FIDO2 hardware security key is required.",
            answerable=True,
            citations=[
                Citation(
                    document="authentication_policy.md",
                    chunk_id="auth-c1",
                    supporting_text="A FIDO2 hardware security key is required.",
                )
            ],
            retrieval_metadata={
                "retrieved_sources": [
                    {
                        "document": "authentication_policy.md",
                        "page": None,
                        "chunk_id": "auth-c1",
                    }
                ]
            },
            latency_ms=12,
        )


class FakeRagasResult:
    value = 0.75


class FakeRagasMetric:
    async def ascore(self, *, response: str, retrieved_contexts: list[str]) -> FakeRagasResult:
        assert response
        assert retrieved_contexts
        return FakeRagasResult()


def test_runner_calculates_metrics_from_executed_result(tmp_path: Path) -> None:
    case = GoldenCase(
        id="auth-001",
        question="What is required?",
        expected_answer="A FIDO2 hardware security key.",
        expected_sources=[ExpectedSource(document="authentication_policy.md")],
        answerable=True,
    )
    report = run_evaluation(
        StubEngine(),
        [case],
        dataset_path=tmp_path / "golden.jsonl",
        profile="test",
        retrieval_k=5,
        thresholds=EvaluationThresholds(),
    )
    assert report.metrics.retrieval_recall_at_k == 1
    assert report.metrics.faithfulness == 1
    assert report.metrics.citation_accuracy == 1
    assert report.metrics.refusal_accuracy == 1
    assert report.metrics.latency.p95_ms == 12
    assert report.quality_gate.passed is True


def test_quality_gate_reports_metric_regressions() -> None:
    cases = [
        CaseMetrics(
            case_id="failed",
            answerable_expected=True,
            answerable_actual=False,
            retrieval_recall_at_k=0.5,
            faithfulness=0,
            citation_accuracy=0,
            refusal_correct=False,
            latency_ms=500,
        )
    ]
    result = evaluate_thresholds(
        aggregate_metrics(cases),
        EvaluationThresholds(max_p95_latency_ms=100),
    )
    assert result.passed is False
    assert len(result.failures) == 5
    assert any("p95_latency_ms" in failure for failure in result.failures)


def test_ragas_adapter_uses_injected_async_metric() -> None:
    result = StubEngine().query("question")
    assert RagasFaithfulnessScorer(FakeRagasMetric()).score(result) == 0.75


def test_dataset_rejects_answerable_case_without_sources(tmp_path: Path) -> None:
    dataset = tmp_path / "invalid.jsonl"
    dataset.write_text(
        '{"id":"bad","question":"Q?","expected_answer":"A",'
        '"expected_sources":[],"answerable":true}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires expected_sources"):
        load_golden_dataset(dataset)
