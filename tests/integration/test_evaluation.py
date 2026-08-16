from pathlib import Path

from app.config import EvaluationThresholds
from evaluation.dataset import load_golden_dataset
from evaluation.profiles import build_deterministic_engine
from evaluation.runner import run_evaluation


def test_deterministic_golden_profile_passes_quality_gate() -> None:
    dataset_path = Path("evaluation/golden_dataset.jsonl")
    report = run_evaluation(
        build_deterministic_engine(Path("data/sample")),
        load_golden_dataset(dataset_path),
        dataset_path=dataset_path,
        profile="deterministic-test",
        retrieval_k=5,
        thresholds=EvaluationThresholds(),
    )
    assert report.questions_evaluated == 6
    assert report.metrics.retrieval_recall_at_k == 1
    assert report.metrics.citation_accuracy == 1
    assert report.metrics.refusal_accuracy == 1
    assert report.quality_gate.passed is True

