from __future__ import annotations

from pathlib import Path

from evaluation.schemas import EvaluationReport


def write_reports(report: EvaluationReport, output_directory: Path) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "golden_evaluation.json"
    markdown_path = output_directory / "golden_evaluation.md"
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def render_markdown(report: EvaluationReport) -> str:
    metrics = report.metrics
    gate = "PASS" if report.quality_gate.passed else "FAIL"
    failures = "\n".join(f"- {failure}" for failure in report.quality_gate.failures)
    if not failures:
        failures = "- None"
    return f"""# Golden RAG evaluation

Profile: `{report.profile}`  
Dataset: `{report.dataset}`  
Questions evaluated: {report.questions_evaluated}  
Faithfulness method: `{report.faithfulness_method}`

## Metrics

| Metric | Result |
|---|---:|
| Retrieval recall@{report.retrieval_k} | {metrics.retrieval_recall_at_k:.3f} |
| Faithfulness | {metrics.faithfulness:.3f} |
| Citation accuracy | {metrics.citation_accuracy:.3f} |
| Refusal accuracy | {metrics.refusal_accuracy:.3f} |
| Mean latency | {metrics.latency.mean_ms:.1f} ms |
| Median latency | {metrics.latency.median_ms:.1f} ms |
| P95 latency | {metrics.latency.p95_ms} ms |

## Quality gate: {gate}

{failures}

## Interpretation

The deterministic profile exercises the real ingestion, hybrid retrieval, answerability, and
citation-validation service with local deterministic providers. Its lexical faithfulness score is
a CI-safe proxy based on answer-token coverage in validated citation text, not an LLM-judged Ragas
score. Inject an explicit LLM-based scorer for release-quality production assessment.
"""
