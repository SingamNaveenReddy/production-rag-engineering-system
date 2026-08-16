from __future__ import annotations

import argparse
from pathlib import Path

from app.config import get_config
from evaluation.dataset import load_golden_dataset
from evaluation.profiles import build_deterministic_engine, build_production_engine
from evaluation.reporting import write_reports
from evaluation.runner import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the offline golden RAG evaluation")
    parser.add_argument(
        "--profile", choices=("deterministic", "production"), default="deterministic"
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-directory", type=Path, default=Path("evaluation/results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()
    dataset_path = args.dataset or config.evaluation.dataset_path
    cases = load_golden_dataset(dataset_path)
    if args.profile == "production":
        engine = build_production_engine(config)
    else:
        engine = build_deterministic_engine(Path("data/sample"))
    report = run_evaluation(
        engine,
        cases,
        dataset_path=dataset_path,
        profile=args.profile,
        retrieval_k=config.evaluation.retrieval_k,
        thresholds=config.evaluation.thresholds,
    )
    json_path, markdown_path = write_reports(report, args.output_directory)
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    print(f"Quality gate: {'PASS' if report.quality_gate.passed else 'FAIL'}")
    if not report.quality_gate.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

