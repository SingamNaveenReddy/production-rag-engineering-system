from __future__ import annotations

import json
from pathlib import Path

from evaluation.schemas import GoldenCase


def load_golden_dataset(path: Path) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    identifiers: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = GoldenCase.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid golden dataset row at line {line_number}") from exc
        if case.id in identifiers:
            raise ValueError(f"Duplicate golden dataset id: {case.id}")
        if case.answerable and not case.expected_sources:
            raise ValueError(f"Answerable case requires expected_sources: {case.id}")
        if not case.answerable and case.expected_sources:
            raise ValueError(f"Unanswerable case must not declare expected_sources: {case.id}")
        identifiers.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError("Golden dataset is empty")
    return cases

