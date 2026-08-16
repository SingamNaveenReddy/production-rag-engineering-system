from pathlib import Path

import yaml

WORKFLOW_PATH = Path(".github/workflows/quality-gate.yml")


def test_quality_gate_runs_for_pull_requests_with_minimal_permissions() -> None:
    workflow = yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    assert "pull_request" in workflow["on"]
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["quality-gate"]["runs-on"] == "ubuntu-latest"


def test_quality_gate_runs_every_required_check() -> None:
    workflow_text = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "python -m ruff check ." in workflow_text
    assert "python -m pytest tests/unit" in workflow_text
    assert "python -m pytest tests/integration" in workflow_text
    assert "python -m evaluation.evaluate" in workflow_text
    assert "--profile deterministic" in workflow_text
