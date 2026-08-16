import json
import subprocess
import sys
from pathlib import Path


def test_evaluation_cli_exits_nonzero_when_quality_regresses(tmp_path: Path) -> None:
    dataset = tmp_path / "regressed.jsonl"
    dataset.write_text(
        '{"id":"regression","question":"What authentication device is mandatory for '
        'administrators?","expected_answer":"A FIDO2 hardware security key.",'
        '"expected_sources":[{"document":"wrong-source.md"}],"answerable":true}\n',
        encoding="utf-8",
    )
    output_directory = tmp_path / "results"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.evaluate",
            "--profile",
            "deterministic",
            "--dataset",
            str(dataset),
            "--output-directory",
            str(output_directory),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    report = json.loads((output_directory / "golden_evaluation.json").read_text(encoding="utf-8"))
    assert completed.returncode == 1
    assert "Quality gate: FAIL" in completed.stdout
    assert report["quality_gate"]["passed"] is False
    assert any("retrieval_recall_at_k" in failure for failure in report["quality_gate"]["failures"])
