"""The strict gate can compare parsed documents from a Hydra CLI."""

import json
import subprocess
import sys
from pathlib import Path


def test_gate_cli_uses_actual_parsed_json_and_config_overrides(tmp_path):
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "-m",
        "vrscore.gate_cli",
        "vacancy=public_data/parsed/accountant.json",
        "resume=public_data/parsed/r2.json",
        f"hydra.run.dir={tmp_path}",
    ]
    result = subprocess.run(
        command, cwd=root, capture_output=True, text=True, check=True
    )
    decision = json.loads(result.stdout)
    assert decision["status"] == "pass"
    assert [entry["filter"] for entry in decision["decisions"]] == [
        "experience",
        "education",
        "work_format",
    ]
    overridden = subprocess.run(
        command + ["filters.order=[work_format,education,experience]"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert [
        entry["filter"] for entry in json.loads(overridden.stdout)["decisions"]
    ] == ["work_format", "education", "experience"]
