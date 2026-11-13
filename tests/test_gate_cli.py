"""The strict gate can compare parsed documents from a Hydra CLI."""

import json
import subprocess
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

from vrscore.gate_cli import main


def test_gate_cli_uses_actual_parsed_json_and_config_overrides(tmp_path, capsys):
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
        config = compose(
            config_name="first_gate_cli",
            overrides=[
                f"vacancy={root / 'public_data/parsed/accountant.json'}",
                f"resume={root / 'public_data/parsed/r2.json'}",
            ],
        )
    main.__wrapped__(config)
    decision = json.loads(capsys.readouterr().out)
    assert decision["status"] == "pass"
    assert [entry["filter"] for entry in decision["decisions"]] == [
        "experience",
        "education",
        "work_format",
        "work_schedule",
    ]
    command = [
        sys.executable,
        "-m",
        "vrscore.gate_cli",
        "vacancy=public_data/parsed/accountant.json",
        "resume=public_data/parsed/r2.json",
        f"hydra.run.dir={tmp_path}",
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
