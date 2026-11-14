"""The demo journey must show actual pipeline stages without fabricating scores."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_stage_rows_preserve_gate_and_scores():
    script = """
const ui = require('./vrscore/web/demo/journey.js');
const data = {initial:['b','a','c'], gate:{passed:['a'],unknown:['c'],rejected:['b']},
  bm25:[{vacancy_id:'c',score:2.3}], ranked:[{vacancy_id:'c',score:0.8}]};
const rows = [0,1,2,3].map(stage => ui.stageRows(data,stage));
process.stdout.write(JSON.stringify(rows));
"""
    result = subprocess.run(
        ["node", "-e", script], cwd=ROOT, capture_output=True, text=True, check=True
    )
    rows = json.loads(result.stdout)
    assert [r["id"] for r in rows[0]] == ["b", "a", "c"]
    assert rows[1][0]["removed"] is True
    assert rows[1][2]["removed"] is False
    assert rows[2][1]["removed"] is True
    assert rows[2][2]["score"] == 2.3
    assert rows[3] == [{"id": "c", "score": 0.8, "removed": False}]
