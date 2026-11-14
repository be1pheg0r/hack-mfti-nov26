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


def test_vacancy_report_preserves_uncertainty_and_escapes_evidence():
    script = """
const ui = require('./vrscore/web/demo/journey.js');
const report = {preliminary:true, requirements:[
 {requirement:'Vue <script>',status:'contradicted',comment:'Проверить контекст',
  vacancy_evidence:'Требуется Vue',resume_evidence:'Не работал',
  vacancy_source:{evidence_kind:'source_block',document_id:'v',text_version:'v1'},
  resume_source:{document_id:'r',text_version:'v1'}},
 {requirement:'SQL',status:'not_stated',vacancy_evidence:'SQL обязателен'}],
 strengths:[],risks:[],questions:['Какой опыт?']};
process.stdout.write(ui.reportHtml(report));
"""
    output = subprocess.run(
        ["node", "-e", script], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    assert "Предварительный" in output
    assert "&lt;script&gt;" in output and "<script>" not in output
    assert "Противоречие" in output and "Не указано" in output
    assert "Не работал" in output and "Требуется Vue" in output
    assert "Фрагмент документа" in output
    assert "Какой опыт?" in output
