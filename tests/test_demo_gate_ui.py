"""Focused regression tests for the demo's Gate 1 presentation."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_MODULE = ROOT / "vrscore/web/demo/app.js"


def test_analyze_button_keeps_original_label_and_gate_remains_available():
    from html.parser import HTMLParser

    class ButtonParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.current = None
            self.labels = {}

        def handle_starttag(self, tag, attrs):
            if tag == "button":
                self.current = dict(attrs).get("id")

        def handle_data(self, data):
            if self.current:
                self.labels[self.current] = self.labels.get(self.current, "") + data

        def handle_endtag(self, tag):
            if tag == "button":
                self.current = None

    parser = ButtonParser()
    parser.feed((ROOT / "vrscore/web/demo/index.html").read_text(encoding="utf-8"))
    assert parser.labels["analyze"].strip() == "АНАЛИЗ"
    assert "GATE 1" in parser.labels["run-gate"]


def run_gate_ui(payload):
    script = r"""
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(process.argv[1], 'utf8')
  .split('// --- Demo application ---')[0];
const sandbox = {URL, Map, Object, String, Array, globalThis: {}};
vm.runInNewContext(source, sandbox);
const ui = sandbox.globalThis.GateUI;
const payload = JSON.parse(process.argv[2]);
process.stdout.write(JSON.stringify({
  summary: ui.summarize(payload), html: ui.render(payload)
}));
"""
    result = subprocess.run(
        ["node", "-e", script, str(APP_MODULE), json.dumps(payload)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_gate_ui_preserves_unknown_and_counts_all_vacancies():
    payload = {
        "resume_id": "r1",
        "vacancies": [
            {
                "id": "v1",
                "title": "Инженер",
                "source_url": "https://example.test/1",
                "gate": {
                    "status": "pass",
                    "decisions": [
                        {
                            "filter": "experience",
                            "status": "pass",
                            "explanation": "Подходит",
                        },
                        {
                            "filter": "education",
                            "status": "pass",
                            "explanation": "Подходит",
                        },
                        {
                            "filter": "work_format",
                            "status": "pass",
                            "explanation": "Подходит",
                        },
                        {
                            "filter": "work_schedule",
                            "status": "pass",
                            "explanation": "Подходит",
                        },
                    ],
                },
            },
            {
                "id": "v2",
                "title": "Аналитик",
                "source_url": None,
                "gate": {
                    "status": "unknown",
                    "decisions": [
                        {
                            "filter": "experience",
                            "status": "unknown",
                            "explanation": "Нет данных",
                        }
                    ],
                },
            },
            {
                "id": "v3",
                "title": "Редактор",
                "source_url": "https://example.test/3",
                "gate": {
                    "status": "fail",
                    "decisions": [
                        {
                            "filter": "education",
                            "status": "fail",
                            "explanation": "Не совпадает",
                        }
                    ],
                },
            },
        ],
    }
    result = run_gate_ui(payload)
    assert result["summary"] == {"total": 3, "pass": 1, "fail": 1, "unknown": 1}
    assert all(title in result["html"] for title in ("Инженер", "Аналитик", "Редактор"))
    assert "Нет данных" in result["html"]
    assert "Недостаточно данных" in result["html"]


def test_gate_ui_escapes_untrusted_content_and_renders_entity_evidence():
    payload = {
        "resume_id": "r1",
        "vacancies": [
            {
                "id": "<v>",
                "title": "<script>alert(1)</script>",
                "source_url": None,
                "gate": {
                    "status": "fail",
                    "decisions": [
                        {
                            "filter": "work_format",
                            "status": "fail",
                            "explanation": (
                                "No candidate format is compatible "
                                "with the vacancy format."
                            ),
                            "requirement": [
                                {
                                    "label": "Формат",
                                    "value": "Офис",
                                    "evidence": "В офисе",
                                    "context": "Москва",
                                }
                            ],
                            "candidate": [
                                {
                                    "label": "Формат",
                                    "value": "Удалённо",
                                    "evidence": "удалённо",
                                    "context": "из дома",
                                }
                            ],
                        }
                    ],
                },
            }
        ],
    }
    html = run_gate_ui(payload)["html"]
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert all(text in html for text in ("В офисе", "Москва", "удалённо", "из дома"))
    assert "Формат работы из резюме несовместим" in html
    assert "work_format" not in html
    assert "Формат работы" in html


def test_gate_ui_marks_skipped_rules_unmet_but_keeps_unknown_status():
    payload = {
        "resume_id": "r1",
        "vacancies": [
            {
                "id": "v1",
                "title": "Unknown",
                "gate": {
                    "status": "unknown",
                    "decisions": [
                        {
                            "filter": "experience",
                            "status": "unknown",
                            "explanation": "Нет данных",
                        },
                    ],
                },
            },
            {
                "id": "v2",
                "title": "Malformed pass",
                "gate": {
                    "status": "pass",
                    "decisions": [
                        {
                            "filter": "experience",
                            "status": "pass",
                            "explanation": "Подходит",
                        },
                    ],
                },
            },
        ],
    }
    result = run_gate_ui(payload)
    assert result["summary"] == {"total": 2, "pass": 0, "fail": 1, "unknown": 1}
    assert "Не проверено" in result["html"]
    assert "Проверка остановлена на предыдущем правиле." in result["html"]
