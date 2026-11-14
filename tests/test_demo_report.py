"""Preliminary reports use extracted source evidence, not assumptions."""

import json
from pathlib import Path


def test_report_distinguishes_evidence_negation_and_missing():
    from vrscore.demo.report import build_report

    def fact(value, polarity=None):
        return {
            "category": "hard_skill",
            "value": value,
            "evidence": value,
            "polarity": polarity,
        }

    report = build_report(
        {"id": "v", "entities": [fact("Python"), fact("Vue 3"), fact("SQL")]},
        {"id": "r", "entities": [fact("Python", "present"), fact("Vue 3", "absent")]},
    )
    assert [item["status"] for item in report["requirements"]] == [
        "matched",
        "partial",
        "not_stated",
    ]
    assert report["requirements"][1]["resume_evidence"] == "Vue 3"
    assert len(report["strengths"]) == 1
    assert not report["questions"]
    assert report["preliminary"] is True


def test_real_ivanov_vue_and_tailwind_are_not_claimed_as_strengths():
    from vrscore.demo.report import build_report

    root = Path(__file__).resolve().parents[1] / "public_data" / "parsed"
    resume = json.loads((root / "r1.json").read_text())
    vacancy = json.loads((root / "frontend.json").read_text())
    report = build_report(vacancy, resume)
    relevant = [
        row
        for row in report["requirements"]
        if str(row["requirement"]).casefold() in {"vue.js", "tailwind css"}
    ]
    assert len(relevant) == 2
    assert all(row["status"] == "contradicted" for row in relevant)


def test_report_service_covers_each_vacancy_without_pipeline(monkeypatch):
    from vrscore.demo.service import DemoService

    service = DemoService(None)

    def extraction(document, kind):
        return {"id": document["id"], "entities": []}

    monkeypatch.setattr(service, "_cached_extraction", extraction)
    report = service.report("r1")
    assert report["resume_id"] == "r1"
    assert {item["vacancy_id"] for item in report["vacancies"]} == {
        item["id"] for item in service.vacancies
    }
