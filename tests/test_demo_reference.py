"""The demo reference order is sourced only from the annotated dataset."""

import json
from pathlib import Path

import pytest

from vrscore.demo.service import DemoService

ROOT = Path(__file__).resolve().parents[1]
GT_DIR = ROOT / ".data" / "mfti_matching_gt"


def _service(gt_dir=GT_DIR):
    if (
        not Path(gt_dir).exists()
        or not (ROOT / ".data/mfti_demo/vacancies.json").exists()
    ):
        pytest.skip("Private benchmark corpus is not installed")
    return DemoService(
        None,
        matching_gt_path=gt_dir,
        extra_vacancies_path=ROOT / ".data" / "mfti_demo" / "vacancies.json",
    )


def test_reference_order_returns_20_ai_labels_and_seven_unranked_vacancies():
    service = _service()
    result = service.reference_order("r1")
    pairs = [
        json.loads(line) for line in (GT_DIR / "pairs.jsonl").read_text().splitlines()
    ]
    vacancies = [
        json.loads(line)
        for line in (GT_DIR / "vacancies.jsonl").read_text().splitlines()
    ]
    expected = sorted(
        (pair for pair in pairs if pair["resume_id"] == "r1"),
        key=lambda pair: pair["rank"],
    )

    assert result["resume_id"] == "r1"
    assert result["disclosure"] == "AI-labeled reference; not an algorithm prediction."
    assert len(result["ranked"]) == 20
    assert len(result["unannotated"]) == 7
    assert [item["vacancy_id"] for item in result["ranked"]] == [
        pair["vacancy_id"] for pair in expected
    ]
    assert [item["rank"] for item in result["ranked"]] == [
        pair["rank"] for pair in expected
    ]
    assert [item["relevance"] for item in result["ranked"]] == [
        pair["relevance"] for pair in expected
    ]
    assert all(
        "rank" not in item and "relevance" not in item for item in result["unannotated"]
    )
    assert {
        item.get("vacancy_id", item.get("id"))
        for item in result["ranked"] + result["unannotated"]
    } == {vacancy["id"] for vacancy in service.extra_vacancies} | {
        vacancy["id"] for vacancy in service.vacancies
    }
    assert all(item["title"] for item in result["ranked"] + result["unannotated"])
    assert len(vacancies) == 20


def test_reference_order_is_independent_of_gate_extraction():
    service = _service()
    assert service.reference_order("synthetic_r10")["ranked"]
    assert service.cache == {}
    assert service.reference_order("synthetic_r10")["ranked"]
    assert service.cache == {}


def test_reference_order_rejects_unknown_resume():
    with pytest.raises(ValueError, match="resume"):
        _service().reference_order("not-a-resume")


def test_reference_rejects_changed_documents_and_keeps_new_vacancies_unranked():
    """Reference labels cannot be silently reused for different source texts."""

    service = _service()
    service.extra_vacancies.append({"id": "new", "title": "New", "text": "New"})
    assert any(
        row["id"] == "new" for row in service.reference_order("r1")["unannotated"]
    )
    service.resumes[0]["text"] += " changed"
    with pytest.raises(ValueError, match="changed"):
        service.reference_order("r1")


def test_missing_optional_reference_does_not_break_demo(tmp_path):
    """An installation without the private dataset can still start."""
    service = DemoService(None, matching_gt_path=tmp_path / "missing")
    assert service.matching_reference is None
