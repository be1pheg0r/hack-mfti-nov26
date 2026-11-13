"""Exercise end-to-end dataset construction without private data."""

import hashlib
import json

import pytest

from vrscore.vacancy_parser.dataset import build_dataset


def test_build_dataset_preserves_input_and_links_every_evidence_span(tmp_path):
    """Produce a separate viewer-compatible corpus with auditable references."""
    source = tmp_path / "input"
    source.mkdir()
    text = "Описание:\nТребования:\nВысшее образование и опыт работы от 3 лет."
    start = text.index("Требования:")
    row = {
        "id": "vacancy_001",
        "kind": "vacancy",
        "sampling_group": "it",
        "title": "Инженер",
        "text": text,
        "attributes": {"description": text[start:]},
        "sections": [
            {
                "source_field": "description",
                "label": "Описание",
                "start": start,
                "end": len(text),
            }
        ],
        "source": {"text_completeness": "public_page_description"},
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    original = json.dumps(row, ensure_ascii=False) + "\n"
    (source / "vacancies.jsonl").write_text(original)
    (source / "resumes.jsonl").write_text("")
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "name": "test",
                "actual": {"resumes": 0, "vacancies": 1},
                "sha256": {},
            }
        )
    )
    (source / "schema.json").write_text("{}")
    output = tmp_path / "output"
    build_dataset(source, output)
    assert (source / "vacancies.jsonl").read_text() == original
    parsed = json.loads((output / "vacancies.jsonl").read_text())
    audit = json.loads((output / "requirements.jsonl").read_text())
    assert parsed["attributes"]["experienceRequirements"] == 3
    assert audit["id"] == parsed["id"]
    assert (
        parsed["text"][audit["items"][0]["start"] : audit["items"][0]["end"]]
        == audit["items"][0]["text"]
    )
    assert (output / "resumes.jsonl").resolve() == (source / "resumes.jsonl").resolve()
    with pytest.raises(FileExistsError):
        build_dataset(source, output)


def test_dataset_build_applies_only_matching_source_evidence(tmp_path):
    """Additional fields are traceable to their exact source and record."""
    source = tmp_path / "input"
    source.mkdir()
    text = "Требования:\nPython"
    row = {
        "id": "vacancy_001",
        "kind": "vacancy",
        "sampling_group": "it",
        "title": "Инженер",
        "text": text,
        "attributes": {"positionRequirements": "Python"},
        "sections": [
            {
                "source_field": "positionRequirements",
                "label": "Требования",
                "start": 12,
                "end": len(text),
            }
        ],
        "source": {"original_url": "https://example.org/1"},
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    (source / "vacancies.jsonl").write_text(json.dumps(row) + "\n")
    (source / "resumes.jsonl").write_text("")
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "name": "test",
                "actual": {"resumes": 0, "vacancies": 1},
                "sha256": {},
            }
        )
    )
    (source / "schema.json").write_text("{}")
    evidence = tmp_path / "evidence.jsonl"
    evidence.write_text(
        json.dumps(
            {
                "id": row["id"],
                "source_url": row["source"]["original_url"],
                "fields": {
                    "salary": {"value": "от 80 000 ₽", "source_label": "salary"}
                },
            }
        )
        + "\n"
    )
    output = tmp_path / "output"
    build_dataset(source, output, evidence_path=evidence)
    parsed = json.loads((output / "vacancies.jsonl").read_text())
    assert parsed["attributes"]["salaryMin"] == 80000
    assert parsed["attributes"]["salaryText"] == "от 80 000 ₽"


def test_reviewed_duplicate_is_kept_as_full_alternative(tmp_path):
    """Resolved duplicates remain recoverable without inflating the corpus."""
    source = tmp_path / "input"
    source.mkdir()
    rows = []
    for number, text in ((1, "Требования: Python"), (2, "Требования: Python и SQL")):
        rows.append(
            {
                "id": f"vacancy_{number:03d}",
                "kind": "vacancy",
                "text": text,
                "attributes": {"positionRequirements": text},
                "sections": [
                    {
                        "source_field": "positionRequirements",
                        "start": 0,
                        "end": len(text),
                    }
                ],
                "source": {},
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        )
    (source / "vacancies.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (source / "resumes.jsonl").write_text("")
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "name": "test",
                "actual": {"resumes": 0, "vacancies": 2},
                "sha256": {},
            }
        )
    )
    (source / "schema.json").write_text("{}")
    resolution = tmp_path / "resolution.json"
    resolution.write_text(
        json.dumps(
            {
                "merges": [
                    {
                        "keep": rows[0]["id"],
                        "remove": rows[1]["id"],
                        "keep_sha256": rows[0]["text_sha256"],
                        "remove_sha256": rows[1]["text_sha256"],
                        "reason": "Same reviewed role",
                    }
                ]
            }
        )
    )
    output = tmp_path / "output"
    result = build_dataset(source, output, resolutions_path=resolution)
    assert result["actual"]["vacancies"] == 1
    alternative = json.loads((output / "alternative_records.jsonl").read_text())
    assert alternative["duplicate_of"] == rows[0]["id"]
    assert alternative["record"]["id"] == rows[1]["id"]
    assert len((output / "parsing_audit.jsonl").read_text().splitlines()) == 2
