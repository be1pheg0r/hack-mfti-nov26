"""Verify source metadata enrichment preserves meaning and provenance."""

import hashlib

from vacancy_parser.evidence import apply_source_evidence


def test_source_evidence_restores_description_and_tags_skills():
    """Recover text only from the same source and retain evidence offsets."""
    title = "Аналитик"
    description = "Требования:\nОпыт"
    text = f"Должность:\n{title}\n\nОписание вакансии:\n{description}"
    start = text.index(description)
    record = {
        "id": "vacancy_001",
        "title": title,
        "text": text,
        "attributes": {"typicalPosition": title, "description": description},
        "sections": [
            {
                "source_field": "typicalPosition",
                "label": "Должность",
                "start": 11,
                "end": 11 + len(title),
            },
            {
                "source_field": "description",
                "label": "Описание",
                "start": start,
                "end": len(text),
            },
        ],
        "source": {"original_url": "https://example.org/1"},
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    evidence = {
        "id": "vacancy_001",
        "source_url": "https://example.org/1",
        "full_description": "Требования:\nОпыт работы от 3 лет.",
        "fields": {
            "experience": {"value": "1–3 года", "source_label": "experience.name"},
            "key_skills": {"value": ["Python", "SQL"], "source_label": "key_skills"},
            "salary_object": {
                "value": {
                    "from": 100000,
                    "to": None,
                    "currency": "RUR",
                    "gross": False,
                },
                "source_label": "salary",
            },
            "base_salary": {
                "value": "90 700 руб.",
                "source_label": "ДОЛЖНОСТНОЙ ОКЛАД",
            },
        },
    }
    output, audit = apply_source_evidence(record, evidence)
    assert output["attributes"]["description"] == evidence["full_description"]
    assert output["attributes"]["skills"] == ["Python", "SQL"]
    assert output["attributes"]["experience"] == "1–3 года"
    assert output["attributes"]["salaryMin"] == 100000
    assert output["attributes"]["baseSalary"] == 90700
    assert "salaryMax" not in output["attributes"]
    assert output["text_sha256"] == hashlib.sha256(output["text"].encode()).hexdigest()
    for item in audit["fields"].values():
        assert output["text"][item["start"] : item["end"]] == item["evidence"]
    assert record["text"] == text
