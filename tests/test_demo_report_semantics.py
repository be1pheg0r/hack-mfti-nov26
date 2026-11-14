"""Regression coverage for conservative, useful vacancy reports."""

from vrscore.demo.report import build_report


def fact(category, value, evidence, **extras):
    return {"category": category, "value": value, "evidence": evidence, **extras}


def test_numeric_experience_is_compared_not_matched_as_a_string():
    vacancy = {
        "id": "v",
        "entities": [fact("experience", 1, "Опыт от 1 года", importance="strict")],
    }
    resume = {
        "id": "r",
        "entities": [fact("experience", 2.83, "Опыт 2 года 10 месяцев")],
    }
    report = build_report(vacancy, resume)
    assert report["requirements"][0]["status"] == "matched"
    assert not report["risks"] and not report["questions"]
    assert report["requirements"][0]["resume_evidence"] == "Опыт 2 года 10 месяцев"
    resume["entities"][0]["value"] = 0.5
    assert build_report(vacancy, resume)["requirements"][0]["status"] == "contradicted"
    resume["entities"][0]["value"] = None
    assert build_report(vacancy, resume)["requirements"][0]["status"] == "not_stated"


def test_specific_work_experience_cannot_be_proven_by_generic_total():
    vacancy = {
        "entities": [
            fact("experience", 1, "Опыт работы с Python от 1 года", importance="strict")
        ]
    }
    resume = {"entities": [fact("experience", 3, "Опыт работы 3 года")]}
    row = build_report(vacancy, resume)["requirements"][0]
    assert row["status"] == "partial"
    assert "уточнить" in row["comment"]


def test_experience_keeps_all_source_citations_without_summing_periods():
    vacancy = {"entities": [fact("experience", 3, "Опыт от 3 лет")]}
    resume = {
        "entities": [
            {
                **fact("experience", 4, "Два периода", derived=True),
                "sources": [
                    fact("experience", 2, "Период 1: 2 года"),
                    fact("experience", 2, "Период 2: 2 года"),
                ],
            }
        ]
    }
    row = build_report(vacancy, resume)["requirements"][0]
    assert row["status"] == "partial"
    assert len(row["resume_sources"]) == 2
    vacancy["entities"][0]["value"] = 1
    row = build_report(vacancy, resume)["requirements"][0]
    assert row["status"] == "matched"
    assert len(row["resume_sources"]) == 2


def test_technical_evidence_scope_deduplication_and_questions():
    vacancy = {
        "id": "v",
        "entities": [
            fact("hard_skill", "HTML", "HTML и CSS", importance="good2have"),
            fact("hard_skill", "HTML5", "Уверенное знание HTML5", importance="strict"),
            fact(
                "hard_skill",
                "Vue 3",
                "Разработка компонентов на Vue 3",
                importance="good2have",
            ),
            fact(
                "hard_skill", "Vue.js", "Коммерческий опыт Vue.js", importance="strict"
            ),
            fact(
                "hard_skill",
                "Tailwind CSS",
                "Практический опыт Tailwind CSS",
                importance="strict",
            ),
            fact("hard_skill", "Figma", "Работа с Figma", importance="good2have"),
            fact("hard_skill", "Figma", "Знание Figma", importance="honestly_desired"),
            fact(
                "hard_skill",
                "Pinia",
                "Опыт работы с Pinia",
                importance="honestly_desired",
            ),
        ],
    }
    resume = {
        "id": "r",
        "entities": [
            fact("hard_skill", "HTML", "Верстка на HTML", polarity="present"),
            fact(
                "hard_skill",
                "Vue 3",
                "Коммерческих проектов на нём нет",
                polarity="absent",
            ),
            fact("hard_skill", "Tailwind", "С Tailwind не работал", polarity="absent"),
        ],
    }
    report = build_report(vacancy, resume)
    rows = {row["requirement"]: row for row in report["requirements"]}
    assert rows["HTML"]["status"] == "matched"
    assert rows["HTML5"]["status"] == "partial"
    assert rows["HTML5"]["resume_evidence"] == "Верстка на HTML"
    assert rows["Vue 3"]["status"] == "partial"
    assert rows["Vue.js"]["status"] == "contradicted"
    assert rows["Tailwind CSS"]["status"] == "contradicted"
    assert rows["Figma"]["status"] == "not_stated"
    assert len(rows["Figma"]["vacancy_sources"]) == 2
    assert "HTML" not in {row["requirement"] for row in report["risks"]}
    assert "HTML5" not in {row["requirement"] for row in report["risks"]}
    assert "Pinia" not in {row["requirement"] for row in report["risks"]}
    assert len(report["questions"]) <= 5
    assert len(report["questions"]) == len(set(report["questions"]))


def test_conflicting_or_unsupported_negation_is_not_overstated():
    vacancy = {"id": "v", "entities": [fact("hard_skill", "Python", "Python")]}
    resume = {
        "id": "r",
        "entities": [
            fact("hard_skill", "Python", "Изучал Python", polarity="present"),
            fact("hard_skill", "Python", "Коммерческого Python нет", polarity="absent"),
        ],
    }
    row = build_report(vacancy, resume)["requirements"][0]
    assert row["status"] == "partial"
    assert len(row["resume_sources"]) == 2
    resume["entities"] = [
        fact("hard_skill", "Python", "Участвовал в проекте", polarity="absent")
    ]
    assert build_report(vacancy, resume)["requirements"][0]["status"] == "partial"


def test_material_contradictions_are_asked_and_versions_stay_distinct():
    vacancy = {
        "entities": [
            fact(
                "hard_skill",
                "Vue.js",
                "Коммерческий опыт Vue.js",
                importance="good2have",
            ),
            fact(
                "hard_skill", "Vue 3", "Коммерческий опыт Vue 3", importance="good2have"
            ),
            fact("hard_skill", "Tailwind CSS", "Опыт Tailwind", importance="good2have"),
        ]
    }
    resume = {
        "entities": [
            fact(
                "hard_skill",
                "Vue 3",
                "Коммерческих проектов на Vue нет",
                polarity="absent",
            ),
            fact("hard_skill", "Tailwind", "С Tailwind не работал", polarity="absent"),
        ]
    }
    result = build_report(vacancy, resume, {"max_questions": 1})
    assert len(result["requirements"]) == 3
    assert len(result["questions"]) == 1
    assert "Vue" in result["questions"][0]
