"""Behavioral parity and retrieval contract for Tantivy exact skills."""

import pytest

from vrscore.feature_extraction.vacancy import default_feature_contracts
from vrscore.indexes import (
    ExactSkillIndex,
    TantivyExactSkillIndex,
    reciprocal_rank_fusion,
)


def test_tantivy_exact_matches_existing_index_and_fuses():
    vacancies = [
        {
            "id": "v1",
            "entities": [
                {
                    "category": "hard_skill",
                    "value": "Vue.js",
                    "importance": "good2have",
                    "evidence": "Vue.js",
                },
                {
                    "category": "hard_skill",
                    "value": "C++",
                    "importance": "strict",
                    "evidence": "C++",
                },
            ],
        },
        {
            "id": "v2",
            "entities": [
                {"category": "soft_skill", "value": "Vue.js", "importance": None},
                {"category": "hard_skill", "value": "C#", "importance": None},
            ],
        },
        {"id": "v3", "entities": []},
    ]
    resume = {
        "id": "r",
        "entities": [
            {
                "category": "hard_skill",
                "value": "Vue",
                "polarity": "present",
                "evidence": "Vue",
            },
            {"category": "hard_skill", "value": "C++", "polarity": "absent"},
            {"category": "hard_skill", "value": "C#", "polarity": "present"},
        ],
    }
    exact = ExactSkillIndex(vacancies, aliases={"vue.js": ["vue"]}).search(resume)
    tantivy = TantivyExactSkillIndex(vacancies, aliases={"vue.js": ["vue"]}).search(
        resume
    )
    assert [(x.vacancy_id, x.score) for x in tantivy.hits] == [
        (x.vacancy_id, x.score) for x in exact.hits
    ]
    assert tantivy.index_name == "tantivy_exact_skills"
    assert tantivy.query_id == exact.query_id == "r"
    assert tantivy.hits[0].matches[0].resume_entity["evidence"] == "Vue"
    assert {x.vacancy_id for x in reciprocal_rank_fusion([exact, tantivy]).hits} == {
        "v1",
        "v2",
    }
    assert TantivyExactSkillIndex([]).search(resume).hits == ()
    with pytest.raises(ValueError, match="Duplicate"):
        TantivyExactSkillIndex(vacancies + [vacancies[0]])


def test_tantivy_accepts_real_feature_contracts():
    c = default_feature_contracts()
    vacancy = c.VacancyFeatures(
        document_id="v",
        text_version="x",
        requirements=(c.Requirement(category="hard_skill", value=".NET"),),
    )
    resume = c.ResumeFeatures(
        document_id="r",
        text_version="x",
        claims=[
            c.ResumeClaim(name=".net", category="hard_skill", polarity="present"),
        ],
    )
    hit = TantivyExactSkillIndex([vacancy]).search(resume).hits[0]
    assert hit.vacancy_id == "v"
    assert hit.matches[0].key == ".net"


def test_tantivy_parity_over_many_hits_and_repeated_evidence():
    vacancies = [
        {
            "id": f"v{i:02}",
            "entities": [
                {
                    "category": "hard_skill",
                    "value": 'C++ "test"',
                    "importance": "strict",
                },
                {"category": "hard_skill", "value": 'C++ "test"', "importance": None},
            ],
        }
        for i in range(35)
    ]
    resume = {
        "id": "r",
        "entities": [
            {"category": "hard_skill", "value": 'c++ "test"', "polarity": "present"},
            {"category": "hard_skill", "value": 'c++ "test"', "polarity": "present"},
        ],
    }
    config = {"importance_weights": {"strict": 2.0}}
    exact = ExactSkillIndex(vacancies, config=config).search(resume)
    tantivy = TantivyExactSkillIndex(vacancies, config=config).search(resume)
    assert len(tantivy.hits) == 35
    assert [(h.vacancy_id, h.score, h.matches) for h in tantivy.hits] == [
        (h.vacancy_id, h.score, h.matches) for h in exact.hits
    ]
