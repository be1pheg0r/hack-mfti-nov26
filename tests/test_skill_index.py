"""Behavioral coverage for lexical indexes and rank fusion."""

import pytest

from vrscore.feature_extraction.vacancy import default_feature_contracts
from vrscore.indexes import ExactSkillIndex, compare_rankings, reciprocal_rank_fusion


def test_exact_skill_index_matches_aliases_without_treating_negation_as_positive():
    vacancies = [
        {
            "id": "front",
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
                    "importance": "honestly_desired",
                    "evidence": "C++",
                },
            ],
        },
        {"id": "other", "entities": [{"category": "hard_skill", "value": "C#"}]},
    ]
    resume = {
        "id": "r",
        "entities": [
            {
                "category": "hard_skill",
                "value": "Vue",
                "polarity": "present",
                "evidence": "работал с Vue",
            },
            {
                "category": "hard_skill",
                "value": "C++",
                "polarity": "absent",
                "evidence": "не работал с C++",
            },
        ],
    }
    index = ExactSkillIndex(vacancies, aliases={"vue.js": ["vue"]})
    result = index.search(resume)
    assert [hit.vacancy_id for hit in result.hits] == ["front"]
    assert result.hits[0].score == 1
    match = result.hits[0].matches[0]
    assert (
        match.key,
        match.vacancy_entity["evidence"],
        match.resume_entity["evidence"],
    ) == ("vue.js", "Vue.js", "работал с Vue")
    assert result.index_name == "exact_skills"


def test_rrf_uses_ranks_not_raw_scores():
    vacancies = [
        {"id": "a", "entities": [{"category": "hard_skill", "value": "x"}]},
        {"id": "b", "entities": [{"category": "hard_skill", "value": "y"}]},
    ]
    index = ExactSkillIndex(vacancies)
    x = index.search(
        {"entities": [{"category": "hard_skill", "value": "x", "polarity": "present"}]}
    )
    y = index.search(
        {"entities": [{"category": "hard_skill", "value": "y", "polarity": "present"}]}
    )
    merged = reciprocal_rank_fusion([x, y], k=60)
    assert [hit.vacancy_id for hit in merged.hits] == ["a", "b"]
    assert merged.hits[0].score == merged.hits[1].score
    compared = compare_rankings([x, y])
    assert compared["a"]["0:exact_skills"] == (1, 1.0)
    assert compared["a"]["1:exact_skills"] is None


def test_unlisted_skills_and_safe_normalization_and_validation():
    vacancies = [
        {
            "id": "v",
            "entities": [
                {"category": "hard_skill", "value": " C++ ", "importance": "good2have"},
                {"category": "hard_skill", "value": "C++", "importance": "good2have"},
                {"category": "hard_skill", "value": "C#"},
                {"category": "hard_skill", "value": "необычный Навык"},
            ],
        }
    ]
    index = ExactSkillIndex(vacancies)
    resume = {
        "id": "r",
        "entities": [
            {"category": "hard_skill", "value": "c++", "polarity": "present"},
            {
                "category": "hard_skill",
                "value": "необычный навык",
                "polarity": "present",
            },
            {"category": "hard_skill", "value": "C#", "polarity": None},
        ],
    }
    hit = index.search(resume).hits[0]
    assert hit.score == 2
    assert {match.key for match in hit.matches} == {"c++", "необычный навык"}
    assert index.search({"entities": []}).hits == ()
    with pytest.raises(ValueError, match="ambiguous"):
        ExactSkillIndex([], aliases={"a": ["shared"], "b": ["shared"]})
    with pytest.raises(ValueError, match="Duplicate vacancy"):
        ExactSkillIndex(vacancies * 2)


def test_index_accepts_extraction_contracts_used_by_llm_notebook():
    contracts = default_feature_contracts()
    vacancy = contracts.VacancyFeatures(
        document_id="vacancy-doc",
        text_version="v1",
        requirements=(
            contracts.Requirement(
                category="hard_skill", value="Python", importance="good2have"
            ),
        ),
    )
    resume = contracts.ResumeFeatures(
        document_id="resume-doc",
        text_version="v2",
        claims=[
            contracts.ResumeClaim(
                name="python", category="hard_skill", polarity="present"
            )
        ],
    )
    result = ExactSkillIndex([vacancy]).search(resume)
    assert result.query_id == "resume-doc"
    assert result.hits[0].vacancy_id == "vacancy-doc"
    assert result.hits[0].matches[0].resume_entity["name"] == "python"
