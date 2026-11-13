"""Behavioral tests for BM25 and fusion with the existing exact index."""

import hashlib

import pytest
from rank_bm25 import BM25Plus

from vrscore.feature_extraction.vacancy import default_feature_contracts
from vrscore.indexes import (
    BM25SkillIndex,
    ExactSkillIndex,
    reciprocal_rank_fusion,
    tokenize_skill,
)


def test_bm25_merges_skill_categories_and_fuses_with_exact():
    vacancies = [
        {
            "id": "v1",
            "entities": [
                {"category": "hard_skill", "value": "Python", "evidence": "Python"},
            ],
        },
        {
            "id": "v2",
            "entities": [
                {
                    "category": "soft_skill",
                    "value": "Python разработка",
                    "evidence": "Python разработка",
                },
            ],
        },
        {
            "id": "v3",
            "entities": [
                {"category": "hard_skill", "value": "C++", "evidence": "C++"},
            ],
        },
    ]
    resume = {
        "id": "r",
        "entities": [
            {
                "category": "hard_skill",
                "value": "Python",
                "polarity": "present",
                "evidence": "Python",
            },
            {"category": "hard_skill", "value": "C++", "polarity": "absent"},
        ],
    }
    exact = ExactSkillIndex(vacancies).search(resume)
    bm25 = BM25SkillIndex(vacancies).search(resume)
    assert [hit.vacancy_id for hit in exact.hits] == ["v1"]
    assert [hit.vacancy_id for hit in bm25.hits] == ["v1", "v2"]
    # Compare with the dependency, not a second homemade BM25 implementation.
    expected = BM25Plus(
        [["python"], ["python", "разработка"], ["c++"]],
        k1=1.2,
        b=0.75,
        delta=0.0,
    ).get_scores(["python"])[0]
    assert bm25.hits[0].score == pytest.approx(expected)
    match = bm25.hits[1].matches[0]
    assert (
        match.key,
        match.vacancy_entity["evidence"],
        match.resume_entity["evidence"],
    ) == ("python", "Python разработка", "Python")
    combined = reciprocal_rank_fusion([exact, bm25], k=60)
    assert [hit.vacancy_id for hit in combined.hits] == ["v1", "v2"]
    assert combined.query_id == "r"
    assert len(combined.hits[0].contributions) == 2
    assert len(combined.hits[1].contributions) == 1
    assert {match.key for match in combined.hits[0].matches} == {"python"}


def test_bm25_uses_real_contracts_and_can_keep_categories_separate():
    contracts = default_feature_contracts()
    vacancy = contracts.VacancyFeatures(
        document_id="v",
        text_version="v1",
        requirements=(contracts.Requirement(category="soft_skill", value="C++"),),
    )
    resume = contracts.ResumeFeatures(
        document_id="r",
        text_version="v2",
        claims=[
            contracts.ResumeClaim(
                name="C++", category="hard_skill", polarity="present"
            ),
            contracts.ResumeClaim(name="C#", category="hard_skill", polarity="absent"),
        ],
    )
    merged = BM25SkillIndex([vacancy]).search(resume)
    assert [hit.vacancy_id for hit in merged.hits] == ["v"]
    separate = BM25SkillIndex([vacancy], config={"merge_skill_categories": False})
    assert separate.search(resume).hits == ()
    assert tokenize_skill("C++ C# .NET Vue.js Node.js 1С") == [
        "c++",
        "c#",
        ".net",
        "vue.js",
        "node.js",
        "1с",
    ]
    assert BM25SkillIndex([]).search(resume).hits == ()
    with pytest.raises(ValueError, match="Duplicate"):
        BM25SkillIndex([vacancy, vacancy])
    with pytest.raises(ValueError, match="k1"):
        BM25SkillIndex([], config={"k1": 0})


def test_stopwords_filter_applies_to_both_documents_and_queries():
    vacancies = [
        {
            "id": "java",
            "entities": [{"category": "hard_skill", "value": "с и в Java"}],
        },
        {"id": "cpp", "entities": [{"category": "hard_skill", "value": "C++"}]},
    ]
    resume = {
        "id": "r",
        "entities": [
            {
                "category": "hard_skill",
                "value": "с и в Python",
                "polarity": "present",
            },
            {"category": "hard_skill", "value": "C++", "polarity": "present"},
        ],
    }
    disabled = BM25SkillIndex(vacancies, config={"stopwords": {"enabled": False}})
    enabled = BM25SkillIndex(vacancies, config={"stopwords": {"enabled": True}})
    assert {x.vacancy_id for x in disabled.search(resume).hits} == {"java", "cpp"}
    assert [x.vacancy_id for x in enabled.search(resume).hits] == ["cpp"]
    only_stop = {
        "id": "r",
        "entities": [
            {"category": "hard_skill", "value": "с и в", "polarity": "present"}
        ],
    }
    assert enabled.search(only_stop).hits == ()
    assert (
        BM25SkillIndex(
            [{"id": "v", "entities": [{"category": "hard_skill", "value": "с и в"}]}]
        )
        .search(resume)
        .hits
        == ()
    )
    override = BM25SkillIndex(
        vacancies, config={"stopwords": {"enabled": True, "keep": ["с"]}}
    )
    assert {x.vacancy_id for x in override.search(resume).hits} == {"java", "cpp"}
    assert {"c++", "c#", ".net", "node.js", "python", "1с"}.isdisjoint(
        enabled.stopwords
    )
    assert {"работа", "работать", "уметь", "без", "не", "нет"}.isdisjoint(
        enabled.stopwords
    )
    with pytest.raises((ValueError, OSError), match="stopword|No such file"):
        BM25SkillIndex(
            [], config={"stopwords": {"enabled": True, "files": ["missing.txt"]}}
        )


def test_russian_stopword_dictionary_does_not_drop_programming_language_c():
    """Latin C is a real skill although it appears in the Russian stoplist."""
    vacancy = {"id": "v", "entities": [{"category": "hard_skill", "value": "C"}]}
    resume = {
        "id": "r",
        "entities": [{"category": "hard_skill", "value": "C", "polarity": "present"}],
    }
    index = BM25SkillIndex([vacancy])
    assert "c" not in index.stopwords
    assert [hit.vacancy_id for hit in index.search(resume).hits] == ["v"]


def test_stopwords_can_be_edited_in_a_local_file(tmp_path, monkeypatch):
    source = tmp_path / "stopwords.txt"
    source.write_text("# comment\nСЛОВО\n\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    index = BM25SkillIndex([], config={"stopwords": {"files": [str(source)]}})
    assert index.stopwords == {"слово"}
    assert (
        index.stopword_sources[str(source)]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert BM25SkillIndex(
        []
    ).stopwords  # Project-relative defaults work from another cwd.
    source.write_text("two words\n", encoding="utf-8")
    with pytest.raises(ValueError, match="one token"):
        BM25SkillIndex([], config={"stopwords": {"files": [str(source)]}})
