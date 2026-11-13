"""Strict first-gate decisions over the repository's parsed entity JSON."""

import json
from pathlib import Path

from vrscore.gates import evaluate_first_gate

DATA = Path(__file__).parents[1] / "public_data" / "parsed"


def test_missing_vacancy_values_use_labeled_defaults_and_resume_null_is_unknown():
    from vrscore.gates import apply_vacancy_defaults

    vacancy = {
        "kind": "vacancy",
        "entities": [
            {"category": "experience", "value": None},
            {"category": "education", "value": None},
        ],
    }
    normalized = apply_vacancy_defaults(vacancy)
    defaults = {e["category"]: e for e in normalized["entities"]}
    expected = {
        "experience": 0,
        "education": "secondary_general",
        "work_schedule": "full_time",
        "work_format": "on_site",
        "employment_type": "contract",
        "salary_min": 0,
        "salary_max": 1000000,
    }
    assert {key: defaults[key]["value"] for key in expected} == expected
    assert all(defaults[key]["derived"] for key in expected)
    assert vacancy["entities"][0]["value"] is None
    resume = {
        "entities": [
            {"category": "experience", "value": 2},
            {"category": "education", "value": "bachelor"},
            {"category": "work_format", "value": "on_site"},
            {"category": "work_schedule", "value": "full_time"},
        ]
    }
    assert evaluate_first_gate(vacancy, resume)["status"] == "pass"
    resume["entities"][1]["value"] = None
    result = evaluate_first_gate(vacancy, resume)
    assert result["status"] == "unknown"
    assert result["decisions"][-1]["status"] == "unknown"
    assert result["decisions"][-1]["reason"] == "resume_not_stated"
    resume["entities"][1]["value"] = "bachelor"
    resume["entities"][2]["value"] = "remote"
    resume["entities"][3]["value"] = "shift"
    assert evaluate_first_gate(vacancy, resume)["status"] == "pass"
    overridden = apply_vacancy_defaults(
        normalized, {"vacancy_defaults": {"salary_max": 12}}
    )
    assert (
        next(
            e["value"] for e in overridden["entities"] if e["category"] == "salary_max"
        )
        == 12
    )


def test_strict_schedule_missing_in_resume_is_not_stated():
    config = {"filters": {"order": ["work_schedule"]}}
    vacancy = {"entities": [{"category": "work_schedule", "value": "full_time"}]}
    resume = {"entities": [{"category": "work_schedule", "value": None}]}
    result = evaluate_first_gate(vacancy, resume, config)
    assert result["status"] == "unknown"
    assert result["decisions"][0]["reason"] == "resume_not_stated"
    resume["entities"][0]["value"] = "full_time"
    assert evaluate_first_gate(vacancy, resume, config)["status"] == "pass"
    resume["entities"][0]["value"] = "shift"
    assert evaluate_first_gate(vacancy, resume, config)["status"] == "fail"


def parsed(identifier):
    return json.loads((DATA / f"{identifier}.json").read_text(encoding="utf-8"))


def test_actual_parsed_entities_gate_in_order_and_preserve_evidence():
    vacancy = parsed("accountant")
    next(e for e in vacancy["entities"] if e["category"] == "experience")["value"] = 2.0
    result = evaluate_first_gate(vacancy, parsed("r1"))
    assert result["status"] == "unknown"
    assert [item["filter"] for item in result["decisions"]] == [
        "experience",
        "education",
    ]
    education = result["decisions"][-1]
    assert education["status"] == "unknown"
    assert education["candidate"][0]["context"] == "Неоконченное высшее"
    assert education["candidate"][0]["evidence"] == "Неоконченное высшее"


def test_actual_parsed_work_format_alternatives_and_overlap_caveat():
    result = evaluate_first_gate(parsed("accountant"), parsed("r2"))
    assert result["status"] == "unknown"
    assert [item["filter"] for item in result["decisions"]] == [
        "experience",
        "education",
        "work_format",
        "work_schedule",
    ]
    experience = result["decisions"][0]
    assert experience["status"] == "pass"
    assert "Пересечения и дубли не проверены" in experience["candidate"][0]["context"]
    assert "overlaps/duplicates were not checked" in experience["explanation"]
    assert len(experience["candidate"][0]["sources"]) == 2
    work_format = result["decisions"][2]
    assert work_format["status"] == "pass"
    assert work_format["candidate"][0]["evidence_kind"] == "source_block"
    for resume_id in ("r3", "r4"):
        experience = evaluate_first_gate(parsed("accountant"), parsed(resume_id))[
            "decisions"
        ][0]
        assert experience["status"] == "pass"
        assert "overlaps/duplicates were not checked" in experience["explanation"]
        assert experience["candidate"][0]["sources"]


def test_rule_objects_share_context_result_and_gate_supports_extension():
    from vrscore.gates import RuleContext

    vacancy = {"kind": "vacancy", "entities": []}
    resume = {"kind": "resume", "entities": []}

    class AlwaysPass:
        name = "custom"

        def evaluate(self, context: RuleContext) -> dict:
            return {"filter": self.name, "status": "pass", "explanation": "custom"}

    result = evaluate_first_gate(
        vacancy,
        resume,
        config={"filters": {"order": ["custom"], "categories": {}}},
        rule_registry={"custom": AlwaysPass},
    )
    assert result == {
        "status": "pass",
        "decisions": [{"filter": "custom", "status": "pass", "explanation": "custom"}],
    }


def test_unknown_short_circuits_and_requirements_ignore_importance():
    vacancy = {
        "kind": "vacancy",
        "entities": [
            {"category": "experience", "value": 2, "importance": "good2have"},
            {"category": "education", "value": "bachelor", "importance": None},
            {"category": "work_format", "value": "remote", "importance": None},
        ],
    }
    resume = {"kind": "resume", "entities": []}
    result = evaluate_first_gate(vacancy, resume)
    assert result["status"] == "unknown"
    assert [item["filter"] for item in result["decisions"]] == ["experience"]


def test_incomplete_education_can_only_count_when_explicitly_configured():
    vacancy = parsed("accountant")
    next(e for e in vacancy["entities"] if e["category"] == "experience")["value"] = 2.0
    config = {
        "education": {"allow_incomplete": True},
    }
    result = evaluate_first_gate(vacancy, parsed("r1"), config=config)
    assert result["decisions"][1]["status"] == "pass"
    assert result["decisions"][1]["candidate"][0]["evidence"] == "Неоконченное высшее"


def test_numeric_and_education_differences_are_inclusive_and_configurable():
    vacancy = {
        "kind": "vacancy",
        "entities": [
            {"category": "experience", "value": 3},
            {"category": "education", "value": "bachelor"},
            {"category": "work_format", "value": "on_site"},
        ],
    }
    resume = {
        "kind": "resume",
        "entities": [
            {"category": "experience", "value": 3},
            {"category": "education", "value": "candidate_of_sciences"},
            {"category": "work_format", "value": "on_site"},
            {"category": "work_schedule", "value": "full_time"},
        ],
    }
    result = evaluate_first_gate(vacancy, resume)
    assert result["status"] == "pass"
    assert result["decisions"][0]["difference"] == 0
    assert result["decisions"][1]["difference"] >= 0


def test_explicitly_absent_format_does_not_satisfy_requirement():
    vacancy = {
        "kind": "vacancy",
        "entities": [
            {"category": "experience", "value": 1},
            {"category": "education", "value": "bachelor"},
            {"category": "work_format", "value": "on_site"},
        ],
    }
    resume = {
        "kind": "resume",
        "entities": [
            {"category": "experience", "value": 1},
            {"category": "education", "value": "bachelor"},
            {"category": "work_format", "value": "on_site", "polarity": "absent"},
        ],
    }
    assert evaluate_first_gate(vacancy, resume)["status"] == "fail"


def test_format_compatibility_rules():
    for required, offered, expected in [
        ("hybrid", {"hybrid"}, "pass"),
        ("hybrid", {"remote"}, "fail"),
        ("hybrid", {"on_site"}, "fail"),
        ("remote", {"hybrid", "remote"}, "pass"),
        ("remote", {"hybrid"}, "pass"),
        ("remote", {"remote"}, "pass"),
        ("remote", {"on_site"}, "fail"),
        ("on_site", {"on_site"}, "pass"),
        ("on_site", {"hybrid"}, "fail"),
        ("on_site", {"remote"}, "fail"),
        ("on_site", {"hybrid", "remote"}, "fail"),
    ]:
        vacancy = {
            "kind": "vacancy",
            "entities": [
                {"category": "experience", "value": 1},
                {"category": "education", "value": "bachelor"},
                {"category": "work_format", "value": required},
            ],
        }
        resume = {
            "kind": "resume",
            "entities": [
                {"category": "experience", "value": 1},
                {"category": "education", "value": "bachelor"},
                *[{"category": "work_format", "value": item} for item in offered],
                {"category": "work_schedule", "value": "full_time"},
            ],
        }
        assert evaluate_first_gate(vacancy, resume)["status"] == expected


def test_unconfirmed_or_invalid_facts_remain_unknown():
    vacancy = {"entities": [{"category": "experience", "value": 1}]}
    for value, polarity in [(3, "absent"), ("invalid", None), (float("nan"), None)]:
        resume = {
            "entities": [
                {"category": "experience", "value": value, "polarity": polarity}
            ]
        }
        assert evaluate_first_gate(vacancy, resume)["status"] == "unknown"


def test_vocabulary_education_levels_and_unrelated_negative_format():
    from vrscore.gates import EducationRule, RuleContext, WorkFormatRule, _config

    config = _config(None)

    def context(category, required, candidate, polarity=None):
        return RuleContext(
            vacancy={"entities": [{"category": category, "value": required}]},
            resume={
                "entities": [
                    {"category": category, "value": candidate, "polarity": polarity}
                ]
            },
            config=config,
        )

    assert (
        EducationRule().evaluate(
            context("education", "secondary_general", "secondary_vocational")
        )["status"]
        == "pass"
    )
    assert (
        EducationRule().evaluate(context("education", "master", "postgraduate"))[
            "status"
        ]
        == "pass"
    )
    assert (
        EducationRule().evaluate(context("education", "master", "higher"))["status"]
        == "unknown"
    )
    assert (
        WorkFormatRule().evaluate(
            context("work_format", "on_site", "remote", "absent")
        )["status"]
        == "unknown"
    )


def test_alternative_education_levels_in_same_quote_do_not_become_both_required():
    from vrscore.gates import EducationRule, RuleContext, _config

    quote = "Среднее профессиональное или высшее образование"
    vacancy = {
        "entities": [
            {"category": "education", "value": value, "evidence": quote}
            for value in ("secondary_vocational", "higher")
        ]
    }
    resume = {"entities": [{"category": "education", "value": "secondary_vocational"}]}
    result = EducationRule().evaluate(RuleContext(vacancy, resume, _config(None)))
    assert result["status"] == "pass"
