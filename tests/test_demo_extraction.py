"""Config-to-provider and evidence integration for demo extraction."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pydantic import ValidationError


def test_configured_resume_extraction_schema_and_evidence():
    from vrscore.demo.service import DemoService

    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="demo",
            overrides=[
                "llm.prompts.extraction.system='CUSTOM SYSTEM'",
                "llm.prompts.extraction.resume='CUSTOM RESUME'",
                "feature_extraction.labels.mobility='Перемещения'",
                "+feature_extraction.vocabulary.categories.custom="
                "{description:Custom,value_type:text}",
            ],
        )

    class Provider:
        def __init__(self):
            self.calls = []

        def structured(self, messages, output_type, *, public_data):
            self.calls.append((messages, output_type, public_data))
            text = messages[-1]["content"]
            quote = "Не готов к переезду" if "Не готов к переезду" in text else None
            raw = (
                []
                if not quote
                else [
                    {
                        "category": "mobility",
                        "value": "переезд",
                        "polarity": "absent",
                        "context": None,
                        "importance": None,
                        "quote": quote,
                    }
                ]
            )
            return output_type.model_validate({"entities": raw})

    provider = Provider()
    service = DemoService(
        provider,
        prompts=OmegaConf.to_container(cfg.llm.prompts.extraction, resolve=True),
        vocabulary=OmegaConf.to_container(
            cfg.feature_extraction.vocabulary, resolve=True
        ),
        labels=OmegaConf.to_container(cfg.feature_extraction.labels, resolve=True),
    )
    result = service.analyze("frontend", resume_id="r2")
    petrov = next(row for row in result["resumes"] if row["id"] == "r2")
    entity = petrov["entities"][0]
    assert (entity["category"], entity["polarity"]) == ("mobility", "absent")
    assert entity["label"] == "Перемещения"
    assert petrov["text"][entity["start"] : entity["end"]] == entity["evidence"]
    messages, schema, public_data = provider.calls[0]
    assert public_data is True
    assert "CUSTOM SYSTEM" in messages[0]["content"]
    assert "CUSTOM RESUME" in messages[0]["content"]
    assert "custom" in messages[-1]["content"]
    assert "mobility" in str(schema.model_json_schema())
    with pytest.raises(ValidationError):
        schema.model_validate(
            {
                "entities": [
                    {
                        "category": "not_a_category",
                        "value": "x",
                        "polarity": None,
                        "context": None,
                        "importance": None,
                        "quote": "x",
                    }
                ]
            }
        )

    class InvalidValues:
        def structured(self, messages, output_type, *, public_data):
            source = messages[-1]["content"].split("<source_document>\n", 1)[1]
            first_line = next(line for line in source.splitlines() if line.strip())
            common = {"polarity": None, "context": None, "importance": None}
            return output_type.model_validate(
                {
                    "entities": [
                        {
                            **common,
                            "category": "hard_skill",
                            "value": "ok",
                            "quote": first_line[:5],
                        },
                        {
                            **common,
                            "category": "education",
                            "value": "not_configured",
                            "quote": first_line[:5],
                        },
                    ]
                }
            )

    row = DemoService(InvalidValues()).analyze("frontend")["resumes"][0]
    assert row["rejected_values"] == 1
    assert len(row["entities"]) == 1

    class BlockReference:
        def structured(self, messages, output_type, *, public_data):
            content = messages[-1]["content"]
            values = []
            if "Опыт работы —7 лет\u00a09 месяцев" in content:
                values = [
                    {
                        "category": "experience",
                        "value": 7.75,
                        "polarity": "present",
                        "context": None,
                        "importance": None,
                        "quote": "Опыт работы —7 лет 9 месяцев",
                    }
                ]
            return output_type.model_validate({"entities": values})

    rows = DemoService(BlockReference()).analyze("frontend")["resumes"]
    row = next(r for r in rows if r["id"] == "r2")
    fact = row["entities"][0]
    assert fact["evidence_kind"] == "source_block"
    assert fact["model_quote"] == "Опыт работы —7 лет 9 месяцев"
    assert fact["evidence"] == "Опыт работы —7 лет\u00a09 месяцев"
    assert row["text"][fact["start"] : fact["end"]] == fact["evidence"]


def test_vacancy_extraction_keeps_requirement_importance_and_null_polarity(tmp_path):
    from vrscore.demo.service import DemoService

    class Provider:
        def __init__(self):
            self.calls = []

        def structured(self, messages, output_type, *, public_data):
            self.calls.append((messages, public_data))
            source = messages[-1]["content"].split("<source_document>\n", 1)[1]
            quote = next(line.strip() for line in source.splitlines() if line.strip())
            return output_type.model_validate(
                {
                    "entities": [
                        {
                            "category": "hard_skill",
                            "value": "Python",
                            "polarity": None,
                            "context": "обязательный навык",
                            "importance": "strict",
                            "quote": quote,
                        },
                        {
                            "category": "hard_skill",
                            "value": "Vue",
                            "polarity": "absent",
                            "context": "must not inherit resume polarity",
                            "importance": "not_configured",
                            "quote": quote,
                        },
                    ]
                }
            )

    provider = Provider()
    service = DemoService(provider, parsed_dir=tmp_path)
    result = service.extract_document(service.vacancies[0], kind="vacancy")
    entity = result["entities"][0]
    assert entity["importance"] == "strict"
    assert entity["polarity"] is None
    assert entity["evidence"] == entity["model_quote"]
    assert result["rejected_values"] == 1
    assert "This document is a vacancy" in provider.calls[0][0][0]["content"]
    assert provider.calls[0][1] is True


def test_every_approved_case_document_can_be_parsed_to_disk(tmp_path):
    import json

    from vrscore.demo.service import DemoService

    class EmptyProvider:
        def structured(self, messages, output_type, *, public_data):
            source = messages[-1]["content"]
            assert "<source_document>\n" in source
            return output_type.model_validate({"entities": []})

    service = DemoService(EmptyProvider(), parsed_dir=tmp_path)
    results = service.extract_all()
    assert len(results["vacancies"]) == 3
    assert len(results["resumes"]) == 4
    assert len(list(tmp_path.glob("*.json"))) == 7
    for item in [*results["vacancies"], *results["resumes"]]:
        saved = json.loads((tmp_path / f"{item['id']}.json").read_text())
        assert saved["document_id"] == item["document_id"]
        assert saved["entities"] == []
    catalog = DemoService(EmptyProvider(), parsed_dir=tmp_path).catalog()
    assert all("entities" in item for item in catalog["vacancies"])


def test_matching_saved_results_are_reused_and_config_changes_invalidate(tmp_path):
    from vrscore.demo.service import DemoService

    class Provider:
        def __init__(self):
            self.calls = 0

        def structured(self, messages, output_type, *, public_data):
            self.calls += 1
            return output_type.model_validate({"entities": []})

    provider = Provider()
    service = DemoService(provider, parsed_dir=tmp_path)
    service.extract_all()
    assert provider.calls == 7
    restarted = DemoService(provider, parsed_dir=tmp_path)
    restarted.catalog()
    restarted.analyze("frontend", resume_id="r1")
    restarted.analyze("frontend", resume_id="r2")
    assert provider.calls == 7
    restarted.resumes[0]["text"] += " changed"
    restarted.analyze("frontend", resume_id="r1")
    assert provider.calls == 8
    changed = dict(restarted.labels, hard_skill="Changed label")
    reconfigured = DemoService(provider, parsed_dir=tmp_path, labels=changed)
    reconfigured.analyze("frontend", resume_id="r1")
    assert provider.calls == 12

    from pydantic import BaseModel

    class Config(BaseModel):
        model: str

    provider.config = Config(model="first")
    runtime = DemoService(provider, parsed_dir=tmp_path, cache_dir=tmp_path / "runtime")
    runtime.catalog()
    runtime.analyze("frontend")
    assert provider.calls == 19
    again = DemoService(provider, parsed_dir=tmp_path, cache_dir=tmp_path / "runtime")
    again.catalog()
    again.analyze("frontend")
    assert provider.calls == 19
    provider.config.model = "second"
    again.analyze("frontend")
    assert provider.calls == 23


def test_experience_is_summed_with_all_source_references():
    from vrscore.demo.service import DemoService

    class Provider:
        def structured(self, messages, output_type, *, public_data):
            text = messages[-1]["content"]
            if "Семенов Алексей" not in text:
                return output_type.model_validate({"entities": []})
            common = {
                "category": "experience",
                "polarity": "present",
                "importance": None,
            }
            return output_type.model_validate(
                {
                    "entities": [
                        {
                            **common,
                            "value": 2.0,
                            "context": "Первая должность",
                            "quote": "март 2023 — настоящее время (2 года)",
                        },
                        {
                            **common,
                            "value": 1.5,
                            "context": "Вторая должность",
                            "quote": "сентябрь 2021 — февраль 2023 (1,5 года)",
                        },
                    ]
                }
            )

    result = DemoService(Provider()).analyze("frontend", resume_id="r4")
    row = next(row for row in result["resumes"] if row["id"] == "r4")
    assert len(row["entities"]) == 1
    total = row["entities"][0]
    assert total["value"] == 3.5
    assert len(total["sources"]) == 2
    assert all(
        row["text"][s["start"] : s["end"]] == s["evidence"] for s in total["sources"]
    )


def test_postprocessing_rules_can_be_disabled_and_preserve_other_entities():
    from vrscore.demo.postprocessing import apply_rules

    first = {"category": "experience", "value": 0.1, "polarity": "present"}
    second = {"category": "experience", "value": 0.2, "polarity": None}
    denied = {"category": "experience", "value": 1.0, "polarity": "absent"}
    skill = {"category": "hard_skill", "value": "Vue", "polarity": "present"}
    entities = [skill, first, denied, second]
    assert apply_rules(entities, []) == entities
    assert apply_rules([first], ["sum_experience"]) == [first]
    result = apply_rules(entities, ["sum_experience"])
    assert [item["value"] for item in result] == ["Vue", 0.3, 1.0]
    assert result[1]["sources"] == [first, second]
    with pytest.raises(ValueError, match="Unknown postprocessing rule"):
        apply_rules(entities, ["unknown"])
