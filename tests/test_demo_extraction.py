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
