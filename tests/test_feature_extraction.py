"""Contracts for extracted resume and vacancy attributes."""

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pydantic import BaseModel, ValidationError

from vrscore.documents import DocxReader
from vrscore.feature_extraction import (
    Requirement,
    ResumeFeatures,
    VacancyFeatures,
    build_feature_contracts,
)


def test_feature_contracts_roundtrip_dataset_attributes_and_json_schema(docx_bytes):
    """Accept dataset fields as input and expose typed JSON output contracts."""
    resume = ResumeFeatures.model_validate(
        {
            "positionName": "Инженер",
            "hardSkills": ["Python", "SQL"],
            "softSkills": ["Командная работа"],
            "experience": 5.5,
            "education": "Высшее",
            "scheduleType": "Полный день",
            "salaryMin": 120000,
            "salaryMax": 180000,
        }
    )
    vacancy = VacancyFeatures.model_validate(
        {
            "requirements": [
                {"category": "education", "value": "higher", "importance": "strict"},
                {
                    "category": "hard_skill",
                    "value": "Python",
                    "importance": "good2have",
                },
            ]
        }
    )

    assert isinstance(resume, BaseModel) and isinstance(vacancy, BaseModel)
    assert ResumeFeatures.model_validate_json(resume.model_dump_json()) == resume
    assert VacancyFeatures.model_validate_json(vacancy.model_dump_json()) == vacancy
    assert resume.model_dump(by_alias=True)["positionName"] == "Инженер"
    assert vacancy.requirements is not None
    assert vacancy.requirements[0].value == "higher"
    assert ResumeFeatures.model_json_schema()["type"] == "object"
    assert VacancyFeatures.model_json_schema()["type"] == "object"
    document = DocxReader().read(
        docx_bytes(lambda doc: doc.add_paragraph("Инженер")), filename="vacancy.docx"
    )
    linked = VacancyFeatures.model_validate(
        {
            "document_id": document.document_id,
            "text_version": document.text_version,
            "requirements": [],
        }
    )
    assert linked.document_id == document.document_id
    assert linked.text_version == document.text_version
    evidence = document.quote(document.blocks[0].block_id, 0, 7, "Инженер")
    linked = VacancyFeatures(
        document_id=document.document_id,
        text_version=document.text_version,
        requirements=(
            Requirement(category="hard_skill", value="Инженер", evidence=(evidence,)),
        ),
    )
    linked.validate_document(document)
    with pytest.raises(ValueError):
        linked.model_copy(update={"document_id": "wrong"}).validate_document(document)
    wrong_quote = evidence.model_copy(update={"text": "forged"})
    bad = VacancyFeatures(
        document_id=document.document_id,
        text_version=document.text_version,
        requirements=(
            Requirement(category="hard_skill", value="x", evidence=(wrong_quote,)),
        ),
    )
    with pytest.raises(ValueError):
        bad.validate_document(document)


def test_feature_contracts_allow_partial_extraction_but_reject_bad_types():
    """Missing fields stay unknown; malformed values and unknown keys fail."""
    assert ResumeFeatures().hard_skills is None
    assert VacancyFeatures().requirements is None
    with pytest.raises(ValidationError):
        ResumeFeatures(hardSkills="Python")
    with pytest.raises(ValidationError):
        VacancyFeatures.model_validate({"unexpected": "value"})
    resume = ResumeFeatures.model_validate(
        {
            "claims": [
                {
                    "name": "Vue.js",
                    "polarity": "absent",
                    "context": "нет коммерческого опыта",
                }
            ]
        }
    )
    assert resume.claims[0].polarity == "absent"


def test_requirement_vocabulary_from_hydra_and_validation():
    """Config overrides drive schema and validation; unknown priority stays unknown."""
    from pathlib import Path

    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="feature_extraction",
            overrides=[
                "vocabulary.categories.education.values.higher='Higher education'",
                "+vocabulary.categories.education.values.doctorate=Doctorate",
                "+vocabulary.categories.custom={description:Custom,value_type:text}",
            ],
        )
    from vrscore.feature_extraction import FeatureVocabulary

    vocabulary = FeatureVocabulary.model_validate(
        OmegaConf.to_container(cfg.vocabulary)
    )
    contracts = build_feature_contracts(vocabulary)
    requirement = contracts.Requirement(category="education", value="doctorate")
    assert requirement.importance is None
    vacancy = contracts.VacancyFeatures(requirements=(requirement,))
    assert vacancy.requirements is not None
    assert vacancy.requirements[0] == requirement
    resume = contracts.ResumeFeatures.model_validate(
        {
            "claims": [
                {"name": "qualification", "category": "custom", "polarity": "present"}
            ]
        }
    )
    assert resume.claims is not None
    assert resume.claims[0].category == "custom"
    with pytest.raises(ValidationError):
        ResumeFeatures.model_validate(
            {"claims": [{"name": "x", "category": "custom", "polarity": "present"}]}
        )
    with pytest.raises(ValidationError):
        contracts.Requirement(category="education", value="unknown")
    with pytest.raises(ValidationError):
        contracts.Requirement(category="experience", value=-1)
    with pytest.raises(ValidationError):
        contracts.Requirement(category="unknown", value="anything")
    with pytest.raises(ValidationError):
        contracts.Requirement(category="education", value="higher", importance="maybe")
    for model in (contracts.Requirement, contracts.VacancyFeatures):
        assert all(field.description for field in model.model_fields.values())
    assert "doctorate" in str(contracts.Requirement.model_json_schema())
