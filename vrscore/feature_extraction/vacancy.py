"""Config-driven vacancy requirement contracts."""

from dataclasses import dataclass
from pathlib import Path
from types import GenericAlias
from typing import Any, ClassVar, Literal

from omegaconf import OmegaConf
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    create_model,
    model_validator,
)

from vrscore.documents.contracts import Evidence, ParsedDocument

from .resume import ResumeClaim as BaseResumeClaim
from .resume import ResumeFeatures as BaseResumeFeatures


class CategoryDefinition(BaseModel):
    """Configuration for one requirement's value domain."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    value_type: Literal["enum", "text", "number"]
    values: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_values(self) -> "CategoryDefinition":
        """Reject invalid vocabularies instead of silently accepting any value."""
        if self.value_type == "enum" and not self.values:
            raise ValueError("An enum category needs at least one value")
        if self.value_type != "enum" and self.values:
            raise ValueError("Only enum categories can define values")
        if any(
            not key.strip() or not desc.strip() for key, desc in self.values.items()
        ):
            raise ValueError("Value codes and descriptions must not be empty")
        return self


class FeatureVocabulary(BaseModel):
    """Catalog shared by all contracts generated for an extraction run."""

    model_config = ConfigDict(extra="forbid")

    importance: dict[str, str] = Field(min_length=1)
    categories: dict[str, CategoryDefinition] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_codes(self) -> "FeatureVocabulary":
        """Reject empty category and importance identifiers and descriptions."""
        if any(
            not code.strip() or not description.strip()
            for code, description in self.importance.items()
        ):
            raise ValueError("Importance codes and descriptions must not be empty")
        if any(not code.strip() for code in self.categories):
            raise ValueError("Category codes must not be empty")
        return self


class _Requirement(BaseModel):
    """One atomic requirement, interpreted under its configured vocabulary."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    _vocabulary: ClassVar[FeatureVocabulary]

    category: str = Field(description="Категория требования из настроенного словаря.")
    value: str | StrictFloat = Field(
        description="Нормализованное значение: код, текст или число по категории."
    )
    importance: str | None = Field(
        default=None,
        description="Обязательность из словаря; null, если в источнике она не указана.",
    )
    context: str | None = Field(
        default=None,
        description="Уточнение требования: область опыта, уровень языка или условие.",
    )
    evidence: tuple[Evidence, ...] = Field(
        default=(), description="Дословные подтверждения в исходном документе вакансии."
    )

    @model_validator(mode="after")
    def validate_value(self) -> "_Requirement":
        """Enforce the category-specific value domain after Pydantic validation."""
        category = type(self)._vocabulary.categories[self.category]
        if category.value_type == "number":
            if not isinstance(self.value, float) or self.value < 0:
                raise ValueError("A numeric requirement must be non-negative")
        elif not isinstance(self.value, str) or not self.value.strip():
            raise ValueError("A textual requirement must be a non-empty string")
        elif category.value_type == "enum" and self.value not in category.values:
            raise ValueError(f"Value is not in configured category {self.category}")
        return self


class _VacancyFeatures(BaseModel):
    """Document identity and a compact collection of atomic requirements."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str | None = Field(
        default=None, description="Идентификатор исходного прочитанного документа."
    )
    text_version: str | None = Field(
        default=None, description="Версия текста исходного документа."
    )
    requirements: tuple[_Requirement, ...] | None = Field(
        default=None, description="Атомарные требования вакансии."
    )

    def validate_document(self, document: ParsedDocument) -> None:
        """Check source identity and each supplied exact quotation."""
        if (self.document_id, self.text_version) != (
            document.document_id,
            document.text_version,
        ):
            raise ValueError("Features refer to a different document or text version")
        for requirement in self.requirements or ():
            for evidence in requirement.evidence:
                document.resolve(evidence)


@dataclass(frozen=True)
class FeatureContracts:
    """Pydantic classes bound to one immutable vocabulary snapshot."""

    Requirement: type[_Requirement]
    VacancyFeatures: type[_VacancyFeatures]
    ResumeClaim: type[BaseResumeClaim]
    ResumeFeatures: type[BaseResumeFeatures]


def build_feature_contracts(vocabulary: FeatureVocabulary | dict) -> FeatureContracts:
    """Build vocabulary-specific requirement and vacancy models.

    Hydra may compose/override the configuration before this function is called.
    Codes and categorical value choices are included in the exported JSON Schema;
    category/value compatibility is also checked by the runtime validator.
    """
    config = FeatureVocabulary.model_validate(vocabulary).model_copy(deep=True)
    literal_factory: Any = Literal
    model_factory: Any = create_model
    category_type: Any = literal_factory[tuple(config.categories)]
    importance_type: Any = literal_factory[tuple(config.importance)] | None
    category_desc = "Категории: " + "; ".join(
        f"{code}: {item.description}" for code, item in config.categories.items()
    )
    value_desc = "Значение по категории. " + "; ".join(
        f"{code}: "
        + (
            ", ".join(f"{key} ({desc})" for key, desc in item.values.items())
            if item.value_type == "enum"
            else item.value_type
        )
        for code, item in config.categories.items()
    )
    importance_desc = "Обязательность или null, если не указана: " + "; ".join(
        f"{code}: {description}" for code, description in config.importance.items()
    )
    conditions = []
    for code, item in config.categories.items():
        if item.value_type == "enum":
            constraint = {"enum": list(item.values)}
        elif item.value_type == "number":
            constraint = {"type": "number", "minimum": 0}
        else:
            constraint = {"type": "string", "minLength": 1}
        conditions.append(
            {
                "if": {
                    "properties": {"category": {"const": code}},
                    "required": ["category"],
                },
                "then": {"properties": {"value": constraint}},
            }
        )
    requirement = model_factory(
        "Requirement",
        __base__=_Requirement,
        __module__=__name__,
        category=(category_type, Field(description=category_desc)),
        value=(str | StrictFloat, Field(description=value_desc)),
        importance=(importance_type, Field(default=None, description=importance_desc)),
        __config__=ConfigDict(json_schema_extra={"allOf": conditions}),
    )
    requirement._vocabulary = config
    vacancy = model_factory(
        "VacancyFeatures",
        __base__=_VacancyFeatures,
        __module__=__name__,
        requirements=(
            GenericAlias(tuple, (requirement, Ellipsis)) | None,
            Field(
                default=None,
                description="Атомарные требования; null — они не извлечены.",
            ),
        ),
    )
    resume_claim = model_factory(
        "ResumeClaim",
        __base__=BaseResumeClaim,
        __module__=__name__,
        category=(category_type | None, Field(default=None, description=category_desc)),
    )
    resume = model_factory(
        "ResumeFeatures",
        __base__=BaseResumeFeatures,
        __module__=__name__,
        claims=(GenericAlias(list, resume_claim) | None, None),
    )
    return FeatureContracts(
        Requirement=requirement,
        VacancyFeatures=vacancy,
        ResumeClaim=resume_claim,
        ResumeFeatures=resume,
    )


def default_feature_contracts() -> FeatureContracts:
    """Load the project's default vocabulary; use the builder for Hydra overrides."""
    path = Path(__file__).resolve().parents[2] / "configs" / "feature_extraction.yaml"
    config = OmegaConf.load(path)
    vocabulary = OmegaConf.to_container(config.vocabulary, resolve=True)
    return build_feature_contracts(FeatureVocabulary.model_validate(vocabulary))
