"""Config-bound LLM extraction schema and category validation for the demo."""

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, create_model

from vrscore.documents.contracts import ParsedDocument
from vrscore.feature_extraction.vacancy import FeatureVocabulary


class Entity(BaseModel):
    """An atomic fact and the model's verbatim source quotation."""

    model_config = ConfigDict(extra="forbid")
    category: str = Field(description="Category code from the supplied catalog")
    value: str | StrictFloat = Field(description="Value according to category type")
    polarity: Literal["present", "absent"] | None
    context: str | None
    importance: str | None
    quote: str = Field(min_length=1, description="Exact continuous source quote")


class Extracted(BaseModel):
    """Structured model response, specialized with a configured category enum."""

    model_config = ConfigDict(extra="forbid")
    entities: list[Entity]


def build_extraction_model(vocabulary: FeatureVocabulary) -> type[Extracted]:
    """Build JSON Schema from the same Hydra vocabulary used in the prompt."""
    literal: Any = Literal
    categories: Any = literal[tuple(vocabulary.categories)]
    entity = create_model(
        "ConfiguredEntity",
        __base__=Entity,
        category=(categories, Field(description="Category from the supplied catalog")),
    )
    return create_model(
        "ConfiguredExtraction", __base__=Extracted, entities=(list[entity], ...)
    )


class VacancyValue(BaseModel):
    """One value and its exact evidence, without model-assigned priority."""

    model_config = ConfigDict(extra="forbid")
    value: str | StrictFloat
    context: str | None
    quote: str = Field(min_length=1)


def build_vacancy_model(
    vocabulary: FeatureVocabulary, template: dict
) -> type[BaseModel]:
    """Build fixed output slots whose categories and priorities come from config."""
    fields = {}
    for name, slot in template.items():
        if slot["category"] not in vocabulary.categories:
            raise ValueError(f"Unknown vacancy category: {slot['category']}")
        if (
            slot["importance"] is not None
            and slot["importance"] not in vocabulary.importance
        ):
            raise ValueError(f"Unknown vacancy priority: {slot['importance']}")
        fields[name] = (list[VacancyValue], Field(description=slot["description"]))
    return create_model(
        "VacancyTemplate", __config__=ConfigDict(extra="forbid"), **fields
    )


def valid_entity(
    entity: Entity, vocabulary: FeatureVocabulary, *, kind: str = "resume"
) -> bool:
    """Enforce category, value, and document-specific extraction semantics."""
    definition = vocabulary.categories.get(entity.category)
    if definition is None or kind not in {"resume", "vacancy"}:
        return False
    allowed_importance = set(vocabulary.importance)
    if kind == "resume" and entity.importance is not None:
        return False
    if kind == "vacancy" and (
        entity.polarity is not None
        or entity.importance is not None
        and entity.importance not in allowed_importance
    ):
        return False
    value = entity.value
    if definition.value_type == "number":
        return isinstance(value, float) and math.isfinite(value) and value >= 0
    if not isinstance(value, str) or not value.strip():
        return False
    return definition.value_type != "enum" or value in definition.values


def source_reference(document: ParsedDocument, quote: str) -> dict | None:
    """Use exact evidence, or a disclosed whole-block reference for typography.

    A model quote is never rewritten into an exact quote. If only a unique
    paragraph matches after normalizing non-breaking spaces and hyphens, cite
    the full original source block and retain the model's quote separately.
    """
    for block in document.blocks:
        offset = block.text.find(quote)
        if offset < 0:
            continue
        start = block.start + offset
        try:
            citation = document.quote(block.block_id, start, start + len(quote), quote)
        except ValueError:
            continue
        return {
            "evidence": citation.text,
            "start": citation.start,
            "end": citation.end,
            "block_id": citation.block_id,
            "evidence_kind": "exact_quote",
            "model_quote": quote,
        }

    table = str.maketrans({"\u00a0": " ", "\u202f": " ", "\u2010": "-", "\u2011": "-"})
    wanted = quote.translate(table)
    blocks = [
        block for block in document.blocks if wanted in block.text.translate(table)
    ]
    if len(blocks) != 1:
        return None
    block = blocks[0]
    return {
        "evidence": block.text,
        "start": block.start,
        "end": block.end,
        "block_id": block.block_id,
        "evidence_kind": "source_block",
        "model_quote": quote,
    }
