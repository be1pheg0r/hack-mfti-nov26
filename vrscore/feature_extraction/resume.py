"""Shared provenance and resume feature contracts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from vrscore.documents.contracts import Evidence, ParsedDocument


class FeatureEntity(BaseModel):
    """Shared immutable provenance for features extracted from one document."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    document_id: str | None = None
    text_version: str | None = None
    evidence: dict[str, tuple[Evidence, ...]] = Field(default_factory=dict)

    def validate_document(self, document: ParsedDocument) -> None:
        """Validate source identity and all supplied field evidence against text."""
        if (self.document_id, self.text_version) != (
            document.document_id,
            document.text_version,
        ):
            raise ValueError("Features refer to a different document or text version")
        for field, citations in self.evidence.items():
            if field not in type(self).model_fields or field in {
                "document_id",
                "text_version",
                "evidence",
            }:
                raise ValueError(f"Unknown feature field: {field}")
            for citation in citations:
                document.resolve(citation)


class ResumeClaim(BaseModel):
    """An explicit positive or negative claim; missing is not negative."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    category: str | None = Field(
        default=None, description="Категория из общего словаря."
    )
    polarity: Literal["present", "absent"]
    context: str | None = None
    evidence: tuple[Evidence, ...] = ()


class ResumeFeatures(FeatureEntity):
    """Extracted resume attributes using the dataset's existing field names."""

    position_name: str | None = Field(default=None, alias="positionName")
    claims: list[ResumeClaim] | None = None
    hard_skills: list[str] | None = Field(default=None, alias="hardSkills")
    soft_skills: list[str] | None = Field(default=None, alias="softSkills")
    skills: list[str] | None = None
    experience: float | None = None
    education: str | None = None
    schedule_type: str | None = Field(default=None, alias="scheduleType")
    salary_min: float | None = Field(default=None, alias="salaryMin")
    salary_max: float | None = Field(default=None, alias="salaryMax")
    typical_position: str | None = Field(default=None, alias="typicalPosition")

    def validate_document(self, document: ParsedDocument) -> None:
        """Validate both field-level and individual claim evidence."""
        super().validate_document(document)
        for claim in self.claims or ():
            for citation in claim.evidence:
                document.resolve(citation)
