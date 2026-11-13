"""Pydantic output contracts for resume and vacancy feature extraction."""

from .resume import FeatureEntity
from .vacancy import (
    FeatureContracts,
    FeatureVocabulary,
    build_feature_contracts,
    default_feature_contracts,
)

_default_contracts = default_feature_contracts()
Requirement = _default_contracts.Requirement
VacancyFeatures = _default_contracts.VacancyFeatures
ResumeClaim = _default_contracts.ResumeClaim
ResumeFeatures = _default_contracts.ResumeFeatures

__all__ = [
    "FeatureContracts",
    "FeatureEntity",
    "FeatureVocabulary",
    "Requirement",
    "ResumeClaim",
    "ResumeFeatures",
    "VacancyFeatures",
    "build_feature_contracts",
    "default_feature_contracts",
]
