"""Common index interfaces and exact skill retrieval."""

from .base import (
    Match,
    SearchHit,
    SearchResults,
    VacancyIndex,
    compare_rankings,
    reciprocal_rank_fusion,
)
from .exact_skills import ExactSkillIndex, Posting, normalize_skill

__all__ = [
    "ExactSkillIndex",
    "Match",
    "Posting",
    "SearchHit",
    "SearchResults",
    "VacancyIndex",
    "compare_rankings",
    "normalize_skill",
    "reciprocal_rank_fusion",
]
