"""Shared retrieval contracts and rank-based fusion for heterogeneous indexes."""

from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel


@dataclass(frozen=True)
class Match:
    """Candidate lexical match with the original entities, not a fit judgment."""

    key: str
    vacancy_entity: dict[str, Any]
    resume_entity: dict[str, Any]


@dataclass(frozen=True)
class SearchHit:
    """One ranked vacancy, its index-specific score, and supporting matches."""

    vacancy_id: str
    score: float
    matches: tuple[Match, ...] = ()
    contributions: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResults:
    """Ordered results for one index and one query, with no implicit cutoff."""

    index_name: str
    query_id: str | None
    hits: tuple[SearchHit, ...]


class VacancyIndex(Protocol):
    """Common interface for retrieval methods operating on extracted resumes."""

    def search(self, resume: dict[str, Any] | BaseModel) -> SearchResults:
        """Rank candidate vacancies for a resume."""
        ...


def compare_rankings(
    results: list[SearchResults],
) -> dict[str, dict[str, tuple[int, float] | None]]:
    """Align ranks and raw scores for each vacancy without combining scores."""
    query_ids = {result.query_id for result in results if result.query_id is not None}
    if len(query_ids) > 1:
        raise ValueError("Cannot compare rankings from different queries")
    channels = [f"{i}:{result.index_name}" for i, result in enumerate(results)]
    rows: dict[str, dict[str, tuple[int, float] | None]] = {}
    for channel, result in zip(channels, results, strict=True):
        seen: set[str] = set()
        for rank, hit in enumerate(result.hits, 1):
            if hit.vacancy_id in seen:
                raise ValueError("An input ranking contains duplicate vacancy IDs")
            seen.add(hit.vacancy_id)
            rows.setdefault(hit.vacancy_id, dict.fromkeys(channels))[channel] = (
                rank,
                hit.score,
            )
    return dict(sorted(rows.items()))


def reciprocal_rank_fusion(
    results: list[SearchResults], *, k: int = 60
) -> SearchResults:
    """Combine ranks rather than incompatible raw index scores.

    Each input is an independent channel; duplicate channel names are permitted
    and identified by their position. Unreturned vacancies are not scored.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")
    query_ids = {result.query_id for result in results if result.query_id is not None}
    if len(query_ids) > 1:
        raise ValueError("Cannot merge rankings from different queries")
    scores: dict[str, float] = {}
    matches: dict[str, list[Match]] = {}
    contributions: dict[str, dict[str, float]] = {}
    for channel, result in enumerate(results):
        seen: set[str] = set()
        for rank, hit in enumerate(result.hits, 1):
            if hit.vacancy_id in seen:
                raise ValueError("An input ranking contains duplicate vacancy IDs")
            seen.add(hit.vacancy_id)
            increment = 1 / (k + rank)
            scores[hit.vacancy_id] = scores.get(hit.vacancy_id, 0.0) + increment
            matches.setdefault(hit.vacancy_id, []).extend(hit.matches)
            contributions.setdefault(hit.vacancy_id, {})[
                f"{channel}:{result.index_name}"
            ] = increment
    ordered = sorted(scores, key=lambda vacancy_id: (-scores[vacancy_id], vacancy_id))
    return SearchResults(
        index_name="rrf",
        query_id=next(iter(query_ids), None),
        hits=tuple(
            SearchHit(
                vacancy_id,
                scores[vacancy_id],
                tuple(matches[vacancy_id]),
                contributions[vacancy_id],
            )
            for vacancy_id in ordered
        ),
    )
