"""In-memory inverted index over explicitly extracted skill entities."""

import unicodedata
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from pydantic import BaseModel

from .base import Match, SearchHit, SearchResults

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "exact_skills_index.yaml"
)


def normalize_skill(value: str) -> str:
    """Normalize case and whitespace without stripping skill punctuation."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


@dataclass(frozen=True)
class Posting:
    """One vacancy entity associated with a canonical skill key."""

    vacancy_id: str
    entity: dict[str, Any]


class ExactSkillIndex:
    """A rebuildable inverted index; exact hits are candidates, not fit decisions.

    Accepts demo records with `id` and `entities` or the Pydantic feature
    contracts from the LLM notebook. Only explicit positive resume claims are
    queries; missing/negative polarity is excluded.
    """

    def __init__(
        self,
        vacancies: list[dict[str, Any] | BaseModel],
        *,
        config: dict[str, Any] | None = None,
        aliases: dict[str, list[str]] | None = None,
    ) -> None:
        settings = OmegaConf.merge(
            OmegaConf.load(DEFAULT_CONFIG), OmegaConf.create(config or {})
        )
        if aliases is not None:
            settings.aliases = OmegaConf.create(aliases)
        self.name = str(settings.name)
        self.categories = frozenset(settings.categories)
        self.weights = dict(settings.importance_weights)
        if any(
            not isinstance(w, (int, float))
            or isinstance(w, bool)
            or not isfinite(w)
            or w < 0
            for w in self.weights.values()
        ):
            raise ValueError("Importance weights must be finite, non-negative numbers")
        if "unknown" not in self.weights:
            raise ValueError("An unknown importance weight is required")
        self._aliases: dict[str, str] = {}
        for canonical, values in settings.aliases.items():
            target = normalize_skill(canonical)
            if not target:
                raise ValueError("Empty canonical skill")
            for value in (canonical, *values):
                key = normalize_skill(value)
                if not key or (key in self._aliases and self._aliases[key] != target):
                    raise ValueError(f"Empty or ambiguous skill alias: {value!r}")
                self._aliases[key] = target
        postings: dict[str, list[Posting]] = defaultdict(list)
        seen_ids: set[str] = set()
        for vacancy in vacancies:
            if isinstance(vacancy, BaseModel):
                vacancy = vacancy.model_dump()
            vacancy_id = vacancy.get("id") or vacancy.get("document_id")
            if not isinstance(vacancy_id, str) or not vacancy_id.strip():
                raise ValueError("Each vacancy needs a non-empty string id")
            if vacancy_id in seen_ids:
                raise ValueError(f"Duplicate vacancy id: {vacancy_id}")
            seen_ids.add(vacancy_id)
            for entity in vacancy.get("entities", vacancy.get("requirements")) or []:
                if (
                    not isinstance(entity, dict)
                    or entity.get("category") not in self.categories
                ):
                    continue
                value = entity.get("value")
                if (
                    not isinstance(value, str)
                    or not value.strip()
                    or entity.get("defaulted")
                ):
                    continue
                key = self._canonical(value)
                postings[key].append(Posting(vacancy_id, deepcopy(entity)))
        self.postings = {key: tuple(value) for key, value in postings.items()}
        self.vacancy_ids = tuple(sorted(seen_ids))

    def _canonical(self, value: str) -> str:
        key = normalize_skill(value)
        return self._aliases.get(key, key)

    def search(self, resume: dict[str, Any] | BaseModel) -> SearchResults:
        """Find positive matches and sum each distinct requirement skill once."""
        if isinstance(resume, BaseModel):
            resume = resume.model_dump()
        grouped: dict[str, list[Match]] = defaultdict(list)
        scores: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
        for entity in resume.get("entities", resume.get("claims")) or []:
            if not isinstance(entity, dict) or entity.get("polarity") != "present":
                continue
            category = entity.get("category")
            value = entity.get("value", entity.get("name"))
            if (
                category not in self.categories
                or not isinstance(value, str)
                or not value.strip()
            ):
                continue
            key = self._canonical(value)
            for posting in self.postings.get(key, ()):
                if posting.entity.get("category") != category:
                    continue
                importance = posting.entity.get("importance") or "unknown"
                if importance not in self.weights:
                    raise ValueError(
                        f"No configured weight for importance {importance!r}"
                    )
                grouped[posting.vacancy_id].append(
                    Match(key, deepcopy(posting.entity), deepcopy(entity))
                )
                unique = (category, key)
                scores[posting.vacancy_id][unique] = max(
                    scores[posting.vacancy_id].get(unique, 0), self.weights[importance]
                )
        totals = {
            vacancy_id: sum(values.values()) for vacancy_id, values in scores.items()
        }
        order = sorted(totals, key=lambda vacancy_id: (-totals[vacancy_id], vacancy_id))
        return SearchResults(
            self.name,
            resume.get("id") or resume.get("document_id"),
            tuple(
                SearchHit(vacancy_id, totals[vacancy_id], tuple(grouped[vacancy_id]))
                for vacancy_id in order
            ),
        )
