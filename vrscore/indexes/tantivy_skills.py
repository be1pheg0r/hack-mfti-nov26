"""Tantivy-backed exact skill retrieval with the baseline scoring policy."""

import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import tantivy
from omegaconf import OmegaConf
from pydantic import BaseModel

from .base import Match, SearchHit, SearchResults
from .exact_skills import DEFAULT_CONFIG as EXACT_CONFIG
from .exact_skills import ExactSkillIndex

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "tantivy_exact_skills_index.yaml"
)


def _term(category: str, value: str) -> str:
    """Encode an exact category/value pair without delimiter collisions."""
    return json.dumps([category, value], ensure_ascii=False, separators=(",", ":"))


class TantivyExactSkillIndex:
    """In-memory Tantivy exact terms with the existing exact index semantics.

    Tantivy performs candidate retrieval; the adapter applies the same explicit
    importance weights, duplicate policy and deterministic ties as ExactSkillIndex.
    Source evidence is held locally, never sent to an external service.
    """

    def __init__(
        self,
        vacancies: list[dict[str, Any] | BaseModel],
        *,
        config: dict[str, Any] | None = None,
        aliases: dict[str, list[str]] | None = None,
    ) -> None:
        settings = OmegaConf.merge(
            OmegaConf.load(EXACT_CONFIG),
            OmegaConf.load(DEFAULT_CONFIG),
            OmegaConf.create(config or {}),
        )
        self._policy = ExactSkillIndex([], config=settings, aliases=aliases)
        self.name = str(settings.name)
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("id", stored=True, tokenizer_name="raw")
        schema_builder.add_text_field(
            "term", tokenizer_name="raw", index_option="basic"
        )
        self._schema = schema_builder.build()
        self._index = tantivy.Index(self._schema)
        writer = self._index.writer(
            heap_size=settings.writer_heap_bytes, num_threads=settings.writer_threads
        )
        self._entities: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = {}
        for vacancy in vacancies:
            if isinstance(vacancy, BaseModel):
                vacancy = vacancy.model_dump()
            vacancy_id = vacancy.get("id") or vacancy.get("document_id")
            if not isinstance(vacancy_id, str) or not vacancy_id.strip():
                raise ValueError("Each vacancy needs a non-empty string id")
            if vacancy_id in self._entities:
                raise ValueError(f"Duplicate vacancy id: {vacancy_id}")
            references: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
            for entity in vacancy.get("entities", vacancy.get("requirements")) or []:
                if not isinstance(entity, dict) or entity.get("defaulted"):
                    continue
                category, value = entity.get("category"), entity.get("value")
                if (
                    category not in self._policy.categories
                    or not isinstance(value, str)
                    or not value.strip()
                ):
                    continue
                key = self._policy._canonical(value)
                references[(category, key)].append(deepcopy(entity))
            self._entities[vacancy_id] = dict(references)
            document = tantivy.Document()
            document.add_text("id", vacancy_id)
            for category, key in references:
                document.add_text("term", _term(category, key))
            writer.add_document(document)
        writer.commit()
        self._index.reload()
        self.vacancy_ids = tuple(sorted(self._entities))

    def search(self, resume: dict[str, Any] | BaseModel) -> SearchResults:
        """Retrieve exact positive skills and preserve matching evidence."""
        if isinstance(resume, BaseModel):
            resume = resume.model_dump()
        grouped: dict[str, list[Match]] = defaultdict(list)
        scores: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
        searcher = self._index.searcher()
        for entity in resume.get("entities", resume.get("claims")) or []:
            if not isinstance(entity, dict) or entity.get("polarity") != "present":
                continue
            category = entity.get("category")
            value = entity.get("value", entity.get("name"))
            if (
                category not in self._policy.categories
                or not isinstance(value, str)
                or not value.strip()
            ):
                continue
            key = self._policy._canonical(value)
            if not self._entities:
                continue
            query = tantivy.Query.term_query(
                self._schema, "term", _term(category, key), index_option="basic"
            )
            for _, address in searcher.search(query, limit=len(self._entities)).hits:
                vacancy_id = searcher.doc(address).to_dict()["id"][0]
                for vacancy_entity in self._entities[vacancy_id][(category, key)]:
                    importance = vacancy_entity.get("importance") or "unknown"
                    if importance not in self._policy.weights:
                        raise ValueError(
                            f"No configured weight for importance {importance!r}"
                        )
                    grouped[vacancy_id].append(
                        Match(key, deepcopy(vacancy_entity), deepcopy(entity))
                    )
                    unique = (category, key)
                    scores[vacancy_id][unique] = max(
                        scores[vacancy_id].get(unique, 0),
                        self._policy.weights[importance],
                    )
        totals = {
            vacancy_id: sum(values.values()) for vacancy_id, values in scores.items()
        }
        ordered = sorted(
            totals, key=lambda vacancy_id: (-totals[vacancy_id], vacancy_id)
        )
        return SearchResults(
            self.name,
            resume.get("id") or resume.get("document_id"),
            tuple(
                SearchHit(vacancy_id, totals[vacancy_id], tuple(grouped[vacancy_id]))
                for vacancy_id in ordered
            ),
        )
