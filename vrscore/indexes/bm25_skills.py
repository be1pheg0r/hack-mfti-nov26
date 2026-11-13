"""Skill retrieval adapter around the rank-bm25 library's BM25Plus."""

import hashlib
import re
from collections import defaultdict
from copy import deepcopy
from math import isfinite
from pathlib import Path
from typing import Any

from omegaconf import ListConfig, OmegaConf
from pydantic import BaseModel
from rank_bm25 import BM25Plus

from .base import Match, SearchHit, SearchResults
from .exact_skills import normalize_skill

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "bm25_skills_index.yaml"
)
# Keep punctuation that distinguishes technology names while discarding separators.
_TOKEN = re.compile(r"(?u)(?<!\w)(?:\.[^\W_]+|[^\W_]+)(?:[.+#-][^\W_]+|[+#]+)*")


def tokenize_skill(value: str) -> list[str]:
    """Tokenize normalized text without conflating C++, C#, .NET and Node.js."""
    return _TOKEN.findall(normalize_skill(value))


class BM25SkillIndex:
    """Rank extracted vacancy skills using rank-bm25 BM25Plus.

    Categories are combined by default, but can be kept distinct via config.
    Only explicitly positive resume claims are queried. The library calculates
    scores over the whole corpus; this adapter returns only lexical matches.
    """

    def __init__(
        self,
        vacancies: list[dict[str, Any] | BaseModel],
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        settings = OmegaConf.merge(
            OmegaConf.load(DEFAULT_CONFIG), OmegaConf.create(config or {})
        )
        self.name = str(settings.name)
        self.categories = frozenset(settings.categories)
        self.merge_skill_categories = settings.merge_skill_categories
        if not isinstance(self.merge_skill_categories, bool):
            raise ValueError("merge_skill_categories must be boolean")
        for key in ("k1", "b", "delta"):
            value = settings[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not isfinite(value)
            ):
                raise ValueError(f"{key} must be a finite number")
        if settings.k1 <= 0 or not 0 <= settings.b <= 1 or settings.delta < 0:
            raise ValueError("Require k1 > 0, 0 <= b <= 1, delta >= 0")
        stop_cfg = settings.stopwords
        if not isinstance(stop_cfg.enabled, bool):
            raise ValueError("stopwords.enabled must be boolean")
        if not isinstance(stop_cfg.preserve_ascii_single_letter, bool):
            raise ValueError("stopwords.preserve_ascii_single_letter must be boolean")
        files = stop_cfg.files
        if not isinstance(files, (list, ListConfig)) or any(
            not isinstance(path, str) or not path.strip() for path in files
        ):
            raise ValueError("stopwords.files must be a list of nonempty paths")
        keep = list(stop_cfg.keep)
        if any(not isinstance(item, str) or not item.strip() for item in keep):
            raise ValueError("stopwords.keep must contain nonempty strings")
        words: set[str] = set()
        self.stopword_sources: dict[str, str] = {}
        if stop_cfg.enabled:
            root = Path(__file__).resolve().parents[2]
            for name in files:
                path = root / name
                try:
                    raw = path.read_bytes()
                    lines = raw.decode("utf-8").splitlines()
                except (OSError, UnicodeError) as exc:
                    raise ValueError(f"Cannot read stopword file: {name}") from exc
                self.stopword_sources[name] = hashlib.sha256(raw).hexdigest()
                for line in lines:
                    value = line.strip()
                    if value and not value.startswith("#"):
                        word = normalize_skill(value)
                        if tokenize_skill(word) != [word]:
                            raise ValueError(f"Stopword is not one token in {name}")
                        words.add(word)
        self.stopwords = frozenset(words - {normalize_skill(word) for word in keep})
        if stop_cfg.preserve_ascii_single_letter:
            self.stopwords = frozenset(
                word
                for word in self.stopwords
                if not (len(word) == 1 and word.isascii() and word.isalpha())
            )
        self._ids: list[str] = []
        self._documents: list[dict[str, list[dict[str, Any]]]] = []
        corpus: list[list[str]] = []
        seen = set()
        for vacancy in vacancies:
            if isinstance(vacancy, BaseModel):
                vacancy = vacancy.model_dump()
            vacancy_id = vacancy.get("id") or vacancy.get("document_id")
            if not isinstance(vacancy_id, str) or not vacancy_id.strip():
                raise ValueError("Each vacancy needs a non-empty string id")
            if vacancy_id in seen:
                raise ValueError(f"Duplicate vacancy id: {vacancy_id}")
            seen.add(vacancy_id)
            tokens: list[str] = []
            references: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for entity in vacancy.get("entities", vacancy.get("requirements")) or []:
                if not isinstance(entity, dict) or entity.get("defaulted"):
                    continue
                category, value = entity.get("category"), entity.get("value")
                if category not in self.categories or not isinstance(value, str):
                    continue
                entity_tokens = [
                    self._key(category, token) for token in self._tokens(value)
                ]
                tokens.extend(entity_tokens)
                for token in dict.fromkeys(entity_tokens):
                    references[token].append(deepcopy(entity))
            self._ids.append(vacancy_id)
            self._documents.append(dict(references))
            corpus.append(tokens)
        self.vacancy_ids = tuple(sorted(seen))
        # rank-bm25 cannot handle an empty corpus; empty documents are still
        # retained in nonempty corpora, including in average length and IDF.
        self._model = (
            BM25Plus(corpus, k1=settings.k1, b=settings.b, delta=settings.delta)
            if any(corpus)
            else None
        )

    def _key(self, category: str, token: str) -> str:
        return token if self.merge_skill_categories else f"{category}\x1f{token}"

    def _tokens(self, value: str) -> list[str]:
        return [token for token in tokenize_skill(value) if token not in self.stopwords]

    def search(self, resume: dict[str, Any] | BaseModel) -> SearchResults:
        """Score a resume against vacancies, preserving evidence for matched tokens."""
        if isinstance(resume, BaseModel):
            resume = resume.model_dump()
        query: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entity in resume.get("entities", resume.get("claims")) or []:
            if not isinstance(entity, dict) or entity.get("polarity") != "present":
                continue
            category = entity.get("category")
            value = entity.get("value", entity.get("name"))
            if category not in self.categories or not isinstance(value, str):
                continue
            for token in dict.fromkeys(self._tokens(value)):
                query[self._key(category, token)].append(deepcopy(entity))
        if self._model is None or not query:
            return SearchResults(
                self.name, resume.get("id") or resume.get("document_id"), ()
            )
        values = self._model.get_scores(list(query))
        hits = []
        for vacancy_id, refs, score in zip(
            self._ids, self._documents, values, strict=True
        ):
            matches = tuple(
                Match(
                    token.split("\x1f", 1)[-1],
                    deepcopy(vacancy_entity),
                    deepcopy(resume_entity),
                )
                for token in query
                if token in refs
                for vacancy_entity in refs[token]
                for resume_entity in query[token]
            )
            if matches:
                hits.append(SearchHit(vacancy_id, float(score), matches))
        hits.sort(key=lambda hit: (-hit.score, hit.vacancy_id))
        return SearchResults(
            self.name, resume.get("id") or resume.get("document_id"), tuple(hits)
        )
