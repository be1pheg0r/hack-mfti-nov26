"""Local gated BM25 retrieval with bounded embedding reranking."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vrscore.gates import evaluate_first_gate
from vrscore.indexes import BM25SkillIndex
from vrscore.indexes.base import SearchResults
from vrscore.indexes.embedding_reranker import EmbeddingShortlistReranker
from vrscore.milvus_cache import MilvusEmbeddingCache
from vrscore.vector_store import MilvusVectorStore


class RetrievalPipeline:
    """Gate vacancies, retrieve lexically, then rerank a bounded shortlist.

    Source text is only supplied to the caller-injected encoder. Document vectors
    are cached in memory per instance; an optional persistent cache stores only
    content hashes and vectors under an explicit trusted encoder fingerprint.
    """

    def __init__(
        self,
        vacancies: Sequence[dict[str, Any]],
        encoder: Any,
        *,
        shortlist_size: int = 10,
        gate_config: dict[str, Any] | str | Path | None = None,
        bm25_config: dict[str, Any] | None = None,
        include_unknown: bool = True,
        document_cache_collection: str | None = None,
        document_batch_size: int = 8,
        milvus_uri: str = "http://127.0.0.1:19530",
        milvus_token: str | None = None,
        milvus_token_file: str | Path | None = ".data/milvus/token",
    ) -> None:
        if (
            isinstance(shortlist_size, bool)
            or not isinstance(shortlist_size, int)
            or shortlist_size < 1
        ):
            raise ValueError("shortlist_size must be a positive integer")
        if not isinstance(include_unknown, bool):
            raise ValueError("include_unknown must be boolean")
        if (
            isinstance(document_batch_size, bool)
            or not isinstance(document_batch_size, int)
            or not 1 <= document_batch_size <= 128
        ):
            raise ValueError("document_batch_size must be an integer from 1 to 128")
        if not callable(getattr(encoder, "encode_documents", None)) or not callable(
            getattr(encoder, "encode_query", None)
        ):
            raise ValueError("encoder must provide encode_documents and encode_query")
        if isinstance(vacancies, (str, bytes)) or not isinstance(vacancies, Sequence):
            raise ValueError("vacancies must be a sequence of documents")
        self.vacancies = [
            self._validate_document(item, "vacancy") for item in vacancies
        ]
        identifiers = [item["id"] for item in self.vacancies]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Duplicate vacancy ID")
        self.encoder = encoder
        self.shortlist_size = shortlist_size
        self.gate_config = deepcopy(gate_config)
        self.bm25_config = deepcopy(bm25_config)
        self.include_unknown = include_unknown
        self.document_batch_size = document_batch_size
        self.milvus_uri = milvus_uri
        self.milvus_token = milvus_token
        self.milvus_token_file = milvus_token_file
        self._document_vectors: dict[tuple[str, str | None], list[float]] = {}
        self._cache: MilvusEmbeddingCache | None = None
        self._cache_fingerprint: str | None = None
        self._cache_dimension: int | None = None
        if document_cache_collection is not None:
            # Never infer cache compatibility from a class name or defaults: the
            # trusted embedding service must explicitly expose its effective config.
            supplied = getattr(encoder, "cache_fingerprint", None)
            required = {"encoder", "prompt", "revision", "dimension"}
            if not isinstance(supplied, Mapping) or not required <= supplied.keys():
                raise ValueError(
                    "persistent cache requires encoder.cache_fingerprint with "
                    "encoder, prompt, revision, and dimension"
                )
            fingerprint_data = dict(supplied)
            dimension = fingerprint_data["dimension"]
            if (
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension < 1
            ):
                raise ValueError("trusted cache fingerprint dimension must be positive")
            try:
                encoded_fingerprint = json.dumps(
                    fingerprint_data,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid trusted encoder cache fingerprint") from exc
            self._cache_fingerprint = hashlib.sha256(encoded_fingerprint).hexdigest()
            self._cache_dimension = dimension
            token = milvus_token or MilvusVectorStore.resolve_token(milvus_token_file)
            self._cache = MilvusEmbeddingCache(
                milvus_uri, token, document_cache_collection, dimension
            )

    def _content_hash(self, text: str, title: str | None) -> str:
        payload = json.dumps([text, title], ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _persistent_get(self, item: Mapping[str, Any]) -> list[float] | None:
        if self._cache is None or self._cache_fingerprint is None:
            return None
        vector = self._cache.get(
            self._cache_fingerprint, self._content_hash(item["text"], item.get("title"))
        )
        return None if vector is None else self._vector(vector, "cached document")

    def _persistent_put(self, item: Mapping[str, Any], vector: list[float]) -> None:
        if self._cache is None or self._cache_fingerprint is None:
            return
        if len(vector) != self._cache_dimension:
            raise ValueError(
                "document vector dimension does not match trusted cache fingerprint"
            )
        self._cache.put(
            self._cache_fingerprint,
            self._content_hash(item["text"], item.get("title")),
            vector,
        )

    @staticmethod
    def _validate_document(document: Any, kind: str) -> dict[str, Any]:
        if not isinstance(document, Mapping):
            raise ValueError(f"Each {kind} must be a document mapping")
        value = deepcopy(dict(document))
        if not isinstance(value.get("id"), str) or not value["id"].strip():
            raise ValueError(f"Each {kind} needs a non-empty string id")
        if value.get("kind") != kind:
            raise ValueError(f"{kind} input must have kind='{kind}'")
        if not isinstance(value.get("text"), str) or not value["text"].strip():
            raise ValueError(f"Each {kind} needs non-empty text")
        if not isinstance(value.get("entities"), list):
            raise ValueError(f"Each {kind} needs an entities list")
        if (
            "title" in value
            and value["title"] is not None
            and not isinstance(value["title"], str)
        ):
            raise ValueError("title must be a string or None")
        return value

    @staticmethod
    def _vector(values: Sequence[float], label: str) -> list[float]:
        if isinstance(values, (str, bytes)):
            raise ValueError(f"{label} encoder output must be a numeric vector")
        try:
            vector = [float(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label} encoder output must be a numeric vector"
            ) from exc
        if not vector or any(not math.isfinite(value) for value in vector):
            raise ValueError(f"{label} encoder output must be non-empty and finite")
        if not any(vector):
            raise ValueError(f"{label} encoder output cannot be a zero vector")
        return vector

    def search(self, resume: dict[str, Any]) -> dict[str, Any]:
        """Return gate evidence, raw lexical ranks/scores, and shortlist scores."""
        query = self._validate_document(resume, "resume")
        gates = {
            vacancy["id"]: evaluate_first_gate(vacancy, query, config=self.gate_config)
            for vacancy in self.vacancies
        }
        eligible = [
            vacancy
            for vacancy in self.vacancies
            if gates[vacancy["id"]]["status"] == "pass"
            or (self.include_unknown and gates[vacancy["id"]]["status"] == "unknown")
        ]
        bm25 = BM25SkillIndex(eligible, config=self.bm25_config)
        lexical = bm25.search(query)
        lexical_hits = list(lexical.hits)
        lexical_rank = {
            hit.vacancy_id: rank for rank, hit in enumerate(lexical_hits, 1)
        }
        shortlist = SearchResults(
            lexical.index_name,
            lexical.query_id,
            tuple(lexical_hits[: self.shortlist_size]),
        )
        returned: list[dict[str, Any]] = []
        if shortlist.hits:
            by_id = {item["id"]: item for item in eligible}
            missing_by_key = {}
            for hit in shortlist.hits:
                item = by_id[hit.vacancy_id]
                key = (item["text"], item.get("title"))
                if key not in self._document_vectors:
                    cached = self._persistent_get(item)
                    if cached is not None:
                        self._document_vectors[key] = cached
                    else:
                        missing_by_key[key] = item
            missing = list(missing_by_key.values())
            # Keep HTTP requests within the service's item limit and avoid
            # excessively padded inference batches.
            for offset in range(0, len(missing), self.document_batch_size):
                batch = missing[offset : offset + self.document_batch_size]
                encoded = self.encoder.encode_documents(
                    [item["text"] for item in batch],
                    titles=[item.get("title") for item in batch],
                )
                if len(encoded) != len(batch):
                    raise ValueError(
                        "encode_documents returned the wrong number of vectors"
                    )
                for item, vector in zip(batch, encoded, strict=True):
                    validated = self._vector(vector, "document")
                    self._document_vectors[(item["text"], item.get("title"))] = (
                        validated
                    )
                    self._persistent_put(item, validated)
            query_vector = self._vector(
                self.encoder.encode_query(query["text"]), "query"
            )
            vectors_by_id = {
                hit.vacancy_id: self._document_vectors[
                    (by_id[hit.vacancy_id]["text"], by_id[hit.vacancy_id].get("title"))
                ]
                for hit in shortlist.hits
            }
            if any(
                len(vector) != len(query_vector) for vector in vectors_by_id.values()
            ):
                raise ValueError("query and document vector dimensions do not match")
            sentinels = {
                identifier: f"__candidate_{position}__"
                for position, identifier in enumerate(vectors_by_id)
            }
            vectors_by_sentinel = {
                sentinels[identifier]: vector
                for identifier, vector in vectors_by_id.items()
            }
            with MilvusVectorStore(
                dimension=len(query_vector),
                collection_name=f"vrscore_rerank_base_{len(query_vector)}",
                uri=self.milvus_uri,
                token=self.milvus_token,
                token_file=self.milvus_token_file,
            ) as store:
                reranker = EmbeddingShortlistReranker(
                    store, encode=lambda token: vectors_by_sentinel[token]
                )
                reranked = reranker.rerank(
                    shortlist,
                    query_vector,
                    {identifier: sentinels[identifier] for identifier in sentinels},
                )
            score_by_id = {hit.vacancy_id: float(hit.score) for hit in reranked.hits}
            raw_by_id = {hit.vacancy_id: hit for hit in lexical_hits}
            for hit in reranked.hits:
                raw = raw_by_id[hit.vacancy_id]
                returned.append(
                    {
                        "id": hit.vacancy_id,
                        "lexical_rank": lexical_rank[hit.vacancy_id],
                        "lexical_score": float(raw.score),
                        "embedding_score": score_by_id[hit.vacancy_id],
                        "matches": [asdict(match) for match in hit.matches],
                    }
                )
        no_hit_ids = [item["id"] for item in eligible if item["id"] not in lexical_rank]
        return {
            "query_id": query["id"],
            "gates": deepcopy(gates),
            "eligible_ids": [item["id"] for item in eligible],
            "lexical_ranking": [
                {
                    "id": hit.vacancy_id,
                    "rank": rank,
                    "score": float(hit.score),
                    "matches": [asdict(match) for match in hit.matches],
                }
                for rank, hit in enumerate(lexical_hits, 1)
            ],
            "results": returned,
            "accounting": {
                "vacancy_count": len(self.vacancies),
                "eligible_count": len(eligible),
                "lexical_hit_count": len(lexical_hits),
                "returned_count": len(returned),
                "unranked_count": len(eligible) - len(returned),
                "no_hit_ids": no_hit_ids,
            },
        }
