"""Tests for Milvus-backed embedding reranking of a lexical shortlist."""

import json
import math

import pytest

from vrscore.indexes.base import Match, SearchHit, SearchResults
from vrscore.indexes.embedding_reranker import EmbeddingShortlistReranker
from vrscore.vector_store import MilvusVectorStore


class FakeIndexParams:
    def __init__(self):
        self.indexes = []

    def add_index(self, **kwargs):
        self.indexes.append(kwargs)


class FakeSchema:
    def __init__(self):
        self.fields = []

    def add_field(self, **kwargs):
        self.fields.append(kwargs)


class FakeMilvusClient:
    def __init__(self):
        self.collections = {}
        self.created = []

    def has_collection(self, collection_name):
        return collection_name in self.collections

    def prepare_index_params(self):
        return FakeIndexParams()

    def create_schema(self, **kwargs):
        return FakeSchema()

    def create_collection(self, collection_name, schema, index_params):
        self.created.append(collection_name)
        fields = []
        for field in schema.fields:
            params = {}
            if "max_length" in field:
                params["max_length"] = str(field["max_length"])
            if "dim" in field:
                params["dim"] = str(field["dim"])
            fields.append(
                {
                    "name": field["field_name"],
                    "type": "VarChar" if field["field_name"] == "id" else "FloatVector",
                    "params": params,
                }
            )
        self.collections[collection_name] = {
            "fields": fields,
            "indexes": index_params.indexes,
            "rows": {},
        }

    def describe_collection(self, collection_name):
        collection = self.collections[collection_name]
        return {"fields": collection["fields"], "indexes": collection["indexes"]}

    def list_indexes(self, collection_name, field_name=""):
        return [
            index["field_name"]
            for index in self.collections[collection_name]["indexes"]
            if not field_name or index["field_name"] == field_name
        ]

    def describe_index(self, collection_name, index_name):
        return next(
            index
            for index in self.collections[collection_name]["indexes"]
            if index["field_name"] == index_name
        )

    def upsert(self, collection_name, data):
        rows = self.collections[collection_name]["rows"]
        rows.update({row["id"]: row["vector"] for row in data})

    def search(self, collection_name, data, limit, filter="", **kwargs):
        rows = self.collections[collection_name]["rows"]
        allowed = set(json.loads(filter.split(" in ", 1)[1])) if filter else None
        query = data[0]
        hits = []
        for identifier, vector in rows.items():
            if allowed is not None and identifier not in allowed:
                continue
            similarity = sum(a * b for a, b in zip(query, vector)) / (
                math.sqrt(sum(a * a for a in query))
                * math.sqrt(sum(b * b for b in vector))
            )
            hits.append({"id": identifier, "distance": similarity})
        hits.sort(key=lambda hit: (-hit["distance"], hit["id"]))
        return [hits[:limit]]

    def query(
        self, collection_name, filter="", output_fields=None, consistency_level=None
    ):
        self.count_consistency = consistency_level
        return [{"count(*)": len(self.collections[collection_name]["rows"])}]

    def get_collection_stats(self, collection_name):
        return {"row_count": str(len(self.collections[collection_name]["rows"]))}

    def drop_collection(self, collection_name):
        self.collections.pop(collection_name, None)

    def delete(self, collection_name, ids):
        rows = self.collections[collection_name]["rows"]
        for identifier in ids:
            rows.pop(identifier, None)


def test_vector_store_upserts_many_in_one_call():
    client = FakeMilvusClient()
    store = MilvusVectorStore(dimension=2, collection_name="persistent", client=client)
    store.upsert_many({"one": [1.0, 0.0], "two": [0.0, 1.0]})
    assert store.count() == 2


def test_reranker_reorders_only_shortlist_and_preserves_evidence():
    client = FakeMilvusClient()
    store = MilvusVectorStore(dimension=2, collection_name="persistent", client=client)
    store.upsert("unrelated", [0.0, 1.0])
    lexical = SearchResults(
        "bm25",
        "resume-1",
        (
            SearchHit(
                "v1",
                10.0,
                (Match("python", {"value": "Python"}, {"name": "Python"}),),
            ),
            SearchHit("v2", 5.0),
        ),
    )
    texts = {"v1": "vacancy one", "v2": "vacancy two"}
    vectors = {"vacancy one": [0.0, 1.0], "vacancy two": [1.0, 0.0]}
    reranker = EmbeddingShortlistReranker(store, encode=lambda text: vectors[text])

    result = reranker.rerank(lexical, [1.0, 0.0], texts)

    assert result.index_name == "embedding_rerank"
    assert result.query_id == "resume-1"
    assert [hit.vacancy_id for hit in result.hits] == ["v2", "v1"]
    assert result.hits[0].score == pytest.approx(1.0)
    assert result.hits[1].matches == lexical.hits[0].matches
    assert {hit.vacancy_id for hit in result.hits} == {"v1", "v2"}
    assert list(client.collections) == ["persistent"]
    assert client.created == ["persistent"]
    assert store.count() == 1


def test_reranker_rejects_missing_text_duplicate_ids_and_bad_vectors():
    client = FakeMilvusClient()
    store = MilvusVectorStore(dimension=2, collection_name="persistent", client=client)
    reranker = EmbeddingShortlistReranker(store, encode=lambda _: [1.0, 0.0])
    one = SearchResults("bm25", None, (SearchHit("v1", 1.0),))
    with pytest.raises(ValueError, match="text"):
        reranker.rerank(one, [1.0, 0.0], {})
    duplicate = SearchResults(
        "bm25", None, (SearchHit("v1", 1.0), SearchHit("v1", 0.5))
    )
    with pytest.raises(ValueError, match="duplicate"):
        reranker.rerank(duplicate, [1.0, 0.0], {"v1": "text"})
    wrong_encoder = EmbeddingShortlistReranker(store, encode=lambda _: [1.0])
    with pytest.raises(ValueError, match="dimension"):
        wrong_encoder.rerank(one, [1.0, 0.0], {"v1": "text"})
    assert list(client.collections) == ["persistent"]


def test_vector_store_search_can_be_restricted_to_shortlist_ids():
    client = FakeMilvusClient()
    store = MilvusVectorStore(dimension=2, collection_name="persistent", client=client)
    store.upsert("short", [1.0, 0.0])
    store.upsert("other", [0.0, 1.0])

    results = store.search([0.0, 1.0], limit=1, record_ids=["short"])

    assert [result.record_id for result in results] == ["short"]
