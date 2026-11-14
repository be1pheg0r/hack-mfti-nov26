"""Synthetic contract tests for the persistent Milvus vector store."""

from pathlib import Path

import pytest

from vrscore.vector_store import MilvusVectorStore


class FakeIndexParams:
    def __init__(self):
        self.indexes = []

    def add_index(self, field_name, index_type, metric_type):
        self.indexes.append(
            {
                "field_name": field_name,
                "index_type": index_type,
                "metric_type": metric_type,
            }
        )


class FakeSchema:
    def __init__(self):
        self.fields = []

    def add_field(self, **kwargs):
        self.fields.append(kwargs)


class FakeMilvusClient:
    """Small in-memory stand-in implementing the MilvusClient boundary."""

    def __init__(self):
        self.collections = {}
        self.calls = []
        self.count_consistency = None

    def has_collection(self, collection_name):
        return collection_name in self.collections

    def create_schema(self, **kwargs):
        return FakeSchema()

    def create_collection(self, collection_name, schema, index_params):
        self.calls.append(("create_collection", collection_name, index_params))
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
            "schema": {"fields": fields, "indexes": index_params.indexes},
            "rows": {},
        }

    def prepare_index_params(self):
        return FakeIndexParams()

    def describe_collection(self, collection_name):
        return self.collections[collection_name]["schema"]

    def list_indexes(self, collection_name, field_name=""):
        return [
            index["field_name"]
            for index in self.collections[collection_name]["schema"]["indexes"]
            if not field_name or index["field_name"] == field_name
        ]

    def describe_index(self, collection_name, index_name):
        return next(
            index
            for index in self.collections[collection_name]["schema"]["indexes"]
            if index["field_name"] == index_name
        )

    def upsert(self, collection_name, data):
        self.collections[collection_name]["rows"].update(
            {row["id"]: row["vector"] for row in data}
        )
        return {"upsert_count": len(data)}

    def search(
        self,
        collection_name,
        data,
        limit,
        filter="",
        output_fields=None,
        search_params=None,
        consistency_level=None,
    ):
        import json
        import math

        self.calls.append(("search", filter, limit))
        rows = self.collections[collection_name]["rows"]
        selected = list(rows)
        if filter:
            selected = [
                key for key in selected if key in json.loads(filter.split(" in ", 1)[1])
            ]
        query = data[0]
        hits = []
        for key in selected:
            vector = rows[key]
            score = sum(x * y for x, y in zip(query, vector)) / (
                math.sqrt(sum(x * x for x in query))
                * math.sqrt(sum(x * x for x in vector))
            )
            hits.append({"id": key, "distance": score})
        return [hits[:limit]]

    def query(
        self, collection_name, filter="", output_fields=None, consistency_level=None
    ):
        self.count_consistency = consistency_level
        return [{"count(*)": len(self.collections[collection_name]["rows"])}]

    def get_collection_stats(self, collection_name):
        return {"row_count": str(len(self.collections[collection_name]["rows"]))}

    def delete(self, collection_name, ids=None, filter=None):
        rows = self.collections[collection_name]["rows"]
        for key in ids or []:
            rows.pop(key, None)

    def drop_collection(self, collection_name):
        self.calls.append(("drop_collection", collection_name))
        self.collections.pop(collection_name, None)

    def close(self):
        self.calls.append(("close",))


def _store(client=None, **kwargs):
    return MilvusVectorStore(
        dimension=2,
        collection_name="test_vectors",
        client=client or FakeMilvusClient(),
        **kwargs,
    )


def test_store_uses_exact_cosine_collection_and_upserts_string_ids():
    client = FakeMilvusClient()
    store = _store(client)

    assert store.upsert("record-1", [1.0, 0.0])
    assert store.count() == 1
    assert client.count_consistency == "Strong"
    schema = client.collections["test_vectors"]["schema"]
    assert schema["fields"][1]["params"]["dim"] == "2"
    assert schema["fields"][0]["params"]["max_length"] == "65535"
    assert client.calls[0][0] == "create_collection"
    assert client.calls[0][2].indexes == [
        {"field_name": "vector", "index_type": "FLAT", "metric_type": "COSINE"}
    ]


def test_upsert_replaces_and_rejects_bad_vectors_and_collection_dimensions():
    store = _store()
    store.upsert("same-id", [1.0, 0.0])
    store.upsert("same-id", [0.0, 1.0])
    assert store.count() == 1
    with pytest.raises(ValueError, match="dimension"):
        store.upsert("bad", [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        store.upsert("bad", [float("nan"), 0.0])
    with pytest.raises(ValueError, match="dimension"):
        MilvusVectorStore(
            dimension=3,
            collection_name="test_vectors",
            client=store.client,
        )


def test_filtered_search_is_injection_safe_and_sorts_ties_by_id():
    client = FakeMilvusClient()
    store = _store(client)
    store.upsert("a", [0.8, 0.0])
    store.upsert("b", [0.8, 0.0])
    store.upsert('x" or id != "', [0.99, 0.0])

    results = store.search([1.0, 0.0], record_ids=["b", "a"], limit=2)

    assert [result.record_id for result in results] == ["a", "b"]
    filter_expression = next(call[1] for call in client.calls if call[0] == "search")
    assert '"x\\" or id' not in filter_expression
    assert store.search([1.0, 0.0], record_ids=[], limit=2) == []


def test_tie_break_is_deterministic_even_at_limit_boundary():
    store = _store()
    for identifier in ("z", "b", "a"):
        store.upsert(identifier, [1.0, 0.0])

    assert [result.record_id for result in store.search([1.0, 0.0], limit=2)] == [
        "a",
        "b",
    ]


def test_search_validates_query_and_upsert_ids():
    store = _store()
    with pytest.raises(ValueError, match="record_id"):
        store.upsert(" ", [1.0, 0.0])
    with pytest.raises(ValueError, match="zero"):
        store.upsert("z", [0.0, 0.0])
    with pytest.raises(ValueError, match="limit"):
        store.search([1.0, 0.0], limit=0)


def test_temporary_collection_is_unique_and_scoped_cleanup_preserves_persistent():
    client = FakeMilvusClient()
    persistent = _store(client)

    with MilvusVectorStore.temporary(
        dimension=2, client=client, collection_prefix="rerank"
    ) as temporary:
        collection = temporary.collection_name
        assert collection != persistent.collection_name
        temporary.upsert("only-here", [1.0, 0.0])
        assert temporary.count() == 1

    assert collection not in client.collections
    assert persistent.collection_name in client.collections
    assert ("drop_collection", collection) in client.calls
    assert ("drop_collection", persistent.collection_name) not in client.calls


def test_explicit_token_is_used_by_constructor_without_environment_mutation(
    monkeypatch,
):
    client = FakeMilvusClient()
    observed = []

    def make_client(**kwargs):
        observed.append(kwargs)
        return client

    monkeypatch.setattr("vrscore.vector_store.MilvusClient", make_client)
    store = MilvusVectorStore(
        dimension=2,
        collection_name="explicit-token",
        uri="http://127.0.0.1:19530",
        token="explicit-secret",
        token_file=None,
    )

    assert observed[0]["token"] == "explicit-secret"
    assert "MILVUS_TOKEN" not in __import__("os").environ
    store.close()


def test_temporary_store_accepts_explicit_token(monkeypatch):
    client = FakeMilvusClient()
    observed = []

    def make_client(**kwargs):
        observed.append(kwargs)
        return client

    monkeypatch.setattr("vrscore.vector_store.MilvusClient", make_client)
    with MilvusVectorStore.temporary(
        dimension=2, token="temporary-secret", token_file=None
    ):
        pass

    assert observed[0]["token"] == "temporary-secret"


def test_token_is_read_from_private_file_and_environment_takes_precedence(
    tmp_path, monkeypatch
):
    token_file = tmp_path / "token"
    token_file.write_text("file-secret\n", encoding="utf-8")
    token_file.chmod(0o600)

    assert MilvusVectorStore.resolve_token(token_file) == "file-secret"
    monkeypatch.setenv("MILVUS_TOKEN", "environment-secret")
    assert MilvusVectorStore.resolve_token(token_file) == "environment-secret"


def test_token_file_rejects_permissions_other_than_0600(tmp_path, monkeypatch):
    monkeypatch.delenv("MILVUS_TOKEN", raising=False)
    token_file = tmp_path / "token"
    token_file.write_text("token", encoding="utf-8")
    token_file.chmod(0o640)

    with pytest.raises(ValueError, match="0600"):
        MilvusVectorStore.resolve_token(token_file)


def test_hydra_config_uses_milvus_endpoint_and_secure_token_file():
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[1] / "configs")
    ):
        config = compose(config_name="vector_store")

    assert config.backend == "milvus"
    assert config.uri == "http://127.0.0.1:19530"
    assert config.token_file == ".data/milvus/token"
