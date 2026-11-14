import hashlib
import json
import sqlite3
import struct

import pytest

from vrscore.milvus_cache import MilvusEmbeddingCache
from vrscore.vector_cache_migration import (
    migrate_malt_cache,
    migrate_pipeline_cache,
    verify_collection,
)


class MemoryMilvus:
    def __init__(self):
        self.collections = {}

    def has_collection(self, collection_name):
        return collection_name in self.collections

    def create_collection(self, collection_name, dimension, schema=None, **kwargs):
        self.collections[collection_name] = {}

    def insert(self, collection_name, data, **kwargs):
        collection = self.collections[collection_name]
        for row in data:
            key = row["id"]
            if key in collection:
                raise ValueError("duplicate")
            collection[key] = row
        return {"insert_count": len(data)}

    def get(self, collection_name, ids, **kwargs):
        c = self.collections[collection_name]
        return [c[i] for i in ids if i in c]

    def query(
        self, collection_name, filter="", output_fields=None, limit=10000, **kwargs
    ):
        return list(self.collections[collection_name].values())[:limit]

    def load_collection(self, collection_name):
        pass


def cache(tmp_path, ddl, rows, meta=()):
    path = tmp_path / "cache.sqlite3"
    db = sqlite3.connect(path)
    db.executescript(ddl)
    for row in meta:
        db.execute("INSERT INTO metadata VALUES (?,?)", row)
    for row in rows:
        db.execute(
            "INSERT INTO "
            + ("embeddings" if len(row) == 4 else "document_embedding_cache")
            + " VALUES ("
            + ",".join("?" * len(row))
            + ")",
            row,
        )
    db.commit()
    db.close()
    return path


def test_malt_cache_migration_is_lossless_resumable_and_conflict_safe(tmp_path):
    vec = struct.pack("<3f", 1.0, 0.0, 0.25)
    path = cache(
        tmp_path,
        "CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL); "
        "CREATE TABLE embeddings(role TEXT NOT NULL,id TEXT NOT NULL,"
        "text_sha TEXT NOT NULL,vector BLOB NOT NULL,PRIMARY KEY(role,id))",
        [("query", "q1", "a" * 64, vec), ("document", "d1", "b" * 64, vec)],
        [("model", "immutable-model@rev:3:fp32:query:document")],
    )
    client = MemoryMilvus()
    report = migrate_malt_cache(path, client, batch_size=1)
    assert report.inserted == 2
    assert migrate_malt_cache(path, client, batch_size=1).inserted == 0
    verification = verify_collection(client, report.collection_name, expected_count=2)
    assert verification.count == 2
    assert verification.vector_hashes_match
    assert verification.payload_hashes_match
    db = sqlite3.connect(path)
    assert db.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 2
    db.close()
    row = next(iter(client.collections[report.collection_name].values()))
    row["vector"] = [9.0, 9.0, 9.0]
    with pytest.raises(ValueError, match="conflict"):
        migrate_malt_cache(path, client)


def test_pipeline_cache_preserves_legacy_fingerprint_as_data(tmp_path):
    vector = [0.125, -0.5, 1.0]
    fp, content_hash = "f" * 64, "c" * 64
    path = cache(
        tmp_path,
        "CREATE TABLE document_embedding_cache("
        "fingerprint TEXT NOT NULL,content_hash TEXT NOT NULL,vector TEXT NOT NULL,"
        "PRIMARY KEY(fingerprint,content_hash))",
        [(fp, content_hash, json.dumps(vector))],
    )
    client = MemoryMilvus()
    report = migrate_pipeline_cache(
        path, client, model_identity="legacy-fingerprint:" + fp
    )
    assert report.inserted == 1
    verification = verify_collection(client, report.collection_name, expected_count=1)
    row = client.collections[report.collection_name].values().__iter__().__next__()
    assert row["fingerprint"] == fp
    assert row["content_hash"] == content_hash
    assert row["model_identity"] == "legacy-fingerprint:" + fp
    assert (
        row["payload_sha256"] == hashlib.sha256(json.dumps(vector).encode()).hexdigest()
    )
    assert verification.vector_hashes_match


def test_malformed_malt_vector_rejected_before_collection_write(tmp_path):
    path = cache(
        tmp_path,
        "CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL); "
        "CREATE TABLE embeddings(role TEXT NOT NULL,id TEXT NOT NULL,"
        "text_sha TEXT NOT NULL,vector BLOB NOT NULL,PRIMARY KEY(role,id))",
        [("query", "q1", "a" * 64, b"not-float32")],
        [("model", "model@revision")],
    )
    client = MemoryMilvus()
    with pytest.raises(ValueError, match="vector"):
        migrate_malt_cache(path, client)
    assert client.collections == {}


def test_milvus_embedding_cache_get_put_and_conflict():
    client = MemoryMilvus()
    cache = MilvusEmbeddingCache("unused", "unused", "cache_contract", 3, client=client)
    fingerprint, content_hash = "a" * 64, "b" * 64
    assert cache.get(fingerprint, content_hash) is None
    cache.put(
        fingerprint,
        content_hash,
        [0.25, -0.5, 1.0],
        metadata={"source_format": "sqlite-v1"},
    )
    assert cache.get(fingerprint, content_hash) == [0.25, -0.5, 1.0]
    cache.put(fingerprint, content_hash, [0.25, -0.5, 1.0])
    assert cache.count() == 1
    with pytest.raises(ValueError, match="conflicting"):
        cache.put(fingerprint, content_hash, [1.0, 0.0, 0.0])
