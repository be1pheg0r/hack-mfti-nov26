"""Fingerprint-aware Milvus embedding cache tests."""

import json

import pytest
from test_vector_cache_migration import MemoryMilvus

from vrscore.milvus_cache import MilvusEmbeddingCache
from vrscore.retrieval_pipeline import RetrievalPipeline


class Encoder:
    cache_fingerprint = {
        "encoder": "model-a",
        "prompt": "document prompt",
        "revision": "rev-a",
        "dimension": 2,
    }

    def encode_documents(self, texts, *, titles=None):
        return [[1.0, 0.0] for _ in texts]

    def encode_query(self, text):
        return [1.0, 0.0]


def doc():
    return {"id": "v1", "kind": "vacancy", "text": "python", "entities": []}


def test_persistent_cache_requires_explicit_trusted_fingerprint():
    class UnfingerprintedEncoder(Encoder):
        cache_fingerprint = None

    with pytest.raises(ValueError, match="cache_fingerprint"):
        RetrievalPipeline(
            [doc()], UnfingerprintedEncoder(), document_cache_collection="test"
        )


def test_milvus_cache_reuses_vectors_and_fingerprint_or_content_invalidates():
    client = MemoryMilvus()
    first = MilvusEmbeddingCache("unused", "unused", "cache", 2, client=client)
    fp, content = "f" * 64, "a" * 64
    first.put(fp, content, [1.0, 0.0])
    reopened = MilvusEmbeddingCache("unused", "unused", "cache", 2, client=client)
    assert reopened.get(fp, content) == [1.0, 0.0]
    assert reopened.get("e" * 64, content) is None
    assert reopened.get(fp, "b" * 64) is None
    assert "python" not in json.dumps(client.collections)


def test_invalid_cached_vector_is_rejected():
    client = MemoryMilvus()
    cache = MilvusEmbeddingCache("unused", "unused", "cache", 2, client=client)
    fp, content = "f" * 64, "a" * 64
    cache.put(fp, content, [1.0, 0.0])
    next(iter(client.collections["cache"].values()))["vector"] = [float("nan"), 0.0]
    with pytest.raises(ValueError, match="invalid cached"):
        cache.get(fp, content)
