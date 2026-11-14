"""Milvus retrieval cache integration."""

import os
from pathlib import Path
from uuid import uuid4

import pytest

from vrscore.retrieval_pipeline import RetrievalPipeline


@pytest.mark.skipif(
    not os.getenv("MILVUS_TEST_URI"), reason="Milvus test server not configured"
)
def test_milvus_cache_survives_pipeline_reconstruction():
    from pymilvus import MilvusClient

    uri = os.environ["MILVUS_TEST_URI"]
    token = Path(os.environ["MILVUS_TEST_TOKEN_FILE"]).read_text().strip()
    collection = "test_cache_" + uuid4().hex

    class Encoder:
        cache_fingerprint = {
            "encoder": "test",
            "revision": "fixed",
            "dimension": 2,
            "prompt": "fixed",
        }

        def __init__(self):
            self.calls = 0

        def encode_documents(self, texts, *, titles=None):
            self.calls += len(texts)
            return [[1.0, 0.0] for _ in texts]

        def encode_query(self, text):
            return [1.0, 0.0]

    skill = {"category": "hard_skill", "value": "python", "polarity": "present"}
    vacancy = {"id": "v", "kind": "vacancy", "text": "python", "entities": [skill]}
    resume = {"id": "r", "kind": "resume", "text": "python", "entities": [skill]}
    try:
        first = Encoder()
        out = RetrievalPipeline(
            [vacancy],
            first,
            milvus_uri=uri,
            milvus_token=token,
            document_cache_collection=collection,
        ).search(resume)
        second = Encoder()
        other = RetrievalPipeline(
            [vacancy],
            second,
            milvus_uri=uri,
            milvus_token=token,
            document_cache_collection=collection,
        ).search(resume)
        assert first.calls == 1
        assert second.calls == 0
        assert out == other
    finally:
        c = MilvusClient(uri=uri, token=token)
        if c.has_collection(collection):
            c.drop_collection(collection)
        c.close()
