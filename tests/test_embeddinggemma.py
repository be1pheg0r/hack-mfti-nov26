"""Contract tests for text-only EmbeddingGemma embeddings."""

import numpy as np
import pytest

from vrscore.embeddinggemma import EmbeddingGemmaEncoder


class FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return np.tile(np.array([1.0] + [0.0] * 127), (len(texts), 1))


def test_query_and_documents_use_retrieval_prompts_and_normalized_128d():
    model = FakeModel()
    encoder = EmbeddingGemmaEncoder(model=model)
    assert len(encoder.encode_query("резюме")) == 128
    assert (
        len(encoder.encode_documents(["навыки", "текст"], titles=["Инженер", None]))
        == 2
    )
    assert model.calls[0][1]["prompt_name"] == "SearchQuery"
    assert model.calls[1][0] == [
        "title: Инженер | text: навыки",
        "title: none | text: текст",
    ]
    assert model.calls[1][1]["prompt"] == ""
    assert all(
        call[1]["truncate_dim"] == 128 and call[1]["normalize_embeddings"]
        for call in model.calls
    )
    with pytest.raises(ValueError):
        encoder.encode_documents(["x"], titles=[])
