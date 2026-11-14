"""Tests for the isolated OpenVINO EmbeddingGemma adapter."""

from __future__ import annotations

import math
import sys
from types import ModuleType

import pytest

from vrscore.embeddinggemma_openvino import OpenVINOEmbeddingGemmaEncoder
from vrscore.embeddinggemma_openvino_worker import _handle


class FakeSentenceTransformer:
    def __init__(self, *_args, **_kwargs):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return [[0.6, 0.8] + [0.0] * 126 for _ in texts]


def test_loads_model_with_openvino_text_only_and_pinned_revision(monkeypatch):
    seen = {}

    class CapturingModel(FakeSentenceTransformer):
        def __init__(self, model_id, **kwargs):
            seen["model_id"] = model_id
            seen.update(kwargs)
            super().__init__()

    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = CapturingModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    encoder = OpenVINOEmbeddingGemmaEncoder(dimension=128)

    assert seen["model_id"] == "google/embeddinggemma-2"
    assert seen["revision"] == "f9972ac22e696a105c482202646979e33e4b5a1e"
    assert seen["backend"] == "openvino"
    assert seen["config_kwargs"] == {"vision_config": None, "audio_config": None}
    assert seen["device"] == "cpu"
    assert encoder.model is not None


def test_query_document_prompts_and_normalized_matryoshka_vectors():
    model = FakeSentenceTransformer()
    encoder = OpenVINOEmbeddingGemmaEncoder(model=model, dimension=128)

    query = encoder.encode_query("resume skills")
    documents = encoder.encode_documents(["Python engineering"], titles=["Engineer"])

    assert query[:2] == pytest.approx([0.6, 0.8])
    assert len(query) == 128
    assert documents[0][:2] == pytest.approx([0.6, 0.8])
    assert len(documents[0]) == 128
    assert model.calls[0] == (
        ["resume skills"],
        {
            "prompt_name": "SearchQuery",
            "truncate_dim": 128,
            "normalize_embeddings": True,
            "batch_size": 8,
            "show_progress_bar": False,
        },
    )
    assert model.calls[1][0] == ["title: Engineer | text: Python engineering"]
    assert model.calls[1][1]["prompt"] == ""


def test_worker_routes_document_requests_without_echoing_input():
    model = FakeSentenceTransformer()
    encoder = OpenVINOEmbeddingGemmaEncoder(model=model)

    response = _handle(
        encoder,
        {"op": "documents", "texts": ["private text"], "titles": ["Role"]},
    )

    assert set(response) == {"vectors"}
    assert model.calls[0][0] == ["title: Role | text: private text"]


def test_worker_rejects_unknown_operations():
    encoder = OpenVINOEmbeddingGemmaEncoder(model=FakeSentenceTransformer())
    with pytest.raises(ValueError, match="op must be"):
        _handle(encoder, {"op": "unknown", "texts": []})


def test_empty_or_invalid_texts_are_rejected_before_inference():
    model = FakeSentenceTransformer()
    encoder = OpenVINOEmbeddingGemmaEncoder(model=model)
    with pytest.raises(ValueError, match="non-empty"):
        encoder.encode_query("  ")
    with pytest.raises(ValueError, match="same length"):
        encoder.encode_documents(["a"], titles=[])
    assert model.calls == []


def test_invalid_output_is_rejected():
    class BadModel(FakeSentenceTransformer):
        def encode(self, texts, **kwargs):
            return [[math.nan] * 128 for _ in texts]

    encoder = OpenVINOEmbeddingGemmaEncoder(model=BadModel())
    with pytest.raises(ValueError, match="invalid embeddings"):
        encoder.encode_query("resume")
