"""Contract tests for optional Sentence Transformers inference backends."""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest

from vrscore.embeddinggemma_backend import EmbeddingGemmaBackendEncoder


class FakeModel:
    def __init__(self, *_args, **_kwargs):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return np.tile(np.array([1.0] + [0.0] * 127), (len(texts), 1))


def fake_sentence_transformers(monkeypatch, model_class):
    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = model_class
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_backend_constructor_passes_model_contract_and_retrieval_prompts(monkeypatch):
    seen = {}

    class CapturingModel(FakeModel):
        def __init__(self, model_id, **kwargs):
            seen["model_id"] = model_id
            seen.update(kwargs)
            super().__init__()

    fake_sentence_transformers(monkeypatch, CapturingModel)
    encoder = EmbeddingGemmaBackendEncoder(backend="onnx", device="cpu")
    assert seen["model_id"] == "google/embeddinggemma-2"
    assert seen["backend"] == "onnx"
    assert seen["config_kwargs"] == {"vision_config": None, "audio_config": None}
    assert seen["revision"] == encoder.revision
    assert len(encoder.encode_query("resume")) == 128
    docs = encoder.encode_documents(["skills"], titles=["Engineer"])
    assert len(docs) == 1 and len(docs[0]) == 128
    assert encoder.model.calls[0][1]["prompt_name"] == "SearchQuery"
    assert encoder.model.calls[1][0] == ["title: Engineer | text: skills"]
    assert encoder.model.calls[1][1]["prompt"] == ""
    for _, kwargs in encoder.model.calls:
        assert kwargs["truncate_dim"] == 128
        assert kwargs["normalize_embeddings"] is True


def test_outputs_are_renormalized_in_float32_after_backend_inference():
    class LowPrecisionModel(FakeModel):
        def encode(self, texts, **kwargs):
            return np.tile(np.array([0.25] * 128, dtype=np.float32), (len(texts), 1))

    encoder = EmbeddingGemmaBackendEncoder(model=LowPrecisionModel())
    vector = encoder.encode_query("resume")
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-6)


def test_invalid_backend_is_rejected_before_loading(monkeypatch):
    def should_not_load(*_args, **_kwargs):
        raise AssertionError("model loader must not run")

    fake_sentence_transformers(monkeypatch, should_not_load)
    with pytest.raises(ValueError, match="backend"):
        EmbeddingGemmaBackendEncoder(backend="tensorflow")


def test_can_wrap_an_existing_model_without_optional_backend_imports():
    model = FakeModel()
    encoder = EmbeddingGemmaBackendEncoder(model=model)
    assert encoder.model is model
    assert len(encoder.encode_query("resume")) == 128


def test_quantization_is_explicit_and_only_accepts_an_onnx_backend(tmp_path):
    encoder = EmbeddingGemmaBackendEncoder(model=FakeModel())
    with pytest.raises(ValueError, match="backend='onnx'"):
        encoder.export_quantized_onnx(tmp_path)


def test_quantized_onnx_export_requires_a_savable_st_model(tmp_path):
    encoder = EmbeddingGemmaBackendEncoder(backend="onnx", model=FakeModel())
    with pytest.raises(TypeError, match="SentenceTransformer ONNX model"):
        encoder.export_quantized_onnx(tmp_path)
