"""Configurable CPU and batching optimizations for the embedding encoder."""

import json
import sys
import threading
from types import SimpleNamespace
from urllib.request import Request, urlopen

import pytest

from vrscore.embedding_server import make_server
from vrscore.embeddinggemma import EmbeddingGemmaEncoder


class LengthModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        # Keep a valid nonzero 128D vector while making result order observable.
        return [[float(len(text))] + [0.0] * 127 for text in texts]


def test_cpu_thread_count_is_configurable_without_changing_non_cpu_devices(monkeypatch):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(set_num_threads=calls.append),
    )

    EmbeddingGemmaEncoder(model=LengthModel(), device="cpu", cpu_threads=3)
    assert calls == [3]

    EmbeddingGemmaEncoder(model=LengthModel(), device="cuda", cpu_threads=5)
    assert calls == [3]


def test_cpu_thread_count_must_be_positive():
    with pytest.raises(ValueError, match="cpu_threads"):
        EmbeddingGemmaEncoder(model=LengthModel(), cpu_threads=0)


def test_query_batches_are_encoded_together_with_original_order():
    model = LengthModel()
    encoder = EmbeddingGemmaEncoder(model=model, batch_size=2)

    vectors = encoder.encode_queries(["short", "long query"])

    assert [vector[0] for vector in vectors] == [5.0, 10.0]
    assert len(model.calls) == 1
    assert model.calls[0][0] == ["short", "long query"]
    assert model.calls[0][1]["prompt_name"] == "SearchQuery"
    assert model.calls[0][1]["batch_size"] == 2


def test_document_length_sorting_restores_input_order():
    model = LengthModel()
    encoder = EmbeddingGemmaEncoder(model=model, batch_size=2, sort_by_length=True)

    vectors = encoder.encode_documents(["a much longer text", "x", "medium"])

    assert [vector[0] for vector in vectors] == [38.0, 21.0, 26.0]
    assert model.calls[0][0] == [
        "title: none | text: x",
        "title: none | text: medium",
        "title: none | text: a much longer text",
    ]
    assert model.calls[0][1]["batch_size"] == 2


def test_http_query_batch_uses_encoder_batch_path():
    model = LengthModel()
    encoder = EmbeddingGemmaEncoder(model=model, batch_size=2)
    server = make_server(encoder, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/embed",
            data=json.dumps(
                {"kind": "query", "texts": ["short", "long query"]}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request) as response:
            payload = json.load(response)
        assert [row[0] for row in payload["embeddings"]] == [5.0, 10.0]
        assert len(model.calls) == 1
        assert model.calls[0][1]["batch_size"] == 2
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_length_sorting_can_be_disabled():
    model = LengthModel()
    encoder = EmbeddingGemmaEncoder(model=model, sort_by_length=False)

    encoder.encode_documents(["longer", "x"])

    assert model.calls[0][0] == [
        "title: none | text: longer",
        "title: none | text: x",
    ]
