"""Integration boundaries for configurable retrieval optimization."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from vrscore.embedding_client import EmbeddingClient
from vrscore.retrieval_pipeline import RetrievalPipeline


def test_pipeline_respects_document_batch_size():
    class Encoder:
        def __init__(self):
            self.sizes = []

        def encode_documents(self, texts, *, titles=None):
            self.sizes.append(len(texts))
            return [[1.0, 0.0] for _ in texts]

        def encode_query(self, text):
            return [1.0, 0.0]

    skill = {"category": "hard_skill", "value": "python", "polarity": "present"}
    vacancies = [
        {"id": str(i), "kind": "vacancy", "text": f"python {i}", "entities": [skill]}
        for i in range(3)
    ]
    resume = {"id": "r", "kind": "resume", "text": "python", "entities": [skill]}
    encoder = Encoder()
    RetrievalPipeline(vacancies, encoder, document_batch_size=1).search(resume)
    assert encoder.sizes == [1, 1, 1]


def test_optional_dynamic_quantization_is_applied():
    import torch

    from vrscore.embeddinggemma import EmbeddingGemmaEncoder

    class First:
        def __init__(self):
            self.auto_model = torch.nn.Sequential(torch.nn.Linear(3, 3))

    class Model:
        def __init__(self):
            self.first = First()

        def __getitem__(self, key):
            return self.first

    model = Model()
    EmbeddingGemmaEncoder(model=model, quantization="dynamic_int8")
    assert "quantized" in type(model.first.auto_model[0]).__module__


def test_model_identity_is_safe_for_cache():
    from vrscore.embeddinggemma import EmbeddingGemmaEncoder

    class Fake:
        pass

    encoder = EmbeddingGemmaEncoder(model=Fake())
    assert encoder.cache_fingerprint is None


def test_server_exposes_fingerprint_on_health_and_embeddings():
    from urllib.request import Request, urlopen

    from vrscore.embedding_server import make_server

    fingerprint = {"encoder": "e", "revision": "r", "dimension": 2, "prompt": "p"}

    class Encoder:
        dimension = 2
        cache_fingerprint = fingerprint

        def encode_query(self, text):
            return [1.0, 0.0]

    server = make_server(Encoder(), host="127.0.0.1", port=0)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}
    )
    thread.start()
    try:
        root = f"http://127.0.0.1:{server.server_port}"
        with urlopen(root + "/health") as response:
            assert json.load(response)["cache_fingerprint"] == fingerprint
        request = Request(
            root + "/embed", data=json.dumps({"kind": "query", "texts": ["x"]}).encode()
        )
        with urlopen(request) as response:
            assert json.load(response)["cache_fingerprint"] == fingerprint
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_client_obtains_and_checks_server_fingerprint():
    fingerprint = {
        "encoder": "test",
        "revision": "immutable",
        "dimension": 2,
        "prompt": "p",
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.respond(
                {"status": "ready", "dimension": 2, "cache_fingerprint": fingerprint}
            )

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.respond(
                {
                    "dimension": 2,
                    "embeddings": [[1.0, 0.0]],
                    "cache_fingerprint": self.server.current,
                }
            )

        def respond(self, payload):
            content = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    server.current = fingerprint.copy()
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}
    )
    thread.start()
    try:
        client = EmbeddingClient(f"http://127.0.0.1:{server.server_port}")
        assert client.cache_fingerprint == fingerprint
        assert client.encode_query("sample") == [1.0, 0.0]
        server.current = {**fingerprint, "revision": "different"}
        with pytest.raises(RuntimeError, match="fingerprint"):
            client.encode_query("sample")
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
