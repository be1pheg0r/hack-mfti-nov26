"""The local embedding service uses the encoder's retrieval contract."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from vrscore.embedding_client import EmbeddingClient
from vrscore.embedding_server import make_server


class Encoder:
    dimension = 128

    def encode_query(self, text):
        return [1.0] * 128

    def encode_documents(self, texts, *, titles=None):
        return [[2.0] * 128 for _ in texts]


def test_http_embedding_contract_and_invalid_input():
    server = make_server(Encoder(), host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        client = EmbeddingClient(base)
        assert client.encode_query("a") == [1.0] * 128
        assert client.encode_documents(["a"], titles=["Title"]) == [[2.0] * 128]
        with pytest.raises(ValueError):
            EmbeddingClient("http://example.com")
        req = urllib.request.Request(
            base + "/embed",
            data=json.dumps({"kind": "query", "texts": ["a"]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as response:
            assert len(json.load(response)["embeddings"][0]) == 128
        req = urllib.request.Request(
            base + "/embed",
            data=b'{"kind":"unknown","texts":["a"]}',
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req)
        assert exc.value.code == 400
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
