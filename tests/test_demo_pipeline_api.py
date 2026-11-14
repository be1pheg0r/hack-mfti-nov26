"""Authenticated demo pipeline API returns real retrieval stages."""

import json
import threading
from functools import partial
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from vrscore.demo.server import Handler
from vrscore.demo.service import DemoService


class Encoder:
    def __init__(self):
        self.document_calls = 0
        self.query_calls = 0

    @property
    def cache_fingerprint(self):
        return {"encoder": "test", "prompt": "v1", "revision": "a", "dimension": 2}

    def encode_documents(self, texts, *, titles=None):
        self.document_calls += 1
        return [[1.0, 0.0] for _ in texts]

    def encode_query(self, text):
        self.query_calls += 1
        return [1.0, 0.0]


def test_pipeline_returns_stages_cached_by_resume_and_honors_shortlist(
    monkeypatch, tmp_path
):
    service = DemoService(None, cache_dir=tmp_path, demo_delay_seconds=0.05)
    fail_accountant = False

    def extraction(document, kind):
        if fail_accountant and kind == "vacancy" and document["id"] == "accountant":
            raise ValueError("extraction unavailable")
        return {
            "id": document["id"],
            "kind": kind,
            "entities": [
                {
                    "category": "experience",
                    "value": (
                        10
                        if kind == "vacancy" and document["id"] == "accountant"
                        else 1
                    ),
                },
                {"category": "education", "value": "bachelor"},
                {"category": "work_format", "value": "on_site"},
                {"category": "work_schedule", "value": "full_time"},
                {
                    "category": "hard_skill",
                    "value": "python",
                    **({"polarity": "present"} if kind == "resume" else {}),
                },
            ],
        }

    monkeypatch.setattr(service, "_cached_extraction", extraction)
    import vrscore.demo.service as demo_service
    import vrscore.retrieval_pipeline as retrieval

    class VectorStore:
        def __init__(self, **kwargs):
            self.rows = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def upsert_many(self, rows):
            self.rows.update(rows)

        def search(self, query, *, limit, record_ids):
            import math

            def cosine(vector):
                return sum(a * b for a, b in zip(query, vector)) / (
                    math.sqrt(sum(a * a for a in query))
                    * math.sqrt(sum(b * b for b in vector))
                )

            return [
                type("Hit", (), {"record_id": key, "score": cosine(self.rows[key])})()
                for key in record_ids
            ]

        def delete_many(self, record_ids):
            for key in record_ids:
                self.rows.pop(key, None)

    monkeypatch.setattr(retrieval, "MilvusVectorStore", VectorStore)
    delays = []
    monkeypatch.setattr(demo_service.time, "sleep", delays.append)
    encoder = Encoder()
    monkeypatch.setattr(service, "embedding_encoder", encoder)
    service.shortlist_size = 2

    result = service.pipeline("r2")
    assert result["resume_id"] == "r2"
    assert set(result) == {
        "resume_id",
        "initial",
        "gate",
        "bm25",
        "ranked",
        "shortlist_size",
    }
    assert set(result["initial"]) == {v["id"] for v in service.vacancies}
    assert result["gate"]["passed"]
    assert result["gate"]["rejected"]
    assert all(set(row) == {"vacancy_id", "score"} for row in result["bm25"])
    assert all(set(row) == {"vacancy_id", "score"} for row in result["ranked"])
    assert len(result["bm25"]) <= 2
    assert result["shortlist_size"] == 2
    assert service.pipeline("r2") == result
    assert delays == [service.demo_delay_seconds]
    assert encoder.document_calls == 1
    assert encoder.query_calls == 1

    restarted = DemoService(None, cache_dir=tmp_path, demo_delay_seconds=0)
    monkeypatch.setattr(restarted, "_cached_extraction", extraction)
    restarted_encoder = Encoder()
    monkeypatch.setattr(restarted, "embedding_encoder", restarted_encoder)
    restarted.shortlist_size = 2
    assert restarted.pipeline("r2") == result
    assert restarted_encoder.document_calls == 0
    assert restarted_encoder.query_calls == 0

    changed = DemoService(None, cache_dir=tmp_path, demo_delay_seconds=0)
    monkeypatch.setattr(changed, "_cached_extraction", extraction)
    changed_encoder = Encoder()
    monkeypatch.setattr(changed, "embedding_encoder", changed_encoder)
    changed.shortlist_size = 3
    changed.pipeline("r2")
    assert changed_encoder.document_calls == 1
    assert changed_encoder.query_calls == 1

    fail_accountant = True
    failed = DemoService(None, cache_dir=tmp_path, demo_delay_seconds=0)
    monkeypatch.setattr(failed, "_cached_extraction", extraction)
    failed_encoder = Encoder()
    monkeypatch.setattr(failed, "embedding_encoder", failed_encoder)
    failed.shortlist_size = 2
    failed_result = failed.pipeline("r2")
    assert failed_result["gate"]["decisions"]["accountant"]["status"] == "unknown"
    assert "accountant" not in {
        row["vacancy_id"] for row in failed_result["bm25"] + failed_result["ranked"]
    }


def test_pipeline_http_auth_validation_and_contract():
    class Service:
        resumes = [{"id": "r1"}]

        def pipeline(self, resume_id):
            return {
                "resume_id": resume_id,
                "initial": ["v1"],
                "gate": {"passed": ["v1"], "rejected": []},
                "bm25": [{"vacancy_id": "v1", "score": 0.4}],
                "ranked": [{"vacancy_id": "v1", "score": 0.9}],
                "shortlist_size": 10,
            }

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_port
    server.RequestHandlerClass = partial(
        Handler, service=Service(), token="secret", port=port
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}/api/demo/pipeline"
    headers = {"Content-Type": "application/json"}
    body = json.dumps({"resume_id": "r1"}).encode()
    try:
        with pytest.raises(HTTPError) as unauthorized:
            urlopen(Request(url, data=body, headers=headers))
        assert unauthorized.value.code == 401
        headers["Authorization"] = "Bearer secret"
        with urlopen(Request(url, data=body, headers=headers)) as response:
            assert json.load(response)["resume_id"] == "r1"
        with pytest.raises(HTTPError) as invalid:
            urlopen(Request(url, data=b'{"resume_id":"missing"}', headers=headers))
        assert invalid.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
