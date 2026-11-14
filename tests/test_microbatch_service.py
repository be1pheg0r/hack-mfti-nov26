"""Concurrency and lifecycle behavior for the local embedding HTTP service."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from vrscore.embedding_server import make_server


class RecordingEncoder:
    dimension = 1

    def __init__(self, *, delay: float = 0.0, fail: bool = False):
        self.delay = delay
        self.fail = fail
        self.calls: list[tuple[str, list[str], list[str | None] | None]] = []
        self.lock = threading.Lock()

    def encode_queries(self, texts):
        with self.lock:
            self.calls.append(("query", list(texts), None))
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("private encoder failure")
        return [[float(len(text))] for text in texts]

    def encode_documents(self, texts, *, titles=None):
        with self.lock:
            self.calls.append(
                ("document", list(texts), list(titles) if titles is not None else None)
            )
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("private encoder failure")
        return [[float(len(text))] for text in texts]


def running_server(encoder, **options):
    server = make_server(encoder, host="127.0.0.1", port=0, **options)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_port}"


def post(url, kind, texts, *, titles=None, timeout=3):
    payload = {"kind": kind, "texts": texts}
    if titles is not None:
        payload["titles"] = titles
    req = urllib.request.Request(
        url + "/embed",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def test_overlapping_requests_are_microbatched_without_mixing_prompts():
    encoder = RecordingEncoder()
    server, thread, url = running_server(
        encoder, max_workers=8, max_queue_size=16, max_batch_size=16, batch_wait_ms=80
    )
    results = {}
    try:
        barrier = threading.Barrier(4)

        def request(key, kind, text, title=None):
            barrier.wait(timeout=2)
            results[key] = post(
                url, kind, [text], titles=[title] if kind == "document" else None
            )["embeddings"]

        workers = [
            threading.Thread(target=request, args=("q1", "query", "query-one")),
            threading.Thread(target=request, args=("q2", "query", "q-two")),
            threading.Thread(
                target=request, args=("d1", "document", "doc-one", "Title A")
            ),
            threading.Thread(
                target=request, args=("d2", "document", "doc-two", "Title B")
            ),
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=3)
        assert all(not worker.is_alive() for worker in workers)
        assert results == {"q1": [[9.0]], "q2": [[5.0]], "d1": [[7.0]], "d2": [[7.0]]}
        calls = {kind: (texts, titles) for kind, texts, titles in encoder.calls}
        assert sorted(calls["query"][0]) == ["q-two", "query-one"]
        document_texts, document_titles = calls["document"]
        assert set(zip(document_texts, document_titles, strict=True)) == {
            ("doc-one", "Title A"),
            ("doc-two", "Title B"),
        }
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_single_large_request_is_split_to_batch_limit():
    encoder = RecordingEncoder()
    server, thread, url = running_server(encoder, max_batch_size=2, batch_wait_ms=0)
    try:
        response = post(url, "query", ["a", "bb", "ccc", "dddd", "eeeee"])
        assert response["embeddings"] == [[1.0], [2.0], [3.0], [4.0], [5.0]]
        assert [len(call[1]) for call in encoder.calls] == [2, 2, 1]
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_encoder_exceptions_are_propagated_as_sanitized_http_errors():
    encoder = RecordingEncoder(fail=True)
    server, thread, url = running_server(encoder, batch_wait_ms=0)
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            post(url, "query", ["do not expose this input"])
        assert exc.value.code == 500
        assert b"private encoder failure" not in exc.value.read()
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_shutdown_drains_accepted_requests_and_stops_dispatcher():
    encoder = RecordingEncoder(delay=0.08)
    server, thread, url = running_server(encoder, batch_wait_ms=80)
    outcome = []
    client_thread = threading.Thread(
        target=lambda: outcome.append(post(url, "query", ["accepted"]))
    )
    client_thread.start()
    time.sleep(0.02)
    server.shutdown()
    thread.join(timeout=2)
    client_thread.join(timeout=2)
    try:
        assert outcome[0]["embeddings"] == [[8.0]]
        assert server.microbatcher.is_stopped
    finally:
        server.server_close()


def test_encoder_value_error_is_not_misreported_as_bad_request():
    encoder = RecordingEncoder(fail=True)

    def fail_with_value_error(texts):
        raise ValueError("private model error")

    encoder.encode_queries = fail_with_value_error
    server, thread, url = running_server(encoder, batch_wait_ms=0)
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            post(url, "query", ["valid input"])
        assert exc.value.code == 500
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_request_timeout_returns_gateway_timeout():
    encoder = RecordingEncoder(delay=0.1)
    server, thread, url = running_server(encoder, batch_wait_ms=0, request_timeout=0.02)
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            post(url, "query", ["slow"], timeout=1)
        assert exc.value.code == 504
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_batch_wait_must_be_bounded_and_queue_full_returns_overloaded():
    with pytest.raises(ValueError):
        make_server(RecordingEncoder(), host="127.0.0.1", port=0, batch_wait_ms=-1)

    encoder = RecordingEncoder(delay=0.2)
    server, thread, url = running_server(
        encoder, max_workers=4, max_queue_size=1, max_batch_size=1, batch_wait_ms=0
    )
    try:
        first = threading.Thread(
            target=lambda: _ignore_error(post(url, "query", ["first"]))
        )
        first.start()
        deadline = time.monotonic() + 1
        while not encoder.calls and time.monotonic() < deadline:
            time.sleep(0.002)
        second = threading.Thread(
            target=lambda: _ignore_error(post(url, "query", ["second"]))
        )
        second.start()
        time.sleep(0.02)
        with pytest.raises(urllib.error.HTTPError) as exc:
            post(url, "query", ["third"])
        assert exc.value.code == 503
        first.join(timeout=2)
        second.join(timeout=2)
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def _ignore_error(call):
    try:
        call
    except Exception:
        pass
