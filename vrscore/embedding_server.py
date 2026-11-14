"""Loopback-only HTTP service for a single in-memory embedding model."""

from __future__ import annotations

import json
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import hydra
from omegaconf import DictConfig

from .embeddinggemma import EmbeddingGemmaEncoder

MAX_BODY = 2_000_000


class _EncoderFailure(RuntimeError):
    """Distinguish model failures from malformed client input."""


class _Request:
    def __init__(self, kind: str, texts: list[str], titles: list[str | None] | None):
        self.kind = kind
        self.texts = texts
        self.titles = titles
        self.future: Future = Future()


class _Microbatcher:
    def __init__(self, encoder, *, max_queue_size, max_batch_size, batch_wait_ms):
        self.encoder = encoder
        self.max_batch_size = max_batch_size
        self.batch_wait = batch_wait_ms / 1000
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._stopping = threading.Event()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, name="embedding-microbatcher")
        self._thread.start()

    @property
    def is_stopped(self):
        return self._stopped.is_set()

    def submit(self, kind, texts, titles):
        if self._stopping.is_set():
            raise RuntimeError("service shutting down")
        request = _Request(kind, texts, titles)
        try:
            self.queue.put_nowait(request)
        except queue.Full as exc:
            raise OverflowError("embedding queue is full") from exc
        return request.future

    def _run(self):
        deferred = []
        pending = None
        try:
            while True:
                if pending is None:
                    try:
                        pending = self.queue.get(timeout=0.05)
                    except queue.Empty:
                        if self._stopping.is_set():
                            break
                        continue
                gathered = [pending]
                pending = None
                count = len(gathered[0].texts)
                deadline = time.monotonic() + self.batch_wait
                while count < self.max_batch_size:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        candidate = self.queue.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if count + len(candidate.texts) > self.max_batch_size:
                        deferred.append(candidate)
                    else:
                        gathered.append(candidate)
                        count += len(candidate.texts)
                kinds = []
                for item in gathered:
                    if item.kind not in kinds:
                        kinds.append(item.kind)
                for kind in kinds:
                    self._encode_group([item for item in gathered if item.kind == kind])
                for item in deferred:
                    self._encode_group([item])
                deferred.clear()
            if pending is not None:
                self._encode_group([pending])
        finally:
            self._stopped.set()

    def _encode_group(self, group):
        try:
            texts = [text for item in group for text in item.texts]
            titles = (
                [
                    title
                    for item in group
                    for title in (item.titles or [None] * len(item.texts))
                ]
                if group[0].kind == "document"
                else None
            )
            vectors = []
            for start in range(0, len(texts), self.max_batch_size):
                chunk = texts[start : start + self.max_batch_size]
                if group[0].kind == "query":
                    encode_queries = getattr(self.encoder, "encode_queries", None)
                    result = (
                        encode_queries(chunk)
                        if callable(encode_queries)
                        else [self.encoder.encode_query(text) for text in chunk]
                    )
                else:
                    result = self.encoder.encode_documents(
                        chunk, titles=titles[start : start + len(chunk)]
                    )
                if len(result) != len(chunk):
                    raise ValueError("encoder returned the wrong number of embeddings")
                vectors.extend(result)
            expected = sum(len(item.texts) for item in group)
            if len(vectors) != expected:
                raise ValueError("encoder returned the wrong number of embeddings")
            offset = 0
            for item in group:
                end = offset + len(item.texts)
                item.future.set_result(vectors[offset:end])
                offset = end
        except Exception as exc:
            failure = _EncoderFailure("embedding failed")
            failure.__cause__ = exc
            for item in group:
                item.future.set_exception(failure)

    def stop(self):
        self._stopping.set()
        self._thread.join()


class _BoundedThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = False
    block_on_close = True

    def __init__(self, address, handler, *, max_workers, max_queue_size):
        self.request_queue_size = max_workers + max_queue_size
        super().__init__(address, handler)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="embedding-http"
        )
        self._slots = threading.BoundedSemaphore(max_workers + max_queue_size)

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            try:
                body = b'{"error":"service overloaded"}'
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\n"
                    b"Content-Type: application/json\r\nContent-Length: "
                    + str(len(body)).encode()
                    + b"\r\n\r\n"
                    + body
                )
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            self._executor.submit(self._process_request, request, client_address)
        except Exception:
            self._slots.release()
            self.shutdown_request(request)
            raise

    def _process_request(self, request, client_address):
        try:
            self.process_request_thread(request, client_address)
        finally:
            self._slots.release()

    def shutdown(self):
        super().shutdown()
        self._executor.shutdown(wait=True, cancel_futures=False)
        self.microbatcher.stop()

    def server_close(self):
        super().server_close()
        self._executor.shutdown(wait=True, cancel_futures=False)
        self.microbatcher.stop()


def make_server(
    encoder: EmbeddingGemmaEncoder,
    *,
    host: str,
    port: int,
    max_workers: int = 8,
    max_queue_size: int = 64,
    max_batch_size: int = 32,
    batch_wait_ms: int = 10,
    request_timeout: float = 120.0,
) -> HTTPServer:
    """Create bounded local HTTP handling and a shared cross-request batcher."""
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("Embedding service only supports loopback binding")
    if max_workers < 1 or max_queue_size < 1 or max_batch_size < 1:
        raise ValueError("worker, queue, and batch limits must be positive")
    if batch_wait_ms < 0 or request_timeout <= 0:
        raise ValueError("batch wait must be nonnegative and request timeout positive")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def respond(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path != "/health":
                self.respond(404, {"error": "not found"})
                return
            self.respond(
                200,
                {
                    "status": "ready",
                    "dimension": encoder.dimension,
                    "cache_fingerprint": getattr(encoder, "cache_fingerprint", None),
                },
            )

        def do_POST(self) -> None:
            if self.path != "/embed":
                self.respond(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > MAX_BODY:
                    raise ValueError("invalid request length")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("expected an object")
                texts = data.get("texts")
                if not isinstance(texts, list) or not texts or len(texts) > 128:
                    raise ValueError(
                        "texts must be a nonempty list of at most 128 items"
                    )
                if any(not isinstance(text, str) or not text.strip() for text in texts):
                    raise ValueError("texts must contain non-empty strings")
                kind = data.get("kind")
                titles = None
                if kind == "document":
                    titles = data.get("titles")
                    if titles is not None and (
                        not isinstance(titles, list)
                        or len(titles) != len(texts)
                        or any(
                            title is not None and not isinstance(title, str)
                            for title in titles
                        )
                    ):
                        raise ValueError(
                            "titles must match texts and contain strings or nulls"
                        )
                elif kind != "query":
                    raise ValueError("kind must be query or document")
                future = self.server.microbatcher.submit(kind, texts, titles)
                embeddings = future.result(timeout=request_timeout)
            except OverflowError:
                self.respond(503, {"error": "service overloaded"})
                return
            except TimeoutError:
                self.respond(504, {"error": "embedding request timed out"})
                return
            except _EncoderFailure:
                self.respond(500, {"error": "embedding failed"})
                return
            except (ValueError, TypeError, UnicodeDecodeError) as exc:
                self.respond(400, {"error": str(exc)})
                return
            except Exception:
                self.respond(500, {"error": "embedding failed"})
                return
            self.respond(
                200,
                {
                    "dimension": encoder.dimension,
                    "embeddings": embeddings,
                    "cache_fingerprint": getattr(encoder, "cache_fingerprint", None),
                },
            )

    server = _BoundedThreadingHTTPServer(
        (host, port), Handler, max_workers=max_workers, max_queue_size=max_queue_size
    )
    server.microbatcher = _Microbatcher(
        encoder,
        max_queue_size=max_queue_size,
        max_batch_size=max_batch_size,
        batch_wait_ms=batch_wait_ms,
    )
    return server


@hydra.main(version_base=None, config_path="../configs", config_name="embeddinggemma")
def main(config: DictConfig) -> None:
    """Load the model once and serve local embedding requests."""
    encoder = EmbeddingGemmaEncoder(
        model_id=config.model_id,
        revision=config.revision,
        dimension=config.dimension,
        device=config.device,
        batch_size=config.batch_size,
        cpu_threads=config.cpu_threads,
        sort_by_length=config.sort_by_length,
        backend=config.backend,
        max_seq_length=config.max_seq_length,
        quantization=config.quantization,
    )
    server = make_server(
        encoder,
        host=config.host,
        port=config.port,
        max_workers=config.max_workers,
        max_queue_size=config.max_queue_size,
        max_batch_size=config.max_batch_size,
        batch_wait_ms=config.batch_wait_ms,
        request_timeout=config.request_timeout,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
