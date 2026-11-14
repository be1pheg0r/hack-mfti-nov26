"""Loopback client for the locally hosted embedding service."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Sequence
from urllib.parse import urlsplit


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class EmbeddingClient:
    """Expose the same query/document methods as the in-process encoder."""

    def __init__(self, url: str = "http://127.0.0.1:18766", timeout: float = 120):
        parts = urlsplit(url)
        if (
            parts.scheme != "http"
            or parts.hostname not in ("127.0.0.1", "localhost")
            or parts.username is not None
            or parts.password is not None
            or parts.path not in ("", "/")
            or parts.query
            or parts.fragment
        ):
            raise ValueError("Only a loopback HTTP service origin is allowed")
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._cache_fingerprint = None
        self._fingerprint_loaded = False

    @property
    def cache_fingerprint(self):
        """Fetch the serving model identity for opt-in persistent caching."""
        if not self._fingerprint_loaded:
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({}), _NoRedirect()
            )
            with opener.open(self.url + "/health", timeout=self.timeout) as response:
                self._cache_fingerprint = json.load(response).get("cache_fingerprint")
            self._fingerprint_loaded = True
        return self._cache_fingerprint

    def _encode(self, kind: str, texts: Sequence[str], **kwargs) -> list[list[float]]:
        if not texts:
            return []
        data = json.dumps({"kind": kind, "texts": list(texts), **kwargs}).encode()
        request = urllib.request.Request(
            self.url + "/embed", data=data, headers={"Content-Type": "application/json"}
        )
        # Never follow redirects with document content.
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect()
        )
        with opener.open(request, timeout=self.timeout) as response:
            result = json.load(response)
        if (
            self._fingerprint_loaded
            and self._cache_fingerprint is not None
            and result.get("cache_fingerprint") != self._cache_fingerprint
        ):
            raise RuntimeError("Embedding server fingerprint changed during retrieval")
        return result["embeddings"]

    def encode_query(self, text: str) -> list[float]:
        """Encode one resume query."""
        return self._encode("query", [text])[0]

    def encode_documents(
        self, texts: Sequence[str], *, titles: Sequence[str | None] | None = None
    ) -> list[list[float]]:
        """Encode vacancy texts with optional titles."""
        return self._encode("document", texts, titles=titles)
