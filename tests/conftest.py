"""Shared in-memory document fixtures."""

from io import BytesIO
from socketserver import BaseServer

import pytest
from docx import Document


@pytest.fixture(autouse=True)
def fast_test_server_shutdown(monkeypatch):
    """Avoid the default half-second shutdown polling delay in test servers."""
    serve_forever = BaseServer.serve_forever

    def serve_with_short_poll(server, poll_interval=0.005):
        return serve_forever(server, poll_interval=poll_interval)

    monkeypatch.setattr(BaseServer, "serve_forever", serve_with_short_poll)


@pytest.fixture
def docx_bytes():
    """Build a real DOCX, optionally customising its content."""

    def build(populate=None):
        document = Document()
        if populate is not None:
            populate(document)
        stream = BytesIO()
        document.save(stream)
        return stream.getvalue()

    return build
