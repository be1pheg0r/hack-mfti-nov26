"""Demo extraction must use grounded provider output and protect its API."""

import pytest


def test_extraction_checks_evidence_and_reuses_cache(tmp_path):
    from shutil import copytree

    from vrscore.demo.extraction import Entity, Extracted
    from vrscore.demo.service import (
        CASE_FILES,
        RESUME_FILES,
        DemoService,
        load_case_documents,
    )

    def entity(category, value, quote):
        return Entity(
            category=category,
            value=value,
            polarity="present",
            context=None,
            importance=None,
            quote=quote,
        )

    class Provider:
        calls = 0

        def structured(self, messages, output_type, *, public_data):
            assert public_data is True
            self.calls += 1
            text = messages[-1]["content"].split("<source_document>\n", 1)[1]
            return (
                Extracted(entities=[entity("hard_skill", "Python", "Python")])
                if "Python" in text
                else Extracted(entities=[])
            )

    provider = Provider()
    service = DemoService(provider)
    from pathlib import Path

    from vrscore.documents.docx import DocxReader

    original = Path("public_data/raw/Иванов Алексей_резюме 1.docx")
    parsed = DocxReader().read(original.read_bytes(), filename=original.name)
    assert service.catalog()["resumes"][0]["text"] == parsed.text
    assert len(service.catalog()["vacancies"][0]["text"]) > 1000
    catalog = service.catalog()
    for entries, file_info in (
        (catalog["resumes"], RESUME_FILES),
        (catalog["vacancies"], CASE_FILES),
    ):
        for entry, info in zip(entries, file_info, strict=True):
            path = original.parent / info[1]
            expected = DocxReader().read(path.read_bytes(), filename=path.name)
            assert entry["text"] == expected.text
    copied = tmp_path / "raw"
    copytree(original.parent, copied)
    (copied / original.name).write_bytes(b"changed document")
    with pytest.raises(ValueError, match="approved"):
        load_case_documents(copied)
    first = service.analyze("frontend")
    assert provider.calls == len(service.resumes)
    ivanov = next(r for r in first["resumes"] if r["id"] == "r1")
    assert ivanov["entities"][0]["evidence"] == "Python"
    service.analyze("frontend")
    assert provider.calls == len(service.resumes) + 1
    assert service.catalog()["resumes"][0]["text"]
    selected = service.analyze("frontend", resume_id="r2")
    assert provider.calls == len(service.resumes) + 2
    assert next(r for r in selected["resumes"] if r["id"] == "r2")["id"] == "r2"
    with pytest.raises(ValueError, match="resume"):
        service.analyze("frontend", resume_id="unknown")

    class BadProvider:
        def structured(self, messages, output_type, *, public_data):
            return Extracted(
                entities=[entity("hard_skill", "x", "несуществующая цитата")]
            )

    with pytest.raises(ValueError, match="quote"):
        DemoService(BadProvider()).analyze("frontend")

    class PartlyBadProvider:
        def structured(self, messages, output_type, *, public_data):
            text = messages[-1]["content"].split("<source_document>\n", 1)[1]
            first_line = next(line for line in text.splitlines() if line.strip())
            return Extracted(
                entities=[
                    entity("hard_skill", "a", first_line[:5]),
                    entity("hard_skill", "b", "несуществующая цитата"),
                ]
            )

    result = DemoService(PartlyBadProvider()).analyze("frontend")
    assert all(
        r["rejected"] == 1 and len(r["entities"]) == 1 for r in result["resumes"]
    )


def test_http_auth_and_cross_origin_denial():
    """The API is inaccessible without a token or from a hostile Origin."""
    import threading
    from functools import partial
    from http.server import ThreadingHTTPServer
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    from vrscore.demo.server import Handler
    from vrscore.demo.service import DemoService

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    service = DemoService(None)
    server.RequestHandlerClass = partial(
        Handler,
        service=service,
        token="test-token",
        port=port,
        public_origin="https://demo.example",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{port}/api/demo"
        with pytest.raises(HTTPError) as unauthorized:
            urlopen(url)
        assert unauthorized.value.code == 401
        headers = {"Authorization": "Bearer test-token"}
        with urlopen(Request(url, headers=headers)) as response:
            assert response.status == 200
        with urlopen(f"http://127.0.0.1:{port}/logo.png") as response:
            assert response.headers.get_content_type() == "image/png"
            assert response.read(8) == b"\x89PNG\r\n\x1a\n"
        with urlopen(f"http://127.0.0.1:{port}/favicon.ico") as response:
            assert response.headers.get_content_type() == "image/vnd.microsoft.icon"
            assert response.read(4) == b"\x00\x00\x01\x00"
        headers["Origin"] = "http://evil.example"
        with pytest.raises(HTTPError) as forbidden:
            urlopen(Request(url, headers=headers))
        assert forbidden.value.code == 403
        headers.update({"Host": "demo.example", "Origin": "https://demo.example"})
        with urlopen(Request(url, headers=headers)) as response:
            assert response.status == 200
        import json

        class FailedProvider:
            def structured(self, messages, output_type, *, public_data):
                raise ValueError("truncated model output")

        service.provider = FailedProvider()
        headers["Content-Type"] = "application/json"
        with pytest.raises(HTTPError) as failed:
            urlopen(
                Request(
                    url + "/analyze",
                    headers=headers,
                    data=json.dumps({"vacancy_id": "frontend"}).encode(),
                )
            )
        assert failed.value.code == 502
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_server_waits_for_inflight_request_on_shutdown():
    """A service restart must not abort a running analysis response."""
    import threading
    from http.server import BaseHTTPRequestHandler
    from urllib.request import urlopen

    from vrscore.demo.server import ThreadingHTTPServer

    entered, release = threading.Event(), threading.Event()

    class SlowHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            entered.set()
            assert release.wait(3)
            body = b'{"ok": true}'
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), SlowHandler)
    assert server.daemon_threads is False
    serving = threading.Thread(target=server.serve_forever)
    serving.start()
    result = []
    client = threading.Thread(
        target=lambda: result.append(
            urlopen(f"http://127.0.0.1:{server.server_port}", timeout=3).read()
        )
    )
    client.start()
    try:
        assert entered.wait(2)
        server.shutdown()
        closing = threading.Thread(target=server.server_close)
        closing.start()
        assert closing.is_alive()
        release.set()
        closing.join(3)
        client.join(3)
        assert result == [b'{"ok": true}']
        assert not closing.is_alive()
    finally:
        release.set()
        serving.join(3)
        client.join(3)
