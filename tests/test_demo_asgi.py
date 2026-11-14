"""ASGI demo server preserves the existing HTTP security contract."""

import asyncio
import json

from vrscore.demo.server import DemoASGI


async def request(app, method="GET", path="/", headers=None, body=b""):
    sent = []
    messages = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive():
        return messages.pop(0)

    async def send(message):
        sent.append(message)

    await app(
        {
            "type": "http",
            "method": method,
            "path": path,
            "headers": [
                (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 18765),
        },
        receive,
        send,
    )
    start = sent[0]
    response_body = b"".join(m.get("body", b"") for m in sent[1:])
    return start["status"], dict(start["headers"]), response_body


class Service:
    resumes = [{"id": "r1"}]
    vacancies = [{"id": "v1"}]

    def catalog(self):
        return {"ok": True}

    def pipeline(self, resume_id):
        return {"resume_id": resume_id}

    def gate(self, resume_id):
        return {"resume_id": resume_id}

    def analyze(self, vacancy_id, resume_id=None):
        return {"vacancy_id": vacancy_id}


def test_asgi_auth_host_origin_headers_and_existing_routes():
    app = DemoASGI(Service(), token="secret", port=18765)
    status, headers, body = asyncio.run(
        request(app, "GET", "/api/demo", {"host": "127.0.0.1:18765"})
    )
    assert status == 401
    assert json.loads(body)["error"] == "Требуется токен"
    assert headers[b"cache-control"] == b"no-store"
    assert headers[b"x-content-type-options"] == b"nosniff"
    assert b"frame-ancestors 'none'" in headers[b"content-security-policy"]

    headers = {
        "host": "127.0.0.1:18765",
        "authorization": "Bearer secret",
        "origin": "http://127.0.0.1:18765",
    }
    status, _, body = asyncio.run(request(app, "GET", "/api/demo", headers))
    assert status == 200 and json.loads(body) == {"ok": True}

    headers["origin"] = "https://attacker.example"
    status, _, _ = asyncio.run(request(app, "GET", "/api/demo", headers))
    assert status == 403


def test_asgi_pipeline_validates_body_and_authenticates():
    app = DemoASGI(Service(), token="secret", port=18765)
    headers = {"host": "127.0.0.1:18765", "content-type": "application/json"}
    status, _, _ = asyncio.run(
        request(app, "POST", "/api/demo/pipeline", headers, b'{"resume_id":"r1"}')
    )
    assert status == 401

    headers["authorization"] = "Bearer secret"
    status, _, body = asyncio.run(
        request(app, "POST", "/api/demo/pipeline", headers, b'{"resume_id":"r1"}')
    )
    assert status == 200 and json.loads(body) == {"resume_id": "r1"}

    status, _, _ = asyncio.run(
        request(app, "POST", "/api/demo/pipeline", headers, b'{"resume_id":"bad"}')
    )
    assert status == 400


def test_public_origin_does_not_allow_http_downgrade():
    app = DemoASGI(
        Service(), token="secret", port=18765, public_origin="https://demo.example"
    )
    headers = {"host": "demo.example", "origin": "http://demo.example"}
    status, _, _ = asyncio.run(request(app, "GET", "/", headers))
    assert status == 403


def test_report_api_is_authenticated_and_checks_resume():
    class ReportService(Service):
        def report(self, resume_id):
            return {"resume_id": resume_id, "vacancies": []}

    app = DemoASGI(ReportService(), token="secret", port=18765)
    headers = {"host": "127.0.0.1:18765", "content-type": "application/json"}
    body = b'{"resume_id":"r1"}'
    status, _, _ = asyncio.run(request(app, "POST", "/api/demo/report", headers, body))
    assert status == 401
    headers["authorization"] = "Bearer secret"
    status, _, result = asyncio.run(
        request(app, "POST", "/api/demo/report", headers, body)
    )
    assert status == 200 and json.loads(result)["resume_id"] == "r1"
    status, _, _ = asyncio.run(
        request(app, "POST", "/api/demo/report", headers, b'{"resume_id":"bad"}')
    )
    assert status == 400
