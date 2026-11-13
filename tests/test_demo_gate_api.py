"""The demo gate endpoint reports real decisions for each available vacancy."""

import json
import threading
from functools import partial
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from vrscore.demo.server import Handler
from vrscore.demo.service import DemoService


def test_demo_gate_uses_cached_extractions_and_never_ranks(monkeypatch):
    service = DemoService(None)

    def cached(document, kind):
        values = {
            "experience": 1,
            "education": "bachelor",
            "work_format": "on_site",
        }
        return {
            "id": document["id"],
            "kind": kind,
            "entities": [
                {"category": category, "value": value}
                for category, value in values.items()
            ],
        }

    monkeypatch.setattr(service, "_cached_extraction", cached)
    result = service.gate("r2")
    assert result["resume_id"] == "r2"
    assert {row["id"] for row in result["vacancies"]} >= {
        "frontend",
        "accountant",
        "curator",
    }
    assert all(row["gate"]["status"] == "pass" for row in result["vacancies"])
    assert all("rank" not in row for row in result["vacancies"])
    with pytest.raises(ValueError, match="resume"):
        service.gate("missing")


def test_failed_original_vacancy_extraction_does_not_hide_preprocessed_results(
    monkeypatch,
):
    service = DemoService(None)
    service.provider = object()

    def cached(document, kind):
        if kind == "vacancy":
            raise ValueError("incomplete response")
        return {"kind": kind, "entities": [{"category": "experience", "value": 1}]}

    monkeypatch.setattr(service, "_cached_extraction", cached)
    assert len(service.catalog()["vacancies"]) == 3
    results = service.gate("r3")["vacancies"]
    assert all(row["gate"]["status"] == "unknown" for row in results)
    assert all(row["extraction_error"] for row in results)


def test_gate_http_requires_auth_and_valid_resume():
    class Service:
        resumes = [{"id": "r2"}]

        def gate(self, resume_id):
            return {"resume_id": resume_id, "vacancies": []}

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_port
    server.RequestHandlerClass = partial(
        Handler, service=Service(), token="token", port=port
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{port}/api/demo/gate"
        data = json.dumps({"resume_id": "r2"}).encode()
        with pytest.raises(HTTPError) as unauthorized:
            urlopen(
                Request(url, data=data, headers={"Content-Type": "application/json"})
            )
        assert unauthorized.value.code == 401
        headers = {"Authorization": "Bearer token", "Content-Type": "application/json"}
        with urlopen(Request(url, data=data, headers=headers)) as response:
            assert json.load(response) == {"resume_id": "r2", "vacancies": []}
        with pytest.raises(HTTPError) as bad:
            urlopen(Request(url, data=b'{"resume_id":"unknown"}', headers=headers))
        assert bad.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_demo_gate_includes_preprocessed_vacancies_and_validates_evidence(
    tmp_path, monkeypatch
):
    path = tmp_path / "vacancies.json"
    record = {
        "id": "mfti-test",
        "kind": "vacancy",
        "title": "Инженер",
        "text": "Опыт 3 года",
        "source_url": "https://hr.mipt.ru/vacancies/123/",
        "document_id": "mfti-test",
        "text_version": "v1",
        "entities": [
            {
                "category": "experience",
                "value": 3,
                "evidence": "3 года",
                "start": 5,
                "end": 11,
            }
        ],
    }
    path.write_text(json.dumps([record], ensure_ascii=False), encoding="utf-8")
    service = DemoService(None, extra_vacancies_path=path)
    monkeypatch.setattr(
        service,
        "_cached_extraction",
        lambda document, kind: {
            "kind": kind,
            "entities": [{"category": "experience", "value": 1}],
        },
    )
    result = service.gate("r2")
    row = next(v for v in result["vacancies"] if v["id"] == "mfti-test")
    assert row["gate"]["status"] == "fail"
    assert row["gate"]["decisions"][0]["requirement"][0]["evidence"] == "3 года"
    assert row["source_url"] == record["source_url"]
    assert any(v["id"] == "mfti-test" for v in service.catalog()["vacancies"])

    record["entities"][0]["evidence"] = "fabricated"
    path.write_text(json.dumps([record], ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence"):
        DemoService(None, extra_vacancies_path=path)


def test_public_vacancy_ambiguous_experience_and_degree_are_not_misrepresented(
    tmp_path,
):
    from vrscore.demo.service import load_extra_vacancies

    text = "Опыт технического руководства. Кандидат наук. Опыт от 3 лет."

    def entity(category, value, quote):
        start = text.index(quote)
        return {
            "category": category,
            "value": value,
            "evidence": quote,
            "start": start,
            "end": start + len(quote),
        }

    record = {
        "id": "x",
        "kind": "vacancy",
        "title": "X",
        "text": text,
        "entities": [
            entity("experience", 3, "Опыт технического руководства"),
            entity("education", "postgraduate", "Кандидат наук"),
            entity("experience", 3, "Опыт от 3 лет"),
        ],
    }
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps([record], ensure_ascii=False), encoding="utf-8")
    entities = load_extra_vacancies(path)[0]["entities"]
    assert [e["value"] for e in entities if e["category"] == "experience"] == [3]
    degree = next(e for e in entities if e["category"] == "education")
    assert degree["value"] == "candidate_of_sciences"
    assert degree["original_value"] == "postgraduate"
