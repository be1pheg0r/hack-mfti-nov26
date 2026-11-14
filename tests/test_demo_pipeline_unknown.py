"""Do not exclude uncertain applicants or rank failed vacancy extraction."""

import pytest

from vrscore.demo.service import DemoService


def test_uncertain_gate_is_included_but_failed_extraction_is_not_ranked(monkeypatch):
    service = DemoService(None, demo_delay_seconds=0)
    monkeypatch.setattr(
        service, "_cached_extraction", lambda doc, kind: {"entities": []}
    )
    monkeypatch.setattr(
        service,
        "_available_vacancy",
        lambda doc: {"entities": [], "extraction_error": doc["id"] == "accountant"},
    )
    monkeypatch.setattr(
        service, "embedding_encoder", type("Encoder", (), {"cache_fingerprint": None})()
    )
    captured = {}

    class Pipeline:
        def __init__(self, vacancies, encoder, **options):
            captured["vacancies"] = vacancies
            captured["options"] = options

        def search(self, resume):
            return {
                "gates": {
                    v["id"]: {"status": "unknown"} for v in captured["vacancies"]
                },
                "lexical_ranking": [],
                "results": [],
            }

    monkeypatch.setattr("vrscore.demo.service.RetrievalPipeline", Pipeline)
    result = service.pipeline("r1")
    assert captured["options"]["include_unknown"] is True
    assert "accountant" not in {v["id"] for v in captured["vacancies"]}
    assert "accountant" in result["gate"]["unknown"]


def test_uncached_private_resume_is_not_sent_to_external_provider(monkeypatch):
    service = DemoService(None, demo_delay_seconds=0)
    resume = dict(service.resumes[0], id="real_r5")
    service.resumes = [resume]
    service.cache = {}

    def forbidden(*args, **kwargs):
        raise AssertionError("external extraction must not be called")

    monkeypatch.setattr(service, "_cached_extraction", forbidden)
    with pytest.raises(ValueError, match="private resume"):
        service.pipeline("real_r5")


def test_direct_private_extraction_is_never_sent_to_external_provider(
    tmp_path, monkeypatch
):
    import hashlib
    import json

    service = DemoService(None, demo_delay_seconds=0)
    resume = dict(service.resumes[0], id="real_r5")
    raw = json.dumps({"text": resume["text"]}).encode()
    (tmp_path / "real_r5.json").write_bytes(raw)
    service.real_resumes_path = tmp_path
    monkeypatch.setitem(
        __import__(
            "vrscore.demo.service", fromlist=["APPROVED_REAL_RESUME_HASHES"]
        ).APPROVED_REAL_RESUME_HASHES,
        "real_r5.json",
        hashlib.sha256(raw).hexdigest(),
    )
    with pytest.raises(ValueError, match="private resume"):
        service.extract_document(resume, kind="resume")
