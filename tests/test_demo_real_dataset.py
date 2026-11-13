"""Private real-resume mode stays opt-in and accepts only frozen fixtures."""

import hashlib
import json

import pytest


def _record(identifier, text):
    return {
        "id": identifier,
        "kind": "resume",
        "title": "Fixture resume title",
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "sampling_group": "it",
        "attributes": {"description": text},
        "sections": [
            {
                "source_field": "description",
                "label": "Описание",
                "start": 0,
                "end": len(text),
            }
        ],
        "source": {"text_completeness": "complete"},
    }


def _install_fixtures(monkeypatch, directory, records):
    from vrscore.demo import service

    directory.mkdir()
    hashes = {}
    for i, record in enumerate(records, 5):
        name = f"real_r{i}.json"
        raw = json.dumps(record, ensure_ascii=False).encode()
        (directory / name).write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(service, "APPROVED_REAL_RESUME_HASHES", hashes, raising=False)
    monkeypatch.setattr(service, "REAL_RESUME_FILES", tuple(hashes), raising=False)


def test_private_input_replaces_synthetic_and_preserves_initial_cases(
    tmp_path, monkeypatch
):
    from vrscore.demo.service import DemoService

    real_dir = tmp_path / "private"
    records = [
        _record(f"private_record_id_{i}", f"Fixture resume {i}. " + "x" * 120)
        for i in range(5, 11)
    ]
    _install_fixtures(monkeypatch, real_dir, records)

    default = DemoService(None)
    enabled = DemoService(None, real_resumes_path=real_dir)
    assert [r["id"] for r in default.resumes] == [
        "r1",
        "r2",
        "r3",
        "r4",
        *(f"synthetic_r{i}" for i in range(5, 11)),
    ]
    assert [r["id"] for r in enabled.resumes] == [
        "r1",
        "r2",
        "r3",
        "r4",
        *(f"real_r{i}" for i in range(5, 11)),
    ]
    assert len(enabled.vacancies) == 3
    assert all("original_" not in json.dumps(r) for r in enabled.catalog()["resumes"])
    assert all("synthetic" not in r for r in enabled.resumes[4:])
    assert enabled.resumes[4]["text"] == records[0]["text"]
    assert enabled.documents["real_r5"].text == records[0]["text"]
    assert not any(key.startswith("synthetic_") for key in enabled.documents)


def test_private_input_fails_closed_when_any_file_changes(tmp_path, monkeypatch):
    from vrscore.demo.service import DemoService

    real_dir = tmp_path / "private"
    records = [
        _record(f"private_record_id_{i}", f"Fixture resume {i}. " + "x" * 120)
        for i in range(5, 11)
    ]
    _install_fixtures(monkeypatch, real_dir, records)
    (real_dir / "real_r8.json").write_text("edited", encoding="utf-8")
    with pytest.raises(ValueError, match="approved"):
        DemoService(None, real_resumes_path=real_dir)


def test_tampering_after_start_blocks_external_extraction(tmp_path, monkeypatch):
    from vrscore.demo.service import DemoService

    real_dir = tmp_path / "private"
    records = [_record(f"fixture_{i}", f"Text {i}. " + "x" * 120) for i in range(6)]
    _install_fixtures(monkeypatch, real_dir, records)
    service = DemoService(None, real_resumes_path=real_dir)
    (real_dir / "real_r5.json").write_text("edited", encoding="utf-8")
    with pytest.raises(ValueError, match="approved"):
        service.extract_document(service.resumes[4], kind="resume")


def test_prepared_record_shape_is_bridged_without_rebuilding_text(
    tmp_path, monkeypatch
):
    from vrscore.demo.service import DemoService

    real_dir = tmp_path / "private"
    records = [
        _record(f"private_record_id_{i}", f"Fixture resume {i}. " + "x" * 120)
        for i in range(5, 11)
    ]
    for record in records:
        record.pop("attributes")
        record.pop("sections")
    _install_fixtures(monkeypatch, real_dir, records)
    service = DemoService(None, real_resumes_path=real_dir)
    assert service.resumes[4]["text"] == records[0]["text"]


def test_demo_hydra_config_defaults_private_input_off_and_configurable():
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        default = compose(config_name="demo")
        configured = compose(
            config_name="demo",
            overrides=[
                "server.real_resumes_enabled=true",
                "server.real_resumes_path=/private/input",
            ],
        )
    assert default.server.real_resumes_enabled is False
    assert configured.server.real_resumes_enabled is True
    assert configured.server.real_resumes_path == "/private/input"


def test_sanitized_resume_ids_work_with_reference_and_current_cache_keys(
    tmp_path, monkeypatch
):
    import hashlib

    from vrscore.demo.service import DemoService

    real_dir = tmp_path / "private"
    records = [
        _record(f"private_record_id_{i}", f"Fixture resume {i}. " + "x" * 120)
        for i in range(5, 11)
    ]
    _install_fixtures(monkeypatch, real_dir, records)
    service = DemoService(None, real_resumes_path=real_dir)
    resume = service.resumes[4]
    assert service.cache_key(resume, "resume")
    assert service.documents["real_r5"].document_id == "real_r5"
    assert all(
        f"private_record_id_{i}" not in json.dumps(service.catalog())
        for i in range(5, 11)
    )

    service.extra_vacancies = [
        {
            "id": f"fixture_v{i}",
            "title": f"Fixture vacancy {i}",
            "text": f"vacancy text {i}",
        }
        for i in range(20)
    ]
    service.matching_reference = {
        "resumes": {
            "real_r5": {
                "text_sha256": hashlib.sha256(resume["text"].encode()).hexdigest()
            }
        },
        "pairs": [
            {
                "resume_id": "real_r5",
                "vacancy_id": f"fixture_v{i}",
                "rank": i + 1,
                "relevance": 3,
            }
            for i in range(20)
        ],
        "vacancies": {
            f"fixture_v{i}": {
                "text_sha256": hashlib.sha256(f"vacancy text {i}".encode()).hexdigest()
            }
            for i in range(20)
        },
    }
    reference = service.reference_order("real_r5")
    assert len(reference["ranked"]) == 20
    assert reference["ranked"][0]["vacancy_id"] == "fixture_v0"


def test_live_private_reference_matches_every_real_resume():
    """When installed, the actual reference must join all approved documents."""
    from pathlib import Path

    from vrscore.demo.service import DemoService

    root = Path(__file__).resolve().parents[1] / ".data"
    resumes, reference, vacancies = (
        root / "mfti_real_work",
        root / "mfti_matching_gt",
        root / "mfti_demo/vacancies.json",
    )
    if not all(p.exists() for p in (resumes, reference, vacancies)):
        pytest.skip("Private benchmark corpus is not installed")
    service = DemoService(
        None,
        real_resumes_path=resumes,
        matching_gt_path=reference,
        extra_vacancies_path=vacancies,
    )
    for resume in service.resumes:
        order = service.reference_order(resume["id"])
        assert len(order["ranked"]) == 20
        assert len(order["unannotated"]) == 7
