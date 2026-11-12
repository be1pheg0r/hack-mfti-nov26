"""Exercise real viewer loading, HTTP responses and private-file isolation."""

import importlib.util
import json
import subprocess
import sys
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest


@pytest.fixture
def viewer():
    """Load the viewer without depending on test runner import paths."""
    path = Path(__file__).resolve().parents[1] / "dataset_viewer" / "server.py"
    assert path.is_file(), "The dataset viewer server is not implemented"
    spec = importlib.util.spec_from_file_location("viewer_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def corpus(tmp_path):
    """Create explicit synthetic fixtures; never include private dataset content."""
    records = [
        {"id": "resume_001", "kind": "resume", "title": "Тестировщик"},
        {"id": "vacancy_001", "kind": "vacancy", "title": "Разработчик"},
    ]
    for filename, record in zip(["resumes.jsonl", "vacancies.jsonl"], records):
        (tmp_path / filename).write_text(json.dumps(record), encoding="utf-8")
    manifest = {
        "name": "fixture",
        "actual": {"resumes": 1, "vacancies": 1},
        "provenance": {"copy_repository": "https://example.invalid"},
        "private_debug_path": "/not-for-browser",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_dataset_loading_and_validation(viewer, corpus):
    """Return actual records and safe metadata; reject corrupt dataset counts."""
    payload = viewer.load_dataset(corpus)
    assert [r["id"] for r in payload["records"]] == ["resume_001", "vacancy_001"]
    assert payload["manifest"]["actual"] == {"resumes": 1, "vacancies": 1}
    assert "private_debug_path" not in payload["manifest"]
    assert len(payload["groups"]) == 5
    (corpus / "vacancies.jsonl").write_text(
        json.dumps({"id": "resume_001", "kind": "vacancy"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Duplicate"):
        viewer.load_dataset(corpus)
    (corpus / "vacancies.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="count"):
        viewer.load_dataset(corpus)


def test_http_isolation_and_security_headers(viewer, corpus, tmp_path):
    """Serve only allowed assets/API; do not disclose arbitrary local files."""
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "index.html").write_text("<!doctype html><title>Atlas</title>")
    (assets / "app.js").write_text("console.log('fixture');")
    (assets / "styles.css").write_text("body { color: black; }")
    (assets / "secret.txt").write_text("do not expose")
    server = viewer.create_server(corpus, assets, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/dataset", timeout=2) as response:
            assert len(json.load(response)["records"]) == 2
            assert response.headers["Cache-Control"] == "no-store"
            assert response.headers["X-Content-Type-Options"] == "nosniff"
        for route, expected in [("/", "text/html"), ("/app.js", "javascript")]:
            with urlopen(base + route, timeout=2) as response:
                assert expected in response.headers["Content-Type"]
        for route in [
            "/.env",
            "/secret.txt",
            "/../manifest.json",
            "/%2e%2e/manifest.json",
            "/api/unknown",
        ]:
            with pytest.raises(HTTPError) as error:
                urlopen(base + route, timeout=2)
            assert error.value.code == 404
        request = Request(base + "/api/dataset", headers={"Host": "evil.example"})
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=2)
        assert error.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    with pytest.raises(ValueError, match="loopback"):
        viewer.create_server(corpus, assets, "0.0.0.0", 0)


def test_module_hydra_overrides():
    """The actual module CLI composes private proxy overrides without starting."""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dataset_viewer",
            "--cfg",
            "job",
            "server.port=9876",
            "server.host=172.18.0.1",
            "server.allow_private_bind=true",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert result.returncode == 0, result.stderr
    assert "port: 9876" in result.stdout
    assert "allow_private_bind: true" in result.stdout


def test_switch_datasets_without_exposing_paths(viewer, corpus, tmp_path):
    """Each explicitly configured dataset is isolated by its selector."""
    import shutil

    medium = tmp_path / "medium"
    shutil.copytree(corpus, medium)
    manifest_path = medium / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["name"] = "trudvsem_medium"
    manifest_path.write_text(json.dumps(manifest))
    (medium / "resumes.jsonl").write_text(
        json.dumps({"id": "resume_001", "kind": "resume", "title": "medium"})
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    server = viewer.create_server(
        corpus, assets, "127.0.0.1", 0, extra_datasets={"medium": medium}
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/datasets") as response:
            catalog = json.load(response)["datasets"]
        assert [entry["id"] for entry in catalog] == ["mini", "medium"]
        assert catalog[1]["name"] == "trudvsem_medium"
        assert str(tmp_path) not in json.dumps(catalog)
        for suffix, title in [("", "Тестировщик"), ("?dataset=medium", "medium")]:
            with urlopen(base + "/api/dataset" + suffix) as response:
                assert json.load(response)["records"][0]["title"] == title
        for query in (
            "?dataset=unknown",
            "?dataset=../medium",
            "?dataset=mini&dataset=medium",
        ):
            with pytest.raises(HTTPError) as error:
                urlopen(base + "/api/dataset" + query)
            assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
