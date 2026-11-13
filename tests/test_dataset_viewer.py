"""Exercise real viewer loading, HTTP responses and private-file isolation."""

import json
import subprocess
import sys
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from vrscore.dataset_viewer import server as viewer_module


@pytest.fixture
def viewer():
    """Provide the dataset viewer module."""
    return viewer_module


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


def test_optional_pairs_are_validated_and_returned(viewer, corpus):
    """Pair records require real references, unique bounded ranks and labels."""
    pairs = [
        {
            "resume_id": "resume_001",
            "vacancy_id": "vacancy_001",
            "rank": 1,
            "relevance": 3,
            "rationale": "Подходит",
            "annotator": "reviewer",
            "evidence": [{"resume_quote": "Python", "vacancy_quote": "Python"}],
        }
    ]
    (corpus / "pairs.jsonl").write_text(json.dumps(pairs[0]), encoding="utf-8")
    payload = viewer.load_dataset(corpus)
    assert payload["pairs"] == pairs
    assert "labels" not in payload["manifest"]

    invalid = [
        {**pairs[0], "vacancy_id": "missing"},
        {**pairs[0], "rank": 21},
        {**pairs[0], "relevance": True},
        {**pairs[0], "evidence": [{"resume_quote": "x"}]},
        {**pairs[0], "resume_id": "other", "vacancy_id": "vacancy_001"},
    ]
    for pair in invalid:
        (corpus / "pairs.jsonl").write_text(json.dumps(pair), encoding="utf-8")
        with pytest.raises(
            ValueError, match="pair|Pair|rank|evidence|reference|resume"
        ):
            viewer.load_dataset(corpus)


def test_pair_rank_and_identity_integrity(viewer, corpus):
    """A resume cannot repeat a pair or a rank, and optional files may be absent."""
    assert viewer.load_dataset(corpus)["pairs"] == []
    (corpus / "vacancies.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"id": "vacancy_001", "kind": "vacancy", "title": "Разработчик"},
                {"id": "vacancy_002", "kind": "vacancy", "title": "Аналитик"},
            )
        ),
        encoding="utf-8",
    )
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    manifest["actual"]["vacancies"] = 2
    (corpus / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    base = {
        "resume_id": "resume_001",
        "vacancy_id": "vacancy_001",
        "rank": 1,
        "relevance": 2,
        "rationale": "",
        "annotator": "reviewer",
        "evidence": [],
    }
    (corpus / "pairs.jsonl").write_text(
        "\n".join(json.dumps(row) for row in (base, {**base, "rank": 2})),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate pair"):
        viewer.load_dataset(corpus)
    (corpus / "pairs.jsonl").write_text(
        "\n".join(
            json.dumps(row) for row in (base, {**base, "vacancy_id": "vacancy_002"})
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="rank"):
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
            "vrscore.dataset_viewer",
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
    assert "assets_dir: vrscore/web/dataset_viewer" in result.stdout
    assert (root / "vrscore/web/dataset_viewer/index.html").is_file()
    for module in (
        "vrscore.vacancy_parser",
        "vrscore.vacancy_parser.partition",
    ):
        result = subprocess.run(
            [sys.executable, "-m", module, "--cfg", "job", "source_dir=fixture"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=3,
        )
        assert result.returncode == 0, result.stderr
        assert "source_dir: fixture" in result.stdout


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
