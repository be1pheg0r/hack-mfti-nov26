"""Exercise discovery of additional private dataset directories."""

import json
import shutil
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from dataset_viewer.server import create_server


def make_corpus(directory, name):
    """Write a small, deterministic viewer-compatible corpus."""
    directory.mkdir()
    (directory / "manifest.json").write_text(
        json.dumps({"name": name, "actual": {"resumes": 1, "vacancies": 1}})
    )
    for filename, record in (
        ("resumes.jsonl", {"id": "resume_001", "kind": "resume"}),
        (
            "vacancies.jsonl",
            {"id": "vacancy_001", "kind": "vacancy", "sampling_group": "science"},
        ),
    ):
        (directory / filename).write_text(json.dumps(record))


def test_discover_compatible_datasets_safely(tmp_path):
    """Discover corpora without leaking paths or following escaping symlinks."""
    corpus = tmp_path / "default"
    make_corpus(corpus, "default")
    root = tmp_path / "datasets"
    root.mkdir()
    medium = root / "trudvsem_medium"
    mfti = root / "mfti_vacancies"
    make_corpus(medium, "medium")
    make_corpus(mfti, "МФТИ")
    (mfti / "resumes.jsonl").unlink()
    (mfti / "resumes.jsonl").symlink_to("../trudvsem_medium/resumes.jsonl")
    bad = root / "bad"
    shutil.copytree(corpus, bad)
    (bad / "vacancies.jsonl").write_text("not json")
    wrong_shape = root / "wrong-shape"
    shutil.copytree(corpus, wrong_shape)
    (wrong_shape / "vacancies.jsonl").write_text("[]")
    escaped = root / "escaped"
    shutil.copytree(corpus, escaped)
    (escaped / "resumes.jsonl").unlink()
    (escaped / "resumes.jsonl").symlink_to(corpus / "resumes.jsonl")
    (root / "outside-link").symlink_to(corpus, target_is_directory=True)
    assets = tmp_path / "assets"
    assets.mkdir()
    server = create_server(
        corpus,
        assets,
        "127.0.0.1",
        0,
        extra_datasets={"medium": medium},
        discovery_root=root,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/datasets", timeout=2) as response:
            catalog = json.load(response)["datasets"]
        assert [item["id"] for item in catalog] == ["mini", "medium", "mfti_vacancies"]
        assert str(tmp_path) not in json.dumps(catalog)
        url = base + "/api/dataset?dataset=mfti_vacancies"
        with urlopen(url, timeout=2) as response:
            data = json.load(response)
        assert data["manifest"]["name"] == "МФТИ"
        assert data["records"][1]["sampling_group"] == "science"
        assert any(group["id"] == "science" for group in data["groups"])
        for dataset in (
            "bad",
            "wrong-shape",
            "escaped",
            "outside-link",
            "trudvsem_medium",
        ):
            with pytest.raises(HTTPError) as error:
                urlopen(base + "/api/dataset?dataset=" + dataset, timeout=2)
            assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_discovery_works_without_default_corpora(tmp_path):
    """A new compatible corpus works even when the built-in ones are absent."""
    root = tmp_path / "datasets"
    root.mkdir()
    make_corpus(root / "new_dataset", "Новый набор")
    assets = tmp_path / "assets"
    assets.mkdir()
    server = create_server(
        root / "missing-mini",
        assets,
        "127.0.0.1",
        0,
        extra_datasets={"medium": root / "missing-medium"},
        discovery_root=root,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/datasets", timeout=2) as response:
            catalog = json.load(response)
        assert catalog["default_dataset"] == "new_dataset"
        with urlopen(base + "/api/dataset", timeout=2) as response:
            assert json.load(response)["manifest"]["name"] == "Новый набор"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
