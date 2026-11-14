"""Tests for the Malt paired shortlist ranking protocol."""

import sqlite3

import numpy as np
import pytest

from vrscore.benchmarking import malt_embedding_run as experiment
from vrscore.benchmarking.malt_embedding_ablation import rerank_shortlist


def test_malt_default_cache_is_milvus():
    assert experiment.DEFAULT_MILVUS_MALT_COLLECTION.startswith(
        "malt_embedding_cache_v1_"
    )


def test_rerank_reorders_only_candidates_by_cosine_and_checks_values():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    got = rerank_shortlist(
        ["a", "b"], np.array([0.0, 1.0]), matrix, {"a": 0, "b": 1, "c": 2}
    )
    assert [x[0] for x in got] == ["b", "a"]
    with pytest.raises(ValueError):
        rerank_shortlist(["a", "a"], np.array([0.0, 1.0]), matrix, {"a": 0})


def test_experiment_scores_matched_pool_and_excludes_zero_ideal(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "OUT", tmp_path)
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE embeddings(role TEXT, id TEXT, vector BLOB)")
    v1, v2 = np.eye(128, dtype="<f4")[:2]
    for role, identifier, vector in [
        ("document", "a", v1),
        ("document", "b", v2),
        ("query", "p1", v2),
        ("query", "p2", v1),
    ]:
        db.execute(
            "INSERT INTO embeddings VALUES (?,?,?)",
            (role, identifier, vector.tobytes()),
        )
    profiles = {"p1": "x", "p2": "y"}
    briefs = {"a": "a", "b": "b"}
    labels = {"p1": {"a": 0, "b": 2}, "p2": {"a": 0, "b": -1}}
    ranks = {p: {m: ["a", "b"] for m in experiment.METHODS} for p in profiles}
    experiment.evaluate(profiles, briefs, labels, ranks, {}, db)
    import json

    result = json.loads((tmp_path / "results.json").read_text())
    summary = result["summary"]["bm25_stop_ru:m10"]
    assert summary["positive_ideal_count"] == 1
    assert summary["macro"]["bm25"]["1"] == 0
    assert summary["macro"]["embedding"]["1"] == 1
    assert summary["paired_ndcg5"]["better"] == 1
    db.close()
