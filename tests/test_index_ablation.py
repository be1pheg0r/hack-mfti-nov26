"""Synthetic-only tests for index and RRF ablations."""

import hashlib
import json

import pytest

from vrscore.benchmarking.index_ablation import run_experiment


def _write_jsonl(path, rows):
    raw = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _fixture(tmp_path):
    gt = tmp_path / "synthetic_gt"
    gt.mkdir()
    resumes = [
        {"id": "r1", "document_id": "rd1", "text_version": "rv1", "text_sha256": "rh1"},
        {"id": "r2", "document_id": "rd2", "text_version": "rv2", "text_sha256": "rh2"},
    ]
    vacancies = [
        {"id": "v1", "document_id": "vd1", "text_version": "vv1", "text_sha256": "vh1"},
        {"id": "v2", "document_id": "vd2", "text_version": "vv2", "text_sha256": "vh2"},
    ]
    pairs = [
        {"resume_id": r, "vacancy_id": v, "relevance": rel}
        for r, labels in (("r1", (3, 0)), ("r2", (0, 1)))
        for v, rel in zip(("v1", "v2"), labels, strict=True)
    ]
    extractions = [
        {
            "id": "r1",
            "kind": "resume",
            "document_id": "rd1",
            "text_version": "rv1",
            "text_sha256": "rh1",
            "entities": [
                {"category": "hard_skill", "value": "Python", "polarity": "present"}
            ],
        },
        {
            "id": "r2",
            "kind": "resume",
            "document_id": "rd2",
            "text_version": "rv2",
            "text_sha256": "rh2",
            "entities": [],
        },
        {
            "id": "v1",
            "kind": "vacancy",
            "document_id": "vd1",
            "text_version": "vv1",
            "text_sha256": "vh1",
            "entities": [
                {"category": "hard_skill", "value": "python", "importance": "strict"}
            ],
        },
        {
            "id": "v2",
            "kind": "vacancy",
            "document_id": "vd2",
            "text_version": "vv2",
            "text_sha256": "vh2",
            "entities": [
                {"category": "hard_skill", "value": "Rust", "importance": "strict"}
            ],
        },
    ]
    hashes = {}
    for filename, rows in (
        ("resumes.jsonl", resumes),
        ("vacancies.jsonl", vacancies),
        ("pairs.jsonl", pairs),
        ("requirements.jsonl", []),
        ("llm_extractions.jsonl", extractions),
    ):
        hashes[filename] = _write_jsonl(gt / filename, rows)
    (gt / "manifest.json").write_text(
        json.dumps(
            {
                "version": "synthetic-v1",
                "sha256": hashes,
                "actual": {"resumes": 2, "vacancies": 2, "pairs": 4, "requirements": 0},
            }
        )
    )
    return gt


def _config(gt):
    return {
        "run_name": "synthetic-index-ablation",
        "gt_dir": str(gt),
        "cutoffs": [1, 2],
        "gate": {"enabled": False, "unknown_policy": "include"},
        "indexes": [
            {"name": "exact", "type": "exact", "config": {}},
            {"name": "bm25", "type": "bm25", "config": {}},
            {"name": "tantivy", "type": "tantivy", "config": {}},
        ],
        "rrf": {
            "enabled": True,
            "k": 60,
            "combinations": [["exact", "bm25"], ["bm25", "tantivy"]],
        },
        "scope": {
            "excluded_methods": {
                "vector_store": {
                    "reason": (
                        "LocalVectorStore requires vectors and no encoder is "
                        "configured in this pipeline."
                    )
                },
                "rrf_exact_tantivy": {
                    "reason": (
                        "Exact and Tantivy share the same exact-skill matching "
                        "signal; this fusion is redundant."
                    )
                },
            }
        },
    }


def test_indexes_and_pairwise_rrf_share_complete_synthetic_query_pool(tmp_path):
    result = run_experiment(_config(_fixture(tmp_path)))
    variants = result["variants"]
    assert set(variants) == {
        "exact",
        "bm25",
        "tantivy",
        "rrf_exact_bm25",
        "rrf_bm25_tantivy",
    }
    assert result["experiment"]["scope"]["excluded_methods"]["vector_store"]["reason"]
    assert result["experiment"]["scope"]["excluded_methods"]["rrf_exact_tantivy"][
        "reason"
    ]
    assert all(
        [q["resume_id"] for q in v["queries"]] == ["r1", "r2"]
        for v in variants.values()
    )
    assert all(
        len(q["relevance_by_vacancy"]) == 2
        for v in variants.values()
        for q in v["queries"]
    )
    assert variants["exact"]["queries"][1]["ranked_ids"] == []
    assert result["errors"] == [] and result["skipped"] == []
    assert "Python" not in json.dumps(result)
    assert result["provenance"]["input_sha256"]["pairs.jsonl"]


def test_gate_include_and_exclude_are_distinct_split_variants(tmp_path):
    config = _config(_fixture(tmp_path))
    config["gate"] = {"enabled": True, "unknown_policy": "include"}
    included = run_experiment(config)
    config["gate"]["unknown_policy"] = "exclude"
    excluded = run_experiment(config)
    assert "exact__gate_include" in included["variants"]
    assert "exact__gate_exclude" in excluded["variants"]
    assert included["variants"]["exact__gate_include"]["queries"][0]["ranked_ids"] == [
        "v1"
    ]
    assert excluded["variants"]["exact__gate_exclude"]["queries"][0]["ranked_ids"] == [
        "v1"
    ]


def test_invalid_rrf_reference_fails_closed(tmp_path):
    config = _config(_fixture(tmp_path))
    config["rrf"]["combinations"] = [["exact", "missing"]]
    with pytest.raises(ValueError, match="unknown|index|variant"):
        run_experiment(config)


def test_timing_reports_separate_index_build_and_query_samples(tmp_path):
    config = _config(_fixture(tmp_path))
    config["timing"] = {"enabled": True, "warmup_iterations": 1, "iterations": 2}
    result = run_experiment(config)
    timing = result["timing"]
    assert set(timing["build_seconds"]) == {"exact", "bm25", "tantivy"}
    assert (
        timing["measurement_scope"]
        == "index.search; RRF measured separately on cached results"
    )
    assert len(timing["query_samples"]["bm25"]["r1"]["samples"]) == 2
    assert len(timing["rrf_samples"]["rrf_exact_bm25"]["r1"]["samples"]) == 2


def test_ablation_compares_stopwords_on_the_same_synthetic_pool(tmp_path):
    gt = _fixture(tmp_path)
    path = gt / "llm_extractions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["entities"][0]["value"] = "с и в Python"
    rows[3]["entities"][0]["value"] = "с и в Rust"
    checksum = _write_jsonl(path, rows)
    manifest_path = gt / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"]["llm_extractions.jsonl"] = checksum
    manifest_path.write_text(json.dumps(manifest))
    config = _config(gt)
    config["indexes"] = [
        {
            "name": "without",
            "type": "bm25",
            "config": {"stopwords": {"enabled": False}},
        },
        {"name": "with", "type": "bm25", "config": {"stopwords": {"enabled": True}}},
    ]
    config["rrf"] = {"enabled": False}
    result = run_experiment(config)
    before = result["variants"]["without"]["queries"][0]
    after = result["variants"]["with"]["queries"][0]
    assert set(before["ranked_ids"]) == {"v1", "v2"}
    assert after["ranked_ids"] == ["v1"]
    assert before["relevance_by_vacancy"] == after["relevance_by_vacancy"]
    assert result["provenance"]["stopwords"]["without"]["count"] == 0
    assert result["provenance"]["stopwords"]["with"]["count"] > 0
