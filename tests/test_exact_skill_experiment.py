"""Contract tests for the private, reproducible exact-skills experiment."""

import hashlib
import json

import pytest

from vrscore.benchmarking.exact_skill_experiment import run_experiment


def _write_jsonl(path, rows):
    raw = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
        for row in rows
    )
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _fixture(tmp_path):
    gt = tmp_path / "gt"
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
        {"resume_id": resume, "vacancy_id": vacancy, "relevance": relevance}
        for resume, labels in (("r1", (3, 0)), ("r2", (0, 1)))
        for vacancy, relevance in zip(("v1", "v2"), labels, strict=True)
    ]
    extraction = [
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
    for name, rows in (
        ("resumes", resumes),
        ("vacancies", vacancies),
        ("pairs", pairs),
        ("requirements", []),
        ("llm_extractions", extraction),
    ):
        hashes[f"{name}.jsonl"] = _write_jsonl(gt / f"{name}.jsonl", rows)
    manifest = {"actual": {"resumes": 2, "vacancies": 2, "pairs": 4}, "sha256": hashes}
    (gt / "manifest.json").write_text(json.dumps(manifest))
    return gt


def test_complete_pool_scores_all_queries_and_zero_hit_as_zero(tmp_path):
    gt = _fixture(tmp_path)
    result = run_experiment({"gt_dir": str(gt), "cutoffs": [1, 5]})
    exact = result["variants"]["exact"]
    assert result["dataset"]["resumes_total"] == 2
    assert len(exact["queries"]) == 2
    assert exact["queries"][1]["ranked_ids"] == []
    assert exact["queries"][1]["metrics"]["ndcg@1"] == 0
    assert exact["mean_metrics"]["ndcg@1"] == pytest.approx(0.5)
    assert len(exact["queries"][0]["relevance_by_vacancy"]) == 2
    assert exact["queries"][0]["hits"][0] == {
        "vacancy_id": "v1",
        "rank": 1,
        "score": 1.0,
    }
    assert exact["queries"][1]["no_hits"] is True
    assert result["errors"] == []
    assert result["skipped"] == []
    assert "Python" not in json.dumps(result)
    fallback = run_experiment(
        {
            "gt_dir": str(gt),
            "cutoffs": [1, 5],
            "run_gate": False,
            "include_full_list_fallback": True,
        }
    )
    assert (
        len(fallback["variants"]["exact_then_vacancy_id"]["queries"][1]["ranked_ids"])
        == 2
    )


def test_incomplete_join_is_fatal_and_never_shrinks_pool(tmp_path):
    gt = _fixture(tmp_path)
    path = gt / "pairs.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()][:-1]
    hashes = _write_jsonl(path, rows)
    manifest = json.loads((gt / "manifest.json").read_text())
    manifest["sha256"]["pairs.jsonl"] = hashes
    (gt / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="complete|200|join|pair"):
        run_experiment({"gt_dir": str(gt)})


def test_extraction_identity_must_match_ground_truth_source(tmp_path):
    gt = _fixture(tmp_path)
    path = gt / "llm_extractions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["text_version"] = "wrong-version"
    hashes = _write_jsonl(path, rows)
    manifest = json.loads((gt / "manifest.json").read_text())
    manifest["sha256"]["llm_extractions.jsonl"] = hashes
    (gt / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity"):
        run_experiment({"gt_dir": str(gt)})


def test_manifest_checksums_are_enforced(tmp_path):
    gt = _fixture(tmp_path)
    with (gt / "resumes.jsonl").open("ab") as stream:
        stream.write(b"{}\n")
    with pytest.raises(ValueError, match="checksum|sha256"):
        run_experiment({"gt_dir": str(gt)})
