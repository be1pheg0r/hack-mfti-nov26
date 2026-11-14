"""Tests for resume vocabulary and isolated BM25 IDF ablation."""

import json
import math

from test_index_ablation import _fixture, _write_jsonl

from vrscore.benchmarking.vocabulary_ablation import (
    build_vocabulary,
    pretrained_index,
    run_experiment,
)


def test_resume_document_frequency_and_bm25_unseen_fallback():
    rows = [
        {"id": "r1", "attributes": {"hardSkills": ["Python", "Python", "C++"]}},
        {"id": "r2", "attributes": {"softSkills": ["Python", "SQL"]}},
    ]
    vocabulary = build_vocabulary(rows)
    assert vocabulary["documents"] == 2
    assert vocabulary["tokens"]["python"] == 2
    assert vocabulary["tokens"]["c++"] == 1
    assert vocabulary["skills"]["python"]["document_frequency"] == 2
    vacancies = [
        {"id": "v1", "entities": [{"category": "hard_skill", "value": "Python Rust"}]},
        {"id": "v2", "entities": [{"category": "hard_skill", "value": "SQL"}]},
    ]
    base, pretrained = pretrained_index(vacancies, vocabulary)
    assert pretrained._model.idf["python"] == math.log(3 / 2)
    assert pretrained._model.idf["rust"] == base._model.idf["rust"]
    query = {
        "id": "q",
        "entities": [
            {"category": "hard_skill", "value": "Rust", "polarity": "present"}
        ],
    }
    assert [hit.vacancy_id for hit in pretrained.search(query).hits] == ["v1"]


def test_end_to_end_excludes_overlap_and_mines_judged_negatives(tmp_path):
    gt = _fixture(tmp_path)
    path = gt / "llm_extractions.jsonl"
    extractions = [json.loads(line) for line in path.read_text().splitlines()]
    extractions[3]["entities"][0]["value"] = "Python"
    checksum = _write_jsonl(path, extractions)
    manifest_path = gt / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"]["llm_extractions.jsonl"] = checksum
    manifest_path.write_text(json.dumps(manifest))
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    checksum = _write_jsonl(
        corpus / "resumes.jsonl",
        [
            {
                "id": "train1",
                "text_sha256": "rh1",
                "attributes": {"skills": ["Python"]},
            },
            {
                "id": "train2",
                "text_sha256": "other",
                "attributes": {"skills": ["Python"]},
            },
        ],
    )
    (corpus / "manifest.json").write_text(
        json.dumps({"sha256": {"resumes.jsonl": checksum}, "actual": {"resumes": 2}})
    )
    output = tmp_path / "output"
    result = run_experiment(
        {
            "gt_dir": str(gt),
            "corpus_dir": str(corpus),
            "output_dir": str(output),
            "cutoffs": [1, 2],
            "exclude_gt_exact_hashes": True,
        }
    )
    assert result["dataset"]["training_resumes"] == 1
    assert result["dataset"]["excluded_training_resumes"] == 1
    mined = json.loads((output / "negative_candidates.json").read_text())
    assert [(m["resume_id"], m["vacancy_id"], m["grade"]) for m in mined] == [
        ("r1", "v2", 0)
    ]
    assert result["variants"]["baseline"]["queries"][0]["ranked_ids"] == ["v1", "v2"]
    assert json.loads((output / "vocabulary.json").read_text())["documents"] == 1
