"""Evaluate locally pretrained resume token IDF against vacancy-corpus BM25."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from vrscore.benchmarking.evaluation import EvaluationCase
from vrscore.benchmarking.exact_skill_experiment import _macro, _validate_dataset
from vrscore.indexes import BM25SkillIndex
from vrscore.indexes.bm25_skills import DEFAULT_CONFIG
from vrscore.indexes.exact_skills import normalize_skill

_FIELDS = ("skills", "hardSkills", "softSkills")
_ROOT = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_vocabulary(
    rows: list[dict[str, Any]], index_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Count each normalized skill and token at most once per resume."""
    normalizer = BM25SkillIndex([], config=index_config)
    ids = set()
    skills: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()
    empty = 0
    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("Resume IDs must be unique nonempty strings")
        ids.add(identifier)
        attributes = row.get("attributes")
        if not isinstance(attributes, dict):
            raise ValueError("Resume attributes must be a mapping")
        found = set()
        for field in _FIELDS:
            values = attributes.get(field, [])
            if not isinstance(values, list) or any(
                not isinstance(v, str) for v in values
            ):
                raise ValueError(f"{field} must be a list of strings")
            if values:
                field_counts[field] += 1
            found.update(
                normalize_skill(value) for value in values if normalize_skill(value)
            )
        if not found:
            empty += 1
        skills.update(found)
        tokens.update({token for value in found for token in normalizer._tokens(value)})
    return {
        "documents": len(rows),
        "documents_without_skills": empty,
        "nonempty_fields": dict(field_counts),
        "skills": {
            skill: {"document_frequency": count}
            for skill, count in sorted(skills.items())
        },
        "tokens": dict(sorted(tokens.items())),
        "stopword_sources_sha256": normalizer.stopword_sources,
        "stopwords_sha256": hashlib.sha256(
            json.dumps(
                sorted(normalizer.stopwords), ensure_ascii=False, separators=(",", ":")
            ).encode()
        ).hexdigest(),
    }


def pretrained_index(
    vacancies: list[dict], vocabulary: dict, index_config: dict[str, Any] | None = None
) -> tuple[BM25SkillIndex, BM25SkillIndex]:
    """Replace only seen-token IDF; unseen tokens retain vacancy-corpus IDF."""
    base = BM25SkillIndex(vacancies, config=index_config)
    trained = BM25SkillIndex(vacancies, config=index_config)
    if not base.merge_skill_categories:
        raise ValueError("This experiment requires merged skill categories")
    if trained._model is not None:
        n = vocabulary["documents"]
        if n < 1:
            raise ValueError("Vocabulary must contain resumes")
        for token in trained._model.idf:
            frequency = vocabulary["tokens"].get(token)
            if frequency is not None:
                if not 1 <= frequency <= n:
                    raise ValueError("Invalid token document frequency")
                trained._model.idf[token] = math.log((n + 1) / frequency)
    return base, trained


def run_experiment(config: dict[str, Any] | DictConfig) -> dict[str, Any]:
    """Build the vocabulary and evaluate paired scores and mining locally."""
    settings = (
        OmegaConf.to_container(config, resolve=True)
        if isinstance(config, DictConfig)
        else dict(config)
    )
    if not isinstance(settings, dict):
        raise ValueError("Experiment settings must be a mapping")
    index_config = settings.get("index", {})
    corpus_dir = Path(settings["corpus_dir"])
    gt_dir = Path(settings["gt_dir"])
    output = Path(settings["output_dir"])
    corpus_manifest = json.loads(
        (corpus_dir / "manifest.json").read_text(encoding="utf-8")
    )
    corpus_path = corpus_dir / "resumes.jsonl"
    if _sha(corpus_path) != corpus_manifest["sha256"]["resumes.jsonl"]:
        raise ValueError("Resume corpus checksum mismatch")
    rows = [
        json.loads(line)
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != corpus_manifest["actual"]["resumes"]:
        raise ValueError("Resume corpus count mismatch")
    gt_manifest, data = _validate_dataset(gt_dir)
    resumes = data["resumes.jsonl"]
    gt_hashes = {r.get("text_sha256") for r in resumes}
    corpus_hashes = {r.get("text_sha256") for r in rows}
    overlap_ids = [r["id"] for r in resumes if r.get("text_sha256") in corpus_hashes]
    excluded = 0
    if settings.get("exclude_gt_exact_hashes", False):
        retained = [row for row in rows if row.get("text_sha256") not in gt_hashes]
        excluded = len(rows) - len(retained)
        rows = retained
    vocabulary = build_vocabulary(rows, index_config)
    vocabulary["source_sha256"] = _sha(corpus_path)
    vocabulary["excluded_exact_gt_hashes"] = excluded
    vacancies = data["vacancies.jsonl"]
    extraction = {x["id"]: x for x in data["llm_extractions.jsonl"]}
    labels = {
        (p["resume_id"], p["vacancy_id"]): p["relevance"] for p in data["pairs.jsonl"]
    }
    cases = [
        EvaluationCase(
            id=r["id"],
            input=r["id"],
            expected={v["id"]: labels[r["id"], v["id"]] for v in vacancies},
        )
        for r in resumes
    ]
    base, trained = pretrained_index(
        [extraction[v["id"]] for v in vacancies], vocabulary, index_config
    )
    indexes = {"baseline": base, "pretrained": trained}
    results = {}
    hits_by_variant = {}
    cutoffs = settings["cutoffs"]
    for name, index in indexes.items():
        hits_by_variant[name] = {
            r["id"]: index.search(extraction[r["id"]]).hits for r in resumes
        }
        rankings = {
            r: [h.vacancy_id for h in hits] for r, hits in hits_by_variant[name].items()
        }
        scores = {
            r: {h.vacancy_id: h.score for h in hits}
            for r, hits in hits_by_variant[name].items()
        }
        result, _ = _macro(cases, rankings, cutoffs, scores)
        result["no_hits"] = sum(not hits for hits in hits_by_variant[name].values())
        result["short_at_max_cutoff"] = sum(
            len(hits) < max(cutoffs) for hits in hits_by_variant[name].values()
        )
        result["retrieved_grade_ge_1"] = sum(
            labels[r, h.vacancy_id] >= 1
            for r, hits in hits_by_variant[name].items()
            for h in hits
        )
        result["top_k_grade_0"] = sum(
            labels[r, h.vacancy_id] == 0
            for r, hits in hits_by_variant[name].items()
            for h in hits[: max(cutoffs)]
        )
        results[name] = result
    mined = []
    for r, hits in hits_by_variant["pretrained"].items():
        baseline_ranks = {
            h.vacancy_id: rank
            for rank, h in enumerate(hits_by_variant["baseline"][r], 1)
        }
        for rank, hit in enumerate(hits, 1):
            if labels[r, hit.vacancy_id] != 0:
                continue
            mined.append(
                {
                    "resume_id": r,
                    "vacancy_id": hit.vacancy_id,
                    "grade": 0,
                    "rank": rank,
                    "baseline_rank": baseline_ranks.get(hit.vacancy_id),
                    "score": hit.score,
                    "matching_tokens": sorted({m.key for m in hit.matches}),
                    "label_status": "AI-judged candidate; human review required",
                }
            )
    mined.sort(key=lambda x: (x["rank"], x["resume_id"], x["vacancy_id"]))
    common = set(base._model.idf) & set(vocabulary["tokens"]) if base._model else set()
    result = {
        "configuration": settings,
        "dataset": {
            "training_resumes": len(rows),
            "excluded_training_resumes": excluded,
            "gt_resumes": len(resumes),
            "gt_vacancies": len(vacancies),
            "gt_pairs": len(labels),
            "gt_version": gt_manifest.get("version"),
            "exact_text_hash_overlap": len(overlap_ids),
            "overlapping_gt_ids": overlap_ids,
            "overlap_caveat": (
                "Different text rendering or source versions may conceal overlap."
            ),
        },
        "vocabulary": {
            "unique_skills": len(vocabulary["skills"]),
            "unique_tokens": len(vocabulary["tokens"]),
            "vacancy_tokens_seen": len(common),
            "vacancy_tokens_total": len(base._model.idf) if base._model else 0,
            "nonempty_fields": vocabulary["nonempty_fields"],
            "documents_without_skills": vocabulary["documents_without_skills"],
        },
        "provenance": {
            "python": sys.version.split()[0],
            "rank_bm25": importlib.metadata.version("rank-bm25"),
            "effective_index_config": OmegaConf.to_container(
                OmegaConf.merge(OmegaConf.load(DEFAULT_CONFIG), index_config),
                resolve=True,
            ),
            "index_config_sha256": _sha(DEFAULT_CONFIG),
            "experiment_config_sha256": _sha(
                _ROOT / "configs/vocabulary_ablation.yaml"
            ),
            "corpus_sha256": _sha(corpus_path),
            "gt_manifest_sha256": _sha(gt_dir / "manifest.json"),
            "inputs_sha256": {
                filename: _sha(gt_dir / filename)
                for filename in (
                    "resumes.jsonl",
                    "vacancies.jsonl",
                    "pairs.jsonl",
                    "requirements.jsonl",
                    "llm_extractions.jsonl",
                )
            },
            "implementation_sha256": {
                str(p.relative_to(_ROOT)): _sha(p)
                for p in (
                    Path(__file__).resolve(),
                    _ROOT / "vrscore/indexes/bm25_skills.py",
                    _ROOT / "vrscore/indexes/exact_skills.py",
                    _ROOT / "vrscore/benchmarking/evaluation.py",
                )
            },
            "stopwords": vocabulary["stopword_sources_sha256"],
            "stopwords_effective_sha256": vocabulary["stopwords_sha256"],
        },
        "variants": results,
        "mining": {
            "grade_zero_returned": len(mined),
            "top_k": max(cutoffs),
            "top_k_count": sum(x["rank"] <= max(cutoffs) for x in mined),
        },
        "errors": [],
        "skipped": [],
    }
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "vocabulary.json", vocabulary)
    result["provenance"]["vocabulary_sha256"] = _sha(output / "vocabulary.json")
    _write(output / "results.json", result)
    _write(output / "negative_candidates.json", mined)
    return result


@hydra.main(
    version_base=None, config_path="../../configs", config_name="vocabulary_ablation"
)
def main(config: DictConfig) -> None:
    """Run the configured vocabulary ablation and persist private artifacts."""
    result = run_experiment(config)
    print(
        json.dumps(
            {
                "dataset": result["dataset"],
                "vocabulary": result["vocabulary"],
                "means": {
                    name: value["mean_metrics"]
                    for name, value in result["variants"].items()
                },
                "mining": result["mining"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
