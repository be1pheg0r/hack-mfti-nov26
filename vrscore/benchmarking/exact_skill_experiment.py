"""Reproducible local exact-skill retrieval evaluation on private GT data."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from vrscore.benchmarking.evaluation import EvaluationCase, evaluate, ndcg_at_k
from vrscore.gates import evaluate_first_gate
from vrscore.indexes import ExactSkillIndex
from vrscore.indexes.exact_skills import DEFAULT_CONFIG as INDEX_DEFAULT_CONFIG

_REQUIRED_FILES = (
    "resumes.jsonl",
    "vacancies.jsonl",
    "pairs.jsonl",
    "requirements.jsonl",
    "llm_extractions.jsonl",
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read valid JSONL: {path.name}") from error


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_dataset(gt_dir: Path) -> tuple[dict, dict[str, list[dict]]]:
    manifest_path = gt_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = manifest.get("sha256", {})
    if not isinstance(hashes, dict):
        raise ValueError("Manifest is missing sha256 checksums")
    rows = {}
    for filename in _REQUIRED_FILES:
        path = gt_dir / filename
        if filename not in hashes or _digest(path) != hashes[filename]:
            raise ValueError(f"Manifest checksum mismatch for {filename}")
        rows[filename] = _jsonl(path)
    resumes = rows["resumes.jsonl"]
    vacancies = rows["vacancies.jsonl"]
    pairs = rows["pairs.jsonl"]
    extractions = rows["llm_extractions.jsonl"]
    resume_ids = [row.get("id") for row in resumes]
    vacancy_ids = [row.get("id") for row in vacancies]
    if len(set(resume_ids)) != len(resume_ids) or None in resume_ids:
        raise ValueError("Duplicate or missing resume IDs")
    if len(set(vacancy_ids)) != len(vacancy_ids) or None in vacancy_ids:
        raise ValueError("Duplicate or missing vacancy IDs")
    if not resumes or not vacancies:
        raise ValueError("Ground truth must contain resumes and vacancies")
    expected_pairs = {(r, v) for r in resume_ids for v in vacancy_ids}
    pair_keys = [(p.get("resume_id"), p.get("vacancy_id")) for p in pairs]
    if len(pair_keys) != len(set(pair_keys)) or set(pair_keys) != expected_pairs:
        raise ValueError(
            "Pair judgments must form the complete unique resume-vacancy join"
        )
    for record_name, records, id_list in (
        ("resume", [x for x in extractions if x.get("kind") == "resume"], resumes),
        ("vacancy", [x for x in extractions if x.get("kind") == "vacancy"], vacancies),
    ):
        by_id = {x.get("id"): x for x in records}
        if len(by_id) != len(records) or set(by_id) != {x["id"] for x in id_list}:
            raise ValueError(
                f"Extraction coverage does not exactly match {record_name} IDs"
            )
        for source in id_list:
            extraction = by_id[source["id"]]
            for field in ("document_id", "text_version", "text_sha256"):
                if extraction.get(field) != source.get(field):
                    raise ValueError(
                        "Extraction identity mismatch for "
                        f"{record_name} {source['id']} ({field})"
                    )
    expected = manifest.get("actual", {})
    counts = (
        ("resumes", len(resumes)),
        ("vacancies", len(vacancies)),
        ("pairs", len(pairs)),
        ("requirements", len(rows["requirements.jsonl"])),
    )
    for key, actual in counts:
        if key in expected and expected[key] != actual:
            raise ValueError(f"Manifest count mismatch for {key}")
    return manifest, rows


def _macro(
    cases: list[EvaluationCase],
    rankings: dict[str, list[str]],
    cutoffs: list[int],
    scores: dict[str, dict[str, float | None]] | None = None,
) -> tuple[dict, list[dict]]:
    metrics_by_query: dict[str, dict[str, float | None]] = {}

    def predictor(query_id):
        return rankings[query_id]

    def scorer(relevance, ranked_ids):
        output = {}
        for cutoff in cutoffs:
            value = ndcg_at_k(relevance, ranked_ids, cutoff)
            output[f"ndcg@{cutoff}"] = (
                0.0
                if value is None and any(relevance.values())
                else (value if value is not None else 0.0)
            )
        return output

    evaluated = evaluate("exact_skills", cases, predictor, scorer)
    if any(case.error_type for case in evaluated.cases):
        raise ValueError(
            "Internal evaluation failed for one or more complete-pool queries"
        )
    for case in evaluated.cases:
        metrics_by_query[case.id] = case.metrics
    # IDCG-zero cases have undefined nDCG and remain in output but not macro means.
    for case in cases:
        if not any(case.expected.values()):
            metrics_by_query[case.id] = {f"ndcg@{cutoff}": None for cutoff in cutoffs}
    means = {}
    for name in (f"ndcg@{cutoff}" for cutoff in cutoffs):
        values = [
            metrics[name]
            for metrics in metrics_by_query.values()
            if metrics[name] is not None
        ]
        means[name] = sum(values) / len(values) if values else None
    query_rows = []
    for case in cases:
        ranking = rankings[case.id]
        rank_map = {vacancy_id: rank for rank, vacancy_id in enumerate(ranking, 1)}
        query_rows.append(
            {
                "resume_id": case.id,
                "ranked_ids": ranking,
                "hits": [
                    {
                        "vacancy_id": vacancy_id,
                        "rank": rank_map[vacancy_id],
                        "score": scores[case.id][vacancy_id]
                        if scores is not None
                        else None,
                    }
                    for vacancy_id in ranking
                ],
                "hit_count": len(ranking),
                "no_hits": not ranking,
                "relevance_by_vacancy": dict(sorted(case.expected.items())),
                "metrics": metrics_by_query[case.id],
            }
        )
    return {"mean_metrics": means, "queries": query_rows}, query_rows


def run_experiment(config: dict[str, Any] | DictConfig) -> dict[str, Any]:
    """Validate all local inputs, run ranking variants, and return text-free scores."""
    settings = (
        OmegaConf.to_container(config, resolve=True)
        if isinstance(config, DictConfig)
        else dict(config)
    )
    gt_dir = Path(settings.get("gt_dir", "public_data/mfti_matching_gt")).resolve()
    manifest, data = _validate_dataset(gt_dir)
    resumes = data["resumes.jsonl"]
    vacancies = data["vacancies.jsonl"]
    extractions = data["llm_extractions.jsonl"]
    resume_gt = {row["id"]: row for row in resumes}
    vacancy_gt = {row["id"]: row for row in vacancies}
    extraction_by_id = {row["id"]: row for row in extractions}
    pair_rows = data["pairs.jsonl"]
    labels = {
        (row["resume_id"], row["vacancy_id"]): row["relevance"] for row in pair_rows
    }
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
        for value in labels.values()
    ):
        raise ValueError("Relevance labels must be nonnegative numbers")
    cuts = settings.get("cutoffs", [1, 2, 3, 4, 5])
    if (
        not cuts
        or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in cuts)
        or len(set(cuts)) != len(cuts)
    ):
        raise ValueError("cutoffs must be unique positive integers")
    index_config = settings.get("index", {})
    index = ExactSkillIndex(
        [extraction_by_id[v] for v in vacancy_gt], config=index_config
    )
    cases = [
        EvaluationCase(id=r, input=r, expected={v: labels[(r, v)] for v in vacancy_gt})
        for r in resume_gt
    ]
    exact_rankings = {}
    raw_hits = {}
    raw_scores = {}
    for resume_id in resume_gt:
        hits = index.search(extraction_by_id[resume_id]).hits
        raw_hits[resume_id] = [hit.vacancy_id for hit in hits]
        raw_scores[resume_id] = {hit.vacancy_id: hit.score for hit in hits}
        exact_rankings[resume_id] = list(raw_hits[resume_id])
    exact, _ = _macro(cases, exact_rankings, cuts, raw_scores)
    variants = {"exact": exact}

    if settings.get("run_gate", True):
        gate_unknown = settings.get("gate_unknown_policy", "include")
        if gate_unknown not in {"include", "exclude"}:
            raise ValueError("gate_unknown_policy must be include or exclude")
        gate_cfg = settings.get("gate_config")
        gate_decisions = {}
        gate_rankings = {}
        for resume_id, resume in resume_gt.items():
            passing = set()
            outcomes = {}
            for vacancy_id, vacancy in vacancy_gt.items():
                result = evaluate_first_gate(
                    extraction_by_id[vacancy_id], extraction_by_id[resume_id], gate_cfg
                )
                status = result["status"]
                outcomes[vacancy_id] = status
                if status == "pass" or (
                    status == "unknown" and gate_unknown == "include"
                ):
                    passing.add(vacancy_id)
            gate_decisions[resume_id] = outcomes
            gate_rankings[resume_id] = [v for v in raw_hits[resume_id] if v in passing]
        gated_scores = {
            r: {v: raw_scores[r][v] for v in gate_rankings[r]} for r in resume_gt
        }
        gated, _ = _macro(cases, gate_rankings, cuts, gated_scores)
        gated["gate_policy"] = {
            "unknown": gate_unknown,
            "fail": "exclude",
            "pass": "include",
        }
        gated["gate_status_by_vacancy"] = gate_decisions
        variants["gate1_exact"] = gated

    if settings.get("include_full_list_fallback", False):
        fallback = {
            r: ranking + sorted(set(vacancy_gt) - set(ranking))
            for r, ranking in exact_rankings.items()
        }
        fallback_scores = {
            r: {v: raw_scores[r].get(v, 0.0) for v in fallback[r]} for r in resume_gt
        }
        variants["exact_then_vacancy_id"] = _macro(
            cases, fallback, cuts, fallback_scores
        )[0]

    return {
        "experiment": {
            "name": settings.get("run_name", "exact-skills-v1"),
            "metric": "internal ndcg_at_k; gain=2^relevance-1",
            "cutoffs": cuts,
            "configuration": settings,
        },
        "dataset": {
            "path": str(gt_dir),
            "manifest_version": manifest.get("version"),
            "resumes_total": len(resumes),
            "vacancies_per_resume": len(vacancies),
            "pairs_total": len(pair_rows),
            "unique_pairs_total": len(labels),
            "extractions_total": len(extractions),
        },
        "errors": [],
        "skipped": [],
        "provenance": {
            "manifest_sha256": _digest(gt_dir / "manifest.json"),
            "input_sha256": {name: _digest(gt_dir / name) for name in _REQUIRED_FILES},
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "code_sha256": _digest(Path(__file__)),
            "implementation_sha256": {
                name: _digest(Path(__file__).resolve().parents[1] / name)
                for name in (
                    "benchmarking/contracts.py",
                    "benchmarking/evaluation.py",
                    "gates.py",
                    "indexes/base.py",
                    "indexes/exact_skills.py",
                )
            },
            "environment_files_sha256": {
                "pyproject.toml": _digest(
                    Path(__file__).resolve().parents[2] / "pyproject.toml"
                ),
                "uv.lock": _digest(Path(__file__).resolve().parents[2] / "uv.lock"),
            },
            "config_files_sha256": {
                "experiment": _digest(
                    Path(__file__).resolve().parents[2]
                    / "configs"
                    / "exact_skills_experiment.yaml"
                ),
                "exact_skill_index": _digest(INDEX_DEFAULT_CONFIG),
                "first_gate": _digest(
                    Path(__file__).resolve().parents[2] / "configs" / "first_gate.yaml"
                ),
            },
            "index": {
                "class": "vrscore.indexes.ExactSkillIndex",
                "config": OmegaConf.to_container(
                    OmegaConf.merge(
                        OmegaConf.load(INDEX_DEFAULT_CONFIG),
                        OmegaConf.create(index_config),
                    ),
                    resolve=True,
                ),
            },
        },
        "variants": variants,
    }


def _write_artifacts(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "results.json"
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


try:
    import hydra
except ImportError:  # pragma: no cover - exercised only in incomplete environments
    hydra = None

if hydra is not None:

    @hydra.main(
        version_base=None,
        config_path="../../configs",
        config_name="exact_skills_experiment",
    )
    def main(config: DictConfig) -> None:
        result = run_experiment(config)
        _write_artifacts(result, Path(config.output_dir))
        print(
            json.dumps(
                {
                    "output_dir": str(Path(config.output_dir).resolve()),
                    "dataset": result["dataset"],
                    "means": {
                        name: value["mean_metrics"]
                        for name, value in result["variants"].items()
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
else:  # pragma: no cover

    def main() -> None:
        raise RuntimeError("hydra-core is required to run this experiment CLI")


if __name__ == "__main__":
    main()
