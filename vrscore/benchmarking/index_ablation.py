"""Reproducible, text-free lexical-index and RRF ablation experiment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from vrscore.benchmarking import BenchmarkSpec, run_benchmark
from vrscore.benchmarking.evaluation import EvaluationCase
from vrscore.benchmarking.exact_skill_experiment import _macro, _validate_dataset
from vrscore.gates import evaluate_first_gate
from vrscore.indexes import BM25SkillIndex, ExactSkillIndex, reciprocal_rank_fusion

_ROOT = Path(__file__).resolve().parents[2]
_INDEX_MODULES = {
    "exact": "indexes/exact_skills.py",
    "bm25": "indexes/bm25_skills.py",
    "tantivy": "indexes/tantivy_skills.py",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _settings(config: dict[str, Any] | DictConfig) -> dict[str, Any]:
    result = (
        OmegaConf.to_container(config, resolve=True)
        if isinstance(config, DictConfig)
        else dict(config)
    )
    if not isinstance(result, dict):
        raise ValueError("Experiment config must be a mapping")
    return result


def run_experiment(config: dict[str, Any] | DictConfig) -> dict[str, Any]:
    """Evaluate every configured index and RRF combination on the same pool."""
    from vrscore.indexes import TantivyExactSkillIndex

    settings = _settings(config)
    gt_dir = Path(settings.get("gt_dir", "public_data/mfti_matching_gt")).resolve()
    manifest, data = _validate_dataset(gt_dir)
    resumes = data["resumes.jsonl"]
    vacancies = data["vacancies.jsonl"]
    extractions = data["llm_extractions.jsonl"]
    resume_by_id = {row["id"]: row for row in resumes}
    vacancy_by_id = {row["id"]: row for row in vacancies}
    extraction_by_id = {row["id"]: row for row in extractions}
    labels = {
        (row["resume_id"], row["vacancy_id"]): row["relevance"]
        for row in data["pairs.jsonl"]
    }
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
        for value in labels.values()
    ):
        raise ValueError("Relevance labels must be nonnegative numbers")
    cutoffs = settings.get("cutoffs", [1, 2, 3, 4, 5])
    if (
        not cutoffs
        or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in cutoffs)
        or len(set(cutoffs)) != len(cutoffs)
    ):
        raise ValueError("cutoffs must be unique positive integers")

    specs = settings.get("indexes")
    if not isinstance(specs, list) or not specs:
        raise ValueError("indexes must be a non-empty list")
    classes = {
        "exact": ExactSkillIndex,
        "bm25": BM25SkillIndex,
        "tantivy": TantivyExactSkillIndex,
    }
    indexes = {}
    build_seconds = {}
    timing_settings = settings.get("timing", {})
    timing_enabled = timing_settings.get("enabled", False)
    if not isinstance(timing_enabled, bool):
        raise ValueError("timing.enabled must be a boolean")
    timing_spec = (
        BenchmarkSpec(
            name="index-search",
            iterations=timing_settings.get("iterations", 3),
            warmup_iterations=timing_settings.get("warmup_iterations", 1),
        )
        if timing_enabled
        else None
    )
    for spec in specs:
        if (
            not isinstance(spec, dict)
            or not isinstance(spec.get("name"), str)
            or not spec["name"]
        ):
            raise ValueError("Each index requires a non-empty name")
        name, kind = spec["name"], spec.get("type")
        if name in indexes:
            raise ValueError(f"Duplicate index name: {name}")
        if kind not in classes:
            raise ValueError(f"Unknown index type: {kind!r}")
        start = time.perf_counter()
        instance = classes[kind](
            [extraction_by_id[v] for v in vacancy_by_id], config=spec.get("config", {})
        )
        if timing_enabled:
            build_seconds[name] = time.perf_counter() - start
        indexes[name] = {
            "instance": instance,
            "type": kind,
            "config": spec.get("config", {}),
        }

    cases = [
        EvaluationCase(
            id=r, input=r, expected={v: labels[(r, v)] for v in vacancy_by_id}
        )
        for r in resume_by_id
    ]
    raw_rankings: dict[str, dict[str, list[str]]] = {}
    raw_scores: dict[str, dict[str, dict[str, float]]] = {}
    search_results = {}
    query_timings: dict[str, dict[str, Any]] = {}
    for name, spec in indexes.items():
        raw_rankings[name], raw_scores[name], search_results[name] = {}, {}, {}
        if timing_spec:
            query_timings[name] = {}
        for resume_id in resume_by_id:
            result = spec["instance"].search(extraction_by_id[resume_id])
            if timing_spec:
                measure = run_benchmark(
                    timing_spec,
                    lambda: spec["instance"].search(extraction_by_id[resume_id]),
                )
                if measure.failed_iterations:
                    raise ValueError(f"Benchmark query failed for {name}")
                query_timings[name][resume_id] = measure.model_dump(mode="json")
            search_results[name][resume_id] = result
            hits = result.hits
            ids = [hit.vacancy_id for hit in hits]
            if len(ids) != len(set(ids)) or not set(ids).issubset(vacancy_by_id):
                raise ValueError(f"Index {name!r} returned invalid vacancy IDs")
            raw_rankings[name][resume_id] = ids
            raw_scores[name][resume_id] = {
                hit.vacancy_id: float(hit.score) for hit in hits
            }

    rankings = {name: dict(values) for name, values in raw_rankings.items()}
    scores = {name: dict(values) for name, values in raw_scores.items()}
    rrf_config = settings.get("rrf", {})
    combinations = (
        rrf_config.get("combinations", []) if rrf_config.get("enabled", True) else []
    )
    rrf_k = rrf_config.get("k", 60)
    rrf_timings: dict[str, dict[str, Any]] = {}
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 0:
        raise ValueError("rrf.k must be a non-negative integer")
    for combination in combinations:
        if (
            not isinstance(combination, list)
            or len(combination) < 2
            or len(set(combination)) != len(combination)
        ):
            raise ValueError(
                "Each RRF combination must contain at least two unique index names"
            )
        if any(name not in indexes for name in combination):
            raise ValueError("RRF combination references an unknown index")
        variant = "rrf_" + "_".join(combination)
        if variant in rankings:
            raise ValueError(f"Duplicate variant name: {variant}")
        rankings[variant], scores[variant] = {}, {}
        if timing_spec:
            rrf_timings[variant] = {}
        for resume_id in resume_by_id:
            channels = [search_results[name][resume_id] for name in combination]
            fused = reciprocal_rank_fusion(channels, k=rrf_k)
            if timing_spec:
                measure = run_benchmark(
                    timing_spec, lambda: reciprocal_rank_fusion(channels, k=rrf_k)
                )
                if measure.failed_iterations:
                    raise ValueError(f"Benchmark RRF failed for {variant}")
                rrf_timings[variant][resume_id] = measure.model_dump(mode="json")
            rankings[variant][resume_id] = [hit.vacancy_id for hit in fused.hits]
            scores[variant][resume_id] = {
                hit.vacancy_id: float(hit.score) for hit in fused.hits
            }

    variants = {
        name: _macro(cases, rankings[name], cutoffs, scores[name])[0]
        for name in rankings
    }
    gate = settings.get("gate", {})
    if gate.get("enabled", False):
        unknown_policy = gate.get("unknown_policy", "include")
        if unknown_policy not in {"include", "exclude"}:
            raise ValueError("gate.unknown_policy must be include or exclude")
        decisions = {}
        for resume_id in resume_by_id:
            outcomes = {}
            for vacancy_id in vacancy_by_id:
                outcomes[vacancy_id] = evaluate_first_gate(
                    extraction_by_id[vacancy_id],
                    extraction_by_id[resume_id],
                    gate.get("config"),
                )["status"]
            decisions[resume_id] = outcomes
        passing = {
            resume_id: {
                vacancy_id
                for vacancy_id, status in outcomes.items()
                if status == "pass"
                or (status == "unknown" and unknown_policy == "include")
            }
            for resume_id, outcomes in decisions.items()
        }
        for name in rankings:
            gated = {
                r: [v for v in rankings[name][r] if v in passing[r]]
                for r in resume_by_id
            }
            gated_scores = {
                r: {v: scores[name][r][v] for v in gated[r]} for r in resume_by_id
            }
            key = f"{name}__gate_{unknown_policy}"
            value = _macro(cases, gated, cutoffs, gated_scores)[0]
            value["gate_policy"] = {
                "unknown": unknown_policy,
                "fail": "exclude",
                "pass": "include",
                "filter_stage": "after_search_and_fusion",
            }
            value["gate_status_by_vacancy"] = decisions
            variants[key] = value

    library_names = {
        "hydra-core": "hydra-core",
        "rank-bm25": "rank-bm25",
        "tantivy": "tantivy",
    }
    implementation_files = [
        "benchmarking/index_ablation.py",
        "benchmarking/exact_skill_experiment.py",
        "benchmarking/evaluation.py",
        "gates.py",
        "indexes/base.py",
        *_INDEX_MODULES.values(),
    ]
    return {
        "experiment": {
            "name": settings.get("run_name", "index-ablation-v1"),
            "metric": "internal ndcg_at_k; gain=2^relevance-1",
            "cutoffs": cutoffs,
            "configuration": settings,
            "scope": settings.get("scope", {}),
        },
        "dataset": {
            "manifest_version": manifest.get("version"),
            "resumes_total": len(resumes),
            "vacancies_per_resume": len(vacancies),
            "pairs_total": len(data["pairs.jsonl"]),
            "unique_pairs_total": len(labels),
            "extractions_total": len(extractions),
        },
        "errors": [],
        "skipped": [],
        "timing": (
            {
                "measurement_scope": (
                    "index.search; RRF measured separately on cached results"
                ),
                "units": "seconds",
                "order": "sequential by configured index and resume",
                "warmup_iterations": timing_spec.warmup_iterations,
                "iterations": timing_spec.iterations,
                "build_seconds": build_seconds,
                "query_samples": query_timings,
                "rrf_samples": rrf_timings,
            }
            if timing_spec
            else None
        ),
        "provenance": {
            "manifest_sha256": _digest(gt_dir / "manifest.json"),
            "input_sha256": {
                name: _digest(gt_dir / name)
                for name in (
                    "resumes.jsonl",
                    "vacancies.jsonl",
                    "pairs.jsonl",
                    "requirements.jsonl",
                    "llm_extractions.jsonl",
                )
            },
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "implementation_sha256": {
                name: _digest(_ROOT / "vrscore" / name) for name in implementation_files
            },
            "environment_files_sha256": {
                name: _digest(_ROOT / name) for name in ("pyproject.toml", "uv.lock")
            },
            "config_files_sha256": {
                name: _digest(_ROOT / "configs" / name)
                for name in (
                    "index_ablation.yaml",
                    "exact_skills_index.yaml",
                    "bm25_skills_index.yaml",
                    "tantivy_exact_skills_index.yaml",
                    "first_gate.yaml",
                )
            },
            "libraries": {
                name: _version(distribution)
                for name, distribution in library_names.items()
            },
            "indexes": {
                name: {
                    "type": spec["type"],
                    "class": (
                        f"{type(spec['instance']).__module__}."
                        f"{type(spec['instance']).__name__}"
                    ),
                    "config": spec["config"],
                }
                for name, spec in indexes.items()
            },
            "stopwords": {
                name: {
                    "count": len(spec["instance"].stopwords),
                    "source_files_sha256": spec["instance"].stopword_sources,
                    "sha256": hashlib.sha256(
                        json.dumps(
                            sorted(spec["instance"].stopwords),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                }
                for name, spec in indexes.items()
                if isinstance(spec["instance"], BM25SkillIndex)
            },
            "rrf": {"k": rrf_k, "combinations": combinations},
        },
        "variants": variants,
    }


def _write_artifacts(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


try:
    import hydra
except ImportError:  # pragma: no cover
    hydra = None

if hydra is not None:

    @hydra.main(
        version_base=None, config_path="../../configs", config_name="index_ablation"
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
                indent=2,
            )
        )
else:  # pragma: no cover

    def main() -> None:
        raise RuntimeError("hydra-core is required to run this experiment CLI")


if __name__ == "__main__":
    main()
