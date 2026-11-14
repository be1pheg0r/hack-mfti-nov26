"""Paired BM25 versus EmbeddingGemma reranking on the complete local GT pool."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from vrscore.benchmarking.evaluation import ndcg_at_k
from vrscore.benchmarking.exact_skill_experiment import _validate_dataset
from vrscore.embedding_client import EmbeddingClient
from vrscore.gates import evaluate_first_gate
from vrscore.indexes import BM25SkillIndex
from vrscore.indexes.base import SearchResults
from vrscore.indexes.embedding_reranker import EmbeddingShortlistReranker
from vrscore.vector_store import MilvusVectorStore

ROOT = Path(__file__).resolve().parents[2]


def compare_rankings(baseline, reranked, labels, cutoffs) -> dict[str, Any]:
    """Compare two permutations using the same full-pool ideal ranking."""
    left = [hit.vacancy_id for hit in baseline.hits]
    right = [hit.vacancy_id for hit in reranked.hits]
    if (
        len(left) != len(set(left))
        or set(left) != set(right)
        or len(left) != len(right)
    ):
        raise ValueError("Both rankings must contain the exact same unique shortlist")
    result = {
        name: {f"ndcg@{k}": ndcg_at_k(labels, ranking, k) for k in cutoffs}
        for name, ranking in (("bm25", left), ("embedding", right))
    }
    for threshold in (1, 2):
        positive = {key for key, grade in labels.items() if grade >= threshold}
        result[f"candidate_recall_grade_{threshold}"] = (
            len(positive.intersection(left)) / len(positive) if positive else None
        )
    return result


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(config: DictConfig) -> dict[str, Any]:
    """Run a paired experiment; all source text stays on the local machine."""
    settings = OmegaConf.to_container(config, resolve=True)
    gt_dir = Path(settings["gt_dir"]).resolve()
    manifest, data = _validate_dataset(gt_dir)
    resumes = data["resumes.jsonl"]
    vacancies = data["vacancies.jsonl"]
    ex = {row["id"]: row for row in data["llm_extractions.jsonl"]}
    for row in resumes + vacancies:
        if hashlib.sha256(row["text"].encode()).hexdigest() != row["text_sha256"]:
            raise ValueError(f"Text checksum mismatch for {row['id']}")
    labels = {
        r["id"]: {
            p["vacancy_id"]: p["relevance"]
            for p in data["pairs.jsonl"]
            if p["resume_id"] == r["id"]
        }
        for r in resumes
    }
    if any(
        isinstance(x, bool) or not isinstance(x, int) or x < 0
        for row in labels.values()
        for x in row.values()
    ):
        raise ValueError("Invalid relevance grade")
    texts = {v["text"]: v["id"] for v in vacancies}
    if len(texts) != len(vacancies):
        raise ValueError("Ambiguous identical vacancy texts")
    client = EmbeddingClient(settings["server"], timeout=settings["timeout_seconds"])
    started = time.perf_counter()
    doc_vectors = {}
    for index, v in enumerate(vacancies, 1):
        doc_vectors[v["id"]] = client.encode_documents(
            [v["text"]], titles=[v["title"]]
        )[0]
        print(f"Encoded vacancies: {index}/{len(vacancies)}", flush=True)
    docs_seconds = time.perf_counter() - started
    started = time.perf_counter()
    query_vectors = {}
    for index, r in enumerate(resumes, 1):
        query_vectors[r["id"]] = client.encode_query(r["text"])
        print(f"Encoded resumes: {index}/{len(resumes)}", flush=True)
    query_seconds = time.perf_counter() - started
    all_vectors = list(doc_vectors.values()) + list(query_vectors.values())
    if any(
        len(v) != 128
        or not all(math.isfinite(x) for x in v)
        or abs(math.sqrt(sum(x * x for x in v)) - 1) > 1e-4
        for v in all_vectors
    ):
        raise ValueError("Invalid model vectors: expected unit-normalized finite 128d")

    vector_settings = OmegaConf.to_container(
        OmegaConf.load(ROOT / "configs/vector_store.yaml"), resolve=True
    )
    vector_settings.update(settings.get("vector_store") or {})
    token_value = vector_settings.get("token_file")
    token_file = (
        ROOT / token_value
        if token_value and not Path(token_value).is_absolute()
        else token_value
    )
    results = {}
    gate_counts = Counter()
    filtered_label_counts = Counter()
    with MilvusVectorStore(
        uri=settings.get("vector_store_uri", vector_settings["uri"]),
        collection_name=settings.get("vector_collection", "embedding_ablation_128d"),
        token_file=token_file,
        dimension=128,
    ) as store:
        reranker = EmbeddingShortlistReranker(
            store, encode=lambda text: doc_vectors[texts[text]]
        )
        for gate_on in settings["gate_modes"]:
            mode = "gate_include_unknown" if gate_on else "no_gate"
            results[mode] = {str(m): [] for m in settings["shortlist_sizes"]}
            for r in resumes:
                rid = r["id"]
                statuses = (
                    {
                        v["id"]: evaluate_first_gate(ex[v["id"]], ex[rid])["status"]
                        for v in vacancies
                    }
                    if gate_on
                    else {}
                )
                if gate_on:
                    gate_counts.update(statuses.values())
                    for vid, status in statuses.items():
                        if status == "fail":
                            filtered_label_counts[str(labels[rid][vid])] += 1
                eligible = [
                    v for v in vacancies if not gate_on or statuses[v["id"]] != "fail"
                ]
                index = BM25SkillIndex([ex[v["id"]] for v in eligible])
                hits = index.search(ex[rid])
                for m in settings["shortlist_sizes"]:
                    baseline = SearchResults(
                        hits.index_name, hits.query_id, hits.hits[:m]
                    )
                    ranked = reranker.rerank(
                        baseline,
                        query_vectors[rid],
                        {v["id"]: v["text"] for v in eligible},
                    )
                    metrics = compare_rankings(
                        baseline, ranked, labels[rid], settings["cutoffs"]
                    )
                    metrics.update(
                        {
                            "resume_id": rid,
                            "eligible_count": len(eligible),
                            "bm25_hit_count_before_cutoff": len(hits.hits),
                            "shortlist_count": len(baseline.hits),
                            "bm25_ids": [h.vacancy_id for h in baseline.hits],
                            "embedding_ids": [h.vacancy_id for h in ranked.hits],
                            "scores": {h.vacancy_id: h.score for h in ranked.hits},
                            "relevance": labels[rid],
                        }
                    )
                    results[mode][str(m)].append(metrics)
    return {
        "config": settings,
        "dataset": {
            "version": manifest["version"],
            "resumes": len(resumes),
            "vacancies": len(vacancies),
            "pairs": len(data["pairs.jsonl"]),
            "grades": dict(Counter(str(p["relevance"]) for p in data["pairs.jsonl"])),
        },
        "gate_status_counts": dict(gate_counts),
        "gate_excluded_by_grade": dict(filtered_label_counts),
        "timing": {
            "vacancy_embedding_seconds": docs_seconds,
            "resume_embedding_seconds": query_seconds,
            "scope": "one sequential local HTTP request per text; no model load",
        },
        "provenance": {
            "dataset_sha256": {p.name: digest(p) for p in gt_dir.glob("*.json*")},
            "code_sha256": {
                str(p.relative_to(ROOT)): digest(p)
                for p in [
                    Path(__file__),
                    ROOT / "vrscore/gates.py",
                    ROOT / "vrscore/indexes/bm25_skills.py",
                    ROOT / "vrscore/indexes/embedding_reranker.py",
                    ROOT / "vrscore/vector_store.py",
                    ROOT / "vrscore/embeddinggemma.py",
                    ROOT / "configs/bm25_skills_index.yaml",
                    ROOT / "configs/first_gate.yaml",
                    ROOT / "configs/stopwords/ru.txt",
                    ROOT / "configs/embeddinggemma.yaml",
                ]
            },
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                p: importlib.metadata.version(p)
                for p in [
                    "rank-bm25",
                    "pymilvus",
                    "torch",
                    "transformers",
                    "sentence-transformers",
                ]
            },
            "model": "google/embeddinggemma-2",
            "model_revision": "f9972ac22e696a105c482202646979e33e4b5a1e",
        },
        "results": results,
    }


@hydra.main(
    version_base=None, config_path="../../configs", config_name="embedding_ablation"
)
def main(config: DictConfig) -> None:
    """Run experiment and save a private, text-free result artifact."""
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = run(config)
    (out / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Results: {out / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
