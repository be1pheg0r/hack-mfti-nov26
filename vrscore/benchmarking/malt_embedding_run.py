"""Full, resumable local EmbeddingGemma-vs-BM25 ablation on transposed Malt."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from vrscore.benchmarking.malt_embedding_ablation import rerank_shortlist
from vrscore.vector_cache_migration import make_milvus_client

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / ".data/malt_ablation"
PREVIOUS = ROOT / ".experiments/malt-full-ablation"
OUT = ROOT / ".experiments/malt-embeddinggemma-ablation-20261114"
MODEL = (
    "google/embeddinggemma-2@f9972ac22e696a105c482202646979e33e4b5a1e"
    ":128:fp32:SearchQuery:Document"
)
SIZE = (10,)
CUTS = (1, 3, 5, 10, 20)
METHODS = ("bm25_stop_ru",)
DEFAULT_MILVUS_MALT_COLLECTION = "malt_embedding_cache_v1_4036fe3811436fc6"


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load():
    manifest = json.loads((ROOT / ".data/malt/manifest.json").read_text())
    source = {x["path"]: x for x in manifest["files"]}
    paths = [
        DATA / (n + ".parquet")
        for n in ("freelancer-corpus", "brief-queries", "brief-qrels")
    ]
    for path in paths:
        if digest(path) != source[path.name]["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {path.name}")
    profiles, briefs, judgments = [pq.read_table(path).to_pylist() for path in paths]
    prof = {x["id"]: x["text"] for x in profiles}
    brief = {x["id"]: x["text"] for x in briefs}
    if (
        len(prof) != len(profiles)
        or len(brief) != len(briefs)
        or len(prof) != 4019
        or len(brief) != 200
    ):
        raise ValueError("Unexpected corpus size or duplicate IDs")
    labels = {pid: {} for pid in prof}
    for row in judgments:
        pid, bid, grade = row["corpus-id"], row["query-id"], row["score"]
        if (
            pid not in labels
            or bid not in brief
            or bid in labels[pid]
            or grade not in (-2, -1, 0, 1, 2, 3)
        ):
            raise ValueError("Malformed, duplicate, or unexpected qrel")
        labels[pid][bid] = grade
    if len(judgments) != 803800 or any(len(x) != len(brief) for x in labels.values()):
        raise ValueError("Qrel matrix must be complete")
    ranking = {}
    with gzip.open(PREVIOUS / "rankings.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            pid = row["profile_id"]
            if pid in ranking or pid not in prof:
                raise ValueError("Duplicate or unknown profile ranking")
            ranking[pid] = {k: row["rankings"][k] for k in METHODS}
            for ids in ranking[pid].values():
                if len(ids) != len(set(ids)) or not set(ids).issubset(brief):
                    raise ValueError("Invalid baseline IDs")
    if set(ranking) != set(prof):
        raise ValueError("Baseline profile coverage mismatch")
    return prof, brief, labels, ranking, {p.name: digest(p) for p in paths}


def metrics(ids, grade, k):
    def gain(x):
        return 2 ** max(x, 0) - 1

    actual = sum(gain(grade[x]) / math.log2(i + 2) for i, x in enumerate(ids[:k]))
    ideal = sum(
        gain(x) / math.log2(i + 2)
        for i, x in enumerate(sorted(grade.values(), reverse=True)[:k])
    )
    return actual / ideal if ideal else None


def evaluate(
    profiles,
    briefs,
    labels,
    ranking,
    hashes,
    conn,
    *,
    milvus_client=None,
    milvus_collection=None,
):
    doc_ids = sorted(
        {bid for pid in profiles for bid in ranking[pid]["bm25_stop_ru"][:10]}
    )
    pos = {key: i for i, key in enumerate(doc_ids)}

    milvus_vectors = None
    if milvus_client is not None:
        if not milvus_collection:
            raise ValueError("Milvus collection is required when using Milvus cache")
        iterator = milvus_client.query_iterator(
            collection_name=milvus_collection,
            batch_size=1000,
            limit=-1,
            filter="",
            output_fields=["role", "record_id", "text_sha", "model_identity", "vector"],
        )
        milvus_vectors = {}
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                for row in batch:
                    source = profiles if row["role"] == "query" else briefs
                    if (
                        row["role"] not in {"query", "document"}
                        or row["record_id"] not in source
                        or row.get("model_identity") != MODEL
                        or row.get("text_sha")
                        != hashlib.sha256(source[row["record_id"]].encode()).hexdigest()
                    ):
                        raise ValueError(
                            "Milvus cache model or source content mismatch"
                        )
                    milvus_vectors[(row["role"], row["record_id"])] = np.asarray(
                        row["vector"], dtype="<f4"
                    )
        finally:
            close = getattr(iterator, "close", None)
            if close:
                close()

    def v(role, key):
        if milvus_vectors is not None:
            value = milvus_vectors.get((role, key))
            if value is None:
                raise ValueError(f"Missing Milvus embedding: {role} {key}")
            return value
        rec = conn.execute(
            "SELECT vector FROM embeddings WHERE role=? AND id=?", (role, key)
        ).fetchone()
        if not rec:
            raise ValueError("Missing embedding")
        return np.frombuffer(rec[0], dtype="<f4").copy()

    matrix = np.stack([v("document", x) for x in doc_ids])
    results = {f"{method}:m{m}": [] for method in METHODS for m in SIZE}

    with gzip.open(OUT / "per-profile.jsonl.gz", "wt", encoding="utf-8") as f:
        for number, pid in enumerate(profiles, 1):
            query = v("query", pid)
            row = {"profile_id": pid, "variants": {}}
            for method in METHODS:
                rank = ranking[pid][method]
                grades = labels[pid]
                for m in SIZE:
                    candidate = rank[:m]
                    scored = rerank_shortlist(candidate, query, matrix, pos)
                    after = [x[0] for x in scored]
                    a = {str(k): metrics(candidate, grades, k) for k in CUTS}
                    b = {str(k): metrics(after, grades, k) for k in CUTS}
                    key = f"{method}:m{m}"
                    data = {
                        "bm25": a,
                        "embedding": b,
                        "candidate_count": len(candidate),
                        "candidate_positive": sum(grades[x] > 0 for x in candidate),
                        "candidate_total_positive": sum(x > 0 for x in grades.values()),
                        "top5_nonpositive_bm25": sum(
                            grades[x] <= 0 for x in candidate[:5]
                        ),
                        "top5_nonpositive_embedding": sum(
                            grades[x] <= 0 for x in after[:5]
                        ),
                        "bm25_ids": candidate,
                        "embedding_ids": after,
                    }
                    results[key].append(data)
                    row["variants"][key] = data
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
            if number % 500 == 0:
                print("evaluated", number, flush=True)
    summary = {}
    rng = np.random.default_rng(42)
    for key, rows in results.items():
        eligible = [r for r in rows if r["bm25"]["5"] is not None]
        diffs = np.array([r["embedding"]["5"] - r["bm25"]["5"] for r in eligible])
        boot = np.array(
            [
                np.mean(rng.choice(diffs, size=len(diffs), replace=True))
                for _ in range(2000)
            ]
        )
        summary[key] = {
            "count": len(rows),
            "positive_ideal_count": len(eligible),
            "macro": {
                kind: {
                    str(k): float(
                        np.mean(
                            [
                                r[kind][str(k)]
                                for r in rows
                                if r[kind][str(k)] is not None
                            ]
                        )
                    )
                    for k in CUTS
                }
                for kind in ("bm25", "embedding")
            },
            "paired_ndcg5": {
                "delta": float(np.mean(diffs)),
                "better": int(np.sum(diffs > 1e-12)),
                "equal": int(np.sum(np.abs(diffs) <= 1e-12)),
                "worse": int(np.sum(diffs < -1e-12)),
                "bootstrap_95_percentile": np.percentile(boot, [2.5, 97.5]).tolist(),
                "bootstrap_seed": 42,
                "bootstrap_draws": 2000,
            },
            "candidate_positive": sum(r["candidate_positive"] for r in rows),
            "total_positive": sum(r["candidate_total_positive"] for r in rows),
            "empty": sum(r["candidate_count"] == 0 for r in rows),
            "short5": sum(r["candidate_count"] < 5 for r in rows),
            "mean_candidates": float(np.mean([r["candidate_count"] for r in rows])),
            "top5_nonpositive_bm25": sum(r["top5_nonpositive_bm25"] for r in rows),
            "top5_nonpositive_embedding": sum(
                r["top5_nonpositive_embedding"] for r in rows
            ),
            "top5_count": sum(min(5, r["candidate_count"]) for r in rows),
        }
    output = {
        "dataset": {
            "profiles": len(profiles),
            "briefs": len(briefs),
            "pairs": sum(len(v) for v in labels.values()),
            "positive_ideal": sum(
                any(x > 0 for x in row.values()) for row in labels.values()
            ),
            "sha256": hashes,
        },
        "protocol": {
            "model": MODEL,
            "no_gate": True,
            "profile_input": "full text, query prompt",
            "document_input": "full brief, title none",
            "lexical_input": "profile skills vs complete brief",
            "m": SIZE,
            "cutoffs": CUTS,
            "gain": "2**max(score,0)-1",
            "idcg_pool": 200,
            "zero_idcg": "exclude from macro",
            "prior_bm25_ranking_sha256": digest(PREVIOUS / "rankings.jsonl.gz"),
            "prior_results_sha256": digest(PREVIOUS / "results.json"),
            "code_sha256": digest(Path(__file__)),
            "reranker_helper_sha256": digest(
                ROOT / "vrscore/benchmarking/malt_embedding_ablation.py"
            ),
        },
        "summary": summary,
    }
    (OUT / "results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("results written", flush=True)


def main():
    p, b, labels, r, h = load()
    needed = {bid for pid in p for bid in r[pid]["bm25_stop_ru"][:10]}
    print(
        "verified:",
        len(p),
        "profiles,",
        len(b),
        "briefs,",
        sum(map(len, labels.values())),
        "pairs",
        flush=True,
    )
    print("unique shortlisted briefs:", len(needed), flush=True)
    milvus_collection = os.environ.get(
        "MILVUS_MALT_COLLECTION", DEFAULT_MILVUS_MALT_COLLECTION
    )
    milvus_client = make_milvus_client()
    conn = None
    try:
        evaluate(
            p,
            b,
            labels,
            r,
            h,
            conn,
            milvus_client=milvus_client,
            milvus_collection=milvus_collection,
        )
    finally:
        milvus_client.close()


if __name__ == "__main__":
    main()
