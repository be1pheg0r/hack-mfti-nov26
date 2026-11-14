"""Configurable public-input LLM, embedding, Milvus, and BM25 benchmark."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from vrscore.benchmarking.exact_skill_experiment import _validate_dataset
from vrscore.demo.service import DemoService
from vrscore.embedding_client import EmbeddingClient
from vrscore.indexes import BM25SkillIndex
from vrscore.llm import LLMConfig, OpenRouterClient
from vrscore.vector_store import MilvusVectorStore


class RecordingChat:
    """Add unique request nonces and retain provider metadata, never prompt text."""

    def __init__(self, chat: Any):
        self.original = chat
        self.last: dict[str, Any] | None = None

    def send(self, **kwargs):
        messages = [dict(message) for message in kwargs["messages"]]
        messages[0]["content"] = (
            f"Independent benchmark request identifier: {uuid.uuid4().hex}. "
            "Ignore this identifier when extracting facts.\n" + messages[0]["content"]
        )
        kwargs["messages"] = messages
        self.last = None
        try:
            response = self.original.send(**kwargs)
        except Exception as exc:
            self.last = {"error_type": type(exc).__name__}
            raise
        usage = getattr(response, "usage", None)
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump(mode="json")
        self.last = {
            "id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "provider": getattr(response, "provider", None),
            "usage": usage
            if isinstance(usage, (dict, list, str, int, float, type(None)))
            else None,
            "finish_reason": response.choices[0].finish_reason
            if response.choices
            else None,
        }
        usage_data = self.last.get("usage")
        details = (
            usage_data.get("prompt_tokens_details", {})
            if isinstance(usage_data, dict)
            else {}
        )
        if not isinstance(details, dict):
            details = {}
        self.last["cached_tokens"] = details.get("cached_tokens")
        self.last["cache_usage_status"] = (
            "reported" if "cached_tokens" in details else "missing"
        )
        if type(self.last["cached_tokens"]) is not int:
            self.last["cache_usage_status"] = "missing"
            raise RuntimeError("Provider did not report cache telemetry")
        if self.last["cached_tokens"]:
            self.last["cache_usage_status"] = "cache_hit_rejected"
            raise RuntimeError("Provider reported cached prompt tokens")
        return response


class RecordingSDK:
    """Expose only the wrapped chat interface expected by OpenRouterClient."""

    def __init__(self, sdk: Any):
        self.chat = RecordingChat(sdk.chat)


def make_row(kind: str, iteration: int, document: dict[str, Any]) -> dict[str, Any]:
    """Create a text-free provenance row for a source document."""
    return {
        "kind": kind,
        "iteration": iteration,
        "document_id": document["id"],
        "text_chars": len(document["text"]),
        "source_sha256": hashlib.sha256(document["text"].encode()).hexdigest(),
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-stage duration, provider cost, and outcomes for a group."""
    successful = [row for row in rows if row.get("status") == "ok"]
    failed = [row for row in rows if row.get("status") != "ok"]
    durations: dict[str, dict[str, float | int]] = {}
    fields = sorted(
        {key for row in successful for key in row if key.endswith("_seconds")}
    )
    for field in fields:
        values = [
            row[field] for row in successful if isinstance(row.get(field), (int, float))
        ]
        if values:
            durations[field] = {
                "count": len(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
    costs = [
        usage.get("cost")
        for row in rows
        if isinstance((llm := row.get("llm")), dict)
        and isinstance((usage := llm.get("usage")), dict)
    ]
    costs = [value for value in costs if isinstance(value, (int, float))]
    return {
        "total": len(rows),
        "success": len(successful),
        "failure": len(failed),
        "llm_cost_reported_total": sum(costs) if costs else None,
        "llm_cost_reported_mean": statistics.mean(costs) if costs else None,
        "llm_cost_reported_count": len(costs),
        "durations_seconds": durations,
    }


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate outcomes globally and separately for each document type."""
    summary = _summarize_rows(rows)
    summary["by_kind"] = {
        kind: _summarize_rows([row for row in rows if row.get("kind") == kind])
        for kind in ("vacancy", "resume")
    }
    return summary


def safe_write_json(path: Path, value: dict[str, Any]) -> None:
    """Create a private artifact once; never overwrite an earlier run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")


def _checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically refresh this run's private progress artifact."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump({"rows": rows}, output, ensure_ascii=False, indent=2)
        output.write("\n")
    temporary.replace(path)


def run(config: DictConfig) -> dict[str, Any]:
    """Execute configured iterations over the approved corpus."""
    settings = OmegaConf.to_container(config, resolve=True)
    if settings["iterations"] < 1:
        raise ValueError("iterations must be positive")
    output_dir = Path(to_absolute_path(settings["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    rows_path, results_path = output_dir / "rows.json", output_dir / "results.json"
    if rows_path.exists() or results_path.exists():
        raise FileExistsError("Benchmark output exists; choose a new output_dir")

    llm_config = OmegaConf.load(to_absolute_path(settings["llm_config"]))
    llm_config.max_tokens = settings["max_tokens"]
    client = OpenRouterClient(
        LLMConfig.model_validate(OmegaConf.to_container(llm_config, resolve=True))
    )
    sdk = client.sdk
    client.sdk = RecordingSDK(sdk)
    service = DemoService(client)
    vacancies = service.vacancies
    resumes = [item for item in service.resumes if item.get("synthetic") is True]
    if len(vacancies) != 3 or len(resumes) != 6:
        raise ValueError("Approved public vacancy/synthetic resume set changed")
    encoder = EmbeddingClient(
        settings["embedding_url"], timeout=settings["timeout_seconds"]
    )
    fingerprint = encoder.cache_fingerprint
    if not fingerprint:
        raise ValueError("Embedding server did not provide a model fingerprint")
    expected_revision = settings.get("expected_embedding_revision")
    if expected_revision and fingerprint.get("revision") != expected_revision:
        raise ValueError(
            "Embedding server revision differs from configured expectation"
        )

    gt_dir = Path(to_absolute_path(settings["gt_dir"]))
    _, dataset = _validate_dataset(gt_dir)
    extraction_by_id = {item["id"]: item for item in dataset["llm_extractions.jsonl"]}
    baseline = [extraction_by_id[item["id"]] for item in dataset["vacancies.jsonl"]]
    normalizer = BM25SkillIndex([])
    collection = settings["collection_prefix"] + uuid.uuid4().hex
    store = MilvusVectorStore(
        dimension=settings["embedding_dimension"],
        collection_name=collection,
        uri=settings["milvus_uri"],
        token_file=to_absolute_path(settings["milvus_token_file"])
        if settings["milvus_token_file"]
        else None,
    )
    rows: list[dict[str, Any]] = []
    ids: list[str] = []
    stopped_reason = None
    try:
        for kind, documents in (("vacancy", vacancies), ("resume", resumes)):
            for iteration in range(1, settings["iterations"] + 1):
                document = documents[(iteration - 1) % len(documents)]
                row = make_row(kind, iteration, document)
                chat = client.sdk.chat
                chat.last = None
                started = time.perf_counter()
                try:
                    stage = time.perf_counter()
                    result = service.extract_document(document, kind=kind)
                    row["llm_extract_seconds"] = time.perf_counter() - stage
                    row["llm"] = chat.last
                    if chat.last is None:
                        raise RuntimeError("Provider returned without usage metadata")
                    row["entities"] = len(result["entities"])
                    row["rejected"] = result["rejected"]
                    stage = time.perf_counter()
                    vector = (
                        encoder.encode_documents(
                            [document["text"]], titles=[document["title"]]
                        )[0]
                        if kind == "vacancy"
                        else encoder.encode_query(document["text"])
                    )
                    row["embedding_seconds"] = time.perf_counter() - stage
                    record_id = f"{kind}_{iteration}_{uuid.uuid4().hex}"
                    stage = time.perf_counter()
                    store.upsert(record_id, vector)
                    ids.append(record_id)
                    row["milvus_upsert_seconds"] = time.perf_counter() - stage
                    if kind == "vacancy":
                        stage = time.perf_counter()
                        BM25SkillIndex(baseline + [{**result, "id": record_id}])
                        row["bm25_rebuild_seconds"] = time.perf_counter() - stage
                    else:
                        stage = time.perf_counter()
                        tokens = {
                            token
                            for entity in result["entities"]
                            if entity.get("category") in normalizer.categories
                            and entity.get("polarity") == "present"
                            and isinstance(entity.get("value"), str)
                            for token in normalizer._tokens(entity["value"])
                        }
                        row["bm25_prepare_query_seconds"] = time.perf_counter() - stage
                        row["query_token_count"] = len(tokens)
                    row["index_seconds"] = (
                        row["embedding_seconds"]
                        + row["milvus_upsert_seconds"]
                        + row.get("bm25_rebuild_seconds", 0)
                        + row.get("bm25_prepare_query_seconds", 0)
                    )
                    row["total_seconds"] = (
                        row["llm_extract_seconds"] + row["index_seconds"]
                    )
                    row["status"] = "ok"
                except Exception as exc:
                    row["status"] = "error"
                    row["error_type"] = type(exc).__name__
                    row["elapsed_before_error_seconds"] = time.perf_counter() - started
                    row["llm"] = chat.last
                    if chat.last is None or "error_type" in chat.last:
                        stopped_reason = (
                            "Provider request failed before response; stopped run"
                        )
                rows.append(row)
                _checkpoint(rows_path, rows)
                if stopped_reason:
                    break
            if stopped_reason:
                break
        returned = (
            store.client.get(
                collection_name=collection,
                ids=ids,
                output_fields=["id"],
                consistency_level="Strong",
            )
            if ids
            else []
        )
        returned_ids = {item["id"] for item in returned}
        if returned_ids != set(ids):
            raise ValueError("Milvus strong-consistency readback mismatch")
        result = {
            "configuration": settings,
            "model": llm_config.model,
            "provider": OmegaConf.to_container(llm_config.provider, resolve=True),
            "embedding_fingerprint": fingerprint,
            "provenance": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "source_sha256": {
                    row["document_id"]: row["source_sha256"]
                    for row in [make_row("vacancy", 1, item) for item in vacancies]
                    + [make_row("resume", 1, item) for item in resumes]
                },
                "ground_truth_manifest_sha256": hashlib.sha256(
                    (gt_dir / "manifest.json").read_bytes()
                ).hexdigest(),
            },
            "milvus": {"collection": collection, "readback_count": len(returned_ids)},
            "summary": build_summary(rows),
            "rows": rows,
            "stopped_reason": stopped_reason,
        }
        safe_write_json(results_path, result)
        return result
    finally:
        try:
            store.client.drop_collection(collection)
        finally:
            store.close()
            if hasattr(sdk, "__exit__"):
                sdk.__exit__(None, None, None)


@hydra.main(
    version_base=None, config_path="../../configs", config_name="indexing_benchmark"
)
def main(config: DictConfig) -> None:
    """Run benchmark and print a concise, text-free summary."""
    result = run(config)
    print(
        json.dumps(
            {"summary": result["summary"], "output_dir": config.output_dir}, indent=2
        )
    )
    if (
        result["summary"]["failure"]
        or result["summary"]["total"] != 2 * config.iterations
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
