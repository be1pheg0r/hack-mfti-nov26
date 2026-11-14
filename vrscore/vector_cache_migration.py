"""Lossless, resumable migration of SQLite embedding caches to Milvus."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vrscore.milvus_cache import MilvusEmbeddingCache

MILVUS_URI = "http://127.0.0.1:19530"


@dataclass(frozen=True)
class MigrationReport:
    collection_name: str
    source_count: int
    inserted: int
    skipped: int
    source_sha256: str


@dataclass(frozen=True)
class VerificationReport:
    count: int
    vector_hashes_match: bool
    payload_hashes_match: bool


def make_milvus_client(token_file: str | Path = "/root/mfti_hack/.data/milvus/token"):
    """Create authenticated MilvusClient; never expose credentials in errors/logs."""
    from pymilvus import MilvusClient

    token = os.environ.get("MILVUS_TOKEN")
    if not token:
        path = Path(token_file)
        if path.exists():
            if path.stat().st_mode & 0o077:
                raise PermissionError("Milvus token file must have mode 0600")
            token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError(
            "Set MILVUS_TOKEN or provide the protected Milvus token file"
        )
    return MilvusClient(uri=MILVUS_URI, token=token)


def malt_cache_key(role: str, record_id: str) -> str:
    if role not in {"query", "document"}:
        raise ValueError("invalid Malt cache role")
    return hashlib.sha256(f"{role}\0{record_id}".encode()).hexdigest()


def get_malt_vector(
    client: Any, collection_name: str, role: str, record_id: str
) -> list[float] | None:
    """Fetch one Malt embedding by its stable role/id key."""
    rows = client.get(
        collection_name=collection_name, ids=[malt_cache_key(role, record_id)]
    )
    if not rows:
        return None
    row = rows[0]
    if row.get("role") != role or row.get("record_id") != record_id:
        raise ValueError("Milvus Malt cache key metadata conflict")
    return [float(x) for x in row["vector"]]


def _collection_name(kind: str, path: Path) -> str:
    identity = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
    return f"{kind}_{identity}"


def _ensure_collection(client: Any, name: str, dimension: int) -> None:
    if client.has_collection(collection_name=name):
        return
    try:
        client.create_collection(
            collection_name=name,
            dimension=dimension,
            id_type="string",
            max_length=64,
            auto_id=False,
            primary_field_name="id",
            vector_field_name="vector",
            enable_dynamic_field=True,
        )
    except TypeError:
        # Small injected clients used by tests may implement a narrower interface.
        client.create_collection(collection_name=name, dimension=dimension)


def _source_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _f32_bytes(values: list[float]) -> bytes:
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("vector must be non-empty and finite")
    return struct.pack("<" + "f" * len(values), *values)


def _vector_hash(values: list[float]) -> str:
    return hashlib.sha256(_f32_bytes(values)).hexdigest()


def _query_all(client: Any, name: str, expected_count: int) -> list[dict]:
    if hasattr(client, "query_iterator"):
        iterator = client.query_iterator(
            collection_name=name,
            batch_size=1000,
            limit=expected_count,
            filter="",
            output_fields=["*"],
        )
        result = []
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                result.extend(batch)
        finally:
            close = getattr(iterator, "close", None)
            if close:
                close()
        return result
    return client.query(
        collection_name=name,
        filter="",
        output_fields=["*"],
        limit=max(expected_count, 1),
    )


def _verify_expected(client: Any, name: str, expected: list[dict]) -> None:
    actual_rows = _query_all(client, name, len(expected))
    if len(actual_rows) != len(expected):
        raise ValueError(
            f"row count mismatch: expected {len(expected)}, found {len(actual_rows)}"
        )
    actual = {row.get("id"): row for row in actual_rows}
    if len(actual) != len(actual_rows):
        raise ValueError("duplicate primary keys found during verification")
    for source in expected:
        row = actual.get(source["id"])
        if row is None:
            raise ValueError("missing migrated row")
        for key, value in source.items():
            if key == "vector":
                if _f32_bytes([float(x) for x in row.get("vector", [])]) != _f32_bytes(
                    value
                ):
                    raise ValueError("vector equality verification failed")
            elif row.get(key) != value:
                raise ValueError(f"metadata equality verification failed for {key}")


def _insert_rows(
    client: Any, name: str, rows: list[dict], batch_size: int
) -> tuple[int, int]:
    inserted = skipped = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        # Exact repeats are resumable; changes to an existing key are fatal.
        existing = client.get(collection_name=name, ids=[r["id"] for r in batch])
        by_id = {r["id"]: r for r in existing}
        pending = []
        for row in batch:
            old = by_id.get(row["id"])
            if old is None:
                pending.append(row)
            elif (
                old.get("vector_sha256") != row["vector_sha256"]
                or _vector_hash([float(x) for x in old.get("vector", [])])
                != row["vector_sha256"]
                or old.get("payload_sha256") != row["payload_sha256"]
                or old.get("model_identity") != row["model_identity"]
            ):
                raise ValueError(f"conflict for migrated cache row {row['id']}")
            else:
                skipped += 1
        if pending:
            client.insert(collection_name=name, data=pending)
            inserted += len(pending)
    return inserted, skipped


def migrate_malt_cache(
    path: str | Path, client: Any | None = None, batch_size: int = 256
) -> MigrationReport:
    """Migrate role/id/text_sha/little-endian float32 BLOB Malt cache."""
    path = Path(path)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        metadata = dict(conn.execute("SELECT key,value FROM metadata"))
        model = metadata.get("model")
        if not model:
            raise ValueError("Malt cache lacks immutable model identity")
        raw = conn.execute(
            "SELECT role,id,text_sha,vector FROM embeddings ORDER BY role,id"
        ).fetchall()
    finally:
        conn.close()
    rows, dimension = [], None
    for role, key, text_sha, blob in raw:
        if role not in {"query", "document"} or len(blob) == 0 or len(blob) % 4:
            raise ValueError("invalid Malt cache row or vector BLOB")
        vector = list(struct.unpack("<" + "f" * (len(blob) // 4), blob))
        if not all(math.isfinite(x) for x in vector):
            raise ValueError("vector contains non-finite values")
        dimension = dimension or len(vector)
        if len(vector) != dimension:
            raise ValueError("inconsistent vector dimensions")
        key_id = hashlib.sha256(f"{role}\0{key}".encode()).hexdigest()
        rows.append(
            {
                "id": key_id,
                "source_key": f"{role}\0{key}",
                "role": role,
                "record_id": key,
                "text_sha": text_sha,
                "model_identity": model,
                "payload_sha256": hashlib.sha256(text_sha.encode()).hexdigest(),
                "vector_sha256": hashlib.sha256(blob).hexdigest(),
                "vector": vector,
            }
        )
    if client is None:
        client = make_milvus_client()
    name = _collection_name("malt_embedding_cache_v1", path)
    _ensure_collection(client, name, dimension or 1)
    inserted, skipped = _insert_rows(client, name, rows, batch_size)
    _verify_expected(client, name, rows)
    return MigrationReport(name, len(rows), inserted, skipped, _source_hash(path))


def migrate_pipeline_cache(
    path: str | Path,
    client: Any | None = None,
    *,
    model_identity: str | None = None,
    batch_size: int = 256,
) -> MigrationReport:
    """Migrate fingerprint/content_hash JSON vectors without merging fingerprints."""
    path = Path(path)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        raw = conn.execute(
            "SELECT fingerprint,content_hash,vector FROM document_embedding_cache "
            "ORDER BY fingerprint,content_hash"
        ).fetchall()
    finally:
        conn.close()
    rows, dimension = [], None
    for fingerprint, content_hash, raw_vector in raw:
        try:
            vector = json.loads(raw_vector)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid pipeline vector JSON") from exc
        if (
            not isinstance(vector, list)
            or not vector
            or any(
                not isinstance(x, (int, float)) or isinstance(x, bool) for x in vector
            )
        ):
            raise ValueError("invalid pipeline vector")
        vector = [float(x) for x in vector]
        packed = _f32_bytes(vector)
        dimension = dimension or len(vector)
        if len(vector) != dimension:
            raise ValueError("inconsistent vector dimensions")
        identity = model_identity or f"legacy-fingerprint:{fingerprint}"
        key_id = hashlib.sha256(f"{fingerprint}\0{content_hash}".encode()).hexdigest()
        rows.append(
            {
                "id": key_id,
                "fingerprint": fingerprint,
                "content_hash": content_hash,
                "model_identity": identity,
                "payload_sha256": hashlib.sha256(raw_vector.encode()).hexdigest(),
                "vector_sha256": hashlib.sha256(packed).hexdigest(),
                "vector": vector,
            }
        )
    name = _collection_name("pipeline_document_cache_v1", path)
    if client is None:
        client = make_milvus_client()
    cache = MilvusEmbeddingCache(MILVUS_URI, "", name, dimension or 1, client=client)
    inserted = skipped = 0
    for row in rows:
        prior = cache.get(row["fingerprint"], row["content_hash"])
        if prior is None:
            cache.put(
                row["fingerprint"],
                row["content_hash"],
                row["vector"],
                metadata={
                    "model_identity": row["model_identity"],
                    "payload_sha256": row["payload_sha256"],
                },
            )
            inserted += 1
        else:
            skipped += 1
    _verify_expected(client, name, rows)
    return MigrationReport(name, len(rows), inserted, skipped, _source_hash(path))


def verify_collection(
    client: Any, collection_name: str, expected_count: int, page_size: int = 1000
) -> VerificationReport:
    """Read back all persisted rows and verify counts plus float32/payload digests."""
    rows = _query_all(client, collection_name, expected_count)
    if len(rows) != expected_count:
        raise ValueError(
            f"row count mismatch: expected {expected_count}, found {len(rows)}"
        )
    vectors_ok = payloads_ok = True
    for row in rows:
        vector = [float(x) for x in row["vector"]]
        vectors_ok &= _vector_hash(vector) == row.get("vector_sha256")
        payloads_ok &= (
            isinstance(row.get("payload_sha256"), str)
            and len(row["payload_sha256"]) == 64
        )
    if not vectors_ok or not payloads_ok:
        raise ValueError("Milvus migration digest verification failed")
    return VerificationReport(len(rows), vectors_ok, payloads_ok)
