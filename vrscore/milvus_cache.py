"""Milvus-backed implementation of the document embedding cache contract."""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Any, Iterable


class MilvusEmbeddingCache:
    """Get/put vectors keyed by the exact fingerprint/content-hash pair.

    Args: uri, token, collection, dimension. An injected client is supported for
    tests; production connections use pymilvus.MilvusClient.
    """

    def __init__(
        self,
        uri: str,
        token: str,
        collection: str,
        dimension: int,
        *,
        client: Any = None,
    ):
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.collection = collection
        self.dimension = dimension
        if client is None:
            from pymilvus import MilvusClient

            client = MilvusClient(uri=uri, token=token)
        self.client = client
        self._ensure_collection()

    @staticmethod
    def _check_hash(value: str, name: str) -> None:
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")

    @staticmethod
    def _key(fingerprint: str, content_hash: str) -> str:
        return hashlib.sha256(f"{fingerprint}\0{content_hash}".encode()).hexdigest()

    def _ensure_collection(self) -> None:
        if not self.client.has_collection(collection_name=self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                dimension=self.dimension,
                id_type="string",
                max_length=64,
                auto_id=False,
                primary_field_name="id",
                vector_field_name="vector",
                enable_dynamic_field=True,
            )

    def get(self, fingerprint: str, content_hash: str) -> list[float] | None:
        self._check_hash(fingerprint, "fingerprint")
        self._check_hash(content_hash, "content_hash")
        rows = self.client.get(
            collection_name=self.collection, ids=[self._key(fingerprint, content_hash)]
        )
        if not rows:
            return None
        row = rows[0]
        if (
            row.get("fingerprint") != fingerprint
            or row.get("content_hash") != content_hash
        ):
            raise ValueError("Milvus cache key metadata conflict")
        vector = [float(x) for x in row["vector"]]
        if len(vector) != self.dimension or not all(math.isfinite(x) for x in vector):
            raise ValueError("invalid cached vector dimension or values")
        return vector

    def put(
        self,
        fingerprint: str,
        content_hash: str,
        vector: Iterable[float],
        *,
        metadata: dict | None = None,
    ) -> None:
        self._check_hash(fingerprint, "fingerprint")
        self._check_hash(content_hash, "content_hash")
        values = [float(x) for x in vector]
        if (
            len(values) != self.dimension
            or not values
            or not all(math.isfinite(x) for x in values)
        ):
            raise ValueError("vector dimension or values invalid")
        # Compare canonical float32 because Milvus stores FLOAT_VECTOR values.
        packed = struct.pack("<" + "f" * self.dimension, *values)
        vector_hash = hashlib.sha256(packed).hexdigest()
        key = self._key(fingerprint, content_hash)
        old = self.get(fingerprint, content_hash)
        if old is not None:
            old_hash = hashlib.sha256(
                struct.pack("<" + "f" * self.dimension, *old)
            ).hexdigest()
            if old_hash != vector_hash:
                raise ValueError("conflicting vector for existing Milvus cache key")
            return
        row = {
            "id": key,
            "fingerprint": fingerprint,
            "content_hash": content_hash,
            "vector": values,
            "vector_sha256": vector_hash,
        }
        if metadata:
            for name, value in metadata.items():
                if name in row or name == "id":
                    raise ValueError(f"reserved cache metadata field: {name}")
                row[name] = value
        self.client.insert(collection_name=self.collection, data=[row])

    def count(self) -> int:
        result = self.client.query(
            collection_name=self.collection,
            filter="",
            output_fields=["id"],
            limit=1000000,
        )
        return len(result)
