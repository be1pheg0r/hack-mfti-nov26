"""Persistent Milvus vector store accepting caller-prepared vectors only."""

from __future__ import annotations

import json
import math
import os
import stat
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from pymilvus import DataType, MilvusClient

DEFAULT_URI = "http://127.0.0.1:19530"
DEFAULT_TOKEN_FILE = Path(".data/milvus/token")


@dataclass(frozen=True)
class SearchResult:
    """A vector record identifier and cosine similarity to the query vector."""

    record_id: str
    score: float


class MilvusVectorStore:
    """Store caller-prepared vectors in a persistent Milvus collection.

    Collection schema is fixed to string primary keys and exact FLAT/COSINE
    search. Persistent collections are never dropped by ``close``.
    """

    def __init__(
        self,
        *,
        dimension: int,
        collection_name: str,
        uri: str = DEFAULT_URI,
        token: str | None = None,
        token_file: str | Path | None = DEFAULT_TOKEN_FILE,
        client: MilvusClient | None = None,
    ) -> None:
        """Open or create a dimension-checked persistent collection."""
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 1
        ):
            raise ValueError("Vector dimension must be a positive integer")
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("collection_name must be a non-empty string")
        if not isinstance(uri, str) or not uri.strip():
            raise ValueError("uri must be a non-empty string")
        self.dimension = dimension
        self.collection_name = collection_name
        self.uri = uri
        self.token_file = token_file
        if token is not None and (not isinstance(token, str) or not token.strip()):
            raise ValueError("token must be a non-empty string when supplied")
        self._token = token if token is not None else self.resolve_token(token_file)
        self._owns_client = client is None
        self.client = client or MilvusClient(uri=uri, token=self._token)
        self._ensure_collection()

    @staticmethod
    def resolve_token(token_file: str | Path | None = DEFAULT_TOKEN_FILE) -> str | None:
        """Read Milvus credentials from env or a private owner-only token file."""
        environment_token = os.environ.get("MILVUS_TOKEN")
        if environment_token:
            return environment_token
        if token_file is None:
            return None
        path = Path(token_file)
        try:
            info = path.lstat()
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ValueError("Milvus token file must be a regular non-symlink file")
        if stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError("Milvus token file must have permissions 0600")
        token = path.read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError("Milvus token file is empty")
        return token

    def _ensure_collection(self) -> None:
        """Create a collection once, handling concurrent creators safely."""
        if not self.client.has_collection(self.collection_name):
            schema = self.client.create_schema(
                auto_id=False, enable_dynamic_field=False
            )
            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=65535,
            )
            schema.add_field(
                field_name="vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=self.dimension,
            )
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector", index_type="FLAT", metric_type="COSINE"
            )
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    schema=schema,
                    index_params=index_params,
                )
            except Exception:
                # A concurrent constructor may have created it after our check.
                if not self.client.has_collection(self.collection_name):
                    raise
        self._validate_collection()

    def _validate_collection(self) -> None:
        """Reject collections whose primary key, vector dimension, or index differs."""
        description = self.client.describe_collection(self.collection_name)
        fields = description.get("fields", [])
        by_name = {field.get("name", field.get("fieldName")): field for field in fields}
        id_field = by_name.get("id")
        vector_field = by_name.get("vector")
        if id_field is None or vector_field is None:
            raise ValueError("Milvus collection must contain id and vector fields")
        id_type = getattr(
            id_field.get("type", id_field.get("dataType", "")), "name", None
        ) or str(id_field.get("type", id_field.get("dataType", "")))
        id_type = id_type.lower()
        if "varchar" not in id_type and "string" not in id_type:
            raise ValueError("Milvus collection id field must use string identifiers")
        vector_type = getattr(
            vector_field.get("type", vector_field.get("dataType", "")), "name", None
        ) or str(vector_field.get("type", vector_field.get("dataType", "")))
        vector_type = vector_type.lower()
        if "floatvector" not in vector_type and "float_vector" not in vector_type:
            raise ValueError("Milvus collection vector field must be FLOAT_VECTOR")
        params = vector_field.get("params", {})
        stored_dimension = params.get("dim", vector_field.get("dim"))
        if stored_dimension is None or int(stored_dimension) != self.dimension:
            raise ValueError(
                "Milvus collection dimension is "
                f"{stored_dimension}, not requested {self.dimension}"
            )
        index_names = self.client.list_indexes(
            self.collection_name, field_name="vector"
        )
        if not index_names:
            raise ValueError("Milvus collection requires a FLAT/COSINE vector index")
        indexes = [
            self.client.describe_index(self.collection_name, index_name)
            for index_name in index_names
        ]
        if not any(
            str(index.get("index_type", "")).upper() == "FLAT"
            and str(index.get("metric_type", "")).upper() == "COSINE"
            for index in indexes
        ):
            raise ValueError("Milvus collection requires a FLAT/COSINE vector index")

    def _validate_vector(self, vector: Sequence[float]) -> list[float]:
        """Validate finite, non-zero values of the configured dimension."""
        if isinstance(vector, (str, bytes)):
            raise ValueError("Vector must be a numeric sequence")
        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise ValueError("Vector must contain numeric values") from exc
        if len(values) != self.dimension:
            raise ValueError(
                f"Vector dimension must be {self.dimension}; received {len(values)}"
            )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Vector values must be finite")
        if not any(values):
            raise ValueError("Cosine distance is undefined for a zero vector")
        return values

    def upsert(self, record_id: str, vector: Sequence[float]) -> bool:
        """Insert or replace a vector under a caller-supplied string identifier."""
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("record_id must be a non-empty string")
        values = self._validate_vector(vector)
        self.client.upsert(
            collection_name=self.collection_name,
            data=[{"id": record_id, "vector": values}],
        )
        return True

    def upsert_many(self, vectors: Mapping[str, Sequence[float]]) -> None:
        """Validate and insert a bounded collection of vectors in one RPC."""
        records = []
        for record_id, vector in vectors.items():
            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError("record_id must be a non-empty string")
            records.append({"id": record_id, "vector": self._validate_vector(vector)})
        if records:
            self.client.upsert(collection_name=self.collection_name, data=records)

    def delete_many(self, record_ids: Sequence[str]) -> None:
        """Remove only the caller's explicitly named records."""
        if record_ids:
            self.client.delete(
                collection_name=self.collection_name, ids=list(record_ids)
            )

    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int = 5,
        record_ids: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        """Return cosine scores in deterministic order, restricted to IDs if set."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("Search limit must be a positive integer")
        values = self._validate_vector(vector)
        expression = ""
        if record_ids is not None:
            if isinstance(record_ids, (str, bytes)):
                raise ValueError("record_ids must be a sequence of strings")
            identifiers = tuple(dict.fromkeys(record_ids))
            if any(
                not isinstance(item, str) or not item.strip() for item in identifiers
            ):
                raise ValueError("record_ids must contain non-empty strings")
            if not identifiers:
                return []
            expression = "id in " + json.dumps(identifiers, ensure_ascii=True)
            candidate_limit = len(identifiers)
        else:
            candidate_limit = self.count()
        if candidate_limit == 0:
            return []
        response = self.client.search(
            collection_name=self.collection_name,
            data=[values],
            limit=max(limit, candidate_limit),
            filter=expression,
            output_fields=[],
            search_params={"metric_type": "COSINE", "params": {}},
            consistency_level="Strong",
        )
        hits = response[0] if response else []
        results = [
            SearchResult(str(hit["id"]), float(hit.get("distance", hit.get("score"))))
            for hit in hits
        ]
        if record_ids is not None:
            allowed = set(identifiers)
            results = [result for result in results if result.record_id in allowed]
        return sorted(results, key=lambda result: (-result.score, result.record_id))[
            :limit
        ]

    def count(self) -> int:
        """Return a strongly consistent count of vectors in the collection."""
        response = self.client.query(
            collection_name=self.collection_name,
            filter="",
            output_fields=["count(*)"],
            consistency_level="Strong",
        )
        return int(response[0]["count(*)"]) if response else 0

    def close(self) -> None:
        """Close an owned client connection without dropping persistent data."""
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "MilvusVectorStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def temporary_collection(
        self, *, collection_prefix: str = "temporary"
    ) -> Iterator["MilvusVectorStore"]:
        """Create and scope a unique temporary collection on this same client."""
        with type(self).temporary(
            dimension=self.dimension,
            uri=self.uri,
            token=self._token,
            token_file=self.token_file,
            collection_prefix=collection_prefix,
            client=self.client,
        ) as store:
            yield store

    @classmethod
    @contextmanager
    def temporary(
        cls,
        *,
        dimension: int,
        uri: str = DEFAULT_URI,
        token: str | None = None,
        token_file: str | Path | None = DEFAULT_TOKEN_FILE,
        collection_prefix: str = "temporary",
        client: MilvusClient | None = None,
    ) -> Iterator["MilvusVectorStore"]:
        """Yield a UUID-isolated collection and drop only that collection on exit."""
        if not isinstance(collection_prefix, str) or not collection_prefix.strip():
            raise ValueError("collection_prefix must be a non-empty string")
        name = f"{collection_prefix}_{uuid.uuid4().hex}"
        store = cls(
            dimension=dimension,
            collection_name=name,
            uri=uri,
            token=token,
            token_file=token_file,
            client=client,
        )
        try:
            yield store
        finally:
            try:
                store.client.drop_collection(collection_name=name)
            finally:
                store.close()
