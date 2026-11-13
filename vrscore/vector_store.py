"""Persistent SQLite vector index accepting vectors from caller code only."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import sqlite_vec


@dataclass(frozen=True)
class SearchResult:
    """A vector record identifier and cosine similarity to the query vector."""

    record_id: str
    score: float


class LocalVectorStore:
    """Persist ready-to-use float vectors in a local sqlite-vec index.

    This class deliberately does not embed, parse, or store source documents.
    Callers supply an ID and a prepared vector; returned results contain IDs and
    similarity scores so document storage stays with the calling application.
    """

    def __init__(self, database_path: str | Path, *, dimension: int) -> None:
        """Open/create a persistent vector index with a fixed vector dimension."""
        if dimension < 1:
            raise ValueError("Vector dimension must be positive")
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        self._connection = sqlite3.connect(self.database_path)
        self._connection.enable_load_extension(True)
        sqlite_vec.load(self._connection)
        self._connection.enable_load_extension(False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS vector_metadata "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        stored_dimension = self._connection.execute(
            "SELECT value FROM vector_metadata WHERE key='dimension'"
        ).fetchone()
        if stored_dimension and int(stored_dimension[0]) != dimension:
            self._connection.close()
            raise ValueError(
                f"Index dimension is {stored_dimension[0]}, not requested {dimension}"
            )
        self._connection.execute(
            "INSERT OR IGNORE INTO vector_metadata VALUES ('dimension', ?)",
            (str(dimension),),
        )
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS vector_ids (record_id TEXT NOT NULL UNIQUE)"
        )
        self._connection.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vector_values USING vec0("
            f"embedding float[{dimension}] distance_metric=cosine)"
        )
        self._connection.commit()

    def _validate(self, vector: Sequence[float]) -> list[float]:
        """Validate caller-supplied finite, non-zero vector values."""
        values = [float(value) for value in vector]
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
        """Insert or replace a prepared vector without storing source JSON."""
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("record_id must be a non-empty string")
        values = self._validate(vector)
        existing = self._connection.execute(
            "SELECT rowid FROM vector_ids WHERE record_id=?", (record_id,)
        ).fetchone()
        with self._connection:
            if existing:
                rowid = existing[0]
                self._connection.execute(
                    "DELETE FROM vector_values WHERE rowid=?", (rowid,)
                )
            else:
                cursor = self._connection.execute(
                    "INSERT INTO vector_ids(record_id) VALUES (?)", (record_id,)
                )
                rowid = cursor.lastrowid
            self._connection.execute(
                "INSERT INTO vector_values(rowid, embedding) VALUES (?, ?)",
                (rowid, sqlite_vec.serialize_float32(values)),
            )
        return True

    def search(self, vector: Sequence[float], *, limit: int = 5) -> list[SearchResult]:
        """Return nearest IDs using sqlite-vec cosine distance and KNN search."""
        if limit < 1:
            raise ValueError("Search limit must be positive")
        values = self._validate(vector)
        rows = self._connection.execute(
            "SELECT ids.record_id, vectors.distance "
            "FROM vector_values AS vectors "
            "JOIN vector_ids AS ids ON ids.rowid=vectors.rowid "
            "WHERE vectors.embedding MATCH ? AND k=? ORDER BY vectors.distance",
            (sqlite_vec.serialize_float32(values), limit),
        ).fetchall()
        return [SearchResult(record_id, 1.0 - distance) for record_id, distance in rows]

    def count(self) -> int:
        """Return the number of vectors in this index."""
        count = self._connection.execute("SELECT count(*) FROM vector_ids").fetchone()[
            0
        ]
        return int(count)

    def close(self) -> None:
        """Close the database connection."""
        self._connection.close()

    def __enter__(self) -> "LocalVectorStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
