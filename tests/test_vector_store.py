"""Persistent local vector-index tests using only synthetic vectors."""

from pathlib import Path

import pytest

from vrscore.vector_store import LocalVectorStore


def test_ready_vectors_persist_and_are_retrievable_after_reopen(tmp_path: Path):
    database = tmp_path / "vectors.sqlite3"
    store = LocalVectorStore(database, dimension=3)
    assert store.upsert("near", [1.0, 0.0, 0.0])
    assert store.upsert("far", [0.0, 1.0, 0.0])
    store.close()

    reopened = LocalVectorStore(database, dimension=3)
    results = reopened.search([0.9, 0.1, 0.0], limit=1)

    assert [result.record_id for result in results] == ["near"]
    assert results[0].score > 0.9
    assert reopened.count() == 2
    reopened.close()


def test_upsert_replaces_vector_and_rejects_invalid_dimensions(tmp_path: Path):
    store = LocalVectorStore(tmp_path / "vectors.sqlite3", dimension=2)
    store.upsert("same-id", [1.0, 0.0])
    store.upsert("same-id", [0.0, 1.0])

    assert store.count() == 1
    assert store.search([0.0, 1.0])[0].score == pytest.approx(1.0)
    with pytest.raises(ValueError, match="dimension"):
        store.upsert("bad", [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        store.upsert("bad", [float("nan"), 0.0])
    store.close()


def test_hydra_config_exposes_vector_operations_without_ingestion_defaults():
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[1] / "configs")
    ):
        config = compose(config_name="vector_store")

    assert config.backend == "sqlite_vec"
    assert config.operation == "search"
    assert config.dimension > 0
    assert config.database_path.endswith("vectors.sqlite3")
