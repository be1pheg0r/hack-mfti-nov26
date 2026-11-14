"""Hydra CLI for upserting and querying caller-prepared vectors."""

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from .vector_store import MilvusVectorStore

ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str) -> Path:
    """Resolve project-relative paths independently of Hydra's working directory."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


@hydra.main(version_base=None, config_path="../configs", config_name="vector_store")
def main(config: DictConfig) -> None:
    """Upsert a supplied vector or search with a supplied query vector."""
    token_file = resolve_path(config.token_file) if config.token_file else None
    if config.backend != "milvus":
        raise ValueError("backend must be 'milvus'")
    with MilvusVectorStore(
        uri=config.uri,
        collection_name=config.collection_name,
        token_file=token_file,
        dimension=config.dimension,
    ) as store:
        if config.operation == "upsert":
            store.upsert(config.record_id, config.vector)
            print(f"Upserted vector for {config.record_id!r}; total={store.count()}")
        elif config.operation == "search":
            results = store.search(
                config.vector, limit=config.limit, record_ids=config.record_ids
            )
            print(
                json.dumps(
                    [
                        {"record_id": result.record_id, "score": result.score}
                        for result in results
                    ],
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            raise ValueError("operation must be 'upsert' or 'search'")


if __name__ == "__main__":
    main()
