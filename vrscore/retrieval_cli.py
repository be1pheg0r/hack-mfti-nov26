"""Run gate, BM25 and embedding reranking on extracted JSON documents."""

import json
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from .embedding_client import EmbeddingClient
from .retrieval_pipeline import RetrievalPipeline


@hydra.main(
    version_base=None, config_path="../configs", config_name="retrieval_pipeline"
)
def main(config: DictConfig) -> None:
    """Print the complete ranking and gate decisions as JSON."""
    resume = json.loads(
        Path(to_absolute_path(config.resume)).read_text(encoding="utf-8")
    )
    vacancies = json.loads(
        Path(to_absolute_path(config.vacancies)).read_text(encoding="utf-8")
    )
    settings = OmegaConf.to_container(config, resolve=True)
    client = EmbeddingClient(settings["server"], timeout=settings["timeout_seconds"])
    pipeline = RetrievalPipeline(
        vacancies,
        client,
        shortlist_size=settings["shortlist_size"],
        gate_config=settings["gate_config"],
        bm25_config=settings["bm25_config"],
        include_unknown=settings["include_unknown"],
        document_cache_collection=settings["document_cache_collection"],
        document_batch_size=settings["document_batch_size"],
        milvus_uri=settings["milvus_uri"],
        milvus_token_file=to_absolute_path(settings["milvus_token_file"])
        if settings["milvus_token_file"]
        else None,
    )
    print(json.dumps(pipeline.search(resume), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
