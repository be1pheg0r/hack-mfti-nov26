"""Hydra CLI for local, versioned DOCX ingestion; no document text in logs."""

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from .contracts import DocumentError
from .docx import ReaderOptions
from .storage import DocumentStore

ROOT = Path(__file__).resolve().parents[2]


def resolve(value: str) -> Path:
    """Resolve project-relative paths consistently with the existing CLIs."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


@hydra.main(
    version_base=None, config_path=str(ROOT / "configs"), config_name="documents"
)
def main(config: DictConfig) -> None:
    """Ingest one DOCX or exit nonzero with a structured, actionable error."""
    try:
        store = DocumentStore(
            resolve(config.storage_root), ReaderOptions(**config.reader)
        )
        result = store.ingest(resolve(config.source_path))
    except DocumentError as error:
        print(
            json.dumps(
                {"error": error.code, "message": str(error)}, ensure_ascii=False
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    print(
        json.dumps(
            {
                "document_id": result.document.document_id,
                "text_version": result.document.text_version,
                "text_sha256": result.document.text_sha256,
                "blocks": len(result.document.blocks),
                "warnings": result.document.warnings,
                "original_path": str(result.original_path),
                "parsed_path": str(result.parsed_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
