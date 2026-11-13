"""Hydra CLI for local redaction of one JSON document."""

import json
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from .deidentification import deidentify

ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str) -> Path:
    """Resolve absolute paths unchanged and relative paths from the repository."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


@hydra.main(
    version_base=None, config_path=str(ROOT / "configs"), config_name="deidentify"
)
def main(config: DictConfig) -> None:
    """Read JSON locally and write a redacted derivative without printing PII."""
    source = _resolve(config.input_path)
    destination = _resolve(config.output_path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
        redacted = deidentify(data, names=list(config.names))
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(json.dumps(redacted, ensure_ascii=False, indent=2) + "\n")
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        print("Unable to process input or write output.", file=sys.stderr)
        raise SystemExit(1) from None
    print(f"Redacted JSON written to {destination}")


if __name__ == "__main__":
    main()
