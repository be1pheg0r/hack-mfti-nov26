"""Hydra entry point for the vacancy parsing pipeline."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from .dataset import build_dataset

ROOT = Path(__file__).resolve().parents[2]


def resolve(value: str) -> Path:
    """Resolve configured project-relative paths independently of Hydra's cwd."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


@hydra.main(
    version_base=None, config_path=str(ROOT / "configs"), config_name="vacancy_parser"
)
def main(config: DictConfig) -> None:
    """Build the configured output corpus, refusing to replace an existing one."""
    manifest = build_dataset(
        resolve(config.source_dir),
        resolve(config.output_dir),
        evidence_path=resolve(config.source_evidence)
        if config.source_evidence
        else None,
        resolutions_path=(
            resolve(config.duplicate_resolutions)
            if config.duplicate_resolutions
            else None
        ),
        vocabulary_path=resolve(config.skill_vocabulary)
        if config.skill_vocabulary
        else None,
    )
    print(
        f"Parsed {manifest['actual']['vacancies']} vacancies into "
        f"{resolve(config.output_dir)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
