"""Run the first gate on two parsed JSON documents with Hydra overrides."""

import json
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from .gates import evaluate_first_gate


@hydra.main(version_base=None, config_path="../configs", config_name="first_gate_cli")
def main(config: DictConfig) -> None:
    """Read the requested files and print a JSON gate decision to stdout."""
    vacancy = json.loads(
        Path(to_absolute_path(config.vacancy)).read_text(encoding="utf-8")
    )
    resume = json.loads(
        Path(to_absolute_path(config.resume)).read_text(encoding="utf-8")
    )
    settings = OmegaConf.to_container(config, resolve=True)
    if not isinstance(settings, dict):
        raise ValueError("First-gate configuration must be a mapping")
    result = evaluate_first_gate(vacancy, resume, config=settings)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
