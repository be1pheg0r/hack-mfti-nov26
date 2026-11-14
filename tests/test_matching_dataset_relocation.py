"""Validate the relocated matching dataset and configured consumers."""

from pathlib import Path

from omegaconf import OmegaConf

from vrscore.benchmarking.exact_skill_experiment import _validate_dataset
from vrscore.dataset_viewer.server import load_dataset

ROOT = Path(__file__).resolve().parents[1]


def test_matching_dataset_and_default_paths():
    """The shipped data remains valid and configured paths all point to it."""
    dataset = ROOT / "public_data/mfti_matching_gt"
    manifest, rows = _validate_dataset(dataset)
    assert manifest["actual"]["pairs"] == len(rows["pairs.jsonl"]) == 200
    assert len(load_dataset(dataset)["records"]) == 30
    for filename in (
        "exact_skills_experiment.yaml",
        "index_ablation.yaml",
        "embedding_ablation.yaml",
        "vocabulary_ablation.yaml",
        "indexing_benchmark.yaml",
    ):
        assert OmegaConf.load(ROOT / "configs" / filename).gt_dir == str(
            dataset.relative_to(ROOT)
        )
    assert OmegaConf.load(ROOT / "configs/demo.yaml").server.matching_gt_path == str(
        dataset.relative_to(ROOT)
    )
    assert OmegaConf.load(ROOT / "configs/dataset_viewer.yaml").extra_datasets[
        "mfti_matching_gt"
    ] == str(dataset.relative_to(ROOT))
