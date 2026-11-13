"""Hydra composition of the selected retrieval index and alternatives."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from vrscore.indexes import BM25SkillIndex, ExactSkillIndex, TantivyExactSkillIndex

CONFIGS = str(Path(__file__).resolve().parents[1] / "configs")


def test_active_index_is_usable_and_can_be_overridden():
    vacancy = {"id": "v", "entities": [{"category": "hard_skill", "value": "C++"}]}
    resume = {
        "id": "r",
        "entities": [{"category": "hard_skill", "value": "C++", "polarity": "present"}],
    }
    with initialize_config_dir(config_dir=CONFIGS, version_base=None):
        cfg = compose(config_name="retrieval_index")
        index = instantiate(cfg.retrieval_method, vacancies=[vacancy])
        assert isinstance(index, BM25SkillIndex)
        assert index.search(resume).hits[0].vacancy_id == "v"
        assert cfg.retrieval_method.config.stopwords.enabled is True
        for name, cls in (
            ("exact", ExactSkillIndex),
            ("tantivy", TantivyExactSkillIndex),
        ):
            alternate = compose(
                config_name="retrieval_index", overrides=[f"retrieval_method={name}"]
            )
            assert isinstance(
                instantiate(alternate.retrieval_method, vacancies=[vacancy]), cls
            )
        no_stop = compose(
            config_name="retrieval_index",
            overrides=["retrieval_method.config.stopwords.enabled=false"],
        )
        assert (
            instantiate(no_stop.retrieval_method, vacancies=[]).stopwords == frozenset()
        )
