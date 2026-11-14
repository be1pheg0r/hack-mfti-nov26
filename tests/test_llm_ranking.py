import hashlib
import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from vrscore.benchmarking.llm_ranking import run, validate_ranking


def test_ranking_must_cover_each_vacancy_exactly_once():
    assert validate_ranking(["v2", "v1"], ["v1", "v2"]) == ["v2", "v1"]
    for ranking in (["v1", "v1"], ["v1"], ["v1", "other"]):
        with pytest.raises(ValueError):
            validate_ranking(ranking, ["v1", "v2"])


def test_attestation_and_sdk_request_with_usage(tmp_path, monkeypatch):
    from vrscore.benchmarking import llm_ranking

    calls = []

    class Usage:
        def model_dump(self, **kwargs):
            return {"prompt_tokens": 20, "completion_tokens": 4, "cost": 0.001}

    class Client:
        def __init__(self, settings):
            self.sdk = SimpleNamespace(chat=SimpleNamespace(send=self.send))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def send(self, **kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(
                content='{"vacancy_ids":["v2","v1"]}', reasoning="Short rationale"
            )
            return SimpleNamespace(
                usage=Usage(),
                model="example",
                id="gen-1",
                choices=[SimpleNamespace(finish_reason="stop", message=message)],
            )

    monkeypatch.setattr(llm_ranking, "OpenRouterClient", Client)
    source = tmp_path / "input.json"
    source.write_text(
        json.dumps({"resume": {"id": "r"}, "vacancies": [{"id": "v1"}, {"id": "v2"}]})
    )
    config = {
        "input_path": str(source),
        "output_path": str(tmp_path / "out.json"),
        "approved_input_sha256": None,
        "vacancy_count": 2,
        "llm": OmegaConf.to_container(OmegaConf.load("configs/llm.yaml")),
        "system_prompt": "rank",
        "user_prompt": "rank",
    }
    with pytest.raises(ValueError, match="attestation"):
        run(config)
    assert not calls
    config["approved_input_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    result = run(config)
    assert result["ranking"] == ["v2", "v1"]
    assert result["reasoning"] == "Short rationale"
    assert json.loads((tmp_path / "out.json").read_text())["usage"]["cost"] == 0.001
    assert calls[0]["model"] == config["llm"]["model"]
    assert calls[0]["provider"] == config["llm"]["provider"]
