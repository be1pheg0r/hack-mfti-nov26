"""Exercise the OpenRouter SDK boundary without transmitting private data."""

import json
from pathlib import Path

import httpx
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from openrouter import OpenRouter
from pydantic import BaseModel, ValidationError

from vrscore.llm import LLMConfig, OpenRouterClient

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


class Answer(BaseModel):
    answer: str


def test_sdk_structured_response_and_privacy():
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "test",
                "created": 1,
                "model": "example/model",
                "object": "chat.completion",
                "system_fingerprint": None,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": '{"answer":"ok"}'},
                    }
                ],
            },
        )

    sdk = OpenRouter(
        api_key="test", client=httpx.Client(transport=httpx.MockTransport(handle))
    )
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        default = compose(config_name="llm")
        assert default.provider.only == ["wafer/us"]
        assert default.provider.allow_fallbacks is False
        config = compose(
            config_name="llm",
            overrides=[
                "model=example/model",
                "provider.only=[example/us]",
                "provider.allow_fallbacks=true",
                "schema_strict=false",
                "temperature=0.5",
                "max_tokens=64",
                "timeout_ms=12345",
                "reasoning_enabled=false",
                "prompts.extraction.system='Extract exact facts.'",
            ],
        )
    client = OpenRouterClient(
        LLMConfig.model_validate(OmegaConf.to_container(config, resolve=True)), sdk=sdk
    )
    assert client.config.prompts.extraction.system == "Extract exact facts."
    messages = [{"role": "user", "content": "public example"}]
    with pytest.raises(ValueError, match="public"):
        client.structured(messages, Answer)
    assert not requests
    assert client.structured(messages, Answer, public_data=True) == Answer(answer="ok")
    assert requests[0]["response_format"]["type"] == "json_schema"
    assert requests[0]["response_format"]["json_schema"]["strict"] is False
    assert requests[0]["provider"]["require_parameters"] is True
    assert client.text(messages, public_data=True) == '{"answer":"ok"}'
    assert "response_format" not in requests[1]
    for request in requests:
        assert request["provider"]["only"] == ["example/us"]
        assert request["provider"]["allow_fallbacks"] is True
        assert request["model"] == "example/model"
        assert request["temperature"] == 0.5
        assert request["max_tokens"] == 64
        assert request["reasoning"]["enabled"] is False
    with pytest.raises(ValidationError):
        Answer.model_validate_json('{"answer": 1}')
    client.close()


def test_configured_credentials_and_timeout(tmp_path, monkeypatch):
    """Use the configured dotenv path, environment variable, and SDK timeout."""
    from vrscore import llm

    env_file = tmp_path / "test.env"
    env_file.write_text("TEST_OPENROUTER_KEY=example-test-key\n")
    monkeypatch.delenv("TEST_OPENROUTER_KEY", raising=False)
    captured = {}
    monkeypatch.setattr(llm, "OpenRouter", lambda **kwargs: captured.update(kwargs))
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        config = compose(
            config_name="llm",
            overrides=[
                "api_key_env=TEST_OPENROUTER_KEY",
                f"dotenv_path={env_file}",
                "timeout_ms=12345",
            ],
        )
    OpenRouterClient(LLMConfig.model_validate(OmegaConf.to_container(config)))
    assert captured == {"api_key": "example-test-key", "timeout_ms": 12345}
    monkeypatch.delenv("TEST_OPENROUTER_KEY", raising=False)
