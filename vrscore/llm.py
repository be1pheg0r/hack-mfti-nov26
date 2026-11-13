"""OpenRouter SDK adapter. Only explicitly public data may leave the system."""

import os
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv
from openrouter import OpenRouter
from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T", bound=BaseModel)


class ProviderConfig(BaseModel):
    """OpenRouter routing policy for every request."""

    model_config = ConfigDict(extra="forbid")

    only: list[str] = Field(min_length=1)
    allow_fallbacks: bool
    require_parameters: bool


class ExtractionPrompts(BaseModel):
    """Instructions used by extraction demonstrations and future consumers."""

    model_config = ConfigDict(extra="forbid")

    system: str = Field(min_length=1)
    resume: str = Field(min_length=1)
    vacancy: str = Field(min_length=1)
    user_intro: str = Field(min_length=1)


class PromptConfig(BaseModel):
    """Named prompts loaded from configuration, not embedded in clients."""

    model_config = ConfigDict(extra="forbid")

    extraction: ExtractionPrompts
    smoke: str = Field(min_length=1)


class LLMConfig(BaseModel):
    """Settings for an OpenRouter chat completion."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    temperature: float = Field(ge=0, le=2)
    max_tokens: int = Field(gt=0)
    timeout_ms: int = Field(gt=0)
    api_key_env: str = Field(min_length=1)
    dotenv_path: Path
    schema_strict: bool
    reasoning_enabled: bool | None
    provider: ProviderConfig
    prompts: PromptConfig


class OpenRouterClient:
    """Synchronous client; callers must attest that request data is public."""

    def __init__(self, config: LLMConfig, *, sdk: OpenRouter | None = None) -> None:
        self.config = config
        if sdk is None:
            dotenv_path = config.dotenv_path
            if not dotenv_path.is_absolute():
                dotenv_path = Path(__file__).resolve().parents[1] / dotenv_path
            load_dotenv(dotenv_path, override=False)
            key = os.getenv(config.api_key_env)
            if not key:
                raise ValueError(f"{config.api_key_env} is not configured")
            sdk = OpenRouter(api_key=key, timeout_ms=config.timeout_ms)
        self.sdk = sdk

    def structured(
        self,
        messages: list[dict[str, Any]],
        output_type: type[T],
        *,
        public_data: bool = False,
    ) -> T:
        """Request and locally validate JSON; never send data without attestation."""
        content = self._send(
            messages,
            public_data=public_data,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": output_type.__name__,
                    "strict": self.config.schema_strict,
                    "schema": output_type.model_json_schema(),
                },
            },
        )
        return output_type.model_validate_json(content)

    def text(self, messages: list[dict[str, Any]], *, public_data: bool = False) -> str:
        """Request a plain-text response for explicitly public messages."""
        return self._send(messages, public_data=public_data)

    def _send(
        self, messages: list[dict[str, Any]], *, public_data: bool, **options: Any
    ) -> str:
        if public_data is not True:
            raise ValueError("Only public, non-confidential data may be sent")
        response = self.sdk.chat.send(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=False,
            reasoning=(
                {"enabled": self.config.reasoning_enabled}
                if self.config.reasoning_enabled is not None
                else None
            ),
            provider=self.config.provider.model_dump(),
            **options,
        )
        if not response.choices:
            raise ValueError("OpenRouter returned no choices")
        choice = response.choices[0]
        if choice.finish_reason != "stop" or not isinstance(
            choice.message.content, str
        ):
            raise ValueError("OpenRouter returned an incomplete or non-text response")
        return choice.message.content

    def close(self) -> None:
        """Close the underlying SDK client."""
        self.sdk.__exit__(None, None, None)

    def __enter__(self) -> "OpenRouterClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
