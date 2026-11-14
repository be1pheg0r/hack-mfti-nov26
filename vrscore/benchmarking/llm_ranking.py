"""Rank an explicitly approved resume and vacancy set using configured LLM."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

from vrscore.llm import LLMConfig, OpenRouterClient


class Ranking(BaseModel):
    """IDs in decreasing order of estimated suitability."""

    vacancy_ids: list[str]


def validate_ranking(ranking: list[str], expected: list[str]) -> list[str]:
    """Reject incomplete, duplicated, or fabricated IDs."""
    if len(expected) != len(set(expected)) or len(ranking) != len(expected):
        raise ValueError("Expected unique IDs and one ranking entry per vacancy")
    if len(set(ranking)) != len(ranking) or set(ranking) != set(expected):
        raise ValueError("Ranking must contain exactly the supplied vacancy IDs")
    return ranking


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Send reviewed input once and record provider-reported usage and result."""
    data = Path(config["input_path"]).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if (
        not config.get("approved_input_sha256")
        or digest != config["approved_input_sha256"]
    ):
        raise ValueError("External-transfer attestation must match the input SHA-256")
    source = json.loads(data)
    resume, vacancies = source["resume"], source["vacancies"]
    if not isinstance(resume, dict) or not isinstance(vacancies, list):
        raise ValueError("Input must contain a resume object and vacancies array")
    if len(vacancies) != config["vacancy_count"]:
        raise ValueError("Unexpected vacancy count")
    ids = [v["id"] for v in vacancies]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Vacancy IDs must be nonempty and unique")
    settings = LLMConfig.model_validate(config["llm"])
    messages = [
        {"role": "system", "content": config["system_prompt"]},
        {
            "role": "user",
            "content": config["user_prompt"]
            + "\n"
            + json.dumps(
                {"resume": resume, "vacancies": vacancies}, ensure_ascii=False
            ),
        },
    ]
    with OpenRouterClient(settings) as client:
        response = client.sdk.chat.send(
            model=settings.model,
            messages=messages,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            stream=False,
            reasoning={"enabled": settings.reasoning_enabled}
            if settings.reasoning_enabled is not None
            else None,
            provider=settings.provider.model_dump(),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "Ranking",
                    "strict": settings.schema_strict,
                    "schema": Ranking.model_json_schema(),
                },
            },
        )
    usage = (
        response.usage.model_dump(mode="json", exclude_unset=True)
        if response.usage
        else None
    )
    reasoning = (
        getattr(response.choices[0].message, "reasoning", None)
        if response.choices
        else None
    )
    result = {
        "reasoning": reasoning if isinstance(reasoning, str) else None,
        "input_sha256": digest,
        "resume_id": resume.get("id"),
        "model": response.model,
        "request_id": response.id,
        "usage": usage,
        "ranking": None,
        "error": None,
    }
    try:
        if not response.choices or response.choices[0].finish_reason != "stop":
            raise ValueError("Response missing or incomplete")
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise ValueError("Response is not text")
        result["ranking"] = validate_ranking(
            Ranking.model_validate_json(content).vacancy_ids, ids
        )
    except (ValueError, TypeError) as exc:
        result["error"] = str(exc)
    output = Path(config["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if result["error"]:
        raise ValueError(f"Ranking failed; usage saved to {output}: {result['error']}")
    return result


@hydra.main(version_base=None, config_path="../../configs", config_name="llm_ranking")
def main(config: DictConfig) -> None:
    """Run from the project root with Hydra overrides."""
    result = run(OmegaConf.to_container(config, resolve=True))
    print(
        json.dumps(
            {
                "model": result["model"],
                "usage": result["usage"],
                "ranked": len(result["ranking"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
