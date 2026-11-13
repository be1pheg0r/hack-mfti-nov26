"""Evidence-preserving strict first-gate filters for parsed vacancy/resume data."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict

from omegaconf import OmegaConf

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "first_gate.yaml"


def _entities(document: dict, category: str) -> list[dict]:
    """Return valid parsed entities for a category without altering evidence."""
    entities = document.get("entities", [])
    if not isinstance(entities, list):
        return []
    return [
        entity
        for entity in entities
        if isinstance(entity, dict)
        and entity.get("category") == category
        and "value" in entity
    ]


def _fact(entity: dict) -> dict:
    """Keep the complete source entity available in the decision result."""
    return dict(entity)


def _config(config: dict | str | Path | None) -> dict:
    defaults = OmegaConf.load(DEFAULT_CONFIG)
    if config is None:
        source = defaults
    elif isinstance(config, (str, Path)):
        source = OmegaConf.merge(defaults, OmegaConf.load(config))
    else:
        source = OmegaConf.merge(defaults, OmegaConf.create(config))
    return OmegaConf.to_container(source, resolve=True)  # type: ignore[return-value]


def _decision(filter_name: str, status: str, **details: Any) -> dict:
    return {"filter": filter_name, "status": status, **details}


def _evaluate_experience(vacancy: dict, resume: dict, category: str) -> dict:
    requirements = _entities(vacancy, category)
    candidates = _entities(resume, category)
    if not requirements:
        return _decision(
            "experience",
            "unknown",
            explanation="Vacancy experience requirement is missing.",
        )
    if not candidates:
        return _decision(
            "experience",
            "unknown",
            requirement=[_fact(e) for e in requirements],
            candidate=[],
            explanation="Resume experience is not stated; it is unknown, not zero.",
        )
    try:
        threshold_values = [float(item["value"]) for item in requirements]
        values = [
            float(item["value"])
            for item in candidates
            if item.get("polarity") != "absent"
        ]
        if not values or any(
            not math.isfinite(value) or value < 0 for value in threshold_values + values
        ):
            raise ValueError("Unconfirmed or invalid experience")
    except (TypeError, ValueError, OverflowError):
        return _decision(
            "experience",
            "unknown",
            requirement=[_fact(e) for e in requirements],
            candidate=[_fact(e) for e in candidates],
            explanation="Experience is invalid or not confirmed.",
        )
    threshold = max(threshold_values)
    candidate_value = max(values)
    difference = candidate_value - threshold
    caveat = any(
        "пересеч" in str(entity.get("context", "")).casefold()
        or "overlap" in str(entity.get("context", "")).casefold()
        for entity in candidates
    )
    explanation = (
        f"Resume experience {candidate_value:g} years minus requirement "
        f"{threshold:g} years = {difference:g}."
    )
    if caveat:
        explanation += (
            " Summed periods may overlap; the source context explicitly says "
            "overlaps/duplicates were not checked."
        )
    return _decision(
        "experience",
        "pass" if difference >= 0 else "fail",
        difference=difference,
        requirement=[_fact(e) for e in requirements],
        candidate=[_fact(e) for e in candidates],
        explanation=explanation,
    )


def _evaluate_education(
    vacancy: dict, resume: dict, category: str, config: dict
) -> dict:
    requirements = _entities(vacancy, category)
    candidates = _entities(resume, category)
    levels = list(config["education"]["levels"])
    aliases = config["education"].get("aliases", {})
    ranks = {level: index for index, level in enumerate(levels)}
    ranks.update(
        {alias: ranks[target] for alias, target in aliases.items() if target in ranks}
    )
    if not requirements:
        return _decision(
            "education",
            "unknown",
            explanation="Vacancy education requirement is missing.",
        )
    if not candidates:
        return _decision(
            "education",
            "unknown",
            requirement=[_fact(e) for e in requirements],
            candidate=[],
            explanation="Resume education is not stated; it is unknown.",
        )
    incomplete_markers = tuple(
        marker.casefold()
        for marker in config["education"].get("incomplete_markers", [])
    )
    complete = []
    for entity in candidates:
        context = str(entity.get("context") or "").casefold()
        value = entity.get("value")
        if any(marker in context for marker in incomplete_markers) and not config[
            "education"
        ].get("allow_incomplete", False):
            continue
        elif entity.get("polarity") == "absent" or value not in ranks:
            continue
        else:
            complete.append(entity)
    if not complete:
        return _decision(
            "education",
            "unknown",
            requirement=[_fact(e) for e in requirements],
            candidate=[_fact(e) for e in candidates],
            explanation=(
                "No confirmed completed education level; "
                "incomplete/unclear evidence remains unknown."
            ),
        )
    required_levels = [entity.get("value") for entity in requirements]
    if any(level not in ranks for level in required_levels):
        return _decision(
            "education",
            "unknown",
            requirement=[_fact(e) for e in requirements],
            candidate=[_fact(e) for e in candidates],
            explanation=(
                "Vacancy contains an education level not present in the "
                "configured ordering."
            ),
        )
    alternative_groups: dict[str, list[int]] = {}
    required_ranks = []
    for entity in requirements:
        quote = entity.get("evidence")
        if isinstance(quote, str) and " или " in quote.casefold():
            alternative_groups.setdefault(quote, []).append(ranks[entity["value"]])
        else:
            required_ranks.append(ranks[entity["value"]])
    required_rank = max(
        required_ranks + [min(group) for group in alternative_groups.values()]
    )
    candidate_rank = max(ranks[entity["value"]] for entity in complete)
    difference = candidate_rank - required_rank
    if difference < 0 and any(entity.get("value") == "higher" for entity in complete):
        return _decision(
            "education",
            "unknown",
            requirement=[_fact(e) for e in requirements],
            candidate=[_fact(e) for e in candidates],
            explanation="Higher education is confirmed, but its degree is unspecified.",
        )
    return _decision(
        "education",
        "pass" if difference >= 0 else "fail",
        difference=difference,
        requirement=[_fact(e) for e in requirements],
        candidate=[_fact(e) for e in candidates],
        explanation=(
            "Highest confirmed completed education rank difference "
            f"(resume - vacancy) is {difference}."
        ),
    )


def _evaluate_work_format(
    vacancy: dict, resume: dict, category: str, config: dict
) -> dict:
    requirements = _entities(vacancy, category)
    all_candidates = _entities(resume, category)
    candidates = [
        entity for entity in all_candidates if entity.get("polarity") != "absent"
    ]
    if not requirements:
        return _decision(
            "work_format", "unknown", explanation="Vacancy work format is missing."
        )
    if not candidates:
        required_values = {e["value"] for e in requirements}
        denied_values = {e["value"] for e in all_candidates}
        if required_values and required_values <= denied_values:
            return _decision(
                "work_format",
                "fail",
                requirement=[_fact(e) for e in requirements],
                candidate=[_fact(e) for e in all_candidates],
                explanation="Resume explicitly excludes the required work format.",
            )
        return _decision(
            "work_format",
            "unknown",
            requirement=[_fact(e) for e in requirements],
            candidate=[_fact(e) for e in all_candidates],
            explanation="Resume work format is not confirmed; it is unknown.",
        )
    compatibility = config["work_format"]["compatibility"]
    required_values = {e["value"] for e in requirements}
    candidate_values = {e["value"] for e in candidates}
    if any(value not in compatibility for value in required_values):
        status = "unknown"
        explanation = "Vacancy work format is not configured."
    else:
        accepted = set().union(
            *(set(compatibility[value]) for value in required_values)
        )
        matched = candidate_values & accepted
        status = "pass" if matched else "fail"
        explanation = (
            f"Compatible candidate format(s): {', '.join(sorted(matched))}."
            if matched
            else "No candidate format is compatible with the vacancy format."
        )
    return _decision(
        "work_format",
        status,
        requirement=[_fact(e) for e in requirements],
        candidate=[_fact(e) for e in candidates],
        explanation=explanation,
    )


class RuleResult(TypedDict):
    """Shared result envelope returned by every gate rule."""

    filter: str
    status: Literal["pass", "fail", "unknown"]
    explanation: str


@dataclass(frozen=True)
class RuleContext:
    """Shared read-only inputs available to every rule in a gate run."""

    vacancy: dict
    resume: dict
    config: dict


class Rule(Protocol):
    """Extension contract for one independently testable gate rule."""

    name: str

    def evaluate(self, context: RuleContext) -> RuleResult:
        """Evaluate this rule using the shared context."""
        ...


class ExperienceRule:
    """Strict experience threshold using resume-minus-vacancy years."""

    name = "experience"

    def evaluate(self, context: RuleContext) -> RuleResult:
        return _evaluate_experience(
            context.vacancy,
            context.resume,
            context.config["filters"]["categories"][self.name],
        )  # type: ignore[return-value]


class EducationRule:
    """Strict education-level comparison, preserving incomplete evidence as unknown."""

    name = "education"

    def evaluate(self, context: RuleContext) -> RuleResult:
        return _evaluate_education(
            context.vacancy,
            context.resume,
            context.config["filters"]["categories"][self.name],
            context.config,
        )  # type: ignore[return-value]


class WorkFormatRule:
    """Strict work-format compatibility rule."""

    name = "work_format"

    def evaluate(self, context: RuleContext) -> RuleResult:
        return _evaluate_work_format(
            context.vacancy,
            context.resume,
            context.config["filters"]["categories"][self.name],
            context.config,
        )  # type: ignore[return-value]


class Gate:
    """Run rule objects in order and stop at the first non-pass result."""

    def __init__(self, rules: list[Rule]):
        self.rules = list(rules)

    def evaluate(self, context: RuleContext) -> dict:
        decisions = []
        for rule in self.rules:
            result = rule.evaluate(context)
            if result.get("status") not in {"pass", "fail", "unknown"}:
                raise ValueError(f"Rule {rule.name} returned an invalid status")
            decisions.append(result)
            if result["status"] != "pass":
                return {"status": result["status"], "decisions": decisions}
        return {"status": "pass", "decisions": decisions}


_BUILTIN_RULES: dict[str, type[Rule]] = {
    "experience": ExperienceRule,
    "education": EducationRule,
    "work_format": WorkFormatRule,
}


def evaluate_first_gate(
    vacancy: dict,
    resume: dict,
    config: dict | str | Path | None = None,
    *,
    rule_registry: dict[str, type[Rule]] | None = None,
) -> dict:
    """Evaluate strict filters in configured order; stop on first non-pass.

    The standalone API accepts parsed JSON documents and preserves each source
    entity verbatim in decisions. Custom Rule classes can extend the registry;
    the Gate engine remains unchanged.
    """
    if vacancy.get("kind") not in (None, "vacancy"):
        raise ValueError("vacancy input must have kind='vacancy'")
    if resume.get("kind") not in (None, "resume"):
        raise ValueError("resume input must have kind='resume'")
    settings = _config(config)
    registry = {**_BUILTIN_RULES, **(rule_registry or {})}
    try:
        rules = [registry[name]() for name in settings["filters"]["order"]]
    except KeyError as exc:
        raise ValueError(f"Unknown first-gate rule: {exc.args[0]}") from exc
    context = RuleContext(vacancy=vacancy, resume=resume, config=settings)
    return Gate(rules).evaluate(context)
