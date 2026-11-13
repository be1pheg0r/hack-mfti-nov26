"""Generic quality evaluation against explicitly supplied reference labels."""

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import Field

from .contracts import BenchmarkContract


class EvaluationCase(BenchmarkContract):
    """A locally held input and its reference output; no labels are inferred."""

    id: str = Field(min_length=1)
    input: Any
    expected: Any


class EvaluationCaseResult(BenchmarkContract):
    """Metrics or a sanitized failure for a single reference case."""

    id: str = Field(min_length=1)
    metrics: dict[str, float] = Field(default_factory=dict)
    error_type: str | None = None


class EvaluationResult(BenchmarkContract):
    """Case results; metric means use only cases with that metric available."""

    name: str = Field(min_length=1)
    cases: tuple[EvaluationCaseResult, ...] = Field(min_length=1)

    @property
    def mean_metrics(self) -> dict[str, float]:
        """Calculate per-metric means over successfully scored cases."""
        values: dict[str, list[float]] = defaultdict(list)
        for case in self.cases:
            for name, value in case.metrics.items():
                values[name].append(value)
        return {name: sum(items) / len(items) for name, items in values.items()}


def evaluate(
    name: str,
    cases: Sequence[EvaluationCase],
    predict: Callable[[Any], Any],
    score: Callable[[Any, Any], dict[str, float]],
) -> EvaluationResult:
    """Score arbitrary module or pipeline outputs against reference cases.

    Failed cases are recorded separately and excluded from metric means. The
    scorer defines metric semantics; neither predictions nor exceptions are saved.
    """
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Evaluation case IDs must be unique")
    results = []
    for case in cases:
        try:
            metrics = score(case.expected, predict(case.input))
            if not metrics or any(
                not key or not math.isfinite(value) for key, value in metrics.items()
            ):
                raise ValueError("Metrics must be nonempty and finite")
            results.append(EvaluationCaseResult(id=case.id, metrics=metrics))
        except Exception as error:
            results.append(
                EvaluationCaseResult(id=case.id, error_type=type(error).__name__)
            )
    return EvaluationResult(name=name, cases=tuple(results))
