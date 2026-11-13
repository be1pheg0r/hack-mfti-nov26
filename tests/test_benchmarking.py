"""Reusable benchmark contracts and runner behavior."""

import pytest
from pydantic import ValidationError

from vrscore.benchmarking import BenchmarkSpec, run_benchmark


def test_runner_excludes_warmups_records_failures_and_summarizes():
    """Capture measured calls only and aggregate successful durations."""
    spec = BenchmarkSpec(name="parser", iterations=3, warmup_iterations=1)
    calls = 0
    ticks = iter((1.0, 1.2, 2.0, 2.1, 3.0, 3.3))

    def operation():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("fixture failure")

    result = run_benchmark(spec, operation, clock=lambda: next(ticks))

    assert calls == 4
    assert len(result.samples) == 3
    assert [sample.iteration for sample in result.samples] == [1, 2, 3]
    assert [sample.success for sample in result.samples] == [True, False, True]
    assert result.samples[1].error_type == "RuntimeError"
    assert result.successful_iterations == 2
    assert result.failed_iterations == 1
    assert result.mean_seconds == pytest.approx(0.25)
    assert result.median_seconds == pytest.approx(0.25)
    assert result.p95_seconds == pytest.approx(0.3)


def test_benchmark_contract_rejects_invalid_iteration_counts():
    """Require a nonempty measured run and nonnegative warmup count."""
    with pytest.raises(ValidationError):
        BenchmarkSpec(name="", iterations=0)
    with pytest.raises(ValidationError):
        BenchmarkSpec(name="parser", iterations=1, warmup_iterations=-1)


def test_evaluation_scores_cases_and_does_not_leak_exception_text():
    """Evaluate predictions against labels without retaining sensitive errors."""
    from vrscore.benchmarking import EvaluationCase, evaluate

    cases = [
        EvaluationCase(id="a", input=1, expected=2),
        EvaluationCase(id="b", input=2, expected=4),
    ]

    def predict(value):
        if value == 2:
            raise RuntimeError("private resume content")
        return value * 2

    result = evaluate(
        "feature extraction",
        cases,
        predict,
        lambda expected, actual: {"exact": float(expected == actual)},
    )
    assert result.cases[0].metrics == {"exact": 1.0}
    assert result.cases[1].error_type == "RuntimeError"
    assert result.mean_metrics == {"exact": 1.0}
    assert "private resume content" not in result.model_dump_json()
