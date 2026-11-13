"""Shared benchmark execution helpers."""

import time
from collections.abc import Callable

from .contracts import BenchmarkResult, BenchmarkSample, BenchmarkSpec


def run_benchmark(
    spec: BenchmarkSpec,
    operation: Callable[[], object],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> BenchmarkResult:
    """Run warmups then timed invocations, retaining individual failures.

    Warmup failures propagate because the operation is not ready to measure.
    Measured failures are recorded and later iterations continue. Timing uses a
    monotonic performance counter by default; inject ``clock`` for deterministic
    tests or a specialized measurement source.
    """
    for _ in range(spec.warmup_iterations):
        operation()

    samples = []
    for iteration in range(1, spec.iterations + 1):
        start = clock()
        try:
            operation()
        except Exception as error:
            duration = max(0.0, clock() - start)
            samples.append(
                BenchmarkSample(
                    iteration=iteration,
                    duration_seconds=duration,
                    success=False,
                    error_type=type(error).__name__,
                )
            )
        else:
            duration = max(0.0, clock() - start)
            samples.append(
                BenchmarkSample(
                    iteration=iteration, duration_seconds=duration, success=True
                )
            )
    return BenchmarkResult(name=spec.name, samples=tuple(samples))
