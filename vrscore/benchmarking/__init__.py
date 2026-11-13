"""Reusable benchmark contracts and execution utilities."""

from .contracts import BenchmarkResult, BenchmarkSample, BenchmarkSpec
from .evaluation import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationResult,
    evaluate,
    ndcg_at_k,
)
from .runner import run_benchmark

__all__ = [
    "BenchmarkResult",
    "BenchmarkSample",
    "BenchmarkSpec",
    "EvaluationCase",
    "EvaluationCaseResult",
    "EvaluationResult",
    "evaluate",
    "ndcg_at_k",
    "run_benchmark",
]
