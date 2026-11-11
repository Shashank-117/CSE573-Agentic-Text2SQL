"""Wrappers around each benchmark's official evaluator."""

from __future__ import annotations

from .spider1 import Spider1Evaluator
from .spider2_lite import Spider2LiteEvaluator

BENCHMARK_EVALUATORS = {
    "spider1": Spider1Evaluator,
    "spider2_lite": Spider2LiteEvaluator,
}


def get_evaluator(benchmark: str, **options):
    """Build the official evaluator. Options (e.g. keep_distinct) are Spider 1 only."""
    if benchmark not in BENCHMARK_EVALUATORS:
        raise ValueError(f"no official evaluator registered for {benchmark!r}")
    return BENCHMARK_EVALUATORS[benchmark](**options)


__all__ = ["Spider1Evaluator", "Spider2LiteEvaluator", "get_evaluator", "BENCHMARK_EVALUATORS"]
