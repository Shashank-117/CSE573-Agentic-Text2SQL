"""Cross-run analysis: repeated-run aggregation and paired system comparison.

- ``aggregate_repeats``: mean / sample std / min / max of EX over repeated
  runs of the same system on the same questions.
- ``compare_runs``: question-by-question comparison of two runs on the same
  questions, with an exact McNemar test and a paired bootstrap confidence
  interval for the EX difference.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .metrics import summarize, within_budget

AGGREGATE_METRICS = (
    "execution_accuracy", "execution_accuracy_within_budget", "execution_success_rate",
    "mean_wall_time_s", "input_tokens", "output_tokens", "model_calls",
)

PAIRED_FIELDS = ("question_id", "db_id", "difficulty", "outcome", "a_correct", "b_correct",
                 "a_status", "b_status", "a_sql", "b_sql")


class RunMismatchError(ValueError):
    """Runs cannot be combined (different benchmark or question set)."""


def _by_question(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for record in records:
        if record["question_id"] in rows:
            raise RunMismatchError(f"duplicate question {record['question_id']} in one run")
        rows[record["question_id"]] = record
    return rows


def _evaluator(records: Iterable[dict[str, Any]]) -> str | None:
    versions = {r.get("evaluator") for r in records if r.get("evaluator")}
    if len(versions) > 1:
        raise RunMismatchError(f"one run scored by several evaluators: {sorted(versions)}")
    return versions.pop() if versions else None


def _check_same_questions(name_a: str, a: dict, name_b: str, b: dict) -> None:
    if set(a) != set(b):
        only_a, only_b = len(set(a) - set(b)), len(set(b) - set(a))
        raise RunMismatchError(
            f"{name_a} and {name_b} cover different questions "
            f"({only_a} only in the first, {only_b} only in the second)"
        )
    benchmarks = {r["benchmark"] for r in (*a.values(), *b.values())}
    if len(benchmarks) != 1:
        raise RunMismatchError(f"runs mix benchmarks: {sorted(benchmarks)}")


# ------------------------------------------------------------ repeated runs

def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def aggregate_repeats(runs: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Aggregate repeated runs, grouped by (system, benchmark, evaluator version).

    ``runs`` maps a run id to its records. Runs in one group must cover the
    same questions, otherwise their EX values are not comparable.
    """
    groups: dict[tuple[str, str, str], list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for run_id, records in runs.items():
        if not records:
            raise RunMismatchError(f"run {run_id} has no records")
        first = records[0]
        key = (first["system_name"], first["benchmark"], _evaluator(records) or "")
        groups[key].append((run_id, records))

    rows = []
    for (system, benchmark, evaluator), members in sorted(groups.items()):
        reference_id, reference = members[0][0], _by_question(members[0][1])
        for run_id, records in members[1:]:
            _check_same_questions(reference_id, reference, run_id, _by_question(records))
        summaries = [summarize(records) for _, records in members]
        row: dict[str, Any] = {
            "system_name": system,
            "benchmark": benchmark,
            "evaluator": evaluator or None,
            "n_repeats": len(members),
            "n_questions": len(reference),
            "run_ids": ";".join(run_id for run_id, _ in members),
            "repeat_indices": ";".join(str(s.get("repeat_index")) for s in summaries),
            "correct_per_repeat": ";".join(str(s["correct"]) for s in summaries),
        }
        for metric in AGGREGATE_METRICS:
            values = [float(s[metric]) for s in summaries if s.get(metric) is not None]
            for stat, value in _stats(values).items():
                row[f"{metric}_{stat}"] = value
        rows.append(row)
    return rows


# ------------------------------------------------------- paired comparison

def mcnemar_exact_p(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, iterations: int = 10_000, seed: int = 0, level: float = 0.95,
) -> tuple[float, float]:
    """Percentile CI for mean(a) - mean(b), resampling questions with replacement."""
    rng = np.random.default_rng(seed)
    diffs = a.astype(float) - b.astype(float)
    samples = rng.integers(0, len(diffs), size=(iterations, len(diffs)))
    means = diffs[samples].mean(axis=1)
    alpha = (1 - level) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


@dataclass(frozen=True)
class Comparison:
    summary: dict[str, Any]
    paired_rows: list[dict[str, Any]]


def _outcome(a_ok: bool, b_ok: bool) -> str:
    return {(True, True): "both", (True, False): "only_a",
            (False, True): "only_b", (False, False): "neither"}[(a_ok, b_ok)]


def compare_runs(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
    within_budget_only: bool = False,
    allow_different_evaluators: bool = False,
    bootstrap_iterations: int = 10_000,
    seed: int = 0,
) -> Comparison:
    """Compare run A against run B on exactly the same questions.

    With ``within_budget_only`` a correct answer that broke the budget counts
    as wrong, which is the right setting for matched-budget comparisons.
    """
    a, b = _by_question(records_a), _by_question(records_b)
    name_a = records_a[0]["run_id"] if records_a else "A"
    name_b = records_b[0]["run_id"] if records_b else "B"
    _check_same_questions(name_a, a, name_b, b)
    eval_a, eval_b = _evaluator(records_a), _evaluator(records_b)
    if eval_a != eval_b and not allow_different_evaluators:
        raise RunMismatchError(f"runs were scored differently: {eval_a} vs {eval_b}")

    def ok(record: dict[str, Any]) -> bool:
        return bool(record["correct"]) and (within_budget(record) or not within_budget_only)

    rows = []
    for qid in sorted(a):
        ra, rb = a[qid], b[qid]
        rows.append({
            "question_id": qid, "db_id": ra.get("db_id"), "difficulty": ra.get("difficulty"),
            "outcome": _outcome(ok(ra), ok(rb)), "a_correct": ok(ra), "b_correct": ok(rb),
            "a_status": ra.get("status"), "b_status": rb.get("status"),
            "a_sql": ra.get("generated_sql"), "b_sql": rb.get("generated_sql"),
        })

    a_vec = np.array([r["a_correct"] for r in rows], dtype=bool)
    b_vec = np.array([r["b_correct"] for r in rows], dtype=bool)
    counts = {k: sum(1 for r in rows if r["outcome"] == k) for k in ("both", "only_a", "only_b", "neither")}
    low, high = paired_bootstrap_ci(a_vec, b_vec, bootstrap_iterations, seed) if rows else (0.0, 0.0)
    n = len(rows)
    summary = {
        "run_a": name_a, "system_a": records_a[0]["system_name"],
        "run_b": name_b, "system_b": records_b[0]["system_name"],
        "benchmark": records_a[0]["benchmark"], "n_questions": n,
        "evaluator_a": eval_a, "evaluator_b": eval_b,
        "within_budget_only": within_budget_only,
        "ex_a": float(a_vec.mean()) if n else 0.0,
        "ex_b": float(b_vec.mean()) if n else 0.0,
        "ex_diff_a_minus_b": float(a_vec.mean() - b_vec.mean()) if n else 0.0,
        "ci95_low": low, "ci95_high": high,
        **{f"n_{k}": v for k, v in counts.items()},
        "mcnemar_exact_p": mcnemar_exact_p(counts["only_a"], counts["only_b"]),
        "bootstrap_iterations": bootstrap_iterations, "bootstrap_seed": seed,
    }
    return Comparison(summary, rows)
