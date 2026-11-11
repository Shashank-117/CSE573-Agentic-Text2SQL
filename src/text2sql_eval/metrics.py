"""Aggregate per-question run records into summary tables.

EX = correct / evaluated, where "evaluated" counts every included
question (unsupported, errors and timeouts are failures, not exclusions).
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .contracts import Budget, PredictionResult, Status, budget_violations

SUMMARY_FIELDS = (
    "run_id", "system_name", "benchmark", "repeat_index", "evaluated", "correct", "execution_accuracy",
    "n_over_budget", "correct_within_budget", "execution_accuracy_within_budget",
    "executed", "execution_success_rate", "supported", "coverage", "accuracy_when_supported",
    "model_calls", "retries", "input_tokens", "output_tokens", "max_input_tokens_single_call",
    "mean_wall_time_s", "p50_wall_time_s", "p90_wall_time_s", "peak_gpu_memory_mb",
    *(f"n_{s}" for s in sorted(Status.ALL)),
)

DIFFICULTY_ORDER = ("easy", "medium", "hard", "extra", "unknown")
DIFFICULTY_FIELDS = ("run_id", "system_name", "benchmark", "repeat_index", "difficulty",
                     "evaluated", "correct", "execution_accuracy")

ERROR_FIELDS = (
    "run_id", "system_name", "benchmark", "question_id", "db_id", "status",
    "execution_error", "eval_detail", "error_message", "generated_sql",
)


def load_records(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def within_budget(record: dict[str, Any]) -> bool:
    """Harness budget verdict; recomputed with default limits for records that predate it."""
    if "within_budget" in record:
        return bool(record["within_budget"])
    probe = PredictionResult(
        question_id="", db_id="", system_name="", generated_sql=None,
        model_calls=int(record.get("model_calls") or 0),
        retry_count=int(record.get("retry_count") or 0),
        input_tokens=int(record.get("input_tokens") or 0),
    )
    return not budget_violations(probe, Budget())


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summary for one homogeneous group (same run, system, benchmark)."""
    rows = list(records)
    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    executed = sum(1 for r in rows if r["status"] == Status.EXECUTED)
    supported = sum(1 for r in rows if r["status"] != Status.UNSUPPORTED)
    statuses = Counter(r["status"] for r in rows)
    times = [float(r.get("wall_time_s") or 0.0) for r in rows]
    gpu = [r["peak_gpu_memory_mb"] for r in rows if r.get("peak_gpu_memory_mb") is not None]
    first = rows[0] if rows else {}
    in_budget = [within_budget(r) for r in rows]
    correct_in_budget = sum(1 for r, ok in zip(rows, in_budget) if ok and r["correct"])

    return {
        "run_id": first.get("run_id"),
        "system_name": first.get("system_name"),
        "benchmark": first.get("benchmark"),
        "repeat_index": first.get("repeat_index"),
        "evaluated": total,
        "correct": correct,
        "execution_accuracy": _ratio(correct, total),
        "n_over_budget": sum(1 for ok in in_budget if not ok),
        "correct_within_budget": correct_in_budget,
        "execution_accuracy_within_budget": _ratio(correct_in_budget, total),
        "executed": executed,
        "execution_success_rate": _ratio(executed, total),
        "supported": supported,
        "coverage": _ratio(supported, total),
        "accuracy_when_supported": _ratio(correct, supported),
        "model_calls": sum(int(r.get("model_calls") or 0) for r in rows),
        "retries": sum(int(r.get("retry_count") or 0) for r in rows),
        "input_tokens": sum(int(r.get("input_tokens") or 0) for r in rows),
        "output_tokens": sum(int(r.get("output_tokens") or 0) for r in rows),
        "max_input_tokens_single_call": max((int(r.get("max_input_tokens_single_call") or 0) for r in rows), default=0),
        "mean_wall_time_s": statistics.fmean(times) if times else None,
        "p50_wall_time_s": _percentile(times, 0.5),
        "p90_wall_time_s": _percentile(times, 0.9),
        "peak_gpu_memory_mb": max(gpu) if gpu else None,  # None = not measured
        **{f"n_{s}": statuses.get(s, 0) for s in sorted(Status.ALL)},
    }


def summarize_by_group(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (record["run_id"], record["system_name"], record["benchmark"], record.get("repeat_index"))
        groups[key].append(record)
    return [summarize(groups[key]) for key in sorted(groups, key=lambda k: tuple(str(x) for x in k))]


def summarize_by_difficulty(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """EX per benchmark difficulty level (Spider 1 easy/medium/hard/extra)."""
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("difficulty") is None:
            continue
        key = (record["run_id"], record["system_name"], record["benchmark"],
               record.get("repeat_index"), record["difficulty"])
        groups[key].append(record)

    def order(key: tuple) -> tuple:
        level = key[4]
        rank = DIFFICULTY_ORDER.index(level) if level in DIFFICULTY_ORDER else len(DIFFICULTY_ORDER)
        return (*(str(x) for x in key[:4]), rank)

    rows = []
    for key in sorted(groups, key=order):
        members = groups[key]
        correct = sum(1 for r in members if r["correct"])
        rows.append({
            "run_id": key[0], "system_name": key[1], "benchmark": key[2], "repeat_index": key[3],
            "difficulty": key[4], "evaluated": len(members), "correct": correct,
            "execution_accuracy": _ratio(correct, len(members)),
        })
    return rows


def _write_csv(path: Path, fields: Iterable[str], rows: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_reports(runs_path: Path, out_dir: Path | None = None) -> dict[str, Path]:
    """Write results_summary.csv and errors.csv next to runs.jsonl."""
    out_dir = Path(out_dir or Path(runs_path).parent)
    records = load_records(runs_path)
    summaries = summarize_by_group(records)
    failures = [r for r in records if not r["correct"]]
    reports = {
        "summary": _write_csv(out_dir / "results_summary.csv", SUMMARY_FIELDS, summaries),
        "errors": _write_csv(out_dir / "errors.csv", ERROR_FIELDS, failures),
    }
    by_difficulty = summarize_by_difficulty(records)
    if by_difficulty:
        reports["by_difficulty"] = _write_csv(out_dir / "results_by_difficulty.csv",
                                              DIFFICULTY_FIELDS, by_difficulty)
    return reports
