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

from .contracts import Status

SUMMARY_FIELDS = (
    "run_id", "system_name", "benchmark", "repeat_index", "evaluated", "correct", "execution_accuracy",
    "executed", "execution_success_rate", "supported", "coverage", "accuracy_when_supported",
    "model_calls", "retries", "input_tokens", "output_tokens", "max_input_tokens_single_call",
    "mean_wall_time_s", "p50_wall_time_s", "p90_wall_time_s", "peak_gpu_memory_mb",
    *(f"n_{s}" for s in sorted(Status.ALL)),
)

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

    return {
        "run_id": first.get("run_id"),
        "system_name": first.get("system_name"),
        "benchmark": first.get("benchmark"),
        "repeat_index": first.get("repeat_index"),
        "evaluated": total,
        "correct": correct,
        "execution_accuracy": _ratio(correct, total),
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
    return {
        "summary": _write_csv(out_dir / "results_summary.csv", SUMMARY_FIELDS, summaries),
        "errors": _write_csv(out_dir / "errors.csv", ERROR_FIELDS, failures),
    }
