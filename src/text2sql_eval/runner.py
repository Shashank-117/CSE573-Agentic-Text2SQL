"""The evaluation runner: one record per (question, system, run), always.

Stages per question:
  manifest entry -> gold-free QuestionInput -> system.predict()
  -> read-only validation + isolated execution (TEMP views if any)
  -> official evaluator -> record -> database-unchanged check.

Every included question produces exactly one record, including failures,
so unsupported / errors / timeouts stay in the EX denominator.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .contracts import Budget, PredictionResult, QuestionInput, Status, Text2SQLSystem, budget_violations
from .manifest import ManifestEntry, to_question_input
from .sandbox import ExecResult, database_fingerprint, run_query
from .view_composition import compose_with_views, views_from_artifacts

_MAX_LOGGED_ERROR = 2_000


class DatabaseModifiedError(RuntimeError):
    """Raised if a database file changed during evaluation. Aborts the run."""


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    system_name: str
    benchmark: str
    manifest: str
    budget: Budget = field(default_factory=Budget)
    repeat_index: int = 0
    seed: int | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _truncate(text: str | None) -> str | None:
    if text is None or len(text) <= _MAX_LOGGED_ERROR:
        return text
    return text[:_MAX_LOGGED_ERROR] + "...[truncated]"


def _safe_predict(system: Text2SQLSystem, question: QuestionInput, budget: Budget) -> PredictionResult:
    started = time.monotonic()
    try:
        prediction = system.predict(question, budget)
    except Exception as exc:  # a crashing system must not crash the run
        return PredictionResult(
            question_id=question.question_id, db_id=question.db_id, system_name=system.name,
            generated_sql=None, status=Status.SYSTEM_ERROR,
            wall_time_s=time.monotonic() - started,
            error_message=f"{type(exc).__name__}: {exc}",
        )
    if not prediction.wall_time_s:
        prediction.wall_time_s = time.monotonic() - started
    return prediction


def _system_failure_status(prediction: PredictionResult) -> str | None:
    """Status if the system itself did not produce runnable SQL, else None."""
    if prediction.status not in ("success", Status.EXECUTED):
        return prediction.status if prediction.status in Status.ALL else Status.SYSTEM_ERROR
    if not prediction.generated_sql or not prediction.generated_sql.strip():
        return Status.PARSE_ERROR
    return None


def _difficulty(evaluator: Any, question: QuestionInput) -> str | None:
    """Benchmark difficulty label, computed after prediction (it is derived from gold)."""
    classify = getattr(evaluator, "difficulty", None)
    return classify(question.question_id, question.db_path) if classify else None


def _execute(question: QuestionInput, prediction: PredictionResult, budget: Budget) -> tuple[ExecResult, str | None]:
    """Run in the sandbox. Returns the execution result and the scored SQL."""
    try:
        views = views_from_artifacts(prediction.intermediate_artifacts)
    except ValueError as exc:
        return ExecResult(Status.VIEW_ERROR, error=str(exc)), None

    with_views = run_query(question.db_path, prediction.generated_sql, views,
                           budget.sql_timeout_s, budget.max_result_rows)
    if not with_views.ok or not views:
        return with_views, prediction.generated_sql if with_views.ok else None

    # Views present: score the equivalent standalone query (views as CTEs).
    scored_sql = compose_with_views(prediction.generated_sql, views)
    standalone = run_query(question.db_path, scored_sql, (), budget.sql_timeout_s, budget.max_result_rows)
    if not standalone.ok:
        return ExecResult(standalone.status, error=f"CTE form failed: {standalone.error}"), None
    return standalone, scored_sql


def evaluate_one(
    system: Text2SQLSystem,
    entry: ManifestEntry,
    evaluator: Any,
    config: RunConfig,
) -> dict[str, Any]:
    question = to_question_input(entry)
    fingerprint = database_fingerprint(question.db_path)
    started = time.monotonic()

    prediction = _safe_predict(system, question, config.budget)
    status = _system_failure_status(prediction)
    execution = ExecResult(status or Status.EXECUTED)
    scored_sql: str | None = None
    correct, evaluator_version, eval_detail = False, getattr(evaluator, "version", None), None

    if status is None:
        execution, scored_sql = _execute(question, prediction, config.budget)
        status = execution.status

    if status == Status.EXECUTED and scored_sql is not None:
        try:
            outcome = evaluator.evaluate(
                question.question_id, question.db_path, scored_sql,
                columns=execution.columns, rows=execution.rows,
            )
            correct, evaluator_version, eval_detail = outcome.correct, outcome.evaluator, outcome.detail
        except Exception as exc:
            status, eval_detail = Status.EVAL_ERROR, f"{type(exc).__name__}: {exc}"

    difficulty = _difficulty(evaluator, question)
    violations = budget_violations(prediction, config.budget)

    if database_fingerprint(question.db_path) != fingerprint:
        raise DatabaseModifiedError(f"{question.db_path} changed while evaluating {question.question_id}")

    return {
        "run_id": config.run_id,
        "repeat_index": config.repeat_index,
        "benchmark": entry.benchmark,
        **prediction.to_dict(),
        "system_status": prediction.status,
        "status": status,
        "correct": bool(correct and status == Status.EXECUTED),
        "within_budget": not violations,
        "budget_violations": violations,
        "difficulty": difficulty,
        "scored_sql": scored_sql,
        "execution_error": _truncate(execution.error),
        "execution_time_s": round(execution.elapsed_s, 4),
        "result_row_count": len(execution.rows),
        "result_truncated": execution.truncated,
        "evaluator": evaluator_version,
        "eval_detail": _truncate(eval_detail),
        "error_message": _truncate(prediction.error_message),
        "harness_wall_time_s": round(time.monotonic() - started, 4),
    }


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as handle:
        return {json.loads(line)["question_id"] for line in handle if line.strip()}


def run(
    system: Text2SQLSystem,
    entries: Iterable[ManifestEntry],
    evaluator: Any,
    config: RunConfig,
    out_dir: Path,
    resume: bool = False,
    progress_every: int = 50,
) -> Path:
    """Evaluate ``entries`` and append one JSON line per question to runs.jsonl."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    done = _completed_ids(runs_path) if resume else set()
    mode = "a" if resume else "w"

    todo = [e for e in entries if e.question_id not in done]
    with runs_path.open(mode, encoding="utf-8") as handle:
        for index, entry in enumerate(todo, start=1):
            record = evaluate_one(system, entry, evaluator, config)
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            if progress_every and (index % progress_every == 0 or index == len(todo)):
                print(f"  [{config.system_name}] {index}/{len(todo)}", flush=True)
    return runs_path
