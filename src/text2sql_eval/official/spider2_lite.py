"""Spider 2.0-Lite (local SQLite subset) scoring via the official comparator.

The official ``evaluation_suite/evaluate.py`` cannot be imported directly:
at import time it redirects stdout/stderr to ``log.txt`` and imports
``google.cloud.bigquery``, and it looks for SQLite files in a directory
that differs from the released layout. We therefore load only its pure
comparison functions (unmodified source, extracted by name via ``ast``)
and record the file's SHA-256 so results are tied to an exact version.

Prediction results are round-tripped through CSV exactly as the official
"sql" mode does (query -> DataFrame -> CSV -> ``pd.read_csv``) so values
are compared with the same types as the gold CSV files.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from ..contracts import EvalOutcome
from ..paths import SPIDER2_EVAL_DIR

EVALUATOR_NAME = "spider2-lite/evaluation_suite"
_OFFICIAL_FUNCTIONS = (
    "load_gold_csv",
    "compare_pandas_table",
    "compare_multi_pandas_table",
    "resolve_gold_paths",
)


def _load_official_functions(source_path: Path) -> dict[str, Any]:
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in _OFFICIAL_FUNCTIONS]
    missing = set(_OFFICIAL_FUNCTIONS) - {n.name for n in wanted}
    if missing:
        raise RuntimeError(f"official evaluator is missing functions: {sorted(missing)}")
    namespace: dict[str, Any] = {"pd": pd, "math": math, "os": os, "re": re, "Path": Path, "lru_cache": lru_cache}
    module = ast.Module(body=wanted, type_ignores=[])
    exec(compile(module, str(source_path), "exec"), namespace)  # noqa: S102 - trusted benchmark source
    return {name: namespace[name] for name in _OFFICIAL_FUNCTIONS}


def _to_official_frame(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> pd.DataFrame:
    """Mimic official sql mode: DataFrame -> CSV -> read_csv."""
    buffer = io.StringIO()
    pd.DataFrame(list(rows), columns=list(columns)).to_csv(buffer, index=False)
    buffer.seek(0)
    return pd.read_csv(buffer)


class Spider2LiteEvaluator:
    """Scores Spider 2.0-Lite local instances against gold exec results."""

    def __init__(self, eval_dir: Path = SPIDER2_EVAL_DIR):
        self.eval_dir = Path(eval_dir)
        source = self.eval_dir / "evaluate.py"
        self._fns = _load_official_functions(source)
        self._standards = self._load_standards()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        self.version = f"{EVALUATOR_NAME}@sha256:{digest}"

    def _load_standards(self) -> dict[str, dict]:
        path = self.eval_dir / "gold" / "spider2lite_eval.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return {row["instance_id"]: row for row in rows}

    def has_gold(self, question_id: str) -> bool:
        gold_paths, _ = self._fns["resolve_gold_paths"](question_id, str(self.eval_dir / "gold" / "exec_result"))
        return bool(gold_paths)

    def gold_sql(self, question_id: str) -> str | None:
        """Gold SQL exists for only some instances. Never give to a model."""
        path = self.eval_dir / "gold" / "sql" / f"{question_id}.sql"
        return path.read_text(encoding="utf-8") if path.exists() else None

    def evaluate(
        self,
        question_id: str,
        db_path: str,
        predicted_sql: str,
        columns: Sequence[str] = (),
        rows: Sequence[Sequence[Any]] = (),
        **_: object,
    ) -> EvalOutcome:
        gold_dir = str(self.eval_dir / "gold" / "exec_result")
        gold_paths, is_single = self._fns["resolve_gold_paths"](question_id, gold_dir)
        if not gold_paths:
            return EvalOutcome(False, self.version, "no matching gold file found")

        try:
            pred = _to_official_frame(columns, rows)
        except pd.errors.EmptyDataError:
            return EvalOutcome(False, self.version, "prediction returned no columns")

        standard = self._standards.get(question_id, {})
        condition_cols = standard.get("condition_cols")
        ignore_order = standard.get("ignore_order", False)
        if is_single:
            gold = self._fns["load_gold_csv"](str(gold_paths[0]))
            score = self._fns["compare_pandas_table"](pred, gold, condition_cols, ignore_order)
        else:
            golds = [self._fns["load_gold_csv"](str(p)) for p in gold_paths]
            score = self._fns["compare_multi_pandas_table"](pred, golds, condition_cols, ignore_order)
        return EvalOutcome(bool(score), self.version, None if score else "result mismatch")
