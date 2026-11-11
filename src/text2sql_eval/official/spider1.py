"""Spider 1.0 execution accuracy via the official test-suite-sql-eval code.

We call ``exec_eval.eval_exec_match`` from taoyds/test-suite-sql-eval
(pinned commit, see scripts/setup_evaluators.sh) instead of re-implementing
result comparison. Only predictions that already executed safely in the
sandbox reach this function, because the official code opens databases
read-write and its timeout cannot interrupt a running SQLite query.

Note: with the original Spider databases this is standard Spider EX.
Downloading the test-suite "distilled" databases into
third_party/test-suite-sql-eval/database would give test-suite EX; that
is a separate, stricter metric and must be reported as such.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from ..contracts import EvalOutcome
from ..paths import SPIDER1_DIR, TEST_SUITE_DIR

EVALUATOR_NAME = "spider1/test-suite-sql-eval"
UNKNOWN_DIFFICULTY = "unknown"


def _evaluator_commit(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@lru_cache(maxsize=1)
def _load_exec_eval(repo: Path):
    if not (repo / "exec_eval.py").is_file():
        raise FileNotFoundError(
            f"test-suite-sql-eval not found at {repo}; run scripts/setup_evaluators.sh"
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return importlib.import_module("exec_eval")


@lru_cache(maxsize=1)
def _load_hardness(repo: Path):
    """Official Spider difficulty classifier (evaluation.Evaluator.eval_hardness)."""
    _load_exec_eval(repo)  # puts the repo on sys.path
    evaluation = importlib.import_module("evaluation")
    process_sql = importlib.import_module("process_sql")
    return evaluation.Evaluator(), process_sql


class Spider1Evaluator:
    """Scores Spider 1.0 predictions; owns access to gold SQL."""

    def __init__(
        self,
        spider_root: Path = SPIDER1_DIR,
        evaluator_repo: Path = TEST_SUITE_DIR,
        keep_distinct: bool = False,
    ):
        # keep_distinct=False is the official default: DISTINCT is stripped
        # from both queries before comparison (so COUNT(*) can match
        # COUNT(DISTINCT x)). True is stricter; report which one was used.
        self.spider_root = Path(spider_root)
        self.evaluator_repo = Path(evaluator_repo)
        self.keep_distinct = keep_distinct
        self._gold = self._load_gold()
        self._schemas: dict[str, object] = {}
        mode = "keep_distinct" if keep_distinct else "default"
        self.version = f"{EVALUATOR_NAME}@{_evaluator_commit(self.evaluator_repo)}[{mode}]"

    def _load_gold(self) -> dict[str, str]:
        import json

        gold: dict[str, str] = {}
        for split, filenames in (("dev", ["dev.json"]), ("train", ["train_spider.json", "train_others.json"])):
            offset = 0
            for filename in filenames:
                path = self.spider_root / filename
                if not path.exists():
                    continue
                rows = json.loads(path.read_text(encoding="utf-8"))
                for i, row in enumerate(rows, start=offset):
                    gold[f"spider1_{split}_{i:06d}"] = row["query"]
                offset += len(rows)
        return gold

    def gold_sql(self, question_id: str) -> str | None:
        """Gold SQL for sanity-check systems and tests. Never give to a model."""
        return self._gold.get(question_id)

    def difficulty(self, question_id: str, db_path: str) -> str:
        """Official Spider hardness (easy/medium/hard/extra) of the *gold* query.

        Used only to group results after scoring; never shown to a system.
        """
        gold = self._gold.get(question_id)
        if gold is None:
            return UNKNOWN_DIFFICULTY
        try:
            classifier, process_sql = _load_hardness(self.evaluator_repo)
            if db_path not in self._schemas:
                self._schemas[db_path] = process_sql.Schema(process_sql.get_schema(str(db_path)))
            return classifier.eval_hardness(process_sql.get_sql(self._schemas[db_path], gold))
        except Exception:  # the official parser rejects a few gold queries
            return UNKNOWN_DIFFICULTY

    def evaluate(self, question_id: str, db_path: str, predicted_sql: str, **_: object) -> EvalOutcome:
        gold = self._gold.get(question_id)
        if gold is None:
            return EvalOutcome(False, self.version, f"no gold SQL for {question_id}")
        exec_eval = _load_exec_eval(self.evaluator_repo)
        try:
            score = exec_eval.eval_exec_match(
                db=str(db_path), p_str=predicted_sql, g_str=gold,
                plug_value=False, keep_distinct=self.keep_distinct,
                progress_bar_for_each_datapoint=False,
            )
        except AssertionError as exc:  # official code asserts gold executes
            raise RuntimeError(f"gold SQL failed for {question_id}: {exc}") from exc
        return EvalOutcome(bool(score), self.version, None if score else "result mismatch")
