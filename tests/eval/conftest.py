import sqlite3
from pathlib import Path

import pytest

from text2sql_eval.contracts import EvalOutcome
from text2sql_eval.manifest import ManifestEntry
from text2sql_eval.paths import PROJECT_ROOT


@pytest.fixture()
def company_db(tmp_path: Path) -> Path:
    """Tiny HR database used for the illustrative 'active employees' example."""
    path = tmp_path / "company.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE dept (dept_id INTEGER PRIMARY KEY, dept_name TEXT);
        CREATE TABLE staff (staff_id INTEGER PRIMARY KEY, dept_id INTEGER, status TEXT,
                            FOREIGN KEY (dept_id) REFERENCES dept(dept_id));
        INSERT INTO dept VALUES (1, 'Sales'), (2, 'Eng');
        INSERT INTO staff VALUES (1, 1, 'A'), (2, 1, 'T'), (3, 2, 'A'), (4, 2, 'A');
        """
    )
    con.commit()
    con.close()
    return path


@pytest.fixture()
def company_entry(company_db: Path) -> ManifestEntry:
    return ManifestEntry(
        question_id="test_000001",
        benchmark="test",
        split="dev",
        db_id="company",
        db_path=str(company_db),  # absolute path is fine: joined onto PROJECT_ROOT
        question="How many active employees are in each department?",
        dialect="sqlite",
        evidence_doc_ids=(),
        included=True,
    )


class ResultSetEvaluator:
    """Pure-Python stand-in for an official evaluator (order-insensitive)."""

    version = "test-evaluator"

    def __init__(self, gold: dict[str, list[tuple]]):
        self.gold = gold
        self.calls: list[str] = []

    def evaluate(self, question_id, db_path, predicted_sql, columns=(), rows=(), **_):
        self.calls.append(predicted_sql)
        correct = sorted(map(tuple, rows)) == sorted(self.gold[question_id])
        return EvalOutcome(correct, self.version, None if correct else "result mismatch")


@pytest.fixture()
def evaluator() -> ResultSetEvaluator:
    return ResultSetEvaluator({"test_000001": [("Sales", 1), ("Eng", 2)]})


@pytest.fixture()
def data_available() -> None:
    if not (PROJECT_ROOT / "data" / "Spider1" / "dev.json").exists():
        pytest.skip("benchmark data not present under data/")
