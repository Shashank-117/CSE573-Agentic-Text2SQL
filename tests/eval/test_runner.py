import json

import pytest

from text2sql_eval.contracts import Budget, PredictionResult, Status
from text2sql_eval.metrics import load_records, summarize, write_reports
from text2sql_eval.runner import RunConfig, evaluate_one, run

CONFIG = RunConfig(run_id="r1", system_name="fake", benchmark="test", manifest="m.jsonl",
                   budget=Budget(sql_timeout_s=0.5))

GOOD_SQL = ("SELECT d.dept_name, COUNT(*) FROM staff s JOIN dept d USING (dept_id) "
            "WHERE s.status = 'A' GROUP BY d.dept_name")
JOIN_BUG_SQL = "SELECT d.dept_name, COUNT(*) FROM staff s JOIN dept d USING (dept_id) GROUP BY d.dept_name"


class FakeSystem:
    name = "fake"

    def __init__(self, sql=None, status="success", artifacts=None, crash=False):
        self.sql, self.status, self.artifacts, self.crash = sql, status, artifacts or {}, crash
        self.seen = []

    def predict(self, question, budget):
        self.seen.append(question)
        if self.crash:
            raise RuntimeError("boom")
        return PredictionResult(question.question_id, question.db_id, self.name, self.sql,
                                status=self.status, intermediate_artifacts=self.artifacts,
                                model_calls=2, input_tokens=100, output_tokens=20)


@pytest.mark.parametrize(
    "system,status,correct",
    [
        (FakeSystem(GOOD_SQL), Status.EXECUTED, True),
        (FakeSystem(JOIN_BUG_SQL), Status.EXECUTED, False),
        (FakeSystem("SELECT nope FROM staff"), Status.EXECUTION_ERROR, False),
        (FakeSystem("DELETE FROM staff"), Status.UNSAFE_SQL, False),
        (FakeSystem(None, status=Status.UNSUPPORTED), Status.UNSUPPORTED, False),
        (FakeSystem(""), Status.PARSE_ERROR, False),
        (FakeSystem(GOOD_SQL, status="weird"), Status.SYSTEM_ERROR, False),
        (FakeSystem(crash=True), Status.SYSTEM_ERROR, False),
        (FakeSystem("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT COUNT(*) FROM c"),
         Status.TIMEOUT, False),
    ],
)
def test_statuses(system, status, correct, company_entry, evaluator):
    record = evaluate_one(system, company_entry, evaluator, CONFIG)
    assert record["status"] == status
    assert record["correct"] is correct
    assert record["question_id"] == company_entry.question_id


def test_only_executed_sql_reaches_evaluator(company_entry, evaluator):
    for sql in ["DELETE FROM staff", "SELECT nope FROM staff", None]:
        evaluate_one(FakeSystem(sql), company_entry, evaluator, CONFIG)
    assert evaluator.calls == []


def test_two_agent_views_scored_as_standalone_cte(company_entry, evaluator):
    views = {"views": [{"name": "Employee", "sql": "SELECT s.staff_id AS employee_id, d.dept_name AS "
                        "department_name FROM staff s JOIN dept d USING (dept_id) WHERE s.status = 'A'"}]}
    system = FakeSystem("SELECT department_name, COUNT(*) FROM Employee GROUP BY 1", artifacts=views)
    record = evaluate_one(system, company_entry, evaluator, CONFIG)
    assert record["status"] == Status.EXECUTED and record["correct"]
    assert record["scored_sql"].startswith('WITH "Employee" AS (')
    assert evaluator.calls == [record["scored_sql"]]


def test_malformed_view_artifact(company_entry, evaluator):
    system = FakeSystem("SELECT 1", artifacts={"views": [{"sql": "SELECT 1"}]})
    assert evaluate_one(system, company_entry, evaluator, CONFIG)["status"] == Status.VIEW_ERROR


def test_evaluator_crash_is_eval_error(company_entry):
    class Broken:
        version = "broken"

        def evaluate(self, *a, **k):
            raise ValueError("gold missing")

    record = evaluate_one(FakeSystem(GOOD_SQL), company_entry, Broken(), CONFIG)
    assert record["status"] == Status.EVAL_ERROR and not record["correct"]


def test_run_writes_one_record_per_question_and_resumes(tmp_path, company_entry, evaluator):
    entries = [company_entry, company_entry.__class__(**{**company_entry.__dict__, "question_id": "test_000002"})]
    evaluator.gold["test_000002"] = []
    runs_path = run(FakeSystem(GOOD_SQL), entries, evaluator, CONFIG, tmp_path, progress_every=0)
    assert [r["question_id"] for r in load_records(runs_path)] == ["test_000001", "test_000002"]

    system = FakeSystem(GOOD_SQL)
    run(system, entries, evaluator, CONFIG, tmp_path, resume=True, progress_every=0)
    assert system.seen == []  # nothing re-run
    assert len(load_records(runs_path)) == 2

    reports = write_reports(runs_path)
    summary_line = reports["summary"].read_text().splitlines()[1]
    assert "0.5" in summary_line  # 1 of 2 correct
    assert "test_000002" in reports["errors"].read_text()


def test_failures_stay_in_denominator():
    records = [
        {"run_id": "r", "system_name": "s", "benchmark": "b", "status": Status.EXECUTED, "correct": True},
        {"run_id": "r", "system_name": "s", "benchmark": "b", "status": Status.UNSUPPORTED, "correct": False},
        {"run_id": "r", "system_name": "s", "benchmark": "b", "status": Status.TIMEOUT, "correct": False},
        {"run_id": "r", "system_name": "s", "benchmark": "b", "status": Status.EXECUTED, "correct": False},
    ]
    summary = summarize(records)
    assert summary["evaluated"] == 4
    assert summary["execution_accuracy"] == 0.25
    assert summary["coverage"] == 0.75
    assert summary["accuracy_when_supported"] == pytest.approx(1 / 3)
    assert summary["n_timeout"] == 1
    assert summary["peak_gpu_memory_mb"] is None  # unmeasured stays missing, not 0


def test_records_are_json_serializable(company_entry, evaluator):
    record = evaluate_one(FakeSystem(GOOD_SQL), company_entry, evaluator, CONFIG)
    json.dumps(record)
