"""Budget checks, difficulty grouping, repeat aggregation, paired comparison, plug-in loading."""

import importlib.util
import sys
from argparse import Namespace

import numpy as np
import pytest

from text2sql_eval.analysis import (
    RunMismatchError,
    aggregate_repeats,
    compare_runs,
    mcnemar_exact_p,
    paired_bootstrap_ci,
)
from text2sql_eval.contracts import Budget, PredictionResult, budget_violations
from text2sql_eval.metrics import summarize, summarize_by_difficulty, within_budget
from text2sql_eval.paths import PROJECT_ROOT
from text2sql_eval.runner import RunConfig, evaluate_one

GOOD_SQL = ("SELECT d.dept_name, COUNT(*) FROM staff s JOIN dept d USING (dept_id) "
            "WHERE s.status = 'A' GROUP BY d.dept_name")


def _record(qid, correct, run_id="r", system="s", benchmark="b", repeat=0, difficulty=None, **extra):
    return {"run_id": run_id, "system_name": system, "benchmark": benchmark, "repeat_index": repeat,
            "question_id": qid, "db_id": "db", "correct": correct,
            "status": "executed" if correct else "execution_error", "difficulty": difficulty,
            "within_budget": True, "wall_time_s": 1.0, **extra}


def _run(pattern, **kwargs):
    return [_record(f"q{i}", bool(ok), **kwargs) for i, ok in enumerate(pattern)]


# ------------------------------------------------------------------ budget

class OverBudgetSystem:
    name = "greedy"

    def __init__(self, calls, retries, tokens):
        self.calls, self.retries, self.tokens = calls, retries, tokens

    def predict(self, question, budget):
        return PredictionResult(question.question_id, question.db_id, self.name, GOOD_SQL,
                                model_calls=self.calls, retry_count=self.retries, input_tokens=self.tokens)


def test_budget_violations_lists_each_exceeded_limit():
    budget = Budget(max_retries=3, max_model_calls=8, max_total_input_tokens=32_000)
    ok = PredictionResult("q", "d", "s", "SELECT 1", model_calls=8, retry_count=3, input_tokens=32_000)
    bad = PredictionResult("q", "d", "s", "SELECT 1", model_calls=9, retry_count=4, input_tokens=32_001)
    assert budget_violations(ok, budget) == []
    assert [v.split()[0] for v in budget_violations(bad, budget)] == ["model_calls", "retries", "input_tokens"]


def test_runner_flags_over_budget_but_keeps_official_verdict(company_entry, evaluator):
    config = RunConfig(run_id="r", system_name="greedy", benchmark="test", manifest="m",
                       budget=Budget(max_model_calls=2))
    record = evaluate_one(OverBudgetSystem(calls=3, retries=0, tokens=10), company_entry, evaluator, config)
    assert record["correct"] is True                       # official EX verdict untouched
    assert record["within_budget"] is False
    assert record["budget_violations"] == ["model_calls 3 > 2"]
    summary = summarize([record])
    assert summary["n_over_budget"] == 1
    assert summary["correct"] == 1 and summary["correct_within_budget"] == 0


def test_within_budget_recomputed_for_old_records():
    old = {"model_calls": 9, "retry_count": 0, "input_tokens": 0}   # no within_budget field
    assert within_budget(old) is False
    assert within_budget({"model_calls": 1}) is True


def test_difficulty_absent_for_evaluators_without_classifier(company_entry, evaluator):
    config = RunConfig(run_id="r", system_name="greedy", benchmark="test", manifest="m")
    record = evaluate_one(OverBudgetSystem(1, 0, 1), company_entry, evaluator, config)
    assert record["difficulty"] is None
    assert summarize_by_difficulty([record]) == []


# -------------------------------------------------------------- difficulty

def test_summarize_by_difficulty_orders_levels():
    records = [_record("q1", True, difficulty="hard"), _record("q2", False, difficulty="easy"),
               _record("q3", True, difficulty="easy"), _record("q4", True, difficulty="extra")]
    rows = summarize_by_difficulty(records)
    assert [r["difficulty"] for r in rows] == ["easy", "hard", "extra"]
    assert rows[0]["correct"] == 1 and rows[0]["evaluated"] == 2


def test_official_spider1_difficulty(data_available):
    from text2sql_eval.manifest import read_manifest
    from text2sql_eval.official import get_evaluator
    from text2sql_eval.paths import MANIFEST_DIR, TEST_SUITE_DIR

    if not (TEST_SUITE_DIR / "evaluation.py").is_file():
        pytest.skip("run scripts/setup_evaluators.sh first")
    pytest.importorskip("nltk")
    evaluator = get_evaluator("spider1")
    entries = read_manifest(MANIFEST_DIR / "spider1_dev.jsonl")[:3]
    labels = [evaluator.difficulty(e.question_id, str(PROJECT_ROOT / e.db_path)) for e in entries]
    assert labels[0] == "easy"                              # "How many singers do we have?"
    assert set(labels) <= {"easy", "medium", "hard", "extra"}
    assert evaluator.difficulty("no_such_question", "x.sqlite") == "unknown"


# ------------------------------------------------------------- statistics

def test_mcnemar_exact_known_values():
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(5, 5) == 1.0
    assert mcnemar_exact_p(10, 2) == pytest.approx(0.0386, abs=1e-4)
    assert mcnemar_exact_p(2, 10) == mcnemar_exact_p(10, 2)


def test_bootstrap_ci_is_deterministic_and_contains_difference():
    rng = np.random.default_rng(1)
    a, b = rng.random(300) < 0.7, rng.random(300) < 0.5
    ci1 = paired_bootstrap_ci(a, b, iterations=2000, seed=0)
    assert ci1 == paired_bootstrap_ci(a, b, iterations=2000, seed=0)
    assert ci1[0] < a.mean() - b.mean() < ci1[1]


def test_bootstrap_ci_identical_runs_is_zero():
    a = np.array([True, False, True, True])
    assert paired_bootstrap_ci(a, a, iterations=500) == (0.0, 0.0)


# ---------------------------------------------------------------- compare

def test_compare_runs_counts_and_summary():
    a = _run([1, 1, 0, 0, 1], run_id="A", system="two_agent")
    b = _run([1, 0, 1, 0, 0], run_id="B", system="single_agent")
    result = compare_runs(a, b, bootstrap_iterations=500)
    s = result.summary
    assert (s["n_both"], s["n_only_a"], s["n_only_b"], s["n_neither"]) == (1, 2, 1, 1)
    assert s["ex_a"] == pytest.approx(0.6) and s["ex_b"] == pytest.approx(0.4)
    assert s["ex_diff_a_minus_b"] == pytest.approx(0.2)
    assert s["mcnemar_exact_p"] == pytest.approx(mcnemar_exact_p(2, 1))
    assert [r["outcome"] for r in result.paired_rows] == ["both", "only_a", "only_b", "neither", "only_a"]


def test_compare_within_budget_counts_over_budget_as_wrong():
    a = _run([1, 1], run_id="A")
    a[0]["within_budget"] = False
    b = _run([1, 1], run_id="B")
    assert compare_runs(a, b, bootstrap_iterations=100).summary["n_only_a"] == 0
    s = compare_runs(a, b, within_budget_only=True, bootstrap_iterations=100).summary
    assert s["n_only_b"] == 1 and s["ex_a"] == pytest.approx(0.5)


def test_compare_rejects_different_question_sets():
    with pytest.raises(RunMismatchError, match="different questions"):
        compare_runs(_run([1, 1, 1], run_id="A"), _run([1, 1], run_id="B"))


def test_compare_rejects_mixed_benchmarks():
    with pytest.raises(RunMismatchError, match="benchmarks"):
        compare_runs(_run([1], run_id="A", benchmark="spider1"), _run([1], run_id="B", benchmark="x"))


# -------------------------------------------------------------- aggregate

def test_aggregate_repeats_mean_and_sample_std():
    runs = {f"rep{i}": _run(p, run_id=f"rep{i}", repeat=i)
            for i, p in enumerate([[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]])}
    other = {"cls": _run([0, 0, 0, 0], run_id="cls", system="classical")}
    rows = aggregate_repeats({**runs, **other})
    by_system = {r["system_name"]: r for r in rows}
    s = by_system["s"]
    assert s["n_repeats"] == 3 and s["n_questions"] == 4
    assert s["execution_accuracy_mean"] == pytest.approx(0.5)
    assert s["execution_accuracy_std"] == pytest.approx(0.25)    # stdev of 0.5, 0.75, 0.25
    assert s["correct_per_repeat"] == "2;3;1"
    assert by_system["classical"]["execution_accuracy_std"] == 0.0


def test_aggregate_rejects_repeats_on_different_questions():
    runs = {"r0": _run([1, 1], run_id="r0"), "r1": _run([1, 1, 1], run_id="r1")}
    with pytest.raises(RunMismatchError):
        aggregate_repeats(runs)


# ---------------------------------------------------------------- plug-in

@pytest.fixture()
def run_eval_module():
    spec = importlib.util.spec_from_file_location("run_eval_script", PROJECT_ROOT / "scripts" / "run_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_external_system_factory_loaded(tmp_path, monkeypatch, run_eval_module):
    (tmp_path / "my_plugin.py").write_text(
        "class S:\n"
        "    name = 'plugin'\n"
        "    def __init__(self, opts): self.opts = opts\n"
        "    def predict(self, question, budget): raise NotImplementedError\n"
        "def build(args):\n"
        "    return S(args.system_options)\n"
        "def broken(args):\n"
        "    return object()\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    args = Namespace(system_options={"views": "on"})
    system = run_eval_module.load_external_system("my_plugin:build", args)
    assert system.name == "plugin" and system.opts == {"views": "on"}
    with pytest.raises(SystemExit, match="must return"):
        run_eval_module.load_external_system("my_plugin:broken", args)
    with pytest.raises(SystemExit, match="not a callable"):
        run_eval_module.load_external_system("my_plugin:missing", args)
    with pytest.raises(SystemExit, match="package.module:factory"):
        run_eval_module.load_external_system("my_plugin", args)
    sys.modules.pop("my_plugin", None)


def test_system_option_parsing(run_eval_module):
    assert run_eval_module._parse_options(["a=1", "b=x=y"]) == {"a": "1", "b": "x=y"}
    with pytest.raises(SystemExit):
        run_eval_module._parse_options(["novalue"])


def test_aggregate_keeps_scoring_modes_apart():
    runs = {"d": _run([1, 0], run_id="d", evaluator="spider1[default]"),
            "k": _run([0, 0], run_id="k", evaluator="spider1[keep_distinct]")}
    rows = aggregate_repeats(runs)
    assert len(rows) == 2 and all(r["n_repeats"] == 1 for r in rows)


def test_compare_rejects_different_evaluators_unless_allowed():
    a = _run([1, 0], run_id="A", evaluator="spider1[default]")
    b = _run([0, 0], run_id="B", evaluator="spider1[keep_distinct]")
    with pytest.raises(RunMismatchError, match="scored differently"):
        compare_runs(a, b)
    assert compare_runs(a, b, allow_different_evaluators=True, bootstrap_iterations=50).summary["n_only_a"] == 1
