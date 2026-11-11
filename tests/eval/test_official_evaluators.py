"""Known-pass / known-fail checks against the official evaluators.

These execute third-party benchmark code (test-suite-sql-eval and the
Spider 2.0-Lite evaluation_suite), so they need both the data and
``scripts/setup_evaluators.sh``.
"""

import json

import pytest

from text2sql_eval.paths import SPIDER1_DATABASE_DIR, SPIDER2_DIR, SPIDER2_EVAL_DIR, SPIDER2_LOCAL_DB_DIR, TEST_SUITE_DIR
from text2sql_eval.sandbox import run_query

DEV0 = "spider1_dev_000000"  # "How many singers do we have?" on concert_singer


@pytest.fixture(scope="module")
def spider1_eval():
    if not (TEST_SUITE_DIR / "exec_eval.py").exists():
        pytest.skip("run scripts/setup_evaluators.sh first")
    from text2sql_eval.official import Spider1Evaluator

    return Spider1Evaluator()


@pytest.fixture(scope="module")
def spider2_eval():
    if not (SPIDER2_EVAL_DIR / "evaluate.py").exists():
        pytest.skip("Spider 2 evaluation suite not present")
    from text2sql_eval.official import Spider2LiteEvaluator

    return Spider2LiteEvaluator()


def _concert_db() -> str:
    return str(SPIDER1_DATABASE_DIR / "concert_singer" / "concert_singer.sqlite")


def test_spider1_gold_passes(spider1_eval):
    gold = spider1_eval.gold_sql(DEV0)
    assert spider1_eval.evaluate(DEV0, _concert_db(), gold).correct


def test_spider1_equivalent_sql_passes(spider1_eval):
    assert spider1_eval.evaluate(DEV0, _concert_db(), "SELECT COUNT(singer_id) FROM singer").correct


# Note: "SELECT COUNT(*) FROM concert" would be *correct* by execution
# (concert also has 6 rows) -- EX judges results, not SQL text.
@pytest.mark.parametrize("sql", ["SELECT 1", "SELECT COUNT(*) + 1 FROM singer", "SELECT name FROM singer"])
def test_spider1_wrong_sql_fails(spider1_eval, sql):
    assert not spider1_eval.evaluate(DEV0, _concert_db(), sql).correct


def test_spider1_version_is_pinned_commit(spider1_eval):
    assert spider1_eval.version.startswith("spider1/test-suite-sql-eval@e97acc5")
    assert spider1_eval.version.endswith("[default]")


def test_spider1_keep_distinct_is_stricter(spider1_eval):
    from text2sql_eval.official import Spider1Evaluator

    qid = "spider1_dev_000055"  # gold: SELECT count(DISTINCT pettype) FROM pets
    db = str(SPIDER1_DATABASE_DIR / "pets_1" / "pets_1.sqlite")
    pred = 'SELECT COUNT(*) FROM "Pets"'
    assert spider1_eval.evaluate(qid, db, pred).correct  # official default strips DISTINCT
    strict = Spider1Evaluator(keep_distinct=True)
    assert not strict.evaluate(qid, db, pred).correct
    assert strict.version.endswith("[keep_distinct]")


def _spider2_local_gold():
    meta = {}
    for line in (SPIDER2_DIR / "spider2-lite.jsonl").read_text().splitlines():
        row = json.loads(line)
        meta[row["instance_id"]] = row["db"]
    ids = sorted(p.stem for p in (SPIDER2_EVAL_DIR / "gold" / "sql").glob("local*.sql"))
    return [(qid, SPIDER2_LOCAL_DB_DIR / f"{meta[qid]}.sqlite") for qid in ids]


def _official_sql_mode_score(spider2_eval, qid, db, sql):
    """Score exactly as evaluate.py's sql mode: read_sql_query -> CSV -> read_csv."""
    import io
    import sqlite3

    import pandas as pd

    fns = spider2_eval._fns
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        frame = pd.read_sql_query(sql, con)
    finally:
        con.close()
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)
    pred = pd.read_csv(buffer)
    paths, single = fns["resolve_gold_paths"](qid, str(SPIDER2_EVAL_DIR / "gold" / "exec_result"))
    std = spider2_eval._standards[qid]
    cond, order = std.get("condition_cols"), std.get("ignore_order", False)
    if single:
        return bool(fns["compare_pandas_table"](pred, fns["load_gold_csv"](str(paths[0])), cond, order))
    golds = [fns["load_gold_csv"](str(p)) for p in paths]
    return bool(fns["compare_multi_pandas_table"](pred, golds, cond, order))


def test_spider2_wrapper_matches_official_sql_mode(spider2_eval):
    """Harness verdict == official sql-mode verdict for every released gold SQL.

    Not every released gold SQL reproduces its gold CSV (e.g. local023's SQL
    returns 2 columns rounded to 3 dp, the CSV has 3 columns). That is a
    benchmark property, so we assert parity, not 100%.
    """
    mismatches, correct = [], 0
    for qid, db in _spider2_local_gold():
        sql = spider2_eval.gold_sql(qid)
        result = run_query(db, sql, timeout_s=300)
        assert result.ok, (qid, result.error)
        ours = spider2_eval.evaluate(qid, str(db), sql, columns=result.columns, rows=result.rows).correct
        correct += ours
        if ours != _official_sql_mode_score(spider2_eval, qid, db, sql):
            mismatches.append(qid)
    assert not mismatches, mismatches
    assert correct > 0


def test_spider2_gold_results_score_correct_for_all_included(spider2_eval):
    """Feeding each gold CSV back as the prediction must score 1 (all 135)."""
    import pandas as pd

    from text2sql_eval.manifest import build_spider2_lite_manifest, select

    failures = []
    for entry in select(build_spider2_lite_manifest()):
        gold_dir = str(SPIDER2_EVAL_DIR / "gold" / "exec_result")
        paths, _ = spider2_eval._fns["resolve_gold_paths"](entry.question_id, gold_dir)
        gold = pd.read_csv(paths[0])
        rows = [tuple(r) for r in gold.itertuples(index=False, name=None)]
        if not spider2_eval.evaluate(entry.question_id, "", "", columns=list(gold.columns), rows=rows).correct:
            failures.append(entry.question_id)
    assert not failures, failures


def test_spider2_wrong_result_fails(spider2_eval):
    qid, db = _spider2_local_gold()[0]
    result = run_query(db, "SELECT 1 AS x")
    assert not spider2_eval.evaluate(qid, str(db), "", columns=result.columns, rows=result.rows).correct
