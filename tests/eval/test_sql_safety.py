import pytest

from text2sql_eval.sql_safety import check_read_only, normalize_sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM t",
        "select count(*) from t;",
        "  WITH x AS (SELECT 1) SELECT * FROM x ;; ",
        "SELECT 'DROP TABLE t' AS s",              # keyword inside a string literal
        'SELECT "update" FROM t',                   # keyword as quoted identifier
        "-- leading comment\nSELECT 1",
        "SELECT created_at, deleted FROM t",        # keyword substrings in names
        "SELECT REPLACE(name, 'a', 'b') FROM t",    # REPLACE() string function
        "SELECT replace (name, '-', '') AS n FROM t",
    ],
)
def test_accepts_single_read_only_statement(sql):
    verdict = check_read_only(sql)
    assert verdict.ok, verdict.reason
    assert not verdict.sql.endswith(";")


@pytest.mark.parametrize(
    "sql,reason_part",
    [
        (None, "empty"),
        ("   ", "empty"),
        ("-- only a comment", "empty"),
        ("DELETE FROM t", "must start"),
        ("UPDATE t SET a = 1", "must start"),
        ("DROP TABLE t", "must start"),
        ("PRAGMA writable_schema = 1", "must start"),
        ("ATTACH DATABASE 'x.db' AS x", "must start"),
        ("SELECT 1; DROP TABLE t", "statement"),
        ("SELECT 1; SELECT 2", "statement"),
        ("WITH d AS (DELETE FROM t RETURNING *) SELECT * FROM d", "forbidden"),
        ("WITH c AS (SELECT 1) REPLACE INTO t VALUES (1)", "forbidden"),
        ("WITH c AS (SELECT 1) INSERT INTO t SELECT * FROM c", "forbidden"),
        ("WITH c AS (SELECT 1) UPDATE t SET a = 1", "forbidden"),
        ("SELECT * FROM t /* */; INSERT INTO t VALUES (1)", "statement"),
    ],
)
def test_rejects_non_read_only(sql, reason_part):
    verdict = check_read_only(sql)
    assert not verdict.ok
    assert reason_part in verdict.reason


def test_normalize_strips_comments_and_semicolons():
    assert normalize_sql("SELECT 1 -- c\n;") == "SELECT 1"
