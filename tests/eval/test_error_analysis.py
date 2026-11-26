"""Error categorization for Spider 1 (normalizer is pure; the matcher test needs the evaluator + data)."""

import pytest

from text2sql_eval.error_analysis import (
    EXTRA_COLUMNS, EXTRA_TABLES, MISSING_TABLES, SAME_SHAPE, WHERE, WRONG_TABLES,
    categorize, normalize_for_spider_parser,
)


@pytest.mark.parametrize("sql,expected", [
    ("SELECT MAX(a) AS m FROM t;", "SELECT MAX(a) FROM t"),
    ("SELECT s.x FROM stadium s LEFT JOIN concert c ON s.id = c.id",
     "SELECT s.x FROM stadium AS s JOIN concert AS c ON s.id = c.id"),
    ("SELECT k, COUNT(*) AS n FROM t GROUP BY k HAVING n > 2 ORDER BY n",
     "SELECT k, COUNT(*) FROM t GROUP BY k HAVING COUNT(*) > 2 ORDER BY COUNT(*)"),
    ("SELECT COUNT(c.id), COUNT(DISTINCT c.k) FROM t AS c", "SELECT COUNT(*), COUNT(DISTINCT c.k) FROM t AS c"),
    ('SELECT "Name" FROM t WHERE x = "Bob"', 'SELECT Name FROM t WHERE x = "Bob"'),
    ("SELECT a FROM t WHERE b = 1", "SELECT a FROM t WHERE b = 1"),
])
def test_normalize_for_spider_parser(sql, expected):
    assert normalize_for_spider_parser(sql) == expected


def _scores(**f1):
    return {name: {"f1": value} for name, value in f1.items()}


@pytest.mark.parametrize("pred,gold,expected", [
    (["a", "b"], ["a"], EXTRA_TABLES),
    (["a"], ["a", "b"], MISSING_TABLES),
    (["a", "c"], ["a", "b"], WRONG_TABLES),
])
def test_table_categories(pred, gold, expected):
    assert categorize({}, pred, gold) == [expected]


def test_extra_columns_and_where_are_both_reported_in_order():
    scores = {**_scores(select=0, where=0), "select(no AGG)": {"f1": 0},
              "select_counts": {"label_total": 1, "pred_total": 2, "matched": 1}}
    assert categorize(scores, ["t"], ["t"]) == [EXTRA_COLUMNS, WHERE]


def test_everything_matching_means_same_shape():
    assert categorize(_scores(select=1, where=1), ["t"], ["t"]) == [SAME_SHAPE]


def test_official_matcher_on_real_query():
    from text2sql_eval.error_analysis import _SpiderParser
    from text2sql_eval.official import get_evaluator
    from text2sql_eval.paths import SPIDER1_DATABASE_DIR

    db = SPIDER1_DATABASE_DIR / "car_1" / "car_1.sqlite"
    if not db.exists():
        pytest.skip("Spider 1 data not available")
    parser = _SpiderParser(get_evaluator("spider1"))
    pred = parser.parse("car_1", str(db), normalize_for_spider_parser(
        "SELECT Model, COUNT(DISTINCT MakeId) AS n FROM car_names GROUP BY Model ORDER BY n DESC LIMIT 1;"))
    gold = parser.parse("car_1", str(db), "SELECT Model FROM CAR_NAMES GROUP BY Model ORDER BY count(*) DESC LIMIT 1")
    assert EXTRA_COLUMNS in categorize(parser.components(pred, gold), ["car_names"], ["car_names"])
