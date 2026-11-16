"""Shared SQL-dialect guidance for every LLM system (single-agent and two-agent).

Both systems must import the same text so neither gets extra help.
Every statement below was checked against the SQLite build the harness runs
(3.37.2, Python's sqlite3). It describes the engine, not any benchmark question.
Change the text -> bump DIALECT_HINTS_VERSION.
"""

from __future__ import annotations

DIALECT_HINTS_VERSION = "sqlite/v1"

SQLITE_HINTS = """\
SQLite notes (the database is SQLite, not PostgreSQL/MySQL/BigQuery):
- Dates are stored as text. Use strftime('%Y', d), strftime('%m', d), date(d, 'start of month'),
  date(d, '+7 days') and julianday(a) - julianday(b) for day differences.
  Not available: EXTRACT, DATE_TRUNC, DATE_FORMAT, DATEDIFF, DATE_ADD, INTERVAL, UNIXEPOCH, NOW().
- Scalar MIN(a, b) / MAX(a, b) replace LEAST / GREATEST.
- Not available: STDDEV, VARIANCE, MEDIAN, PERCENTILE_CONT, STRING_AGG, FORMAT, ILIKE.
  Compute variance as AVG(x*x) - AVG(x)*AVG(x); use GROUP_CONCAT(x, ',') and printf('%.2f', x).
- Math functions exist: SQRT, POWER, LN, EXP, SIN, COS, ACOS, RADIANS, PI, FLOOR, CEIL.
  There are no geographic ST_* functions; write the haversine formula explicitly.
- Integer division truncates (7/2 = 3). Use x * 1.0 / y or CAST(x AS REAL) / y for ratios.
- RIGHT JOIN and FULL OUTER JOIN are not supported; use LEFT JOIN (swap tables or UNION two LEFT JOINs).
- In a UNION / UNION ALL, ORDER BY and LIMIT may only appear once, at the very end.
- Aggregates cannot appear in WHERE; filter them with HAVING or in an outer query.
- Window functions (ROW_NUMBER, RANK, LAG, LEAD, SUM(...) OVER) and CTEs (WITH) are supported.
- Concatenate strings with ||. Quote identifiers containing spaces with double quotes."""

_HINTS = {"sqlite": SQLITE_HINTS}

# Replacements for functions SQLite reports as missing (used in repair feedback).
SQLITE_FUNCTION_REPLACEMENTS = {
    "EXTRACT": "strftime('%Y', d) / strftime('%m', d) etc.",
    "DATE_TRUNC": "date(d, 'start of month') or strftime('%Y-%m', d)",
    "DATE_FORMAT": "strftime(format, d)",
    "TO_CHAR": "strftime(format, d)",
    "DATEDIFF": "julianday(a) - julianday(b)",
    "DATE_ADD": "date(d, '+N days')",
    "DATE_SUB": "date(d, '-N days')",
    "UNIXEPOCH": "strftime('%s', d)",
    "NOW": "date('now')",
    "LEAST": "scalar MIN(a, b)",
    "GREATEST": "scalar MAX(a, b)",
    "STDDEV": "SQRT(AVG(x*x) - AVG(x)*AVG(x))",
    "STDEV": "SQRT(AVG(x*x) - AVG(x)*AVG(x))",
    "STDDEV_POP": "SQRT(AVG(x*x) - AVG(x)*AVG(x))",
    "STDDEV_SAMP": "the sample formula with SUM and COUNT",
    "VARIANCE": "AVG(x*x) - AVG(x)*AVG(x)",
    "MEDIAN": "ROW_NUMBER() / COUNT(*) OVER () to pick the middle row(s)",
    "PERCENTILE_CONT": "ROW_NUMBER() / COUNT(*) OVER () to pick the row at that rank",
    "STRING_AGG": "GROUP_CONCAT(x, ',')",
    "FORMAT": "printf(format, x)",
    "ILIKE": "LOWER(x) LIKE LOWER(pattern)",
    "ST_DISTANCE": "the haversine formula with RADIANS, SIN, COS, ACOS",
    "ST_DISTANCE_SPHERE": "the haversine formula with RADIANS, SIN, COS, ACOS",
}


def dialect_hints(dialect: str | None) -> str:
    """Guidance text for a dialect ('' when there is none)."""
    return _HINTS.get((dialect or "").lower(), "")
