"""Turn (views + primary SQL) into one standalone query for official scoring.

The two-agent system produces idealized views plus primary SQL over them.
Official evaluators only accept a single query against the real database,
so we inline each view as a common table expression (CTE):

    views: Employee := SELECT ...
    primary: SELECT department_name, COUNT(*) FROM Employee GROUP BY 1
    ->  WITH "Employee" AS (SELECT ...) SELECT department_name, ...

A non-recursive CTE has the same semantics as a view for a read-only
query, so this does not change the answer. The harness also executes the
original form (TEMP views + primary SQL) in the sandbox and records both.
"""

from __future__ import annotations

import re
from typing import Sequence

from .sandbox import ViewDefinition
from .sql_safety import normalize_sql

_LEADING_WITH = re.compile(r"^\s*WITH(\s+RECURSIVE)?\s+", re.IGNORECASE)


def compose_with_views(primary_sql: str, views: Sequence[ViewDefinition]) -> str:
    """Return ``primary_sql`` with ``views`` prepended as CTEs."""
    body = normalize_sql(primary_sql)
    if not views:
        return body

    ctes = ", ".join(f'"{v.name}" AS ({normalize_sql(v.sql)})' for v in views)
    match = _LEADING_WITH.match(body)
    if match is None:
        return f"WITH {ctes} {body}"

    # Primary SQL already has a WITH clause: merge into a single clause.
    recursive = "RECURSIVE " if match.group(1) else ""
    return f"WITH {recursive}{ctes}, {body[match.end():]}"


def views_from_artifacts(artifacts: dict) -> list[ViewDefinition]:
    """Parse ``intermediate_artifacts['views']`` from a PredictionResult."""
    raw = artifacts.get("views") or []
    views = []
    for item in raw:
        if not isinstance(item, dict) or "name" not in item or "sql" not in item:
            raise ValueError(f"malformed view artifact: {item!r}")
        views.append(ViewDefinition(name=str(item["name"]), sql=str(item["sql"])))
    return views
