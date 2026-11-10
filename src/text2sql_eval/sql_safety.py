"""Static read-only checks for model-generated SQL.

This is the first of three layers. The sandbox also opens databases in
read-only mode and installs a SQLite authorizer, so a query that slips
past these checks still cannot modify data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import sqlparse

_ALLOWED_FIRST_KEYWORDS = frozenset({"SELECT", "WITH", "VALUES"})

# Keywords that never belong in a read-only query, matched as whole words
# outside string literals and comments.
_FORBIDDEN_KEYWORDS = frozenset(
    {
        "INSERT", "UPDATE", "DELETE", "REPLACE", "MERGE", "UPSERT",
        "CREATE", "DROP", "ALTER", "TRUNCATE", "RENAME",
        "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REINDEX", "ANALYZE",
        "GRANT", "REVOKE", "BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT",
        "RELEASE",
    }
)

_STRING_OR_IDENT = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|`[^`]*`|\[[^\]]*\]")
_WORD = re.compile(r"[A-Za-z_]+")


@dataclass(frozen=True)
class SafetyVerdict:
    ok: bool
    sql: str | None
    reason: str | None = None


def _strip_literals(sql: str) -> str:
    """Blank out string literals and quoted identifiers."""
    return _STRING_OR_IDENT.sub(" ", sql)


def normalize_sql(sql: str) -> str:
    """Remove comments, surrounding whitespace and trailing semicolons."""
    stripped = sqlparse.format(sql, strip_comments=True).strip()
    while stripped.endswith(";"):
        stripped = stripped[:-1].rstrip()
    return stripped


def check_read_only(sql: str | None) -> SafetyVerdict:
    """Accept exactly one SELECT/WITH/VALUES statement, nothing else."""
    if sql is None or not sql.strip():
        return SafetyVerdict(False, None, "empty SQL")

    cleaned = normalize_sql(sql)
    if not cleaned:
        return SafetyVerdict(False, None, "empty SQL after removing comments")

    statements = [s for s in sqlparse.split(cleaned) if s.strip().rstrip(";").strip()]
    if len(statements) != 1:
        return SafetyVerdict(False, None, f"expected 1 statement, found {len(statements)}")

    words = [w.upper() for w in _WORD.findall(_strip_literals(cleaned))]
    if not words or words[0] not in _ALLOWED_FIRST_KEYWORDS:
        first = words[0] if words else "?"
        return SafetyVerdict(False, None, f"statement must start with SELECT/WITH, got {first}")

    forbidden = sorted(set(words) & _FORBIDDEN_KEYWORDS)
    if forbidden:
        return SafetyVerdict(False, None, f"forbidden keyword(s): {', '.join(forbidden)}")

    return SafetyVerdict(True, cleaned)
