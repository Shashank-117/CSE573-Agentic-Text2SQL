"""Schema-grounded explanations of SQLite errors for the repair loop.

Uses only the database schema (which the model is already shown), never gold
SQL or gold results. In the v1 pilot on Spider 2.0-Lite local, 125 of 145
"no such column" errors named a column that exists in a *different* table,
so pointing at the right table is the most useful feedback.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from .dialect_hints import SQLITE_FUNCTION_REPLACEMENTS

_MAX_LISTED_COLUMNS = 40
_TABLE_REF = re.compile(
    r"\b(?:FROM|JOIN)\s+(\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|[\w.]+)(?:\s+(?:AS\s+)?(\"[^\"]+\"|[A-Za-z_]\w*))?",
    re.IGNORECASE,
)
_NOT_ALIASES = frozenset({
    "WHERE", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "CROSS", "FULL", "NATURAL", "ON", "USING",
    "GROUP", "ORDER", "LIMIT", "HAVING", "UNION", "EXCEPT", "INTERSECT", "WINDOW", "AS",
})


@dataclass(frozen=True)
class SchemaIndex:
    """Lower-cased lookup of table -> columns for one database."""

    columns_by_table: dict[str, tuple[str, ...]]   # display names, keyed by lower-case table

    @classmethod
    def from_tables(cls, tables: dict[str, list[str]]) -> "SchemaIndex":
        return cls({name.lower(): (name, *cols) for name, cols in tables.items()})

    def table_name(self, key: str) -> str:
        return self.columns_by_table[key][0]

    def columns(self, key: str) -> tuple[str, ...]:
        return self.columns_by_table[key][1:]

    def tables_with_column(self, column: str) -> list[str]:
        wanted = column.lower()
        return [self.table_name(t) for t in self.columns_by_table
                if any(c.lower() == wanted for c in self.columns(t))]

    def similar_columns(self, column: str, limit: int = 5) -> list[str]:
        everything = {f"{self.table_name(t)}.{c}" for t in self.columns_by_table for c in self.columns(t)}
        bare = {item.split(".", 1)[1].lower(): item for item in everything}
        names = difflib.get_close_matches(column.lower(), list(bare), n=limit, cutoff=0.6)
        return [bare[n] for n in names]


def _unquote(name: str) -> str:
    return name.strip('"`[]')


def table_aliases(sql: str) -> dict[str, str]:
    """Map alias (and bare table name) -> table name, from FROM/JOIN clauses."""
    aliases: dict[str, str] = {}
    for match in _TABLE_REF.finditer(sql):
        table = _unquote(match.group(1)).split(".")[-1]
        aliases[table.lower()] = table
        alias = match.group(2)
        if alias and alias.upper() not in _NOT_ALIASES:
            aliases[_unquote(alias).lower()] = table
    return aliases


def _column_list(schema: SchemaIndex, table_key: str) -> str:
    cols = schema.columns(table_key)
    shown = ", ".join(cols[:_MAX_LISTED_COLUMNS])
    return shown + (", ..." if len(cols) > _MAX_LISTED_COLUMNS else "")


def _explain_missing_column(reference: str, sql: str, schema: SchemaIndex) -> str:
    qualifier, _, column = reference.rpartition(".")
    column = _unquote(column)
    hints = []
    owners = schema.tables_with_column(column)
    if qualifier:
        table = table_aliases(sql).get(_unquote(qualifier).lower())
        if table and table.lower() in schema.columns_by_table:
            key = table.lower()
            hints.append(f'Table "{schema.table_name(key)}" (used as {qualifier}) has no column "{column}". '
                         f"Its columns are: {_column_list(schema, key)}.")
        elif table is None:
            hints.append(f'"{qualifier}" is not a table or alias defined in this query.')
    if owners:
        hints.append(f'Column "{column}" exists in: {", ".join(owners)}. Join that table, or use the right alias.')
    else:
        similar = schema.similar_columns(column)
        hints.append(f'No table has a column named "{column}".'
                     + (f" Similar columns: {', '.join(similar)}." if similar else
                        " If it is an alias from SELECT, repeat the expression instead."))
    return " ".join(hints)


def explain_error(error: str, sql: str, schema: SchemaIndex | None) -> str:
    """Extra guidance for a database error, or '' if there is nothing useful to add."""
    if match := re.search(r"no such column: (\S+)", error):
        return _explain_missing_column(match.group(1), sql, schema) if schema else ""
    if match := re.search(r"no such function: (\w+)", error):
        name = match.group(1).upper()
        replacement = SQLITE_FUNCTION_REPLACEMENTS.get(name)
        return f"SQLite has no {name}(); use {replacement}." if replacement else \
            f"SQLite has no {name}(); rewrite it with built-in SQLite functions."
    if match := re.search(r"ambiguous column name: (\S+)", error):
        column = match.group(1).split(".")[-1]
        owners = schema.tables_with_column(column) if schema else []
        where = f" It exists in: {', '.join(owners)}." if owners else ""
        return f'Qualify "{column}" with a table alias.{where}'
    if match := re.search(r"no such table: (\S+)", error):
        if not schema:
            return ""
        names = [schema.table_name(t) for t in schema.columns_by_table]
        close = difflib.get_close_matches(_unquote(match.group(1)), names, n=3, cutoff=0.5)
        return f"Available tables: {', '.join(names)}." + (f" Did you mean {', '.join(close)}?" if close else "")
    if "RIGHT and FULL OUTER JOINs are not currently supported" in error:
        return "Rewrite with LEFT JOIN (swap the table order)."
    if "ORDER BY clause should come after" in error:
        return "Put a single ORDER BY / LIMIT after the last SELECT of the UNION, or wrap the UNION in a subquery."
    if "misuse of aggregate" in error:
        return "Aggregates cannot be used in WHERE or nested directly; use HAVING or compute them in a subquery/CTE."
    return ""
