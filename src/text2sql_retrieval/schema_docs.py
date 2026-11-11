"""Render a physical SQLite schema as per-table documents for prompting.

Each table becomes a CREATE TABLE-style block (columns, types, primary and
foreign keys) plus a few sample rows. Sample rows are read through a
read-only connection with a deadline and are part of the "physical schema
access" that both LLM systems are allowed (keep it identical across them).
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from text2sql_data.schema import extract_sqlite_schema

_MAX_VALUE_CHARS = 40
_SAMPLE_TIMEOUT_S = 2.0


@dataclass(frozen=True)
class TableDoc:
    doc_id: str          # "<db_id>::<table>"
    table: str
    ddl: str             # CREATE TABLE-style text
    sample: str          # rendered sample rows ("" if none)
    references: tuple[str, ...]  # tables this one has foreign keys to
    index_text: str      # text used for lexical retrieval

    @property
    def prompt_text(self) -> str:
        return f"{self.ddl}\n{self.sample}".rstrip()


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _split_identifier(name: str) -> str:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return re.sub(r"[_\W]+", " ", spaced).lower().strip()


def _render_ddl(table: dict) -> str:
    lines = []
    for column in table["columns"]:
        parts = [f"  {_quote(column['name'])}", column.get("type") or ""]
        if column.get("primary_key"):
            parts.append("PRIMARY KEY")
        lines.append(" ".join(p for p in parts if p))
    for fk in table.get("foreign_keys", []):
        lines.append(
            f"  FOREIGN KEY ({_quote(fk['column'])}) REFERENCES "
            f"{_quote(fk['references_table'])}({_quote(str(fk['references_column']))})"
        )
    return f"CREATE TABLE {_quote(table['name'])} (\n" + ",\n".join(lines) + "\n);"


def _format_value(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bytes):
        return "<blob>"
    text = str(value).replace("\n", " ")
    return text if len(text) <= _MAX_VALUE_CHARS else text[: _MAX_VALUE_CHARS - 3] + "..."


def _sample_rows(connection: sqlite3.Connection, table: str, limit: int) -> str:
    if limit <= 0:
        return ""
    deadline = time.monotonic() + _SAMPLE_TIMEOUT_S
    connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        cursor = connection.execute(f"SELECT * FROM {_quote(table)} LIMIT {int(limit)}")
        rows = cursor.fetchall()
        columns = [d[0] for d in cursor.description or ()]
    except sqlite3.Error:
        return ""
    finally:
        connection.set_progress_handler(None, 0)
    if not rows:
        return f"/* table {table} has no rows */"
    body = "\n".join(" | ".join(_format_value(v) for v in row) for row in rows)
    return f"/* {len(rows)} sample rows from {table}:\n{' | '.join(columns)}\n{body}\n*/"


def _open_read_only(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    connection.text_factory = lambda b: b.decode(errors="replace")
    return connection


@lru_cache(maxsize=64)
def build_table_docs(db_path: str, db_id: str, sample_rows: int = 3) -> tuple[TableDoc, ...]:
    """All tables of one database as TableDocs (cached per db/sample size)."""
    schema = extract_sqlite_schema(db_path, db_id=db_id)
    connection = _open_read_only(Path(db_path))
    try:
        docs = []
        for table in schema["tables"]:
            name = table["name"]
            columns = " ".join(_split_identifier(c["name"]) for c in table["columns"])
            docs.append(
                TableDoc(
                    doc_id=f"{db_id}::{name}",
                    table=name,
                    ddl=_render_ddl(table),
                    sample=_sample_rows(connection, name, sample_rows),
                    references=tuple(sorted({fk["references_table"] for fk in table.get("foreign_keys", [])})),
                    index_text=f"{_split_identifier(name)} {_split_identifier(name)} {columns}",
                )
            )
        return tuple(docs)
    finally:
        connection.close()
