"""Isolated, read-only SQLite execution.

Safety layers (defense in depth):
1. ``sql_safety.check_read_only`` rejects non-SELECT text up front.
2. The database file is opened with ``mode=ro`` so SQLite refuses writes.
3. An authorizer denies every action except reads, functions and creating
   TEMP views, so even exotic statements cannot modify anything.

Views live only in the connection's TEMP schema (in memory). Each call to
``run_query`` opens a fresh connection and closes it afterwards, so views
never leak between questions, attempts or runs.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .contracts import Status
from .sql_safety import check_read_only

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_PROGRESS_STEPS = 10_000

_ALWAYS_ALLOWED = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
)
_TEMP_SCHEMA_TABLE = "sqlite_temp_master"


@dataclass(frozen=True)
class ViewDefinition:
    name: str
    sql: str  # a single SELECT statement


@dataclass(frozen=True)
class ExecResult:
    status: str  # Status.EXECUTED / EXECUTION_ERROR / TIMEOUT / UNSAFE_SQL / VIEW_ERROR
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[Any, ...], ...] = ()
    truncated: bool = False
    error: str | None = None
    elapsed_s: float = 0.0
    views_created: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.status == Status.EXECUTED


def _authorizer(allow_view_creation: bool):
    def authorize(action: int, arg1: Any, arg2: Any, db_name: Any, source: Any) -> int:
        if action in _ALWAYS_ALLOWED:
            return sqlite3.SQLITE_OK
        if allow_view_creation:
            if action == sqlite3.SQLITE_CREATE_TEMP_VIEW:
                return sqlite3.SQLITE_OK
            # Creating a TEMP view writes one row into the TEMP schema table.
            if action in (sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE) and arg1 == _TEMP_SCHEMA_TABLE:
                return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    return authorize


def _connect_read_only(db_path: Path) -> sqlite3.Connection:
    if not db_path.is_file():
        raise FileNotFoundError(f"database not found: {db_path}")
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
    connection.text_factory = lambda b: b.decode(errors="replace")
    return connection


def _install_deadline(connection: sqlite3.Connection, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    connection.set_progress_handler(lambda: int(time.monotonic() > deadline), _PROGRESS_STEPS)


def _create_views(connection: sqlite3.Connection, views: Iterable[ViewDefinition]) -> list[str]:
    created: list[str] = []
    connection.set_authorizer(_authorizer(allow_view_creation=True))
    for view in views:
        if not _IDENTIFIER.match(view.name):
            raise ValueError(f"invalid view name: {view.name!r}")
        verdict = check_read_only(view.sql)
        if not verdict.ok:
            raise ValueError(f"view {view.name} is not read-only: {verdict.reason}")
        connection.execute(f'CREATE TEMP VIEW "{view.name}" AS {verdict.sql}')
        created.append(view.name)
    connection.set_authorizer(_authorizer(allow_view_creation=False))
    return created


def _classify_error(exc: sqlite3.Error) -> str:
    return Status.TIMEOUT if "interrupted" in str(exc).lower() else Status.EXECUTION_ERROR


def run_query(
    db_path: str | Path,
    sql: str | None,
    views: Sequence[ViewDefinition] = (),
    timeout_s: float = 30.0,
    max_rows: int = 100_000,
) -> ExecResult:
    """Execute one read-only query (optionally over TEMP views) in isolation."""
    verdict = check_read_only(sql)
    if not verdict.ok:
        return ExecResult(Status.UNSAFE_SQL, error=verdict.reason)

    started = time.monotonic()
    connection = _connect_read_only(Path(db_path))
    try:
        _install_deadline(connection, timeout_s)
        try:
            created = _create_views(connection, views)
        except (sqlite3.Error, ValueError) as exc:
            timed_out = isinstance(exc, sqlite3.Error) and _classify_error(exc) == Status.TIMEOUT
            status = Status.TIMEOUT if timed_out else Status.VIEW_ERROR
            return ExecResult(status, error=f"view creation failed: {exc}",
                              elapsed_s=time.monotonic() - started)

        try:
            cursor = connection.execute(verdict.sql)
            rows = cursor.fetchmany(max_rows + 1)
        except sqlite3.Error as exc:
            return ExecResult(_classify_error(exc), error=str(exc),
                              elapsed_s=time.monotonic() - started,
                              views_created=tuple(created))

        columns = tuple(d[0] for d in (cursor.description or ()))
        return ExecResult(
            Status.EXECUTED,
            columns=columns,
            rows=tuple(tuple(r) for r in rows[:max_rows]),
            truncated=len(rows) > max_rows,
            elapsed_s=time.monotonic() - started,
            views_created=tuple(created),
        )
    finally:
        connection.close()


def database_fingerprint(db_path: str | Path) -> tuple[int, int]:
    """Cheap change detector: (size in bytes, mtime in ns)."""
    stat = Path(db_path).stat()
    return stat.st_size, stat.st_mtime_ns
