"""Extract a SQL query from free-form model output."""

from __future__ import annotations

import re

_FENCED = re.compile(r"```[ \t]*(?:sql|sqlite)?[ \t]*\n(.*?)```", re.IGNORECASE | re.DOTALL)
_UNCLOSED = re.compile(r"```[ \t]*(?:sql|sqlite)?[ \t]*\n(.*)$", re.IGNORECASE | re.DOTALL)
_STARTS_LIKE_SQL = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)
_SQL_START = re.compile(r"\b(SELECT|WITH)\b", re.IGNORECASE)


def extract_sql(text: str | None) -> str | None:
    """Return the SQL in ``text`` or None.

    Preference: last fenced block, then an unclosed fence (output cut off
    at max tokens), then raw text that starts with SELECT/WITH, then the
    text from the first SELECT/WITH keyword onward.
    """
    if not text or not text.strip():
        return None
    blocks = [b.strip() for b in _FENCED.findall(text) if b.strip()]
    if blocks:
        return blocks[-1]
    unclosed = _UNCLOSED.search(text)
    if unclosed and unclosed.group(1).strip():
        return unclosed.group(1).strip()
    if _STARTS_LIKE_SQL.match(text):
        return text.strip()
    start = _SQL_START.search(text)
    return text[start.start():].strip() if start else None
