"""Versioned prompt templates. Change the text -> bump the version.

The version string is stored in every prediction's artifacts, so results
can always be traced back to the exact prompt that produced them.
"""

from __future__ import annotations

from typing import Sequence

SINGLE_AGENT_PROMPT_VERSION = "single_agent/v1"  # kept for v1 reruns / ablations
SINGLE_AGENT_PROMPT_VERSION_V2 = "single_agent/v2"

SINGLE_AGENT_SYSTEM = (
    "You are an expert data analyst who writes correct {dialect} SQL.\n"
    "Rules:\n"
    "- Use only tables and columns that appear in the schema.\n"
    "- Write exactly one read-only query (SELECT or WITH ... SELECT).\n"
    "- Return only the columns the question asks for, in the order asked.\n"
    "- Quote identifiers that contain spaces or special characters with double quotes.\n"
    "- Answer with the SQL inside a single ```sql code block and nothing else."
)

SINGLE_AGENT_USER = (
    "Database schema ({n_tables} tables{partial_note}):\n"
    "{schema}\n"
    "{evidence_block}"
    "Question: {question}\n\n"
    "SQL:"
)

# v2: same rules plus column ownership and shared dialect guidance
# (text2sql_agents.dialect_hints, also used by the two-agent system).
SINGLE_AGENT_SYSTEM_V2 = (
    SINGLE_AGENT_SYSTEM.replace(
        "- Use only tables and columns that appear in the schema.\n",
        "- Use only tables and columns that appear in the schema.\n"
        "- A column belongs only to the table it is listed under: before writing alias.column,\n"
        "  check that this column is in that alias's table, and join the owning table if it is not.\n",
    )
    + "{dialect_hints_block}"
)

EVIDENCE_BLOCK = "\nReference documentation provided with the question:\n{evidence}\n\n"

REPAIR_USER = (
    "The previous SQL failed.\n"
    "SQL:\n```sql\n{sql}\n```\n"
    "Problem: {problem}\n\n"
    "Write a corrected query for the same question. "
    "Answer with the SQL inside a single ```sql code block and nothing else."
)

REPAIR_USER_V2 = (
    "The previous SQL failed.\n"
    "SQL:\n```sql\n{sql}\n```\n"
    "Problem: {problem}\n"
    "{guidance}"
    "{history}"
    "\nWrite a corrected query for the same question. "
    "Answer with the SQL inside a single ```sql code block and nothing else."
)

REPEATED_SQL_NOTE = (
    "This is the same SQL as an earlier attempt that already failed. "
    "Do not repeat it: change the tables, joins or functions involved."
)


def render_repair_v2(sql: str, problem: str, guidance: str, failed_before: Sequence[str]) -> str:
    history = ""
    if failed_before:
        listed = "\n".join(f"- {' '.join(q.split())}" for q in failed_before)
        history = f"Earlier queries that also failed (do not repeat them):\n{listed}\n"
    return REPAIR_USER_V2.format(
        sql=sql, problem=problem,
        guidance=f"How to fix: {guidance}\n" if guidance else "",
        history=history,
    )


def render_single_agent_system_v2(dialect: str, hints: str) -> str:
    return SINGLE_AGENT_SYSTEM_V2.format(
        dialect=dialect, dialect_hints_block=f"\n\n{hints}" if hints else "",
    )


def render_single_agent_user(
    question: str,
    schema_blocks: Sequence[str],
    evidence: str,
    total_tables: int,
) -> str:
    partial = "" if len(schema_blocks) == total_tables else f", showing the {len(schema_blocks)} most relevant"
    return SINGLE_AGENT_USER.format(
        n_tables=total_tables,
        partial_note=partial,
        schema="\n\n".join(schema_blocks),
        evidence_block=EVIDENCE_BLOCK.format(evidence=evidence) if evidence else "\n",
        question=question.strip(),
    )
