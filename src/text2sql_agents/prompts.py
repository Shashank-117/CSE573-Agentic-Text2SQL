"""Versioned prompt templates. Change the text -> bump the version.

The version string is stored in every prediction's artifacts, so results
can always be traced back to the exact prompt that produced them.
"""

from __future__ import annotations

from typing import Sequence

SINGLE_AGENT_PROMPT_VERSION = "single_agent/v1"

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

EVIDENCE_BLOCK = "\nReference documentation provided with the question:\n{evidence}\n\n"

REPAIR_USER = (
    "The previous SQL failed.\n"
    "SQL:\n```sql\n{sql}\n```\n"
    "Problem: {problem}\n\n"
    "Write a corrected query for the same question. "
    "Answer with the SQL inside a single ```sql code block and nothing else."
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
