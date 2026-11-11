"""Single-agent LLM baseline: question + physical schema -> SQL, with repair.

Loop (bounded by Budget):
  prompt(schema, evidence, question) -> LLM -> extract SQL
  -> read-only check -> execute in sandbox (no gold involved)
  -> on failure, show the model its SQL and the error, retry (max 3).

All model calls, tokens and retries are accounted for. The final SQL is
returned even if its last execution failed, so the harness records the
failure category; the system never sees gold SQL or gold results.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from text2sql_eval.contracts import Budget, PredictionResult, QuestionInput, Status
from text2sql_eval.sandbox import run_query
from text2sql_eval.sql_safety import check_read_only
from text2sql_llm import GenerationParams, GpuPeakSampler, LLMClient, LLMError, Message
from text2sql_retrieval import SchemaSelection, build_table_docs, select_tables

from .prompts import (
    REPAIR_USER,
    SINGLE_AGENT_PROMPT_VERSION,
    SINGLE_AGENT_SYSTEM,
    render_single_agent_user,
)
from .sql_extraction import extract_sql

_MAX_LOGGED_OUTPUT = 4_000
_FEEDBACK_ROWS = 5


@dataclass(frozen=True)
class SingleAgentConfig:
    schema_token_budget: int = 6_000
    schema_top_k: int | None = None
    sample_rows: int = 3
    evidence_token_budget: int = 2_000
    retry_on_empty_result: bool = False
    chars_per_token: float = 3.5  # only for budgeting prompts; reports use server counts
    measure_gpu: bool = True
    params: GenerationParams = field(default_factory=GenerationParams)

    def estimate_tokens(self, text: str) -> int:
        return int(len(text) / self.chars_per_token) + 1


@dataclass
class _Attempt:
    sql: str | None
    problem: str | None
    input_tokens: int
    output_tokens: int
    latency_s: float
    finish_reason: str | None
    raw_output: str


@dataclass
class _Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    max_single_input: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.max_single_input = max(self.max_single_input, input_tokens)


class SingleAgentSystem:
    name = "single_agent_llm"

    def __init__(self, llm: LLMClient, config: SingleAgentConfig | None = None):
        self.llm = llm
        self.config = config or SingleAgentConfig()

    def describe(self) -> dict[str, Any]:
        return {
            "system": self.name,
            "prompt_version": SINGLE_AGENT_PROMPT_VERSION,
            "llm": self.llm.describe(),
            "config": asdict(self.config),
        }

    # ------------------------------------------------------------ context

    def _evidence_text(self, question: QuestionInput) -> str:
        if not question.evidence:
            return ""
        text = "\n\n".join(f"[{d.doc_id}]\n{d.text.strip()}" for d in question.evidence)
        max_chars = int(self.config.evidence_token_budget * self.config.chars_per_token)
        return text if len(text) <= max_chars else text[:max_chars] + "\n[... truncated]"

    def _select_schema(self, question: QuestionInput) -> tuple[SchemaSelection, int]:
        docs = build_table_docs(question.db_path, question.db_id, self.config.sample_rows)
        selection = select_tables(question.question, docs, self.config.schema_token_budget,
                                  self.config.estimate_tokens, self.config.schema_top_k)
        return selection, len(docs)

    def build_messages(self, question: QuestionInput) -> tuple[list[Message], SchemaSelection]:
        selection, total = self._select_schema(question)
        user = render_single_agent_user(
            question.question, [t.prompt_text for t in selection.tables],
            self._evidence_text(question), total,
        )
        system = SINGLE_AGENT_SYSTEM.format(dialect=question.dialect.upper() if question.dialect else "SQL")
        return [Message("system", system), Message("user", user)], selection

    # --------------------------------------------------------------- loop

    def _check(self, sql: str | None, question: QuestionInput, budget: Budget) -> str | None:
        """Return a problem description for the model, or None if SQL is OK."""
        if sql is None:
            return "No SQL code block was found in your answer."
        verdict = check_read_only(sql)
        if not verdict.ok:
            return f"The query is not a single read-only SELECT: {verdict.reason}."
        result = run_query(question.db_path, sql, timeout_s=budget.sql_timeout_s, max_rows=_FEEDBACK_ROWS)
        if not result.ok:
            if result.status == Status.TIMEOUT:
                return f"The query did not finish within {budget.sql_timeout_s:.0f} seconds; simplify it."
            return f"Database error: {result.error}"
        if self.config.retry_on_empty_result and not result.rows:
            return "The query ran but returned no rows; check filters and literal values against the sample rows."
        return None

    def _within_budget(self, usage: _Usage, messages: list[Message], budget: Budget) -> bool:
        if usage.calls >= budget.max_model_calls:
            return False
        upcoming = sum(self.config.estimate_tokens(m.content) for m in messages)
        return usage.input_tokens + upcoming <= budget.max_total_input_tokens

    def predict(self, question: QuestionInput, budget: Budget) -> PredictionResult:
        started = time.monotonic()
        messages, selection = self.build_messages(question)
        usage, attempts = _Usage(), []
        final_sql, stop_reason, error = None, "max_retries", None

        with GpuPeakSampler(enabled=self.config.measure_gpu) as gpu:
            for _ in range(1 + budget.max_retries):
                if not self._within_budget(usage, messages, budget):
                    stop_reason = "budget_exhausted"
                    break
                try:
                    generation = self.llm.generate(messages, self.config.params)
                except LLMError as exc:
                    stop_reason, error = "llm_error", str(exc)
                    break
                usage.add(generation.input_tokens, generation.output_tokens)
                sql = extract_sql(generation.text)
                problem = self._check(sql, question, budget)
                attempts.append(_Attempt(sql, problem, generation.input_tokens, generation.output_tokens,
                                         round(generation.latency_s, 3), generation.finish_reason,
                                         generation.text[:_MAX_LOGGED_OUTPUT]))
                if problem is None:
                    final_sql, stop_reason = sql, "ok"
                    break
                messages = messages[:2] + [
                    Message("assistant", generation.text),
                    Message("user", REPAIR_USER.format(sql=sql or "(none)", problem=problem)),
                ]

        last_sql = final_sql or next((a.sql for a in reversed(attempts) if a.sql), None)
        status = "success" if last_sql else (Status.SYSTEM_ERROR if stop_reason == "llm_error" else Status.PARSE_ERROR)
        return PredictionResult(
            question_id=question.question_id,
            db_id=question.db_id,
            system_name=self.name,
            generated_sql=last_sql,
            status=status,
            model_calls=usage.calls,
            retry_count=max(0, usage.calls - 1),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            max_input_tokens_single_call=usage.max_single_input,
            wall_time_s=time.monotonic() - started,
            peak_gpu_memory_mb=gpu.peak_mb,
            referenced_tables=_referenced_tables(last_sql, selection),
            retrieved_document_ids=[t.doc_id for t in selection.tables] + [d.doc_id for d in question.evidence],
            intermediate_artifacts={
                "prompt_version": SINGLE_AGENT_PROMPT_VERSION,
                "stop_reason": stop_reason,
                "schema_complete": selection.complete,
                "schema_estimated_tokens": selection.estimated_tokens,
                "bm25_ranking": list(selection.ranked_ids),
                "attempts": [asdict(a) for a in attempts],
            },
            error_message=error or (attempts[-1].problem if attempts and final_sql is None else None),
        )


def _referenced_tables(sql: str | None, selection: SchemaSelection) -> list[str]:
    """Lexical approximation: selected tables whose name appears in the SQL."""
    if not sql:
        return []
    lowered = sql.lower()
    return [t.table for t in selection.tables
            if re.search(rf'(?<![\w"]){re.escape(t.table.lower())}(?![\w"])|"{re.escape(t.table.lower())}"', lowered)]
