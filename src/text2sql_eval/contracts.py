"""Shared data contracts between the evaluation harness and every system.

Every system (classical, single-agent, two-agent) implements
``predict(question: QuestionInput, budget: Budget) -> PredictionResult``.

``QuestionInput`` deliberately has no gold SQL / gold result field: the
harness is the only component allowed to touch gold data, and only when
scoring. See docs/CONTRACTS.md for the human-readable version.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


class Status:
    """Pipeline status of one prediction (independent of correctness)."""

    EXECUTED = "executed"                # SQL ran; see `correct` for EX
    UNSUPPORTED = "unsupported"          # system declined to answer
    PARSE_ERROR = "parse_error"          # system could not produce SQL
    UNSAFE_SQL = "unsafe_sql"            # rejected by read-only validator
    COMPILE_ERROR = "compile_error"      # TRC -> view compilation failed
    VIEW_ERROR = "view_error"            # compiled view failed to create
    EXECUTION_ERROR = "execution_error"  # SQL raised a database error
    TIMEOUT = "timeout"                  # SQL exceeded the time limit
    SYSTEM_ERROR = "system_error"        # system raised an exception
    EVAL_ERROR = "eval_error"            # official evaluator failed

    # Statuses a *system* may report. The harness assigns the rest.
    SYSTEM_REPORTABLE = frozenset(
        {"success", UNSUPPORTED, PARSE_ERROR, COMPILE_ERROR, TIMEOUT}
    )

    ALL = frozenset(
        {
            EXECUTED, UNSUPPORTED, PARSE_ERROR, UNSAFE_SQL, COMPILE_ERROR,
            VIEW_ERROR, EXECUTION_ERROR, TIMEOUT, SYSTEM_ERROR, EVAL_ERROR,
        }
    )


@dataclass(frozen=True)
class EvidenceDoc:
    """A piece of benchmark-provided evidence (e.g. Spider2 external doc)."""

    doc_id: str
    text: str


@dataclass(frozen=True)
class QuestionInput:
    """What a system is allowed to see for one question. No gold data."""

    question_id: str
    benchmark: str
    db_id: str
    db_path: str
    question: str
    dialect: str = "sqlite"
    evidence: tuple[EvidenceDoc, ...] = ()


@dataclass(frozen=True)
class Budget:
    """Resource limits shared by all systems in a matched comparison."""

    max_retries: int = 3
    max_model_calls: int = 8
    max_total_input_tokens: int = 32_000
    sql_timeout_s: float = 30.0
    max_result_rows: int = 100_000


@dataclass
class PredictionResult:
    """What a system returns. Field names follow the team spec."""

    question_id: str
    db_id: str
    system_name: str
    generated_sql: str | None
    status: str = "success"
    model_calls: int = 0
    retry_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    max_input_tokens_single_call: int = 0
    wall_time_s: float = 0.0
    peak_gpu_memory_mb: float | None = None
    referenced_tables: list[str] = field(default_factory=list)
    referenced_columns: list[str] = field(default_factory=list)
    retrieved_document_ids: list[str] = field(default_factory=list)
    # Two-agent systems put compiled views here:
    #   {"views": [{"name": "Employee", "sql": "SELECT ..."}], ...}
    intermediate_artifacts: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Text2SQLSystem(Protocol):
    """Adapter interface every system implements."""

    name: str

    def predict(self, question: QuestionInput, budget: Budget) -> PredictionResult:
        ...


@dataclass(frozen=True)
class EvalOutcome:
    """Result of official scoring for one prediction."""

    correct: bool
    evaluator: str
    detail: str | None = None
