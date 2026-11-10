"""Sanity-check systems for validating the harness itself.

``GoldOracleSystem`` returns gold SQL and must NEVER be reported as a
result. It exists only to verify that correct SQL scores ~100% under
the official evaluator. ``AlwaysUnsupportedSystem`` must score 0%.
"""

from __future__ import annotations

from ..contracts import Budget, PredictionResult, QuestionInput, Status


class GoldOracleSystem:
    """Harness self-test only: reads gold SQL from the evaluator."""

    name = "sanity_gold_oracle"

    def __init__(self, evaluator) -> None:
        self._evaluator = evaluator

    def predict(self, question: QuestionInput, budget: Budget) -> PredictionResult:
        gold = self._evaluator.gold_sql(question.question_id)
        return PredictionResult(
            question_id=question.question_id,
            db_id=question.db_id,
            system_name=self.name,
            generated_sql=gold,
            status="success" if gold else Status.UNSUPPORTED,
            error_message=None if gold else "no gold SQL available for this instance",
        )


class AlwaysUnsupportedSystem:
    name = "sanity_always_unsupported"

    def predict(self, question: QuestionInput, budget: Budget) -> PredictionResult:
        return PredictionResult(
            question_id=question.question_id,
            db_id=question.db_id,
            system_name=self.name,
            generated_sql=None,
            status=Status.UNSUPPORTED,
        )


class ConstantSQLSystem:
    """Returns the same SQL for every question (exercises error paths)."""

    def __init__(self, sql: str, name: str = "sanity_constant_sql") -> None:
        self.sql = sql
        self.name = name

    def predict(self, question: QuestionInput, budget: Budget) -> PredictionResult:
        return PredictionResult(
            question_id=question.question_id,
            db_id=question.db_id,
            system_name=self.name,
            generated_sql=self.sql,
        )
