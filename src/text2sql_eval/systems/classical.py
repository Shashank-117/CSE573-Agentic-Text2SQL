"""Adapter: Shashank's classical TF-IDF + k-NN baseline -> harness contract.

The model is trained only on Spider 1 *train*. At prediction time it gets
the gold-free QuestionInput plus the physical schema extracted from the
database file; the Text2SQLExample it receives has ``gold_sql=None``.
"""

from __future__ import annotations

import time
from functools import lru_cache

from text2sql_data.baseline import ClassicalBaseline
from text2sql_data.datasets import load_spider1
from text2sql_data.models import Text2SQLExample
from text2sql_data.schema import extract_sqlite_schema

from ..contracts import Budget, PredictionResult, QuestionInput, Status
from ..paths import SPIDER1_DIR


@lru_cache(maxsize=256)
def _schema(db_path: str, db_id: str) -> dict:
    return extract_sqlite_schema(db_path, db_id=db_id)


class ClassicalBaselineSystem:
    name = "classical_tfidf_knn"

    def __init__(self, n_neighbors: int = 3) -> None:
        train, _ = load_spider1(SPIDER1_DIR)
        self._model = ClassicalBaseline(n_neighbors=n_neighbors).fit(train)

    def predict(self, question: QuestionInput, budget: Budget) -> PredictionResult:
        started = time.monotonic()
        example = Text2SQLExample(
            example_id=question.question_id,
            dataset=question.benchmark,
            split="evaluation",
            db_id=question.db_id,
            question=question.question,
            gold_sql=None,
        )
        prediction = self._model.predict(example, _schema(question.db_path, question.db_id))
        table, column = prediction.matched_table, prediction.matched_column
        return PredictionResult(
            question_id=question.question_id,
            db_id=question.db_id,
            system_name=self.name,
            generated_sql=prediction.predicted_sql,
            status="success" if prediction.supported else Status.UNSUPPORTED,
            wall_time_s=time.monotonic() - started,
            referenced_tables=[table] if table else [],
            referenced_columns=[f"{table}.{column}"] if table and column else [],
            intermediate_artifacts={
                "predicted_intent": prediction.predicted_intent,
                "nearest_train_question": prediction.nearest_question,
            },
            error_message=None if prediction.supported else f"intent {prediction.predicted_intent} not supported",
        )
