import sqlite3
from pathlib import Path
from typing import Dict, Iterable

from .models import BaselinePrediction


def execute_sql(db_path: str | Path, sql: str):
    connection = sqlite3.connect(str(db_path))

    try:
        cursor = connection.cursor()
        cursor.execute(sql)
        return cursor.fetchall()

    finally:
        connection.close()


def execution_match(
    db_path: str | Path,
    predicted_sql: str | None,
    gold_sql: str | None,
) -> bool:

    if not predicted_sql or not gold_sql:
        return False

    try:
        predicted_result = execute_sql(
            db_path,
            predicted_sql,
        )

        gold_result = execute_sql(
            db_path,
            gold_sql,
        )

        # ORDER BY means row order is part of the expected result.
        if "ORDER BY" in gold_sql.upper():
            return predicted_result == gold_result

        # Otherwise compare rows without depending on order.
        try:
            return sorted(predicted_result) == sorted(gold_result)

        except TypeError:
            # Fallback for values that cannot be directly sorted.
            return {
                repr(row)
                for row in predicted_result
            } == {
                repr(row)
                for row in gold_result
            }

    except sqlite3.Error:
        return False

def summarize(
    predictions: Iterable[BaselinePrediction],
    db_paths: Dict[str, str | Path],
) -> Dict:

    predictions = list(predictions)

    total = len(predictions)

    supported = sum(
        1
        for prediction in predictions
        if prediction.supported
    )

    execution_success = 0
    execution_correct = 0
    missing_database = 0

    for prediction in predictions:

        db_path = db_paths.get(
            prediction.db_id
        )

        if db_path is None:
            missing_database += 1
            continue

        if not prediction.predicted_sql:
            continue

        # First check whether the predicted SQL executes.
        try:
            execute_sql(
                db_path,
                prediction.predicted_sql,
            )

            execution_success += 1

        except sqlite3.Error:
            continue

        # Then compare its result against gold SQL.
        if execution_match(
            db_path,
            prediction.predicted_sql,
            prediction.gold_sql,
        ):
            execution_correct += 1

    coverage = (
        supported / total
        if total
        else 0.0
    )

    execution_success_rate = (
        execution_success / total
        if total
        else 0.0
    )

    execution_accuracy_overall = (
        execution_correct / total
        if total
        else 0.0
    )

    execution_accuracy_supported = (
        execution_correct / supported
        if supported
        else 0.0
    )

    return {
        "total_questions": total,
        "supported_questions": supported,
        "unsupported_questions": (
            total - supported
        ),
        "coverage": coverage,

        "execution_success": execution_success,
        "execution_success_rate": (
            execution_success_rate
        ),

        "execution_correct": execution_correct,

        "execution_accuracy_overall": (
            execution_accuracy_overall
        ),

        "execution_accuracy_supported": (
            execution_accuracy_supported
        ),

        "missing_database": missing_database,
    }