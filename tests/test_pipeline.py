import json
import sqlite3

from text2sql_data.baseline import (
    ClassicalBaseline,
    lexical_schema_match,
    render_template,
)
from text2sql_data.datasets import (
    load_normalized,
    load_spider1,
    load_spider2_lite,
    load_spider2_local,
    load_spider2_sqlite,
)
from text2sql_data.intent import sql_to_intent
from text2sql_data.models import (
    BaselinePrediction,
    Text2SQLExample,
)
from text2sql_data.paths import (
    SPIDER1_DIR,
    SPIDER2_DIR,
)
from text2sql_data.schema import (
    extract_sqlite_schema,
)


def test_text2sql_example():
    example = Text2SQLExample(
        example_id="test_1",
        dataset="test",
        split="train",
        db_id="test_db",
        question="How many users are there?",
        gold_sql="SELECT COUNT(*) FROM users",
        dialect="sqlite",
        metadata={"source": "test"},
    )

    result = example.to_dict()

    assert result["example_id"] == "test_1"
    assert result["db_id"] == "test_db"
    assert result["metadata"]["source"] == "test"


def test_spider1_loader():
    train, dev = load_spider1(
        SPIDER1_DIR
    )

    assert len(train) == 8659
    assert len(dev) == 1034

    first = train[0]

    assert first.dataset == "spider1"
    assert first.split == "train"
    assert first.db_id
    assert first.question
    assert first.gold_sql
    assert first.dialect == "sqlite"


def test_spider2_loader():
    examples = load_spider2_lite(
        SPIDER2_DIR
    )

    assert len(examples) == 547

    sqlite_examples = [
        x
        for x in examples
        if x.dialect == "sqlite"
    ]

    assert len(sqlite_examples) == 123


def test_spider2_local_loader():
    sqlite_examples = load_spider2_sqlite(
        SPIDER2_DIR
    )

    local_examples = load_spider2_local(
        SPIDER2_DIR
    )

    assert len(sqlite_examples) == 123
    assert len(local_examples) == 123

    for example in local_examples:
        assert example.dialect == "sqlite"

        assert example.metadata[
            "locally_executable"
        ] is True

        assert example.metadata[
            "local_db_path"
        ] is not None


def test_normalized_loader():
    train = load_normalized(
        "outputs/normalized/spider1_train.json"
    )

    dev = load_normalized(
        "outputs/normalized/spider1_dev.json"
    )

    assert len(train) == 8659
    assert len(dev) == 1034

    assert train[0].question
    assert train[0].gold_sql
    assert train[0].db_id


def test_intent_detection():
    assert (
        sql_to_intent(
            "SELECT COUNT(*) FROM users"
        )
        == "COUNT"
    )

    assert (
        sql_to_intent(
            "SELECT COUNT(*) FROM users WHERE age > 20"
        )
        == "COUNT_FILTER"
    )

    assert (
        sql_to_intent(
            "SELECT SUM(price) FROM orders"
        )
        == "SUM"
    )

    assert (
        sql_to_intent(
            "SELECT name FROM users"
        )
        == "SELECT"
    )

    assert (
        sql_to_intent(
            "SELECT name FROM users WHERE age > 20"
        )
        == "FILTER"
    )


def test_schema_extraction(tmp_path):
    db_path = tmp_path / "company.sqlite"

    connection = sqlite3.connect(
        str(db_path)
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE departments (
            department_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE employees (
            employee_id INTEGER PRIMARY KEY,
            name TEXT,
            department_id INTEGER,
            FOREIGN KEY(department_id)
                REFERENCES departments(department_id)
        )
        """
    )

    connection.commit()
    connection.close()

    schema = extract_sqlite_schema(
        db_path,
        db_id="company",
    )

    assert schema["db_id"] == "company"
    assert schema["dialect"] == "sqlite"

    table_names = {
        table["name"]
        for table in schema["tables"]
    }

    assert "employees" in table_names
    assert "departments" in table_names

    employees = next(
        table
        for table in schema["tables"]
        if table["name"] == "employees"
    )

    assert len(
        employees["foreign_keys"]
    ) == 1

    fk = employees["foreign_keys"][0]

    assert (
        fk["references_table"]
        == "departments"
    )


def test_lexical_schema_match():
    schema = {
        "tables": [
            {
                "name": "employees",
                "columns": [
                    {
                        "name": "employee_id",
                        "type": "INTEGER",
                    },
                    {
                        "name": "salary",
                        "type": "REAL",
                    },
                ],
            },
            {
                "name": "departments",
                "columns": [
                    {
                        "name": "department_id",
                        "type": "INTEGER",
                    }
                ],
            },
        ]
    }

    table, column = lexical_schema_match(
        "What is the salary of employees?",
        schema,
    )

    assert table == "employees"
    assert column == "salary"


def test_sql_templates():
    assert (
        render_template(
            "COUNT",
            "employees",
            None,
        )
        == 'SELECT COUNT(*) FROM "employees";'
    )

    assert (
        render_template(
            "SUM",
            "employees",
            "salary",
        )
        == 'SELECT SUM("salary") FROM "employees";'
    )

    assert (
        render_template(
            "FILTER",
            "employees",
            "salary",
        )
        is None
    )


def test_baseline_prediction():
    train = [
        Text2SQLExample(
            example_id="train_1",
            dataset="test",
            split="train",
            db_id="company",
            question="How many employees are there?",
            gold_sql=(
                "SELECT COUNT(*) FROM employees"
            ),
        ),
        Text2SQLExample(
            example_id="train_2",
            dataset="test",
            split="train",
            db_id="company",
            question="Show employee salary",
            gold_sql=(
                "SELECT salary FROM employees"
            ),
        ),
        Text2SQLExample(
            example_id="train_3",
            dataset="test",
            split="train",
            db_id="company",
            question="List employees",
            gold_sql=(
                "SELECT * FROM employees"
            ),
        ),
    ]

    schema = {
        "tables": [
            {
                "name": "employees",
                "columns": [
                    {
                        "name": "employee_id",
                        "type": "INTEGER",
                    },
                    {
                        "name": "salary",
                        "type": "REAL",
                    },
                ],
            }
        ]
    }

    model = ClassicalBaseline(
        n_neighbors=1
    )

    model.fit(train)

    test_example = Text2SQLExample(
        example_id="dev_1",
        dataset="test",
        split="dev",
        db_id="company",
        question="How many employees are there?",
        gold_sql=(
            "SELECT COUNT(*) FROM employees"
        ),
    )

    prediction = model.predict(
        test_example,
        schema,
    )

    assert isinstance(
        prediction,
        BaselinePrediction,
    )

    assert prediction.example_id == "dev_1"
    assert prediction.predicted_intent == "COUNT"
    assert prediction.supported is True
    assert prediction.predicted_sql is not None