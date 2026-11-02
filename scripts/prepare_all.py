import json
from pathlib import Path

from text2sql_data.datasets import (
    load_spider1,
    load_spider2_lite,
    load_spider2_sqlite,
    load_spider2_local,
)

from text2sql_data.paths import (
    SPIDER1_DIR,
    SPIDER1_DATABASE_DIR,
    SPIDER2_DIR,
    SPIDER2_LOCAL_DB_DIR,
    NORMALIZED_DIR,
    SPIDER1_SCHEMA_DIR,
    SPIDER2_SCHEMA_DIR,
    RETRIEVAL_DIR,
    TRAINING_DIR,
)

from text2sql_data.schema import extract_sqlite_schema


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def example_to_dict(example):
    return example.to_dict()


def find_spider1_databases():
    """
    Spider 1 databases normally look like:

        database/
            concert_singer/
                concert_singer.sqlite
    """

    databases = {}

    for db_path in SPIDER1_DATABASE_DIR.rglob("*.sqlite"):
        databases[db_path.stem] = db_path

    return databases


def find_spider2_databases():
    """
    Spider 2 local DBs are stored directly in:

        spider2-localdb/*.sqlite
    """

    databases = {}

    for db_path in SPIDER2_LOCAL_DB_DIR.glob("*.sqlite"):
        databases[db_path.stem] = db_path

    return databases


def export_schemas(
    databases,
    output_dir,
    dataset_name,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    schemas = {}

    success = 0
    failed = 0

    for db_id, db_path in sorted(databases.items()):
        try:
            schema = extract_sqlite_schema(
                db_path,
                db_id=db_id,
            )

            schemas[db_id] = schema

            write_json(
                output_dir / f"{db_id}.json",
                schema,
            )

            success += 1

        except Exception as exc:
            failed += 1

            print(
                f"  WARNING: failed schema extraction "
                f"for {dataset_name}/{db_id}: {exc}"
            )

    return schemas, success, failed


def schema_to_documents(
    db_id,
    schema,
    dataset,
):
    """
    Convert schema JSON into simple table-level documents
    that Jay can later index using BM25.
    """

    documents = []

    tables = schema.get("tables", [])

    for table in tables:
        table_name = table.get("name", "")

        lines = [
            f"Database: {db_id}",
            f"Table: {table_name}",
            "",
            "Columns:",
        ]

        for column in table.get("columns", []):
            name = column.get("name", "")
            data_type = column.get("type", "")

            description = f"{name} {data_type}".strip()

            if column.get("primary_key"):
                description += " PRIMARY KEY"

            if column.get("nullable") is False:
                description += " NOT NULL"

            lines.append(description)

        foreign_keys = table.get(
            "foreign_keys",
            [],
        )

        if foreign_keys:
            lines.extend(
                [
                    "",
                    "Foreign Keys:",
                ]
            )

            for fk in foreign_keys:
                column = fk.get("column", "")
                ref_table = fk.get(
                    "references_table",
                    "",
                )
                ref_column = fk.get(
                    "references_column",
                    "",
                )

                lines.append(
                    f"{table_name}.{column} -> "
                    f"{ref_table}.{ref_column}"
                )

        documents.append(
            {
                "document_id": (
                    f"{dataset}::{db_id}::{table_name}"
                ),
                "dataset": dataset,
                "db_id": db_id,
                "table": table_name,
                "text": "\n".join(lines),
            }
        )

    return documents


def build_retrieval_documents(
    schemas,
    dataset,
):
    documents = []

    for db_id, schema in schemas.items():
        documents.extend(
            schema_to_documents(
                db_id,
                schema,
                dataset,
            )
        )

    return documents


def build_training_examples(
    train_examples,
    schemas,
):
    """
    Prepare Spider 1 examples for Nolan/Jay.

    ontology and declared_schema remain None because those
    belong to later components of the project.
    """

    rows = []

    for example in train_examples:
        schema = schemas.get(
            example.db_id
        )

        rows.append(
            {
                "example_id": example.example_id,
                "dataset": example.dataset,
                "db_id": example.db_id,
                "question": example.question,
                "gold_sql": example.gold_sql,
                "physical_schema": schema,
                "ontology": None,
                "declared_schema": None,
            }
        )

    return rows


def main():
    print("=" * 60)
    print("CSE 573 Text2SQL Data Preparation")
    print("=" * 60)

    # ----------------------------------------------------------
    # Spider 1
    # ----------------------------------------------------------

    print("\n[1/6] Loading Spider 1...")

    spider1_train, spider1_dev = (
        load_spider1(
            SPIDER1_DIR
        )
    )

    print(
        "  Train:",
        len(spider1_train),
    )

    print(
        "  Dev:",
        len(spider1_dev),
    )

    write_json(
        NORMALIZED_DIR
        / "spider1_train.json",
        [
            example_to_dict(x)
            for x in spider1_train
        ],
    )

    write_json(
        NORMALIZED_DIR
        / "spider1_dev.json",
        [
            example_to_dict(x)
            for x in spider1_dev
        ],
    )

    # ----------------------------------------------------------
    # Spider 2
    # ----------------------------------------------------------

    print("\n[2/6] Loading Spider 2.0-Lite...")

    spider2_all = load_spider2_lite(
        SPIDER2_DIR
    )

    spider2_sqlite = (
        load_spider2_sqlite(
            SPIDER2_DIR
        )
    )

    spider2_local = (
        load_spider2_local(
            SPIDER2_DIR
        )
    )

    print(
        "  Total:",
        len(spider2_all),
    )

    print(
        "  SQLite:",
        len(spider2_sqlite),
    )

    print(
        "  Locally executable:",
        len(spider2_local),
    )

    write_json(
        NORMALIZED_DIR
        / "spider2_lite_all.json",
        [
            example_to_dict(x)
            for x in spider2_all
        ],
    )

    write_json(
        NORMALIZED_DIR
        / "spider2_lite_sqlite.json",
        [
            example_to_dict(x)
            for x in spider2_local
        ],
    )

    # ----------------------------------------------------------
    # Spider 1 schemas
    # ----------------------------------------------------------

    print("\n[3/6] Extracting Spider 1 schemas...")

    spider1_databases = (
        find_spider1_databases()
    )

    print(
        "  SQLite DB files:",
        len(spider1_databases),
    )

    (
        spider1_schemas,
        spider1_success,
        spider1_failed,
    ) = export_schemas(
        spider1_databases,
        SPIDER1_SCHEMA_DIR,
        "spider1",
    )

    print(
        "  Extracted:",
        spider1_success,
    )

    print(
        "  Failed:",
        spider1_failed,
    )

    # ----------------------------------------------------------
    # Spider 2 schemas
    # ----------------------------------------------------------

    print("\n[4/6] Extracting Spider 2 schemas...")

    spider2_databases = (
        find_spider2_databases()
    )

    print(
        "  SQLite DB files:",
        len(spider2_databases),
    )

    (
        spider2_schemas,
        spider2_success,
        spider2_failed,
    ) = export_schemas(
        spider2_databases,
        SPIDER2_SCHEMA_DIR,
        "spider2_lite",
    )

    print(
        "  Extracted:",
        spider2_success,
    )

    print(
        "  Failed:",
        spider2_failed,
    )

    # ----------------------------------------------------------
    # Retrieval documents
    # ----------------------------------------------------------

    print("\n[5/6] Building retrieval documents...")

    spider1_docs = (
        build_retrieval_documents(
            spider1_schemas,
            "spider1",
        )
    )

    spider2_docs = (
        build_retrieval_documents(
            spider2_schemas,
            "spider2_lite",
        )
    )

    write_jsonl(
        RETRIEVAL_DIR
        / "spider1_schema_docs.jsonl",
        spider1_docs,
    )

    write_jsonl(
        RETRIEVAL_DIR
        / "spider2_schema_docs.jsonl",
        spider2_docs,
    )

    print(
        "  Spider 1 documents:",
        len(spider1_docs),
    )

    print(
        "  Spider 2 documents:",
        len(spider2_docs),
    )

    # ----------------------------------------------------------
    # Training examples
    # ----------------------------------------------------------

    print("\n[6/6] Preparing agent training examples...")

    training_rows = (
        build_training_examples(
            spider1_train,
            spider1_schemas,
        )
    )

    write_jsonl(
        TRAINING_DIR
        / "primary_agent_train.jsonl",
        training_rows,
    )

    with_schema = sum(
        1
        for row in training_rows
        if row["physical_schema"] is not None
    )

    print(
        "  Training examples:",
        len(training_rows),
    )

    print(
        "  Examples with physical schema:",
        with_schema,
    )

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    print("\n" + "=" * 60)
    print("Preparation complete")
    print("=" * 60)

    print(
        "\nNormalized data:",
        NORMALIZED_DIR,
    )

    print(
        "Spider 1 schemas:",
        SPIDER1_SCHEMA_DIR,
    )

    print(
        "Spider 2 schemas:",
        SPIDER2_SCHEMA_DIR,
    )

    print(
        "Retrieval documents:",
        RETRIEVAL_DIR,
    )

    print(
        "Training examples:",
        TRAINING_DIR,
    )


if __name__ == "__main__":
    main()