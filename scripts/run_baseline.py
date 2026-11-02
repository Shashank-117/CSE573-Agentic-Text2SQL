import argparse
from pathlib import Path

from text2sql_data.baseline import (
    ClassicalBaseline,
)

from text2sql_data.datasets import (
    load_normalized,
)

from text2sql_data.evaluate import (
    summarize,
)

from text2sql_data.io_utils import (
    read_json,
    write_json,
)

from text2sql_data.schema import (
    find_sqlite_databases,
)

from text2sql_data.paths import (
    NORMALIZED_DIR,
    SPIDER1_SCHEMA_DIR,
    SPIDER1_DATABASE_DIR,
    BASELINE_DIR,
)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run the classical TF-IDF + k-NN "
            "Text2SQL baseline."
        )
    )

    parser.add_argument(
        "--train",
        default=str(
            NORMALIZED_DIR
            / "spider1_train.json"
        ),
    )

    parser.add_argument(
        "--dev",
        default=str(
            NORMALIZED_DIR
            / "spider1_dev.json"
        ),
    )

    parser.add_argument(
        "--schemas",
        default=str(
            SPIDER1_SCHEMA_DIR
        ),
    )

    parser.add_argument(
        "--database-root",
        default=str(
            SPIDER1_DATABASE_DIR
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            BASELINE_DIR
            / "spider1_dev_predictions.json"
        ),
    )

    args = parser.parse_args()

    print("=" * 60)
    print("CSE 573 Classical Text2SQL Baseline")
    print("=" * 60)

    # ------------------------------------------------------
    # Load normalized data
    # ------------------------------------------------------

    print("\n[1/5] Loading normalized datasets...")

    train = load_normalized(
        args.train
    )

    dev = load_normalized(
        args.dev
    )

    print(
        "  Train:",
        len(train),
    )

    print(
        "  Dev:",
        len(dev),
    )

    # ------------------------------------------------------
    # Train baseline
    # ------------------------------------------------------

    print("\n[2/5] Training TF-IDF + k-NN...")

    model = ClassicalBaseline(
        n_neighbors=3
    )

    model.fit(train)

    print("  Training complete.")

    # ------------------------------------------------------
    # Generate predictions
    # ------------------------------------------------------

    print("\n[3/5] Generating predictions...")

    schema_root = Path(
        args.schemas
    )

    predictions = []

    missing_schemas = 0

    for index, example in enumerate(dev):

        schema_path = (
            schema_root
            / f"{example.db_id}.json"
        )

        if not schema_path.exists():
            missing_schemas += 1
            continue

        schema = read_json(
            schema_path
        )

        prediction = model.predict(
            example,
            schema,
        )

        predictions.append(
            prediction
        )

        if (
            (index + 1) % 100 == 0
            or index + 1 == len(dev)
        ):
            print(
                f"  Processed "
                f"{index + 1}/{len(dev)}"
            )

    print(
        "  Predictions:",
        len(predictions),
    )

    print(
        "  Missing schemas:",
        missing_schemas,
    )

    # ------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------

    print("\n[4/5] Saving predictions...")

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json(
        [
            prediction.to_dict()
            for prediction in predictions
        ],
        output_path,
    )

    print(
        "  Saved:",
        output_path,
    )

    # ------------------------------------------------------
    # Build DB map
    # ------------------------------------------------------

    db_paths = {}

    for db_path in find_sqlite_databases(
        args.database_root
    ):

        # Spider 1 structure:
        #
        # database/
        #   concert_singer/
        #       concert_singer.sqlite

        db_id = db_path.parent.name

        db_paths[db_id] = db_path

    # ------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------

    print("\n[5/5] Evaluating predictions...")

    metrics = summarize(
        predictions,
        db_paths,
    )

    metrics["missing_schema"] = (
        missing_schemas
    )

    metrics["training_examples"] = (
        len(train)
    )

    metrics["development_examples"] = (
        len(dev)
    )

    metrics_path = (
        output_path.with_name(
            "metrics.json"
        )
    )

    write_json(
        metrics,
        metrics_path,
    )

    print("\n" + "=" * 60)
    print("Baseline Results")
    print("=" * 60)

    for key, value in metrics.items():

        if isinstance(value, float):
            print(
                f"{key}: {value:.4f}"
            )
        else:
            print(
                f"{key}: {value}"
            )

    print(
        "\nPredictions:",
        output_path,
    )

    print(
        "Metrics:",
        metrics_path,
    )


if __name__ == "__main__":
    main()