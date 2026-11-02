import argparse
from pathlib import Path

from text2sql_data.datasets import export_agent_training_examples, load_normalized, normalize_spider_dataset
from text2sql_data.schema import export_all_schemas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/spider")
    parser.add_argument("--output-dir", default="outputs/normalized")
    parser.add_argument("--schema-output", default="outputs/schemas")
    args = parser.parse_args()

    counts = normalize_spider_dataset(args.data_dir, args.output_dir)
    db_root = Path(args.data_dir) / "database"
    schema_count = export_all_schemas(db_root, args.schema_output) if db_root.exists() else 0

    train_path = Path(args.output_dir) / "spider_train.json"
    if train_path.exists():
        export_agent_training_examples(load_normalized(train_path), Path(args.output_dir) / "primary_agent_train.json")

    print(f"Normalized Spider: {counts}")
    print(f"Exported schemas: {schema_count}")


if __name__ == "__main__":
    main()
