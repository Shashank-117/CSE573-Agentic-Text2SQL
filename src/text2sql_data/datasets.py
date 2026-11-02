import json
from pathlib import Path
from typing import Dict, List, Optional, Set

from .models import Text2SQLExample


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def load_json(path: Path):
    """Load a normal JSON file."""
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    """Load a JSONL file."""
    path = Path(path)

    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows


# ---------------------------------------------------------------------
# Spider 1.0
# ---------------------------------------------------------------------

def _normalize_spider1_rows(
    rows: List[dict],
    split: str,
    start_index: int = 0,
) -> List[Text2SQLExample]:
    """
    Convert raw Spider 1.0 rows into the common Text2SQLExample format.
    """

    examples = []

    for index, row in enumerate(rows, start=start_index):
        example = Text2SQLExample(
            example_id=f"spider1_{split}_{index:06d}",
            dataset="spider1",
            split=split,
            db_id=row["db_id"],
            question=row["question"],
            gold_sql=row.get("query"),
            dialect="sqlite",
            external_knowledge=None,
            metadata={
                "question_toks": row.get("question_toks"),
                "query_toks": row.get("query_toks"),
                "query_toks_no_value": row.get(
                    "query_toks_no_value"
                ),
                "sql": row.get("sql"),
            },
        )

        examples.append(example)

    return examples


def load_spider1(
    spider_root: Path,
    include_train_others: bool = True,
):
    """
    Load Spider 1.0 train and dev data.

    Expected layout:

        spider_root/
            train_spider.json
            train_others.json
            dev.json
            database/
            tables.json

    Returns:
        (train_examples, dev_examples)
    """

    spider_root = Path(spider_root)

    train_path = spider_root / "train_spider.json"
    train_others_path = spider_root / "train_others.json"
    dev_path = spider_root / "dev.json"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Spider train file not found: {train_path}"
        )

    if not dev_path.exists():
        raise FileNotFoundError(
            f"Spider dev file not found: {dev_path}"
        )

    train_rows = load_json(train_path)
    dev_rows = load_json(dev_path)

    train_examples = _normalize_spider1_rows(
        train_rows,
        split="train",
    )

    # Spider also provides train_others.json.
    # Keep it in the training split, but give every example
    # a unique ID.
    if include_train_others and train_others_path.exists():
        other_rows = load_json(train_others_path)

        other_examples = _normalize_spider1_rows(
            other_rows,
            split="train",
            start_index=len(train_examples),
        )

        train_examples.extend(other_examples)

    dev_examples = _normalize_spider1_rows(
        dev_rows,
        split="dev",
    )

    return train_examples, dev_examples


# Backward-compatible name in case older starter scripts use it.
def load_spider(
    spider_root: Path,
    include_train_others: bool = True,
):
    return load_spider1(
        spider_root,
        include_train_others=include_train_others,
    )


# ---------------------------------------------------------------------
# Spider 2.0-Lite
# ---------------------------------------------------------------------

def _directory_names(path: Path) -> Set[str]:
    """
    Return names of all direct child directories.
    """

    path = Path(path)

    if not path.exists():
        return set()

    return {
        item.name
        for item in path.iterdir()
        if item.is_dir()
    }


def get_spider2_database_sets(
    spider2_root: Path,
) -> Dict[str, Set[str]]:
    """
    Discover database -> dialect membership from the official
    Spider 2.0-Lite resource directory.

    Expected structure:

        spider2-lite/
            resource/
                databases/
                    sqlite/
                    snowflake/
                    bigquery/

    Each dialect directory contains one directory per database.
    """

    spider2_root = Path(spider2_root)

    database_root = (
        spider2_root
        / "resource"
        / "databases"
    )

    return {
        "sqlite": _directory_names(
            database_root / "sqlite"
        ),
        "snowflake": _directory_names(
            database_root / "snowflake"
        ),
        "bigquery": _directory_names(
            database_root / "bigquery"
        ),
    }


def get_spider2_dialect(
    db_id: str,
    database_sets: Dict[str, Set[str]],
) -> str:
    """
    Determine the SQL dialect/database platform for a Spider2 DB.
    """

    if db_id in database_sets.get("sqlite", set()):
        return "sqlite"

    if db_id in database_sets.get("snowflake", set()):
        return "snowflake"

    if db_id in database_sets.get("bigquery", set()):
        return "bigquery"

    return "unknown"


def get_spider2_local_db_path(
    db_id: str,
    spider2_root: Path,
) -> Optional[Path]:
    """
    Return the local SQLite database path for a Spider2 database
    when one exists.

    Expected location:

        resource/databases/spider2-localdb/<db_id>.sqlite
    """

    spider2_root = Path(spider2_root)

    local_db_dir = (
        spider2_root
        / "resource"
        / "databases"
        / "spider2-localdb"
    )

    direct_match = local_db_dir / f"{db_id}.sqlite"

    if direct_match.exists():
        return direct_match

    # Case-insensitive fallback.
    # This protects us from names such as:
    # Baseball.sqlite vs a differently-cased db identifier.
    if local_db_dir.exists():
        target = db_id.lower()

        for db_path in local_db_dir.glob("*.sqlite"):
            if db_path.stem.lower() == target:
                return db_path

    return None


def load_spider2_lite(
    spider2_root: Path,
) -> List[Text2SQLExample]:
    """
    Load the complete Spider 2.0-Lite benchmark.

    The downloaded release has rows such as:

        {
            "instance_id": "local002",
            "db": "E_commerce",
            "question": "...",
            "external_knowledge": null
        }

    Dialect information is determined from the official
    resource/databases directories instead of guessing from
    instance_id.
    """

    spider2_root = Path(spider2_root)

    jsonl_path = spider2_root / "spider2-lite.jsonl"

    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Spider2-Lite file not found: {jsonl_path}"
        )

    rows = load_jsonl(jsonl_path)

    database_sets = get_spider2_database_sets(
        spider2_root
    )

    examples = []

    for row in rows:
        instance_id = row["instance_id"]
        db_id = row["db"]

        dialect = get_spider2_dialect(
            db_id,
            database_sets,
        )

        local_db_path = None

        if dialect == "sqlite":
            found_path = get_spider2_local_db_path(
                db_id,
                spider2_root,
            )

            if found_path is not None:
                local_db_path = str(found_path)

        example = Text2SQLExample(
            example_id=instance_id,
            dataset="spider2_lite",
            split="evaluation",
            db_id=db_id,
            question=row["question"],
            gold_sql=None,
            dialect=dialect,
            external_knowledge=row.get(
                "external_knowledge"
            ),
            metadata={
                "local_db_path": local_db_path,
                "locally_executable": (
                    dialect == "sqlite"
                    and local_db_path is not None
                ),
            },
        )

        examples.append(example)

    return examples


def load_spider2_sqlite(
    spider2_root: Path,
) -> List[Text2SQLExample]:
    """
    Load only Spider2-Lite examples belonging to SQLite DBs.
    """

    examples = load_spider2_lite(spider2_root)

    return [
        example
        for example in examples
        if example.dialect == "sqlite"
    ]


def load_spider2_local(
    spider2_root: Path,
) -> List[Text2SQLExample]:
    """
    Load only Spider2-Lite examples that have an actual local
    SQLite database available.

    This is the safest subset for local execution.
    """

    examples = load_spider2_lite(spider2_root)

    return [
        example
        for example in examples
        if example.metadata.get(
            "locally_executable",
            False,
        )
    ]


# ---------------------------------------------------------------------
# BIRD
#
# Kept so the starter remains extensible even though BIRD is not
# currently part of our main local pipeline.
# ---------------------------------------------------------------------

def _find_bird_json(
    bird_root: Path,
    split: str,
) -> Optional[Path]:
    """
    Look for common BIRD train/dev JSON layouts.
    """

    bird_root = Path(bird_root)

    candidates = [
        bird_root / f"{split}.json",
        bird_root / split / f"{split}.json",
        bird_root / split / f"{split}_data.json",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def load_bird_split(
    bird_root: Path,
    split: str,
) -> List[Text2SQLExample]:
    """
    Load a BIRD split when present.

    BIRD is optional in the current project, so this function
    is retained for future use but is not required by
    prepare_all.py.
    """

    path = _find_bird_json(
        bird_root,
        split,
    )

    if path is None:
        raise FileNotFoundError(
            f"Could not find BIRD {split} JSON under "
            f"{bird_root}"
        )

    rows = load_json(path)

    examples = []

    for index, row in enumerate(rows):
        db_id = (
            row.get("db_id")
            or row.get("db")
            or ""
        )

        question = (
            row.get("question")
            or row.get("instruction")
            or ""
        )

        gold_sql = (
            row.get("SQL")
            or row.get("sql")
            or row.get("query")
        )

        evidence = row.get("evidence")

        example = Text2SQLExample(
            example_id=(
                f"bird_{split}_{index:06d}"
            ),
            dataset="bird",
            split=split,
            db_id=db_id,
            question=question,
            gold_sql=gold_sql,
            dialect="sqlite",
            external_knowledge=evidence,
            metadata={
                "evidence": evidence,
            },
        )

        examples.append(example)

    return examples


def load_bird(
    bird_root: Path,
):
    """
    Load available BIRD train/dev splits.

    Returns:
        (train_examples, dev_examples)
    """

    bird_root = Path(bird_root)

    train_examples = []
    dev_examples = []

    try:
        train_examples = load_bird_split(
            bird_root,
            "train",
        )
    except FileNotFoundError:
        pass

    try:
        dev_examples = load_bird_split(
            bird_root,
            "dev",
        )
    except FileNotFoundError:
        pass

    return train_examples, dev_examples

def load_normalized(
    path: str | Path,
) -> List[Text2SQLExample]:
    """
    Load one of the normalized JSON datasets generated by
    prepare_all.py back into Text2SQLExample objects.
    """

    rows = load_json(Path(path))

    examples = []

    for row in rows:
        example = Text2SQLExample(
            example_id=row["example_id"],
            dataset=row["dataset"],
            split=row["split"],
            db_id=row["db_id"],
            question=row["question"],
            gold_sql=row.get("gold_sql"),
            dialect=row.get(
                "dialect",
                "sqlite",
            ),
            external_knowledge=row.get(
                "external_knowledge"
            ),
            metadata=row.get(
                "metadata",
                {},
            ),
        )

        examples.append(example)

    return examples