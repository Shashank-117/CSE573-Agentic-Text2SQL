from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Spider 1.0
SPIDER1_DIR = DATA_DIR / "Spider1"
SPIDER1_DATABASE_DIR = SPIDER1_DIR / "database"

# Spider 2.0-Lite
SPIDER2_DIR = DATA_DIR / "Spider2" / "spider2-lite"
SPIDER2_JSONL = SPIDER2_DIR / "spider2-lite.jsonl"
SPIDER2_RESOURCE_DIR = SPIDER2_DIR / "resource"

SPIDER2_DATABASE_RESOURCE_DIR = (
    SPIDER2_RESOURCE_DIR / "databases"
)

SPIDER2_SQLITE_METADATA_DIR = (
    SPIDER2_DATABASE_RESOURCE_DIR / "sqlite"
)

SPIDER2_LOCAL_DB_DIR = (
    SPIDER2_DATABASE_RESOURCE_DIR / "spider2-localdb"
)

SPIDER2_DOCUMENTS_DIR = (
    SPIDER2_RESOURCE_DIR / "documents"
)

# Outputs
NORMALIZED_DIR = OUTPUT_DIR / "normalized"

SCHEMA_DIR = OUTPUT_DIR / "schemas"
SPIDER1_SCHEMA_DIR = SCHEMA_DIR / "spider1"
SPIDER2_SCHEMA_DIR = SCHEMA_DIR / "spider2"

RETRIEVAL_DIR = OUTPUT_DIR / "retrieval"
TRAINING_DIR = OUTPUT_DIR / "training"
BASELINE_DIR = OUTPUT_DIR / "baseline_results"