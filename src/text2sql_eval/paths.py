"""Filesystem locations used by the evaluation harness."""

from text2sql_data.paths import (  # reuse the team's dataset paths
    DATA_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    SPIDER1_DATABASE_DIR,
    SPIDER1_DIR,
    SPIDER2_DIR,
    SPIDER2_DOCUMENTS_DIR,
    SPIDER2_LOCAL_DB_DIR,
)

SPIDER2_EVAL_DIR = SPIDER2_DIR / "evaluation_suite"
TEST_SUITE_DIR = PROJECT_ROOT / "third_party" / "test-suite-sql-eval"

# Committed to git (small, and needed to reproduce any run).
MANIFEST_DIR = PROJECT_ROOT / "manifests"
RUNS_DIR = OUTPUT_DIR / "runs"

__all__ = [
    "DATA_DIR", "OUTPUT_DIR", "PROJECT_ROOT", "SPIDER1_DATABASE_DIR", "SPIDER1_DIR",
    "SPIDER2_DIR", "SPIDER2_DOCUMENTS_DIR", "SPIDER2_LOCAL_DB_DIR", "SPIDER2_EVAL_DIR",
    "TEST_SUITE_DIR", "MANIFEST_DIR", "RUNS_DIR",
]
