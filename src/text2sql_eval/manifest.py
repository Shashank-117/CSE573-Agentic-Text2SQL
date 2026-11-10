"""Question manifests: which questions are evaluated, and why others are not.

A manifest is a JSONL file with one ``ManifestEntry`` per benchmark
question. Excluded questions stay in the file with a reason, so reported
subsets are always explicit. Manifests never contain gold SQL or results.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

from .contracts import EvidenceDoc, QuestionInput
from .paths import (
    PROJECT_ROOT,
    SPIDER1_DATABASE_DIR,
    SPIDER1_DIR,
    SPIDER2_DIR,
    SPIDER2_DOCUMENTS_DIR,
    SPIDER2_EVAL_DIR,
    SPIDER2_LOCAL_DB_DIR,
)

_SPIDER2_DIALECT_BY_PREFIX = (("local", "sqlite"), ("bq", "bigquery"), ("ga", "bigquery"), ("sf", "snowflake"))


@dataclass(frozen=True)
class ManifestEntry:
    question_id: str
    benchmark: str
    split: str
    db_id: str
    db_path: str | None  # relative to project root
    question: str
    dialect: str
    evidence_doc_ids: tuple[str, ...]
    included: bool
    exclusion_reason: str | None = None

    def to_dict(self) -> dict:
        return {**asdict(self), "evidence_doc_ids": list(self.evidence_doc_ids)}

    @classmethod
    def from_dict(cls, row: dict) -> "ManifestEntry":
        return cls(**{**row, "evidence_doc_ids": tuple(row.get("evidence_doc_ids") or ())})


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


# ---------------------------------------------------------------- Spider 1

def build_spider1_manifest(split: str = "dev", spider_root: Path = SPIDER1_DIR) -> list[ManifestEntry]:
    """Spider 1 dev (or train). IDs match text2sql_data's example_id scheme."""
    files = {"dev": ["dev.json"], "train": ["train_spider.json", "train_others.json"]}[split]
    rows: list[dict] = []
    for name in files:
        rows.extend(json.loads((Path(spider_root) / name).read_text(encoding="utf-8")))

    entries = []
    for index, row in enumerate(rows):
        db_file = SPIDER1_DATABASE_DIR / row["db_id"] / f"{row['db_id']}.sqlite"
        exists = db_file.is_file()
        entries.append(
            ManifestEntry(
                question_id=f"spider1_{split}_{index:06d}",
                benchmark="spider1",
                split=split,
                db_id=row["db_id"],
                db_path=_relative(db_file) if exists else None,
                question=row["question"],
                dialect="sqlite",
                evidence_doc_ids=(),
                included=exists,
                exclusion_reason=None if exists else "SQLite database file missing",
            )
        )
    return entries


# ---------------------------------------------------------- Spider 2-Lite

def _spider2_dialect(instance_id: str) -> str:
    for prefix, dialect in _SPIDER2_DIALECT_BY_PREFIX:
        if instance_id.startswith(prefix):
            return dialect
    return "unknown"


def _spider2_gold_ids(eval_dir: Path) -> set[str]:
    """Instance IDs with an evaluation standard and at least one gold CSV."""
    standards_path = eval_dir / "gold" / "spider2lite_eval.jsonl"
    standards = {json.loads(line)["instance_id"] for line in standards_path.read_text().splitlines() if line.strip()}
    csv_ids = {re.sub(r"(_[a-z])?\.csv$", "", p.name) for p in (eval_dir / "gold" / "exec_result").glob("*.csv")}
    return standards & csv_ids


def _spider2_local_db(db_id: str, local_db_dir: Path) -> Path | None:
    direct = local_db_dir / f"{db_id}.sqlite"
    if direct.is_file():
        return direct
    matches = [p for p in local_db_dir.glob("*.sqlite") if p.stem.lower() == db_id.lower()]
    return matches[0] if matches else None


def _spider2_exclusion(dialect: str, db_file: Path | None, has_gold: bool) -> str | None:
    if dialect in ("bigquery", "snowflake"):
        return f"dialect={dialect}: needs cloud warehouse access (out of current local-SQLite scope)"
    if dialect != "sqlite":
        return "unknown dialect for instance id"
    if db_file is None:
        return "local SQLite database file missing"
    if not has_gold:
        return "no official gold execution result"
    return None


def build_spider2_lite_manifest(spider2_root: Path = SPIDER2_DIR) -> list[ManifestEntry]:
    """All 547 Spider 2.0-Lite instances; only local SQLite ones included."""
    jsonl = Path(spider2_root) / "spider2-lite.jsonl"
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    gold_ids = _spider2_gold_ids(SPIDER2_EVAL_DIR)

    entries = []
    for row in rows:
        qid, db_id = row["instance_id"], row["db"]
        dialect = _spider2_dialect(qid)
        db_file = _spider2_local_db(db_id, SPIDER2_LOCAL_DB_DIR) if dialect == "sqlite" else None
        reason = _spider2_exclusion(dialect, db_file, qid in gold_ids)
        knowledge = row.get("external_knowledge")
        entries.append(
            ManifestEntry(
                question_id=qid,
                benchmark="spider2_lite",
                split="evaluation",
                db_id=db_id,
                db_path=_relative(db_file) if db_file else None,
                question=row["question"],
                dialect=dialect,
                evidence_doc_ids=(knowledge,) if knowledge else (),
                included=reason is None,
                exclusion_reason=reason,
            )
        )
    return entries


# ------------------------------------------------------------------- I/O

def write_manifest(entries: Iterable[ManifestEntry], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
    return path


def read_manifest(path: Path) -> list[ManifestEntry]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ManifestEntry.from_dict(json.loads(line)) for line in lines if line.strip()]


def select(
    entries: Iterable[ManifestEntry],
    question_ids: Iterable[str] | None = None,
    limit: int | None = None,
    sample: int | None = None,
    sample_seed: int = 0,
) -> list[ManifestEntry]:
    """Included entries, optionally restricted to IDs, a seeded random sample
    (kept in manifest order), and/or the first N."""
    chosen = [e for e in entries if e.included]
    if question_ids is not None:
        wanted = set(question_ids)
        chosen = [e for e in chosen if e.question_id in wanted]
    if sample is not None and sample < len(chosen):
        picked = set(random.Random(sample_seed).sample(range(len(chosen)), sample))
        chosen = [e for i, e in enumerate(chosen) if i in picked]
    return chosen[:limit] if limit is not None else chosen


def to_question_input(entry: ManifestEntry, documents_dir: Path = SPIDER2_DOCUMENTS_DIR) -> QuestionInput:
    """Build the gold-free input a system receives for one question."""
    if not entry.included or entry.db_path is None:
        raise ValueError(f"{entry.question_id} is excluded: {entry.exclusion_reason}")
    evidence = tuple(
        EvidenceDoc(doc_id=doc_id, text=(Path(documents_dir) / doc_id).read_text(encoding="utf-8"))
        for doc_id in entry.evidence_doc_ids
        if (Path(documents_dir) / doc_id).is_file()
    )
    return QuestionInput(
        question_id=entry.question_id,
        benchmark=entry.benchmark,
        db_id=entry.db_id,
        db_path=str(PROJECT_ROOT / entry.db_path),
        question=entry.question,
        dialect=entry.dialect,
        evidence=evidence,
    )


def with_exclusion(entry: ManifestEntry, reason: str) -> ManifestEntry:
    """Return a copy of ``entry`` marked excluded (no mutation)."""
    return replace(entry, included=False, exclusion_reason=reason)
