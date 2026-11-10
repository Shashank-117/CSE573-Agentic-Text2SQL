"""Manifest correctness and the no-gold-leakage guarantee (needs data/)."""

import dataclasses
import json

import pytest

from text2sql_eval.contracts import QuestionInput
from text2sql_eval.manifest import (
    build_spider1_manifest,
    build_spider2_lite_manifest,
    read_manifest,
    select,
    to_question_input,
    with_exclusion,
    write_manifest,
)
from text2sql_eval.paths import SPIDER1_DIR, SPIDER2_EVAL_DIR

GOLD_FIELD_HINTS = ("gold", "query", "answer", "label", "result")


@pytest.fixture(scope="module")
def spider1(request):
    if not (SPIDER1_DIR / "dev.json").exists():
        pytest.skip("Spider 1 data not present")
    return build_spider1_manifest("dev")


@pytest.fixture(scope="module")
def spider2():
    if not SPIDER2_EVAL_DIR.exists():
        pytest.skip("Spider 2 data not present")
    return build_spider2_lite_manifest()


def test_spider1_manifest(spider1):
    assert len(spider1) == 1034 and all(e.included for e in spider1)
    assert spider1[0].question_id == "spider1_dev_000000"
    assert len({e.question_id for e in spider1}) == 1034


def test_spider2_manifest_counts_and_reasons(spider2):
    assert len(spider2) == 547
    included = [e for e in spider2 if e.included]
    assert len(included) == 135
    assert all(e.question_id.startswith("local") and e.dialect == "sqlite" for e in included)
    assert all(e.exclusion_reason for e in spider2 if not e.included)
    # The two DBs whose metadata folder is spelled differently must be kept.
    assert {"sqlite-sakila", "Db-IMDB"} <= {e.db_id for e in included}


def test_manifest_round_trip_and_select(tmp_path, spider2):
    path = write_manifest(spider2, tmp_path / "m.jsonl")
    assert read_manifest(path) == spider2
    assert len(select(spider2, limit=5)) == 5
    assert select(spider2, question_ids=["local002", "bq011"])[0].question_id == "local002"  # bq011 excluded
    excluded = with_exclusion(spider2[0], "manual")
    assert not excluded.included and spider2[0] is not excluded


def test_seeded_sample_is_reproducible_and_ordered(spider1):
    a = select(spider1, sample=50, sample_seed=7)
    assert a == select(spider1, sample=50, sample_seed=7)
    assert a != select(spider1, sample=50, sample_seed=8)
    assert len(a) == 50 and [e.question_id for e in a] == sorted(e.question_id for e in a)
    assert len({e.db_id for e in a}) > 10  # spread across databases, unlike --limit


def test_question_input_has_no_gold_fields():
    names = [f.name.lower() for f in dataclasses.fields(QuestionInput)]
    assert not [n for n in names if any(h in n for h in GOLD_FIELD_HINTS)]


def test_no_gold_sql_reaches_systems(spider1):
    gold = [row["query"].strip() for row in json.loads((SPIDER1_DIR / "dev.json").read_text())]
    for entry, gold_sql in zip(spider1, gold):
        payload = json.dumps(dataclasses.asdict(to_question_input(entry)))
        assert gold_sql not in payload, entry.question_id
        assert gold_sql not in json.dumps(entry.to_dict()), entry.question_id


def test_spider2_inputs_carry_only_benchmark_evidence(spider2):
    gold_sql_dir = SPIDER2_EVAL_DIR / "gold" / "sql"
    for entry in select(spider2):
        question = to_question_input(entry)
        assert [d.doc_id for d in question.evidence] == list(entry.evidence_doc_ids)
        gold_file = gold_sql_dir / f"{entry.question_id}.sql"
        if gold_file.exists():
            assert gold_file.read_text().strip() not in json.dumps(dataclasses.asdict(question))


def test_excluded_entry_cannot_become_input(spider2):
    excluded = next(e for e in spider2 if not e.included)
    with pytest.raises(ValueError):
        to_question_input(excluded)
