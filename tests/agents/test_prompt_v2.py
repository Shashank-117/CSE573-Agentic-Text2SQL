"""Prompt v2: shared SQLite guidance, schema-grounded repair feedback, repeated-SQL detection."""

import sqlite3

import pytest

from text2sql_agents import SingleAgentConfig, SingleAgentSystem
from text2sql_agents.dialect_hints import SQLITE_FUNCTION_REPLACEMENTS, dialect_hints
from text2sql_agents.prompts import REPEATED_SQL_NOTE, SINGLE_AGENT_SYSTEM
from text2sql_agents.repair_feedback import SchemaIndex, explain_error, table_aliases
from text2sql_eval.contracts import Budget, QuestionInput
from text2sql_llm import ScriptedLLM
from text2sql_retrieval import table_columns

GOOD = ("```sql\nSELECT d.dept_name, COUNT(*) FROM staff s JOIN dept d USING (dept_id) "
        "WHERE s.status = 'A' GROUP BY d.dept_name\n```")
WRONG_TABLE = "```sql\nSELECT s.dept_name, COUNT(*) FROM staff s GROUP BY s.dept_name\n```"


def _question(db):
    return QuestionInput("q1", "test", "company", str(db), "How many active employees are in each department?")


def _system(llm, **config):
    return SingleAgentSystem(llm, SingleAgentConfig(measure_gpu=False, **config))


@pytest.fixture()
def schema(company_db):
    return SchemaIndex.from_tables({t: list(c) for t, c in table_columns(str(company_db)).items()})


# ------------------------------------------------- guidance matches the engine

@pytest.mark.parametrize("expr", [
    "strftime('%Y', '2020-03-04')", "date('2020-03-04', 'start of month')", "date('2020-03-04', '+7 days')",
    "julianday('2020-03-04') - julianday('2020-03-01')", "MIN(1, 2)", "MAX(1, 2)", "SQRT(4)", "POWER(2, 3)",
    "LN(2)", "EXP(1)", "SIN(1)", "COS(1)", "ACOS(0.5)", "RADIANS(180)", "PI()", "FLOOR(2.5)", "CEIL(2.1)",
    "printf('%.2f', 1.234)", "'a' || 'b'", "strftime('%s', '2020-01-01')",
])
def test_functions_the_guidance_recommends_exist(expr):
    sqlite3.connect(":memory:").execute(f"SELECT {expr}").fetchone()


@pytest.mark.parametrize("sql", [
    "SELECT EXTRACT(YEAR FROM '2020-01-01')", "SELECT DATE_TRUNC('month', '2020-01-01')",
    "SELECT DATEDIFF('2020-01-02', '2020-01-01')", "SELECT LEAST(1, 2)", "SELECT GREATEST(1, 2)",
    "SELECT STDDEV(1)", "SELECT MEDIAN(1)", "SELECT STRING_AGG('a', ',')", "SELECT FORMAT('%d', 1)",
    "SELECT * FROM (SELECT 1 a) FULL OUTER JOIN (SELECT 1 b) ON 1",
])
def test_constructs_the_guidance_forbids_really_fail(sql):
    with pytest.raises(sqlite3.Error):
        sqlite3.connect(":memory:").execute(sql).fetchall()


def test_dialect_hints_only_for_sqlite():
    assert "strftime" in dialect_hints("sqlite") and "FULL OUTER JOIN" in dialect_hints("SQLite")
    assert dialect_hints("bigquery") == "" and dialect_hints(None) == ""


# ------------------------------------------------------- repair feedback

def test_table_aliases():
    sql = 'SELECT * FROM staff s JOIN "dept" AS d ON s.dept_id = d.dept_id LEFT JOIN audit_log WHERE 1'
    assert table_aliases(sql) == {"staff": "staff", "s": "staff", "dept": "dept", "d": "dept",
                                  "audit_log": "audit_log"}


def test_column_from_wrong_table_points_to_owner(schema):
    sql = "SELECT s.dept_name FROM staff s"
    text = explain_error("Database error: no such column: s.dept_name", sql, schema)
    assert 'Table "staff" (used as s) has no column "dept_name"' in text
    assert "staff_id, dept_id, status" in text
    assert 'Column "dept_name" exists in: dept' in text


def test_invented_column_suggests_similar(schema):
    text = explain_error("Database error: no such column: dept_nam", "SELECT dept_nam FROM dept", schema)
    assert 'No table has a column named "dept_nam"' in text and "dept.dept_name" in text


def test_unknown_alias(schema):
    text = explain_error("Database error: no such column: x.status", "SELECT x.status FROM staff s", schema)
    assert '"x" is not a table or alias' in text and "exists in: staff" in text


@pytest.mark.parametrize("error,expected", [
    ("Database error: no such function: DATE_TRUNC", SQLITE_FUNCTION_REPLACEMENTS["DATE_TRUNC"]),
    ("Database error: no such function: my_udf", "rewrite it with built-in SQLite functions"),
    ("Database error: ambiguous column name: dept_id", "exists in: dept, staff"),
    ("Database error: no such table: depts", "Did you mean dept"),
    ("Database error: RIGHT and FULL OUTER JOINs are not currently supported", "LEFT JOIN"),
    ("Database error: ORDER BY clause should come after UNION ALL not before", "single ORDER BY"),
    ("Database error: misuse of aggregate function COUNT()", "HAVING"),
    ("Database error: near \"(\": syntax error", ""),
])
def test_explain_error_messages(schema, error, expected):
    text = explain_error(error, "SELECT 1", schema)
    assert expected in text if expected else text == ""


def test_table_columns_lists_every_table(company_db):
    columns = table_columns(str(company_db))
    assert columns["dept"] == ("dept_id", "dept_name") and "audit_log" in columns


# ----------------------------------------------------- the agent with v2

def test_v2_system_prompt_has_guidance_and_column_rule(company_db):
    llm = ScriptedLLM([GOOD])
    result = _system(llm).predict(_question(company_db), Budget())
    system = llm.calls[0][0].content
    assert "SQLite notes" in system and "column belongs only to the table" in system
    assert result.intermediate_artifacts["prompt_version"] == "single_agent/v2"


def test_v1_reproduces_original_prompt_and_feedback(company_db):
    llm = ScriptedLLM([WRONG_TABLE, GOOD])
    result = _system(llm, prompt_version="v1").predict(_question(company_db), Budget())
    assert llm.calls[0][0].content == SINGLE_AGENT_SYSTEM.format(dialect="SQLITE")
    assert "How to fix" not in llm.calls[1][-1].content
    assert result.intermediate_artifacts["features"] == {
        "dialect_hints": False, "grounded_feedback": False, "repeat_feedback": False}


def test_grounded_feedback_reaches_the_model(company_db):
    llm = ScriptedLLM([WRONG_TABLE, GOOD])
    result = _system(llm).predict(_question(company_db), Budget())
    repair = llm.calls[1][-1].content
    assert "How to fix:" in repair and 'Column "dept_name" exists in: dept' in repair
    assert result.intermediate_artifacts["attempts"][0]["guidance"].startswith('Table "staff"')
    assert result.model_calls == 2


def test_repeated_sql_is_flagged_and_history_listed(company_db):
    bad = "```sql\nSELECT bad FROM staff\n```"
    other = "```sql\nSELECT worse FROM staff\n```"
    llm = ScriptedLLM([bad, bad, other, GOOD])
    result = _system(llm).predict(_question(company_db), Budget(max_retries=3))
    attempts = result.intermediate_artifacts["attempts"]
    assert [a["repeated"] for a in attempts] == [False, True, False, False]
    assert attempts[1]["problem"].startswith(REPEATED_SQL_NOTE)
    second_repair = llm.calls[2][-1].content
    assert REPEATED_SQL_NOTE in second_repair
    third_repair = llm.calls[3][-1].content
    assert "Earlier queries that also failed" in third_repair and "- SELECT bad FROM staff" in third_repair
    assert third_repair.count("- SELECT bad FROM staff") == 1        # history has no duplicates
    assert result.intermediate_artifacts["stop_reason"] == "ok"


@pytest.mark.parametrize("flag,absent", [
    ("dialect_hints", "SQLite notes"),
    ("grounded_feedback", "How to fix:"),
    ("repeat_feedback", REPEATED_SQL_NOTE),
])
def test_each_feature_can_be_ablated(company_db, flag, absent):
    llm = ScriptedLLM([WRONG_TABLE, WRONG_TABLE, GOOD])
    result = _system(llm, **{flag: False}).predict(_question(company_db), Budget())
    prompts = " ".join(m.content for call in llm.calls for m in call)
    assert absent not in prompts
    assert result.intermediate_artifacts["features"][flag] is False


def test_unknown_prompt_version_rejected():
    with pytest.raises(ValueError, match="prompt_version"):
        SingleAgentConfig(prompt_version="v9")
