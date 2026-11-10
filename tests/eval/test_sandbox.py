import sqlite3

from text2sql_eval.contracts import Status
from text2sql_eval.sandbox import ViewDefinition, database_fingerprint, run_query


def _count_staff(db):
    con = sqlite3.connect(db)
    try:
        return con.execute("SELECT COUNT(*) FROM staff").fetchone()[0]
    finally:
        con.close()


def test_executes_select(company_db):
    result = run_query(company_db, "SELECT dept_name FROM dept ORDER BY dept_id")
    assert result.status == Status.EXECUTED
    assert result.columns == ("dept_name",)
    assert result.rows == (("Sales",), ("Eng",))


def test_temp_views_work_and_do_not_persist(company_db):
    view = ViewDefinition("Employee", "SELECT s.staff_id AS employee_id, d.dept_name AS department_name "
                                      "FROM staff s JOIN dept d USING (dept_id) WHERE s.status = 'A'")
    result = run_query(company_db, "SELECT department_name, COUNT(*) FROM Employee GROUP BY 1", [view])
    assert result.ok and sorted(result.rows) == [("Eng", 2), ("Sales", 1)]
    assert result.views_created == ("Employee",)

    # A fresh connection must not see the view.
    again = run_query(company_db, "SELECT * FROM Employee")
    assert again.status == Status.EXECUTION_ERROR
    assert "no such table" in again.error


def test_mutations_rejected_and_data_unchanged(company_db):
    before = (_count_staff(company_db), database_fingerprint(company_db))
    for sql in ["DELETE FROM staff", "SELECT 1; DELETE FROM staff", "DROP TABLE staff"]:
        assert run_query(company_db, sql).status == Status.UNSAFE_SQL
    assert (_count_staff(company_db), database_fingerprint(company_db)) == before


def test_authorizer_blocks_what_static_check_misses(company_db):
    result = run_query(company_db, "SELECT load_extension('evil')")
    assert result.status == Status.EXECUTION_ERROR
    assert "not authorized" in result.error


def test_view_body_must_be_read_only(company_db):
    bad = ViewDefinition("v", "DELETE FROM staff")
    result = run_query(company_db, "SELECT * FROM v", [bad])
    assert result.status == Status.VIEW_ERROR
    assert _count_staff(company_db) == 4


def test_invalid_view_name_rejected(company_db):
    result = run_query(company_db, "SELECT 1", [ViewDefinition('x"; DROP TABLE staff; --', "SELECT 1")])
    assert result.status == Status.VIEW_ERROR


def test_timeout(company_db):
    endless = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"
    result = run_query(company_db, endless, timeout_s=0.3)
    assert result.status == Status.TIMEOUT


def test_execution_error(company_db):
    result = run_query(company_db, "SELECT no_such_column FROM staff")
    assert result.status == Status.EXECUTION_ERROR
    assert "no such column" in result.error


def test_row_limit_truncates(company_db):
    result = run_query(company_db, "SELECT * FROM staff", max_rows=2)
    assert result.ok and len(result.rows) == 2 and result.truncated
