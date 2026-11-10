import pytest

from text2sql_eval.sandbox import ViewDefinition, run_query
from text2sql_eval.view_composition import compose_with_views, views_from_artifacts

EMPLOYEE = ViewDefinition(
    "Employee",
    "SELECT s.staff_id AS employee_id, d.dept_name AS department_name "
    "FROM staff s JOIN dept d USING (dept_id) WHERE s.status = 'A';",
)
PRIMARY = "SELECT department_name, COUNT(*) FROM Employee GROUP BY department_name"


def test_no_views_returns_normalized_sql():
    assert compose_with_views("SELECT 1;", []) == "SELECT 1"


def test_cte_form_matches_view_form(company_db):
    via_views = run_query(company_db, PRIMARY, [EMPLOYEE])
    via_cte = run_query(company_db, compose_with_views(PRIMARY, [EMPLOYEE]))
    assert via_views.ok and via_cte.ok
    assert sorted(via_views.rows) == sorted(via_cte.rows) == [("Eng", 2), ("Sales", 1)]


def test_merges_with_existing_with_clause(company_db):
    primary = "WITH big AS (SELECT * FROM Employee) SELECT COUNT(*) FROM big"
    composed = compose_with_views(primary, [EMPLOYEE])
    assert composed.upper().count("WITH ") == 1
    assert run_query(company_db, composed).rows == ((3,),)


def test_preserves_recursive_keyword():
    composed = compose_with_views("WITH RECURSIVE r(x) AS (SELECT 1) SELECT * FROM r", [EMPLOYEE])
    assert composed.startswith('WITH RECURSIVE "Employee" AS (')


def test_views_from_artifacts():
    views = views_from_artifacts({"views": [{"name": "Employee", "sql": "SELECT 1"}]})
    assert views == [ViewDefinition("Employee", "SELECT 1")]
    assert views_from_artifacts({}) == []
    with pytest.raises(ValueError):
        views_from_artifacts({"views": [{"name": "x"}]})
