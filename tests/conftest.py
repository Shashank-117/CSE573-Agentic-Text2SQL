import sqlite3
from pathlib import Path

import pytest


@pytest.fixture()
def company_db(tmp_path: Path) -> Path:
    """Tiny HR database used for the illustrative 'active employees' example."""
    path = tmp_path / "company.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE dept (dept_id INTEGER PRIMARY KEY, dept_name TEXT);
        CREATE TABLE staff (staff_id INTEGER PRIMARY KEY, dept_id INTEGER, status TEXT,
                            FOREIGN KEY (dept_id) REFERENCES dept(dept_id));
        CREATE TABLE audit_log (log_id INTEGER PRIMARY KEY, note TEXT);
        INSERT INTO dept VALUES (1, 'Sales'), (2, 'Eng');
        INSERT INTO staff VALUES (1, 1, 'A'), (2, 1, 'T'), (3, 2, 'A'), (4, 2, 'A');
        """
    )
    con.commit()
    con.close()
    return path
