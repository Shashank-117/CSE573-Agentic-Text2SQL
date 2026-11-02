import sqlite3
from pathlib import Path
from typing import Dict, List

from .io_utils import write_json


def extract_sqlite_schema(db_path: str | Path, db_id: str | None = None) -> Dict:
    db_path = Path(db_path)
    connection = sqlite3.connect(str(db_path))
    cursor = connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
    table_names = [row[0] for row in cursor.fetchall()]

    tables: List[Dict] = []
    for table_name in table_names:
        safe = table_name.replace('"', '""')
        cursor.execute(f'PRAGMA table_info("{safe}")')
        columns = [
            {
                "name": row[1],
                "type": row[2] or "UNKNOWN",
                "nullable": not bool(row[3]),
                "default": row[4],
                "primary_key": bool(row[5]),
            }
            for row in cursor.fetchall()
        ]

        cursor.execute(f'PRAGMA foreign_key_list("{safe}")')
        foreign_keys = [
            {
                "column": row[3],
                "references_table": row[2],
                "references_column": row[4],
            }
            for row in cursor.fetchall()
        ]
        tables.append({"name": table_name, "columns": columns, "foreign_keys": foreign_keys})

    connection.close()
    return {"db_id": db_id or db_path.stem, "dialect": "sqlite", "path": str(db_path), "tables": tables}


def find_sqlite_databases(database_root: str | Path) -> List[Path]:
    root = Path(database_root)
    return sorted(list(root.rglob("*.sqlite")) + list(root.rglob("*.db")))


def export_all_schemas(database_root: str | Path, output_dir: str | Path) -> int:
    output_dir = Path(output_dir)
    count = 0
    for db_path in find_sqlite_databases(database_root):
        db_id = db_path.parent.name if db_path.parent != Path(database_root) else db_path.stem
        schema = extract_sqlite_schema(db_path, db_id=db_id)
        write_json(schema, output_dir / f"{db_id}.json")
        count += 1
    return count
