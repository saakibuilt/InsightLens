"""Creates Snowflake database, schema, and tables defined in schema.sql."""
from __future__ import annotations

import sys
from pathlib import Path

from insightlens.config import ConfigError, load_config
from insightlens.storage.snowflake_client import (
    SnowflakeConnectionError,
    execute_script,
    open_connection,
)

"""@ 2026 Developed by Saksham Nirula"""


def main() -> int:
    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"[setup_database] Configuration error: {exc}", file=sys.stderr)
        return 1

    schema_path = Path(__file__).resolve().parents[1] / "src" / "insightlens" / "storage" / "schema.sql"
    if not schema_path.exists():
        print(f"[setup_database] Schema file missing: {schema_path}", file=sys.stderr)
        return 1

    sql_text = schema_path.read_text()

    try:
        with open_connection(cfg.snowflake) as conn:
            print("[setup_database] Dropping existing tables to reset the database...")
            conn.cursor().execute("DROP TABLE IF EXISTS CHUNKS;")
            conn.cursor().execute("DROP TABLE IF EXISTS DOCUMENTS;")
            
            execute_script(conn, sql_text)
    except SnowflakeConnectionError as exc:
        print(f"[setup_database] {exc}", file=sys.stderr)
        return 1

    print("[setup_database] Schema applied successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
