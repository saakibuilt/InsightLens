"""Snowflake connection management."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import snowflake.connector
from snowflake.connector import SnowflakeConnection
from snowflake.connector.errors import DatabaseError, ProgrammingError

from insightlens.config import SnowflakeConfig


class SnowflakeConnectionError(Exception):
    """Raised when Snowflake cannot be reached or authenticated."""


@contextmanager
def open_connection(cfg: SnowflakeConfig) -> Iterator[SnowflakeConnection]:
    """Open a Snowflake connection with the given config. Yields a connection."""
    try:
        conn = snowflake.connector.connect(
            account=cfg.account,
            user=cfg.user,
            password=cfg.password,
            warehouse=cfg.warehouse,
            database=cfg.database,
            schema=cfg.schema,
            role=cfg.role,
        )
    except DatabaseError as exc:
        raise SnowflakeConnectionError(
            f"Failed to authenticate to Snowflake account '{cfg.account}' as user '{cfg.user}'. "
            f"Underlying error: {exc}"
        ) from exc

    try:
        yield conn
    finally:
        conn.close()


def execute_script(conn: SnowflakeConnection, sql_text: str) -> None:
    """Execute a multi-statement SQL script."""
    cursor = conn.cursor()
    try:
        for statement in _split_statements(sql_text):
            if statement.strip():
                cursor.execute(statement)
    except ProgrammingError as exc:
        raise SnowflakeConnectionError(
            f"Snowflake rejected SQL statement. Underlying error: {exc.msg} "
            f"(error code: {exc.errno})"
        ) from exc
    finally:
        cursor.close()


def _split_statements(sql_text: str) -> list[str]:
    result = []
    for stmt in sql_text.split(";"):
        stripped = stmt.strip()
        if not stripped:
            continue
        non_comment = "\n".join(
            line for line in stripped.splitlines() if not line.strip().startswith("--")
        ).strip()
        if non_comment:
            result.append(stripped)
    return result
