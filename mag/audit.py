"""Run scripts/audit.sql. Each A-* query returns violating rows. Empty means clean."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path


def _queries(sql_text: str) -> list[tuple[str, str]]:
    parts = re.split(r"\n(?=-- A-)", sql_text.strip())
    found: list[tuple[str, str]] = []
    for part in parts:
        match = re.match(r"-- (A-\S+)", part)
        if not match:
            continue
        lines = [
            line for line in part.splitlines()
            if line.strip() and not line.strip().startswith("--")
        ]
        statement = "\n".join(lines).strip().rstrip(";")
        if statement:
            found.append((match.group(1), statement))
    return found


def run_audit(conn: sqlite3.Connection, sql_path: Path) -> list[tuple[str, list[tuple]]]:
    violations: list[tuple[str, list[tuple]]] = []
    for name, statement in _queries(sql_path.read_text()):
        rows = conn.execute(statement).fetchall()
        if rows:
            violations.append((name, [tuple(row) for row in rows]))
    return violations
