"""Add sound_info.original_path and stop serving the loudnorm wav.

Additive and idempotent. Run it once against the operator store; mag.db.connect
also runs the same migration automatically on launch.

    python scripts/migrate_original_path.py
    python scripts/migrate_original_path.py --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mag.db import DB_PATH, connect, init_db


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate sound_info to original_path")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    parser.add_argument("--sample", type=int, default=3, help="rows to print in full")
    args = parser.parse_args()

    conn = _open(args.db, migrate=not args.dry_run)
    if not args.dry_run:
        init_db(conn)
    for line in _report(conn):
        print(f"before {line}")
    if args.dry_run:
        print("dry-run: nothing written")
        conn.close()
        return
    from mag.db import migrate_original_path

    result = migrate_original_path(conn)
    for key, value in result.items():
        print(f"{key}={value}")
    for line in _report(conn):
        print(f"after {line}")
    for row in conn.execute(
        "SELECT asset_id, original_path, norm_path FROM sound_info ORDER BY asset_id LIMIT ?",
        (args.sample,),
    ):
        print(f"sample id={row['asset_id']} original={row['original_path']} norm={row['norm_path']}")
    print(f"user_version={conn.execute('PRAGMA user_version').fetchone()[0]}")
    conn.close()


def _open(path: Path, *, migrate: bool) -> sqlite3.Connection:
    """connect() runs the migration itself, so a dry run bypasses it."""
    if migrate:
        return connect(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _report(conn: sqlite3.Connection) -> list[str]:
    total = int(conn.execute("SELECT COUNT(*) FROM sound_info").fetchone()[0])
    norm_rows = int(conn.execute(
        "SELECT COUNT(*) FROM sound_info WHERE norm_path LIKE '%sounds\\norm%' OR norm_path LIKE '%sounds/norm%'"
    ).fetchone()[0])
    columns = "original_path" in {
        str(row[1]) for row in conn.execute("PRAGMA table_info(sound_info)")
    }
    return [
        f"sound_info rows={total} has_original_path={int(columns)} "
        f"rows_still_pointing_at_norm={norm_rows}"
    ]


if __name__ == "__main__":
    main()
