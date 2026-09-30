"""Audit A-1 to A-8 against the operator store. Empty means clean.

    python scripts/audit.py
    python scripts/audit.py --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mag.audit import run_audit
from mag.db import connect, init_db
from mag.paths import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="MAG audit A-1 to A-8")
    parser.add_argument("--sql", type=Path, default=ROOT / "scripts" / "audit.sql")
    parser.add_argument("--verbose", action="store_true", help="print every violating row")
    args = parser.parse_args()
    conn = connect()
    init_db(conn)
    violations = run_audit(conn, args.sql)
    names = [name for name, _rows in violations] or []
    for name, rows in violations:
        print(f"{name}: {len(rows)} violating row(s)")
        if args.verbose:
            for row in rows[:20]:
                print(f"  {row}")
            if len(rows) > 20:
                print(f"  ... {len(rows) - 20} more")
    if violations:
        print(f"audit=FAIL checks_with_rows={','.join(names)}")
        raise SystemExit(1)
    total = conn.execute("SELECT COUNT(*) FROM impressions").fetchone()[0]
    print(f"audit=clean impressions={total}")


if __name__ == "__main__":
    main()
