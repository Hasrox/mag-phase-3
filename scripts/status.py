"""Where every phase stands, in one command.

    python scripts/status.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mag.audit import run_audit
from mag.catalog import seed_catalog
from mag.config import load_config
from mag.db import connect, init_db
from mag.gate import sha256_file
from mag.phase1 import format_status as phase1_format
from mag.phase1 import operator_status as phase1_status
from mag.phase2 import format_status as phase2_format
from mag.phase2 import operator_status as phase2_status
from mag.phase3 import coverage, dry_run, format_coverage
from mag.phase4 import LOCK_PATH, S1_PATH
from mag.paths import MODELS, ROOT
from mag.serve import pool_images


def _count(conn, sql: str, *params) -> int:
    return int(conn.execute(sql, params).fetchone()[0])


def main() -> None:
    cfg = load_config()
    conn = connect()
    init_db(conn)
    seed_catalog(conn)

    print("=== phase 1: store, vocabulary, import")
    for line in phase1_format(phase1_status(conn)):
        print(line)

    print("\n=== phase 2: model acceptance and offline tagger")
    models = sorted(MODELS.glob("*.gguf"))
    model_sha = ""
    for path in models:
        print(f"model file {path.name} {path.stat().st_size / 1e9:.2f} GB")
        if "mmproj" not in path.name:
            model_sha = sha256_file(path)
    for line in phase2_format(phase2_status(conn, model_sha=model_sha)):
        print(line)

    print("\n=== phase 3: composer and renderer")
    for line in format_coverage(coverage(conn)).splitlines():
        print(line)

    print("\n=== phase 4: UI and random baseline")
    print(f"pool_images={len(pool_images(conn, None))}")
    print(f"impressions={conn.execute('SELECT COUNT(*) FROM impressions').fetchone()[0]}")
    print(f"rated={_count(conn, 'SELECT COUNT(*) FROM impressions WHERE outcome = ?', 'rated')}")
    print(f"tag_runs={conn.execute('SELECT COUNT(*) FROM tag_runs').fetchone()[0]}")
    print(f"accepted_runs={conn.execute('SELECT COUNT(*) FROM tag_runs WHERE accepted = 1').fetchone()[0]}")
    print(f"gold_images={conn.execute('SELECT COUNT(*) FROM assets WHERE is_gold = 1').fetchone()[0]}")
    print(f"s1_record={'present' if S1_PATH.exists() else 'missing'}")
    print(f"phase4_lock={'present' if LOCK_PATH.exists() else 'missing'}")

    print("\n=== audit A-1..A-8")
    violations = run_audit(conn, ROOT / "scripts" / "audit.sql")
    if violations:
        for name, rows in violations:
            print(f"{name}: {len(rows)} violating row(s)")
    else:
        print("audit=clean")

    print("\n=== serve path")
    print("sound source=assets.path (not sound_info.norm_path)")
    norm_rows = int(conn.execute(
        "SELECT COUNT(*) FROM sound_info WHERE norm_path LIKE '%norm%'"
    ).fetchone()[0])
    missing = 0
    for row in conn.execute("SELECT path FROM assets WHERE kind = 'sound'"):
        target = Path(row["path"])
        target = target if target.is_absolute() else ROOT / target
        if not target.is_file():
            missing += 1
    print(f"sounds_pointing_at_norm={norm_rows} sounds_missing_on_disk={missing}")


if __name__ == "__main__":
    main()
