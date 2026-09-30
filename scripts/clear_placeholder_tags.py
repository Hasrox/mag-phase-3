"""Snapshot, clear or restore the inbox placeholder tags.

    python scripts/clear_placeholder_tags.py --dry-run
    python scripts/clear_placeholder_tags.py
    python scripts/clear_placeholder_tags.py --restore data/placeholder_tags.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mag.db import connect, init_db
from mag.placeholders import (
    SNAPSHOT_PATH,
    clear_placeholders,
    load_snapshot,
    restore_placeholders,
    snapshot_placeholders,
)
from mag.serve import pool_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Clear inbox placeholder tags")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--restore", type=Path, default=None)
    parser.add_argument("--snapshot-out", type=Path, default=SNAPSHOT_PATH)
    args = parser.parse_args()

    conn = connect()
    init_db(conn)
    pool_before = len(pool_images(conn, None))

    if args.restore:
        snapshot = load_snapshot(args.restore)
        count = restore_placeholders(conn, snapshot)
        print(f"restored assets={count} from {args.restore}")
    else:
        snapshot = snapshot_placeholders(conn)
        args.snapshot_out.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot_out.write_text(
            snapshot.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"snapshot={args.snapshot_out} assets={snapshot.asset_count} "
            f"tags={snapshot.tag_count}"
        )
        count = clear_placeholders(conn, snapshot, dry_run=args.dry_run)
        print(f"cleared assets={count} dry_run={args.dry_run}")

    pool_after = len(pool_images(conn, None))
    print(f"pool before={pool_before} after={pool_after}")
    if pool_after == 0 and not args.restore:
        print(
            "pool is empty: nothing is tagged until a tag run succeeds. "
            "That is the honest state, not a failure."
        )


if __name__ == "__main__":
    main()
