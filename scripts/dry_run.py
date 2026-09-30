"""Phase 3 dry run. Prints 20 pinned and 20 unfiltered compositions. No HTTP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mag.config import load_config
from mag.db import connect, init_db
from mag.phase3 import coverage, dry_run, format_coverage


def main() -> None:
    parser = argparse.ArgumentParser(description="MAG Phase 3 dry run")
    parser.add_argument("--topic", default="animals")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument("--out", type=Path, default=Path("assets/render_cache/dry_run"))
    args = parser.parse_args()
    cfg = load_config()
    conn = connect()
    init_db(conn)
    print(format_coverage(coverage(conn)))
    pinned = dry_run(conn, cfg, topic=args.topic, n=args.n, seed=args.seed, out_dir=args.out / "pinned")
    open_topic = dry_run(conn, cfg, topic=None, n=args.n, seed=args.seed + 1, out_dir=args.out / "open")
    for label, rows in ((args.topic, pinned), ("none", open_topic)):
        print(f"--- topic={label} n={len(rows)}")
        for row in rows:
            names = ",".join(sorted(row["features"]))
            print(f"{row['template']}\t{row['sound_class']}\t{row['play_end_s']}\t{row['text']}\t{names}")
    print(f"dry_run pinned={len(pinned)} open={len(open_topic)} http=0")


if __name__ == "__main__":
    main()
