"""Record the S-1 smoke verdict. S-1 is ten manual generations.

    python scripts/record_s1.py --from-db
    python scripts/record_s1.py --stars 3,3,4,2,5,3,1,4,3,2 --flags none

--from-db takes the last ten rated impressions of the active profile. S-1
failure does not block Phase 5; it blocks any claim that the product is fun.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mag.config import load_config
from mag.db import connect, init_db
from mag.phase4 import record_s1, s1_verdict
from mag.serve import active_profile

FLAG_KINDS = ("bad_tags", "unreadable_caption", "sound_clash", "unsafe", "broken_asset")


def last_ten(conn, cfg) -> tuple[list[int], list[str]]:
    profile = active_profile(conn, cfg)
    rows = conn.execute(
        """SELECT i.stars, f.kind
           FROM impressions i
           LEFT JOIN flags f ON f.impression_id = i.id
           WHERE i.profile_id = ? AND i.outcome = 'rated'
           ORDER BY i.id DESC LIMIT 10""",
        (int(profile["id"]),),
    ).fetchall()
    stars = [int(row["stars"]) for row in rows if row["stars"] is not None]
    flags = [str(row["kind"]) for row in rows if row["kind"]]
    return list(reversed(stars)), flags


def main() -> None:
    parser = argparse.ArgumentParser(description="Record the S-1 smoke verdict")
    parser.add_argument("--from-db", action="store_true")
    parser.add_argument("--stars", help="comma-separated 1..5, exactly ten")
    parser.add_argument("--flags", default="none", help="comma-separated flag kinds, or none")
    args = parser.parse_args()

    cfg = load_config()
    conn = connect()
    init_db(conn)

    if args.stars:
        stars = [int(item) for item in args.stars.split(",") if item.strip()]
        flags = [] if args.flags == "none" else [f.strip() for f in args.flags.split(",") if f.strip()]
    elif args.from_db:
        stars, flags = last_ten(conn, cfg)
    else:
        parser.error("pass --from-db or --stars")

    if len(stars) != 10:
        print(f"s1 needs exactly ten rated impressions; found {len(stars)}")
        print("rate ten memes first, then rerun, or pass --stars explicitly")
        raise SystemExit(1)
    unknown = [flag for flag in flags if flag not in FLAG_KINDS]
    if unknown:
        raise SystemExit(f"unknown flag kinds: {unknown}")

    verdict = record_s1(stars, flags)
    for key, value in verdict.items():
        print(f"{key}={value}")
    print(f"s1_passed={verdict['passed']}")
    print("note: S-1 failing does not block Phase 5. It blocks a fun claim.")


if __name__ == "__main__":
    main()
