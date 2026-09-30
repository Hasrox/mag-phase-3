"""Review CLI. List review rows, set human tags, approve, or defer.

    python scripts/review.py list
    python scripts/review.py set --id 4 --safety ok --emotion deadpan --topic animals
    python scripts/review.py approve --id 4
    python scripts/review.py defer --id 4 --note "operator deferred"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mag.config import load_config
from mag.db import connect, init_db
from mag.review import approve, defer, queue, set_human


def main() -> None:
    parser = argparse.ArgumentParser(description="MAG Phase 2 review queue")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    set_p = sub.add_parser("set")
    set_p.add_argument("--id", type=int, required=True)
    set_p.add_argument("--safety")
    set_p.add_argument("--emotion")
    set_p.add_argument("--family")
    set_p.add_argument("--intensity")
    set_p.add_argument("--topic", action="append", default=[])
    set_p.add_argument("--subject")
    set_p.add_argument("--action")
    approve_p = sub.add_parser("approve")
    approve_p.add_argument("--id", type=int, required=True)
    defer_p = sub.add_parser("defer")
    defer_p.add_argument("--id", type=int, action="append", required=True)
    defer_p.add_argument("--note", required=True)
    args = parser.parse_args()
    conn = connect()
    init_db(conn)
    cfg = load_config()
    if args.cmd == "list":
        rows = queue(conn)
        if not rows:
            print("review queue empty")
            return
        for row in rows:
            print(f"{row['id']} gold={row['is_gold']} conf={row['tag_conf']} {row['subject']} {row['path']}")
        return
    if args.cmd == "set":
        tags = {
            key: getattr(args, key)
            for key in ("safety", "emotion", "family", "intensity")
            if getattr(args, key)
        }
        if args.topic:
            tags["topic"] = args.topic
        info = {}
        if args.subject:
            info["subject"] = args.subject
        if args.action:
            info["action"] = args.action
        set_human(conn, args.id, tags, info or None)
        print(f"human tags set on {args.id}")
        return
    if args.cmd == "approve":
        state = approve(conn, args.id, cfg.runtime.tag_conf_min)
        print(f"{args.id} -> {state}")
        return
    defer(args.id, args.note)
    print(f"deferred {args.id}")


if __name__ == "__main__":
    main()
