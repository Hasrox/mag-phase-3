"""Batch tagger. Called by tag_run.sh after the gate passes. Not a button.

Walks untagged images, calls the server with the vocab grammar, validates,
and writes tags stamped with the run. A failed parse is retried once, then
left untagged. Does not accept the run; gold scoring does that.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mag.config import load_config
from mag.db import connect, init_db
from mag.gate import load_record, sha256_file
from mag.promote import open_run, rubric_hash, write_model_tags
from mag.tagger import tag_image


def main() -> None:
    parser = argparse.ArgumentParser(description="MAG Phase 2 batch tag")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--mmproj", required=True, type=Path)
    parser.add_argument("--gate-record", required=True, type=Path)
    parser.add_argument("--rubric", type=Path, default=Path("config/safety_rubric.txt"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config()
    record = load_record(args.gate_record)
    if not record.pass_:
        raise SystemExit("tagger refuse: gate record is not a pass")
    if sha256_file(args.model) != record.model_sha256 or sha256_file(args.mmproj) != record.mmproj_sha256:
        raise SystemExit("tagger refuse: model hash does not match the gate record")
    conn = connect()
    init_db(conn)
    run_id = open_run(
        conn,
        model_sha256=record.model_sha256,
        mmproj_sha256=record.mmproj_sha256,
        llama_commit=record.llama_commit,
        prompt_version=cfg.runtime.prompt_version,
        rubric_hash=rubric_hash(args.rubric),
        vocab_version="1.1",
        gate_ref=str(args.gate_record),
    )
    rows = conn.execute(
        """SELECT id, path FROM assets
           WHERE kind = 'image' AND state = 'new' AND is_gold = 0
           ORDER BY id"""
    ).fetchall()
    if args.limit:
        rows = rows[: args.limit]
    rubric = args.rubric.read_text()
    tagged = left = 0
    root = Path(__file__).resolve().parents[1]
    for row in rows:
        attempts = tag_image(
            root / row["path"], cfg, args.gate_record, args.model, args.mmproj, rubric,
        )
        payload = next((item.payload for item in attempts if item.payload is not None), None)
        if payload is None:
            left += 1
            continue
        conf = next(item.conf for item in attempts if item.payload is not None)
        write_model_tags(conn, int(row["id"]), payload, run_id, conf)
        tagged += 1
    print(f"run={run_id} tagged={tagged} untagged={left} accepted=0")


if __name__ == "__main__":
    main()
