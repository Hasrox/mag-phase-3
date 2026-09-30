"""Build the 40-image gold manifest for the operator to hand-label.

S-4 needs 40 images tagged on every axis by a person, stratified across topics
and emotions, never in the pool. This samples them and writes the manifest with
empty label fields. It does not label anything: the labels are the operator's
job, and a model labelling its own gold set would make S-4 meaningless.

    python scripts/make_gold_manifest.py
    python scripts/make_gold_manifest.py --force

After filling the labels in:
    python scripts/import_store.py --gold data/gold/manifest.json
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mag.db import connect, init_db, live_tags
from mag.paths import DATA
from mag.vocab import values

MANIFEST = DATA / "gold" / "manifest.json"
LABEL_FIELDS = (
    "emotion", "family", "intensity", "topics", "safety",
    "subject_usable", "action_usable",
)


def build_gold_manifest(conn: sqlite3.Connection, n: int = 40, seed: int = 1103) -> list[dict]:
    """A deterministic stratified sample. Same input, same manifest.

    Sampled from every stored image, not just active ones: a gold image is
    excluded from the pool by is_gold, not by state, and right now nothing is
    active because the placeholder tags have been cleared.
    """
    rows = conn.execute(
        """SELECT a.id, a.path FROM assets a
           WHERE a.kind = 'image' AND a.is_gold = 0
           ORDER BY a.id"""
    ).fetchall()
    if len(rows) < n:
        raise ValueError(f"only {len(rows)} stored images; need {n} for a gold set")

    buckets: dict[tuple[str, str], list[sqlite3.Row]] = {}
    for row in rows:
        tags = live_tags(conn, int(row["id"]))
        topic = (tags.get("topic") or ["everyday"])[0]
        emotion = (tags.get("emotion") or ["none"])[0]
        buckets.setdefault((topic, emotion), []).append(row)

    rng = random.Random(seed)
    order = sorted(buckets)
    rng.shuffle(order)
    picked: list[sqlite3.Row] = []
    # Round-robin across strata so no single topic dominates the set.
    while len(picked) < n and any(buckets[key] for key in order):
        for key in order:
            if buckets[key] and len(picked) < n:
                picked.append(buckets[key].pop())

    manifest: list[dict] = []
    for row in picked[:n]:
        asset_id = int(row["id"])
        tags = live_tags(conn, asset_id)
        manifest.append({
            "file": str(row["path"]).replace("\\", "/"),
            "license_note": "local collection",
            "is_gold": True,
            "source": "gold",
            "current_tags": tags,
            "labels": {field: None for field in LABEL_FIELDS},
        })
    return manifest


def stratify_warning(manifest: list[dict]) -> str:
    """Say so when the sample could not actually be stratified."""
    topics = [row["current_tags"].get("topic", ["everyday"])[0] for row in manifest]
    emotions = [row["current_tags"].get("emotion", ["none"])[0] for row in manifest]
    if len(set(topics)) == 1 or len(set(emotions)) == 1:
        return (
            "WARNING: the sample is not stratified. The store holds no real tags yet, "
            "so every image fell into one bucket. The 40 images are still valid gold "
            "rows, but the set is not spread across topics and emotions as section 5.3 "
            "requires. Re-run this after a tag run succeeds, or accept the weaker set."
        )
    return f"stratified across topics={sorted(set(topics))} emotions={sorted(set(emotions))}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample the 40-image gold set")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument("--out", type=Path, default=MANIFEST)
    parser.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        print(f"{args.out} exists; refusing to overwrite without --force")
        raise SystemExit(1)
    conn = connect()
    init_db(conn)
    try:
        manifest = build_gold_manifest(conn, args.n, args.seed)
    except ValueError as exc:
        print(f"cannot build the gold set: {exc}")
        raise SystemExit(1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} images={len(manifest)}")
    print(stratify_warning(manifest))
    print("current_tags is what the store believes now; labels is what you must fill in.")
    print("Keep these images out of the pool: import_store --gold flags is_gold.")


if __name__ == "__main__":
    main()
