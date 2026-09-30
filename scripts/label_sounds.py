"""Bulk-label sounds from a CSV. The model has no audio input, so this is the
operator's job and nothing automates it.

CSV header: id,class,intensity,topics,fit_emotion
  id            store asset id of the sound
  class         sound_class, from the vocabulary
  intensity     low | mid | high
  topics        semicolon-separated, may be empty
  fit_emotion   semicolon-separated, may be empty

Every value is validated against the vocab table before anything is written; an
unknown value aborts the whole file rather than half-applying it.

    python scripts/label_sounds.py --csv sounds.csv --dry-run
    python scripts/label_sounds.py --csv sounds.csv
    python scripts/label_sounds.py --template
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mag.db import connect, init_db
from mag.importing import _write_tag
from mag.vocab import values

HEADER = ["id", "class", "intensity", "topics", "fit_emotion"]


def label_sounds_from_csv(
    conn, path: Path, *, dry_run: bool = True,
) -> list[tuple[int, str]]:
    """Validate the whole file, then write. Returns (asset_id, summary) pairs."""
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    if not rows:
        raise ValueError("csv has no rows")
    missing = [name for name in HEADER if name not in rows[0]]
    if missing:
        raise ValueError(f"csv is missing columns: {missing}")

    plans: list[tuple[int, dict]] = []
    for line, row in enumerate(rows, start=2):
        try:
            asset_id = int(str(row["id"]).strip())
        except ValueError as exc:
            raise ValueError(f"line {line}: id must be an integer") from exc
        exists = conn.execute(
            "SELECT 1 FROM assets WHERE id = ? AND kind = 'sound'", (asset_id,)
        ).fetchone()
        if exists is None:
            raise ValueError(f"line {line}: {asset_id} is not a sound in the store")
        tags: dict[str, list[str]] = {}
        for axis, column in (
            ("sound_class", "class"),
            ("intensity", "intensity"),
        ):
            value = str(row[column]).strip()
            if value not in values(axis):
                raise ValueError(f"line {line}: {column}={value!r} is outside {axis} vocab")
            tags[axis] = [value]
        for axis, column in (("topic", "topics"), ("fit_emotion", "fit_emotion")):
            raw = str(row[column] or "").strip()
            if not raw:
                continue
            items = [item.strip() for item in raw.split(";") if item.strip()]
            for item in items:
                if item not in values(axis):
                    raise ValueError(f"line {line}: {column}={item!r} is outside {axis} vocab")
            tags[axis] = items
        plans.append((asset_id, tags))

    if dry_run:
        return [(asset_id, _summary(tags)) for asset_id, tags in plans]

    applied: list[tuple[int, str]] = []
    for asset_id, tags in plans:
        for axis, items in tags.items():
            conn.execute(
                "UPDATE tags SET superseded = 1 WHERE asset_id = ? AND axis = ? AND superseded = 0",
                (asset_id, axis),
            )
            for item in items:
                _write_tag(conn, asset_id, axis, item, "human")
        applied.append((asset_id, _summary(tags)))
    conn.commit()
    return applied


def _summary(tags: dict[str, list[str]]) -> str:
    return " ".join(f"{axis}={'/'.join(values_)}" for axis, values_ in sorted(tags.items()))


def template(rows: list[dict]) -> str:
    lines = [",".join(HEADER)]
    for row in rows:
        lines.append(
            f"{row['id']},{row.get('class', 'other')},{row.get('intensity', 'mid')},"
            f"{row.get('topics', '')},{row.get('fit_emotion', '')}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Label sounds from a CSV")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--template", action="store_true", help="print a CSV to fill")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--apply", action="store_true", help="write, do not just report")
    args = parser.parse_args()

    conn = connect()
    init_db(conn)
    if args.template:
        rows = conn.execute(
            "SELECT a.id, a.path FROM assets a WHERE a.kind = 'sound' ORDER BY a.id"
        ).fetchall()
        for row in rows:
            print(f"# {row['id']}\t{row['path']}", file=sys.stderr)
        print(template([{"id": row["id"]} for row in rows]), end="")
        return
    if not args.csv:
        parser.error("--csv is required unless --template is used")
    applied = label_sounds_from_csv(conn, args.csv, dry_run=not args.apply)
    for asset_id, summary in applied:
        print(f"{asset_id}\t{summary}")
    print(f"sounds={len(applied)} applied={args.apply}")


if __name__ == "__main__":
    main()
