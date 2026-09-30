"""Repoint sound rows at the files that are actually on disk.

The operator's assets/sounds holds the original filenames (akh.mp3, ...). Every
assets.path in the store points at assets/sounds/<sha256>.mp3, which does not
exist. The only path that resolved was the loudnorm wav, which is why the same
wav kept playing even after the serve path was fixed.

This walks assets/sounds, hashes each file, matches it to a store row by
sha256, and repoints assets.path plus sound_info.original_path and norm_path.

    python scripts/repair_sound_paths.py --dry-run
    python scripts/repair_sound_paths.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mag.db import connect, init_db
from mag.paths import ASSETS, ROOT


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Repoint sound rows at real files")
    parser.add_argument("--assets", type=Path, default=ASSETS / "sounds")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    conn = connect()
    init_db(conn)
    rows = conn.execute(
        "SELECT a.id, a.path, a.sha256, s.asset_id FROM assets a"
        " JOIN sound_info s ON s.asset_id = a.id WHERE a.kind = 'sound' ORDER BY a.id"
    ).fetchall()

    by_hash: dict[str, list[Path]] = {}
    for path in sorted(args.assets.rglob("*")):
        if not path.is_file() or path.suffix.lower() in {".wav"} and "norm" in path.parts:
            continue
        if path.parent.name == "norm":
            continue
        by_hash.setdefault(sha256_file(path), []).append(path)

    print(f"files under {args.assets}: {sum(len(v) for v in by_hash.values())}")
    print(f"sound rows: {len(rows)}")

    repointed = unresolved = already_ok = ambiguous = 0
    for row in rows:
        target = Path(row["path"])
        if not target.is_absolute():
            target = ROOT / target
        if target.is_file():
            already_ok += 1
            continue
        matches = by_hash.get(row["sha256"], [])
        if not matches:
            unresolved += 1
            print(f"  id={row['id']} UNRESOLVED sha={row['sha256'][:12]} path={row['path']}")
            continue
        if len(matches) > 1:
            ambiguous += 1
        chosen = matches[0]
        print(f"  id={row['id']} {row['path']} -> {_rel(chosen)}")
        repointed += 1
        if args.dry_run:
            continue
        relative = _rel(chosen)
        conn.execute("UPDATE assets SET path = ? WHERE id = ?", (relative, row["id"]))
        conn.execute(
            "UPDATE sound_info SET original_path = ?, norm_path = ? WHERE asset_id = ?",
            (relative, relative, row["id"]),
        )
    if not args.dry_run:
        conn.commit()
    print(
        f"repointed={repointed} already_ok={already_ok} "
        f"unresolved={unresolved} ambiguous={ambiguous} dry_run={args.dry_run}"
    )
    if unresolved and not args.dry_run:
        raise SystemExit("some rows could not be matched; nothing was deleted")


if __name__ == "__main__":
    main()
