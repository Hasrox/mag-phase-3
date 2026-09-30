"""Inbox placeholder tags: snapshot, clear, restore.

mag/inbox.py writes placeholder tags with source='human' so a dropped file
enters the pool on first launch. That is a scaffold, not provenance. Both
mag.importing._write_tag and mag.promote._write_model refuse to overwrite a
live human tag, and scripts/tag_batch.py only selects state='new', so a real
tag run against this store would write nothing at all.

Clearing supersedes the placeholder tags and returns the images to state='new'
so the tagger can own them. A snapshot is taken first, because the placeholders
are the only tags the store has until a model run succeeds.

    python scripts/clear_placeholder_tags.py --dry-run
    python scripts/clear_placeholder_tags.py
    python scripts/clear_placeholder_tags.py --restore data/placeholder_tags.json
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from mag.db import live_tags
from mag.paths import DATA

SNAPSHOT_PATH = DATA / "placeholder_tags.json"
# The axes the inbox fills in mag/inbox.py::_image_row.
PLACEHOLDER_AXES = {"emotion", "intensity", "family", "caption_zone", "safety", "topic"}
PLACEHOLDER_ACTIONS = {"waiting"}
MULTI_AXES = {"topic", "fit_emotion"}


class PlaceholderAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    asset_id: int
    path: str
    subject: str | None = None
    action: str | None = None
    tags: dict[str, list[str]]


class PlaceholderSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    created_at: str
    asset_count: int
    tag_count: int
    note: str
    assets: list[PlaceholderAsset]


def _is_placeholder(conn: sqlite3.Connection, asset_id: int) -> bool:
    """True when the live tags are the inbox scaffold and nothing else.

    mag/inbox.py writes a fixed tag set with a fixed action, "waiting", and a
    subject taken from the filename. An image the operator has since tagged for
    real has different values, so it is left alone. This is a heuristic on the
    scaffold, not a provenance column: tags carry no record of who wrote them
    beyond source, and the inbox writes source='human' so a dropped file enters
    the pool at all.
    """
    rows = conn.execute(
        "SELECT axis, source FROM tags WHERE asset_id = ? AND superseded = 0",
        (asset_id,),
    ).fetchall()
    if not rows:
        return False
    if any(str(row["source"]) != "human" for row in rows):
        return False
    if not all(str(row["axis"]) in PLACEHOLDER_AXES for row in rows):
        return False
    info = conn.execute(
        "SELECT action FROM image_info WHERE asset_id = ?", (asset_id,)
    ).fetchone()
    return info is not None and str(info["action"] or "") in PLACEHOLDER_ACTIONS


def snapshot_placeholders(conn: sqlite3.Connection) -> PlaceholderSnapshot:
    rows = conn.execute(
        """SELECT a.id, a.path, i.subject, i.action
           FROM assets a LEFT JOIN image_info i ON i.asset_id = a.id
           WHERE a.kind = 'image' AND a.is_gold = 0 ORDER BY a.id"""
    ).fetchall()
    assets: list[PlaceholderAsset] = []
    tag_count = 0
    for row in rows:
        asset_id = int(row["id"])
        if not _is_placeholder(conn, asset_id):
            continue
        tags = live_tags(conn, asset_id)
        tag_count += sum(len(values) for values in tags.values())
        assets.append(PlaceholderAsset(
            asset_id=asset_id,
            path=str(row["path"]),
            subject=row["subject"],
            action=row["action"],
            tags=tags,
        ))
    return PlaceholderSnapshot(
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        asset_count=len(assets),
        tag_count=tag_count,
        note=(
            "Placeholder tags written by mag.inbox at import. Clearing them lets "
            "the tagger own these images. Restore with "
            "scripts/clear_placeholder_tags.py --restore."
        ),
        assets=assets,
    )


def clear_placeholders(
    conn: sqlite3.Connection, snapshot: PlaceholderSnapshot, *, dry_run: bool = False,
) -> int:
    """Supersede the placeholder tags and return the images to state='new'."""
    cleared = 0
    for asset in snapshot.assets:
        if dry_run:
            cleared += 1
            continue
        conn.execute(
            "UPDATE tags SET superseded = 1 WHERE asset_id = ? AND superseded = 0",
            (asset.asset_id,),
        )
        conn.execute(
            "UPDATE image_info SET subject = NULL, action = NULL WHERE asset_id = ?",
            (asset.asset_id,),
        )
        conn.execute("UPDATE assets SET state = 'new' WHERE id = ?", (asset.asset_id,))
        cleared += 1
    if not dry_run:
        conn.commit()
    return cleared


def restore_placeholders(conn: sqlite3.Connection, snapshot: PlaceholderSnapshot) -> int:
    """Write the placeholders back as human tags and return images to active.

    live_tags returns a list for every axis, but write_human_tags compares
    safety against the scalar "ok", so a restored list would leave the image in
    review. Single-valued axes are unwrapped here.
    """
    from mag.importing import write_human_tags

    restored = 0
    for asset in snapshot.assets:
        tags: dict = {}
        for axis, values in asset.tags.items():
            tags[axis] = values if axis in MULTI_AXES else (values[0] if values else "")
            if not tags[axis]:
                tags.pop(axis)
        info = {
            key: value
            for key, value in (("subject", asset.subject), ("action", asset.action))
            if value
        }
        write_human_tags(conn, asset.asset_id, tags, info or None)
        restored += 1
    conn.commit()
    return restored


def load_snapshot(path: Path | None = None) -> PlaceholderSnapshot:
    return PlaceholderSnapshot.model_validate_json(
        (path or SNAPSHOT_PATH).read_text(encoding="utf-8")
    )
