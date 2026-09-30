"""Review queue. List, human override, approve, explicit defer. No model call."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from mag.importing import write_human_tags
from mag.paths import DATA
from mag.promote import promote_asset

DEFERRED = DATA / "review_deferred.json"


def queue(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT a.id, a.path, a.is_gold, a.state, i.subject, i.action, i.tag_conf
           FROM assets a LEFT JOIN image_info i ON i.asset_id = a.id
           WHERE a.kind = 'image' AND a.state = 'review'
           ORDER BY a.id"""
    ).fetchall()


def deferred_ids(path: Path | None = None) -> set[int]:
    file = path or DEFERRED
    if not file.exists():
        return set()
    payload = json.loads(file.read_text())
    return {int(item) for item in payload.get("asset_ids", [])}


def defer(asset_ids: list[int], note: str, path: Path | None = None) -> None:
    file = path or DEFERRED
    file.parent.mkdir(parents=True, exist_ok=True)
    current = deferred_ids(file)
    current.update(asset_ids)
    file.write_text(json.dumps({"note": note, "asset_ids": sorted(current)}, indent=2) + "\n")


def set_human(conn: sqlite3.Connection, asset_id: int, tags: dict, info: dict | None = None) -> None:
    write_human_tags(conn, asset_id, tags, info)


def approve(conn: sqlite3.Connection, asset_id: int, conf_min: float) -> str:
    safety = conn.execute(
        """SELECT value FROM tags
           WHERE asset_id = ? AND axis = 'safety' AND source = 'human' AND superseded = 0""",
        (asset_id,),
    ).fetchone()
    if safety is None or safety["value"] != "ok":
        raise ValueError("approve requires a live human safety=ok tag")
    return promote_asset(conn, asset_id, conf_min)
