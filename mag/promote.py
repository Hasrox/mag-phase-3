"""Tag-run provenance and the section 5.3 state machine.

new -> tagged (valid tags from an accepted run) -> active
(safety ok, confidence acceptable, no open flag).
review holds safety review, low confidence, reject, and flagged rows.
Human tags are never overwritten. Gold stays out of active.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from mag.goldscore import THRESHOLDS, GoldScores
from mag.tagger import TagPayload, grammar

ENUM_AXES = (
    "subject_kind", "emotion", "intensity", "family", "setting", "caption_zone", "safety",
)


def schema_hash() -> str:
    return hashlib.sha256(grammar().encode()).hexdigest()


def open_run(
    conn: sqlite3.Connection,
    *,
    model_sha256: str,
    mmproj_sha256: str,
    llama_commit: str,
    prompt_version: str,
    rubric_hash: str,
    vocab_version: str,
    gate_ref: str,
) -> int:
    cur = conn.execute(
        """INSERT INTO tag_runs(
               model_sha256, mmproj_sha256, llama_commit, prompt_version,
               rubric_hash, vocab_version, schema_hash, gate_ref, accepted
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)""",
        (
            model_sha256, mmproj_sha256, llama_commit, prompt_version,
            rubric_hash, vocab_version, schema_hash(), gate_ref,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def write_model_tags(
    conn: sqlite3.Connection,
    asset_id: int,
    payload: TagPayload,
    run_id: int,
    conf: float | None,
) -> None:
    """Stamp model tags. An axis with a live human tag is left alone."""
    accepted = conn.execute("SELECT accepted FROM tag_runs WHERE id = ?", (run_id,)).fetchone()
    if accepted is None:
        raise ValueError(f"tag run {run_id} does not exist")
    for axis in ENUM_AXES:
        _write_model(conn, asset_id, axis, getattr(payload, axis), run_id)
    for topic in payload.topics:
        _write_model(conn, asset_id, "topic", topic, run_id)
    conn.execute(
        """UPDATE image_info
           SET subject = ?, action = ?, subject_plural = ?, text_in_image = ?,
               profanity_in_image = ?, tag_conf = ?
           WHERE asset_id = ?""",
        (
            payload.subject, payload.action, int(payload.subject_plural),
            int(payload.text_in_image), int(payload.profanity_in_image), conf, asset_id,
        ),
    )
    # Mark the row tagged so a batch run resumes instead of re-tagging it.
    # Promotion to active still waits for an accepted run.
    conn.execute("UPDATE assets SET state = 'tagged' WHERE id = ?", (asset_id,))
    conn.commit()


def _write_model(conn: sqlite3.Connection, asset_id: int, axis: str, value: str, run_id: int) -> None:
    human = conn.execute(
        """SELECT 1 FROM tags
           WHERE asset_id = ? AND axis = ? AND source = 'human' AND superseded = 0""",
        (asset_id, axis),
    ).fetchone()
    if human:
        return
    known = conn.execute(
        "SELECT 1 FROM vocab WHERE axis = ? AND value = ? AND active = 1",
        (axis, value),
    ).fetchone()
    if known is None:
        raise ValueError(f"unknown vocab {axis}:{value}")
    if axis != "topic":
        conn.execute(
            "UPDATE tags SET superseded = 1 WHERE asset_id = ? AND axis = ? AND superseded = 0",
            (asset_id, axis),
        )
    else:
        conn.execute(
            """UPDATE tags SET superseded = 1
               WHERE asset_id = ? AND axis = 'topic' AND source = 'model' AND superseded = 0""",
            (asset_id,),
        )
    conn.execute(
        "INSERT INTO tags(asset_id, axis, value, source, run_id) VALUES (?, ?, ?, 'model', ?)",
        (asset_id, axis, value, run_id),
    )


def _open_flag(conn: sqlite3.Connection, asset_id: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM flags WHERE a_id = ? AND resolved_at IS NULL",
        (asset_id,),
    ).fetchone()
    return row is not None


def promote_asset(conn: sqlite3.Connection, asset_id: int, conf_min: float) -> str:
    """Apply the state machine. Requires an accepted run or a human tag."""
    row = conn.execute(
        "SELECT is_gold, state FROM assets WHERE id = ? AND kind = 'image'",
        (asset_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"image {asset_id} not in store")
    if row["state"] in ("retired", "quarantine"):
        return row["state"]
    live = conn.execute(
        """SELECT t.source, t.axis, t.value, r.accepted
           FROM tags t LEFT JOIN tag_runs r ON r.id = t.run_id
           WHERE t.asset_id = ? AND t.superseded = 0""",
        (asset_id,),
    ).fetchall()
    proven = [
        item for item in live
        if item["source"] == "human" or (item["source"] == "model" and item["accepted"] == 1)
    ]
    if not proven:
        conn.execute("UPDATE assets SET state = 'new' WHERE id = ?", (asset_id,))
        conn.commit()
        return "new"
    safety = next((item["value"] for item in proven if item["axis"] == "safety"), None)
    info = conn.execute("SELECT tag_conf FROM image_info WHERE asset_id = ?", (asset_id,)).fetchone()
    conf = None if info is None else info["tag_conf"]
    if int(row["is_gold"]) or _open_flag(conn, asset_id) or safety != "ok":
        state = "review"
    elif conf is not None and float(conf) < conf_min:
        state = "review"
    else:
        state = "active"
    if state == "active":
        conn.execute("UPDATE assets SET state = 'active' WHERE id = ?", (asset_id,))
    else:
        conn.execute("UPDATE assets SET state = 'review' WHERE id = ?", (asset_id,))
    conn.commit()
    return state


def record_scores(conn: sqlite3.Connection, run_id: int, scores: GoldScores) -> None:
    conn.execute(
        "UPDATE tag_runs SET gold_scores = ? WHERE id = ?",
        (scores.model_dump_json(), run_id),
    )
    conn.commit()


def accept_run(conn: sqlite3.Connection, run_id: int, scores: GoldScores, conf_min: float) -> int:
    """Mark accepted only when S-4 and S-5 passed. Then promote every image this run tagged."""
    if not scores.accepted:
        raise ValueError("tag run cannot be accepted: S-4 or S-5 failed")
    blob = json.loads(scores.model_dump_json())
    blob["thresholds"] = THRESHOLDS
    conn.execute(
        "UPDATE tag_runs SET accepted = 1, gold_scores = ? WHERE id = ?",
        (json.dumps(blob), run_id),
    )
    rows = conn.execute(
        "SELECT DISTINCT asset_id FROM tags WHERE run_id = ? AND superseded = 0",
        (run_id,),
    ).fetchall()
    for row in rows:
        promote_asset(conn, int(row["asset_id"]), conf_min)
    conn.commit()
    return len(rows)


def rubric_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
