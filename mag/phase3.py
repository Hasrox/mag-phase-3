"""Phase 3 exit harness. Composer and renderer. No model call, no HTTP."""

from __future__ import annotations

import random
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from mag.catalog import seed_catalog
from mag.composer import blocked_pairs, captions_for, load_templates, sound_candidates
from mag.config import Config
from mag.features import build_features, registry_names
from mag.paths import ROOT
from mag.render import RenderRejected, render_image
from mag.vocab import values


@dataclass
class Coverage:
    templates: int
    mechanisms: dict[str, int]
    first_person: int
    emotion_legal: dict[str, int]
    image_legal_rate: float
    met: bool
    gaps: list[str] = field(default_factory=list)


def coverage(conn: sqlite3.Connection) -> Coverage:
    templates = load_templates(conn)
    mechanisms = Counter(item["style"].get("mechanism") for item in templates)
    first_person = sum(1 for item in templates if item["style"].get("voice") == "first_person")
    emotion_legal = {}
    for emotion in values("emotion"):
        if emotion == "none":
            continue
        emotion_legal[emotion] = sum(
            1 for item in templates
            if not item["requires"].get("emotion") or emotion in item["requires"]["emotion"]
        )
    images = conn.execute(
        "SELECT id FROM assets WHERE kind = 'image' AND state = 'active' AND is_gold = 0"
    ).fetchall()
    legal_counts = []
    for row in images:
        tags = _tags(conn, int(row["id"]))
        info = conn.execute("SELECT subject, action, profanity_in_image FROM image_info WHERE asset_id = ?", (row["id"],)).fetchone()
        legal_counts.append(len(captions_for(tags, dict(info) if info else {}, templates, random.Random(0), set())))
    rate = (
        sum(1 for count in legal_counts if count >= 3) / len(legal_counts)
        if legal_counts else 1.0
    )
    gaps = []
    if not 24 <= len(templates) <= 32:
        gaps.append(f"templates={len(templates)}; target 24 to 32")
    for mechanism in values("mechanism"):
        if mechanisms[mechanism] < 2:
            gaps.append(f"mechanism {mechanism} has {mechanisms[mechanism]}")
    if first_person < 4:
        gaps.append(f"first_person={first_person}; target 4")
    for emotion, count in emotion_legal.items():
        if count < 4:
            gaps.append(f"emotion {emotion} legal on {count}")
    if images and rate < 0.90:
        gaps.append(f"images with 3 legal templates={rate:.2f}; target 0.90")
    return Coverage(len(templates), dict(mechanisms), first_person, emotion_legal, rate, not gaps, gaps)


def _tags(conn: sqlite3.Connection, asset_id: int) -> dict[str, list[str]]:
    rows = conn.execute(
        "SELECT axis, value FROM tags WHERE asset_id = ? AND superseded = 0", (asset_id,)
    ).fetchall()
    out: dict[str, list[str]] = {}
    for row in rows:
        out.setdefault(row["axis"], []).append(row["value"])
    return out


def dry_run(conn, cfg, *, topic, n, seed, out_dir):
    """Render n compositions. Order is arbitrary. Zero HTTP."""
    seed_catalog(conn)
    templates = load_templates(conn)
    rng = random.Random(seed)
    images = conn.execute(
        "SELECT id, path FROM assets WHERE kind = 'image' AND state = 'active' AND is_gold = 0"
    ).fetchall()
    if topic:
        images = [row for row in images if topic in _tags(conn, int(row["id"])).get("topic", [])]
    if not images:
        raise RuntimeError("dry run has no pool images")
    known = set(registry_names())
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    previous_template = None
    previous_sound = None
    while len(records) < n:
        image = rng.choice(images)
        tags = _tags(conn, int(image["id"]))
        info = conn.execute(
            "SELECT subject, action, profanity_in_image FROM image_info WHERE asset_id = ?",
            (image["id"],),
        ).fetchone()
        captions = captions_for(tags, dict(info), templates, rng, set())
        captions = [item for item in captions if item.template_id != previous_template]
        if not captions:
            continue
        caption = rng.choice(captions)
        blocked = blocked_pairs(conn, int(image["id"]))
        sounds = sound_candidates(
            conn, tags, caption.style, rng, cfg,
            previous_sound=previous_sound, blocked_sounds=blocked,
        )
        sound = rng.choice(sounds)
        features = build_features(int(image["id"]), tags, caption, sound, _tags(conn, sound.sound_id) if sound.sound_id else {})
        src = ROOT / image["path"] if not Path(image["path"]).is_absolute() else Path(image["path"])
        suffix = ".gif" if src.suffix.lower() == ".gif" else ".png"
        dest = out_dir / f"{len(records):02d}{suffix}"
        try:
            render_image(src, caption.text, caption.render, dest, cfg.render)
        except RenderRejected:
            continue
        records.append({
            "image_id": int(image["id"]),
            "topic": topic or "",
            "template": caption.template_key,
            "text": caption.text,
            "sound_class": sound.sound_class,
            "play_end_s": sound.play_end_s,
            "render": str(dest),
            "features": features,
        })
        previous_template = caption.template_id
        previous_sound = sound.sound_id
    return records


def format_coverage(status: Coverage) -> str:
    lines = [
        f"templates={status.templates} first_person={status.first_person}",
        f"image_legal_rate={status.image_legal_rate:.2f}",
        f"phase3_coverage={'met' if status.met else 'open'}",
    ]
    for gap in status.gaps:
        lines.append(f"gap: {gap}")
    return "\n".join(lines)


def main() -> None:
    from mag.db import connect, init_db
    conn = connect()
    init_db(conn)
    seed_catalog(conn)
    print(format_coverage(coverage(conn)))
    print("phase3_exit=open until a dry run prints 20+20 rendered compositions")


if __name__ == "__main__":
    main()
