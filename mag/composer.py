"""Slot fill, validators, and the sound candidate rule. No model call."""

from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
from dataclasses import dataclass, field

from mag.catalog import EMOTION_WORD
from mag.config import Config
from mag.paths import CONFIG
from mag.sound import play_window

SLOT = re.compile(r"\{([a-z_]+)\}")
BRACE = re.compile(r"[{}]")
ALLOWED_SLOTS = {"subject", "action", "emotion_word"}


class CandidateDropped(ValueError):
    """A caption or pair failed a composer rule. Never repaired by a model."""


@dataclass
class Caption:
    template_id: int
    template_key: str
    variant: int
    text: str
    slots: dict[str, str]
    render: str
    style: dict[str, str]
    length_bucket: str
    profanity: bool


@dataclass
class SoundPick:
    sound_id: int | None
    sound_class: str
    role: str
    prior: float
    play_end_s: float


@dataclass
class Composition:
    image_id: int
    caption: Caption
    sound: SoundPick
    features: dict[str, float] = field(default_factory=dict)


def _one(tags: dict[str, list[str]], axis: str) -> str:
    values = tags.get(axis) or []
    return values[0] if values else ""


def length_bucket(text: str) -> str:
    size = len(text)
    if size <= 30:
        return "short"
    if size <= 70:
        return "mid"
    return "long"


def load_wordlist(path=None) -> set[str]:
    file = path or (CONFIG.parent / "profanity.txt")
    if not file.exists():
        return set()
    return {line.strip().lower() for line in file.read_text().splitlines() if line.strip()}


def fill_slots(pattern: str, slots: dict[str, str]) -> str:
    unknown = set(SLOT.findall(pattern)) - ALLOWED_SLOTS
    if unknown:
        raise CandidateDropped(f"invented slot {sorted(unknown)}")
    missing = [name for name in SLOT.findall(pattern) if not slots.get(name)]
    if missing:
        raise CandidateDropped(f"empty slot {missing}")
    text = pattern.format(**slots)
    if BRACE.search(text) or SLOT.search(text):
        raise CandidateDropped("unfilled slot or stray brace")
    return text


def validate_caption(text, *, max_chars, max_lines, profanity_in_image, wordlist):
    lines = text.split("\n")
    if len(lines) > max_lines or any(len(line) > max_chars for line in lines):
        raise CandidateDropped("overflow")
    words = re.findall(r"[a-z0-9']+", text.lower())
    if any(left == right for left, right in zip(words, words[1:])):
        raise CandidateDropped("repeated adjacent word")
    profanity = profanity_in_image or any(word in wordlist for word in words)
    return length_bucket(text), profanity


def legal(requires: dict, tags: dict[str, list[str]], style: dict[str, str]) -> bool:
    del style
    for axis, allowed in requires.items():
        if not allowed:
            continue
        have = set(tags.get(axis) or [])
        if axis == "caption_zone":
            have = {_one(tags, "caption_zone")}
        if not have.intersection(allowed):
            return False
    return True


def zone_covers(style: dict[str, str], zone: str, render: str) -> bool:
    if render == "bar":
        return True
    needed = {
        "single_line": {"top", "bottom", "both"},
        "two_part": {"both", "top", "bottom"},
        "label_pair": {"both"},
        "top_bottom": {"both"},
        "nobody_me": {"both"},
    }.get(style.get("structure", ""), {"both"})
    return zone in needed


def load_templates(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, key, patterns, render, requires, max_chars, max_lines FROM templates WHERE active = 1"
    ).fetchall()
    out = []
    for row in rows:
        tags = conn.execute(
            "SELECT axis, value FROM template_tags WHERE template_id = ?", (row["id"],)
        ).fetchall()
        style = {item["axis"]: item["value"] for item in tags}
        out.append({
            "id": int(row["id"]),
            "key": row["key"],
            "patterns": json.loads(row["patterns"]),
            "render": row["render"],
            "requires": json.loads(row["requires"]),
            "max_chars": int(row["max_chars"]),
            "max_lines": int(row["max_lines"]),
            "style": style,
        })
    return out


def captions_for(image_tags, info, templates, rng, wordlist):
    emotion = _one(image_tags, "emotion") or "none"
    slots = {
        "subject": (info.get("subject") or "").strip().lower(),
        "action": (info.get("action") or "").strip().lower(),
        "emotion_word": EMOTION_WORD.get(emotion, ""),
    }
    zone = _one(image_tags, "caption_zone") or "none"
    found = []
    for template in templates:
        if not legal(template["requires"], image_tags, template["style"]):
            continue
        if not zone_covers(template["style"], zone, template["render"]):
            continue
        variant = rng.randrange(len(template["patterns"]))
        try:
            text = fill_slots(template["patterns"][variant], slots)
            bucket, profanity = validate_caption(
                text,
                max_chars=template["max_chars"],
                max_lines=template["max_lines"],
                profanity_in_image=bool(info.get("profanity_in_image")),
                wordlist=wordlist,
            )
        except CandidateDropped:
            continue
        found.append(Caption(
            template_id=template["id"], template_key=template["key"], variant=variant,
            text=text, slots=slots, render=template["render"], style=template["style"],
            length_bucket=bucket, profanity=profanity,
        ))
    return found


def _prior_score(priors, image_tags, style, sound_class) -> float:
    have = {f"emotion:{_one(image_tags, 'emotion')}", f"mech:{style.get('mechanism', '')}"}
    best = 0.0
    for row in priors:
        pair = {row["feature_a"], row["feature_b"]}
        if f"sndclass:{sound_class}" in pair and pair & have:
            best = max(best, float(row["mean"]))
    return best


def _tags(conn, asset_id: int) -> dict[str, list[str]]:
    rows = conn.execute(
        "SELECT axis, value FROM tags WHERE asset_id = ? AND superseded = 0", (asset_id,)
    ).fetchall()
    out: dict[str, list[str]] = {}
    for row in rows:
        out.setdefault(row["axis"], []).append(row["value"])
    return out


def sound_candidates(conn, image_tags, style, rng, cfg, *, previous_sound=None, blocked_sounds=None, recent_sounds=None):
    """Top 2 by prior, 1 explore, and silence. Silence is exempt from the consecutive rule.

    Placeholder tags make every image share one prior, so the last 8 sounds are
    kept out of the prior slots. Otherwise the same file returns every other draw.
    """
    blocked = blocked_sounds or set()
    recent = set(recent_sounds or [])
    if previous_sound:
        recent.add(previous_sound)
    priors = conn.execute(
        "SELECT feature_a, feature_b, mean FROM pair_priors WHERE active = 1"
    ).fetchall()
    sounds = conn.execute(
        """SELECT a.id, s.duration_ms FROM assets a JOIN sound_info s ON s.asset_id = a.id
           WHERE a.kind = 'sound' AND a.state = 'active' AND a.is_gold = 0"""
    ).fetchall()
    scored = []
    for row in sounds:
        if int(row["id"]) in blocked:
            continue
        sound_class = _one(_tags(conn, int(row["id"])), "sound_class") or "other"
        scored.append((
            int(row["id"]), sound_class, int(row["duration_ms"]),
            _prior_score(priors, image_tags, style, sound_class),
        ))
    scored.sort(key=lambda item: (-item[3], item[0]))
    chosen = []
    pool = [item for item in scored if item[0] not in recent] or [
        item for item in scored if item[0] != previous_sound
    ]
    for sound_id, sound_class, duration_ms, prior in pool[:2]:
        chosen.append(SoundPick(
            sound_id, sound_class, "prior", prior,
            play_window(duration_ms / 1000, cfg.sound)[1],
        ))
    rest = [
        item for item in pool[2:]
        if item[0] not in {pick.sound_id for pick in chosen}
    ]
    if rest:
        sound_id, sound_class, duration_ms, prior = rng.choice(rest)
        chosen.append(SoundPick(
            sound_id, sound_class, "explore", prior,
            play_window(duration_ms / 1000, cfg.sound)[1],
        ))
    chosen.append(SoundPick(
        None, "silence", "silence",
        _prior_score(priors, image_tags, style, "silence"), 0.0,
    ))
    return chosen


def blocked_pairs(conn, image_id: int) -> set[int]:
    rows = conn.execute(
        "SELECT b_id FROM flags WHERE a_id = ? AND resolved_at IS NULL AND b_id IS NOT NULL",
        (image_id,),
    ).fetchall()
    return {int(row["b_id"]) for row in rows}


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
