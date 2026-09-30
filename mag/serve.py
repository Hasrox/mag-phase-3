"""Play-side serve path. Phase 4 body is the holdout draw.

score_and_pick is the seam. Phase 5 replaces the non-holdout body, not the UI.
This module does not load weights, does not update weights, and does not open a socket.
"""

from __future__ import annotations

import json
import random
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone

from mag.composer import (
    Caption,
    SoundPick,
    captions_for,
    load_templates,
    load_wordlist,
    sound_candidates,
    text_sha,
)
from mag.config import Config
from mag.db import live_tags
from mag.features import TIERS, build_features
from mag.paths import RENDER_CACHE, ROOT
from mag.render import RenderRejected, cache_path, render_image

POLICIES = ("coverage", "bandit", "holdout")
FLAG_KINDS = (
    "bad_tags",
    "unreadable_caption",
    "sound_clash",
    "unsafe",
    "broken_asset",
)
ASSET_FLAGS = {"bad_tags", "unsafe", "broken_asset"}


@dataclass
class ServeResult:
    impression_id: int | None
    policy: str
    image_id: int | None
    template_id: int | None
    sound_id: int | None
    text: str
    render_path: str
    sound_path: str
    pool_images: int
    candidate_count: int
    reason: str
    features: dict[str, float] = field(default_factory=dict)


def predict(features: dict[str, float]) -> tuple[float | None, float | None]:
    """No weights in Phase 4. Phase 5 replaces this body."""
    del features
    return None, None


def pick_candidate(policy: str, candidates: list[dict], rng: random.Random) -> dict:
    """Phase 4 ranking is the holdout draw for every slot.

    Phase 5 replaces the coverage and bandit branches. The candidate
    generator stays the one in candidates_for.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy}")
    if not candidates:
        raise RuntimeError("pick_candidate called with an empty pool")
    return rng.choice(candidates)


def status_line(topic: str | None, rated: int, pool_size: int) -> str:
    """Operator status. Policy is intentionally absent (section 9)."""
    label = topic or "none"
    return f"topic={label} rated={rated} pool={pool_size}"


def cooldown_k(pool_size: int, cap: int) -> int:
    return min(cap, pool_size // 4)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def abandon_stale(conn: sqlite3.Connection, minutes: int) -> int:
    """Pending rows older than the window become abandoned. They never train."""
    cur = conn.execute(
        """UPDATE impressions
           SET outcome = 'abandoned', resolved_at = ?
           WHERE outcome = 'pending'
             AND served_at <= strftime('%Y-%m-%dT%H:%M:%fZ', 'now', ?)""",
        (_now(), f"-{int(minutes)} minutes"),
    )
    conn.commit()
    return int(cur.rowcount)


def active_profile(conn: sqlite3.Connection, cfg: Config) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM profiles WHERE archived_at IS NULL ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row:
        return row
    cur = conn.execute(
        """INSERT INTO profiles(config_hash, weights_path, note, rng_seed)
           VALUES (?, NULL, 'phase4', ?)""",
        (cfg.config_hash, random.SystemRandom().randrange(1, 2**31)),
    )
    conn.commit()
    return conn.execute("SELECT * FROM profiles WHERE id = ?", (cur.lastrowid,)).fetchone()


def open_session(conn: sqlite3.Connection, profile_id: int, topic: str | None) -> sqlite3.Row:
    row = conn.execute(
        """SELECT * FROM sessions
           WHERE profile_id = ? AND ended_at IS NULL
           ORDER BY id DESC LIMIT 1""",
        (profile_id,),
    ).fetchone()
    if row and (row["topic_filter"] or None) == (topic or None):
        return row
    if row:
        conn.execute("UPDATE sessions SET ended_at = ? WHERE id = ?", (_now(), row["id"]))
    cur = conn.execute(
        "INSERT INTO sessions(profile_id, topic_filter) VALUES (?, ?)",
        (profile_id, topic),
    )
    conn.commit()
    return conn.execute("SELECT * FROM sessions WHERE id = ?", (cur.lastrowid,)).fetchone()


def reset_topic(conn: sqlite3.Connection, profile_id: int) -> sqlite3.Row:
    """Clear the filter only. Does not archive the profile."""
    return open_session(conn, profile_id, None)


def reset_profile(conn: sqlite3.Connection, cfg: Config, archive_dir: Path) -> sqlite3.Row:
    """New profile row. Archive an empty weight cache. History stays."""
    current = conn.execute(
        "SELECT * FROM profiles WHERE archived_at IS NULL ORDER BY id DESC LIMIT 1"
    ).fetchone()
    archive_dir.mkdir(parents=True, exist_ok=True)
    if current:
        empty = archive_dir / f"profile-{current['id']}.empty"
        empty.write_bytes(b"")
        conn.execute(
            "UPDATE profiles SET archived_at = ?, weights_path = ? WHERE id = ?",
            (_now(), str(empty), current["id"]),
        )
        conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE profile_id = ? AND ended_at IS NULL",
            (_now(), current["id"]),
        )
    cur = conn.execute(
        """INSERT INTO profiles(config_hash, weights_path, note, rng_seed)
           VALUES (?, NULL, 'reset', ?)""",
        (cfg.config_hash, random.SystemRandom().randrange(1, 2**31)),
    )
    conn.commit()
    return conn.execute("SELECT * FROM profiles WHERE id = ?", (cur.lastrowid,)).fetchone()


def pool_images(conn: sqlite3.Connection, topic: str | None) -> list[sqlite3.Row]:
    """G1 before cooldown. Active, not gold, safety ok, accepted-or-human tags, no open asset flag."""
    rows = conn.execute(
        """SELECT a.id, a.path, a.sha256
           FROM assets a
           WHERE a.kind = 'image' AND a.state = 'active' AND a.is_gold = 0
             AND NOT EXISTS (
               SELECT 1 FROM flags f
               WHERE f.a_id = a.id AND f.b_id IS NULL AND f.resolved_at IS NULL
             )"""
    ).fetchall()
    kept = []
    for row in rows:
        tags = live_tags(conn, int(row["id"]))
        if "ok" not in tags.get("safety", []):
            continue
        if topic and topic not in tags.get("topic", []):
            continue
        if not tags.get("topic"):
            continue
        if not _tags_accepted(conn, int(row["id"])):
            continue
        kept.append(row)
    return kept


def _tags_accepted(conn: sqlite3.Connection, asset_id: int) -> bool:
    rows = conn.execute(
        "SELECT source, run_id FROM tags WHERE asset_id = ? AND superseded = 0",
        (asset_id,),
    ).fetchall()
    if not rows:
        return False
    for row in rows:
        if row["source"] == "human":
            continue
        if row["source"] == "import":
            return False
        accepted = conn.execute(
            "SELECT accepted FROM tag_runs WHERE id = ?", (row["run_id"],)
        ).fetchone()
        if not accepted or not accepted["accepted"]:
            return False
    return True


def _recent(conn: sqlite3.Connection, profile_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT id, image_id, template_id, sound_id, composition_id, outcome, cooldown_k
           FROM impressions WHERE profile_id = ? ORDER BY id""",
        (profile_id,),
    ).fetchall()


def _cooling(recent: list[sqlite3.Row], image_id: int, k: int) -> bool:
    """True if image_id was served inside the last k impressions, with skip stretch."""
    if k <= 0 or not recent:
        return False
    window = recent[-k:]
    for row in window:
        if int(row["image_id"]) != image_id:
            continue
        if row["outcome"] == "skipped":
            continue
        return True
    # A skip stretches that image by skip_cooldown_mult. Stored cooldown_k is the stretch.
    for row in recent:
        if int(row["image_id"]) != image_id or row["outcome"] != "skipped":
            continue
        between = sum(1 for item in recent if item["id"] > row["id"])
        if between < int(row["cooldown_k"] or k):
            return True
    return False


def _composition_cooling(recent: list[sqlite3.Row], composition_id: int, window: int) -> bool:
    tail = recent[-window:]
    return any(row["composition_id"] == composition_id for row in tail)


def _blocked(conn: sqlite3.Connection, image_id: int, kind: str) -> set[int]:
    rows = conn.execute(
        """SELECT b_id FROM flags
           WHERE a_id = ? AND kind = ? AND resolved_at IS NULL AND b_id IS NOT NULL""",
        (image_id, kind),
    ).fetchall()
    return {int(row["b_id"]) for row in rows}


_FIT = {"topic_overlap", "emotion_fit", "int_match", "int_far"}
_FATIGUE = {"fat_tpl", "fat_snd", "fat_mech", "img_seen"}


def _tier(name: str) -> tuple[str, float]:
    if name in {"bias", "profanity"} or name in _FIT or name in _FATIGUE:
        if name in _FIT:
            return TIERS["fit"]
        if name in _FATIGUE:
            return TIERS["fatigue"]
        return TIERS[name]
    if name.startswith(("img:", "tpl:", "snd:")):
        return TIERS["id"]
    if "*" in name:
        return TIERS["cross"]
    return TIERS.get(name.split(":")[0], ("A", 1.0))


def _ensure_registry(conn: sqlite3.Connection, features: dict[str, float]) -> None:
    for name in features:
        tier, lam = _tier(name)
        conn.execute(
            "INSERT INTO feature_registry(name, tier, lambda) VALUES (?, ?, ?) ON CONFLICT(name) DO NOTHING",
            (name, tier, lam),
        )


def _upsert_caption(conn: sqlite3.Connection, image_id: int, caption: Caption) -> int:
    digest = text_sha(caption.text)
    found = conn.execute(
        """SELECT id FROM captions
           WHERE image_id = ? AND template_id = ? AND variant = ? AND text_sha = ?""",
        (image_id, caption.template_id, caption.variant, digest),
    ).fetchone()
    if found:
        return int(found["id"])
    cur = conn.execute(
        """INSERT INTO captions(image_id, template_id, variant, slots, text, text_sha)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (image_id, caption.template_id, caption.variant, json.dumps(caption.slots), caption.text, digest),
    )
    return int(cur.lastrowid)


def _upsert_composition(conn: sqlite3.Connection, image_id: int, caption_id: int, sound_id: int | None) -> int:
    found = conn.execute(
        """SELECT id FROM compositions
           WHERE image_id = ? AND caption_id = ? AND ifnull(sound_id, 0) = ifnull(?, 0)""",
        (image_id, caption_id, sound_id),
    ).fetchone()
    if found:
        return int(found["id"])
    cur = conn.execute(
        "INSERT INTO compositions(image_id, caption_id, sound_id) VALUES (?, ?, ?)",
        (image_id, caption_id, sound_id),
    )
    return int(cur.lastrowid)


def _sound_path(conn: sqlite3.Connection, sound_id: int | None) -> str:
    """The stored original file. assets.path is the single source of truth.

    This used to read sound_info.norm_path, which pointed at the loudnorm wav
    under assets/sounds/norm. Loudnorm was rejected by the operator on
    2026-09-30, so the play loop serves the original file and never a derived
    copy. Silence (sound_id None) returns "", which is a legal draw, not a
    missing file.
    """
    if sound_id is None:
        return ""
    row = conn.execute(
        "SELECT path FROM assets WHERE id = ? AND kind = 'sound'", (sound_id,)
    ).fetchone()
    if not row or not row["path"]:
        return ""
    path = Path(row["path"])
    return str(path if path.is_absolute() else ROOT / path)


def _sound_file_exists(conn: sqlite3.Connection, sound_id: int | None) -> bool:
    """False for silence (a legal candidate) and for a file that is gone."""
    if sound_id is None:
        return True
    path = _sound_path(conn, sound_id)
    return bool(path) and Path(path).is_file()


def recent_sound_ids(recent: list[sqlite3.Row], limit: int = 8) -> list[int]:
    """Sound ids from the last `limit` impressions, oldest first.

    mag.composer.sound_candidates can already exclude these from its prior
    slots, but mag/serve.py never passed them. Placeholder tags give every image
    the same prior, so without this the same file won every other draw.
    """
    ids: list[int] = []
    for row in list(recent)[-limit:] if limit > 0 else []:
        if row["sound_id"]:
            ids.append(int(row["sound_id"]))
    return ids


def candidates_for(
    conn: sqlite3.Connection,
    cfg: Config,
    *,
    topic: str | None,
    profile_id: int,
    rng: random.Random,
) -> tuple[list[dict], int, int, str]:
    """Same generator the bandit will use. Returns candidates, pool size, raw pool, reason."""
    raw = pool_images(conn, topic)
    if len(raw) < cfg.pool.min_images:
        return [], len(raw), len(raw), f"pool images {len(raw)} < {cfg.pool.min_images}"
    recent = _recent(conn, profile_id)
    previous_template = int(recent[-1]["template_id"]) if recent and recent[-1]["template_id"] else None
    previous_sound = int(recent[-1]["sound_id"]) if recent and recent[-1]["sound_id"] else None
    # Placeholder tags give every image the same prior, so the last eight served
    # sounds are kept out of the prior slots. Both halves are required: the
    # composer half landed in 6130a65, this call site is what makes it apply.
    recent_sounds = recent_sound_ids(recent, cfg.pool.sound_repeat_window)
    k = cooldown_k(len(raw), cfg.pool.image_cooldown_cap)
    templates = load_templates(conn)
    wordlist = load_wordlist()
    found: list[dict] = []
    eligible_images = 0
    for image in raw:
        image_id = int(image["id"])
        if _cooling(recent, image_id, k):
            continue
        tags = live_tags(conn, image_id)
        info = conn.execute(
            "SELECT subject, action, profanity_in_image FROM image_info WHERE asset_id = ?",
            (image_id,),
        ).fetchone()
        blocked_tpl = _blocked(conn, image_id, "unreadable_caption")
        captions = [
            item for item in captions_for(tags, dict(info) if info else {}, templates, rng, wordlist)
            if item.template_id != previous_template and item.template_id not in blocked_tpl
        ]
        if not captions:
            continue
        eligible_images += 1
        blocked_snd = _blocked(conn, image_id, "sound_clash")
        for caption in captions:
            sounds = sound_candidates(
                conn, tags, caption.style, rng, cfg,
                previous_sound=previous_sound, blocked_sounds=blocked_snd,
                recent_sounds=recent_sounds,
            )
            for sound in sounds:
                # A row whose file is gone is a broken asset, not silence. Silence
                # has sound_id None and is always kept.
                if not _sound_file_exists(conn, sound.sound_id):
                    continue
                found.append({
                    "image": image,
                    "tags": tags,
                    "info": dict(info) if info else {},
                    "caption": caption,
                    "sound": sound,
                })
    if len(found) < cfg.pool.min_compositions:
        return [], eligible_images, len(raw), (
            f"candidate compositions {len(found)} < {cfg.pool.min_compositions}"
        )
    return found, eligible_images, len(raw), ""


def _policy_for(recent: list[sqlite3.Row], profile_seed: int, block_size: int) -> tuple[str, int, int]:
    index = len(recent)
    block_id = index // block_size + 1
    block_pos = index % block_size
    holdout_pos = random.Random(profile_seed + block_id).randrange(block_size)
    policy = "holdout" if block_pos == holdout_pos else "coverage"
    return policy, block_id, block_pos


def score_and_pick(
    conn: sqlite3.Connection,
    cfg: Config,
    *,
    topic: str | None,
    profile_id: int,
    session_id: int,
    rng: random.Random | None = None,
    cache_dir: Path | None = None,
) -> ServeResult:
    """Insert the impression at serve time. Phase 5 replaces pick_candidate's body."""
    started = time.perf_counter()
    profile = conn.execute("SELECT * FROM profiles WHERE id = ?", (profile_id,)).fetchone()
    recent = _recent(conn, profile_id)
    # Seeding only from the profile gives the same stream on every call, so the
    # draw never advances and the same candidate is chosen until a cooldown
    # removes it. Fold in the impression count so each serve advances the
    # sequence. An explicit rng, which tests and Phase 5 pass, is respected.
    rng = rng or random.Random(
        int(profile["rng_seed"]) + int(profile_id) + len(recent) * 1_000_003
    )
    policy, block_id, block_pos = _policy_for(recent, int(profile["rng_seed"]), cfg.ranker.block_size)
    candidates, eligible, raw_pool, reason = candidates_for(
        conn, cfg, topic=topic, profile_id=profile_id, rng=rng,
    )
    if reason:
        return ServeResult(
            None, policy, None, None, None, "", "", "", raw_pool, len(candidates), reason,
        )
    seed = rng.randrange(1, 2**31)
    draw = random.Random(seed)
    # Uniform over images, then captions, then sounds. Same generator, three draws.
    images = sorted({int(item["image"]["id"]) for item in candidates})
    chosen_image = draw.choice(images)
    captions = [item for item in candidates if int(item["image"]["id"]) == chosen_image]
    caption_ids = sorted({item["caption"].template_id for item in captions})
    chosen_template = draw.choice(caption_ids)
    sounds = [item for item in captions if item["caption"].template_id == chosen_template]
    chosen = draw.choice(sounds)
    propensity = (1 / len(images)) * (1 / len(caption_ids)) * (1 / max(len(sounds), 1))
    # Phase 4 ignores policy for the draw. The logged policy still marks the holdout slot.
    picked = pick_candidate(policy, [chosen], draw)
    pred_mean, pred_bonus = predict(picked.get("features", {}))
    image = picked["image"]
    caption: Caption = picked["caption"]
    sound: SoundPick = picked["sound"]
    tags = picked["tags"]
    sound_tags = live_tags(conn, sound.sound_id) if sound.sound_id else {}
    recent_templates = [int(row["template_id"]) for row in recent if row["template_id"]]
    recent_sounds = [int(row["sound_id"]) for row in recent if row["sound_id"]]
    recent_mechs = []
    for row in recent:
        if not row["template_id"]:
            continue
        mech = conn.execute(
            "SELECT value FROM template_tags WHERE template_id = ? AND axis = 'mechanism'",
            (row["template_id"],),
        ).fetchone()
        if mech:
            recent_mechs.append(mech["value"])
    image_seen = sum(1 for row in recent if int(row["image_id"]) == int(image["id"]))
    features = build_features(
        int(image["id"]), tags, caption, sound, sound_tags,
        recent_templates=recent_templates,
        recent_sounds=recent_sounds,
        recent_mechs=recent_mechs,
        image_seen=image_seen,
    )
    _ensure_registry(conn, features)
    src = Path(image["path"])
    src = src if src.is_absolute() else ROOT / src
    cache = cache_dir or RENDER_CACHE
    # Keyed by the composition, not the RNG seed. The seed changes every serve,
    # so a seed-keyed cache never reuses a file and never replaces a stale one.
    dest = cache_path(cache, str(image["sha256"]), caption.text, caption.render, src.suffix)
    if not dest.exists():
        try:
            render_image(src, caption.text, caption.render, dest, cfg.render)
        except RenderRejected as exc:
            dest.unlink(missing_ok=True)
            conn.execute(
                "INSERT INTO render_rejects(session_id, image_id, template_id, reason) VALUES (?, ?, ?, ?)",
                (session_id, int(image["id"]), caption.template_id, str(exc)),
            )
            conn.commit()
            return ServeResult(
                None, policy, int(image["id"]), caption.template_id, sound.sound_id,
                caption.text, "", "", raw_pool, len(candidates), "render rejected",
            )
    caption_id = _upsert_caption(conn, int(image["id"]), caption)
    composition_id = _upsert_composition(conn, int(image["id"]), caption_id, sound.sound_id)
    if _composition_cooling(recent, composition_id, cfg.pool.composition_cooldown):
        return ServeResult(
            None, policy, int(image["id"]), caption.template_id, sound.sound_id,
            caption.text, str(dest), "", raw_pool, len(candidates), "composition cooldown",
        )
    k = cooldown_k(raw_pool, cfg.pool.image_cooldown_cap)
    latency_ms = int((time.perf_counter() - started) * 1000)
    cur = conn.execute(
        """INSERT INTO impressions(
             session_id, profile_id, composition_id, image_id, template_id, variant, sound_id,
             topic_filter, policy, block_id, block_pos, capped, bonus_won, propensity,
             pred_mean, pred_bonus, candidate_count, seed, config_hash, weights_version,
             features_json, outcome, latency_ms, cooldown_k
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, ?, ?, ?, 0, ?, 'pending', ?, ?)""",
        (
            session_id, profile_id, composition_id, int(image["id"]), caption.template_id,
            caption.variant, sound.sound_id, topic, policy, block_id, block_pos,
            propensity if policy == "holdout" else None,
            pred_mean, pred_bonus, len(candidates), seed, cfg.config_hash,
            json.dumps(features), latency_ms, k,
        ),
    )
    conn.commit()
    return ServeResult(
        int(cur.lastrowid), policy, int(image["id"]), caption.template_id, sound.sound_id,
        caption.text, str(dest), _sound_path(conn, sound.sound_id), raw_pool, len(candidates), "",
        features,
    )


def rate(conn: sqlite3.Connection, cfg: Config, impression_id: int, stars: int, attribution: list[str] | None = None) -> None:
    if stars not in range(1, 6):
        raise ValueError("stars must be 1..5")
    chips = [item for item in (attribution or []) if item in {"image", "caption", "sound"}]
    conn.execute(
        """UPDATE impressions
           SET outcome = 'rated', stars = ?, reward = ?, resolved_at = ?, attribution = ?
           WHERE id = ? AND outcome = 'pending'""",
        (stars, cfg.reward.of(stars), _now(), json.dumps(chips), impression_id),
    )
    conn.commit()


def skip(conn: sqlite3.Connection, cfg: Config, impression_id: int) -> None:
    row = conn.execute("SELECT cooldown_k FROM impressions WHERE id = ?", (impression_id,)).fetchone()
    stretched = int(row["cooldown_k"] or 0) * cfg.pool.skip_cooldown_mult
    conn.execute(
        """UPDATE impressions
           SET outcome = 'skipped', stars = NULL, reward = NULL,
               train_value = ?, train_weight = ?, cooldown_k = ?, resolved_at = ?
           WHERE id = ? AND outcome = 'pending'""",
        (cfg.ranker.skip_train_value, cfg.ranker.skip_train_weight, stretched, _now(), impression_id),
    )
    conn.commit()


def flag(conn: sqlite3.Connection, impression_id: int, kind: str) -> None:
    if kind not in FLAG_KINDS:
        raise ValueError(f"unknown flag {kind}")
    row = conn.execute(
        "SELECT image_id, template_id, sound_id FROM impressions WHERE id = ?",
        (impression_id,),
    ).fetchone()
    if kind in ASSET_FLAGS:
        a_id, b_id = int(row["image_id"]), None
        conn.execute("UPDATE assets SET state = 'review' WHERE id = ?", (a_id,))
    elif kind == "unreadable_caption":
        a_id, b_id = int(row["image_id"]), int(row["template_id"])
    else:
        a_id, b_id = int(row["image_id"]), row["sound_id"]
    conn.execute(
        "INSERT INTO flags(impression_id, kind, a_id, b_id) VALUES (?, ?, ?, ?)",
        (impression_id, kind, a_id, b_id),
    )
    conn.execute(
        """UPDATE impressions
           SET outcome = 'flagged', reward = NULL, resolved_at = ?
           WHERE id = ? AND outcome = 'pending'""",
        (_now(), impression_id),
    )
    conn.commit()


def rated_count(conn: sqlite3.Connection, profile_id: int) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM impressions WHERE profile_id = ? AND outcome = 'rated'",
        (profile_id,),
    ).fetchone()
    return int(row[0])
