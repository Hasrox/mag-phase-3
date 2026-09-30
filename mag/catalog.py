"""Authored templates and Appendix B priors. Phase 3 source, loaded into the store."""

from __future__ import annotations

import json
import sqlite3

from mag.vocab import values

EMOTION_WORD = {
    "deadpan": "blank",
    "angry": "furious",
    "smug": "smug",
    "crying": "wrecked",
    "shocked": "stunned",
    "none": "still",
}

TEMPLATES: list[dict] = [
    {"key": "label_is", "patterns": ["{subject} is just {action}", "this {subject} is {action}"], "render": "bar", "requires": {}, "max_chars": 48, "max_lines": 1, "mechanism": "label", "voice": "third_person", "register": "dry", "structure": "single_line"},
    {"key": "label_called", "patterns": ["they call this {subject} {action}", "known as {subject}, {action}"], "render": "bar", "requires": {}, "max_chars": 56, "max_lines": 1, "mechanism": "label", "voice": "third_person", "register": "mock_formal", "structure": "single_line"},
    {"key": "label_emotion", "patterns": ["{emotion_word} {subject}", "a {emotion_word} {subject}"], "render": "impact", "requires": {"caption_zone": ["bottom", "both"]}, "max_chars": 32, "max_lines": 1, "mechanism": "label", "voice": "third_person", "register": "dry", "structure": "single_line"},
    {"key": "label_pair", "patterns": ["{subject}\n{action}"], "render": "impact", "requires": {"caption_zone": ["both"]}, "max_chars": 28, "max_lines": 2, "mechanism": "label", "voice": "third_person", "register": "dry", "structure": "label_pair"},
    {"key": "contrast_me", "patterns": ["me: calm\n{subject}: {action}", "expectation: peace\n{subject}: {action}"], "render": "bar", "requires": {}, "max_chars": 40, "max_lines": 2, "mechanism": "contrast", "voice": "first_person", "register": "dry", "structure": "two_part"},
    {"key": "contrast_but", "patterns": ["{subject}, but {action}", "{subject}. actually {action}"], "render": "bar", "requires": {}, "max_chars": 48, "max_lines": 1, "mechanism": "contrast", "voice": "third_person", "register": "dry", "structure": "two_part"},
    {"key": "contrast_smug", "patterns": ["{subject} thinks it won\nstill {action}"], "render": "impact", "requires": {"emotion": ["smug"], "caption_zone": ["both"]}, "max_chars": 32, "max_lines": 2, "mechanism": "contrast", "voice": "third_person", "register": "chaotic", "structure": "top_bottom"},
    {"key": "under_fine", "patterns": ["this is fine", "{subject} is fine"], "render": "bar", "requires": {}, "max_chars": 24, "max_lines": 1, "mechanism": "understatement", "voice": "third_person", "register": "dry", "structure": "single_line"},
    {"key": "under_me", "patterns": ["i am simply {action}", "no notes. {subject} is {action}"], "render": "bar", "requires": {}, "max_chars": 40, "max_lines": 1, "mechanism": "understatement", "voice": "first_person", "register": "sincere", "structure": "single_line"},
    {"key": "under_still", "patterns": ["{emotion_word}. anyway.", "still {action}"], "render": "impact", "requires": {"caption_zone": ["bottom", "both"]}, "max_chars": 24, "max_lines": 1, "mechanism": "understatement", "voice": "first_person", "register": "dry", "structure": "single_line"},
    {"key": "escal_then", "patterns": ["first {subject}\nthen {action}", "it started with {subject}\nnow {action}"], "render": "bar", "requires": {}, "max_chars": 40, "max_lines": 2, "mechanism": "escalation", "voice": "third_person", "register": "chaotic", "structure": "two_part"},
    {"key": "escal_me", "patterns": ["i saw {subject}\ni am {action}"], "render": "bar", "requires": {}, "max_chars": 36, "max_lines": 2, "mechanism": "escalation", "voice": "first_person", "register": "chaotic", "structure": "two_part"},
    {"key": "escal_peak", "patterns": ["{subject}\n{action} harder"], "render": "impact", "requires": {"caption_zone": ["both"]}, "max_chars": 28, "max_lines": 2, "mechanism": "escalation", "voice": "imperative", "register": "chaotic", "structure": "top_bottom"},
    {"key": "rel_me", "patterns": ["me, {action}", "me when {subject} is {action}"], "render": "bar", "requires": {}, "max_chars": 40, "max_lines": 1, "mechanism": "relatable", "voice": "first_person", "register": "sincere", "structure": "single_line"},
    {"key": "rel_us", "patterns": ["all of us, {action}", "everyone and this {subject}"], "render": "bar", "requires": {}, "max_chars": 40, "max_lines": 1, "mechanism": "relatable", "voice": "first_person", "register": "sincere", "structure": "single_line"},
    {"key": "rel_cry", "patterns": ["{emotion_word} because {subject}"], "render": "impact", "requires": {"emotion": ["crying"], "caption_zone": ["bottom", "both"]}, "max_chars": 36, "max_lines": 1, "mechanism": "relatable", "voice": "first_person", "register": "sincere", "structure": "single_line"},
    {"key": "absurd_why", "patterns": ["why is {subject} {action}", "{subject} should not be {action}"], "render": "bar", "requires": {}, "max_chars": 48, "max_lines": 1, "mechanism": "absurd", "voice": "second_person", "register": "chaotic", "structure": "single_line"},
    {"key": "absurd_law", "patterns": ["illegal: {subject} {action}", "the rules did not cover {subject}"], "render": "bar", "requires": {}, "max_chars": 48, "max_lines": 1, "mechanism": "absurd", "voice": "third_person", "register": "mock_formal", "structure": "single_line"},
    {"key": "absurd_top", "patterns": ["{subject}\n{action} on purpose"], "render": "impact", "requires": {"caption_zone": ["both"]}, "max_chars": 32, "max_lines": 2, "mechanism": "absurd", "voice": "third_person", "register": "chaotic", "structure": "top_bottom"},
    {"key": "own_me", "patterns": ["i did this", "this was me, {action}"], "render": "bar", "requires": {}, "max_chars": 32, "max_lines": 1, "mechanism": "self_own", "voice": "first_person", "register": "dry", "structure": "single_line"},
    {"key": "own_again", "patterns": ["{subject} again\nsame {action}"], "render": "bar", "requires": {}, "max_chars": 32, "max_lines": 2, "mechanism": "self_own", "voice": "first_person", "register": "dry", "structure": "two_part"},
    {"key": "own_impact", "patterns": ["my fault\n{subject}"], "render": "impact", "requires": {"caption_zone": ["both"]}, "max_chars": 24, "max_lines": 2, "mechanism": "self_own", "voice": "first_person", "register": "sincere", "structure": "nobody_me"},
    {"key": "pov_you", "patterns": ["pov: you are {subject}", "pov: {subject} {action}"], "render": "bar", "requires": {}, "max_chars": 40, "max_lines": 1, "mechanism": "pov", "voice": "second_person", "register": "dry", "structure": "single_line"},
    {"key": "pov_watch", "patterns": ["you, watching {subject}", "you notice {subject} {action}"], "render": "bar", "requires": {}, "max_chars": 40, "max_lines": 1, "mechanism": "pov", "voice": "second_person", "register": "dry", "structure": "single_line"},
    {"key": "pov_order", "patterns": ["look at {subject}\nkeep {action}"], "render": "impact", "requires": {"caption_zone": ["both"]}, "max_chars": 28, "max_lines": 2, "mechanism": "pov", "voice": "imperative", "register": "mock_formal", "structure": "top_bottom"},
    {"key": "anti_nothing", "patterns": ["and then nothing", "{subject}. the end."], "render": "bar", "requires": {}, "max_chars": 28, "max_lines": 1, "mechanism": "anti_joke", "voice": "third_person", "register": "dry", "structure": "single_line"},
    {"key": "anti_setup", "patterns": ["{subject} walks in\nnothing happens"], "render": "bar", "requires": {}, "max_chars": 32, "max_lines": 2, "mechanism": "anti_joke", "voice": "third_person", "register": "dry", "structure": "two_part"},
    {"key": "anti_impact", "patterns": ["no punchline\n{subject}"], "render": "impact", "requires": {"caption_zone": ["both"]}, "max_chars": 24, "max_lines": 2, "mechanism": "anti_joke", "voice": "third_person", "register": "dry", "structure": "nobody_me"},
]

PRIORS: list[dict] = [
    {"feature_a": "emotion:crying", "feature_b": "sndclass:trombone", "mean": 0.75, "note": "Sad trombone on defeat."},
    {"feature_a": "emotion:shocked", "feature_b": "sndclass:gasp", "mean": 0.75, "note": "Sound matches the face."},
    {"feature_a": "emotion:shocked", "feature_b": "sndclass:boom", "mean": 0.65, "note": "Impact on the reveal."},
    {"feature_a": "emotion:smug", "feature_b": "sndclass:sting", "mean": 0.70, "note": "Punctuates the smirk."},
    {"feature_a": "emotion:angry", "feature_b": "sndclass:boom", "mean": 0.65, "note": "Escalation lands."},
    {"feature_a": "emotion:deadpan", "feature_b": "sndclass:silence", "mean": 0.70, "note": "Silence is the joke."},
    {"feature_a": "emotion:crying", "feature_b": "sndclass:boom", "mean": 0.35, "note": "Mismatch."},
    {"feature_a": "emotion:deadpan", "feature_b": "sndclass:gasp", "mean": 0.35, "note": "Mismatch."},
    {"feature_a": "mech:understatement", "feature_b": "sndclass:silence", "mean": 0.70, "note": "Do not explain it."},
    {"feature_a": "mech:contrast", "feature_b": "sndclass:scratch", "mean": 0.65, "note": "Record scratch at the swerve."},
    {"feature_a": "mech:escalation", "feature_b": "sndclass:boom", "mean": 0.65, "note": "Peak of the build."},
    {"feature_a": "mech:self_own", "feature_b": "sndclass:trombone", "mean": 0.70, "note": "Fail sting."},
    {"feature_a": "mech:anti_joke", "feature_b": "sndclass:silence", "mean": 0.65, "note": "Nothing happens, on purpose."},
    {"feature_a": "emotion:deadpan", "feature_b": "mech:understatement", "mean": 0.70, "note": "Face and caption agree."},
    {"feature_a": "emotion:smug", "feature_b": "mech:contrast", "mean": 0.65, "note": "Smug against reality."},
    {"feature_a": "emotion:crying", "feature_b": "mech:relatable", "mean": 0.70, "note": "Shared pain."},
]


def seed_catalog(conn: sqlite3.Connection) -> tuple[int, int]:
    for axis in ("mechanism", "voice", "register", "structure"):
        if not values(axis):
            raise RuntimeError(f"vocab missing {axis}")
    for row in TEMPLATES:
        existing = conn.execute("SELECT id FROM templates WHERE key = ?", (row["key"],)).fetchone()
        if existing:
            template_id = int(existing["id"])
        else:
            cur = conn.execute(
                "INSERT INTO templates(key, patterns, render, requires, max_chars, max_lines) VALUES (?, ?, ?, ?, ?, ?)",
                (row["key"], json.dumps(row["patterns"]), row["render"], json.dumps(row["requires"]), row["max_chars"], row["max_lines"]),
            )
            template_id = int(cur.lastrowid)
        for axis in ("mechanism", "voice", "register", "structure"):
            conn.execute(
                "INSERT INTO template_tags(template_id, axis, value) VALUES (?, ?, ?) ON CONFLICT(template_id, axis) DO UPDATE SET value = excluded.value",
                (template_id, axis, row[axis]),
            )
    for row in PRIORS:
        found = conn.execute(
            "SELECT id FROM pair_priors WHERE feature_a = ? AND feature_b = ?",
            (row["feature_a"], row["feature_b"]),
        ).fetchone()
        if found:
            continue
        conn.execute(
            "INSERT INTO pair_priors(feature_a, feature_b, mean, strength, note) VALUES (?, ?, ?, 2.0, ?)",
            (row["feature_a"], row["feature_b"], row["mean"], row["note"]),
        )
    conn.commit()
    templates = int(conn.execute("SELECT COUNT(*) FROM templates WHERE active = 1").fetchone()[0])
    priors = int(conn.execute("SELECT COUNT(*) FROM pair_priors WHERE active = 1").fetchone()[0])
    return templates, priors
