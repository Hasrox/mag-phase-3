"""Controlled vocabularies. Feature names, the grammar, and validators all read this."""

from __future__ import annotations

VOCAB: list[tuple[str, str, str]] = [
    ("topic", "gaming", "Content domain."),
    ("topic", "toilet", "Content domain."),
    ("topic", "everyday", "Content domain."),
    ("topic", "food", "Content domain."),
    ("topic", "work", "Content domain."),
    ("topic", "animals", "Content domain."),
    ("topic", "internet", "Content domain. Operator may substitute."),
    ("topic", "relationships", "Content domain. Operator may substitute."),
    ("emotion", "deadpan", "Affect of the main subject."),
    ("emotion", "angry", "Affect of the main subject."),
    ("emotion", "smug", "Affect of the main subject."),
    ("emotion", "crying", "Affect of the main subject."),
    ("emotion", "shocked", "Affect of the main subject."),
    ("emotion", "none", "Affect of the main subject."),
    ("intensity", "low", "Energy of the image, or loudness class of the sound."),
    ("intensity", "mid", "Energy of the image, or loudness class of the sound."),
    ("intensity", "high", "Energy of the image, or loudness class of the sound."),
    ("family", "reaction", "Layout the image supports."),
    ("family", "object_label", "Layout the image supports."),
    ("family", "comparison", "Layout the image supports."),
    ("family", "sign", "Layout the image supports."),
    ("family", "other", "Layout the image supports."),
    ("setting", "indoor", "Template legality only. Not a feature."),
    ("setting", "outdoor", "Template legality only. Not a feature."),
    ("setting", "screen", "Template legality only. Not a feature."),
    ("setting", "abstract", "Template legality only. Not a feature."),
    ("setting", "unknown", "Template legality only. Not a feature."),
    ("subject_kind", "person", "Agreement and pronouns in slots."),
    ("subject_kind", "animal", "Agreement and pronouns in slots."),
    ("subject_kind", "object", "Agreement and pronouns in slots."),
    ("subject_kind", "character", "Agreement and pronouns in slots."),
    ("subject_kind", "group", "Agreement and pronouns in slots."),
    ("caption_zone", "top", "Free area for impact-style text."),
    ("caption_zone", "bottom", "Free area for impact-style text."),
    ("caption_zone", "both", "Free area for impact-style text."),
    ("caption_zone", "none", "Free area for impact-style text."),
    ("safety", "ok", "May enter the pool."),
    ("safety", "review", "Held until the operator resolves it."),
    ("safety", "reject", "Never served."),
    ("sound_class", "sting", "Short musical punctuation."),
    ("sound_class", "trombone", "Fail sting."),
    ("sound_class", "scratch", "Record scratch."),
    ("sound_class", "gasp", "Intake of breath."),
    ("sound_class", "boom", "Impact."),
    ("sound_class", "other", "Anything else with a file."),
    ("sound_class", "silence", "Virtual. No file. Candidate for every caption."),
    ("fit_emotion", "deadpan", "Emotion this sound suits. Operator-set."),
    ("fit_emotion", "angry", "Emotion this sound suits. Operator-set."),
    ("fit_emotion", "smug", "Emotion this sound suits. Operator-set."),
    ("fit_emotion", "crying", "Emotion this sound suits. Operator-set."),
    ("fit_emotion", "shocked", "Emotion this sound suits. Operator-set."),
    ("fit_emotion", "none", "Emotion this sound suits. Operator-set."),
    ("mechanism", "label", "How the joke works."),
    ("mechanism", "contrast", "How the joke works."),
    ("mechanism", "understatement", "How the joke works."),
    ("mechanism", "escalation", "How the joke works."),
    ("mechanism", "relatable", "How the joke works."),
    ("mechanism", "absurd", "How the joke works."),
    ("mechanism", "self_own", "How the joke works."),
    ("mechanism", "pov", "How the joke works."),
    ("mechanism", "anti_joke", "How the joke works."),
    ("voice", "first_person", "Who is speaking."),
    ("voice", "second_person", "Who is speaking."),
    ("voice", "third_person", "Who is speaking."),
    ("voice", "imperative", "Who is speaking."),
    ("register", "dry", "Tone of delivery."),
    ("register", "chaotic", "Tone of delivery."),
    ("register", "sincere", "Tone of delivery."),
    ("register", "mock_formal", "Tone of delivery."),
    ("structure", "single_line", "Shape of the text."),
    ("structure", "two_part", "Shape of the text."),
    ("structure", "label_pair", "Shape of the text."),
    ("structure", "top_bottom", "Shape of the text."),
    ("structure", "nobody_me", "Shape of the text."),
]

SINGLE_IMAGE_AXES = (
    "emotion",
    "intensity",
    "family",
    "setting",
    "caption_zone",
    "subject_kind",
    "safety",
)
MULTI_AXES = ("topic", "fit_emotion")

FEATURE_AXES = {
    "topic": "topic",
    "emotion": "emotion",
    "intensity": "intensity",
    "family": "family",
    "mechanism": "mech",
    "voice": "voice",
    "register": "reg",
    "structure": "struct",
    "sound_class": "sndclass",
}


def values(axis: str, active_only: bool = True) -> list[str]:
    return [v for a, v, _d in VOCAB if a == axis]


def write_vocab_file(path) -> None:
    """The table is loaded from this file. The constant above is the 1.1 seed."""
    import json
    from pathlib import Path

    payload = [
        {"axis": axis, "value": value, "description": description, "since_version": "1.1"}
        for axis, value, description in VOCAB
    ]
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def load_vocab_file(path) -> list[tuple[str, str, str]]:
    import json
    from pathlib import Path

    file = Path(path)
    if not file.exists():
        write_vocab_file(file)
    rows = json.loads(file.read_text())
    return [(row["axis"], row["value"], row["description"]) for row in rows]
