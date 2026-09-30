"""Named sparse features. Section 6.2. No positional vectors."""

from __future__ import annotations

import math

from mag.composer import Caption, SoundPick
from mag.vocab import values

TIERS = {
    "bias": ("A", 0.1),
    "topic": ("A", 1.0),
    "emotion": ("A", 1.0),
    "intensity": ("A", 1.0),
    "family": ("A", 1.0),
    "mech": ("A", 1.0),
    "voice": ("A", 1.0),
    "reg": ("A", 1.0),
    "struct": ("A", 1.0),
    "len": ("A", 1.0),
    "profanity": ("A", 1.0),
    "sndclass": ("A", 1.0),
    "sndint": ("A", 1.0),
    "fit": ("A", 1.0),
    "fatigue": ("A", 1.0),
    "cross": ("B", 4.0),
    "id": ("C", 3.0),
}
LENGTHS = ("short", "mid", "long")
INTENSITY = {"low": 0, "mid": 1, "high": 2}


def registry_names() -> list[str]:
    names = ["bias"]
    names += [f"topic:{item}" for item in values("topic")]
    names += [f"emotion:{item}" for item in values("emotion")]
    names += [f"intensity:{item}" for item in values("intensity")]
    names += [f"family:{item}" for item in values("family")]
    names += [f"mech:{item}" for item in values("mechanism")]
    names += [f"voice:{item}" for item in values("voice")]
    names += [f"reg:{item}" for item in values("register")]
    names += [f"struct:{item}" for item in values("structure")]
    names += [f"len:{item}" for item in LENGTHS]
    names.append("profanity")
    names += [f"sndclass:{item}" for item in values("sound_class")]
    names += [f"sndint:{item}" for item in values("intensity")]
    names += ["topic_overlap", "emotion_fit", "int_match", "int_far"]
    names += ["fat_tpl", "fat_snd", "fat_mech", "img_seen"]
    for emotion in values("emotion"):
        for sound in values("sound_class"):
            names.append(f"emotion:{emotion}*sndclass:{sound}")
    for mech in values("mechanism"):
        for sound in values("sound_class"):
            names.append(f"mech:{mech}*sndclass:{sound}")
    for emotion in values("emotion"):
        for mech in values("mechanism"):
            names.append(f"emotion:{emotion}*mech:{mech}")
    return names


def _one(tags: dict[str, list[str]], axis: str, default: str = "") -> str:
    values_ = tags.get(axis) or []
    return values_[0] if values_ else default


def build_features(image_id, image_tags, caption, sound, sound_tags, *, recent_templates=None, recent_sounds=None, recent_mechs=None, image_seen=0):
    features = {"bias": 1.0}
    topics = image_tags.get("topic") or []
    scale = 1.0 / math.sqrt(len(topics)) if topics else 0.0
    for topic in topics:
        features[f"topic:{topic}"] = scale
    for axis, prefix in (("emotion", "emotion"), ("intensity", "intensity"), ("family", "family")):
        value = _one(image_tags, axis)
        if value:
            features[f"{prefix}:{value}"] = 1.0
    style = caption.style
    features[f"mech:{style['mechanism']}"] = 1.0
    features[f"voice:{style['voice']}"] = 1.0
    features[f"reg:{style['register']}"] = 1.0
    features[f"struct:{style['structure']}"] = 1.0
    features[f"len:{caption.length_bucket}"] = 1.0
    if caption.profanity:
        features["profanity"] = 1.0
    features[f"sndclass:{sound.sound_class}"] = 1.0
    if sound.sound_class != "silence":
        intensity = _one(sound_tags, "intensity")
        if intensity:
            features[f"sndint:{intensity}"] = 1.0
    sound_topics = set(sound_tags.get("topic") or [])
    features["topic_overlap"] = len(set(topics) & sound_topics) / max(len(topics), 1)
    fit = set(sound_tags.get("fit_emotion") or [])
    features["emotion_fit"] = 1.0 if _one(image_tags, "emotion") in fit else 0.0
    image_int = INTENSITY.get(_one(image_tags, "intensity"), 1)
    sound_int = INTENSITY.get(_one(sound_tags, "intensity"), image_int)
    features["int_match"] = 1.0 if image_int == sound_int else 0.0
    features["int_far"] = 1.0 if abs(image_int - sound_int) > 1 else 0.0
    recent_templates = recent_templates or []
    recent_sounds = recent_sounds or []
    recent_mechs = recent_mechs or []
    features["fat_tpl"] = recent_templates[-10:].count(caption.template_id) / 10
    features["fat_snd"] = 0.0 if sound.sound_id is None else recent_sounds[-10:].count(sound.sound_id) / 10
    features["fat_mech"] = recent_mechs[-5:].count(style["mechanism"]) / 5
    features["img_seen"] = min(image_seen, 3) / 3
    emotion = _one(image_tags, "emotion")
    if emotion:
        features[f"emotion:{emotion}*sndclass:{sound.sound_class}"] = 1.0
        features[f"emotion:{emotion}*mech:{style['mechanism']}"] = 1.0
    features[f"mech:{style['mechanism']}*sndclass:{sound.sound_class}"] = 1.0
    features[f"img:{image_id}"] = 1.0
    features[f"tpl:{caption.template_id}"] = 1.0
    features[f"snd:{sound.sound_id if sound.sound_id is not None else 0}"] = 1.0
    return features


def one_hot_sums(features: dict[str, float]) -> dict[str, float]:
    groups = {
        "emotion": [key for key in features if key.startswith("emotion:") and "*" not in key],
        "mech": [key for key in features if key.startswith("mech:") and "*" not in key],
        "sndclass": [key for key in features if key.startswith("sndclass:") and "*" not in key],
        "topic": [key for key in features if key.startswith("topic:")],
    }
    return {name: sum(features[key] for key in keys) for name, keys in groups.items()}
