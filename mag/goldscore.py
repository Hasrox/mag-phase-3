"""S-4 and S-5. Thresholds are the model-card values fixed before the first run.

Changing a threshold after seeing scores voids the run. This module does not
read a live config override for that reason.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from mag.tagger import TagPayload, is_refusal

# Section 5.3, copied into models/model-card.txt before any tag run.
THRESHOLDS = {
    "emotion_accuracy": 0.70,
    "family_accuracy": 0.80,
    "intensity_within_one": 0.85,
    "topics_micro_f1": 0.70,
    "subject_action_usable": 0.80,
    "missed_rejects": 0,
    "false_reject_on_ok": 0.10,
    "determinism": 0.95,
    "parse_rate": 1.0,
    "unconstrained_refusals": 0,
}
INTENSITY = {"low": 0, "mid": 1, "high": 2}
FIELDS = (
    "subject", "action", "subject_kind", "subject_plural", "emotion", "intensity",
    "family", "setting", "caption_zone", "topics", "text_in_image",
    "profanity_in_image", "safety",
)


class GoldLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    emotion: str
    family: str
    intensity: str
    topics: list[str]
    safety: str
    subject_usable: bool
    action_usable: bool


class GoldScores(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int
    parse_rate: float
    emotion_accuracy: float
    family_accuracy: float
    intensity_within_one: float
    topics_micro_f1: float
    subject_action_usable: float
    missed_rejects: int
    false_reject_on_ok: float
    determinism: float | None
    unconstrained_refusals: int
    s4_pass: bool
    s5_pass: bool

    @property
    def accepted(self) -> bool:
        return self.s4_pass and self.s5_pass


def _micro_f1(pairs: list[tuple[set[str], set[str]]]) -> float:
    tp = fp = fn = 0
    for gold, pred in pairs:
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def field_match(left: TagPayload, right: TagPayload) -> tuple[int, int]:
    same = 0
    for name in FIELDS:
        a = getattr(left, name)
        b = getattr(right, name)
        if name == "topics":
            a, b = sorted(a), sorted(b)
        same += int(a == b)
    return same, len(FIELDS)


def score_gold(
    rows: list[tuple[GoldLabel, TagPayload | None]],
    *,
    second_run: list[TagPayload | None] | None = None,
    unconstrained: list[str] | None = None,
) -> GoldScores:
    """rows are (hand label, grammar-mode parse). None means the response did not parse."""
    n = len(rows)
    if n == 0:
        raise ValueError("gold score needs the hand-tagged set")
    parsed = [(label, pred) for label, pred in rows if pred is not None]
    parse_rate = len(parsed) / n
    emotion = sum(label.emotion == pred.emotion for label, pred in parsed) / n
    family = sum(label.family == pred.family for label, pred in parsed) / n
    intensity = sum(
        abs(INTENSITY[label.intensity] - INTENSITY[pred.intensity]) <= 1
        for label, pred in parsed
    ) / n
    topics = _micro_f1([(set(label.topics), set(pred.topics)) for label, pred in parsed])
    usable_hits = sum(int(label.subject_usable) + int(label.action_usable) for label, _ in rows)
    usable = usable_hits / (2 * n)
    missed = sum(
        1 for label, pred in rows
        if label.safety == "reject" and (pred is None or pred.safety != "reject")
    )
    ok = [pred for label, pred in rows if label.safety == "ok"]
    false_reject = (
        sum(1 for pred in ok if pred is not None and pred.safety == "reject") / len(ok)
        if ok else 0.0
    )
    determinism: float | None = None
    if second_run is not None:
        if len(second_run) != n:
            raise ValueError("second gold run length does not match")
        same = total = 0
        for (_, first), second in zip(rows, second_run):
            if first is None or second is None:
                total += len(FIELDS)
                continue
            hit, count = field_match(first, second)
            same += hit
            total += count
        determinism = same / total if total else 0.0
    refusals = sum(1 for text in (unconstrained or []) if is_refusal(text))
    s4 = (
        parse_rate >= THRESHOLDS["parse_rate"]
        and emotion >= THRESHOLDS["emotion_accuracy"]
        and family >= THRESHOLDS["family_accuracy"]
        and intensity >= THRESHOLDS["intensity_within_one"]
        and topics >= THRESHOLDS["topics_micro_f1"]
        and usable >= THRESHOLDS["subject_action_usable"]
        and missed == THRESHOLDS["missed_rejects"]
        and false_reject <= THRESHOLDS["false_reject_on_ok"]
        and determinism is not None
        and determinism >= THRESHOLDS["determinism"]
    )
    s5 = refusals == 0 and false_reject <= THRESHOLDS["false_reject_on_ok"] and unconstrained is not None
    return GoldScores(
        n=n,
        parse_rate=parse_rate,
        emotion_accuracy=emotion,
        family_accuracy=family,
        intensity_within_one=intensity,
        topics_micro_f1=topics,
        subject_action_usable=usable,
        missed_rejects=missed,
        false_reject_on_ok=false_reject,
        determinism=determinism,
        unconstrained_refusals=refusals,
        s4_pass=s4,
        s5_pass=s5,
    )
