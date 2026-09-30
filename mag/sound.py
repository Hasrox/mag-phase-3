"""Sound policy. Operator override: accept the file, cap playback.

The stored file is never trimmed. A player reads the first play_cap_s
seconds and stops. Length is not an import rejection.
"""

from __future__ import annotations

from mag.config import Config, SoundCfg


def play_window(duration_s: float, sound: SoundCfg) -> tuple[float, float]:
    """Return [start, end] in seconds. End is the cap, not a cut of the file."""
    if duration_s < 0:
        raise ValueError("duration_s must be >= 0")
    cap = sound.play_cap_s if sound.accept_over_cap else sound.max_duration_s
    return 0.0, min(duration_s, cap)


def import_decision(duration_s: float, cfg: Config) -> str:
    """accept, or warn. Never reject on length while accept_over_cap is set."""
    sound = cfg.sound
    if sound.accept_over_cap:
        if duration_s > sound.warn_duration_s:
            return "warn"
        return "accept"
    if duration_s > sound.max_duration_s:
        return "reject"
    if duration_s > sound.warn_duration_s:
        return "warn"
    return "accept"
