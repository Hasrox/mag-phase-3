"""One file log for the play loop. data/mag.log, plus the console."""

from __future__ import annotations

import logging
from pathlib import Path

from mag.paths import DATA

_READY = False


def get_log() -> logging.Logger:
    global _READY
    log = logging.getLogger("mag")
    if _READY:
        return log
    DATA.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(DATA / "mag.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    log.addHandler(file_handler)
    log.addHandler(stream)
    _READY = True
    return log
