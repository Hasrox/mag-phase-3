"""One MAG_ROOT for the whole test process.

mag.paths.ROOT is read from MAG_ROOT at import time, and every stored asset path
is relative to it. A fixture store therefore has to BE the MAG_ROOT, but the
first mag import in the process wins. Every test module imports this helper
FIRST so they all agree on one root, then uses its own subdirectory.

    import _scratch_root  # noqa: F401  (must precede any mag import)
    SCRATCH = _scratch_root.mag_root() / "phase4"
"""

from __future__ import annotations

import atexit
import os
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

_ROOT = PROJECT / "data" / "test_scratch"
shutil.rmtree(_ROOT, ignore_errors=True)
_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["MAG_ROOT"] = str(_ROOT)


@atexit.register
def _cleanup() -> None:
    shutil.rmtree(_ROOT, ignore_errors=True)


def mag_root() -> Path:
    return _ROOT


def real_config(name: str) -> Path:
    """config/mag.toml and friends live in the real project, not the scratch root."""
    return PROJECT / "config" / name
