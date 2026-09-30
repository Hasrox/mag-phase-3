"""Project roots. The package lives next to config/, data/, and assets/."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("MAG_ROOT", Path(__file__).resolve().parents[1]))
CONFIG = ROOT / "config" / "mag.toml"
ASSETS = ROOT / "assets"
DATA = ROOT / "data"
DB_PATH = DATA / "mag.sqlite"
MODELS = ROOT / "models"
RUNTIME = ROOT / "runtime"
GATES = RUNTIME / "gates"
PROFILES = DATA / "profiles"
RENDER_CACHE = ASSETS / "render_cache"
GOLD = DATA / "gold"
