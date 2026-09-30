"""Drop-folder import. No manifest.

Put images and GIFs in assets/inbox/images, sounds in assets/inbox/sounds.
Files already sitting under assets/ are picked up too. A file whose bytes are
already in the store is left alone, including its tags.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from mag.config import Config
from mag.importing import ImportRejected, import_image, import_sound, write_human_tags
from mag.paths import ASSETS

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
SOUND_EXT = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}
SKIP_PARTS = {"quarantine", "render_cache", "fonts", "norm"}
LICENSE = "local collection"


@dataclass
class AbsorbResult:
    added: int = 0
    skipped: int = 0
    rejected: list[str] = field(default_factory=list)

    def line(self) -> str:
        tail = f" rejected={len(self.rejected)}" if self.rejected else ""
        return f"media added={self.added} already_there={self.skipped}{tail}"


def ensure_inbox(assets: Path | None = None) -> tuple[Path, Path]:
    root = assets or ASSETS
    images = root / "inbox" / "images"
    sounds = root / "inbox" / "sounds"
    images.mkdir(parents=True, exist_ok=True)
    sounds.mkdir(parents=True, exist_ok=True)
    return images, sounds


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _known(conn: sqlite3.Connection, digest: str) -> bool:
    row = conn.execute("SELECT 1 FROM assets WHERE sha256 = ?", (digest,)).fetchone()
    return row is not None


def _skip(path: Path) -> bool:
    return bool(SKIP_PARTS.intersection(path.parts))


def _subject(path: Path) -> str:
    stem = path.stem.replace("_", " ").replace("-", " ").strip()
    return (stem[:40] or "thing")


def _sound_class(path: Path) -> str:
    for part in path.parts:
        if part in {"sting", "trombone", "scratch", "gasp", "boom", "other"}:
            return part
    return "other"


def _image_row(path: Path) -> dict:
    subject = _subject(path)
    return {
        "license_note": LICENSE,
        "info": {"subject": subject, "action": "waiting"},
        "tags": {
            "emotion": "deadpan",
            "intensity": "mid",
            "family": "other",
            "caption_zone": "both",
            "safety": "ok",
            "topic": ["everyday"],
        },
    }


def absorb(conn: sqlite3.Connection, cfg: Config, assets: Path | None = None) -> AbsorbResult:
    """Import new images, GIFs, and sounds. Does not retag files already stored."""
    root = assets or ASSETS
    ensure_inbox(root)
    result = AbsorbResult()
    if not root.exists():
        return result
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _skip(path):
            continue
        suffix = path.suffix.lower()
        if suffix not in IMAGE_EXT and suffix not in SOUND_EXT:
            continue
        try:
            digest = _sha(path)
        except OSError as exc:
            result.rejected.append(f"{path.name}: {exc}")
            continue
        if _known(conn, digest):
            result.skipped += 1
            continue
        try:
            if suffix in IMAGE_EXT:
                row = _image_row(path)
                asset_id = import_image(conn, path, row["license_note"], source="inbox")
                write_human_tags(conn, asset_id, row["tags"], row["info"])
            else:
                import_sound(
                    conn,
                    {
                        "file": str(path),
                        "license_note": LICENSE,
                        "class": _sound_class(path),
                        "intensity": "mid",
                        "source": "inbox",
                    },
                    cfg,
                )
        except (ImportRejected, ValueError, OSError, FileNotFoundError) as exc:
            result.rejected.append(f"{path.name}: {exc}")
            continue
        result.added += 1
    return result
