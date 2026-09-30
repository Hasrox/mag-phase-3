"""Asset import. Copies files, hashes, probes sound with ffmpeg. Never calls the model."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from mag.config import Config
from mag.paths import ASSETS


class ImportRejected(ValueError):
    """A file failed a Phase 1 rule. No asset row is written."""


@dataclass
class ImportResult:
    added: int = 0
    reused: int = 0
    ids: list[int] = field(default_factory=list)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assets(assets_root: Path | None) -> Path:
    return assets_root or ASSETS


def _reject_quarantine(src: Path, assets: Path) -> None:
    quarantine = (assets / "quarantine").resolve()
    try:
        src.resolve().relative_to(quarantine)
    except ValueError:
        return
    raise ImportRejected("import does not see assets/quarantine")


def _rel(path: Path, assets: Path) -> str:
    return str(path.relative_to(assets.parent))


def import_image(
    conn: sqlite3.Connection,
    src: Path,
    license_note: str,
    *,
    source: str = "",
    is_gold: bool = False,
    assets_root: Path | None = None,
) -> int:
    note = license_note.strip()
    if not note:
        raise ImportRejected("license note is required")
    assets = _assets(assets_root)
    _reject_quarantine(src, assets)
    digest = sha256_file(src)
    existing = conn.execute("SELECT id FROM assets WHERE sha256 = ?", (digest,)).fetchone()
    if existing:
        return int(existing["id"])
    dest_dir = assets / "images"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{digest}{src.suffix.lower() or '.img'}"
    if not dest.exists():
        shutil.copy2(src, dest)
    try:
        with Image.open(dest) as img:
            img.verify()
        with Image.open(dest) as img:
            width, height = img.size
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise ImportRejected(f"not an image: {src.name}") from exc
    if width < 1 or height < 1:
        dest.unlink(missing_ok=True)
        raise ImportRejected("image has no pixels")
    state = "review" if is_gold else "new"
    cur = conn.execute(
        """INSERT INTO assets(kind, path, sha256, license_note, source, state, is_gold)
           VALUES ('image', ?, ?, ?, ?, ?, ?)""",
        (_rel(dest, assets), digest, note, source, state, int(is_gold)),
    )
    conn.execute(
        "INSERT INTO image_info(asset_id, width, height) VALUES (?, ?, ?)",
        (cur.lastrowid, width, height),
    )
    conn.commit()
    return int(cur.lastrowid)


def _probe_duration_ms(path: Path) -> int:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        text=True,
    )
    duration = float(json.loads(out)["format"]["duration"])
    return int(round(duration * 1000))


def _onset_ms(path: Path, threshold_db: float) -> int:
    """Leading silence only. A gap later in the file is not the onset."""
    proc = subprocess.run(
        ["ffmpeg", "-i", str(path), "-af", f"silencedetect=n={threshold_db}dB:d=0.02", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    start = None
    for line in proc.stderr.splitlines():
        if "silence_start:" in line and start is None:
            start = float(line.split("silence_start:")[1].split()[0])
        if "silence_end:" in line and start is not None and start < 0.05:
            return int(float(line.split("silence_end:")[1].split()[0]) * 1000)
    return 0


def _loudness(path: Path) -> float:
    proc = subprocess.run(
        ["ffmpeg", "-i", str(path), "-af", "loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    start = proc.stderr.rfind("{")
    end = proc.stderr.rfind("}")
    if start < 0 or end < 0:
        raise RuntimeError("loudnorm produced no measurement")
    payload = json.loads(proc.stderr[start : end + 1])
    return float(payload["input_i"])


def _ffmpeg_json(stderr: str) -> dict:
    start = stderr.rfind("{")
    end = stderr.rfind("}")
    if start < 0 or end < start:
        raise ImportRejected("ffmpeg produced no measurement")
    return json.loads(stderr[start : end + 1])


def import_sound(
    conn: sqlite3.Connection,
    manifest: dict,
    cfg: Config,
    *,
    assets_root: Path | None = None,
) -> int:
    note = str(manifest.get("license_note", "")).strip()
    if not note:
        raise ImportRejected("license note is required")
    assets = _assets(assets_root)
    src = Path(manifest["file"])
    _reject_quarantine(src, assets)
    if not src.is_file():
        raise ImportRejected(f"sound file missing: {src}")
    duration_ms = _probe_duration_ms(src)
    # accept_over_cap stores the whole file. Playback stops at play_cap_s.
    if not cfg.sound.accept_over_cap and duration_ms > cfg.sound.max_duration_s * 1000:
        raise ImportRejected(
            f"sound is {duration_ms} ms; hard limit is {cfg.sound.max_duration_s} s"
        )
    digest = sha256_file(src)
    existing = conn.execute("SELECT id FROM assets WHERE sha256 = ?", (digest,)).fetchone()
    if existing:
        return int(existing["id"])
    dest = assets / "sounds" / f"{digest}{src.suffix.lower() or '.wav'}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(src, dest)
    original = _rel(dest, assets)
    # Loudnorm was rejected by the operator on 2026-09-30. The file is stored
    # whole and served as-is. Duration, onset and loudness are measurements for
    # reporting only; none of them is a gate and none of them re-encodes audio.
    onset = _onset_ms(dest, cfg.sound.onset_db)
    try:
        loudness = _loudness(dest)
    except Exception:
        loudness = None
    loud_warn = (
        loudness is not None
        and abs(loudness - cfg.sound.target_lufs) > cfg.sound.lufs_tolerance
    )
    if loud_warn:
        print(
            f"note {src.name}: measured loudness {loudness:.2f} LUFS is outside "
            f"{cfg.sound.lufs_tolerance} LU of {cfg.sound.target_lufs}. "
            f"Measurement only; the file is served unaltered."
        )
    cur = conn.execute(
        """INSERT INTO assets(kind, path, sha256, license_note, source, state, is_gold)
           VALUES ('sound', ?, ?, ?, ?, 'active', 0)""",
        (original, digest, note, manifest.get("source", "")),
    )
    asset_id = int(cur.lastrowid)
    conn.execute(
        """INSERT INTO sound_info(
               asset_id, duration_ms, onset_ms, loudness_lufs, norm_path, original_path, warn_long
           ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            asset_id,
            duration_ms,
            onset,
            loudness,
            # norm_path is legacy and stays NOT NULL. It holds the original path
            # so it can never point at a wav the play loop stopped serving.
            original,
            original,
            int(duration_ms > cfg.sound.warn_duration_s * 1000 or loud_warn),
        ),
    )
    _write_tag(conn, asset_id, "sound_class", manifest["class"], "import")
    _write_tag(conn, asset_id, "intensity", manifest["intensity"], "import")
    for topic in manifest.get("topics", []):
        _write_tag(conn, asset_id, "topic", topic, "import")
    for emotion in manifest.get("fit_emotion", []):
        _write_tag(conn, asset_id, "fit_emotion", emotion, "import")
    conn.commit()
    return asset_id


def _write_tag(conn, asset_id: int, axis: str, value: str, source: str, run_id: int | None = None) -> None:
    known = conn.execute(
        "SELECT 1 FROM vocab WHERE axis = ? AND value = ? AND active = 1",
        (axis, value),
    ).fetchone()
    if known is None:
        raise ValueError(f"unknown vocab {axis}:{value}")
    human = conn.execute(
        """SELECT 1 FROM tags
           WHERE asset_id = ? AND axis = ? AND source = 'human' AND superseded = 0""",
        (asset_id, axis),
    ).fetchone()
    if human and source != "human":
        return
    if axis != "topic" and axis != "fit_emotion":
        conn.execute(
            "UPDATE tags SET superseded = 1 WHERE asset_id = ? AND axis = ? AND superseded = 0",
            (asset_id, axis),
        )
    conn.execute(
        "INSERT INTO tags(asset_id, axis, value, source, run_id) VALUES (?, ?, ?, ?, ?)",
        (asset_id, axis, value, source, run_id),
    )


def write_human_tags(conn: sqlite3.Connection, asset_id: int, tags: dict, info: dict | None = None) -> None:
    for axis, value in tags.items():
        values = value if isinstance(value, list) else [value]
        if axis in ("topic", "fit_emotion"):
            conn.execute(
                "UPDATE tags SET superseded = 1 WHERE asset_id = ? AND axis = ? AND superseded = 0",
                (asset_id, axis),
            )
        for item in values:
            _write_tag(conn, asset_id, axis, item, "human")
    if info:
        conn.execute(
            """UPDATE image_info
               SET subject = ?, action = ?, subject_plural = ?, text_in_image = ?, profanity_in_image = ?
               WHERE asset_id = ?""",
            (
                info.get("subject"),
                info.get("action"),
                int(info.get("subject_plural", False)),
                int(info.get("text_in_image", False)),
                int(info.get("profanity_in_image", False)),
                asset_id,
            ),
        )
    safety = tags.get("safety")
    gold = conn.execute("SELECT is_gold FROM assets WHERE id = ?", (asset_id,)).fetchone()
    if gold and int(gold["is_gold"]):
        state = "review"
    else:
        state = "active" if safety == "ok" else "review"
    conn.execute("UPDATE assets SET state = ? WHERE id = ?", (state, asset_id))
    conn.commit()


def _asset_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0])


def import_batch(
    conn: sqlite3.Connection,
    images: list[dict],
    sounds: list[dict],
    cfg: Config,
    *,
    assets_root: Path | None = None,
) -> ImportResult:
    """Idempotent store load. A second call with the same bytes adds zero rows."""
    before = _asset_count(conn)
    result = ImportResult()
    for row in images:
        asset_id = import_image(
            conn,
            Path(row["file"]),
            str(row.get("license_note", "")),
            source=str(row.get("source", "")),
            is_gold=bool(row.get("is_gold", False)),
            assets_root=assets_root,
        )
        result.ids.append(asset_id)
        if row.get("tags") or row.get("info"):
            write_human_tags(conn, asset_id, row.get("tags") or {}, row.get("info"))
    for row in sounds:
        asset_id = import_sound(conn, row, cfg, assets_root=assets_root)
        result.ids.append(asset_id)
    added = _asset_count(conn) - before
    result.added = added
    result.reused = len(result.ids) - added
    return result


def import_gold(
    conn: sqlite3.Connection,
    manifest_path: Path,
    cfg: Config,
    *,
    assets_root: Path | None = None,
) -> ImportResult:
    """Hand-tagged gold. Flagged is_gold. Never enters pool_images."""
    payload = json.loads(manifest_path.read_text())
    rows = []
    base = manifest_path.parent
    for row in payload:
        src = Path(row["file"])
        if not src.is_absolute():
            src = base / src
        copied = dict(row)
        copied["file"] = str(src)
        copied["is_gold"] = True
        rows.append(copied)
    return import_batch(conn, rows, [], cfg, assets_root=assets_root)


def ensure_layout(root: Path) -> None:
    """Section 5.1. Quarantine exists and is invisible to import.

    assets/sounds/norm is deliberately absent: loudnorm was rejected by the
    operator, so import never writes a derived audio copy.
    """
    for rel in (
        "assets/images",
        "assets/sounds",
        "assets/fonts",
        "assets/render_cache",
        "assets/quarantine",
        "data/gold",
        "data/profiles",
        "models",
        "runtime/gates",
    ):
        (root / rel).mkdir(parents=True, exist_ok=True)
