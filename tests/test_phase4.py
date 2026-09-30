"""Phase 4 exit rules. No model, no GPU, no network.

Proves the two fixes the operator was told about: the play loop serves the
original stored sound file rather than a loudnorm copy, and it does not return
one file every other draw.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])

# Must precede every mag import: it fixes MAG_ROOT for the whole process.
import _scratch_root  # noqa: E402

from PIL import Image  # noqa: E402

from mag.audit import run_audit  # noqa: E402
from mag.catalog import seed_catalog  # noqa: E402
from mag.config import load_config  # noqa: E402
from mag.db import connect, init_db  # noqa: E402
from mag.importing import (  # noqa: E402
    ensure_layout, import_image, import_sound, write_human_tags,
)
from mag.render import cache_path  # noqa: E402
from mag.serve import (  # noqa: E402
    _sound_file_exists,
    _sound_path,
    active_profile,
    candidates_for,
    open_session,
    recent_sound_ids,
    score_and_pick,
    status_line,
)
from mag.ui import _sound_note  # noqa: E402

IMAGES = 25
SOUNDS = 6
NOTE = "operator fixture"
# This store must BE the MAG_ROOT: mag.serve resolves assets.path against
# mag.paths.ROOT, so a nested directory would resolve to the wrong place.
SCRATCH = _scratch_root.mag_root()
REAL_CONFIG = _scratch_root.real_config("mag.toml")
REAL_VOCAB = _scratch_root.real_config("vocab.json")
REAL_AUDIT = Path(__file__).resolve().parents[1] / "scripts" / "audit.sql"


def _png(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (900, 700)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _tone(path: Path, seconds: float, freq: int = 440) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"sine=frequency={freq}:duration={seconds}",
            "-c:a", "libmp3lame", str(path),
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _store(root: Path):
    """A throwaway store with the same placeholder tags the inbox writes."""
    assets = root / "assets"
    ensure_layout(root)
    conn = connect(root / "data" / "mag.sqlite")
    init_db(conn, REAL_VOCAB)
    cfg = load_config(REAL_CONFIG)
    seed_catalog(conn)
    src = assets / "inbox" / "images"
    for index in range(IMAGES):
        path = src / f"thing_{index}.png"
        _png(path, (20 + index * 3, 60, 120))
        asset_id = import_image(conn, path, NOTE, assets_root=assets)
        write_human_tags(conn, asset_id, {
            "emotion": "deadpan",
            "intensity": "mid",
            "family": "other",
            "caption_zone": "both",
            "safety": "ok",
            "topic": ["everyday"],
        }, {"subject": f"thing {index}", "action": "waiting"})
    for index in range(SOUNDS):
        path = assets / "inbox" / "sounds" / f"tone_{index}.mp3"
        _tone(path, 0.4 + index * 0.05, 220 + index * 60)
        import_sound(conn, {
            "file": str(path),
            "license_note": NOTE,
            "class": "sting",
            "intensity": "mid",
            "source": "fixture",
        }, cfg, assets_root=assets)
    return conn, cfg


def _session(conn, cfg) -> tuple[int, int]:
    profile = active_profile(conn, cfg)
    return int(profile["id"]), int(open_session(conn, int(profile["id"]), None)["id"])


def _resolve(relative: str) -> str:
    path = Path(relative)
    return str(path if path.is_absolute() else SCRATCH / path)


class Phase4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # phase1 may have left a store here; the phase4 store owns the root.
        for name in ("assets", "data", "config", "models", "runtime"):
            shutil.rmtree(SCRATCH / name, ignore_errors=True)
        ensure_layout(SCRATCH)
        cls.root = SCRATCH
        cls.conn, cls.cfg = _store(cls.root)
        cls.cache = SCRATCH / "assets" / "render_cache"
        cls.cache.mkdir(parents=True, exist_ok=True)
        cls.profile_id, cls.session_id = _session(cls.conn, cls.cfg)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.conn.close()

    def _serve(self, count: int) -> list[int | None]:
        served: list[int | None] = []
        for _ in range(count):
            result = score_and_pick(
                self.conn, self.cfg, topic=None, profile_id=self.profile_id,
                session_id=self.session_id, cache_dir=self.cache,
            )
            if result.impression_id is not None:
                served.append(result.sound_id)
        return served

    def test_serves_the_original_file_not_a_normalized_copy(self) -> None:
        rows = self.conn.execute(
            """SELECT a.id, a.path, s.original_path
               FROM assets a JOIN sound_info s ON s.asset_id = a.id
               WHERE a.kind = 'sound' ORDER BY a.id"""
        ).fetchall()
        self.assertTrue(rows)
        for row in rows:
            served = _sound_path(self.conn, int(row["id"]))
            self.assertTrue(served.endswith(".mp3"), served)
            parts = served.replace("\\", "/").lower().split("/")
            self.assertNotIn("norm", parts[-2], served)
            self.assertTrue(Path(served).is_file(), served)
            self.assertEqual(served, _resolve(row["path"]))
        self.assertEqual(int(self.conn.execute(
            "SELECT COUNT(*) FROM sound_info WHERE norm_path LIKE '%norm%'"
        ).fetchone()[0]), 0)

    def test_silence_is_not_a_missing_file(self) -> None:
        self.assertEqual(_sound_path(self.conn, None), "")
        self.assertTrue(_sound_file_exists(self.conn, None))
        self.assertIn("silence", _sound_note(""))
        sound_id = int(self.conn.execute(
            "SELECT id FROM assets WHERE kind = 'sound' ORDER BY id LIMIT 1"
        ).fetchone()[0])
        self.assertIn("audio", _sound_note(_sound_path(self.conn, sound_id)))

    def test_no_sound_repeats_inside_the_repeat_window(self) -> None:
        """The half that was missing on main: recent_sounds was never passed."""
        served = self._serve(14)
        self.assertGreaterEqual(len(served), 10, served)
        non_silent = [item for item in served if item is not None]
        self.assertGreaterEqual(len(non_silent), 6, "expected real sound draws too")
        for index in range(len(non_silent) - 1):
            self.assertNotEqual(non_silent[index], non_silent[index + 1], "consecutive repeat")
        window = max(self.cfg.pool.sound_repeat_window, 1)
        for index in range(len(non_silent) - window):
            block = non_silent[index:index + window + 1]
            self.assertGreater(len(set(block)), 1, f"one sound filled a window: {block}")
        self.assertGreaterEqual(len(set(non_silent)), SOUNDS - 1, non_silent)

    def test_recent_sound_ids_reads_the_log(self) -> None:
        rows = self.conn.execute("SELECT id, sound_id FROM impressions ORDER BY id").fetchall()
        self.assertTrue(rows)
        self.assertEqual(
            recent_sound_ids(rows, 3),
            [int(row["sound_id"]) for row in rows[-3:] if row["sound_id"]],
        )
        self.assertEqual(recent_sound_ids([], 8), [])

    def test_render_is_cached_by_composition(self) -> None:
        result = score_and_pick(
            self.conn, self.cfg, topic=None, profile_id=self.profile_id,
            session_id=self.session_id, cache_dir=self.cache,
        )
        self.assertIsNotNone(result.impression_id, result.reason)
        self.assertTrue(result.render_path.endswith(".png"), result.render_path)
        row = self.conn.execute(
            "SELECT a.sha256, c.text, t.render FROM impressions i"
            " JOIN assets a ON a.id = i.image_id"
            " JOIN captions c ON c.image_id = i.image_id AND c.template_id = i.template_id"
            "   AND c.variant = i.variant"
            " JOIN templates t ON t.id = i.template_id WHERE i.id = ?",
            (result.impression_id,),
        ).fetchone()
        expected = cache_path(self.cache, row["sha256"], row["text"], row["render"])
        self.assertEqual(Path(result.render_path).name, expected.name)
        stamp = expected.stat().st_mtime_ns
        self._serve(1)
        self.assertEqual(expected.stat().st_mtime_ns, stamp, "cache entry was rewritten")

    def test_status_line_does_not_leak_policy(self) -> None:
        line = status_line("animals", 7, 42)
        for word in ("policy", "holdout", "bandit", "coverage"):
            self.assertNotIn(word, line)

    def test_audit_is_clean_after_serves(self) -> None:
        self.assertEqual(run_audit(self.conn, REAL_AUDIT), [])

    def test_flagged_image_leaves_the_pool(self) -> None:
        def pool() -> int:
            # index 2 is the raw pool size, before cooldown.
            return candidates_for(
                self.conn, self.cfg, topic=None,
                profile_id=self.profile_id, rng=random.Random(0),
            )[2]
        before = pool()
        # Serve here rather than relying on another test having run first.
        self._serve(1)
        image_id = int(self.conn.execute(
            "SELECT image_id FROM impressions WHERE image_id IS NOT NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()[0])
        self.conn.execute(
            "INSERT INTO flags(impression_id, kind, a_id, b_id)"
            " SELECT id, 'bad_tags', ?, NULL FROM impressions WHERE image_id = ? ORDER BY id DESC LIMIT 1",
            (image_id, image_id),
        )
        self.conn.commit()
        self.assertEqual(pool(), before - 1)


if __name__ == "__main__":
    unittest.main()

