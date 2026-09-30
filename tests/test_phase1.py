"""Phase 1 exit rules. No model, no GPU."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Must precede every mag import: it fixes MAG_ROOT for the whole process.
import _scratch_root  # noqa: E402

from PIL import Image  # noqa: E402

from mag.audit import run_audit  # noqa: E402
from mag.config import load_config  # noqa: E402
from mag.db import connect, init_db  # noqa: E402
from mag.importing import (  # noqa: E402
    ImportRejected, ensure_layout, import_batch, import_image, import_sound,
)
from mag.phase1 import probes_cover_topics  # noqa: E402


def _tone(path: Path, seconds: float, volume_db: float = -28.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"sine=frequency=440:duration={seconds},volume={volume_db}dB",
            "-c:a", "pcm_s16le", str(path),
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color).save(path)


class Phase1Tests(unittest.TestCase):
    def setUp(self) -> None:
        # Under the shared MAG_ROOT so nothing resolves outside it.
        self.root = _scratch_root.mag_root()
        shutil.rmtree(self.root, ignore_errors=True)
        self.assets = self.root / "assets"
        ensure_layout(self.root)
        self.conn = connect(self.root / "data" / "mag.sqlite")
        init_db(self.conn, _scratch_root.real_config("vocab.json"))
        self.cfg = load_config(_scratch_root.real_config("mag.toml"))

    def tearDown(self) -> None:
        self.conn.close()
        for name in ("assets", "data", "config", "models", "runtime"):
            shutil.rmtree(self.root / name, ignore_errors=True)

    def test_exit_rules(self) -> None:
        for name in ("assets", "data"):
            shutil.rmtree(self.root / name, ignore_errors=True)
        image = self.root / "src" / "cat.png"
        gold = self.root / "src" / "gold.png"
        short = self.root / "src" / "short.wav"
        long_ok = self.root / "src" / "sting.wav"
        too_long = self.root / "src" / "over.wav"
        _png(image, (12, 40, 80))
        _png(gold, (80, 12, 12))
        _tone(short, 0.4)
        _tone(long_ok, 1.2, -30.0)
        _tone(too_long, 5.2)
        note = "operator fixture, not a seed license"
        first = import_batch(
            self.conn,
            [
                {"file": image, "license_note": note, "source": "fixture"},
                {
                    "file": gold,
                    "license_note": note,
                    "is_gold": True,
                    "tags": {"safety": "ok", "topic": ["animals"], "emotion": "deadpan"},
                    "info": {"subject": "cat", "action": "staring"},
                },
            ],
            [
                {
                    "file": short,
                    "license_note": note,
                    "class": "sting",
                    "intensity": "low",
                    "topics": ["animals"],
                    "fit_emotion": ["deadpan"],
                },
                {
                    "file": long_ok,
                    "license_note": note,
                    "class": "trombone",
                    "intensity": "mid",
                    "fit_emotion": ["smug"],
                },
            ],
            self.cfg,
            assets_root=self.assets,
        )
        self.assertEqual(first.added, 4)
        second = import_batch(
            self.conn,
            [{"file": image, "license_note": note}, {"file": gold, "license_note": note, "is_gold": True}],
            [{"file": short, "license_note": note, "class": "sting", "intensity": "low"}],
            self.cfg,
            assets_root=self.assets,
        )
        self.assertEqual(second.added, 0)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0], 4)

        with self.assertRaises(ImportRejected):
            import_image(self.conn, image, "  ", assets_root=self.assets)
        hidden = self.assets / "quarantine" / "nope.png"
        _png(hidden, (1, 1, 1))
        with self.assertRaises(ImportRejected):
            import_image(self.conn, hidden, note, assets_root=self.assets)

        # accept_over_cap is true by operator override: an over-cap sound is
        # stored whole, never rejected on length, and never re-encoded.
        over_id = import_sound(
            self.conn,
            {"file": too_long, "license_note": note, "class": "boom", "intensity": "high"},
            self.cfg,
            assets_root=self.assets,
        )
        over = self.conn.execute(
            """SELECT s.duration_ms, s.original_path, s.loudness_lufs
               FROM sound_info s WHERE s.asset_id = ?""",
            (over_id,),
        ).fetchone()
        self.assertGreater(over["duration_ms"], 4000)
        original = Path(over["original_path"])
        self.assertTrue((self.root / original).is_file(), over["original_path"])
        self.assertNotIn("norm", str(original).replace("\\", "/").lower())
        # Loudnorm was rejected. Import must not write a derived audio copy.
        self.assertFalse((self.assets / "sounds" / "norm").exists())

        # Loudnorm was rejected, so loudness is a measurement of the original
        # file, not a guarantee. It is recorded and it is not a gate.
        row = self.conn.execute(
            """SELECT s.loudness_lufs, s.duration_ms, s.original_path
               FROM sound_info s JOIN assets a ON a.id = s.asset_id
               WHERE s.duration_ms >= 1000 AND s.asset_id <> ?""",
            (over_id,),
        ).fetchone()
        self.assertIsNotNone(row["loudness_lufs"])
        self.assertTrue((self.root / row["original_path"]).is_file())

        pool = {r["id"] for r in self.conn.execute("SELECT id FROM pool_images")}
        gold_ids = {r["id"] for r in self.conn.execute("SELECT id FROM assets WHERE is_gold = 1")}
        self.assertTrue(gold_ids)
        self.assertTrue(gold_ids.isdisjoint(pool))
        self.assertFalse(pool, "un-tagged new images stay out of the pool")

        violations = run_audit(self.conn, ROOT / "scripts" / "audit.sql")
        self.assertEqual(violations, [])
        covered, missing = probes_cover_topics(ROOT / "data" / "gold" / "probes.json")
        self.assertTrue(covered, missing)
        self.assertEqual(len(json.loads((ROOT / "data" / "gold" / "probes.json").read_text())), 20)


if __name__ == "__main__":
    unittest.main()
