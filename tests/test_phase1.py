"""Phase 1 exit rules. No model, no GPU."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from PIL import Image

from mag.audit import run_audit
from mag.config import load_config
from mag.db import connect, init_db
from mag.importing import ImportRejected, ensure_layout, import_batch, import_image, import_sound
from mag.phase1 import probes_cover_topics


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
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.assets = self.root / "assets"
        ensure_layout(self.root)
        self.conn = connect(self.root / "data" / "mag.sqlite")
        init_db(self.conn, ROOT / "config" / "vocab.json")
        self.cfg = load_config(ROOT / "config" / "mag.toml")

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def test_exit_rules(self) -> None:
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
        with self.assertRaises(ImportRejected):
            import_sound(
                self.conn,
                {"file": too_long, "license_note": note, "class": "boom", "intensity": "high"},
                self.cfg,
                assets_root=self.assets,
            )
        hidden = self.assets / "quarantine" / "nope.png"
        _png(hidden, (1, 1, 1))
        with self.assertRaises(ImportRejected):
            import_image(self.conn, hidden, note, assets_root=self.assets)

        row = self.conn.execute(
            """SELECT s.loudness_lufs, s.duration_ms, s.norm_path
               FROM sound_info s JOIN assets a ON a.id = s.asset_id
               WHERE s.duration_ms >= 1000"""
        ).fetchone()
        self.assertLess(abs(row["loudness_lufs"] - self.cfg.sound.target_lufs), self.cfg.sound.lufs_tolerance)
        self.assertTrue((self.root / row["norm_path"]).is_file())

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
