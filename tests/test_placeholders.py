"""Placeholder tags: snapshot, clear, restore.

These are the tags mag.inbox writes so a dropped file enters the pool. Clearing
them is what lets a real tag run write anything at all, so the round trip has
to be exact.
"""

from __future__ import annotations

import os
import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])

# Must precede every mag import: it fixes MAG_ROOT for the whole process.
import _scratch_root  # noqa: E402

from PIL import Image  # noqa: E402

from mag.catalog import seed_catalog  # noqa: E402
from mag.config import load_config  # noqa: E402
from mag.db import connect, init_db  # noqa: E402
from mag.importing import import_image, write_human_tags  # noqa: E402
from mag.placeholders import (  # noqa: E402
    clear_placeholders,
    restore_placeholders,
    snapshot_placeholders,
)
from mag.serve import pool_images  # noqa: E402

SCRATCH = _scratch_root.mag_root()
NOTE = "operator fixture"
PLACEHOLDER = {
    "emotion": "deadpan",
    "intensity": "mid",
    "family": "other",
    "caption_zone": "both",
    "safety": "ok",
    "topic": ["everyday"],
}


def _image(root: Path, index: int) -> int:
    path = root / "assets" / "inbox" / "images" / f"thing_{index}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Distinct bytes per index: assets.sha256 is unique, so identical images
    # would dedupe into a single row.
    Image.new("RGB", (900, 700), (20 + index * 7, 60, 120)).save(path)
    asset_id = import_image(conn_for(root), path, NOTE, assets_root=root / "assets")
    write_human_tags(
        conn_for(root), asset_id, PLACEHOLDER,
        {"subject": f"thing {index}", "action": "waiting"},
    )
    return asset_id


_CONNECTIONS: dict = {}


def _open_store(root: Path):
    for name in ("assets", "data"):
        shutil.rmtree(root / name, ignore_errors=True)
    (root / "assets" / "inbox" / "images").mkdir(parents=True, exist_ok=True)
    conn = connect(root / "data" / "mag.sqlite")
    init_db(conn, _scratch_root.real_config("vocab.json"))
    seed_catalog(conn)
    _CONNECTIONS[root] = conn
    return conn


def conn_for(root: Path):
    if root not in _CONNECTIONS:
        raise AssertionError("call _open_store first")
    return _CONNECTIONS[root]


class PlaceholderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.conn = _open_store(SCRATCH)
        cls.ids = [_image(SCRATCH, index) for index in range(4)]
        # One image the operator has since hand-tagged for real. It must survive.
        cls.kept = _image(SCRATCH, 99)
        write_human_tags(cls.conn, cls.kept, {
            "emotion": "crying", "intensity": "high", "family": "reaction",
            "caption_zone": "top", "safety": "ok", "topic": ["animals"],
        }, {"subject": "real cat", "action": "crying"})

    @classmethod
    def tearDownClass(cls) -> None:
        cls.conn.close()
        _CONNECTIONS.clear()

    def _tags(self, asset_id: int) -> dict[str, list[str]]:
        rows = self.conn.execute(
            "SELECT axis, value FROM tags WHERE asset_id = ? AND superseded = 0",
            (asset_id,),
        ).fetchall()
        out: dict[str, list[str]] = {}
        for row in rows:
            out.setdefault(row["axis"], []).append(row["value"])
        return out

    def test_snapshot_only_catches_placeholders(self) -> None:
        snapshot = snapshot_placeholders(self.conn)
        captured = {asset.asset_id for asset in snapshot.assets}
        self.assertEqual(captured, set(self.ids))
        self.assertNotIn(self.kept, captured)
        self.assertEqual(snapshot.asset_count, 4)
        self.assertEqual(snapshot.tag_count, 4 * 6)

    def test_clear_empties_the_pool_and_restore_brings_it_back(self) -> None:
        before_pool = len(pool_images(self.conn, None))
        self.assertEqual(before_pool, 5)
        snapshot = snapshot_placeholders(self.conn)
        self.assertEqual(clear_placeholders(self.conn, snapshot), 4)
        for asset_id in self.ids:
            self.assertEqual(self._tags(asset_id), {})
            state = self.conn.execute(
                "SELECT state FROM assets WHERE id = ?", (asset_id,)
            ).fetchone()[0]
            self.assertEqual(state, "new", asset_id)
        # The operator's real tags are untouched and keep their image in the pool.
        self.assertEqual(self._tags(self.kept)["emotion"], ["crying"])
        self.assertEqual(len(pool_images(self.conn, None)), 1)

        self.assertEqual(restore_placeholders(self.conn, snapshot), 4)
        for asset_id in self.ids:
            self.assertEqual(self._tags(asset_id)["emotion"], ["deadpan"])
            state = self.conn.execute(
                "SELECT state FROM assets WHERE id = ?", (asset_id,)
            ).fetchone()[0]
            self.assertEqual(state, "active", asset_id)
        self.assertEqual(len(pool_images(self.conn, None)), before_pool)

    def test_clear_is_idempotent(self) -> None:
        snapshot = snapshot_placeholders(self.conn)
        clear_placeholders(self.conn, snapshot)
        again = snapshot_placeholders(self.conn)
        self.assertEqual(again.asset_count, 0)
        clear_placeholders(self.conn, again)
        self.assertEqual(clear_placeholders(self.conn, again), 0)
        restore_placeholders(self.conn, snapshot)

    def test_dry_run_writes_nothing(self) -> None:
        snapshot = snapshot_placeholders(self.conn)
        self.assertEqual(clear_placeholders(self.conn, snapshot, dry_run=True), 4)
        self.assertEqual(len(pool_images(self.conn, None)), 5)
        self.assertEqual(self._tags(self.ids[0])["emotion"], ["deadpan"])


if __name__ == "__main__":
    unittest.main()
