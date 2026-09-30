"""Renderer rules. No model, no GPU.

The operator saw a hard-coded (18, 18, 18) caption bar and a caption that never
changed size. Both had the same cause: assets/fonts/ was empty, so _font fell
back to ImageFont.load_default(), which ignores the requested size.
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

from mag.config import RenderCfg, load_config  # noqa: E402
from mag.render import (  # noqa: E402
    RenderRejected, _font, _font_paths, cache_path, render_image,
)

SCRATCH = _scratch_root.mag_root() / "render"
REAL_CONFIG = _scratch_root.real_config("mag.toml")
# A real truetype on the machine, so the test does not depend on a bundled font.
WINDOWS_BOLD = Path("C:/Windows/Fonts/arialbd.ttf")


def _cfg(**overrides) -> RenderCfg:
    base = {
        "min_font_px": 18,
        "max_font_px": 64,
        "bar_height": 96,
        "margin": 16,
        "font_file": str(_scratch_root.real_config("fonts") / "DejaVuSans-Bold.ttf"),
        "font_fallbacks": [str(WINDOWS_BOLD)],
    }
    base.update(overrides)
    return RenderCfg(**base)


class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        shutil.rmtree(SCRATCH, ignore_errors=True)
        SCRATCH.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(SCRATCH, ignore_errors=True)

    def test_requested_size_is_the_size_used(self) -> None:
        cfg = _cfg()
        for size in (18, 32, 64):
            font = _font(cfg, size)
            self.assertEqual(font.size, size)

    def test_shrinks_to_fit_then_rejects(self) -> None:
        cfg = _cfg()
        sizes = {size: _font(cfg, size).size for size in (64, 40, 18)}
        self.assertEqual(sizes, {64: 64, 40: 40, 18: 18})

    def test_missing_font_raises_instead_of_defaulting(self) -> None:
        cfg = _cfg(font_file="assets/fonts/nope.ttf", font_fallbacks=[])
        with self.assertRaises(RenderRejected):
            _font(cfg, 32)

    def test_fallback_resolves_when_configured_font_is_missing(self) -> None:
        cfg = _cfg(font_file="assets/fonts/nope.ttf")
        paths = _font_paths(cfg)
        self.assertTrue(paths[0].name.endswith("nope.ttf"))
        self.assertEqual(paths[1], WINDOWS_BOLD)
        self.assertEqual(_font(cfg, 32).size, 32)

    def test_cache_path_is_stable_and_specific(self) -> None:
        first = cache_path(SCRATCH, "abc123", "this was me", "bar")
        self.assertEqual(first, cache_path(SCRATCH, "abc123", "this was me", "bar"))
        self.assertNotEqual(first, cache_path(SCRATCH, "abc123", "this was me", "impact"))
        self.assertNotEqual(first, cache_path(SCRATCH, "abc123", "something else", "bar"))
        self.assertNotEqual(first, cache_path(SCRATCH, "def456", "this was me", "bar"))
        self.assertTrue(first.name.endswith(".png"))

    def test_gif_source_keeps_the_gif_suffix(self) -> None:
        path = cache_path(SCRATCH, "abc123", "text", "bar", ".gif")
        self.assertTrue(path.name.endswith(".gif"), path.name)

    def test_bar_layout_is_white_with_black_text(self) -> None:
        src = SCRATCH / "src.png"
        dest = SCRATCH / "out.png"
        src.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (300, 200), (10, 20, 30)).save(src)
        render_image(src, "this was me", "bar", dest, _cfg())
        with Image.open(dest) as drawn:
            self.assertEqual(drawn.size, (300, 200 + 96))
            bar = drawn.crop((0, 0, 300, 96)).convert("RGB")
            self.assertEqual(bar.getpixel((2, 2)), (255, 255, 255))
            # Black text pixels must exist inside the bar.
            self.assertTrue(
                any(sum(px) < 200 for px in bar.getdata()),
                "bar is white with no black text",
            )

    def test_oversized_caption_is_rejected_not_shrunk_forever(self) -> None:
        src = SCRATCH / "small.png"
        dest = SCRATCH / "small_out.png"
        src.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (60, 40), (0, 0, 0)).save(src)
        with self.assertRaises(RenderRejected):
            render_image(src, "a very long caption that cannot fit at any size", "bar", dest, _cfg())
        self.assertFalse(dest.exists())

    def test_shipped_config_resolves_a_font(self) -> None:
        cfg = load_config(REAL_CONFIG).render
        resolved = [path for path in _font_paths(cfg) if path.exists()]
        self.assertTrue(resolved, f"no font in {_font_paths(cfg)}")
        self.assertEqual(_font(cfg, cfg.max_font_px).size, cfg.max_font_px)


if __name__ == "__main__":
    unittest.main()
