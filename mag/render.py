"""Pillow renderer. Text only. GIF frames stay frames. The original file is not modified."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageSequence

from mag.config import RenderCfg
from mag.paths import ROOT


class RenderRejected(ValueError):
    """Fit failed at the minimum font size. The candidate is dropped, not repaired."""


def _font(cfg: RenderCfg, size: int):
    path = ROOT / cfg.font_file
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _fits(draw, text: str, font, width: int, margin: int) -> bool:
    for line in text.split("\n"):
        box = draw.textbbox((0, 0), line, font=font)
        if box[2] - box[0] > width - 2 * margin:
            return False
    return True


def _pick_font(draw, text: str, cfg: RenderCfg, width: int):
    for size in range(cfg.max_font_px, cfg.min_font_px - 1, -2):
        font = _font(cfg, size)
        if _fits(draw, text, font, width, cfg.margin):
            return font
    raise RenderRejected("caption does not fit at min_font_px")


def _draw_bar(image, text: str, cfg: RenderCfg):
    base = image.convert("RGB")
    canvas = Image.new("RGB", (base.width, base.height + cfg.bar_height), (255, 255, 255))
    canvas.paste(base, (0, cfg.bar_height))
    draw = ImageDraw.Draw(canvas)
    font = _pick_font(draw, text, cfg, canvas.width)
    draw.multiline_text((cfg.margin, cfg.margin), text, font=font, fill=(0, 0, 0), spacing=4)
    return canvas


def _draw_impact(image, text: str, cfg: RenderCfg):
    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas)
    font = _pick_font(draw, text, cfg, canvas.width)
    lines = text.split("\n")
    line_h = cfg.max_font_px + 6
    y = cfg.margin if len(lines) == 1 else canvas.height - cfg.margin - line_h * len(lines)
    for line in lines:
        x = cfg.margin
        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0, 255))
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
        y += line_h
    return canvas


def render_image(src: Path, text: str, layout: str, dest: Path, cfg: RenderCfg) -> Path:
    """Draw the caption onto a copy. Animated GIF is written as GIF, one caption per frame."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        animated = bool(getattr(image, "is_animated", False)) and getattr(image, "n_frames", 1) > 1
        if animated:
            frames = []
            duration = image.info.get("duration", 100)
            for frame in ImageSequence.Iterator(image):
                drawn = _draw_impact(frame, text, cfg) if layout == "impact" else _draw_bar(frame, text, cfg)
                frames.append(drawn.convert("P", palette=Image.Palette.ADAPTIVE))
            frames[0].save(dest, save_all=True, append_images=frames[1:], loop=0, duration=duration, format="GIF")
            return dest
        drawn = _draw_impact(image, text, cfg) if layout == "impact" else _draw_bar(image, text, cfg)
        if dest.suffix.lower() == ".jpg":
            drawn.convert("RGB").save(dest, format="JPEG", quality=90)
        else:
            drawn.save(dest)
    return dest


def cache_path(cache_dir: Path, image_sha: str, text: str, layout: str) -> Path:
    digest = hashlib.sha256(f"{image_sha}|{layout}|{text}".encode()).hexdigest()
    suffix = ".gif" if image_sha.endswith("gif") else ".png"
    return cache_dir / f"{digest}{suffix}"
