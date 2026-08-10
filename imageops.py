"""Shared image ops for the alpaca edit stack (deterministic, no external deps).

fill_band_deterministic: seamless blank-panel synthesis for text-line removal.
  Vertical interpolation between the two clean gap rows (preserves the
  diagonal glare streak), plus per-column local texture matching from a clean
  panel strip, plus soft edge ramps. Never blends original glyph pixels.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def fill_band_deterministic(
    image: Image.Image,
    y0: int,
    y1: int,
    gap_above: int,
    gap_below: int,
    texture_rows: tuple[int, int] | None = None,
    seed: int = 7,
) -> Image.Image:
    """Replace rows [y0, y1) with synthesized panel content.

    gap_above / gap_below: clean (glyph-free) row indices flanking the band;
    the per-column tone is a linear blend of those two rows, which keeps the
    glare streak continuous through the band. texture_rows: clean strip used
    to match local noise level per column.
    """
    a = np.asarray(image.convert("RGB")).astype(float)
    h, w, _ = a.shape
    y0, y1 = max(0, y0), min(h, y1)
    if gap_above < 0 or gap_below >= h or gap_above >= gap_below:
        raise ValueError("gap rows must bracket the band")

    tr0, tr1 = texture_rows or (gap_below, min(gap_below + 6, h))
    rows = a[y1:y1 + (y1 - y0)] if y1 + (y1 - y0) <= h else a[tr0:tr1]
    band_h = y1 - y0

    base = np.empty((band_h, w, 3), dtype=float)
    t = np.linspace(0.0, 1.0, band_h)
    for k in range(band_h):
        base[k] = a[gap_above] * (1.0 - t[k]) + a[gap_below] * t[k]

    local_std = np.zeros((band_h, w, 1), dtype=float)
    strip = rows.astype(float)
    if strip.shape[0] >= 3:
        for k in range(band_h):
            for ch in range(3):
                col = strip[:, :, ch]
                spread = col.std(axis=0)
                local_std[k, :, 0] = np.minimum(spread, 6.0)

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, (band_h, w, 3)) * local_std

    edge = np.minimum(np.arange(band_h), np.arange(band_h)[::-1])
    edge = np.clip(edge / 3.0, 0.0, 1.0)[:, None, None]
    fill = np.clip(base + noise * edge, 0, 255)

    out = a.copy()
    out[y0:y1] = fill
    return Image.fromarray(np.clip(out, 0, 255).astype("uint8"))


def draw_text(
    image: Image.Image,
    text: str,
    band: tuple[int, int],
    font_path: str | None = None,
    font_size: int | None = None,
    color: tuple[int, int, int] = (255, 255, 255),
    align: str = "center",
    seed: int = 7,
) -> Image.Image:
    """Draw text centered in a horizontal band with optional light grain."""
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    y0, y1 = band
    band_h = y1 - y0
    size = font_size or max(10, int(band_h * 0.85))
    try:
        font: ImageFont.ImageFont = ImageFont.truetype(
            font_path or "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", size
        )
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (out.width - tw) / 2 - bbox[0]
    if align == "left":
        x = 0.0
    y = (y0 + y1) / 2 - th / 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=color)
    return out
