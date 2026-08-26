#!/usr/bin/env python3
"""CLI: deterministic text editing on images (alpaca stack, no external models).

Erase a horizontal text line (synthesized blank panel) and/or draw new text
into the same band. Pure PIL/numpy - pixel-exact, no diffusion artifacts.

Usage:
  textcompose.py photo.jpg --band 548:558 --erase-only --output clean.png
  textcompose.py photo.jpg --band 548:558 --text "SUNRISE.ORG" --output out.png
  textcompose.py photo.jpg --band 548:558 --text "SUNRISE.ORG" --post vintage
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from imageops import draw_text, fill_band_deterministic  # noqa: E402

SEPIA = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])


def vintage(img: Image.Image, seed: int = 7) -> Image.Image:
    a = np.asarray(img.convert("RGB")).astype(float)
    r = a @ SEPIA.T
    r[:, :, 0] += 42.0
    r[:, :, 1] = r[:, :, 1] * 0.88 + 18.0
    r[:, :, 2] = r[:, :, 2] * 0.66 - 6.0
    grain = np.random.default_rng(seed).normal(0.0, 12.0, r.shape)
    r = np.clip(r + grain, 0, 255)
    h, w, _ = r.shape
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    r = r * (1.0 - 0.30 * np.clip(d - 0.35, 0, 1) ** 2)[:, :, None]
    return Image.fromarray(np.clip(r, 0, 255).astype("uint8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic text editing (alpaca stack)")
    parser.add_argument("image", type=Path, help="Source image")
    parser.add_argument("--band", required=True, help="Text line rows to replace, INCLUSIVE 'y0:y1'")
    parser.add_argument("--gap-above", type=int, default=None, help="Clean row above the band")
    parser.add_argument("--gap-below", type=int, default=None, help="Clean row below the band")
    parser.add_argument("--text", default=None, help="New text to draw (omit for erase-only)")
    parser.add_argument("--font", default=None, help="TTF font path")
    parser.add_argument("--font-size", type=int, default=None, help="Font size in px")
    parser.add_argument("--color", default="255,255,255", help="Text color 'r,g,b'")
    parser.add_argument(
        "--method",
        choices=["model", "fill"],
        default="model",
        help="model = qwen-image-edit via sd-server (default); fill = deterministic PIL fallback",
    )
    parser.add_argument("--post", choices=["none", "vintage"], default="none")
    parser.add_argument("--output", type=Path, default=None, help="Output path")
    parser.add_argument("--verify", action="store_true", help="Print pixel containment stats")
    args = parser.parse_args()

    if not args.image.exists():
        sys.exit(f"ERROR: image not found: {args.image}")
    try:
        y0, y1 = (int(v) for v in args.band.split(":"))
        y1 += 1
    except ValueError:
        sys.exit("ERROR: --band must be 'y0:y1' (inclusive)")
    color: tuple[int, int, int] = tuple(int(v) for v in args.color.split(","))  # type: ignore[assignment]

    image = Image.open(args.image).convert("RGB")
    gap_above = args.gap_above if args.gap_above is not None else y0 - 1
    gap_below = args.gap_below if args.gap_below is not None else y1

    if args.method == "model":
        from imageedit_model import text_edit_model

        result, info = text_edit_model(image, (y0, y1), args.text)
        print(f"[textcompose] {info}", file=sys.stderr)
    else:
        result = fill_band_deterministic(image, y0, y1, gap_above, gap_below)
        if args.text:
            result = draw_text(result, args.text, (y0, y1), args.font, args.font_size, color)

    if args.verify:
        o = np.asarray(image).astype(int)
        r = np.asarray(result).astype(int)
        d = np.abs(o - r).sum(axis=2) > 15
        rows = np.where(d.any(axis=1))[0]
        print(f"[verify] changed rows {(rows.min(), rows.max()) if len(rows) else None}, {int(d.sum())} px")

    if args.post == "vintage":
        result = vintage(result)

    out = args.output or args.image.with_name(f"{args.image.stem}_{'composed' if args.text else 'cleaned'}.png")
    result.save(out)
    print(f"[textcompose] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
