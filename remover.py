#!/usr/bin/env python3
"""CLI for artifact-free object/text removal via LaMa inpainting.

Runs simple-lama-inpainting (big-lama.pt, deterministic CNN - cannot
hallucinate text back) on CPU or CUDA. Supports a ready-made mask PNG or a
horizontal band (--band y0:y1) built automatically.

Usage:
  remover.py photo.jpg mask.png --output out.png
  remover.py photo.jpg --band 547:559 --post vintage --verify

Run under the iopaint venv:  ~/.venvs/iopaint/bin/python3 remover.py ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SEPIA = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])


def vintage(img: Image.Image, seed: int = 7) -> Image.Image:
    """Proven vintage post: sepia matrix + warm lift + grain + vignette."""
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


def build_band_mask(image: Image.Image, y0: int, y1: int, expand: int = 1) -> Image.Image:
    w, h = image.size
    y0, y1 = max(0, y0 - expand), min(h, y1 + expand)
    mask = Image.new("L", (w, h), 0)
    mask.paste(255, (0, y0, w, y1))
    return mask


def load_mask(mask_path: Path, image_size: tuple[int, int]) -> Image.Image:
    mask = Image.open(mask_path).convert("L")
    if mask.size != image_size:
        sys.exit(f"ERROR: mask size {mask.size} != image size {image_size}")
    return mask


def verify(image: Image.Image, result: Image.Image, mask: Image.Image) -> dict:
    """Containment check: report change bbox, outside-mask damage, band stats."""
    o = np.asarray(image.convert("RGB")).astype(int)
    r = np.asarray(result.convert("RGB")).astype(int)
    m = np.asarray(mask.convert("L"))
    diff = np.abs(o - r).sum(axis=2)
    changed = diff > 15
    rows = np.where(changed.any(axis=1))[0]
    bbox_rows = (int(rows.min()), int(rows.max())) if len(rows) else (None, None)
    outside = changed & (m < 128)
    band = m >= 128
    return {
        "changed_rows": bbox_rows,
        "outside_mask_px": int(outside.sum()),
        "outside_mask_pct": round(float(outside.sum() / max(changed.sum(), 1)) * 100, 2),
        "band_mean_diff": round(float(diff[band].mean()), 2),
        "band_std": round(float(o[band].std()), 2),
        "result_band_std": round(float(r[band].std()), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="LaMa inpainting for clean object/text removal")
    parser.add_argument("image", type=Path, help="Source image")
    parser.add_argument("mask", type=Path, nargs="?", default=None, help="Mask PNG (white = remove)")
    parser.add_argument("--band", help="Remove horizontal band 'y0:y1' instead of a mask file")
    parser.add_argument("--expand", type=int, default=1, help="Pixels to add above/below a --band")
    parser.add_argument("--output", type=Path, default=None, help="Output path")
    parser.add_argument("--post", choices=["none", "vintage"], default="none", help="Post-processing")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--verify", action="store_true", help="Print pixel containment verification")
    args = parser.parse_args()

    if not args.image.exists():
        sys.exit(f"ERROR: image not found: {args.image}")
    if not args.mask and not args.band:
        sys.exit("ERROR: provide a mask PNG or --band y0:y1")

    image = Image.open(args.image).convert("RGB")
    if args.band:
        try:
            y0, y1 = (int(v) for v in args.band.split(":"))
        except ValueError:
            sys.exit("ERROR: --band must be 'y0:y1'")
        mask = build_band_mask(image, y0, y1, args.expand)
    else:
        mask = load_mask(args.mask, image.size)

    try:
        from simple_lama_inpainting import SimpleLama
    except ImportError as e:
        sys.exit(f"ERROR: simple-lama-inpainting missing: {e} (use ~/.venvs/iopaint)")

    lama = SimpleLama(args.device)
    result = lama(image, mask)

    if args.verify:
        print(json.dumps(verify(image, result, mask), indent=2))

    if args.post == "vintage":
        result = vintage(result)

    out = args.output or args.image.with_name(f"{args.image.stem}_removed.png")
    result.save(out)
    print(f"[remover] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
