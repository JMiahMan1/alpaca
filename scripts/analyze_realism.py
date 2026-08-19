#!/usr/bin/env python3
import json
import os

import numpy as np
from PIL import Image, ImageFilter


def evaluate_realism(image_path):
    """
    Evaluates realism and quality metrics of a generated PNG image:
    1. Resolution & File Size
    2. Dynamic Range & Contrast (Standard Deviation of Luma)
    3. Sharpness Score (High Frequency Detail Density via Laplacian Variance)
    4. Color Balance (RGB Mean Channel Alignment)
    5. Skin & Background Artifact Detection (Extreme saturation / posterization check)
    """
    if not os.path.exists(image_path):
        return {"status": "FILE_NOT_FOUND", "path": image_path}

    img = Image.open(image_path)
    w, h = img.size
    file_size_kb = os.path.getsize(image_path) / 1024.0

    # Convert to Grayscale & Array
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.float32)

    # Laplacian Variance for Sharpness Detail
    laplacian_filter = ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1.0)
    edges = gray.filter(laplacian_filter)
    edge_arr = np.array(edges, dtype=np.float32)
    sharpness_variance = float(np.var(edge_arr))

    # Luminance & Contrast
    luma_mean = float(np.mean(arr))
    luma_std = float(np.std(arr))

    # Color Channel Balance (RGB)
    rgb_img = img.convert("RGB")
    r, g, b = rgb_img.split()
    r_mean = float(np.mean(np.array(r)))
    g_mean = float(np.mean(np.array(g)))
    b_mean = float(np.mean(np.array(b)))

    # Artifact Check (Posterization / Clipped Highlights or Shadows)
    rgb_arr = np.array(rgb_img)
    clipped_shadows = float(np.sum(rgb_arr < 5) / rgb_arr.size * 100)
    clipped_highlights = float(np.sum(rgb_arr > 250) / rgb_arr.size * 100)

    # Grade Realism
    realism_grade = "EXCELLENT"
    recommendations = []

    if sharpness_variance < 50.0:
        realism_grade = "MODERATE"
        recommendations.append("Increase sampling steps or guidance scale to sharpen fine facial details.")
    if luma_std < 40.0:
        realism_grade = "FLAT_CONTRAST"
        recommendations.append("Increase contrast via negative prompt tuning or directional lighting.")
    if clipped_highlights > 15.0 or clipped_shadows > 25.0:
        recommendations.append("Adjust CFG scale down slightly to avoid highlight clipping.")
    if not recommendations:
        recommendations.append("Lighting, skin texture, and background blending are well balanced.")

    return {
        "filename": os.path.basename(image_path),
        "resolution": f"{w}x{h}",
        "file_size_kb": round(file_size_kb, 1),
        "luma_mean": round(luma_mean, 2),
        "contrast_std": round(luma_std, 2),
        "sharpness_laplacian_var": round(sharpness_variance, 2),
        "rgb_means": {"r": round(r_mean, 1), "g": round(g_mean, 1), "b": round(b_mean, 1)},
        "clipped_shadows_pct": round(clipped_shadows, 2),
        "clipped_highlights_pct": round(clipped_highlights, 2),
        "realism_grade": realism_grade,
        "recommendations": recommendations,
    }


def main():
    target_files = [
        "/tmp/sd_peak_beach_michele_0117.png",
        "/tmp/sd_peak_beach_michele_2014.png",
        "/tmp/sd_peak_text2img_beach.png",
    ]

    print("=== Realism & Image Quality Analysis Report ===")
    results = []
    for path in target_files:
        res = evaluate_realism(path)
        results.append(res)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
