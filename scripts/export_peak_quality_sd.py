#!/usr/bin/env python3
import base64
import json
import os
import time

import httpx
from PIL import Image, ImageStat


def analyze_image_quality(img_path):
    """Analyzes image sharpness, contrast, brightness, and resolution."""
    try:
        img = Image.open(img_path)
        w, h = img.size
        stat = ImageStat.Stat(img.convert('L'))
        mean_brightness = stat.mean[0]
        std_dev_contrast = stat.stddev[0]

        pixels = list(img.convert('L').get_flattened_data())
        diffs = [abs(pixels[i] - pixels[i-1]) for i in range(1, len(pixels))]
        sharpness_score = sum(diffs) / max(1, len(diffs))

        return {
            "resolution": f"{w}x{h}",
            "file_size_kb": round(os.path.getsize(img_path) / 1024, 1),
            "brightness_mean": round(mean_brightness, 2),
            "contrast_stddev": round(std_dev_contrast, 2),
            "sharpness_score": round(sharpness_score, 2),
            "status": "VALIDATED"
        }
    except Exception as e:
        return {"error": str(e), "status": "FAILED"}

def main():
    tmp_dir = "/tmp"
    art_dir = "/home/jeremiah/.gemini/antigravity-cli/brain/b0f68df3-fae3-4c02-9194-7bb52bde4e89"
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(art_dir, exist_ok=True)

    results = []

    # ------------------------------------------------------------------------
    # 1. HIGH-QUALITY TROPICAL BEACH ENVIRONMENT EDIT (Michele 000_0117.jpg)
    # ------------------------------------------------------------------------
    source1 = "/home/jeremiah/Desktop/Michele/000_0117.jpg"
    if os.path.exists(source1):
        print("[1/3] Generating High-Quality Tropical Beach Environment Edit (000_0117.jpg)...", flush=True)
        t0 = time.time()
        prompt1 = (
            'An ultra-realistic RAW photograph of the woman standing on a sunny tropical white sand beach at golden hour sunset, '
            'crystal turquoise ocean waves softly breaking in background, warm directional sunlight, detailed skin texture, '
            'natural hair strands flowing in breeze, 8k resolution, professional 85mm portrait lens, f/1.8 aperture'
            '<sd_cpp_extra_args>{"strength": 0.60, "negative_prompt": "blurry, low quality, deformed, CGI, 3D render, plastic skin, distorted features, oversaturated, indoor, dark room"}</sd_cpp_extra_args>'
        )
        with open(source1, "rb") as pf:
            files = {"image": (os.path.basename(source1), pf.read(), "image/jpeg")}
        data = {
            "model": "qwen-image-edit-rapid-aio:q4_k",
            "prompt": prompt1,
            "size": "512x512",
            "n": "1",
            "steps": "25",
            "guidance": "7.5"
        }
        resp1 = httpx.post("http://localhost:5000/api/sd/edit", data=data, files=files, timeout=600.0)
        if resp1.status_code == 200:
            raw1 = base64.b64decode(resp1.json()["data"][0]["b64_json"])
            tmp_path1 = os.path.join(tmp_dir, "sd_peak_beach_michele_0117.png")
            art_path1 = os.path.join(art_dir, "sd_peak_beach_michele_0117.png")
            for p in [tmp_path1, art_path1]:
                with open(p, "wb") as f:
                    f.write(raw1)
            quality1 = analyze_image_quality(tmp_path1)
            print(f"✅ Beach Edit 1 completed in {time.time()-t0:.2f}s | Saved to {tmp_path1} | Quality: {quality1}", flush=True)
            results.append(("Tropical Beach Environment Edit (000_0117.jpg)", tmp_path1, art_path1, quality1))
        else:
            print(f"❌ Beach edit 1 failed: {resp1.status_code} - {resp1.text}", flush=True)

    # ------------------------------------------------------------------------
    # 2. HIGH-QUALITY COASTAL SUNSET BEACH EDIT (Michele 20140715_084728.jpg)
    # ------------------------------------------------------------------------
    source2 = "/home/jeremiah/Desktop/Michele/20140715_084728.jpg"
    if os.path.exists(source2):
        print("[2/3] Generating High-Quality Coastal Beach Sunset Edit (20140715_084728.jpg)...", flush=True)
        t0 = time.time()
        prompt2 = (
            'A stunning cinematic 8k photograph of the person on a sun-drenched coastal beach during golden sunset, '
            'soft sea spray, golden reflections on wet sand, vibrant blue sky with soft evening clouds, sharp focus, '
            'natural skin tones, high dynamic range, photorealistic'
            '<sd_cpp_extra_args>{"strength": 0.58, "negative_prompt": "blurry, low contrast, grain, noise, cgi, plastic skin, distorted features, indoor, dark"}</sd_cpp_extra_args>'
        )
        with open(source2, "rb") as pf2:
            files2 = {"image": (os.path.basename(source2), pf2.read(), "image/jpeg")}
        data2 = {
            "model": "qwen-image-edit-rapid-aio:q4_k",
            "prompt": prompt2,
            "size": "512x512",
            "n": "1",
            "steps": "25",
            "guidance": "7.5"
        }
        resp2 = httpx.post("http://localhost:5000/api/sd/edit", data=data2, files=files2, timeout=600.0)
        if resp2.status_code == 200:
            raw2 = base64.b64decode(resp2.json()["data"][0]["b64_json"])
            tmp_path2 = os.path.join(tmp_dir, "sd_peak_beach_michele_2014.png")
            art_path2 = os.path.join(art_dir, "sd_peak_beach_michele_2014.png")
            for p in [tmp_path2, art_path2]:
                with open(p, "wb") as f:
                    f.write(raw2)
            quality2 = analyze_image_quality(tmp_path2)
            print(f"✅ Beach Edit 2 completed in {time.time()-t0:.2f}s | Saved to {tmp_path2} | Quality: {quality2}", flush=True)
            results.append(("Coastal Sunset Beach Edit (20140715_084728.jpg)", tmp_path2, art_path2, quality2))
        else:
            print(f"❌ Beach edit 2 failed: {resp2.status_code} - {resp2.text}", flush=True)

    # ------------------------------------------------------------------------
    # 3. HIGH-END LUXURY TROPICAL BEACH TEXT-TO-IMAGE PORTRAIT
    # ------------------------------------------------------------------------
    print("[3/3] Generating High-End 8K Text-to-Image Luxury Beach Portrait...", flush=True)
    t0 = time.time()
    gen_payload = {
        "model": "qwen-image-edit-rapid-aio:q4_k",
        "prompt": "An 8k RAW photorealistic portrait photograph of a striking model on a luxury tropical beach at sunset, turquoise ocean waves, cinematic golden hour lighting, detailed skin textures, natural hair, professional 85mm lens f/1.4, masterpiece, ultra detailed",
        "size": "512x512",
        "n": 1,
        "negative_prompt": "blurry, low quality, deformed, face distortion, plastic skin, cgi, 3d render, oversaturated",
        "steps": 25,
        "guidance": 7.5
    }
    resp3 = httpx.post("http://localhost:5000/api/sd/generate", json=gen_payload, timeout=600.0)
    if resp3.status_code == 200:
        raw3 = base64.b64decode(resp3.json()["data"][0]["b64_json"])
        tmp_path3 = os.path.join(tmp_dir, "sd_peak_text2img_beach.png")
        art_path3 = os.path.join(art_dir, "sd_peak_text2img_beach.png")
        for p in [tmp_path3, art_path3]:
            with open(p, "wb") as f:
                f.write(raw3)
        quality3 = analyze_image_quality(tmp_path3)
        print(f"✅ Peak Text-to-Image completed in {time.time()-t0:.2f}s | Saved to {tmp_path3} | Quality: {quality3}", flush=True)
        results.append(("Peak 8K Text-to-Image Luxury Beach Portrait", tmp_path3, art_path3, quality3))
    else:
        print(f"❌ Peak text-to-image failed: {resp3.status_code} - {resp3.text}", flush=True)

    print("\n=== Peak Quality Image Export Summary ===", flush=True)
    for title, tmp_p, _art_p, qual in results:
        print(f"Title: {title}", flush=True)
        print(f"  /tmp Path: {tmp_p}", flush=True)
        print(f"  Quality Metrics: {json.dumps(qual)}", flush=True)

if __name__ == "__main__":
    main()
