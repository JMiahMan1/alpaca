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

        pixels = list(img.convert('L').getdata())
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

    # 1. Flyer Generation
    print("[1/3] Generating Grand Opening Flyer Poster...")
    flyer_payload = {
        "model": "qwen-image-edit-rapid-aio:q4_k",
        "prompt": "flyer graphic design, main title text reading \"GRAND OPENING SALE\", subtext reading \"UP TO 50% OFF THIS WEEKEND\", product sale promo poster, vibrant blue and gold lighting, sharp typography, clean layout, 8k resolution",
        "size": "512x512",
        "n": 1,
        "negative_prompt": "garbled text, distorted letters, bad typography, misspelled text, blurry letters",
        "steps": 20
    }
    t0 = time.time()
    resp1 = httpx.post("http://localhost:5000/api/sd/generate", json=flyer_payload, timeout=300.0)
    if resp1.status_code == 200:
        raw1 = base64.b64decode(resp1.json()["data"][0]["b64_json"])
        tmp_path1 = os.path.join(tmp_dir, "sd_flyer_grand_opening.png")
        art_path1 = os.path.join(art_dir, "sd_flyer_grand_opening.png")
        for p in [tmp_path1, art_path1]:
            with open(p, "wb") as f:
                f.write(raw1)
        quality1 = analyze_image_quality(tmp_path1)
        print(f"✅ Flyer generated in {time.time()-t0:.2f}s | Saved to {tmp_path1} | Quality: {quality1}")
        results.append(("Flyer Graphic (Grand Opening)", tmp_path1, art_path1, quality1))
    else:
        print(f"❌ Flyer generation failed: {resp1.status_code} - {resp1.text}")

    # 2. Photo Edit Michele 000_0117.jpg
    source1 = "/home/jeremiah/Desktop/Michele/000_0117.jpg"
    if os.path.exists(source1):
        print(f"[2/3] Editing photo {source1}...")
        t0 = time.time()
        prompt1 = '8k RAW photo, portrait photograph of subject, detailed skin texture, natural soft studio lighting, sharp focus, 85mm lens f/1.8<sd_cpp_extra_args>{"strength": 0.45, "negative_prompt": "cgi, 3d render, plastic skin, distorted features, low quality"}</sd_cpp_extra_args>'
        with open(source1, "rb") as pf:
            files = {"image": (os.path.basename(source1), pf.read(), "image/jpeg")}
        data = {"model": "qwen-image-edit-rapid-aio:q4_k", "prompt": prompt1, "size": "512x512", "n": "1"}
        resp2 = httpx.post("http://localhost:5000/api/sd/edit", data=data, files=files, timeout=300.0)
        if resp2.status_code == 200:
            raw2 = base64.b64decode(resp2.json()["data"][0]["b64_json"])
            tmp_path2 = os.path.join(tmp_dir, "sd_photo_edit_michele_0117.png")
            art_path2 = os.path.join(art_dir, "sd_photo_edit_michele_0117.png")
            for p in [tmp_path2, art_path2]:
                with open(p, "wb") as f:
                    f.write(raw2)
            quality2 = analyze_image_quality(tmp_path2)
            print(f"✅ Photo Edit 1 completed in {time.time()-t0:.2f}s | Saved to {tmp_path2} | Quality: {quality2}")
            results.append(("Photo Edit (000_0117.jpg)", tmp_path2, art_path2, quality2))
        else:
            print(f"❌ Photo edit 1 failed: {resp2.status_code} - {resp2.text}")

    # 3. Photo Retouch Michele 20140715_084728.jpg
    source2 = "/home/jeremiah/Desktop/Michele/20140715_084728.jpg"
    if os.path.exists(source2):
        print(f"[3/3] Retouching photo {source2}...")
        t0 = time.time()
        prompt2 = 'cinematic photo color grading, balanced lighting, deep contrast, natural skin tones, professional photography<sd_cpp_extra_args>{"strength": 0.35, "negative_prompt": "flat color, oversaturated, washed out, noisy, artifact"}</sd_cpp_extra_args>'
        with open(source2, "rb") as pf2:
            files2 = {"image": (os.path.basename(source2), pf2.read(), "image/jpeg")}
        data2 = {"model": "qwen-image-edit-rapid-aio:q4_k", "prompt": prompt2, "size": "512x512", "n": "1"}
        resp3 = httpx.post("http://localhost:5000/api/sd/edit", data=data2, files=files2, timeout=300.0)
        if resp3.status_code == 200:
            raw3 = base64.b64decode(resp3.json()["data"][0]["b64_json"])
            tmp_path3 = os.path.join(tmp_dir, "sd_photo_retouch_michele_20140715.png")
            art_path3 = os.path.join(art_dir, "sd_photo_retouch_michele_20140715.png")
            for p in [tmp_path3, art_path3]:
                with open(p, "wb") as f:
                    f.write(raw3)
            quality3 = analyze_image_quality(tmp_path3)
            print(f"✅ Photo Edit 2 completed in {time.time()-t0:.2f}s | Saved to {tmp_path3} | Quality: {quality3}")
            results.append(("Photo Retouch (20140715_084728.jpg)", tmp_path3, art_path3, quality3))
        else:
            print(f"❌ Photo edit 2 failed: {resp3.status_code} - {resp3.text}")

    print("\n--- Summary of Exported Images ---")
    for title, tmp_p, _art_p, q in results:
        print(f"Image: {title}")
        print(f"  /tmp Path: {tmp_p}")
        print(f"  Quality: {json.dumps(q)}")

if __name__ == "__main__":
    main()
