#!/usr/bin/env python3
"""benchmark-image.py - performance sweep for sd-server image models.

Sweeps GPU-layer offload x diffusion steps per image model and records
images/sec, peak VRAM, and sample outputs for human quality review:

    sudo python3 benchmark-image.py [--models m1,m2] [--out ...]

Flow per model:
  1. Unload the audio backend (VRAM isolation; the proxy evicts any loaded
     LLM automatically when the SD model loads).
  2. For each gpu_layers in {30, 40, 50}: write the router .profile.json
     override, load the model via the web API (triggers an sd-server
     restart only when the config actually changed), then generate at each
     step count while polling nvidia-smi for peak VRAM.
  3. Keep the fastest config's gpu_layers in the profile and take a final
     1024px sample. Quality is NOT auto-graded: samples are saved under
     data/sd_bench/samples/ for human review.

Step grids are family-aware: distilled "rapid" models sweep few-step
budgets (4/8); full models sweep quality budgets (10/20).
"""

import argparse
import base64
import glob
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent
ROUTER_DIR = Path(os.getenv("ROUTER_MODELS_DIR", REPO / ".alpaca-router"))
WEB_URL = os.getenv("WEB_URL", "http://localhost:5000")
OUT_DEFAULT = REPO / "data" / "sd_bench" / "image_bench.json"
SAMPLES_DIR = REPO / "data" / "sd_bench" / "samples"

PROMPT = "a red barn in a green field at sunset, photorealistic, sharp focus"
SIZE = "768x768"
FINAL_SIZE = "1024x1024"
SEED = 42
GPU_LAYERS_GRID = [30, 40, 50]


def steps_for(model: str) -> list:
    name = model.lower()
    if "rapid" in name or "turbo" in name or "lightning" in name or "distill" in name:
        return [4, 8]
    return [10, 20]


def vram_used_mb() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


class PeakVram:
    """Poll nvidia-smi in the background; report the peak while active."""

    def __init__(self) -> None:
        self.peak: int = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            v = vram_used_mb()
            if v is not None and v > self.peak:
                self.peak = v
            time.sleep(1.0)

    def __enter__(self) -> "PeakVram":
        base = vram_used_mb() or 0
        self.peak = base
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


def list_models(client: httpx.Client) -> list:
    try:
        r = client.get(f"{WEB_URL}/api/sd/models", timeout=15.0)
        data = r.json()
    except Exception as e:
        print(f"[img] model list failed: {e}")
        return []
    items = data.get("data") or data.get("models") or []
    names = []
    for it in items:
        if isinstance(it, str):
            names.append(it)
        elif isinstance(it, dict):
            names.append(it.get("id") or it.get("name") or "")
    return [n for n in names if n]


def set_profile_gpu_layers(model: str, layers: int) -> list:
    """Write n-gpu-layers into every router profile matching the model name.

    Returns the profile paths touched (for logging). The proxy's
    ensure_sd_model_loaded reads the profile on load and restarts sd-server
    only when the effective config changed.
    """
    touched = []
    for path in glob.glob(str(ROUTER_DIR / "*.profile.json")):
        if model.split(":")[0].replace("_", "-") not in os.path.basename(path).replace("_", "-"):
            continue
        try:
            prof = json.loads(Path(path).read_text())
        except Exception:
            prof = {}
        prof["n-gpu-layers"] = layers
        Path(path).write_text(json.dumps(prof, indent=2) + "\n")
        touched.append(path)
    return touched


def unload_audio(client: httpx.Client) -> None:
    try:
        r = client.post(f"{WEB_URL}/api/audio/unload", timeout=60.0)
        print(f"[img] audio unload -> HTTP {r.status_code}")
    except Exception as e:
        print(f"[img] audio unload skipped ({str(e)[:80]})")


def load_model(client: httpx.Client, model: str) -> tuple:
    t0 = time.time()
    try:
        r = client.post(f"{WEB_URL}/api/sd/load", json={"model": model}, timeout=600.0)
        body = r.json()
        if r.status_code == 200:
            return True, round(time.time() - t0, 1), ""
        return False, round(time.time() - t0, 1), str(body.get("error") or body)[:200]
    except Exception as e:
        return False, round(time.time() - t0, 1), str(e)[:200]


def generate(client: httpx.Client, model: str, steps: int, size: str) -> dict:
    payload = {"model": model, "prompt": PROMPT, "steps": steps, "size": size, "n": 1, "seed": SEED}
    t0 = time.time()
    peak = PeakVram()
    try:
        with peak:
            r = client.post(f"{WEB_URL}/api/sd/generate", json=payload, timeout=600.0)
        elapsed = time.time() - t0
        body = r.json()
        if r.status_code != 200:
            return {"ok": False, "error": str(body.get("error") or body)[:200], "elapsed_s": round(elapsed, 1)}
        items = body.get("data") or []
        b64 = (items[0].get("b64_json") if items else None) or body.get("b64_json")
        return {
            "ok": bool(b64),
            "elapsed_s": round(elapsed, 2),
            "it_per_s": round(steps / elapsed, 2) if elapsed > 0 else 0,
            "peak_vram_mb": peak.peak,
            "b64": b64,
            "error": None if b64 else "no image in response",
        }
    except Exception as e:
        return {"ok": False, "error": str(e)[:200], "elapsed_s": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep sd-server GPU layers x steps per image model.")
    ap.add_argument("--models", default="", help="comma-separated model names (default: all from /api/sd/models)")
    ap.add_argument("--gpu-layers", default=",".join(map(str, GPU_LAYERS_GRID)))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()

    layers_grid = [int(x) for x in args.gpu_layers.split(",") if x.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text())
        except Exception:
            results = {}

    def save() -> None:
        out_path.write_text(json.dumps(results, indent=2))

    with httpx.Client() as client:
        models = [m.strip() for m in args.models.split(",") if m.strip()] or list_models(client)
        if not models:
            print("[img] no image models found; pass --models explicitly")
            return
        print(f"[img] models: {models}")
        unload_audio(client)

        for model in models:
            print(f"\n[img] === {model} ===")
            model_res: dict = {"model": model, "configs": [], "winner": None}
            best = None
            for layers in layers_grid:
                touched = set_profile_gpu_layers(model, layers)
                print(f"[img] gpu_layers={layers} (profiles: {len(touched)})")
                ok, load_s, err = load_model(client, model)
                print(f"[img]   load: {'OK' if ok else 'FAIL'} in {load_s}s {err}")
                if not ok:
                    model_res["configs"].append({"gpu_layers": layers, "ok": False, "error": err})
                    continue
                for steps in steps_for(model):
                    print(f"[img]   steps={steps} size={SIZE} ... ", end="", flush=True)
                    g = generate(client, model, steps, SIZE)
                    if g.get("ok"):
                        print(f"{g['elapsed_s']}s  {g['it_per_s']} it/s  peak {g['peak_vram_mb']} MB")
                        fname = SAMPLES_DIR / f"{model.replace(':', '_')}_{layers}l_{steps}s.png"
                        try:
                            fname.write_bytes(base64.b64decode(g.pop("b64")))
                            g["sample"] = str(fname)
                        except Exception as e:
                            g["sample"] = f"save failed: {e}"
                    else:
                        print(f"FAIL ({g.get('error')})")
                        g.pop("b64", None)
                    cfg = {"gpu_layers": layers, "steps": steps, "size": SIZE, "load_s": load_s, **g}
                    model_res["configs"].append(cfg)
                    if g.get("ok") and (best is None or g["it_per_s"] > best["it_per_s"]):
                        best = cfg
            if best is not None:
                wl, ws = best["gpu_layers"], best["steps"]
                set_profile_gpu_layers(model, wl)
                load_model(client, model)
                print(
                    f"[img] winner: {wl} layers x {ws} steps ({best['it_per_s']} it/s); final {FINAL_SIZE} sample ..."
                )
                g = generate(client, model, ws, FINAL_SIZE)
                if g.get("ok") and g.get("b64"):
                    fname = SAMPLES_DIR / f"{model.replace(':', '_')}_WINNER_{wl}l_{ws}s_1024.png"
                    try:
                        fname.write_bytes(base64.b64decode(g.pop("b64")))
                        best["final_sample"] = str(fname)
                    except Exception as e:
                        best["final_sample"] = f"save failed: {e}"
                model_res["winner"] = best
                print(f"[img] WINNER {model}: {wl} layers, {ws} steps, {best['it_per_s']} it/s")
            else:
                print(f"[img] no passing config for {model}")
            results[model] = model_res
            save()

    print(f"\n[img] done -> {out_path} (samples in {SAMPLES_DIR})")


if __name__ == "__main__":
    main()
