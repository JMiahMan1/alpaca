#!/usr/bin/env python3
"""benchmark-audio.py - performance baselines for the audio-server.

Audio models (Kokoro-82M TTS, MusicGen-small) have almost no launch-time
tunables, so this script measures rather than sweeps:

  * TTS cold start (first request after /api/unload, includes model load)
  * TTS warm latency/RTF x3 (same 400-char passage, default voice)
  * MusicGen 10s clip x2 (fixed seed for comparability)
  * VRAM footprint of each engine (nvidia-smi before/after)
  * Idle-unload verification (models should drop after AUDIO_IDLE_UNLOAD_S)

Usage:
    python3 benchmark-audio.py [--out ...] [--skip-idle-wait]

Results -> data/audio_bench/audio_bench.json. The "optimization" output is
the contention policy: how much VRAM each engine pins while loaded, which
is what the LLM/image sweeps must budget around on the 8 GB card.
"""

import argparse
import contextlib
import json
import subprocess
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent
AUDIO_URL = "http://localhost:8082"
OUT_DEFAULT = REPO / "data" / "audio_bench" / "audio_bench.json"

TTS_TEXT = (
    "The transformer architecture replaced recurrent networks with self-attention, "
    "enabling parallel training and Gardens of forking paths in language modeling. "
    "On a single eight-gigabyte graphics card, every megabyte of video memory is "
    "budgeted between model weights, key-value cache, and compute scratch space. "
    "This passage measures text to speech latency. The quick brown fox jumps over "
    "the lazy dog while the benchmark clock keeps ticking in the background."
)
MUSIC_PROMPT = "lo-fi hip hop beat, warm Rhodes piano, vinyl crackle, 90 bpm"
MUSIC_DURATION_S = 10
MUSIC_SEED = 7
IDLE_POLL_S = 30
IDLE_TIMEOUT_S = 300


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


def health(client: httpx.Client) -> dict:
    try:
        return client.get(f"{AUDIO_URL}/health", timeout=10.0).json()
    except Exception as e:
        return {"status": "unreachable", "error": str(e)[:120]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure audio-server TTS/music latency, RTF, and VRAM footprint.")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--skip-idle-wait", action="store_true", help="skip the idle-unload verification wait")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res: dict = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "tts": {}, "music": {}, "vram": {}, "idle": {}}

    with httpx.Client() as client:
        h0 = health(client)
        print(
            f"[aud] health: {h0.get('status')} (tts loaded={((h0.get('tts') or {}).get('loaded'))}, "
            f"music loaded={((h0.get('music') or {}).get('loaded'))})"
        )
        if h0.get("status") != "ok":
            res["error"] = h0
            out_path.write_text(json.dumps(res, indent=2))
            print("[aud] audio-server unreachable, aborting")
            return

        # Cold TTS: force unload first so the first request includes model load.
        try:
            client.post(f"{AUDIO_URL}/api/unload", timeout=60.0)
            print("[aud] unloaded both engines for cold-start measurement")
        except Exception as e:
            print(f"[aud] unload skipped ({str(e)[:80]})")
        time.sleep(3)

        v_before = vram_used_mb()
        t0 = time.time()
        cold = client.post(f"{AUDIO_URL}/api/tts", json={"text": TTS_TEXT}, timeout=600.0).json()
        cold_wall = round(time.time() - t0, 2)
        v_after_cold = vram_used_mb()
        cold_meta = cold.get("meta") or {}
        res["tts"]["cold"] = {
            "wall_s": cold_wall,
            "elapsed_s": cold_meta.get("elapsed_s"),
            "rtf": cold_meta.get("rtf"),
            "duration_s": cold_meta.get("duration_s"),
            "chars": cold_meta.get("chars"),
            "error": cold.get("error"),
        }
        print(
            f"[aud] TTS cold: wall {cold_wall}s, engine {cold_meta.get('elapsed_s')}s, "
            f"rtf {cold_meta.get('rtf')} (vram {v_before}->{v_after_cold} MB)"
        )

        # Warm TTS x3.
        warms = []
        for i in range(1, 4):
            r = client.post(f"{AUDIO_URL}/api/tts", json={"text": TTS_TEXT}, timeout=600.0).json()
            m = r.get("meta") or {}
            warms.append({"elapsed_s": m.get("elapsed_s"), "rtf": m.get("rtf"), "error": r.get("error")})
            print(f"[aud] TTS warm {i}/3: {m.get('elapsed_s')}s, rtf {m.get('rtf')}")
        res["tts"]["warm"] = warms
        v_after_warm = vram_used_mb()

        # Music x2 (fixed seed).
        clips = []
        for i in range(1, 3):
            r = client.post(
                f"{AUDIO_URL}/api/music",
                json={"prompt": MUSIC_PROMPT, "duration_s": MUSIC_DURATION_S, "seed": MUSIC_SEED},
                timeout=900.0,
            ).json()
            m = r.get("meta") or {}
            clips.append(
                {
                    "elapsed_s": m.get("elapsed_s"),
                    "rtf": m.get("rtf"),
                    "duration_s": m.get("duration_s"),
                    "tokens": m.get("tokens"),
                    "error": r.get("error"),
                }
            )
            print(f"[aud] music {i}/2: {m.get('elapsed_s')}s for {m.get('duration_s')}s audio, rtf {m.get('rtf')}")
        res["music"]["clips"] = clips
        v_after_music = vram_used_mb()
        res["vram"] = {
            "before_mb": v_before,
            "after_tts_cold_mb": v_after_cold,
            "after_tts_warm_mb": v_after_warm,
            "after_music_mb": v_after_music,
        }

        # Idle-unload verification: engines should drop after AUDIO_IDLE_UNLOAD_S.
        if not args.skip_idle_wait:
            print(f"[aud] waiting for idle unload (poll every {IDLE_POLL_S}s, up to {IDLE_TIMEOUT_S}s) ...")
            t_start = time.time()
            unloaded_at = None
            while time.time() - t_start < IDLE_TIMEOUT_S:
                time.sleep(IDLE_POLL_S)
                h = health(client)
                tts_on = ((h.get("tts") or {}).get("loaded")) is True
                mus_on = ((h.get("music") or {}).get("loaded")) is True
                if not tts_on and not mus_on:
                    unloaded_at = round(time.time() - t_start, 1)
                    break
            res["idle"] = {
                "both_unloaded": unloaded_at is not None,
                "unloaded_after_s": unloaded_at,
                "final_vram_mb": vram_used_mb(),
            }
            print(f"[aud] idle unload: {'OK in ' + str(unloaded_at) + 's' if unloaded_at else 'NOT observed'}")
        else:
            res["idle"] = {"skipped": True}

        # Leave the box clean: drop audio models so the LLM gets its VRAM back.
        with contextlib.suppress(Exception):
            client.post(f"{AUDIO_URL}/api/unload", timeout=60.0)

    out_path.write_text(json.dumps(res, indent=2))
    print(f"[aud] done -> {out_path}")


if __name__ == "__main__":
    main()
