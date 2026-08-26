#!/usr/bin/env python3
"""
audio_server.py

Alpaca audio generation server: voice (TTS) + music generation behind one
FastAPI service, sized for an RTX 4060 (8 GB shared with llama-server).

Voice   : Kokoro-82M  (hexgrad/Kokoro-82M, Apache-2.0) - 24 kHz narration.
Music   : MusicGen    (facebook/musicgen-small, weights CC-BY-NC) - 32 kHz clips.

VRAM discipline (the card is shared with llama-server):
- Models load lazily on first use and are unloaded when the OTHER model is
  requested or after AUDIO_IDLE_UNLOAD_S seconds of inactivity.
- torch.cuda.empty_cache() after every generation so freed blocks return to
  the driver instead of being hoarded by the allocator.

Endpoints:
  GET  /health        -> status, loaded models, VRAM usage, voice list
  POST /api/tts       -> {text, voice?, speed?, lang?}          -> wav b64
  POST /api/music     -> {prompt, duration_s?, temperature?, guidance_scale?, seed?, top_k?} -> wav b64
  POST /api/unload    -> free all VRAM immediately
"""

import asyncio
import base64
import io
import logging
import os
import time
import wave

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("audio_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

IDLE_UNLOAD_S = int(os.getenv("AUDIO_IDLE_UNLOAD_S", "180"))
DEVICE = os.getenv("AUDIO_DEVICE", "cuda")
TTS_MODEL_ID = os.getenv("TTS_MODEL_ID", "hexgrad/Kokoro-82M")
MUSIC_MODEL_ID = os.getenv("MUSIC_MODEL_ID", "facebook/musicgen-small")
MAX_MUSIC_SECONDS = float(os.getenv("AUDIO_MAX_MUSIC_S", "30"))
MAX_TTS_CHARS = int(os.getenv("AUDIO_MAX_TTS_CHARS", "4000"))

KOKORO_VOICES = [
    "af_heart",
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
]

MUSIC_PRESETS = [
    "lo-fi hip hop beat with warm vinyl crackle, mellow keys, relaxed drums",
    "upbeat synthwave chase theme, driving arpeggios, retro 80s lead",
    "epic orchestral boss battle, pounding timpani, brass fanfare",
    "gentle acoustic folk loop, fingerpicked guitar, soft shaker",
    "dark ambient dungeon crawler drone, distant echoes, low pulses",
    "chiptune arcade boss fight, square leads, fast snare rolls",
]

_state: dict[str, object] = {
    "tts": None,  # KPipeline instance once loaded
    "music": None,  # {"model": ..., "processor": ...} once loaded
    "loading": set(),  # model names currently loading
    "last_used": {"tts": 0.0, "music": 0.0},
}
_lock = asyncio.Lock()

app_start = time.time()
app = FastAPI(title="alpaca-audio-server")


def _torch():
    import torch

    return torch


def _device() -> str:
    t = _torch()
    if DEVICE == "cuda" and not t.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    return DEVICE


def _wav_bytes(samples_f32, sample_rate: int) -> bytes:
    """Encode a mono float32 numpy array (-1..1) as a 16-bit PCM WAV."""
    import numpy as np

    arr = np.asarray(samples_f32, dtype=np.float32)
    peak = float(np.max(np.abs(arr))) if arr.size else 1.0
    if peak > 0:
        arr = arr / max(peak, 1e-6)
    pcm16 = (arr * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def _free_vram_mb() -> tuple[int | None, int | None]:
    try:
        t = _torch()
        if not t.cuda.is_available():
            return None, None
        free_b, total_b = t.cuda.mem_get_info()
        return int(free_b // (1024 * 1024)), int(total_b // (1024 * 1024))
    except Exception as e:  # pragma: no cover - diagnostics only
        logger.debug(f"mem_get_info failed: {e}")
        return None, None


def _empty_cache() -> None:
    try:
        t = _torch()
        if t.cuda.is_available():
            t.cuda.empty_cache()
    except Exception as e:  # pragma: no cover
        logger.debug(f"empty_cache failed: {e}")


async def _unload_model(name: str) -> bool:
    """Drop a loaded model and release its CUDA memory. Returns True if it was loaded."""
    async with _lock:
        if name == "tts" and _state["tts"] is not None:
            _state["tts"] = None
        elif name == "music" and _state["music"] is not None:
            _state["music"] = None
        else:
            return False
        await asyncio.to_thread(_empty_cache)
        logger.info(f"[audio] unloaded {name}, VRAM returned to driver")
        return True


async def _ensure_model(name: str):
    """Load tts/music into memory, evicting the other model first."""
    if name == "tts":
        if _state["tts"] is not None:
            last_used: dict[str, float] = _state["last_used"]  # type: ignore[assignment]
            last_used["tts"] = time.time()
            return _state["tts"]
    else:
        if _state["music"] is not None:
            last_used2: dict[str, float] = _state["last_used"]  # type: ignore[assignment]
            last_used2["music"] = time.time()
            return _state["music"]

    other = "music" if name == "tts" else "tts"
    evicted = await _unload_model(other)

    loading: set[str] = _state["loading"]  # type: ignore[assignment]
    if name in loading:
        raise RuntimeError(f"{name} model is already loading")
    loading.add(name)
    logger.info(f"[audio] loading {name} ({'evicted ' + other + ' first, ' if evicted else ''}device={_device()}) ...")
    try:
        if name == "tts":

            def _load_tts():
                from kokoro import KPipeline

                return KPipeline(lang_code="a", repo_id=TTS_MODEL_ID)  # 'a' = American English

            _state["tts"] = await asyncio.to_thread(_load_tts)
            last_used3: dict[str, float] = _state["last_used"]  # type: ignore[assignment]
            last_used3["tts"] = time.time()
            return _state["tts"]

        def _load_music():
            from transformers import AutoProcessor, MusicgenForConditionalGeneration

            proc = AutoProcessor.from_pretrained(MUSIC_MODEL_ID)
            model = MusicgenForConditionalGeneration.from_pretrained(
                MUSIC_MODEL_ID,
                torch_dtype="float16" if _device() == "cuda" else "float32",
            ).to(_device())
            return {"model": model, "processor": proc}

        _state["music"] = await asyncio.to_thread(_load_music)
        last_used4: dict[str, float] = _state["last_used"]  # type: ignore[assignment]
        last_used4["music"] = time.time()
        return _state["music"]
    finally:
        loading.discard(name)


async def _idle_unloader() -> None:
    while True:
        await asyncio.sleep(15)
        now = time.time()
        for name in ("tts", "music"):
            if _state[name] is not None and now - _state["last_used"][name] > IDLE_UNLOAD_S:  # type: ignore[index]
                logger.info(f"[audio] idle timeout ({IDLE_UNLOAD_S}s) -> unloading {name}")
                await _unload_model(name)


_idle_unload_task: asyncio.Task | None = None


@app.on_event("startup")
async def _startup() -> None:
    global _idle_unload_task
    _idle_unload_task = asyncio.create_task(_idle_unloader())
    logger.info("[audio] audio-server up (models load lazily on first use)")


# --------------------------------------------------------------------------- #
# Health                                                                       #
# --------------------------------------------------------------------------- #


@app.get("/health")
@app.get("/api/status")
async def health():
    free_mb, total_mb = _free_vram_mb()
    return {
        "status": "ok",
        "service": "audio-server",
        "uptime_s": round(time.time() - app_start, 1),
        "tts": {
            "model": TTS_MODEL_ID,
            "loaded": _state["tts"] is not None,
            "voices": KOKORO_VOICES,
        },
        "music": {
            "model": MUSIC_MODEL_ID,
            "loaded": _state["music"] is not None,
            "max_duration_s": MAX_MUSIC_SECONDS,
            "presets": MUSIC_PRESETS,
        },
        "vram_free_mb": free_mb,
        "vram_total_mb": total_mb,
    }


# --------------------------------------------------------------------------- #
# Voice / TTS                                                                  #
# --------------------------------------------------------------------------- #


@app.post("/api/tts")
async def api_tts(request: Request):
    data = await request.json()
    text = str(data.get("text", "")).strip()
    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)
    if len(text) > MAX_TTS_CHARS:
        return JSONResponse({"error": f"text exceeds {MAX_TTS_CHARS} chars"}, status_code=400)
    voice = str(data.get("voice", "af_heart"))
    if voice not in KOKORO_VOICES:
        return JSONResponse({"error": f"unknown voice '{voice}'"}, status_code=400)
    speed = float(data.get("speed", 1.0))
    if not 0.5 <= speed <= 2.0:
        return JSONResponse({"error": "speed must be within 0.5..2.0"}, status_code=400)

    t0 = time.perf_counter()
    try:
        pipe = await _ensure_model("tts")
    except Exception as e:
        logger.exception("TTS load failed")
        return JSONResponse({"error": f"TTS model failed to load: {e}"}, status_code=503)

    try:
        import numpy as np

        chunks: list = []
        sr = 24000
        for result in pipe(text, voice=voice, speed=speed):
            audio = getattr(result, "audio", None)
            if audio is None:
                continue
            chunks.append(np.asarray(audio, dtype=np.float32))
            rate = getattr(audio, "sample_rate", None)
            if isinstance(rate, (int, float)) and int(rate) > 0:
                sr = int(rate)
        if not chunks:
            return JSONResponse({"error": "TTS produced no audio"}, status_code=502)
        merged = np.concatenate(chunks)
        elapsed = time.perf_counter() - t0
        duration = len(merged) / float(sr)
        wav = await asyncio.to_thread(_wav_bytes, merged, sr)
        b64 = base64.b64encode(wav).decode("ascii")
        _empty_cache()
        return {
            "audio_b64": b64,
            "mime": "audio/wav",
            "meta": {
                "engine": "kokoro",
                "model": TTS_MODEL_ID,
                "voice": voice,
                "speed": speed,
                "duration_s": round(duration, 2),
                "sample_rate": sr,
                "elapsed_s": round(elapsed, 2),
                "rtf": round(elapsed / max(duration, 1e-6), 4),
                "chars": len(text),
                "chunks": len(chunks),
            },
        }
    except Exception as e:
        logger.exception("TTS generation failed")
        return JSONResponse({"error": f"TTS generation failed: {e}"}, status_code=500)


# --------------------------------------------------------------------------- #
# Music                                                                        #
# --------------------------------------------------------------------------- #


@app.post("/api/music")
async def api_music(request: Request):
    data = await request.json()
    prompt = str(data.get("prompt", data.get("tags", ""))).strip()
    if not prompt:
        return JSONResponse({"error": "prompt is required"}, status_code=400)
    duration_s = float(data.get("duration_s", 10))
    if not 2 <= duration_s <= MAX_MUSIC_SECONDS:
        return JSONResponse(
            {"error": f"duration_s must be within 2..{MAX_MUSIC_SECONDS:g}"},
            status_code=400,
        )
    temperature = float(data.get("temperature", 1.0))
    guidance = float(data.get("guidance_scale", 3.0))
    seed = data.get("seed")

    t0 = time.perf_counter()
    try:
        bundle = await _ensure_model("music")
    except Exception as e:
        logger.exception("Music model load failed")
        return JSONResponse({"error": f"music model failed to load: {e}"}, status_code=503)

    try:
        import numpy as np
        import torch

        assert isinstance(bundle, dict)
        model = bundle["model"]
        proc = bundle["processor"]

        gen_kwargs: dict = {
            "do_sample": True,
            "top_k": int(data.get("top_k", 250)),
        }
        if seed is not None:
            torch.manual_seed(int(seed))

        inputs = proc(text=[prompt], padding=True, return_tensors="pt").to(_device())
        max_tokens = min(int(duration_s * 50), 1500)
        sr = model.config.audio_encoder.sampling_rate
        with torch.no_grad():
            audio = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                guidance_scale=guidance,
                **gen_kwargs,
            )
        arr = audio[0, 0].float().cpu().numpy().astype(np.float32)
        elapsed = time.perf_counter() - t0
        actual_dur = len(arr) / float(sr)
        wav = await asyncio.to_thread(_wav_bytes, arr, sr)
        b64 = base64.b64encode(wav).decode("ascii")
        _empty_cache()
        return {
            "audio_b64": b64,
            "mime": "audio/wav",
            "meta": {
                "engine": "musicgen",
                "model": MUSIC_MODEL_ID,
                "prompt": prompt[:200],
                "requested_duration_s": duration_s,
                "duration_s": round(actual_dur, 2),
                "sample_rate": sr,
                "tokens": max_tokens,
                "seed": seed,
                "temperature": temperature,
                "guidance_scale": guidance,
                "elapsed_s": round(elapsed, 2),
                "rtf": round(elapsed / max(actual_dur, 1e-6), 4),
            },
        }
    except Exception as e:
        logger.exception("Music generation failed")
        return JSONResponse({"error": f"music generation failed: {e}"}, status_code=500)


@app.post("/api/unload")
async def api_unload():
    a = await _unload_model("tts")
    b = await _unload_model("music")
    return {"unloaded": [n for n, ok in (("tts", a), ("music", b)) if ok], "freed": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("AUDIO_PORT", "8082")))
