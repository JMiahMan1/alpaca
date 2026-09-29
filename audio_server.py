#!/usr/bin/env python3
"""
audio_server.py

Alpaca audio generation server: voice (TTS) + music generation behind one
FastAPI service, sized for an RTX 4060 (8 GB shared with llama-server).

Voice   : Kokoro-82M  (hexgrad/Kokoro-82M, Apache-2.0) - 24 kHz narration.
          + optional OpenVoice V2 tone color (myshell-ai/OpenVoiceV2, MIT) for custom voices.
Music   : MusicGen    (facebook/musicgen-small, weights CC-BY-NC) - 32 kHz clips.

VRAM discipline (the card is shared with llama-server):
- Models load lazily on first use and are unloaded when the OTHER model is
  requested or after AUDIO_IDLE_UNLOAD_S seconds of inactivity.
- torch.cuda.empty_cache() after every generation so freed blocks return to
  the driver instead of being hoarded by the allocator.

Endpoints:
  GET  /health        -> status, loaded models, VRAM usage, voice list
  POST /api/tts       -> {text, voice?, speed?, sentence_pause_s?, paragraph_pause_s?, normalize?} -> wav b64
                         voice may blend several, e.g. "am_michael,am_fenrir"
                         normalize (default true) runs tts_text + audio/tts_lexicon.json
                         clone=<voice id>, clone_tau? re-timbres Kokoro into a custom voice
  POST /api/tts/normalize -> {text} -> {text, sentences}  preview of what will be spoken
  GET  /api/voices/prompts -> read-aloud script for recording a custom voice
  GET  /api/voices    -> custom voice profiles
  POST /api/voices    -> {name, consent, recordings: [{prompt_id, audio_b64}]} -> profile
  PATCH  /api/voices/<id> -> {name} rename (names are unique)
  DELETE /api/voices/<id>
  POST  /api/voices/identify   -> {audio_b64, threshold?} -> who is speaking
  POST  /api/voices/calibrate  -> {clips: [{voice_id, audio_b64}]} -> own vs impostor scores
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

import tts_text
import voice_clone

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
        if name == "tts" and (_state["tts"] is not None or voice_clone.loaded()):
            # The OpenVoice converter only runs alongside Kokoro, so it shares its lifetime.
            _state["tts"] = None
            voice_clone.unload()
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
            active = _state[name] is not None or (name == "tts" and voice_clone.loaded())
            if active and now - _state["last_used"][name] > IDLE_UNLOAD_S:  # type: ignore[index]
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
        "clone": {
            "engine": "openvoice-v2",
            "loaded": voice_clone.loaded(),
            "profiles": len(voice_clone.list_profiles()),
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


DEFAULT_SENTENCE_PAUSE_S = float(os.getenv("TTS_SENTENCE_PAUSE_S", "0.32"))
DEFAULT_PARAGRAPH_PAUSE_S = float(os.getenv("TTS_PARAGRAPH_PAUSE_S", "0.7"))

def _trim_and_fade(audio, sr: int, threshold: float = 0.004, margin_s: float = 0.03, fade_s: float = 0.008):
    """Trim edge silence to a consistent margin and apply short fades to avoid clicks."""
    import numpy as np

    loud = np.flatnonzero(np.abs(audio) > threshold)
    if loud.size == 0:
        return audio[:0]
    margin = int(margin_s * sr)
    audio = audio[max(loud[0] - margin, 0):min(loud[-1] + margin, len(audio))].copy()
    n = min(int(fade_s * sr), len(audio) // 2)
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        audio[:n] *= ramp
        audio[-n:] *= ramp[::-1]
    return audio


@app.post("/api/tts/normalize")
async def api_tts_normalize(request: Request):
    """Preview what the narrator will actually read (after lexicon + normalization)."""
    data = await request.json()
    text = str(data.get("text", ""))
    normalized = tts_text.normalize(text)
    return {"text": normalized, "sentences": [s for p in tts_text.paragraphs(normalized) for s in tts_text.sentences(p)]}


# --------------------------------------------------------------------------- #
# Custom voices (OpenVoice V2 tone color on top of Kokoro)                     #
# --------------------------------------------------------------------------- #


@app.get("/api/voices/prompts")
async def api_voice_prompts():
    """Read-aloud script for building a custom voice; every prompt is required."""
    return {"prompts": voice_clone.PROMPTS, "min_total_speech_s": voice_clone.MIN_TOTAL_SPEECH_S}


@app.get("/api/voices")
async def api_voices_list():
    return {"voices": voice_clone.list_profiles()}


@app.post("/api/voices")
async def api_voices_create(request: Request):
    """{name, consent: true, recordings: [{prompt_id, audio_b64}]} -> profile with quality report."""
    data = await request.json()
    if data.get("consent") is not True:
        return JSONResponse({"error": "confirm this is your voice or that the speaker agreed to be cloned"},
                            status_code=400)
    try:
        recordings = [(str(r.get("prompt_id", "")), base64.b64decode(r["audio_b64"]))
                      for r in data.get("recordings") or []]
    except Exception:
        return JSONResponse({"error": "recordings must be [{prompt_id, audio_b64}]"}, status_code=400)
    missing = [p["id"] for p in voice_clone.PROMPTS if p["id"] not in {pid for pid, _ in recordings}]
    if missing:
        return JSONResponse({"error": f"missing recordings for: {', '.join(missing)}"}, status_code=400)
    try:
        async with _lock:
            meta = await asyncio.to_thread(voice_clone.create_profile, str(data.get("name", "")), recordings)
            _state["last_used"]["tts"] = time.time()  # type: ignore[index]
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    except Exception as e:
        logger.exception("custom voice creation failed")
        return JSONResponse({"error": f"custom voice creation failed: {e}"}, status_code=500)
    finally:
        _empty_cache()
    return {"voice": meta}


@app.patch("/api/voices/{pid}")
async def api_voices_rename(pid: str, request: Request):
    """{name} -> renamed profile. Names are unique (case-insensitive)."""
    data = await request.json()
    try:
        meta = voice_clone.rename_profile(pid, str(data.get("name", "")))
    except KeyError:
        return JSONResponse({"error": f"unknown custom voice '{pid}'"}, status_code=404)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=409)
    return {"voice": meta}


@app.delete("/api/voices/{pid}")
async def api_voices_delete(pid: str):
    try:
        voice_clone.delete_profile(pid)
    except KeyError:
        return JSONResponse({"error": f"unknown custom voice '{pid}'"}, status_code=404)
    return {"deleted": pid}


# Identification lives beside the enrolment routes, not in the dashboard, because
# the comparison needs the reference encoder - which only exists in this
# container - and because Raven reaches it through the same tool surface.
#
# POST, not GET: the clip is audio. No route above claims POST /api/voices/{id},
# so these two paths are unambiguous.


def _clip_from(data: dict, key: str = "audio_b64") -> bytes:
    """Decode one base64 clip, with the same limits the enrolment path applies."""
    raw = data.get(key)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{key} must be a base64-encoded audio clip")
    try:
        blob = base64.b64decode(raw, validate=True)
    except Exception as e:
        raise ValueError(f"{key} is not valid base64: {e}") from e
    if not blob:
        raise ValueError(f"{key} decoded to zero bytes")
    if len(blob) > voice_clone.MAX_UPLOAD_BYTES:
        raise ValueError(f"{key} is {len(blob) // 1048576} MB; the limit is {voice_clone.MAX_UPLOAD_BYTES // 1048576} MB")
    return blob


@app.post("/api/voices/identify")
async def api_voices_identify(request: Request):
    """{audio_b64, threshold?} -> every enrolled voice ranked, plus a verdict.

    `threshold` is optional: without it the floor is derived from the enrolled
    cohort's own intra-speaker spread, so an install that has never calibrated
    still returns an honest answer.
    """
    data = await request.json()
    try:
        clip = _clip_from(data)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    threshold = data.get("threshold")
    if threshold is not None:
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            return JSONResponse({"error": "threshold must be a number between 0 and 1"}, status_code=400)
        if not 0.0 <= threshold <= 1.0:
            return JSONResponse({"error": f"threshold must be between 0 and 1, got {threshold}"},
                                status_code=400)
    try:
        async with _lock:
            result = await asyncio.to_thread(voice_clone.identify, clip, threshold)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    except Exception as e:
        logger.exception("voice identification failed")
        return JSONResponse({"error": f"voice identification failed: {e}"}, status_code=500)
    if not result.get("ok"):
        return JSONResponse(result, status_code=404 if "no enrolled voices" in str(result.get("error")) else 422)
    return result


@app.post("/api/voices/calibrate")
async def api_voices_calibrate(request: Request):
    """{clips: [{voice_id, audio_b64}]} -> own vs impostor scores, and a suggested threshold."""
    data = await request.json()
    raw_clips = data.get("clips")
    if not isinstance(raw_clips, list) or not raw_clips:
        return JSONResponse({"error": "clips must be a non-empty [{voice_id, audio_b64}] list"}, status_code=400)
    if len(raw_clips) > 20:
        return JSONResponse({"error": f"calibrate at most 20 clips at a time, got {len(raw_clips)}"}, status_code=400)
    clips = []
    for i, c in enumerate(raw_clips):
        if not isinstance(c, dict):
            return JSONResponse({"error": f"clip {i + 1} must be an object {{voice_id, audio_b64}}"}, status_code=400)
        try:
            clips.append((str(c.get("voice_id", "")), _clip_from(c)))
        except ValueError as e:
            return JSONResponse({"error": f"clip {i + 1}: {e}"}, status_code=400)
    try:
        async with _lock:
            result = await asyncio.to_thread(voice_clone.calibrate, clips)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    except Exception as e:
        logger.exception("voice calibration failed")
        return JSONResponse({"error": f"voice calibration failed: {e}"}, status_code=500)
    if not result.get("ok"):
        return JSONResponse(result, status_code=404)
    return result


@app.post("/api/tts")
async def api_tts(request: Request):
    data = await request.json()
    text = str(data.get("text", "")).strip()
    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)
    if len(text) > MAX_TTS_CHARS:
        return JSONResponse({"error": f"text exceeds {MAX_TTS_CHARS} chars"}, status_code=400)
    # A comma-separated list blends voices (Kokoro averages the style vectors),
    # e.g. "am_michael,am_fenrir" for a warmer, steadier narrator.
    voice = str(data.get("voice", "af_heart")).replace(" ", "")
    unknown = [v for v in voice.split(",") if v not in KOKORO_VOICES]
    if not voice or unknown:
        return JSONResponse({"error": f"unknown voice '{','.join(unknown) or voice}'"}, status_code=400)
    speed = float(data.get("speed", 1.0))
    if not 0.5 <= speed <= 2.0:
        return JSONResponse({"error": "speed must be within 0.5..2.0"}, status_code=400)
    sentence_pause = float(data.get("sentence_pause_s", DEFAULT_SENTENCE_PAUSE_S))
    paragraph_pause = float(data.get("paragraph_pause_s", DEFAULT_PARAGRAPH_PAUSE_S))
    if not (0.0 <= sentence_pause <= 3.0 and 0.0 <= paragraph_pause <= 5.0):
        return JSONResponse({"error": "pauses must be within 0..3s (sentence) and 0..5s (paragraph)"}, status_code=400)
    normalized = bool(data.get("normalize", True))
    if normalized:
        text = tts_text.normalize(text)
    # Optional custom voice: Kokoro speaks, OpenVoice re-timbres each sentence.
    clone_id = str(data.get("clone") or "").strip()
    clone_meta = None
    clone_tau = float(data.get("clone_tau", voice_clone.DEFAULT_TAU))
    if clone_id:
        try:
            clone_meta = voice_clone.get_profile(clone_id)
        except KeyError:
            return JSONResponse({"error": f"unknown custom voice '{clone_id}'"}, status_code=404)
        if not 0.1 <= clone_tau <= 1.0:
            return JSONResponse({"error": "clone_tau must be within 0.1..1.0"}, status_code=400)

    t0 = time.perf_counter()
    try:
        pipe = await _ensure_model("tts")
    except Exception as e:
        logger.exception("TTS load failed")
        return JSONResponse({"error": f"TTS model failed to load: {e}"}, status_code=503)

    try:
        import numpy as np

        sr = 24000
        pieces: list = []
        n_chunks = 0

        def _synth(sentence: str):
            out = []
            for result in pipe(sentence, voice=voice, speed=speed, split_pattern=None):
                audio = getattr(result, "audio", None)
                if audio is not None:
                    out.append(np.asarray(audio, dtype=np.float32))
            return out

        # Synthesize sentence by sentence so Kokoro never splits mid-sentence at
        # its token limit, then join with controlled pauses and short fades.
        # Holding _lock keeps the idle unloader / music eviction from pulling
        # the model out from under a long narration.
        async with _lock:
            convert = None
            if clone_meta:
                def _synth_source(script: str):
                    chunks = [np.asarray(r.audio, dtype=np.float32) for r in pipe(script, voice=voice, speed=1.0)
                              if getattr(r, "audio", None) is not None]
                    return np.concatenate(chunks), sr

                src_se = await asyncio.to_thread(voice_clone.source_se, voice, _synth_source)
                tgt_se = voice_clone.target_se(clone_id)

                def convert(a):
                    return voice_clone.convert(a, sr, src_se, tgt_se, clone_tau)

            for p_idx, paragraph in enumerate(tts_text.paragraphs(text)):
                for s_idx, sentence in enumerate(tts_text.sentences(paragraph)):
                    for audio in await asyncio.to_thread(_synth, sentence):
                        if convert is not None:
                            audio = await asyncio.to_thread(convert, audio)
                        audio = _trim_and_fade(audio, sr)
                        if audio.size == 0:
                            continue
                        if pieces:
                            gap = paragraph_pause if (s_idx == 0 and p_idx > 0) else sentence_pause
                            pieces.append(np.zeros(int(gap * sr), dtype=np.float32))
                        pieces.append(audio)
                        n_chunks += 1
            _state["last_used"]["tts"] = time.time()  # type: ignore[index]
        if not pieces:
            return JSONResponse({"error": "TTS produced no audio"}, status_code=502)
        merged = np.concatenate(pieces)
        # Pitch is not the converter's job: it moves timbre and leaves F0 alone,
        # so a clone lands on the *source voice's* pitch. Correct the merged
        # result once rather than each sentence - it is the same shift for all
        # of them, and one vocoder pass over the whole utterance is cheaper and
        # more consistent than one per chunk. See voice_clone.correct_pitch.
        clone_meta_out = None
        if clone_meta:
            target_f0 = clone_meta.get("median_f0_hz")
            try:
                merged, pitch_report = await asyncio.to_thread(
                    voice_clone.correct_pitch, merged, sr, target_f0
                )
            except Exception as exc:
                # `correct_pitch` is total by contract, but the audio is already
                # made and a caller must never lose it to a measurement failure.
                logger.warning(f"[audio] pitch correction raised: {exc}")
                pitch_report = {"corrected": False, "reason": f"pitch correction failed: {exc}"}
            try:
                similarity = await asyncio.to_thread(
                    voice_clone.clone_similarity, merged, sr, clone_id
                )
            except Exception as exc:
                logger.warning(f"[audio] clone similarity raised: {exc}")
                similarity = None
            spread = clone_meta.get("intraspeaker_spread")
            clone_meta_out = {
                "id": clone_id,
                "name": clone_meta["name"],
                "tau": clone_tau,
                "pitch": pitch_report,
                # Measured, not asserted: how close the result sounds to the
                # enrolled voice, next to how close the speaker's own
                # re-recordings sound to each other. A listener who thinks the
                # clone is wrong can read which half of that pair is at fault.
                "similarity": similarity,
                "intraspeaker_spread": spread,
                "as_close_as_their_own_re_recordings": (
                    None if similarity is None or spread is None else similarity >= spread
                ),
                "note": (
                    "OpenVoice transfers tone colour onto a Kokoro voice. It "
                    "reproduces timbre and (after pitch correction) pitch, not "
                    "the speaker's accent, rhythm or phrasing."
                ),
            }
        # Lead-in/out padding so players don't clip the first/last syllable.
        pad = np.zeros(int(0.15 * sr), dtype=np.float32)
        merged = np.concatenate([pad, merged, pad])
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
                "chunks": n_chunks,
                "sentence_pause_s": sentence_pause,
                "paragraph_pause_s": paragraph_pause,
                "normalized": normalized,
                "clone": clone_meta_out,
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
