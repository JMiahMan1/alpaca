"""
voice_clone.py

Custom narrator voices for Alpaca's audio-server: Kokoro speaks, then the
OpenVoice V2 tone-color converter (myshell-ai/OpenVoiceV2, MIT) re-timbres
the speech toward a recorded speaker. Kokoro keeps pronunciation and pacing;
OpenVoice only changes how the voice sounds.

A profile is built from a few short read-aloud recordings (see PROMPTS). Each
recording is decoded with ffmpeg, stripped of silence, quality-checked, and
reduced to a 256-d speaker embedding ("SE") averaged over ~10 s windows.
Converting also needs the SE of the Kokoro voice being converted *from*;
that is computed once per Kokoro voice by synthesizing the same prompts and
cached next to the profiles.

Storage (VOICES_DIR, a docker volume):
  <id>/meta.json          name, created, per-recording quality report
  <id>/se.pt              target speaker embedding
  <id>/recordings/NN.wav  cleaned 22.05 kHz mono takes (kept for re-extraction)
  _sources/<voice>.pt     cached Kokoro source embeddings

Converted audio carries OpenVoice's inaudible WavMark watermark, which marks
it as synthetic.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time

logger = logging.getLogger("voice_clone")

VOICES_DIR = os.getenv("VOICES_DIR", "/data/voices")
CONVERTER_REPO = os.getenv("OPENVOICE_REPO_ID", "myshell-ai/OpenVoiceV2")
DEFAULT_TAU = float(os.getenv("OPENVOICE_TAU", "0.3"))
SR = 22050  # OpenVoice V2 converter sample rate
SE_WINDOW_S = 10.0
MIN_TOTAL_SPEECH_S = 30.0
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# Read-aloud script. Three takes, ~20-40 s each, cover the sounds of English
# and a range of intonation, which is what the tone-color embedding needs.
PROMPTS = [
    {
        "id": "rainbow",
        "title": "1 · The Rainbow Passage",
        "why": "A classic phonetically balanced passage used by speech clinicians. It covers nearly every English sound.",
        "tip": "Read at your normal narration pace, as if speaking to a room.",
        "min_s": 15,
        "text": (
            "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. "
            "The rainbow is a division of white light into many beautiful colors. These take the shape "
            "of a long round arch, with its path high above, and its two ends apparently beyond the horizon. "
            "There is, according to legend, a boiling pot of gold at one end. People look, but no one ever "
            "finds it. When a man looks for something beyond his reach, his friends say he is looking for "
            "the pot of gold at the end of the rainbow."
        ),
    },
    {
        "id": "harvard",
        "title": "2 · Clear sentences",
        "why": "Harvard sentences (IEEE) are short and packed with contrasting consonants, so the embedding learns crisp articulation.",
        "tip": "Pause briefly between sentences. Keep the same distance from the mic.",
        "min_s": 12,
        "text": (
            "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background. "
            "It's easy to tell the depth of a well. These days a chicken leg is a rare dish. "
            "Rice is often served in round bowls. The juice of lemons makes fine punch. "
            "The box was thrown beside the parked truck. Four hours of steady work faced us. "
            "A large size in stockings is hard to sell. The boy was there when the sun rose."
        ),
    },
    {
        "id": "expressive",
        "title": "3 · Natural expression",
        "why": "Questions, emphasis, and numbers capture your pitch range and how your voice rises and falls.",
        "tip": "Read it like you mean it: let questions rise and let the exclamation land.",
        "min_s": 12,
        "text": (
            "Good evening, everyone, and thank you for being here. Have you ever wondered how a small idea "
            "grows into something that changes thousands of lives? In eighteen ninety-five, a few people "
            "rented a simple hall and opened the doors to anyone who would come. Was it easy? Not at all! "
            "But they kept going, one week at a time, and that is the story I want to tell you tonight."
        ),
    },
]

# Source-embedding script for Kokoro voices: the same material the speaker read.
_SOURCE_SCRIPT = " ".join(p["text"] for p in PROMPTS)

_state: dict = {"converter": None}
_load_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #


def _converter():
    """Load the OpenVoice V2 tone-color converter (downloads once to the HF cache)."""
    if _state["converter"] is not None:
        return _state["converter"]
    with _load_lock:
        if _state["converter"] is None:
            import torch
            from huggingface_hub import hf_hub_download
            from openvoice.api import ToneColorConverter

            cfg = hf_hub_download(CONVERTER_REPO, "converter/config.json")
            ckpt = hf_hub_download(CONVERTER_REPO, "converter/checkpoint.pth")
            device = "cuda:0" if torch.cuda.is_available() and os.getenv("AUDIO_DEVICE", "cuda") == "cuda" else "cpu"
            # Watermarking is on by default (passing enable_watermark trips the base class).
            conv = ToneColorConverter(cfg, device=device)
            conv.load_ckpt(ckpt)
            _state["converter"] = conv
            logger.info(f"[voice_clone] OpenVoice V2 converter loaded on {device}")
    return _state["converter"]


def loaded() -> bool:
    return _state["converter"] is not None


def unload() -> bool:
    was = _state["converter"] is not None
    _state["converter"] = None
    return was


def _resample(audio, sr_from: int, sr_to: int):
    import numpy as np
    import torch
    import torchaudio.functional as AF

    if sr_from == sr_to:
        return np.asarray(audio, dtype=np.float32)
    t = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    return AF.resample(t, sr_from, sr_to).numpy()


def _spec(conv, audio_22k):
    import torch
    from openvoice.mel_processing import spectrogram_torch

    hps = conv.hps
    y = torch.from_numpy(audio_22k).float().to(conv.device).unsqueeze(0)
    return spectrogram_torch(y, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length,
                             hps.data.win_length, center=False).to(conv.device)


def _embed_windows(windows_22k: list):
    """One OpenVoice reference-encoder embedding per usable window.

    Kept separate from `_embed` so a profile can retain the individual window
    embeddings: the spread *between* a speaker's own windows is the only honest
    yardstick for how similar two recordings have to be to be the same person.
    """
    import torch

    conv = _converter()
    gs = []
    with torch.no_grad():
        for w in windows_22k:
            if len(w) < SR:  # < 1 s carries too little timbre
                continue
            gs.append(conv.model.ref_enc(_spec(conv, w).transpose(1, 2)).unsqueeze(-1).cpu())
    if not gs:
        raise ValueError("not enough speech to build a voice embedding")
    return gs


def _embed(windows_22k: list):
    """Average OpenVoice reference-encoder embeddings over audio windows."""
    import torch

    return torch.stack(_embed_windows(windows_22k)).mean(0).cpu()


def _cosine(a, b) -> float:
    """Cosine similarity of two embeddings, as a plain float (no torch needed)."""
    import numpy as np

    x = np.asarray(_as_f32(a), dtype="float64").ravel()
    y = np.asarray(_as_f32(b), dtype="float64").ravel()
    if x.size != y.size or x.size == 0:
        return 0.0
    d = float(np.linalg.norm(x) * np.linalg.norm(y))
    return 0.0 if d <= 1e-12 else float(np.dot(x, y) / d)


def _as_f32(t):
    """numpy view of a tensor, or the input unchanged if it is already one."""
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy()
    return t


def _pairwise_spread(vectors: list) -> float:
    """Mean (1 - cosine) over every pair. 0.0 for fewer than two vectors."""
    if len(vectors) < 2:
        return 0.0
    tot = n = 0.0
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            tot += 1.0 - _cosine(vectors[i], vectors[j])
            n += 1
    return tot / n if n else 0.0


# The identification threshold is derived from the enrolled cohort rather than
# picked, because "0.82" means nothing on its own. Every profile records how far
# its own takes are from each other; a match has to beat the *worst* speaker's
# own variation by a margin, or the same person re-recorded on a different day
# would read as a stranger. The clamp keeps a single wildly-inconsistent
# enrolment from making the threshold unusable, and a floor keeps a suspiciously
# tight cohort from making it trivially low.
SPREAD_MARGIN = 1.5
MIN_IDENTITY_THRESHOLD = 0.05
MAX_IDENTITY_THRESHOLD = 0.95


def identity_threshold(spreads: list[float]) -> float:
    """Cohort-derived cosine floor above which two recordings are the same voice."""
    real = [s for s in spreads if s is not None]
    if not real:
        return 0.75
    return max(MIN_IDENTITY_THRESHOLD, min(MAX_IDENTITY_THRESHOLD, SPREAD_MARGIN * max(real)))


def convert(audio, sr: int, src_se, tgt_se, tau: float = DEFAULT_TAU):
    """Re-timbre one utterance from src_se toward tgt_se; returns audio at `sr`."""
    import torch

    conv = _converter()
    x = _resample(audio, sr, SR)
    with torch.no_grad():
        spec = _spec(conv, x)
        lengths = torch.LongTensor([spec.size(-1)]).to(conv.device)
        y = conv.model.voice_conversion(spec, lengths, sid_src=src_se.to(conv.device),
                                        sid_tgt=tgt_se.to(conv.device), tau=tau)[0][0, 0]
        y = y.data.cpu().float().numpy()
    y = conv.add_watermark(y, "alpaca") if len(y) >= 16000 else y
    return _resample(y, SR, sr)


# --------------------------------------------------------------------------- #
# Recordings                                                                   #
# --------------------------------------------------------------------------- #


def decode_to_22k(raw: bytes):
    """Decode any browser/phone recording (webm, ogg, m4a, mp3, wav) to 22.05 kHz mono float32."""
    import numpy as np

    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("recording is larger than 25 MB")
    with tempfile.NamedTemporaryFile(suffix=".bin") as f:
        f.write(raw)
        f.flush()
        # Gentle highpass removes rumble/handling noise before analysis.
        p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", f.name, "-af", "highpass=f=70",
                            "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, timeout=120)
    if p.returncode != 0 or not p.stdout:
        raise ValueError("could not decode the recording: " + p.stderr.decode(errors="ignore")[-200:])
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def _frames_db(audio, frame: int):
    import numpy as np

    n = len(audio) // frame
    if n == 0:
        return np.array([-120.0])
    fr = audio[: n * frame].reshape(n, frame)
    return 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-9)


def analyze(audio) -> dict:
    """Speech-only audio + a quality report (duration, loudness, clipping, SNR estimate)."""
    import numpy as np

    frame = int(0.03 * SR)
    db = _frames_db(audio, frame)
    noise_db = float(np.percentile(db, 10))
    speech_ref = float(np.percentile(db, 95))
    # Frames well above the noise floor count as speech; keep 150 ms of hangover
    # so word endings aren't clipped.
    thresh = max(noise_db + 10, speech_ref - 35)
    voiced = db > thresh
    hang = int(0.15 / 0.03)
    keep = np.convolve(voiced.astype(np.int32), np.ones(2 * hang + 1, dtype=np.int32), mode="same") > 0
    speech = audio[: len(keep) * frame].reshape(len(keep), frame)[keep].reshape(-1)
    clip_ratio = float((np.abs(audio) > 0.99).mean()) if len(audio) else 0.0
    report = {
        "duration_s": round(len(audio) / SR, 1),
        "speech_s": round(len(speech) / SR, 1),
        "level_db": round(speech_ref, 1),
        "noise_db": round(noise_db, 1),
        "snr_db": round(speech_ref - noise_db, 1),
        "clipping_pct": round(100 * clip_ratio, 2),
        "warnings": [],
    }
    if report["snr_db"] < 20:
        report["warnings"].append("background noise is noticeable; a quieter room will improve the match")
    if speech_ref < -38:
        report["warnings"].append("recording is quiet; move closer to the mic (15-30 cm)")
    if clip_ratio > 0.002:
        report["warnings"].append("recording clips (too loud); move back a little or lower mic gain")
    return {"speech": speech.astype(np.float32), "report": report}


def _windows(speech, seconds: float = SE_WINDOW_S) -> list:
    n = int(seconds * SR)
    return [speech[i:i + n] for i in range(0, len(speech), n)]


# --------------------------------------------------------------------------- #
# Profiles                                                                     #
# --------------------------------------------------------------------------- #


def _pdir(pid: str) -> str:
    if not re.fullmatch(r"[a-z0-9-]{3,64}", pid or ""):
        raise KeyError(pid)
    return os.path.join(VOICES_DIR, pid)


def list_profiles() -> list[dict]:
    out = []
    if not os.path.isdir(VOICES_DIR):
        return out
    for pid in sorted(os.listdir(VOICES_DIR)):
        meta = os.path.join(VOICES_DIR, pid, "meta.json")
        if not pid.startswith("_") and os.path.isfile(meta):
            try:
                with open(meta, encoding="utf-8") as fh:
                    out.append(json.load(fh))
            except Exception as e:
                logger.warning(f"[voice_clone] unreadable profile {pid}: {e}")
    return sorted(out, key=lambda m: m.get("created", 0), reverse=True)


def get_profile(pid: str) -> dict:
    meta = os.path.join(_pdir(pid), "meta.json")
    if not os.path.isfile(meta):
        raise KeyError(pid)
    return json.load(open(meta, encoding="utf-8"))


def delete_profile(pid: str) -> None:
    d = _pdir(pid)
    if not os.path.isfile(os.path.join(d, "meta.json")):
        raise KeyError(pid)
    shutil.rmtree(d)


def _check_name(name: str, exclude_id: str | None = None) -> str:
    """Names are how people pick a voice, so they must be unique (case-insensitive)."""
    name = re.sub(r"\s+", " ", (name or "").strip())[:60]
    if not name:
        raise ValueError("give the voice a name")
    for p in list_profiles():
        if p["id"] != exclude_id and p["name"].casefold() == name.casefold():
            raise ValueError(f'a voice named "{p["name"]}" already exists; choose another name, or rename or delete that one')
    return name


def rename_profile(pid: str, name: str) -> dict:
    meta = get_profile(pid)
    meta["name"] = _check_name(name, exclude_id=pid)
    path = os.path.join(_pdir(pid), "meta.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    os.replace(tmp, path)
    return meta


def create_profile(name: str, recordings: list[tuple[str, bytes]]) -> dict:
    """Build a profile from [(prompt_id, raw_audio_bytes), ...]. Raises ValueError on unusable input."""
    import soundfile as sf
    import torch

    name = _check_name(name)
    if not recordings:
        raise ValueError("at least one recording is required")
    takes, reports = [], []
    for i, (prompt_id, raw) in enumerate(recordings):
        a = analyze(decode_to_22k(raw))
        rep = {"prompt_id": prompt_id, **a["report"]}
        min_s = next((p["min_s"] for p in PROMPTS if p["id"] == prompt_id), 8) * 0.6
        if rep["speech_s"] < min_s:
            raise ValueError(f"recording {i + 1} has only {rep['speech_s']}s of speech; please read the whole passage")
        takes.append(a["speech"])
        reports.append(rep)
    total = sum(len(t) for t in takes) / SR
    if total < MIN_TOTAL_SPEECH_S * 0.8:
        raise ValueError(f"only {total:.0f}s of speech in total; about {MIN_TOTAL_SPEECH_S:.0f}s is needed")

    # Embed every window once, then derive both the profile centroid and a
    # per-take embedding from it. The per-take embeddings are kept because the
    # gap between a speaker's own takes is what makes an identification score
    # interpretable ("0.62, and this person's own takes differ by 0.11").
    take_ses, all_windows = [], []
    for t in takes:
        ws = _embed_windows(_windows(t))
        take_ses.append(torch.stack(ws).mean(0).cpu())
        all_windows.extend(ws)
    se = torch.stack(all_windows).mean(0).cpu()
    spread = _pairwise_spread(take_ses) if len(take_ses) > 1 else _pairwise_spread(all_windows)

    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "voice"
    pid = f"{slug}-{secrets.token_hex(3)}"
    d = _pdir(pid)
    os.makedirs(os.path.join(d, "recordings"), exist_ok=True)
    for i, t in enumerate(takes, 1):
        sf.write(os.path.join(d, "recordings", f"{i:02d}.wav"), t, SR, subtype="PCM_16")
    torch.save(se, os.path.join(d, "se.pt"))
    torch.save(torch.stack(take_ses), os.path.join(d, "takes.pt"))
    meta = {
        "id": pid,
        "name": name,
        "created": int(time.time()),
        "speech_s": round(total, 1),
        "recordings": reports,
        "warnings": sorted({w for r in reports for w in r["warnings"]}),
        "engine": "openvoice-v2",
        "intraspeaker_spread": round(float(spread), 4),
        "take_count": len(takes),
    }
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    logger.info(f"[voice_clone] created profile {pid} from {len(takes)} takes, {total:.0f}s speech")
    return meta


def target_se(pid: str):
    import torch

    return torch.load(os.path.join(_pdir(pid), "se.pt"), map_location="cpu", weights_only=True)


# --------------------------------------------------------------------------- #
# Identification                                                                #
# --------------------------------------------------------------------------- #


def _centroids() -> list[tuple[dict, object]]:
    """Every enrolled profile with a loadable centroid, newest first.

    A profile whose files are unreadable is skipped rather than fatal: one
    corrupted enrolment must not make every speaker unidentifiable.
    """
    out = []
    for p in list_profiles():
        try:
            out.append((p, target_se(p["id"])))
        except Exception as e:
            logger.warning(f"[voice_clone] skipping profile {p.get('id')}: {e}")
    return out


def _best_window(windows: list, centroid) -> tuple[float, int]:
    """Best-matching single window, and its index.

    The *best* window rather than the mean: a clip of half a minute contains
    breaths, a cough and a door, and one clean window is real evidence while the
    average of a clean window and a door is not.
    """
    best, idx = -1.0, -1
    for i, w in enumerate(windows):
        s = _cosine(w, centroid)
        if s > best:
            best, idx = s, i
    return best, idx


def _score_against(windows: list, cohort: list[tuple[dict, object]]) -> list[dict]:
    ranked = []
    for p, c in cohort:
        score, window = _best_window(windows, c)
        ranked.append({
            "id": p["id"],
            "name": p.get("name") or p["id"],
            "score": round(float(score), 4),
            "best_window": window,
            "intraspeaker_spread": p.get("intraspeaker_spread"),
        })
    ranked.sort(key=lambda r: -r["score"])
    return ranked


def identify(raw: bytes, threshold: float | None = None) -> dict:
    """Who is speaking? Returns every profile ranked, plus a calibrated verdict.

    This reuses the tone-colour embedding the cloner already writes, so it costs
    no extra model and no extra VRAM on a card that llama-server and sd-server
    also need. The `openvoice` package ships no verifier subpackage, so a
    dedicated WavLM speaker-verification model is not available here; this
    function is the seam where one would drop in - it takes audio and returns a
    ranking, and knows nothing about how the scores are produced.
    """
    speech = analyze(decode_to_22k(raw))["speech"]
    windows = _embed_windows(_windows(speech))
    cohort = _centroids()
    if not cohort:
        return {"ok": False, "error": "no enrolled voices to compare against", "candidates": []}
    ranked = _score_against(windows, cohort)
    if not windows:
        return {"ok": False, "error": "not enough speech in the clip to identify", "candidates": ranked}

    spread_key = "intraspeaker_spread"
    thr = threshold if threshold is not None else identity_threshold(
        [p.get(spread_key) for p, _ in cohort]
    )
    top = ranked[0]
    runner_up = ranked[1]["score"] if len(ranked) > 1 else None
    return {
        "ok": True,
        "engine": "openvoice-v2-reference-encoder",
        "matched": top["id"] if top["score"] >= thr else None,
        "matched_name": top["name"] if top["score"] >= thr else None,
        "score": top["score"],
        "threshold": round(float(thr), 4),
        "margin": round(float(top["score"] - runner_up), 4) if runner_up is not None else None,
        "runner_up": ranked[1]["id"] if len(ranked) > 1 else None,
        "windows_scored": len(windows),
        "candidates": ranked,
    }


def calibrate(clips: list[tuple[str, bytes]]) -> dict:
    """Score labelled clips against their own profile and against the impostors.

    The only way to choose a threshold honestly. For each clip it reports the
    score against the profile it is *supposed* to be and the best score against
    any other profile, then suggests a boundary between the two populations.
    Callers can override the suggested value; `identify` keeps deriving its own
    threshold from the enrolled cohort so an uncalibrated install still works.
    """
    cohort = _centroids()
    if not cohort:
        return {"ok": False, "error": "no enrolled voices to compare against", "rows": []}

    rows = []
    for label, raw in clips:
        speech = analyze(decode_to_22k(raw))["speech"]
        windows = _embed_windows(_windows(speech))
        ranked = _score_against(windows, cohort)
        own = next((r for r in ranked if r["id"] == label), None)
        other = next((r for r in ranked if r["id"] != label), None)
        rows.append({
            "label": label,
            "own_score": own["score"] if own else None,
            "impostor_id": other["id"] if other else None,
            "impostor_score": other["score"] if other else None,
            "margin": round(own["score"] - other["score"], 4) if own and other else None,
            "windows_scored": len(windows),
        })

    known = [r["own_score"] for r in rows if r["own_score"] is not None]
    impostor = [r["impostor_score"] for r in rows if r["impostor_score"] is not None]
    suggested = None
    if known and impostor:
        suggested = round((min(known) + max(impostor)) / 2.0, 4)
    weakest = min((r for r in rows if r["margin"] is not None), key=lambda r: r["margin"], default=None)
    return {
        "ok": True,
        "rows": rows,
        "clips": len(rows),
        "suggested_threshold": suggested,
        "lowest_own_score": round(min(known), 4) if known else None,
        "highest_impostor_score": round(max(impostor), 4) if impostor else None,
        "cohort_threshold": round(identity_threshold([p.get("intraspeaker_spread") for p, _ in cohort]), 4),
        "weakest_clip": weakest["label"] if weakest else None,
    }


def profile_spread(pid: str):
    """A profile's own take-to-take spread, or None if it predates this record."""
    try:
        return get_profile(pid).get("intraspeaker_spread")
    except KeyError:
        return None


def source_se(voice: str, synth):
    """SE for a Kokoro voice, cached. `synth(text) -> (audio, sr)` renders Kokoro speech."""
    import torch

    key = re.sub(r"[^a-z0-9_,]+", "", voice.lower()).replace(",", "+")
    path = os.path.join(VOICES_DIR, "_sources", f"{key}.pt")
    if os.path.isfile(path):
        return torch.load(path, map_location="cpu", weights_only=True)
    audio, sr = synth(_SOURCE_SCRIPT)
    speech = analyze(_resample(audio, sr, SR))["speech"]
    se = _embed(_windows(speech))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(se, path)
    logger.info(f"[voice_clone] cached source embedding for Kokoro voice {voice}")
    return se
