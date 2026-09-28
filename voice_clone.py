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
import math
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


#: A window quieter than the clip's loudest window by more than this is not
#: evidence about a speaker. Cosine is level-invariant, so without a gate a
#: 45 dB-down stretch of mains hum scores HIGHER against a centroid than the
#: voiced audio next to it, and `identify` - which trusts its best window -
#: names whoever the hum happens to resemble.
WINDOW_GATE_DB = 20.0


def _gate_windows(windows_22k: list) -> list:
    """Drop windows far below the loudest one. Never returns fewer than one."""
    import numpy as np

    if len(windows_22k) < 2:
        return windows_22k
    rms = np.array([float(np.sqrt(np.mean(np.asarray(w, dtype="float64") ** 2))) for w in windows_22k])
    peak = float(rms.max())
    if peak <= 0.0:
        return windows_22k
    loud = rms >= peak * (10.0 ** (-WINDOW_GATE_DB / 20.0))
    if not loud.any():
        # Everything is quiet relative to the others: keep the best one rather
        # than refusing a clip `analyze` already accepted.
        return [windows_22k[int(rms.argmax())]]
    return [w for w, keep in zip(windows_22k, loud.tolist(), strict=True) if keep]


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


def _as_number(value: object, default: float = 0.0) -> float:
    """A number, or `default`. `analyze` returns a plain dict, so every numeric
    field in it is `object` to a checker even though it is always a float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _cohort_spreads(cohort: list[tuple[dict, object]]) -> list[float | None]:
    """Each profile's recorded spread, or None where there is not a usable one.

    A profile enrolled before spreads were recorded has no value, and a
    non-numeric one is a corrupt meta.json. Both become None, which
    `identity_threshold` already skips - passing the raw value through put a str
    into arithmetic, where it raised.
    """
    return [float(p["intraspeaker_spread"]) if isinstance(p.get("intraspeaker_spread"), (int, float)) else None for p, _ in cohort]


def _self_spread(per_take_windows: list[list]) -> float:
    """How far this speaker is from themselves, in the units `identify` scores.

    Hold one take out, score its best window against the centroid of the other
    takes, and keep the worst (largest) deficit. That is deliberately the same
    operation `identify` performs on a new clip, so the recorded number and the
    score it is compared against are drawn from one population.

    Measuring the distance between take *centroids* instead - which is what
    this used to do - compares two heavily-averaged vectors, and averaging
    shrinks the deviation. The recorded spread then came out several times
    LARGER than a genuine re-record's score, so a threshold derived from it
    admitted speakers the cohort itself would have rejected.

    One take cannot be held out, so the fallback is the spread among that
    take's own windows: a less faithful yardstick, but honest rather than 0.0,
    which would claim a certainty there is no evidence for.
    """
    import torch

    if len(per_take_windows) < 2:
        return _pairwise_spread(per_take_windows[0]) if per_take_windows else 0.0
    worst = 0.0
    for i, held in enumerate(per_take_windows):
        rest = [w for j, take in enumerate(per_take_windows) if j != i for w in take]
        if not rest:
            continue
        centroid = torch.stack(rest).mean(0).cpu()
        deficit = 1.0 - _best_window(held, centroid)[0]
        if deficit > worst:
            worst = deficit
    return worst


# The identification threshold is derived from the enrolled cohort rather than
# picked, because "0.82" means nothing on its own. Every profile records how far
# its own takes are from each other; a match has to beat the *worst* speaker's
# own variation by a margin, or the same person re-recorded on a different day
# would read as a stranger.
#
# The unit matters and is easy to get backwards. `intraspeaker_spread` is a
# DISTANCE (`1 - cosine`, so 0 means identical and larger means worse). The
# score `identify` reports is a SIMILARITY (a cosine, so larger is better).
# Comparing the two directly is the same as demanding `cos >= 1.5 * (1 - cos)`,
# which every positive cosine satisfies - the floor then became the 0.05 clamp
# and any two unrelated voices matched. So the function below returns a
# similarity floor, and the clamps bound a deficit on the way in.
SPREAD_MARGIN = 1.5
MIN_SPREAD_DEFICIT = 0.0
MAX_SPREAD_DEFICIT = 0.95
#: The derived floor is a cosine, so its useful range is narrow: a cohort whose
#: own re-records are 0.02 apart yields a floor near 0.97, and clamping that to
#: a "sensible-looking" 0.95 silently loosened it. The floor bound only exists to
#: stop an outlier enrolment demanding a threshold of exactly 1.0.
MIN_IDENTITY_THRESHOLD = 0.05
MAX_IDENTITY_THRESHOLD = 0.995


def identity_threshold(spreads: list[float | None]) -> float:
    """Cohort-derived cosine FLOOR above which two recordings are the same voice.

    ``spreads`` are the recorded per-profile distances (`1 - cosine`). The
    result is a similarity, comparable against the score `identify` reports.
    """
    real = [float(s) for s in spreads if s is not None and math.isfinite(float(s))]
    if not real:
        return 0.75
    deficit = max(MIN_SPREAD_DEFICIT, min(MAX_SPREAD_DEFICIT, SPREAD_MARGIN * max(real)))
    return max(MIN_IDENTITY_THRESHOLD, min(MAX_IDENTITY_THRESHOLD, 1.0 - deficit))


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
                    loaded = json.load(fh)
                # A `meta.json` holding a list, a string or null parses fine and
                # then breaks every `.get()` downstream - which made one
                # malformed file raise AttributeError out of `identify` for the
                # whole install, rather than skipping that one voice. Validate
                # the shape where the file is read, once.
                if not isinstance(loaded, dict):
                    raise ValueError(f"meta.json holds {type(loaded).__name__}, not an object")
                loaded.setdefault("id", pid)
                spread = loaded.get("intraspeaker_spread")
                if spread is not None and not isinstance(spread, (int, float)):
                    # A spread that is a string reaches identity_threshold and
                    # fails the multiply. Drop it: the profile is still
                    # identifiable, it just does not inform the threshold.
                    logger.warning(f"[voice_clone] profile {pid} has a non-numeric intraspeaker_spread; ignoring it")
                    loaded["intraspeaker_spread"] = None
                out.append(loaded)
            except Exception as e:
                logger.warning(f"[voice_clone] unreadable profile {pid}: {e}")
    return sorted(out, key=lambda m: m.get("created", 0) or 0, reverse=True)


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
        min_s = _as_number(next((p.get("min_s") for p in PROMPTS if p["id"] == prompt_id), 8), 8.0) * 0.6
        if _as_number(rep.get("speech_s"), 0.0) < min_s:
            raise ValueError(f"recording {i + 1} has only {rep['speech_s']}s of speech; please read the whole passage")
        takes.append(a["speech"])
        reports.append(rep)
    total = sum(len(t) for t in takes) / SR
    if total < MIN_TOTAL_SPEECH_S * 0.8:
        raise ValueError(f"only {total:.0f}s of speech in total; about {MIN_TOTAL_SPEECH_S:.0f}s is needed")

    # Embed every window once, then derive the profile centroid, a per-take
    # embedding, and the honest self-distance. The per-take WINDOW vectors are
    # kept only as long as it takes to measure that distance - a hold-one-out
    # measurement needs them, and `takes.pt` then stores the take centroids so
    # the number stays auditable after the windows are gone.
    per_take_windows, take_ses, all_windows = [], [], []
    for t in takes:
        ws = _embed_windows(_windows(t))
        per_take_windows.append(ws)
        take_ses.append(torch.stack(ws).mean(0).cpu())
        all_windows.extend(ws)
    se = torch.stack(all_windows).mean(0).cpu()
    spread = _self_spread(per_take_windows)

    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "voice"
    pid = f"{slug}-{secrets.token_hex(3)}"
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
    # meta.json is written last and is the only thing that makes a profile
    # visible, so a failure part-way leaves a directory that list_profiles and
    # _centroids both skip - and delete_profile refuses, because it keys on
    # meta.json. The result is an orphan the operator can neither see nor
    # remove, so unwind the whole directory rather than leave it behind. The
    # try covers the meta write too: that is the likeliest of the four to fail
    # (it is the only one that serialises everything else first) and leaving it
    # outside would leave exactly the orphan this is here to prevent.
    d = _pdir(pid)
    try:
        os.makedirs(os.path.join(d, "recordings"), exist_ok=True)
        for i, t in enumerate(takes, 1):
            sf.write(os.path.join(d, "recordings", f"{i:02d}.wav"), t, SR, subtype="PCM_16")
        torch.save(se, os.path.join(d, "se.pt"))
        torch.save(torch.stack(take_ses), os.path.join(d, "takes.pt"))
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1)
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise
    logger.info(f"[voice_clone] created profile {pid} from {len(takes)} takes, {total:.0f}s speech")
    return meta


def discard_profile_dir(pid: str) -> None:
    """Remove a profile directory whether or not it was ever finished.

    `delete_profile` is deliberately strict - it refuses anything without a
    `meta.json`, so a half-written enrolment can never be destroyed through the
    API by accident. That strictness is what leaves the orphan undeletable, so
    the enrolment path unwinds its own directory instead (see `create_profile`)
    and this exists for the one case a caller cannot unwind itself: a profile
    whose `meta.json` became unreadable after the fact.
    """
    d = _pdir(pid)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)


def target_se(pid: str):
    import torch

    return torch.load(os.path.join(_pdir(pid), "se.pt"), map_location="cpu", weights_only=True)


# --------------------------------------------------------------------------- #
# Identification                                                                #
# --------------------------------------------------------------------------- #


def _centroids() -> list[tuple[dict, object]]:
    """Every enrolled profile with a usable centroid, newest first.

    A profile whose files are unreadable is skipped rather than fatal: one
    corrupted enrolment must not make every speaker unidentifiable. "Usable"
    means finite and non-zero, not merely loadable - a zero vector has an
    undefined cosine, and one that loads but scores 0.0 forever would otherwise
    sit in every ranking as a permanent nonsense candidate that an agent
    rendering "closest is X" could surface.

    Dimensionality is deliberately NOT pinned to a constant: this must keep
    working if the encoder's output width changes, and a mis-shaped vector is
    already handled correctly by `_cosine`, which returns 0.0 on a mismatch.
    """
    import numpy as np

    out = []
    for p in list_profiles():
        try:
            c = target_se(p["id"])
            v = np.asarray(_as_f32(c), dtype="float64").ravel()
            if v.ndim != 1 or not v.size:
                raise ValueError("centroid is not a 1-D vector")
            if not np.isfinite(v).all():
                raise ValueError("centroid contains non-finite values")
            if not np.any(v):
                raise ValueError("centroid is all zeros; its cosine is undefined")
            out.append((p, c))
        except Exception as e:
            logger.warning(f"[voice_clone] skipping profile {p.get('id')}: {e}")
    return out


def _best_window(windows: list, centroid) -> tuple[float, int]:
    """Best-matching single window, and its index.

    The *best* window rather than the mean: a clip of half a minute contains
    breaths, a cough and a door, and one clean window is real evidence while the
    average of a clean window and a door is not.

    Non-finite scores are skipped rather than compared. A NaN centroid makes
    every comparison false, which left the initial -1.0 sentinel in the
    payload: a score of -1.0 is below the 0.0 a dimension mismatch returns, so
    it read as "worse than unrelated" when it really meant "not evaluated".
    """
    best, idx = None, -1
    for i, w in enumerate(windows):
        s = _cosine(w, centroid)
        if not math.isfinite(s):
            continue
        if best is None or s > best:
            best, idx = s, i
    return (0.0, -1) if best is None else (best, idx)


def _score_against(windows: list, cohort: list[tuple[dict, object]]) -> list[dict]:
    import numpy as np

    ranked = []
    dim = len(np.asarray(_as_f32(windows[0])).ravel()) if windows else None
    for p, c in cohort:
        # A centroid whose width does not match the embeddings this encoder
        # currently produces belongs to a different model version. `_cosine`
        # answers 0.0 for a mismatch, which is safe but leaves the profile
        # sitting in every ranking as a permanent dead candidate.
        if dim is not None and len(np.asarray(_as_f32(c)).ravel()) != dim:
            logger.warning(f"[voice_clone] skipping profile {p.get('id')}: centroid width does not match the encoder")
            continue
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
    cohort = _centroids()
    if not cohort:
        # Checked BEFORE embedding: this path loads the reference encoder, and
        # answering "nobody is enrolled" should not cost a checkpoint load (and
        # a Hugging Face fetch on a cold cache).
        return {"ok": False, "error": "no enrolled voices to compare against", "candidates": []}
    speech = analyze(decode_to_22k(raw))["speech"]
    windows = _embed_windows(_gate_windows(_windows(speech)))
    ranked = _score_against(windows, cohort)
    if not windows:
        return {"ok": False, "error": "not enough speech in the clip to identify", "candidates": ranked}

    spreads = _cohort_spreads(cohort)
    thr = threshold if threshold is not None else identity_threshold(spreads)
    top = ranked[0]
    runner_up = ranked[1]["score"] if len(ranked) > 1 else None
    margin = top["score"] - runner_up if runner_up is not None else None

    # A score alone is not a decision, and this is why. The reference encoder
    # was trained to carry TONE COLOUR, not to verify identity, so two different
    # people can embed almost identically - a low-pitched stranger measured
    # 0.98 against a profile whose own re-records measured 0.99. No threshold
    # separates those, because they are the same number. The only other piece
    # of evidence available is how far the runner-up is below the winner: a
    # genuine speaker beats the field by more than one profile varies from
    # itself, and a clip the data genuinely cannot place does not.
    real = [float(s) for s in spreads if s is not None and math.isfinite(float(s))]
    decisive_margin = max(real) if real else 0.0
    confident = top["score"] >= thr and (margin is None or margin >= decisive_margin)
    # The ranking is returned either way: an agent asked "who is speaking" needs
    # "closest is X, but I would not stake anything on it" far more than a bare
    # "no".
    reason = None
    if top["score"] < thr:
        reason = f"best score {top['score']:.4f} is below the cohort floor {thr:.4f}"
    elif not confident:
        reason = (
            f"{top['name']} led by only {margin:.4f}, which is inside this cohort's own "
            f"speaker-to-speaker variation ({decisive_margin:.4f}); the data cannot separate them"
        )
    return {
        "ok": True,
        "engine": "openvoice-v2-reference-encoder",
        "matched": top["id"] if confident else None,
        "matched_name": top["name"] if confident else None,
        "score": top["score"],
        "threshold": round(float(thr), 4),
        "margin": round(float(margin), 4) if margin is not None else None,
        "min_margin": round(float(decisive_margin), 4),
        "decisive": bool(confident),
        "reason": reason,
        "closest": {"id": top["id"], "name": top["name"], "score": top["score"]},
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

    # Annotated because the two append sites below have different value types -
    # one carries a str error, the other carries scores - and an inferred join
    # erases the fact that the score columns are genuinely optional.
    rows: list[dict[str, object]] = []
    for label, raw in clips:
        try:
            speech = analyze(decode_to_22k(raw))["speech"]
            windows = _embed_windows(_gate_windows(_windows(speech)))
        except Exception as e:
            # Per-clip, not per-batch. One truncated upload out of twenty
            # previously raised out of the whole loop, and the route turned
            # that into a 422 that discarded nineteen valid measurements.
            rows.append({"label": label, "own_score": None, "impostor_id": None, "impostor_score": None, "margin": None, "error": str(e)})
            continue
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

    def _scores(row: dict[str, object], key: str) -> float | None:
        """A score column, or None. A non-numeric value is a defect in the row,
        and dropping it is safer than letting it reach min()/max()."""
        v = row.get(key)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    usable = [r for r in rows if _scores(r, "own_score") is not None and _scores(r, "impostor_score") is not None]
    known = [s for s in (_scores(r, "own_score") for r in usable) if s is not None]
    impostor = [s for s in (_scores(r, "impostor_score") for r in usable) if s is not None]
    suggested = None
    if known and impostor:
        suggested = round((min(known) + max(impostor)) / 2.0, 4)
    with_margin = [(_scores(r, "margin"), r["label"]) for r in usable if _scores(r, "margin") is not None]
    weakest_label = min(with_margin)[1] if with_margin else None
    return {
        "ok": True,
        "rows": rows,
        "clips": len(rows),
        "usable_clips": len(usable),
        "suggested_threshold": suggested,
        "lowest_own_score": round(min(known), 4) if known else None,
        "highest_impostor_score": round(max(impostor), 4) if impostor else None,
        # True when the two populations touch or cross - a swapped label, or a
        # cohort that cannot be told apart at all. The midpoint is meaningless
        # in that case, and nothing in the payload said so.
        "populations_overlap": bool(known and impostor and min(known) <= max(impostor)),
        "cohort_threshold": round(identity_threshold(_cohort_spreads(cohort)), 4),
        "weakest_clip": weakest_label,
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
