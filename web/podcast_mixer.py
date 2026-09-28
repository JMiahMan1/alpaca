"""Procedural music beds, ducking envelopes, fades, and the podcast mixer.

Why this module exists, and why it is here rather than in audio_server.py
=============================================================================

A podcast bed is a narrow job: sustain a harmony, keep a gentle pulse going,
and never compete with the speech sitting on top of it. That is textbook
procedural synthesis, and asking a song generator for it is the wrong
instrument. MusicGen is a 30-second-window song model that lives on the same
8 GB card as llama-server and sd-server, emits at 32 kHz, and is licensed
CC-BY-NC. Asking it for ten minutes of unobtrusive pad costs ten minutes of
VRAM that the other two services need, produces something that drifts out of
its trained distribution the moment it exceeds 30 s, and imports a
non-commercial restriction on a podcast bed.

So: **the bed is synthesized here**, at 24 kHz to match Kokoro exactly, so
nothing in the normal path needs resampling at all. The episode length is
whatever the script needs.

MusicGen is still used, and still earns its keep, for the short **theme
sting** -- 10 seconds of show identity is precisely what it is good at. That
one needs a 32 kHz -> 24 kHz resample, which `resample_linear` handles.

Everything here is numpy plus the stdlib `wave` module. `Dockerfile.web` has
no ffmpeg, no scipy, no librosa and no pydub, and does not need them.

The one thing that must NOT be reused from audio_server
------------------------------------------------------

`audio_server._wav_bytes` peak-normalises before writing PCM. That is right
for a single synthesis, where the caller has no level reference. It is
catastrophic for a final mix: the bed is deliberately ~20 dB below the speech,
so normalising the mix to its own peak re-boosts the bed over the speech and
buries the dialogue. `encode_wav` below writes the samples as they are, with
only a soft clip, and that difference is the whole reason this module has its
own encoder.
"""

from __future__ import annotations

import base64
import io
import math
import re
import wave
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Kokoro's output rate. Generating the bed here means the mixer sums arrays
# that already agree on their length, and a MusicGen sting is the only thing
# in the whole path that has to be resampled.
TARGET_SR = 24000

#: Bed level relative to the speech, in dB. ~20 dB is the usual broadcast
#: convention for music under speech: present enough to feel intentional,
#: quiet enough that every consonant stays intelligible.
DEFAULT_DUCK_DB = 20.0

#: Seconds of bed fade-in at the head and fade-out at the tail. Without these
#: a loop either starts abruptly (a click) or stops abruptly (a worse click).
DEFAULT_EDGE_FADE_S = 1.2

#: Attack and release of the ducking envelope, in seconds. A duck that is too
#: slow lets the bed cover the first syllable; too fast and the bed audibly
#: pumps between words.
DEFAULT_DUCK_ATTACK_S = 0.18
DEFAULT_DUCK_RELEASE_S = 0.45

#: How much *further* the bed drops while somebody is speaking, in dB, on top
#: of its resting level. This is deliberately small and separate from
#: `DEFAULT_DUCK_DB`: the bed's resting level already places it that far under
#: the speech, and spending the same number twice put it at twice the
#: requested depth (an inaudible bed under dialogue). The sidechain only has to
#: stop a syllable from landing on a chord change.
DEFAULT_BED_SIDECHAIN_DB = 6.0

#: Absolute ceiling on a generated bed's peak. Well under 1.0 so the sum with
#: speech cannot clip before the final encode.
BED_PEAK_CEILING = 0.7

#: Twelve semitone offsets used by `name_to_semitone`, indexed by the letter
#: pair. C is index 0.
_SEMITONE_BY_PAIR = {
    "c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11,
}
_PITCH_CLASS_NAMES = ("c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b")

#: Preset beds. `key` is a note name ("a", "f#m"), `mood` picks the waveform
#: recipe, the rest shape it.
BED_PRESETS: dict[str, dict[str, Any]] = {
    "ambient_warm": {
        "label": "Ambient Warm",
        "prompt": "slow warm pad, soft Rhodes chords, gentle pulse, unobtrusive",
        "key": "a", "bpm": 72, "density": 0.45, "brightness": 0.35, "mood": "warm",
    },
    "lofi_calm": {
        "label": "Lo-Fi Calm",
        "prompt": "dusty lo-fi keys, vinyl warmth, relaxed brushed pulse",
        "key": "f#m", "bpm": 78, "density": 0.55, "brightness": 0.3, "mood": "dusty",
    },
    "minimal_pulse": {
        "label": "Minimal Pulse",
        "prompt": "sparse sustained tones with a slow pulse, almost no melody",
        "key": "d", "bpm": 66, "density": 0.22, "brightness": 0.5, "mood": "sparse",
    },
    "acoustic_morning": {
        "label": "Acoustic Morning",
        "prompt": "light fingerpicked guitar figure over a soft pad, bright and hopeful",
        "key": "g", "bpm": 96, "density": 0.62, "brightness": 0.68, "mood": "warm",
    },
    "deep_focus": {
        "label": "Deep Focus",
        "prompt": "low sustained drone with a barely-there heartbeat pulse",
        "key": "c", "bpm": 60, "density": 0.3, "brightness": 0.18, "mood": "sparse",
    },
}

#: Curated two-host pairs. A podcast with one voice is an audiobook, and a
#: pair where both voices share a timbre is a monologue with a cough. Every
#: pair is one af_* and one am_* voice, because the difference between them is
#: the fastest cue a listener has to who is talking. ``clone_*`` slots are
#: filled in by the caller from a saved voice profile; the mixer warns rather
#: than silently applying a cross-gender tone colour, which is the one thing
#: OpenVoice does badly.
HOST_PAIRS: list[dict[str, Any]] = [
    {
        "id": "duo_warm",
        "label": "Warm & Measured",
        "a": {"name": "Ada", "voice": "af_nicole", "role": "host", "gender": "f"},
        "b": {"name": "Rowan", "voice": "am_michael", "role": "cohost", "gender": "m"},
        "description": "A calm, unhurried pair that suits explainers and interviews.",
    },
    {
        "id": "duo_bright",
        "label": "Bright & Quick",
        "a": {"name": "Juno", "voice": "af_sky", "role": "host", "gender": "f"},
        "b": {"name": "Kit", "voice": "am_fenrir", "role": "cohost", "gender": "m"},
        "description": "Fast, energetic pairing for news, reviews and game shows.",
    },
    {
        "id": "duo_deep",
        "label": "Low & Gravelled",
        "a": {"name": "Vera", "voice": "af_heart", "role": "host", "gender": "f"},
        "b": {"name": "Omar", "voice": "am_adam", "role": "cohost", "gender": "m"},
        "description": "Lower, steadier timbres for documentary and narrative work.",
    },
    {
        "id": "duo_witty",
        "label": "Witty & Loose",
        "a": {"name": "Pip", "voice": "af_bella", "role": "host", "gender": "f"},
        "b": {"name": "Des", "voice": "am_puck", "role": "cohost", "gender": "m"},
        "description": "Lighter, more conversational pairing for chat formats.",
    },
]

# Two accepted tag shapes, and the difference is not cosmetic:
#
#   [host_a] Welcome to the show.      <- bracketed, NO colon
#   host_b: Glad to be here.           <- bare key, colon required
#
# A bracketed tag must not require a colon, because the bracket already
# delimits it; requiring one would reject the most common form a model
# produces. A bare tag must require one, or "Here is the plan: ship it"
# would be read as a speaker turn.
_BRACKET_TAG_RE = re.compile(
    r"^\s*[\[(]\s*(?P<who>[A-Za-z][A-Za-z0-9_ .-]{0,40}?)\s*[\])]\s*[:\-\u2013]?\s*(?P<text>\S.*)$"
)
_COLON_TAG_RE = re.compile(
    r"^\s*[\[(]?\s*(?P<who>[A-Za-z_][A-Za-z0-9_ .-]{0,40}?)\s*[\])]?\s*:\s*(?P<text>\S.*)$"
)


#: A speaker tag is a name, not a clause. Two words covers every form in
#: use -- "ADA", "host_a", "Host A", "Dr. Chen" -- and rejects prose like
#: "Here is the plan: ship it", which is the reason this bound exists.
_MAX_TAG_WORDS = 2


def match_speaker_tag(line: str) -> tuple[str, str] | None:
    """Return ``(speaker, text)`` for a tagged line, or ``None``.

    A colon alone is not enough. Ordinary prose is full of them, and a
    sentence like ``Here is the plan: ship it`` otherwise becomes a turn
    whose speaker is "Here is the plan" and whose text is "ship it" -- which
    then gets synthesised in somebody's voice. So the tag must also be
    short enough to be a name.
    """
    for pattern in (_BRACKET_TAG_RE, _COLON_TAG_RE):
        m = pattern.match(line)
        if m:
            who = m.group("who").strip()
            if len(who.split()) > _MAX_TAG_WORDS:
                return None
            return who, m.group("text").strip()
    return None


# ---------------------------------------------------------------------------
# small signal helpers
# ---------------------------------------------------------------------------


def db_to_lin(db: float) -> float:
    """Decibels to a linear amplitude ratio."""
    return float(10.0 ** (db / 20.0))


def lin_to_db(lin: float) -> float:
    """A linear amplitude ratio to decibels, floored at -120."""
    if lin <= 1e-6:
        return -120.0
    return float(20.0 * math.log10(lin))


def soft_clip(x: np.ndarray) -> np.ndarray:
    """tanh-style limiter that bounds the result without a hard edge.

    Hard clipping a sum of speech and bed is audible as a crackle on every
    peak. This is the difference between a mix that occasionally reaches
    full scale gracefully and one that does not.
    """
    return np.tanh(x)


def normalize_peak(x: np.ndarray, target: float = 1.0) -> np.ndarray:
    """Scale so the loudest sample sits at ``target``."""
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 1e-9:
        return x
    return x * (target / peak)


def fade_envelope(n: int, fade_in_s: float, fade_out_s: float, sr: int) -> np.ndarray:
    """A 0..1 ramp up, flat, then a ramp down.

    Both fades are clamped to half the buffer so a short buffer gets a
    triangle rather than a ramp that never reaches full height.
    """
    env = np.ones(n, dtype=np.float32)
    if n <= 0:
        return env
    fin = max(0, min(int(fade_in_s * sr), n // 2))
    fout = max(0, min(int(fade_out_s * sr), n // 2))
    if fin > 0:
        env[:fin] = np.linspace(0.0, 1.0, fin, dtype=np.float32)
    if fout > 0:
        env[n - fout:] = np.linspace(1.0, 0.0, fout, dtype=np.float32)
    return env


def duck_envelope(
    speech: np.ndarray,
    sr: int,
    duck_db: float = DEFAULT_DUCK_DB,
    attack_s: float = DEFAULT_DUCK_ATTACK_S,
    release_s: float = DEFAULT_DUCK_RELEASE_S,
    threshold: float = 0.02,
) -> np.ndarray:
    """A 1.0 (bed at full level) to ``duck_gain`` gain curve, 1 sample per speech sample.

    This is a sidechain envelope follower, not a compressor: the bed is
    pulled down while anyone is speaking and released back afterwards. The
    gain is computed from a smoothed speech-level signal, so it rises and
    falls with the talk instead of snapping on every syllable.

    ``threshold`` is the speech level below which the bed is left alone, so
    the pauses between sentences do not get pumped.
    """
    if speech.size == 0:
        return np.ones(0, dtype=np.float32)

    # Instantaneous power in short windows, then a one-pole smoother. The
    # smoother's coefficients ARE the attack and release, which keeps the
    # whole thing one vectorised pass with no Python loop over samples.
    # The window must never be wider than the buffer, or the reshape below has
    # no samples to fill. A clip shorter than one 10 ms window is not a real
    # episode, but it is a reachable request body and must not be a 500.
    win = max(1, min(int(0.01 * sr), speech.size))
    n_win = max(1, speech.size // win)
    trimmed = speech[: n_win * win].reshape(n_win, win)
    level = np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1)).astype(np.float32)

    gate = (level > threshold).astype(np.float32)
    # Expand the window-level control back up to sample resolution.
    target = np.repeat(gate, win)[: speech.size]
    if target.size < speech.size:
        target = np.pad(target, (0, speech.size - target.size))

    a_att = math.exp(-1.0 / max(attack_s * sr, 1.0))
    a_rel = math.exp(-1.0 / max(release_s * sr, 1.0))

    gain = np.empty(speech.size, dtype=np.float32)
    # Separate one-pole constants for opening and closing: the classic
    # asymmetric envelope follower. Written as a loop because the two-pole
    # recurrence is genuinely sequential, and one pass over a 20-minute
    # buffer is milliseconds.
    opening = float(target[0])
    closing = float(target[0])
    for i in range(speech.size):
        t = target[i]
        opening = a_att * opening + (1.0 - a_att) * t
        closing = a_rel * closing + (1.0 - a_rel) * t
        # The lower of the two is the envelope: fast to duck, slow to return.
        gain[i] = opening if opening < closing else closing

    return 1.0 - (1.0 - db_to_lin(-abs(duck_db))) * gain


def resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Linear-interpolation resample.

    Good enough for one purpose only: MusicGen emits 32 kHz and a sting is a
    slowly-varying pad, where linear interpolation's imaging artefacts sit
    far below the bed. It is not a general resampler and the docstring says
    so, because the obvious next caller will reach for it anyway.
    """
    if src_sr <= 0 or dst_sr <= 0 or src_sr == dst_sr or x.size == 0:
        # A WAV header can declare a framerate of 0. Passing that through to
        # the ratio below is a ZeroDivisionError on a request that arrived over
        # HTTP, so an unreadable rate returns the audio untouched instead.
        return np.asarray(x, dtype=np.float32)
    n_out = max(1, round(x.size * dst_sr / src_sr))
    src_idx = np.linspace(0.0, x.size - 1, n_out, dtype=np.float64)
    return np.interp(src_idx, np.arange(x.size, dtype=np.float64), x).astype(np.float32)


def loop_to_length(x: np.ndarray, n: int, crossfade_s: float = 1.0, sr: int = TARGET_SR) -> np.ndarray:
    """Repeat ``x`` until it is exactly ``n`` samples, crossfading the joins.

    Without a crossfade the join is a step discontinuity, which is a click
    once per loop period -- for a 10 s bed in a 12 minute episode, 72 clicks.

    The overlap is a real crossfade: the outgoing copy's tail and the
    incoming copy's head occupy the same samples and are blended, rather
    than the faded tail being appended *before* the sample. Appending it
    would leave the same step at the front of the fade and drop the bed to
    silence once per loop, which is audible as a heartbeat in the wrong
    place.
    """
    if x.size == 0 or n <= 0:
        return np.zeros(max(0, n), dtype=np.float32)
    if x.size >= n:
        return x[:n].astype(np.float32)

    src = x.astype(np.float32)
    fade = max(0, min(int(crossfade_s * sr), src.size // 2))
    out = np.empty(n, dtype=np.float32)
    written = 0
    first = True
    while written < n:
        take = min(src.size, n - written)
        if first or fade == 0:
            out[written : written + take] = src[:take]
            first = False
        else:
            k = min(fade, take)
            t = np.linspace(0.0, 1.0, k, dtype=np.float32)
            if written >= k:
                prev_tail = out[written - k : written]
            else:  # pragma: no cover - only when the fade is longer than a copy
                prev_tail = np.concatenate([out[:written], np.zeros(k - written, dtype=np.float32)])
            out[written : written + k] = prev_tail * (1.0 - t) + src[:k] * t
            out[written + k : written + take] = src[k:take]
        written += take
    return out


# ---------------------------------------------------------------------------
# music theory, just enough of it
# ---------------------------------------------------------------------------


def name_to_semitone(name: str) -> int:
    """"a3" -> 57, "f#2" -> 42, "c" -> 60 (MIDI numbers).

    A note with no octave is read as octave 4; a name that is not a note at
    all falls back to C3 (48). A podcast bed does not need a key signature,
    but it does need a root to build a fifth and an octave from, and
    accepting "a" as well as "a3" is what makes the presets writable.
    """
    m = re.fullmatch(r"\s*([a-gA-G])([#b]?)(-?\d{1,2})?\s*", name or "")
    if not m:
        return 48
    letter, accidental, octave = m.group(1).lower(), m.group(2), m.group(3)
    semi = _SEMITONE_BY_PAIR[letter] + (1 if accidental == "#" else -1 if accidental == "b" else 0)
    oct_num = int(octave) if octave is not None else 4
    return semi + (oct_num + 1) * 12


def scale_intervals(mood: str) -> list[int]:
    """A chord vocabulary per mood, in semitones from the root.

    All major/minor-adjacent triads so far that they cannot sound wrong
    under speech. Nothing exotic: a bed that grabs attention has failed.
    """
    if mood == "sparse":
        return [0, 7, 12, 19]
    if mood == "dusty":
        return [0, 3, 7, 10, 15]
    if mood == "warm":
        return [0, 4, 7, 11, 14]
    return [0, 4, 7, 12]


# ---------------------------------------------------------------------------
# bed synthesis
# ---------------------------------------------------------------------------


def synthesize_bed(
    duration_s: float,
    sr: int = TARGET_SR,
    key: str = "a",
    bpm: float = 72.0,
    density: float = 0.45,
    brightness: float = 0.4,
    mood: str = "warm",
    seed: int = 0,
    fade_s: float = DEFAULT_EDGE_FADE_S,
) -> np.ndarray:
    """Generate a bed of exactly ``duration_s`` seconds.

    The recipe is a slow detuned pad plus an optional soft pulse, which is
    what "unobtrusive" turns into when it is made of oscillators:

    * **pad** -- three sine partials per chord tone, each detuned by a few
      cents so the chord slowly beats against itself. Beat frequencies of
      well under 1 Hz are what stop a sine pad sounding like a test tone.
    * **pulse** -- a filtered click train on the beat, at low level. It gives
      the bed a sense of time without a drum kit.
    * **lowpass** -- a one-pole filter whose cutoff moves with ``brightness``,
      so a "deep focus" bed has almost no top end and an "acoustic morning"
      bed has some.

    Deterministic in ``seed``: the same arguments always produce the same
    bed, so a re-render of the same script is the same mix and a bad bed can
    be reproduced from a one-line description.
    """
    n = max(1, int(duration_s * sr))
    root = name_to_semitone(key)
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / sr

    out = np.zeros(n, dtype=np.float64)
    intervals = scale_intervals(mood)

    # --- pad -------------------------------------------------------------
    # Density chooses how many chord tones are voiced, and the order is
    # fixed (not shuffled) so the chord always has its root.
    voices = max(2, round(len(intervals) * max(0.15, min(density, 1.0))))
    for i, semi in enumerate(intervals[:voices]):
        freq = 440.0 * (2.0 ** ((root + semi - 69) / 12.0))
        if freq >= sr / 2.2:
            continue
        amp = 1.0 / (1.0 + i * 0.55)
        for harmonic, h_amp in ((1, 1.0), (2, 0.32 * brightness), (3, 0.14 * brightness)):
            f = freq * harmonic
            if f >= sr / 2.2:
                continue
            # +-4 cents of detune. The two halves beat at 2*4 cents, which at
            # A2-ish frequencies is well under 1 Hz: a slow shimmer, not a wobble.
            for cents in (-4.0, 4.0):
                f2 = f * (2.0 ** (cents / 1200.0))
                out += amp * h_amp * np.sin(2.0 * math.pi * f2 * t)
    if np.max(np.abs(out)) > 1e-9:
        out /= np.max(np.abs(out))

    # --- pulse -----------------------------------------------------------
    if density > 0.05 and bpm > 1.0:
        beat = 60.0 / bpm
        n_beats = math.ceil(duration_s / beat) + 1
        pulse = np.zeros(n, dtype=np.float64)
        decay = max(0.08, 0.55 * brightness)
        for b in range(n_beats):
            start = int(b * beat * sr)
            if start >= n:
                break
            length = min(n - start, int(decay * 3 * sr))
            if length <= 2:
                continue
            env = np.exp(-np.arange(length) / max(1.0, decay * sr))
            # A pitch drop is what makes a click sound struck rather than ticked.
            sweep = np.sin(2.0 * math.pi * (180.0 * np.exp(-np.arange(length) / max(1.0, 0.06 * sr)) * np.arange(length) / sr))
            gain = 0.16 * density
            pulse[start : start + length] += env * sweep * gain * (1.0 if b % 2 == 0 else 0.72)
        out += pulse

    # --- one-pole lowpass ------------------------------------------------
    # Brightness 0.0 -> 400 Hz, 1.0 -> 9 kHz. A 24 kHz sample rate with no
    # filter at all would put the pad's harmonics in the same band as the
    # sibilance, which is exactly what a bed must not do.
    cutoff = 400.0 + brightness * 8600.0
    alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff / sr)
    filtered = np.empty_like(out)
    acc = 0.0
    for i in range(n):
        acc += alpha * (out[i] - acc)
        filtered[i] = acc
    out = filtered

    # --- level -----------------------------------------------------------
    out = normalize_peak(out, BED_PEAK_CEILING)
    # Slow tremolo so the bed breathes instead of sitting perfectly still.
    lfo = 1.0 + 0.06 * np.sin(2.0 * math.pi * 0.07 * t + float(rng.random()) * 0.1)
    out = out * lfo
    out = out * fade_envelope(n, fade_s, fade_s, sr)
    return out.astype(np.float32)


def bed_preset(preset_id: str) -> dict[str, Any]:
    """Look up a preset, falling back to ``ambient_warm`` for an unknown id.

    A bad preset id from a stale UI should still produce a usable bed rather
    than a 400 on a render the user already waited for.
    """
    return dict(BED_PRESETS.get(preset_id) or BED_PRESETS["ambient_warm"])


def resample_musicgen_sting(wav_bytes: bytes, target_sr: int = TARGET_SR) -> np.ndarray:
    """Decode a MusicGen WAV and bring it to the mixer's rate.

    The only resampling in the whole podcast path. Returns float samples in
    [-1, 1].
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        src_sr = w.getframerate()
        n_ch = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported wav sample width: {width}")
    if n_ch > 1:
        data = data.reshape(-1, n_ch).mean(axis=1)
    return resample_linear(data, src_sr, target_sr)


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------


def encode_wav(x: np.ndarray, sr: int = TARGET_SR, clip: bool = True) -> bytes:
    """Encode float samples as 16-bit PCM mono WAV, **without normalising**.

    The non-normalising part is the entire point. The bed sits ~20 dB under
    the speech by design; any peak normalisation here would rescale the whole
    mix so the quietest thing in it becomes the loudest, which is the
    opposite of what ducking is for.

    ``clip=True`` still guards against wraparound on integer conversion, but
    it is a soft safety net, not a level control.
    """
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if clip:
        arr = soft_clip(np.clip(arr, -1.0, 1.0))
    pcm = np.round(arr * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def decode_wav(b: bytes) -> tuple[np.ndarray, int]:
    """Decode 8/16/32-bit mono or multi-channel WAV to float samples + rate."""
    with wave.open(io.BytesIO(b), "rb") as w:
        sr = w.getframerate()
        n_ch = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported wav sample width: {width}")
    if n_ch > 1:
        data = data.reshape(-1, n_ch).mean(axis=1)
    return data.astype(np.float32), sr


# ---------------------------------------------------------------------------
# script parsing
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """One speaker turn: who talks, what they say, and how they say it."""

    speaker: str
    text: str
    line_no: int = 0
    #: Index into the caller-supplied host roster, or None for an unknown
    #: speaker (the text is kept, the voice is not guessed).
    host_index: int | None = None
    #: A markdown heading or an ALL-CAPS section label. Kept for display and
    #: dropped before synthesis: reading "# Episode 1" aloud in a host's
    #: voice is the single most jarring way this pipeline can be wrong.
    is_heading: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker,
            "text": self.text,
            "line_no": self.line_no,
            "host_index": self.host_index,
            "is_heading": self.is_heading,
        }


#: Prefixes that mark a tag as a role rather than a clause -- the form a
#: model uses for a guest, a narrator, or a second host it invented.
_SPEAKER_LABEL_PREFIXES = ("host", "guest", "narrator", "speaker", "cohost", "hosting")


def _looks_like_speaker_label(tag: str) -> bool:
    """Whether an *unresolved* tag should start a new turn.

    A guest host is worth a real turn rather than being glued onto the
    previous speaker's line, but ordinary prose is full of colons and
    "Here: we ship" must not become a speaker called Here. The test is
    deliberately narrow: a role prefix, ALL CAPS, or a two-word name.

    The residual is documented rather than solved: a *two-word clause*
    before a colon ("And so: the tests passed") is still read as a turn.
    Tightening it further would mean refusing guest speakers, and the draft
    prompt tells the model to use the host keys it was given.
    """
    t = tag.strip()
    if not t:
        return False
    canon = re.sub(r"[^a-z0-9]+", "", t.lower())
    if canon.startswith(_SPEAKER_LABEL_PREFIXES):
        return True
    letters = [c for c in t if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        return True
    return len(t.split()) >= 2


def parse_script(
    script: str,
    hosts: Sequence[Any] | None = None,
    default_host: str = "a",
) -> list[Turn]:
    """Turn a tagged two-host script into a list of :class:`Turn`.

    Recognised forms, case-insensitive on the tag:

    ```
    [host_a] Welcome to the show.
    host_b: Glad to be here.
    ADA: And I'm Ada.
    ```

    ``hosts`` may be host keys (``["a", "b"]``) or :class:`HostSpec` objects,
    in which case a tag naming either the key or the host's display name
    resolves -- a model that writes ``Ada:`` has still identified the host,
    and refusing to because the key is ``a`` would silently drop the turn to
    a default voice.

    A line with no recognised tag continues the previous turn, so a wrapped
    paragraph is not silently dropped -- which matters because a model
    writing a two-host script very often wraps long lines. Blank lines
    separate turns, and anything before the first tag is kept as an
    unattributed turn rather than discarded, so no words are ever lost.
    """
    aliases: list[set[str]] = []
    for h in hosts if hosts is not None else ["a", "b"]:
        if isinstance(h, HostSpec):
            names = {h.name.strip().lower(), (h.key or h.name).strip().lower()}
        else:
            names = {str(h).strip().lower()}
        names.discard("")
        aliases.append(names)

    turns: list[Turn] = []

    def _canon(tag: str) -> str:
        """Fold a speaker tag to a comparable form.

        Models write ``host_a``, ``HOST A``, ``Host-A`` and ``host a`` for the
        same host, and only the first is in the roster. Separators and case are
        therefore not part of the identity.
        """
        return re.sub(r"[^a-z0-9]+", "", tag.strip().lower().rstrip(":"))

    def _resolve(speaker: str) -> int | None:
        c = _canon(speaker)
        if not c:
            return None
        for i, names in enumerate(aliases):
            forms = {_canon(n) for n in names}
            forms |= {f"host{n}" for n in forms}
            if c in forms:
                return i
        return None

    for i, raw in enumerate((script or "").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        bare = _BARE_TAG_RE.match(line)
        if bare and not _looks_like_header(line):
            # "[host_b]" with nothing after it. `match_speaker_tag` needs
            # trailing text, so without this the label falls into the
            # continuation branch below and the PREVIOUS host reads "host b"
            # aloud while this speaker's actual line is lost.
            turns.append(Turn(speaker=(bare.group("bracketed") or bare.group("colon")).strip(), text="", line_no=i))
            continue
        tag = match_speaker_tag(line)
        if tag and _resolve(tag[0]) is not None:
            turns.append(Turn(speaker=tag[0], text=tag[1], line_no=i))
        elif tag and _looks_like_speaker_label(tag[0]):
            # A tag naming someone we do not have a voice for still starts a
            # new turn. Merging it into the previous line would make the
            # previous host read the other host's label aloud. A tag that
            # does not look like a label at all is prose with a colon in it.
            turns.append(Turn(speaker=tag[0], text=tag[1], line_no=i))
        elif turns and not turns[-1].is_heading and not _looks_like_header(line):
            turns[-1].text = (turns[-1].text + " " + line).strip()
        else:
            turns.append(Turn(speaker="", text=line, line_no=i, is_heading=_looks_like_header(line)))

    for idx, t in enumerate(turns):
        if t.is_heading:
            t.host_index = None
            continue
        if t.speaker:
            resolved = _resolve(t.speaker)
            if resolved is None:
                # Keep the name for display, but the voice falls back to the
                # nearest previous host rather than being guessed.
                nearest = _last_host(turns, idx)
                t.host_index = _resolve(default_host) if nearest is None else nearest
            else:
                t.host_index = resolved
        if t.host_index is None:
            t.host_index = _resolve(default_host) if idx == 0 else _last_host(turns, idx)
    return turns


def _last_host(turns: Sequence[Turn], idx: int) -> int | None:
    """The host index of the nearest preceding spoken turn, skipping headings."""
    for j in range(idx - 1, -1, -1):
        if not turns[j].is_heading and turns[j].host_index is not None:
            return turns[j].host_index
    return None


def spoken_turns(turns: Sequence[Turn]) -> list[Turn]:
    """Drop headings, leaving only the turns that should be synthesised."""
    return [t for t in turns if not t.is_heading and t.text.strip()]


#: A line that is nothing but a speaker label, e.g. "[host_b]" or "Rowan:".
#: `match_speaker_tag` requires trailing text, so these need their own pattern.
#: A bracket pair or a trailing colon is REQUIRED - without one this matches
#: every short line in the script and turns the whole episode into labels.
_BARE_TAG_RE = re.compile(
    r"^\s*(?:[\[(]\s*(?P<bracketed>[^\[\]()\n]{1,40}?)\s*[\])]|(?P<colon>[^\[\]()\n:]{1,40}?)\s*[:\u2013-])\s*$"
)

_HEADING_RE = re.compile(r"^\s*#{1,6}\s+\S|^\s*(?:intro|outro|host|script|episode)\s*$", re.I)


def _looks_like_header(line: str) -> bool:
    """Markdown headings and bare section labels start a new turn.

    Without this a `# Episode 12` line gets glued onto whatever the model
    said before it.
    """
    if _HEADING_RE.match(line):
        return True
    # ALL CAPS with no sentence punctuation reads as a label, not dialogue.
    letters = [c for c in line if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters) and not line.endswith((".", "!", "?"))


def split_for_tts(text: str, max_chars: int = 3800) -> list[str]:
    """Split a turn into chunks the audio-server will accept.

    The audio-server caps a request at 4000 characters of **input** text
    (checked before normalisation, so a chapter that normalises to longer is
    fine as long as the input fits). A long host turn would otherwise be a
    silent 400, so the split happens here at sentence boundaries.

    ``max_chars`` defaults below the server's own cap, leaving room for the
    few characters of a speaker tag.
    """
    max_chars = max(1, int(max_chars))
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    try:
        from tts_text import sentences
    except Exception:  # pragma: no cover - only if tts_text is unavailable
        sentences = None
    units = list(sentences(text)) if sentences else re.split(r"(?<=[.!?])\s+", text)

    chunks: list[str] = []
    cur = ""
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        if len(unit) > max_chars:
            # A single sentence longer than the cap: hard-split it, because
            # dropping it would lose the speaker's words.
            for off in range(0, len(unit), max_chars):
                chunks.append(unit[off : off + max_chars])
            continue
        if cur and len(cur) + 1 + len(unit) > max_chars:
            chunks.append(cur)
            cur = unit
        else:
            cur = f"{cur} {unit}".strip()
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------------------
# the mix
# ---------------------------------------------------------------------------


@dataclass
class HostSpec:
    """Everything the mixer needs to know about one voice."""

    name: str
    voice: str
    #: Stable per-episode key ("a", "b") that a script tag resolves against.
    #: Distinct from ``name``: a model that writes ``host_a:`` and a model that
    #: writes ``Ada:`` are identifying the same host, and the mixer has to be
    #: able to answer to both.
    key: str = ""
    role: str = "host"
    gender: str = ""
    clone: str | None = None
    #: OpenVoice tau for the clone. Lower is closer to the target speaker.
    clone_tau: float = 0.3
    speed: float = 1.0
    #: Sentences of wiggle room the pace gets, so a host is not metronomic.
    speed_jitter: float = 0.04
    pause_after_s: float = 0.35

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "voice": self.voice,
            "key": self.key or self.name,
            "role": self.role,
            "gender": self.gender,
            "clone": self.clone,
            "clone_tau": self.clone_tau,
            "speed": self.speed,
            "speed_jitter": self.speed_jitter,
            "pause_after_s": self.pause_after_s,
        }


def voice_gender(voice: str) -> str:
    """Kokoro voices are named ``af_``/``am_``/``bf_``/``bm_``; the first two
    letters are the speaker's gender and the first is the language family.

    This matters because OpenVoice degrades badly when the tone colour is
    taken from a voice of a different gender, and the mixer warns about it
    rather than producing a subtly wrong result.
    """
    v = (voice or "").strip().lower()
    if v.startswith("am") or v.startswith("bm"):
        return "m"
    if v.startswith("af") or v.startswith("bf"):
        return "f"
    return ""


def cross_gender_warning(host: HostSpec) -> str | None:
    """A human-readable warning if this clone would be applied cross-gender."""
    if not host.clone:
        return None
    src = voice_gender(host.voice)
    tgt = (host.gender or "").strip().lower()[:1]
    if src and tgt and src != tgt:
        return (
            f"host {host.name!r} is a {tgt} speaker but its Kokoro source "
            f"{host.voice!r} is {src}; OpenVoice tone colour transfers poorly "
            "across genders -- pick a matching source voice or drop the clone"
        )
    return None


@dataclass
class MixResult:
    """The finished episode plus everything needed to explain how it was made."""

    wav: bytes
    duration_s: float
    speech_duration_s: float
    bed_duration_s: float
    turn_count: int
    bed_preset: str
    bed_duck_db: float
    warnings: list[str] = field(default_factory=list)
    per_turn: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_s": round(self.duration_s, 2),
            "speech_duration_s": round(self.speech_duration_s, 2),
            "bed_duration_s": round(self.bed_duration_s, 2),
            "turn_count": self.turn_count,
            "bed_preset": self.bed_preset,
            "bed_duck_db": self.bed_duck_db,
            "warnings": list(self.warnings),
            "per_turn": list(self.per_turn),
        }


def mix_podcast(
    segments: Iterable[dict[str, Any]],
    hosts: Sequence[HostSpec],
    bed: np.ndarray | None = None,
    sr: int = TARGET_SR,
    duck_db: float = DEFAULT_DUCK_DB,
    edge_fade_s: float = DEFAULT_EDGE_FADE_S,
    duck_attack_s: float = DEFAULT_DUCK_ATTACK_S,
    duck_release_s: float = DEFAULT_DUCK_RELEASE_S,
    bed_preset_id: str = "",
    master_gain: float = 1.0,
    seed: int = 0,
) -> MixResult:
    """Sum speech turns and an optional bed into one finished episode.

    ``segments`` are the already-synthesised turns, in order:

    ```
    {"wav": <bytes>, "host_index": 0, "text": "...", "line_no": 3}
    ```

    ``host_index`` selects a voice for pacing and pauses; the audio itself
    was produced upstream, because the audio-server is the only thing that
    can hold a Kokoro model and the web container has no GPU.

    The bed is placed under everything at ``-duck_db`` and pulled down
    further while anyone speaks, then the whole mix is edge-faded so it
    neither starts nor stops abruptly.
    """
    segs = [s for s in segments if s and s.get("wav")]
    warnings: list[str] = []
    per_turn: list[dict[str, Any]] = []

    speech_parts: list[np.ndarray] = []
    pos = 0.0
    for i, seg in enumerate(segs):
        audio, seg_sr = decode_wav(seg["wav"])
        if seg_sr != sr:
            audio = resample_linear(audio, seg_sr, sr)
        audio = audio.astype(np.float32)

        idx = seg.get("host_index")
        host = hosts[idx] if isinstance(idx, int) and 0 <= idx < len(hosts) else None
        if host is None:
            warnings.append(f"turn {i} has no usable host index ({idx!r}); using the first host's pacing")
            host = hosts[0] if hosts else HostSpec(name="?", voice="af_heart")

        warn = cross_gender_warning(host)
        if warn:
            warnings.append(warn)

        # Inter-turn gap. A beat of silence between speakers is most of what
        # makes a two-host script listenable rather than read.
        gap = 0.0 if i == len(segs) - 1 else max(0.0, host.pause_after_s)
        speech_parts.append(audio)
        if gap:
            speech_parts.append(np.zeros(int(gap * sr), dtype=np.float32))

        per_turn.append(
            {
                "index": i,
                "host_index": idx,
                "host": host.name,
                "voice": host.voice,
                "clone": host.clone,
                "text": seg.get("text", ""),
                "line_no": seg.get("line_no"),
                "duration_s": round(audio.size / sr, 2),
                "start_s": round(pos, 2),
                "speed": seg.get("speed", host.speed),
            }
        )
        pos += (audio.size + int(gap * sr)) / sr

    speech = np.concatenate(speech_parts).astype(np.float32) if speech_parts else np.zeros(0, dtype=np.float32)

    n = speech.size
    mixed = speech.copy()
    if bed is not None and bed.size and n:
        b = resample_linear(np.asarray(bed, dtype=np.float32), sr, sr)
        b = loop_to_length(b, n, crossfade_s=max(0.25, edge_fade_s), sr=sr)
        b = normalize_peak(b, BED_PEAK_CEILING) * db_to_lin(-abs(duck_db))
        # The resting level above already spends `duck_db`. The envelope is a
        # multiplier that runs 1.0 in a gap down to db_to_lin(-sidechain_db)
        # under speech, so giving it `duck_db` too would square the number.
        env = duck_envelope(
            speech, sr, duck_db=min(abs(duck_db), DEFAULT_BED_SIDECHAIN_DB), attack_s=duck_attack_s, release_s=duck_release_s
        )
        mixed = speech + b * env

    if n:
        mixed = mixed * fade_envelope(n, edge_fade_s, edge_fade_s, sr)
    mixed = soft_clip(np.asarray(mixed, dtype=np.float32) * max(0.0, master_gain))

    return MixResult(
        # clip=False: the limiter already ran above, on the summed mix. Running
        # it again would be a second pass over the same samples and cost a
        # further ~1.7 dB on a peak that is already near full scale.
        wav=encode_wav(mixed, sr, clip=False),
        duration_s=round(n / sr, 2) if n else 0.0,
        speech_duration_s=round(speech.size / sr, 2),
        bed_duration_s=round(float(bed.size) / sr, 2) if bed is not None and bed.size else 0.0,
        turn_count=len(segs),
        bed_preset=bed_preset_id,
        bed_duck_db=duck_db,
        warnings=warnings,
        per_turn=per_turn,
    )


def host_roster(pair_id: str, voice_profiles: dict[str, dict[str, Any]] | None = None) -> list[HostSpec]:
    """Build the two :class:`HostSpec` for a curated pair, wiring in clones.

    ``voice_profiles`` maps a saved voice-profile id to its metadata. A
    profile is only attached to a host when the ``host_clone_<pair>_<slot>``
    key asks for it -- attaching every available clone by default would
    silently replace a curated voice with a stranger's.
    """
    pair = next((p for p in HOST_PAIRS if p["id"] == pair_id), None)
    if pair is None:
        pair = HOST_PAIRS[0]
    out: list[HostSpec] = []
    for slot in ("a", "b"):
        raw = pair[slot]
        out.append(
            HostSpec(
                name=raw["name"],
                voice=raw["voice"],
                key=slot,
                role=raw.get("role", "host"),
                gender=raw.get("gender", ""),
            )
        )
    for i in range(len(out)):
        key = f"host_clone_{pair['id']}_{'ab'[i]}"
        pid = (voice_profiles or {}).get(key) or None
        if pid:
            out[i].clone = pid
    return out


def curate_voice_pair(candidates: Sequence[str]) -> str:
    """Pick the pair whose voices are closest to ``candidates``.

    Used by the Raven tools: the LLM says "I want a calm, lower, male voice"
    and this turns that into a concrete pair id without the model having to
    know the roster. Falls back to the first pair when nothing matches, since
    a usable mix beats a refusal.
    """
    wanted = {c.strip().lower() for c in candidates if c and c.strip()}
    if not wanted:
        return HOST_PAIRS[0]["id"]
    for pair in HOST_PAIRS:
        voices = {pair["a"]["voice"], pair["b"]["voice"]}
        if voices & wanted:
            return pair["id"]
    return HOST_PAIRS[0]["id"]


def mix_to_data_uri(wav: bytes) -> str:
    """``data:audio/wav;base64,...`` for embedding the episode in a page."""
    return "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii")
