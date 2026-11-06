"""Make Kokoro's narration sound like a game show: an announcer, and a paper boy.

Kokoro reads text evenly, like an audiobook. A game-show announcer is the
same voice delivered differently: quicker, compressed so every word lands at
the same loudness, a lift in the presence band, and a little room around it.
That part is signal processing, done here on the clip the audio server
returns.

"Extra! Extra!" is a different character altogether: a newsboy shouting on a
street corner. The line's opening call is split off and given to him (a
younger Kokoro voice, sped and pitched up, narrowed and driven like a shout,
with a slap echo off the buildings); the announcer reads the rest.

Stdlib only (``array``, ``wave``, ``math``): the arcade image has no numpy,
and a few seconds of 24 kHz mono is a fraction of a second of pure Python.
"""
from __future__ import annotations

import array
import io
import math
import os
import re
import wave

#: The newsboy. No Kokoro voice is a boy (every male voice measured 88-166 Hz),
#: and pitching a man up by a third sounded like a machine. A bright young
#: voice (~185 Hz) lifted only a little (~2.4 semitones) stays human.
PAPERBOY_VOICE = os.getenv("PARTY_PAPERBOY_VOICE", "af_kore")
PAPERBOY_SPEED = 1.05
PAPERBOY_PITCH = 1.15
PAPERBOY_CALL = "Extra! Extra! Read all about it!"
#: Kokoro's own pace is the most natural; cadence varies it per sentence.
ANNOUNCER_SPEED = 1.0

_SENTENCE = re.compile(r"(?<=[.!?…])\s+(?=\S)")


def sentences(text: str) -> list[str]:
    """A line as a host breathes it: one sentence at a time."""
    return [s.strip() for s in _SENTENCE.split((text or "").strip()) if s.strip()]


def cadence(sentence: str, base: float = ANNOUNCER_SPEED) -> tuple[float, float]:
    """``(speed, pause after in ms)`` for one sentence, the way a person reads it.

    Questions slow down and leave room for the answer, exclamations push
    forward, a trailing "..." hangs before the punchline. A little variation
    (fixed per sentence) keeps a run of sentences from sounding metronomic.
    """
    s = sentence.rstrip()
    jitter = ((sum(map(ord, s)) % 9) - 4) * 0.006  # ±2.4 %, same every time for the same words
    if s.endswith(("...", "…")):
        return base * 0.96 + jitter, 650.0
    if s.endswith("?"):
        return base * 0.95 + jitter, 420.0
    if s.endswith("!"):
        return base * 1.05 + jitter, 300.0
    return base + jitter, 340.0

_CALL = re.compile(r"^\s*((?:extra[!.,]?\s*){1,3})(read all about it[!.]?\s*)?", re.IGNORECASE)


def split_show_line(text: str) -> list[tuple[str, str]]:
    """``[(role, text), ...]``: an opening "Extra! Extra!" is the paper boy's, the rest the announcer's."""
    clean = (text or "").strip()
    m = _CALL.match(clean)
    if not m or not m.group(1).strip():
        return [("announcer", clean)] if clean else []
    rest = clean[m.end():].strip()
    parts = [("paperboy", PAPERBOY_CALL)]
    if rest:
        parts.append(("announcer", rest))
    return parts


# --------------------------------------------------------------------------- wav io

def read_wav(data: bytes) -> tuple[list[float], int]:
    with wave.open(io.BytesIO(data)) as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError("expected 16-bit PCM")
    s = array.array("h")
    s.frombytes(raw)
    if ch > 1:
        return [sum(s[i:i + ch]) / (ch * 32768.0) for i in range(0, len(s), ch)], rate
    return [x / 32768.0 for x in s], rate


def write_wav(samples: list[float], rate: int) -> bytes:
    out = array.array("h", (max(-32767, min(32767, int(x * 32767.0))) for x in samples))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(out.tobytes())
    return buf.getvalue()


# --------------------------------------------------------------------------- building blocks

def _alpha(cutoff_hz: float, rate: int) -> float:
    return 1.0 - math.exp(-2.0 * math.pi * cutoff_hz / rate)


def highpass(x: list[float], rate: int, cutoff_hz: float) -> list[float]:
    a, lp, out = _alpha(cutoff_hz, rate), 0.0, []
    for v in x:
        lp += a * (v - lp)
        out.append(v - lp)
    return out


def lowpass(x: list[float], rate: int, cutoff_hz: float) -> list[float]:
    a, lp, out = _alpha(cutoff_hz, rate), 0.0, []
    for v in x:
        lp += a * (v - lp)
        out.append(lp)
    return out


def presence(x: list[float], rate: int, cutoff_hz: float = 2500.0, gain: float = 0.6) -> list[float]:
    """High-shelf lift: add back the part above ``cutoff_hz`` (gain 0.6 ~ +4 dB)."""
    a, lp, out = _alpha(cutoff_hz, rate), 0.0, []
    for v in x:
        lp += a * (v - lp)
        out.append(v + gain * (v - lp))
    return out


def compress(x: list[float], rate: int, threshold_db: float = -20.0, ratio: float = 3.5,
             attack_ms: float = 4.0, release_ms: float = 140.0) -> list[float]:
    """Feed-forward peak compressor: even loudness, the way a broadcast voice sounds."""
    att = math.exp(-1.0 / (rate * attack_ms / 1000.0))
    rel = math.exp(-1.0 / (rate * release_ms / 1000.0))
    thr = 10 ** (threshold_db / 20.0)
    env, out = 0.0, []
    for v in x:
        level = abs(v)
        env = att * env + (1 - att) * level if level > env else rel * env + (1 - rel) * level
        gain = (thr * (env / thr) ** (1.0 / ratio)) / env if env > thr else 1.0
        out.append(v * gain)
    return out


def echoes(x: list[float], rate: int, taps: list[tuple[float, float]], tail_lowpass_hz: float = 3000.0) -> list[float]:
    """Add delayed, darkened copies (ms, gain): a room for the announcer, a street for the newsboy."""
    longest = max((int(rate * ms / 1000.0) for ms, _ in taps), default=0)
    dark = lowpass(x, rate, tail_lowpass_hz)
    out = x + [0.0] * longest
    for ms, g in taps:
        d = int(rate * ms / 1000.0)
        for i, v in enumerate(dark):
            out[i + d] += g * v
    return out


def drive(x: list[float], amount: float = 1.8) -> list[float]:
    """Soft clipping: the edge of a voice pushed to a shout."""
    norm = math.tanh(amount)
    return [math.tanh(amount * v) / norm for v in x]


def resample(x: list[float], factor: float) -> list[float]:
    """Play ``factor`` times faster (and higher). Linear interpolation is plenty for a shout."""
    if factor == 1.0 or not x:
        return list(x)
    n = int(len(x) / factor)
    last = len(x) - 1
    out = []
    for i in range(n):
        p = i * factor
        j = int(p)
        f = p - j
        a = x[j]
        b = x[j + 1] if j < last else a
        out.append(a + (b - a) * f)
    return out


def normalize(x: list[float], peak: float = 0.89) -> list[float]:
    top = max((abs(v) for v in x), default=0.0)
    return [v * peak / top for v in x] if top > 1e-6 else list(x)


def silence(rate: int, ms: float) -> list[float]:
    return [0.0] * int(rate * ms / 1000.0)


# --------------------------------------------------------------------------- the two voices

def announcer_fx(x: list[float], rate: int) -> list[float]:
    # Gentle on purpose: heavy compression and a big room are what made it
    # sound synthetic. A light leveller, a little presence, a hint of studio.
    y = highpass(normalize(x, 0.9), rate, 80.0)
    y = presence(y, rate, 2600.0, 0.45)
    y = compress(y, rate, -24.0, 2.5, attack_ms=6.0, release_ms=180.0)
    y = echoes(normalize(y, 0.9), rate, [(38.0, 0.06), (61.0, 0.035)], 3200.0)
    return normalize(y, 0.89)


def paperboy_fx(x: list[float], rate: int) -> list[float]:
    y = resample(x, PAPERBOY_PITCH)
    y = highpass(y, rate, 260.0)
    y = lowpass(y, rate, 4200.0)
    y = compress(normalize(y, 0.9), rate, -22.0, 3.0, attack_ms=3.0, release_ms=110.0)
    y = drive(normalize(y, 0.95), 1.5)
    y = echoes(y, rate, [(135.0, 0.24), (270.0, 0.11), (420.0, 0.05)], 2400.0)
    return normalize(y, 0.93)


def _trim(x: list[float], rate: int, floor: float = 0.012, keep_ms: float = 40.0) -> list[float]:
    """Drop the dead air Kokoro leaves around a clip, so the pauses are ours."""
    keep = int(rate * keep_ms / 1000.0)
    idx = [i for i, v in enumerate(x) if abs(v) > floor]
    if not idx:
        return x
    return x[max(0, idx[0] - keep): min(len(x), idx[-1] + keep)]


def assemble(parts: list[tuple[str, bytes]] | list[tuple[str, bytes, float]]) -> bytes:
    """``[(role, kokoro wav[, pause after ms]), ...]`` -> one processed show line.

    Each role's sentences are joined with their own pauses first and then
    processed together, so the voice's character stays even across them.
    """
    rate = 0
    groups: list[tuple[str, list[float], float]] = []
    for part in parts:
        role, wav = part[0], part[1]
        gap = float(part[2]) if len(part) > 2 else 260.0  # type: ignore[misc]
        samples, r = read_wav(wav)
        if rate and r != rate:
            samples = resample(samples, r / rate)
        rate = rate or r
        samples = _trim(samples, rate)
        if groups and groups[-1][0] == role:
            _, prev, prev_gap = groups[-1]
            groups[-1] = (role, prev + silence(rate, prev_gap) + samples, gap)
        else:
            groups.append((role, samples, gap))
    out: list[float] = []
    for i, (role, samples, gap) in enumerate(groups):
        out += paperboy_fx(samples, rate) if role == "paperboy" else announcer_fx(samples, rate)
        if i < len(groups) - 1:
            out += silence(rate, max(gap, 380.0))
    return write_wav(out, rate or 24000)
