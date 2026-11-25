"""Seamless, bar-exact loops from a generated music clip (numpy only).

A MusicGen clip crossfaded end-into-start at whatever length it came out makes a
click-free seam, but the BEAT stumbles: a 23.4 s clip at about 122 bpm is 48.8
beats, so every repeat lands about 80 ms off the grid. ``loop_to_bars`` fixes
that:

1. ``estimate_beat`` finds the tempo and the first beat. A coarse lag comes from
   the autocorrelation of an onset envelope; period and phase are then refined
   together by fitting a beat grid to the onsets over the whole clip, because a
   loop of 48 beats shows even a fraction of a BPM as a stumble at the seam.
2. ``bar_aligned`` starts the clip on that first beat and cuts it to a whole
   number of 4-beat bars plus the crossfade tail.
3. ``make_loop`` folds the tail into the head (equal power) and drops it, so
   the loop is exactly N bars long and played back to back keeps the grid.

Only a clip with a real beat is cut to bars. Two gates, both measured on real
material: the autocorrelation peak must stand ``TEMPO_MIN_CONFIDENCE`` times
over the mean of its band (noise and steady tones scored 2.1-2.4, beat tracks
14-20, a MusicGen bed 5.0), and the onset envelope must vary by at least
``ONSET_MIN_VARIATION`` of the clip's RMS (tones and noise 0.03, beat tracks
0.5, the MusicGen bed 0.18). Anything else is only crossfaded.

**Loop exactly once.** Keep a clip unlooped and loop it when it is used. Looping
at creation and again on load folds the tail twice: the second pass eats
another crossfade off a loop that was already whole bars.

**Keep loudness by RMS, not peak** (``normalise_rms``), with a soft knee instead
of clipping the rare peak that ends up past full scale.

Ported from SharedLLM's hold music (``services/homehub/hold_music.py``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: A loop is cut to whole bars of this many beats.
BEATS_PER_BAR = 4
#: Tempo search range.
TEMPO_MIN_BPM = 70.0
TEMPO_MAX_BPM = 200.0
#: Autocorrelation peak over its band's mean needed before a clip is called rhythmic.
TEMPO_MIN_CONFIDENCE = 3.5
#: Onset envelope standard deviation over clip RMS needed, so a steady tone or noise gets no "beat".
ONSET_MIN_VARIATION = 0.1
#: Default end-into-start crossfade.
DEFAULT_CROSSFADE_S = 0.5
_ONSET_HOP = 256
#: The grid search: periods within +-3% of the coarse lag, in this many steps, phases in half hops.
_FINE_SPAN = 0.03
_FINE_STEPS = 121


@dataclass(frozen=True)
class Beat:
    """A clip's steady beat: tempo, the first beat (in samples) and how clearly it stood out."""

    bpm: float
    start: float
    confidence: float


@dataclass(frozen=True)
class LoopInfo:
    """What ``loop_to_bars`` did: the tempo and bar count it cut to (None when it only crossfaded)."""

    bpm: float | None
    bars: int | None
    loop_s: float
    confidence: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "bpm": None if self.bpm is None else round(self.bpm, 2),
            "bars": self.bars,
            "loop_s": round(self.loop_s, 3),
            "beat_confidence": None if self.confidence is None else round(self.confidence, 2),
        }


def _mono(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    return a.mean(axis=1) if a.ndim == 2 else a


def _onsets(x: np.ndarray) -> np.ndarray:
    """Onset strength per hop: the positive rise of the short-term level, mean removed."""
    frames = len(x) // _ONSET_HOP
    if frames < 4:
        return np.zeros(0)
    level = np.sqrt(np.mean(x[: frames * _ONSET_HOP].reshape(frames, _ONSET_HOP) ** 2, axis=1))
    rise = np.maximum(0.0, np.diff(level, prepend=level[0]))
    return rise - rise.mean()


def _beat_scores(onsets: np.ndarray, period: float, phases: np.ndarray) -> np.ndarray:
    """How strongly onsets land on a beat grid of ``period`` hops, for every phase at once.

    Linear interpolation between hops; only the beats every phase has inside the clip are counted, so
    the scores compare like with like. One numpy pass per period keeps a clip's first load fast.
    """
    beats = int((len(onsets) - 1 - phases.max()) // period)
    if beats < 4:
        return np.full(len(phases), -np.inf)
    t = phases[:, None] + period * np.arange(beats)[None, :]
    i = t.astype(int)
    f = t - i
    return np.asarray(np.mean(onsets[i] * (1 - f) + onsets[i + 1] * f, axis=1))


def estimate_beat(x: np.ndarray, rate: int) -> Beat | None:
    """The clip's tempo and first beat, or None when it has no clear steady beat."""
    x = _mono(x)
    if rate <= 0 or x.size == 0:
        return None
    on = _onsets(x)
    if len(on) < 16:
        return None
    level = float(np.sqrt(np.mean(x**2)))
    if level <= 1e-9 or float(on.std()) / level < ONSET_MIN_VARIATION:
        return None
    hop_s = _ONSET_HOP / rate
    lo = max(1, int(60 / TEMPO_MAX_BPM / hop_s))
    hi = min(len(on) - 1, int(60 / TEMPO_MIN_BPM / hop_s) + 1)
    if hi <= lo + 1:
        return None
    ac = np.correlate(on, on, "full")[len(on) - 1 :]
    band = ac[lo:hi]
    mean_abs = float(np.mean(np.abs(band))) if band.size else 0.0
    if mean_abs <= 0:
        return None
    confidence = float(band.max() / (mean_abs + 1e-12))
    if confidence < TEMPO_MIN_CONFIDENCE:
        return None
    lag = lo + int(np.argmax(band))
    best = (float("-inf"), float(lag), 0.0)
    for period in np.linspace(lag * (1 - _FINE_SPAN), lag * (1 + _FINE_SPAN), _FINE_STEPS).tolist():
        phases = np.arange(0.0, float(period), 0.5)
        scores = _beat_scores(on, float(period), phases)
        k = int(np.argmax(scores))
        if scores[k] > best[0]:
            best = (float(scores[k]), float(period), float(phases[k]))
    _, period, phase = best
    return Beat(bpm=60.0 / (period * hop_s), start=phase * _ONSET_HOP, confidence=confidence)


def _bar_cut(x: np.ndarray, rate: int, crossfade_s: float, beat: Beat | None) -> tuple[int, int, int] | None:
    """(start, loop length, bars) for a clip with a beat, or None when it is not cut to bars."""
    if beat is None:
        return None
    bar = BEATS_PER_BAR * 60.0 * rate / beat.bpm
    n = int(rate * crossfade_s)
    s = round(beat.start)
    bars = int((len(x) - s - n) // bar)
    if bars < 2:
        return None
    return s, round(bars * bar), bars


def bar_aligned(x: np.ndarray, rate: int, crossfade_s: float = DEFAULT_CROSSFADE_S) -> np.ndarray:
    """Cut ``x`` so that, after ``make_loop`` folds the crossfade away, it is whole bars long.

    It starts on the first beat and keeps as many whole bars as fit, plus the
    crossfade tail. A clip with no clear steady beat is returned unchanged.
    """
    cut = _bar_cut(x, rate, crossfade_s, estimate_beat(x, rate))
    if cut is None:
        return x
    s, length, _ = cut
    return x[s : s + length + int(rate * crossfade_s)]


def make_loop(x: np.ndarray, rate: int, crossfade_s: float = DEFAULT_CROSSFADE_S) -> np.ndarray:
    """Fold the last ``crossfade_s`` into the first, equal power, so back to back there is no seam.

    The tail is then dropped: the loop's last sample is the one just before the
    tail began, and the blended head carries on from it. Call it once per clip.
    """
    n = int(rate * crossfade_s)
    if n <= 0 or len(x) < 3 * n:
        return x
    body = np.array(x[: len(x) - n], dtype=np.float64, copy=True)
    tail = np.asarray(x[len(x) - n :], dtype=np.float64)
    t = (np.arange(n) + 0.5) / n
    if body.ndim == 2:
        t = t[:, None]
    body[:n] = body[:n] * np.sin(0.5 * np.pi * t) + tail * np.cos(0.5 * np.pi * t)
    return body


def loop_to_bars(
    x: np.ndarray, rate: int, crossfade_s: float = DEFAULT_CROSSFADE_S
) -> tuple[np.ndarray, LoopInfo]:
    """A clip -> a seamless loop (whole bars when it has a beat) and what was done to it.

    Pass the clip unlooped, exactly once (see the module notes).
    """
    beat = estimate_beat(x, rate)
    cut = _bar_cut(x, rate, crossfade_s, beat)
    if cut is not None:
        s, length, bars = cut
        x = x[s : s + length + int(rate * crossfade_s)]
    y = make_loop(x, rate, crossfade_s)
    if cut is None or beat is None:
        return y, LoopInfo(bpm=None, bars=None, loop_s=len(y) / rate if rate > 0 else 0.0, confidence=None)
    return y, LoopInfo(bpm=beat.bpm, bars=bars, loop_s=len(y) / rate, confidence=beat.confidence)


def normalise_rms(x: np.ndarray, dbfs: float = -24.0, knee: float = 0.8) -> np.ndarray:
    """RMS to ``dbfs`` (not peak); a peak past ``knee`` is rounded off softly toward full scale, never clipped."""
    a = np.asarray(x, dtype=np.float64)
    rms = float(np.sqrt(np.mean(a**2))) if a.size else 0.0
    if rms <= 1e-9:
        return a.copy()
    y = a * (10 ** (dbfs / 20) / rms)
    over = np.abs(y) > knee
    if over.any():
        y = y.copy()
        y[over] = np.sign(y[over]) * (knee + (1 - knee) * np.tanh((np.abs(y[over]) - knee) / (1 - knee)))
    return y
