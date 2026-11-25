"""audio_loops: bar-exact seamless loops from a generated clip, on fake audio only."""
from __future__ import annotations

import time

import numpy as np
import pytest

import audio_loops as al

RATE = 16000


def _pulses(bpm: float, seconds: float, rate: int = RATE, offset: float = 0.123, seed: int = 0) -> np.ndarray:
    """A click track with an accent every bar, over a little noise, starting ``offset`` s in."""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    x = 0.02 * rng.standard_normal(n)
    beat = 60.0 * rate / bpm
    k = 0
    while offset * rate + k * beat < n:
        i = round(offset * rate + k * beat)
        length = min(n - i, int(0.08 * rate))
        env = np.exp(-np.arange(length) / (0.015 * rate))
        x[i : i + length] += (1.0 if k % 4 == 0 else 0.6) * env * np.sin(2 * np.pi * 90 * np.arange(length) / rate)
        k += 1
    return x


def _click_times(x: np.ndarray, rate: int = RATE) -> np.ndarray:
    """Where each click starts: the first sample of each run over half the peak level."""
    loud = np.abs(x) > 0.3
    starts = np.nonzero(loud & ~np.concatenate([[False], loud[:-1]]))[0]
    keep = [starts[0]]
    for s in starts[1:]:
        if s - keep[-1] > 0.15 * rate:  # one onset per click, not one per cycle of its tone
            keep.append(s)
    return np.asarray(keep) / rate


def _tempo_match(found: float, true: float, tol_pct: float = 0.1) -> bool:
    """Within ``tol_pct`` of the tempo, or of its half or double (a bar of 4 such beats is still whole beats)."""
    return any(abs(found - m * true) / (m * true) * 100 <= tol_pct for m in (0.5, 1.0, 2.0))


@pytest.mark.parametrize(("bpm", "seconds"), [(122, 23.4), (90, 19.7), (140, 21.3), (75.5, 27.1), (118, 29.3)])
def test_a_beat_is_found_and_the_loop_is_whole_bars(bpm, seconds):
    x = _pulses(bpm, seconds)
    beat = al.estimate_beat(x, RATE)
    assert beat is not None and _tempo_match(beat.bpm, bpm)
    assert beat.confidence >= al.TEMPO_MIN_CONFIDENCE

    loop, info = al.loop_to_bars(x, RATE)
    assert info.bars is not None and info.bars >= 2 and info.bpm == pytest.approx(beat.bpm)
    beats = len(loop) / (60.0 * RATE / bpm)
    assert abs(beats - round(beats)) < 0.03, beats  # whole beats of the true tempo
    assert round(beats) % al.BEATS_PER_BAR == 0  # and whole bars
    assert info.loop_s == pytest.approx(len(loop) / RATE)


@pytest.mark.parametrize(("bpm", "seconds"), [(122, 23.4), (101.3, 25.0)])
def test_played_back_to_back_the_beat_does_not_stumble(bpm, seconds):
    loop, _ = al.loop_to_bars(_pulses(bpm, seconds), RATE)
    tiled = np.concatenate([loop, loop, loop])
    gaps = np.diff(_click_times(tiled))
    period = 60.0 / bpm
    # Every gap, including the two across a seam, is a whole number of beats to within 15 ms.
    off = np.abs(gaps - np.round(gaps / period) * period)
    assert off.max() < 0.015, off.max()
    again = al.estimate_beat(tiled, RATE)
    assert again is not None and _tempo_match(again.bpm, bpm)


def test_an_arbitrary_length_crossfade_alone_would_stumble():
    """The failure this module exists for: 48.8 beats looped lands ~80% of a beat off the grid each repeat."""
    bpm, seconds = 122, 23.4
    x = _pulses(bpm, seconds, offset=0.0)
    plain = al.make_loop(x, RATE)
    gaps = np.diff(_click_times(np.concatenate([plain, plain])))
    period = 60.0 / bpm
    assert np.abs(gaps - np.round(gaps / period) * period).max() > 0.05


def test_the_seam_jump_is_no_bigger_than_the_musics_own_steps():
    rng = np.random.default_rng(3)
    t = np.arange(int(24.7 * RATE)) / RATE
    pad = 0.2 * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 277.2 * t) + np.sin(2 * np.pi * 329.6 * t))
    x = pad + _pulses(118, 24.7, seed=4) + 0.01 * rng.standard_normal(t.size)
    loop, info = al.loop_to_bars(x, RATE)
    assert info.bars is not None
    p99 = np.percentile(np.abs(np.diff(loop)), 99)
    assert abs(loop[0] - loop[-1]) <= p99


@pytest.mark.parametrize("kind", ["noise", "tone", "silence"])
def test_no_beat_means_no_cut_only_the_crossfade(kind):
    rng = np.random.default_rng(1)
    n = 20 * RATE
    x = {
        "noise": 0.3 * rng.standard_normal(n),
        "tone": 0.3 * np.sin(2 * np.pi * 440 * np.arange(n) / RATE),
        "silence": np.zeros(n),
    }[kind]
    assert al.estimate_beat(x, RATE) is None
    assert al.bar_aligned(x, RATE) is x
    loop, info = al.loop_to_bars(x, RATE)
    assert info.bpm is None and info.bars is None
    assert len(loop) == n - int(RATE * al.DEFAULT_CROSSFADE_S)


def test_too_short_to_hold_two_bars_is_left_alone():
    x = _pulses(80, 4.0)  # 5.3 beats: not two whole bars plus the tail
    assert al.bar_aligned(x, RATE) is x
    assert al.loop_to_bars(x, RATE)[1].bars is None


def test_loop_exactly_once():
    """A clip is kept unlooped and looped when used: a second fold eats another crossfade off whole bars."""
    bpm = 122
    once, info = al.loop_to_bars(_pulses(bpm, 23.4), RATE)
    bar = al.BEATS_PER_BAR * 60.0 * RATE / bpm
    assert abs(len(once) / bar - info.bars) < 0.01
    twice = al.make_loop(once, RATE)
    assert len(twice) == len(once) - int(RATE * al.DEFAULT_CROSSFADE_S)
    assert abs(len(twice) / bar - round(len(twice) / bar)) > 0.1  # no longer whole bars


def test_make_loop_is_equal_power_and_keeps_stereo():
    n = int(RATE * al.DEFAULT_CROSSFADE_S)
    x = np.ones((4 * n, 2))
    y = al.make_loop(x, RATE)
    assert y.shape == (3 * n, 2)
    # sin + cos peaks at sqrt(2) halfway for identical material: power, not amplitude, is kept.
    assert y[n // 2, 0] == pytest.approx(np.sqrt(2), abs=1e-3)
    assert al.make_loop(np.ones(10), RATE) is not None  # too short to fold: returned as is


def test_normalise_rms_targets_rms_and_rounds_peaks_softly():
    rng = np.random.default_rng(2)
    x = 0.01 * rng.standard_normal(RATE)
    y = al.normalise_rms(x, -24.0)
    assert 20 * np.log10(np.sqrt(np.mean(y**2))) == pytest.approx(-24.0, abs=0.05)
    # A square wave at -3 dBFS RMS has every sample at 0.71, under the knee: untouched by it.
    sq = np.where(np.arange(RATE) % 100 < 50, 1.0, -1.0)
    assert np.abs(al.normalise_rms(sq, -3.0)).max() == pytest.approx(10 ** (-3 / 20))
    # At -1 dBFS every sample (0.89) is past the 0.8 knee: rounded toward full scale, not clipped.
    k = np.abs(al.normalise_rms(sq, -1.0))
    assert 0.8 < k.max() < 10 ** (-1 / 20) and k.max() < 1.0
    ramp = al.normalise_rms(np.linspace(-1, 1, 1001), -3.0)
    assert np.all(np.diff(ramp) > 0) and np.abs(ramp).max() <= 1.0  # monotonic, never past full scale
    assert not al.normalise_rms(np.zeros(10)).any()


def test_a_30_second_clip_at_32k_prepares_in_under_a_second():
    x = _pulses(122, 30.0, rate=32000)
    t0 = time.perf_counter()
    _, info = al.loop_to_bars(x, 32000)
    elapsed = time.perf_counter() - t0
    assert info.bars is not None
    assert elapsed < 1.0, f"{elapsed:.2f}s"
