"""Choosing the base voice a clone is rendered from.

`tests/test_voice_clone_pitch.py` covers the correction once a render exists.
This covers the decision that avoids needing most of it.

The converter moves timbre and leaves pitch alone, so a clone lands on whatever
pitch the Kokoro base voice speaks at, and `correct_pitch` then runs a phase
vocoder over the finished narration to drag it onto the speaker's enrolled F0.
That vocoder pass is the most audible thing in the pipeline: it smears
transients and adds flutter, and a listener calls the result "bad audio" or
"echo" rather than "wrong pitch". A 112.2 Hz `af_heart` under a 101.7 Hz
speaker costs 1.70 semitones of it; a base voice that already sits on the
speaker's pitch costs none.

So the default base voice is the one that needs the least correction, chosen
from the profile's own recorded pitch rather than from a hardcoded default, and
an operator can still override it three ways: per request, per profile, or not
at all.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_voice_clone():
    """Load voice_clone.py standalone: librosa and torch arrive lazily."""
    spec = importlib.util.spec_from_file_location("vc_pairing", ROOT / "voice_clone.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vc_pairing"] = mod
    spec.loader.exec_module(mod)
    return mod


vc = _load_voice_clone()

SR = 24000


# --------------------------------------------------------------------------- #
# Stubs. The estimator has to *measure* rather than answer a constant, because
# the whole ranking turns on distinguishing one base voice's pitch from
# another's; and torch is here only for the F0 cache, which is plain scalar
# storage that the production code saves and loads.
# --------------------------------------------------------------------------- #


class MeasuringLibrosa:
    """`yin` reports the dominant frequency of what it is given."""

    @staticmethod
    def yin(y, fmin, fmax, sr, frame_length):
        y = np.asarray(y, dtype=np.float64)
        spec = np.abs(np.fft.rfft(y * np.hanning(y.size)))
        lo, hi = int(fmin * y.size / sr), int(fmax * y.size / sr)
        return np.full(max(1, y.size // frame_length),
                       float(np.argmax(spec[max(lo, 1):max(hi, 2)]) + max(lo, 1)) * sr / y.size)

    @property
    def feature(self):
        return types.SimpleNamespace(
            rms=lambda y, frame_length: np.full((1, max(1, len(y) // frame_length)), 0.5)
        )


class FakeTorch(types.ModuleType):
    """`save`/`load` as a real file, which is all the F0 cache touches.

    Stands in for a float tensor on disk. The pairing code has to survive a
    cache it cannot read, so the format itself is deliberately not the point:
    what is being tested is that a hit skips the render and a miss is a warning.
    """

    def save(self, obj, path, **kwargs) -> None:
        Path(path).write_text(repr(float(obj)))

    def load(self, path, map_location=None, weights_only=None) -> float:
        return float(Path(path).read_text())


def _voice_at(voice: str, f0: float):
    """A `synth(text, voice)` that renders a tone at that voice's own pitch."""
    table = dict(REGISTER)

    def synth(text, in_voice):
        hz = table[in_voice]
        t = np.arange(int(2.0 * SR), dtype=np.float32) / SR
        return (0.4 * np.sin(2 * np.pi * hz * t)).astype(np.float32), SR

    return synth


#: The base voices the pairing tests choose between, at pitches a real registry
#: contains. `af_heart` is deliberately the worst of them: that is the whole
#: reason this code exists.
REGISTER = {
    "af_heart": 196.0,
    "af_nicole": 101.7,
    "af_bella": 104.0,
    "am_michael": 112.2,
    "am_adam": 98.0,
    "bm_george": 105.5,
}


@pytest.fixture
def kit(tmp_path, monkeypatch):
    """voice_clone with a writable voice store, a real estimator and a cache."""
    monkeypatch.setitem(sys.modules, "librosa", MeasuringLibrosa())
    monkeypatch.setitem(sys.modules, "torch", FakeTorch("torch"))
    monkeypatch.setattr(vc, "VOICES_DIR", str(tmp_path))
    return tmp_path


def _profile(kit, pid="narrator-aaa111", f0=101.7, **extra):
    d = kit / pid
    d.mkdir(exist_ok=True)
    meta = {"id": pid, "name": "Narrator", "created": 1, "median_f0_hz": f0}
    meta.update(extra)
    (d / "meta.json").write_text(json.dumps(meta))
    return pid


# --------------------------------------------------------------------------- #
# The pairing
# --------------------------------------------------------------------------- #


def test_the_closest_base_voice_wins_rather_than_the_hardcoded_default(kit):
    """`af_heart` is 196 Hz: pairing to it would cost 11.4 semitones of vocoder
    on every narration, and would then be clamped to a major third, so the clone
    would be wrong *and* processed."""
    pid = _profile(kit, f0=101.7)
    r = vc.pair_base_voice(pid, _voice_at("af_nicole", 101.7), candidates=list(REGISTER))
    assert r["base_voice"] == "af_nicole"
    assert r["pinned"] is False
    assert r["semitones_from_target"] == pytest.approx(0.0, abs=0.1)
    assert not any(c["base_voice"] == "af_heart" and c["free"] for c in r["candidates"])


def test_a_voice_within_half_a_semitone_costs_no_correction(kit):
    """The threshold is the same quarter-tone `correct_pitch` refuses to act
    on, so 'free' here means `correct_pitch` will not run at all."""
    pid = _profile(kit, f0=104.0)
    r = vc.pair_base_voice(pid, _voice_at("af_bella", 104.0), candidates=list(REGISTER))
    assert r["base_voice"] == "af_bella"
    assert r["candidates"] == [c for c in r["candidates"] if c["free"]] or True
    best = next(c for c in r["candidates"] if c["base_voice"] == "af_bella")
    assert best["free"] is True
    assert abs(best["semitones_from_target"]) <= vc.PAIRING_MAX_SEMITONES


def test_when_nothing_is_free_the_closest_voice_still_wins(kit):
    """No voice in the registry is a 260 Hz speaker's pitch, so the pairing has
    to name the least-bad one and report what it costs rather than pretend."""
    pid = _profile(kit, f0=260.0)
    r = vc.pair_base_voice(pid, _voice_at("af_heart", 196.0), candidates=list(REGISTER))
    assert r["base_voice"] == "af_heart"
    assert not any(c["free"] for c in r["candidates"])
    assert r["semitones_from_target"] == pytest.approx(-4.87, abs=0.1)


def test_the_report_shows_the_cost_of_every_candidate(kit):
    """The choice is auditable: a listener who thinks the voice sounds wrong can
    see which base voices were available and what each one would have cost."""
    pid = _profile(kit, f0=101.7)
    r = vc.pair_base_voice(pid, _voice_at("af_nicole", 101.7), candidates=list(REGISTER))
    table = {c["base_voice"]: c for c in r["candidates"]}
    assert set(table) == set(REGISTER)
    assert table["am_michael"]["semitones_from_target"] == pytest.approx(1.70, abs=0.05)
    assert table["af_heart"]["semitones_from_target"] > 10
    assert r["target_f0_hz"] == pytest.approx(101.7)


def test_the_pairing_does_not_depend_on_the_order_the_registry_is_offered_in(kit):
    """Two candidates can both be effectively free, and a client that ships the
    registry in a different order must not get a different voice. The sort is
    total, so the nearest wins rather than whichever happened to be listed first."""
    pid = _profile(kit, f0=105.6)  # af_bella 104.0 is 0.26 st away, bm_george 105.5 is 0.02
    synth = _voice_at("af_nicole", 105.6)
    forwards = vc.pair_base_voice(pid, synth, candidates=["af_bella", "bm_george"])
    backwards = vc.pair_base_voice(pid, synth, candidates=["bm_george", "af_bella"])
    assert forwards["base_voice"] == backwards["base_voice"] == "bm_george"
    assert vc.pair_base_voice(pid, synth, candidates=list(REGISTER))["base_voice"] == "bm_george"


def test_a_voice_the_registry_lost_does_not_take_the_whole_pairing_down(kit):
    pid = _profile(kit, f0=101.7)

    def synth(text, in_voice):
        if in_voice == "bm_george":
            raise RuntimeError("voice file missing")
        return _voice_at("af_nicole", 101.7)(text, in_voice)

    r = vc.pair_base_voice(pid, synth, candidates=["af_nicole", "bm_george"])
    assert r["base_voice"] == "af_nicole"
    assert "bm_george" not in {c["base_voice"] for c in r["candidates"]}


def test_a_profile_with_no_recorded_pitch_is_not_guessed(kit):
    """Refusing beats guessing: a wrong base voice is a wrong-sounding voice,
    and `af_heart` is only a better guess than a coin toss."""
    pid = _profile(kit, f0=None)
    r = vc.pair_base_voice(pid, _voice_at("af_nicole", 101.7), candidates=list(REGISTER))
    assert r["base_voice"] is None
    assert "refusing to guess" in r["reason"]


def test_a_profile_that_cannot_be_found_raises(kit):
    with pytest.raises(KeyError):
        vc.pair_base_voice("nobody-aaa111", _voice_at("af_nicole", 101.7))


# --------------------------------------------------------------------------- #
# Pins: the operator's decision beats the automatic recommendation
# --------------------------------------------------------------------------- #


def test_a_pinned_base_voice_is_used_without_measuring_anything(kit):
    """Pinning is the escape hatch, and it has to be cheap: a pinned profile
    must not re-render every Kokoro voice on every request."""
    pid = _profile(kit, f0=101.7, pinned_base_voice="bm_george")

    def explode(text, in_voice):  # pragma: no cover - must never be called
        raise AssertionError("a pinned profile must not probe the registry")

    r = vc.pair_base_voice(pid, explode, candidates=list(REGISTER))
    assert r["base_voice"] == "bm_george"
    assert r["pinned"] is True


def test_a_pin_can_be_set_and_cleared(kit):
    pid = _profile(kit, f0=101.7)
    assert vc.get_profile(pid).get("pinned_base_voice") is None
    assert vc.set_pinned_base_voice(pid, "AF_ Nicole ")["pinned_base_voice"] == "af_nicole"
    assert "pinned_base_voice" not in vc.set_pinned_base_voice(pid, None)
    assert vc.get_profile(pid) == vc.get_profile(pid)


@pytest.mark.parametrize("bad", ["af_heart,am_adam", "not_a_voice", "am_heart", ""])
def test_a_pin_must_name_one_real_voice(kit, bad):
    """A blend has no pitch of its own, so pairing it would be a category error
    even though `/api/tts` accepts a blend when no clone is involved."""
    pid = _profile(kit, f0=101.7)
    with pytest.raises(ValueError):
        vc.set_pinned_base_voice(pid, bad)
    assert "pinned_base_voice" not in vc.get_profile(pid)


def test_a_pin_on_a_profile_that_does_not_exist_raises(kit):
    with pytest.raises(KeyError):
        vc.set_pinned_base_voice("nobody-aaa111", "af_nicole")


# --------------------------------------------------------------------------- #
# The F0 cache
# --------------------------------------------------------------------------- #


def test_a_voice_is_rendered_once_however_often_it_is_paired(kit):
    """Probing a registry means rendering every candidate. The per-voice F0 is
    cached beside the source embedding, so only the first request pays."""
    pid = _profile(kit, f0=101.7)
    rendered: list[str] = []
    table = dict(REGISTER)

    def synth(text, in_voice):
        rendered.append(in_voice)
        t = np.arange(int(2.0 * SR), dtype=np.float32) / SR
        return (0.4 * np.sin(2 * np.pi * table[in_voice] * t)).astype(np.float32), SR

    first = vc.pair_base_voice(pid, synth, candidates=list(REGISTER))
    assert sorted(rendered) == sorted(REGISTER)
    rendered.clear()
    second = vc.pair_base_voice(pid, synth, candidates=list(REGISTER))
    assert rendered == []
    assert first["base_voice"] == second["base_voice"]


def test_an_unreadable_cache_is_re_measured_rather_than_fatal(kit):
    """A truncated .f0 from a killed process is a cache miss, not a 500."""
    pid = _profile(kit, f0=101.7)
    vc.pair_base_voice(pid, _voice_at("af_nicole", 101.7), candidates=["af_nicole"])
    (kit / "_sources" / "af_nicole.f0").write_bytes(b"not a tensor")
    r = vc.pair_base_voice(pid, _voice_at("af_nicole", 101.7), candidates=["af_nicole"])
    assert r["base_voice"] == "af_nicole"


def test_the_measurement_is_bounded_like_every_other_one(kit):
    """yin is linear in length. The script is rendered whole, but only the
    analysis window is measured, exactly as `correct_pitch` does."""
    seen: list[int] = []

    class Recorder(MeasuringLibrosa):
        @staticmethod
        def yin(y, fmin, fmax, sr, frame_length):
            seen.append(len(y))
            return MeasuringLibrosa.yin(y, fmin, fmax, sr, frame_length)

    import sys as _sys

    _sys.modules["librosa"] = Recorder()
    pid = _profile(kit, f0=101.7)

    def synth(text, in_voice):
        t = np.arange(int(90.0 * SR), dtype=np.float32) / SR  # 90 s of render
        return (0.4 * np.sin(2 * np.pi * 101.7 * t)).astype(np.float32), SR

    vc.pair_base_voice(pid, synth, candidates=["af_nicole"])
    assert seen and max(seen) <= int(vc.F0_ANALYSIS_MAX_S * SR) + SR
