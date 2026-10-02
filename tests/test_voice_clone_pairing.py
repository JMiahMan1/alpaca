"""Choosing the base voice a clone is rendered from.

`tests/test_voice_clone_pitch.py` covers the correction once a render exists.
This covers the decision that avoids needing most of it.

`correct_pitch` measures how far the finished narration landed from the enrolled
F0 and vocoder-shifts the difference. That shift is the most audible thing in the
pipeline: it smears transients and adds flutter, and a listener calls the result
"bad audio" or "echo" rather than "wrong pitch". So the default base voice is the
one whose *converted* audio lands nearest the speaker's enrolled pitch, measured
by converting candidates rather than by reading their own register - the
converter does not preserve a base voice's pitch, and by how much it moves it is
not monotonic in the voice's own F0. See `converted_voice_f0` for the numbers
that made the difference.

The choice is then overridable three ways: per request, per profile, or not at
all.
"""

from __future__ import annotations

import importlib.util
import json
import math
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
    """`yin` reports the dominant frequency of what it is given, to a tenth of a
    hertz.

    Sub-bin accuracy is not decoration here: the ranking turns on differences of a
    few hundredths of a semitone, and a stub that could only answer to the nearest
    FFT bin would decide some of the tests for reasons that have nothing to do
    with the code.
    """

    @staticmethod
    def yin(y, fmin, fmax, sr, frame_length):
        y = np.asarray(y, dtype=np.float64)
        spec = np.abs(np.fft.rfft(y * np.hanning(y.size)))
        lo, hi = max(int(fmin * y.size / sr), 1), max(int(fmax * y.size / sr), 2)
        k = int(np.argmax(spec[lo:hi])) + lo
        # Parabolic interpolation over the three bins around the peak.
        a, b, c = spec[max(k - 1, 0)], spec[k], spec[min(k + 1, len(spec) - 1)]
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if denom else 0.0
        return np.full(max(1, y.size // frame_length), (k + delta) * sr / y.size)

    @property
    def feature(self):
        return types.SimpleNamespace(
            rms=lambda y, frame_length: np.full((1, max(1, len(y) // frame_length)), 0.5)
        )


class FakeTorch(types.ModuleType):
    """`save`/`load` as a real file, which is all the F0 cache touches.

    Stands in for the `{base_f0_hz, converted_f0_hz}` pair the production code
    writes. The pairing code has to survive a cache it cannot read, so the format
    itself is deliberately not the point: what is being tested is that a hit skips
    the render and a miss is a warning.
    """

    def save(self, obj, path, **kwargs) -> None:
        Path(path).write_text(json.dumps(obj))

    def load(self, path, map_location=None, weights_only=None) -> dict:
        return json.loads(Path(path).read_text())


def _tone(hz: float, seconds: float = 2.0, sr: int = SR):
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    return (0.4 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _voice_at(voice: str, f0: float):
    """A `synth(text, voice)` that renders a tone at that voice's own pitch."""
    table = dict(REGISTER)

    def synth(text, in_voice):
        return _tone(table[in_voice]), SR

    return synth


#: The base voices the pairing tests choose between, at the pitches a real
#: registry contains. The second column is where each one *lands after
#: conversion*, taken from a real profile, and it is deliberately not ordered the
#: same way as the first: `am_echo` is the nearest voice on its own pitch and the
#: second worst after conversion, while `am_liam` is the furthest of the three
#: male voices on its own pitch and the best-placed one. Pairing on the first
#: column chose `am_echo` and cost 2.02 semitones of vocoder.
REGISTER = {
    "af_heart": 188.7,
    "am_echo": 106.9,
    "am_liam": 125.5,
    "af_nicole": 151.2,
}

CONVERTED = {
    "af_heart": 99.6,
    "am_echo": 116.0,
    "am_liam": 104.9,
    "af_nicole": 101.7,
}


def _fake_conversion_stack():
    """`source_se` / `target_se` / `convert` replaced with a converter that moves
    pitch the way the real one does: not predictably from the base voice.

    `source_se` hands back the voice's own name so `convert` can look up where
    that voice lands. That is the whole coupling under test - the pairing has to
    render and convert the *same* candidate, because a base voice's own pitch and
    its converted pitch are two different numbers.
    """
    def source_se(voice, synth):
        return voice

    def target_se(pid):
        return pid

    def convert(audio, sr, src_se, tgt_se, tau=vc.DEFAULT_TAU, seed=vc.CONVERT_SEED):
        return _tone(CONVERTED[src_se], seconds=len(audio) / sr, sr=sr)

    return source_se, target_se, convert


@pytest.fixture
def kit(tmp_path, monkeypatch):
    """voice_clone with a writable voice store, a real estimator and a cache."""
    monkeypatch.setitem(sys.modules, "librosa", MeasuringLibrosa())
    monkeypatch.setitem(sys.modules, "torch", FakeTorch("torch"))
    monkeypatch.setattr(vc, "VOICES_DIR", str(tmp_path))
    src, tgt, conv = _fake_conversion_stack()
    monkeypatch.setattr(vc, "source_se", src)
    monkeypatch.setattr(vc, "target_se", tgt)
    monkeypatch.setattr(vc, "convert", conv)
    # Similarity needs the real reference encoder. Without it every probe is
    # uncomparable, which is the case the pitch-only tests below describe.
    monkeypatch.setattr(vc, "clone_similarity", lambda audio, sr, pid: None)
    return tmp_path


def _profile(kit, pid="narrator-aaa111", f0=103.2, **extra):
    d = kit / pid
    d.mkdir(exist_ok=True)
    meta = {"id": pid, "name": "Narrator", "created": 1, "median_f0_hz": f0}
    meta.update(extra)
    (d / "meta.json").write_text(json.dumps(meta))
    return pid


# --------------------------------------------------------------------------- #
# The pairing: it has to rank the converted pitch, not the base pitch
# --------------------------------------------------------------------------- #


def test_the_choice_follows_the_converted_pitch_and_not_the_base_pitch(kit):
    """The regression this whole change exists for.

    `am_echo` speaks 106.9 Hz on its own, 0.6 semitones from this speaker, and is
    therefore the obvious answer to "which base voice is nearest". But it lands at
    116.0 Hz once converted, so `correct_pitch` would vocode a whole narration by
    2.02 semitones to drag it back. Ranking on the base voice's own pitch picks the
    expensive one; ranking on what the voice will actually sound like picks
    `af_nicole`, which lands at 101.7 Hz -- 0.25 semitones out, inside the free
    band, so the vocoder never runs.
    """
    pid = _profile(kit, f0=103.2)
    r = vc.pair_base_voice(pid, _voice_at("am_liam", 125.5), candidates=list(REGISTER))

    assert r["base_voice"] == "af_nicole"
    assert r["semitones_from_target"] == pytest.approx(-0.26, abs=0.05)
    assert r["candidates"][0]["free"] is True

    # The regression itself: the nearest voice *by its own pitch* must never win.
    # Computed from base_f0_hz rather than a `base_semitones` field, because the
    # candidate dict only reports the distance it was ranked on.
    table = {c["base_voice"]: c for c in r["candidates"]}
    base_st = lambda f: abs(12 * math.log2(f / 103.2))  # noqa: E731
    assert base_st(table["am_echo"]["base_f0_hz"]) < base_st(table["af_nicole"]["base_f0_hz"]), (
        "am_echo is expected to be the nearest on base pitch -- if that stops being true, "
        "this test is no longer testing what it claims"
    )
    assert r["base_voice"] != "am_echo"
    assert table["am_echo"]["converter_offset_semitones"] > 1.0, "am_echo should be the expensive one, so the point stands"


def test_the_report_says_what_the_converter_did_to_each_voice(kit):
    """The offset is the part that was invisible, and it is the reason a base
    voice's own pitch cannot be used to pick one. A listener who thinks the voice
    sounds wrong can now see what each candidate would have cost, and what the
    conversion did to it."""
    pid = _profile(kit, f0=103.2)
    r = vc.pair_base_voice(pid, _voice_at("am_liam", 125.5), candidates=list(REGISTER))
    table = {c["base_voice"]: c for c in r["candidates"]}
    assert set(table) == set(REGISTER)
    assert table["am_echo"]["base_f0_hz"] == 106.9
    assert table["am_echo"]["converted_f0_hz"] == 116.0
    assert table["am_echo"]["converter_offset_semitones"] == pytest.approx(1.42, abs=0.05)
    assert table["am_echo"]["semitones_from_target"] == pytest.approx(2.02, abs=0.05)
    assert table["am_echo"]["free"] is False
    assert table["am_liam"]["converter_offset_semitones"] == pytest.approx(-3.13, abs=0.05)

    # The winner is af_nicole, not am_liam: af_nicle lands nearer the target once
    # converted (101.7 Hz vs 104.9 Hz for a 103.2 Hz speaker), so the report has
    # to follow the converted column too.
    assert r["base_voice"] == "af_nicole"
    assert r["converted_f0_hz"] == 101.7
    assert r["target_f0_hz"] == pytest.approx(103.2)


def test_the_conversion_probe_runs_at_the_tau_the_narration_will_use(kit):
    """The offset depends on how hard the conversion pulls, so a probe at a
    different tau would be measuring a different pipeline."""
    seen: list[float] = []
    real_convert = vc.convert

    def spy(audio, sr, src_se, tgt_se, tau=vc.DEFAULT_TAU, seed=vc.CONVERT_SEED):
        seen.append(tau)
        return real_convert(audio, sr, src_se, tgt_se, tau=tau, seed=seed)

    vc.convert = spy
    try:
        pid = _profile(kit, f0=103.2)
        vc.pair_base_voice(pid, _voice_at("am_liam", 125.5), candidates=list(REGISTER), tau=0.15)
    finally:
        vc.convert = real_convert
    assert seen and all(t == 0.15 for t in seen)


def test_a_paired_voice_is_one_correct_pitch_will_not_touch(kit):
    """The whole point: the pairing's 'free' band is the same band the correction
    refuses to act inside, so a free pairing means no vocoder runs at all."""
    pid = _profile(kit, f0=103.2)
    r = vc.pair_base_voice(pid, _voice_at("am_liam", 125.5), candidates=list(REGISTER))
    best = next(c for c in r["candidates"] if c["base_voice"] == r["base_voice"])
    assert best["free"] is True
    assert abs(best["semitones_from_target"]) <= vc.PAIRING_MAX_SEMITONES
    assert vc.PAIRING_MAX_SEMITONES == vc.MIN_PITCH_SHIFT_SEMITONES


def test_when_nothing_is_free_the_closest_voice_still_wins(kit):
    """No voice in the registry is a 260 Hz speaker's pitch, so the pairing has
    to name the least-bad one and report what it costs rather than pretend."""
    pid = _profile(kit, f0=260.0)
    r = vc.pair_base_voice(pid, _voice_at("af_heart", 188.7), candidates=list(REGISTER))
    assert r["base_voice"] == "am_echo"
    assert not any(c["free"] for c in r["candidates"])
    assert r["semitones_from_target"] == pytest.approx(-13.97, abs=0.1)


def test_the_pairing_does_not_depend_on_the_order_the_registry_is_offered_in(kit):
    """Two candidates can both be effectively free, and a client that ships the
    registry in a different order must not get a different voice. The sort is
    total, so the nearest wins rather than whichever happened to be listed
    first."""
    pid = _profile(kit, f0=103.2)  # am_liam 104.9 is 0.28 st, af_nicole 101.7 is 0.23
    synth = _voice_at("am_liam", 125.5)
    forwards = vc.pair_base_voice(pid, synth, candidates=["am_liam", "af_nicole"])
    backwards = vc.pair_base_voice(pid, synth, candidates=["af_nicole", "am_liam"])
    assert forwards["base_voice"] == backwards["base_voice"] == "af_nicole"
    assert vc.pair_base_voice(pid, synth, candidates=list(REGISTER))["base_voice"] == "af_nicole"


def test_a_voice_the_registry_lost_does_not_take_the_whole_pairing_down(kit):
    pid = _profile(kit, f0=103.2)

    def synth(text, in_voice):
        if in_voice == "af_nicole":
            raise RuntimeError("voice file missing")
        return _tone(REGISTER[in_voice]), SR

    r = vc.pair_base_voice(pid, synth, candidates=["am_liam", "af_nicole"])
    assert r["base_voice"] == "am_liam"
    assert "af_nicole" not in {c["base_voice"] for c in r["candidates"]}


def test_a_voice_the_converter_failed_on_does_not_take_the_pairing_down(kit):
    """A candidate can render and still fail to convert, which is a different
    failure from a missing voice file and has to be survived the same way."""
    real_convert = vc.convert

    def flaky(audio, sr, src_se, tgt_se, tau=vc.DEFAULT_TAU, seed=vc.CONVERT_SEED):
        if src_se == "af_nicole":
            raise RuntimeError("converter ran out of memory")
        return real_convert(audio, sr, src_se, tgt_se, tau=tau, seed=seed)

    vc.convert = flaky
    try:
        pid = _profile(kit, f0=103.2)
        r = vc.pair_base_voice(pid, _voice_at("am_liam", 125.5), candidates=list(REGISTER))
    finally:
        vc.convert = real_convert
    assert r["base_voice"] == "am_liam"
    assert "af_nicole" not in {c["base_voice"] for c in r["candidates"]}


def test_a_profile_with_no_recorded_pitch_is_not_guessed(kit):
    """Refusing beats guessing: a wrong base voice is a wrong-sounding voice,
    and the fallback is only a better guess than a coin toss."""
    pid = _profile(kit, f0=None)
    r = vc.pair_base_voice(pid, _voice_at("am_liam", 125.5), candidates=list(REGISTER))
    assert r["base_voice"] is None
    assert "refusing to guess" in r["reason"]


def test_a_profile_that_cannot_be_found_raises(kit):
    with pytest.raises(KeyError):
        vc.pair_base_voice("nobody-aaa111", _voice_at("am_liam", 125.5))


# --------------------------------------------------------------------------- #
# Pins: the operator's decision beats the automatic recommendation
# --------------------------------------------------------------------------- #


def test_a_pinned_base_voice_is_used_without_measuring_anything(kit):
    """Pinning is the escape hatch, and it has to be cheap: a pinned profile must
    not render and convert every Kokoro voice on every request."""
    pid = _profile(kit, f0=103.2, pinned_base_voice="af_nicole")

    def explode(text, in_voice):  # pragma: no cover - must never be called
        raise AssertionError("a pinned profile must not probe the registry")

    r = vc.pair_base_voice(pid, explode, candidates=list(REGISTER))
    assert r["base_voice"] == "af_nicole"
    assert r["pinned"] is True


def test_a_pin_can_be_set_and_cleared(kit):
    pid = _profile(kit, f0=103.2)
    assert vc.get_profile(pid).get("pinned_base_voice") is None
    assert vc.set_pinned_base_voice(pid, "am_liam")["pinned_base_voice"] == "am_liam"
    assert "pinned_base_voice" not in vc.set_pinned_base_voice(pid, None)
    assert vc.get_profile(pid) == vc.get_profile(pid)


@pytest.mark.parametrize("bad", ["af_heart,am_adam", "not_a_voice", "am_heart", ""])
def test_a_pin_must_name_one_real_voice(kit, bad):
    """A blend has no pitch of its own, so pairing it would be a category error
    even though `/api/tts` accepts a blend when no clone is involved."""
    pid = _profile(kit, f0=103.2)
    with pytest.raises(ValueError):
        vc.set_pinned_base_voice(pid, bad)
    assert "pinned_base_voice" not in vc.get_profile(pid)


def test_a_pin_on_a_profile_that_does_not_exist_raises(kit):
    with pytest.raises(KeyError):
        vc.set_pinned_base_voice("nobody-aaa111", "af_nicole")


# --------------------------------------------------------------------------- #
# The F0 cache
# --------------------------------------------------------------------------- #


def test_a_voice_is_measured_once_however_often_it_is_paired(kit):
    """Probing a registry means rendering and converting every candidate. The
    measurement is cached per voice *and* profile, so only the first request for
    a speaker pays."""
    pid = _profile(kit, f0=103.2)
    rendered: list[str] = []
    real_source_se = vc.source_se

    def counting_source_se(voice, synth):
        rendered.append(voice)
        return real_source_se(voice, synth)

    vc.source_se = counting_source_se
    try:
        first = vc.pair_base_voice(pid, _voice_at("am_liam", 125.5), candidates=list(REGISTER))
        assert sorted(rendered) == sorted(REGISTER)
        rendered.clear()
        second = vc.pair_base_voice(pid, _voice_at("am_liam", 125.5), candidates=list(REGISTER))
    finally:
        vc.source_se = real_source_se
    assert rendered == []
    assert first["base_voice"] == second["base_voice"] == first["base_voice"]


def test_the_cache_is_per_profile_not_just_per_voice(kit):
    """The same voice converts differently for every speaker, so one voice's
    measurement cannot answer for another profile. Sharing it would reintroduce
    exactly the bug this cache was widened to fix."""
    measured = vc.converted_voice_f0("narrator-aaa111", "am_liam", _voice_at("am_liam", 125.5))
    assert measured["converted_f0_hz"] == pytest.approx(104.9, abs=0.05)
    kit2 = kit / "other"
    kit2.mkdir()
    meta = {"id": "other", "name": "Other", "created": 1, "median_f0_hz": 103.2}
    (kit2 / "meta.json").write_text(json.dumps(meta))
    assert vc.converted_voice_f0("other", "am_liam", _voice_at("am_liam", 125.5))["converted_f0_hz"] == pytest.approx(104.9, abs=0.05)
    assert (kit / "_sources" / "am_liam~narrator-aaa111~0.3.converted.f0").is_file()
    assert (kit / "_sources" / "am_liam~other~0.3.converted.f0").is_file()


def test_the_cache_is_also_per_conversion_strength(kit):
    """`tau` is the strength of the conversion being measured, and a caller may
    ask for anything in 0.1..1.0. A cache written at one strength says nothing
    about where another strength will put the voice, so a shared key would hand
    back a ranking measured under different conditions than the narration it is
    ranking for -- silently, because the numbers look perfectly plausible."""
    for tau in (0.3, 0.9):
        vc.converted_voice_f0("narrator-aaa111", "am_liam", _voice_at("am_liam", 125.5), tau=tau)
    sources = sorted(p.name for p in (kit / "_sources").glob("am_liam~narrator-aaa111*"))
    assert sources == [
        "am_liam~narrator-aaa111~0.3.converted.f0",
        "am_liam~narrator-aaa111~0.9.converted.f0",
    ], sources


def test_a_stale_base_pitch_cache_is_not_read(kit):
    """The old cache held the base voice's own pitch, under a name that has no
    profile in it. It must not be mistaken for a measurement of the converted
    audio, which is a different number and the whole point of the pairing."""
    pid = _profile(kit, f0=103.2)
    vc.pair_base_voice(pid, _voice_at("am_liam", 125.5), candidates=["am_liam"])
    stale = kit / "_sources" / "am_liam.f0"
    stale.write_text(json.dumps(125.5))
    r = vc.pair_base_voice(pid, _voice_at("am_liam", 125.5), candidates=["am_liam"])
    assert r["converted_f0_hz"] == 104.9
    assert r["candidates"][0]["converted_f0_hz"] == 104.9


def test_an_unreadable_cache_is_re_measured_rather_than_fatal(kit):
    """A truncated file from a killed process is a cache miss, not a 500."""
    pid = _profile(kit, f0=103.2)
    vc.pair_base_voice(pid, _voice_at("am_liam", 125.5), candidates=["am_liam"])
    (kit / "_sources" / "am_liam~narrator-aaa111.converted.f0").write_bytes(b"not a measurement")
    r = vc.pair_base_voice(pid, _voice_at("am_liam", 125.5), candidates=["am_liam"])
    assert r["base_voice"] == "am_liam"
    assert r["converted_f0_hz"] == 104.9


def test_the_measurement_is_bounded_like_every_other_one(kit):
    """yin is linear in length, and pairing now converts as well as renders, so
    the probe window is deliberately shorter than the analysis window."""
    seen: list[int] = []

    class Recorder(MeasuringLibrosa):
        @staticmethod
        def yin(y, fmin, fmax, sr, frame_length):
            seen.append(len(y))
            return MeasuringLibrosa.yin(y, fmin, fmax, sr, frame_length)

    sys.modules["librosa"] = Recorder()
    pid = _profile(kit, f0=103.2)

    def synth(text, in_voice):
        return _tone(REGISTER[in_voice], seconds=90.0), SR  # 90 s of render

    vc.pair_base_voice(pid, synth, candidates=["am_liam"])
    assert seen and max(seen) <= int(vc.F0_ANALYSIS_MAX_S * SR) + SR
    assert max(seen) <= int(vc.PAIRING_PROBE_S * SR) + SR


# --------------------------------------------------------------------------- #
# Inside the free band, sounding like the speaker beats a closer pitch probe
# --------------------------------------------------------------------------- #


def test_among_free_voices_the_one_that_sounds_most_like_the_speaker_wins(kit, monkeypatch):
    """am_liam (0.28 st) and af_nicole (0.23 st) are both free. The pitch probe
    is not precise enough to separate them, so similarity decides."""
    scores = {104.9: 0.93, 101.7: 0.91}
    monkeypatch.setattr(
        vc, "clone_similarity", lambda audio, sr, pid: scores[round(vc.median_f0(audio, sr), 1)]
    )
    pid = _profile(kit, f0=103.2)
    r = vc.pair_base_voice(pid, _voice_at("am_liam", 125.5), candidates=["af_nicole", "am_liam"])
    assert r["base_voice"] == "am_liam"
    assert "similarity 0.930" in r["reason"]


def test_similarity_never_lets_a_voice_outside_the_free_band_win(kit, monkeypatch):
    """am_echo lands 2 semitones out. However much it sounds like the speaker,
    choosing it means vocoding the whole narration."""
    monkeypatch.setattr(
        vc, "clone_similarity",
        lambda audio, sr, pid: 0.99 if vc.median_f0(audio, sr) > 110 else 0.80,
    )
    pid = _profile(kit, f0=103.2)
    r = vc.pair_base_voice(pid, _voice_at("am_liam", 125.5), candidates=["am_echo", "af_nicole"])
    assert r["base_voice"] == "af_nicole"


def test_a_probe_cached_without_similarity_is_measured_again(kit, monkeypatch):
    monkeypatch.setattr(vc, "clone_similarity", lambda audio, sr, pid: 0.9)
    pid = _profile(kit, f0=103.2)
    path = kit / "_sources" / f"{vc._source_key('am_liam')}~{pid}~{vc.DEFAULT_TAU:g}.converted.f0"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"base_f0_hz": 125.5, "converted_f0_hz": 104.9}))
    probe = vc.converted_voice_f0(pid, "am_liam", _voice_at("am_liam", 125.5))
    assert probe["similarity"] == 0.9
    assert json.loads(path.read_text())["similarity"] == 0.9
