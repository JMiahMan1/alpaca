"""Pitch fidelity of a cloned voice.

`tests/test_voice_clone_identity.py` covers WHO a clip belongs to. This covers
the two things that decide whether a clone is *usable*, which turned out to be
different problems:

* TIMBRE is the converter's job, and it does it well - cosine to the speaker's
  own centroid rises from ~0.5 to ~0.85.
* PITCH is not. The converter leaves F0 alone, so a clone of a 101.7 Hz voice
  built on Kokoro's 112.2 Hz `am_michael` comes out 1.6 semitones sharp, and no
  value of `tau` moves it. That is enough for a listener to say "that is not
  me" about an otherwise good clone.

So the pitch is measured at enrolment, corrected after conversion, and both
numbers are reported. The tests below pin that contract hermetically: librosa
lives only in the audio image, so the estimator and the vocoder are injected.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_voice_clone():
    """Load voice_clone.py with librosa and soundfile stubbed out.

    The module imports both lazily inside functions precisely so that a machine
    without them can still import it; these tests lean on that.
    """
    spec = importlib.util.spec_from_file_location("vc_pitch", ROOT / "voice_clone.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vc_pitch"] = mod
    spec.loader.exec_module(mod)
    return mod


vc = _load_voice_clone()


# --------------------------------------------------------------------------- #
# A fake librosa: a yin that answers with a caller-chosen F0, and a vocoder
# that is a plain resample (which is what pitch_shift's *effect* is, minus the
# phase preservation we do not need to assert).
# --------------------------------------------------------------------------- #


class FakeLibrosa:
    def __init__(self):
        self.f0 = 120.0
        self.measured_on = None
        self.shifts: list[float] = []

    def yin(self, y, fmin, fmax, sr, frame_length):
        self.measured_on = y
        n = max(1, len(y) // 2048)
        return np.full(n, self.f0, dtype=np.float64)

    @property
    def feature(self):
        class _RMS:
            @staticmethod
            def rms(y, frame_length):
                return np.full((1, max(1, len(y) // 2048)), 0.5, dtype=np.float64)

        return types.SimpleNamespace(rms=_RMS.rms)

    @property
    def effects(self):
        fake = self

        class _PitchShift:
            @staticmethod
            def pitch_shift(y, sr, n_steps):
                fake.shifts.append(float(n_steps))
                return y

        return types.SimpleNamespace(pitch_shift=_PitchShift.pitch_shift)


@pytest.fixture
def fake_librosa(monkeypatch):
    fl = FakeLibrosa()
    monkeypatch.setitem(sys.modules, "librosa", fl)
    return fl


def _tone(f0: float, seconds: float, sr: int = 24000) -> np.ndarray:
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    return (0.4 * np.sin(2 * np.pi * f0 * t)).astype(np.float32)


# --------------------------------------------------------------------------- #
# median_f0
# --------------------------------------------------------------------------- #


def test_median_f0_returns_the_measured_median(fake_librosa):
    fake_librosa.f0 = 101.7
    assert vc.median_f0(_tone(101.7, 2.0), 24000) == pytest.approx(101.7)


def test_median_f0_rejects_audio_too_short_to_have_a_pitch(fake_librosa):
    assert vc.median_f0(_tone(100, 0.05), 24000) is None
    assert vc.median_f0(_tone(100, 0.0), 24000) is None
    assert vc.median_f0(None, 24000) is None


def test_median_f0_is_none_when_nothing_is_voiced(fake_librosa):
    fake_librosa.f0 = 0.0  # yin reports 0 for unvoiced frames
    assert vc.median_f0(_tone(100, 2.0), 24000) is None


def test_the_measurement_window_is_bounded(fake_librosa):
    """yin is linear in length and a whole episode is minutes, so the
    correction measures a bounded slice rather than the whole render."""
    fake_librosa.f0 = 110.0
    long_audio = _tone(110.0, vc.F0_ANALYSIS_MAX_S * 4)
    vc.correct_pitch(long_audio, 24000, 110.0)
    assert len(fake_librosa.measured_on) == pytest.approx(
        int(vc.F0_ANALYSIS_MAX_S * 24000), rel=0.01
    )


# --------------------------------------------------------------------------- #
# correct_pitch
# --------------------------------------------------------------------------- #


def test_a_clone_that_is_almost_two_semitones_sharp_is_pulled_onto_the_enrolled_pitch(fake_librosa):
    """The measured failure this exists for: a 112.2 Hz source voice and a
    101.7 Hz person is 1.70 semitones sharp - 12*log2(101.7/112.2)."""
    fake_librosa.f0 = 112.2
    audio, report = vc.correct_pitch(_tone(112.2, 3.0), 24000, 101.7)
    assert report["corrected"] is True
    assert report["applied_semitones"] == pytest.approx(-1.70, abs=0.02)
    assert fake_librosa.shifts == [pytest.approx(-1.70, abs=0.02)]
    assert audio is not None


def test_an_already_correct_clone_is_left_alone(fake_librosa):
    """Under a quarter tone the vocoder is not worth running."""
    fake_librosa.f0 = 120.0
    _, report = vc.correct_pitch(_tone(120, 3.0), 24000, 120.6)
    assert report["corrected"] is False
    assert report["applied_semitones"] == 0.0
    assert "quarter tone" in report["reason"]
    assert fake_librosa.shifts == []


def test_a_sharp_clone_is_pulled_down_not_up(fake_librosa):
    fake_librosa.f0 = 150.0
    _, report = vc.correct_pitch(_tone(150, 3.0), 24000, 100.0)
    assert report["applied_semitones"] < 0


def test_a_shallow_clone_is_pulled_up(fake_librosa):
    fake_librosa.f0 = 90.0
    _, report = vc.correct_pitch(_tone(90, 3.0), 24000, 110.0)
    assert report["applied_semitones"] > 0


def test_a_profile_with_no_recorded_pitch_says_so_instead_of_guessing(fake_librosa):
    """A profile enrolled before pitch was recorded must not be silently left sharp."""
    audio, report = vc.correct_pitch(_tone(112, 3.0), 24000, None)
    assert report["corrected"] is False
    assert report["target_f0_hz"] is None
    assert "re-enrol" in report["reason"]
    assert audio is not None
    assert fake_librosa.shifts == []


def test_the_shift_is_clamped_to_a_major_third(fake_librosa):
    """Past that the vocoder does more damage than the mismatch it fixes."""
    fake_librosa.f0 = 60.0
    _, report = vc.correct_pitch(_tone(60, 3.0), 24000, 300.0)
    assert report["applied_semitones"] == vc.MAX_PITCH_SHIFT_SEMITONES
    assert report["clamped_from_semitones"] > vc.MAX_PITCH_SHIFT_SEMITONES


def test_unvoiced_audio_is_reported_not_guessed_at(fake_librosa):
    fake_librosa.f0 = 0.0
    audio, report = vc.correct_pitch(_tone(100, 3.0), 24000, 110.0)
    assert report["corrected"] is False
    assert report["reason"] == "no voiced audio to measure"
    assert audio is not None


def test_a_failed_vocoder_never_costs_the_user_their_audio(fake_librosa, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("librosa exploded")

    monkeypatch.setattr(vc, "_vocoder_shift", boom)
    audio, report = vc.correct_pitch(_tone(112, 3.0), 24000, 100.0)
    assert audio is not None, "the un-corrected audio must be returned, not lost"
    assert report["corrected"] is False
    assert "librosa exploded" in report["reason"]


def test_the_report_names_the_target_and_what_was_measured(fake_librosa):
    """The numbers exist so a listener who thinks the clone is wrong can see
    which side of the comparison is at fault."""
    fake_librosa.f0 = 112.2
    _, report = vc.correct_pitch(_tone(112, 3.0), 24000, 101.7)
    assert report["target_f0_hz"] == pytest.approx(101.7)
    assert report["measured_f0_hz"] == pytest.approx(112.2)
    assert report["offset_semitones"] == pytest.approx(-1.70, abs=0.02)
