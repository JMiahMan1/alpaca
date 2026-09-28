"""Speaker identification and the per-take spread that makes its scores mean something.

`identify` reuses the OpenVoice reference-encoder embedding the cloner already
writes, so it needs no second model and no extra VRAM on a card that
llama-server and sd-server share. The openvoice package ships no `verifier/`
subpackage, so a dedicated WavLM speaker-verification model is unavailable
here; these tests pin the *seam* - audio in, a ranking out - so a real verifier
can replace the scoring without changing the contract.

The reference encoder is faked rather than loaded: `_converter` and `_spec` are
patched so `_embed_windows` runs for real against a deterministic stand-in, and
the arithmetic under test (cosine, spread, threshold, ranking, calibration) is
the production code path.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("voice_clone_under_test", REPO / "voice_clone.py")
assert _spec and _spec.loader
vc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vc)


# --------------------------------------------------------------------------
# A tensor that answers only what the module under test asks of one
# --------------------------------------------------------------------------


class _T:
    """`unsqueeze`, `cpu`, `mean`, and numpy interop - the whole surface used."""

    def __init__(self, a) -> None:
        self.a = np.asarray(a, dtype="float32")

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        return self.a if dtype is None else self.a.astype(dtype)

    def unsqueeze(self, axis: int) -> _T:
        return _T(np.expand_dims(self.a, axis))

    def transpose(self, *axes) -> _T:
        # NumPy prepends the omitted axes, so a partial `transpose(1, 2)` is
        # really `transpose(1, 2, 0)`. Expanding it here sidesteps a partial-axis
        # failure in numpy 2.5.2 on Python 3.14 (`a.transpose(1, 2)` on a 3-D
        # array raises "axes don't match array" while `swapaxes`/`mT`/`T` and any
        # full permutation work). Production runs on torch inside the container
        # against a pinned numpy, so this is a local-build quirk, not a repo bug -
        # but the stand-in should not be the thing that decides it.
        taken = tuple(int(x) for x in axes)
        rest = tuple(i for i in range(self.a.ndim) if i not in taken)
        return _T(np.transpose(self.a, taken + rest))

    def cpu(self) -> _T:
        return self

    def mean(self, axis: int = 0) -> _T:
        return _T(self.a.mean(axis=axis))

    def __len__(self) -> int:
        return len(self.a)


class _FakeTorch(types.ModuleType):
    """Just the four torch entry points the module under test imports."""

    def __init__(self) -> None:
        super().__init__("torch")
        self._saved: dict[str, np.ndarray] = {}

    @staticmethod
    def no_grad():
        return contextlib.nullcontext()

    @staticmethod
    def stack(items, dim: int = 0) -> _T:
        return _T(np.stack([np.asarray(i) for i in items], axis=dim))

    @staticmethod
    def tensor(data, dtype=None) -> _T:
        return _T(np.asarray(data, dtype=dtype or "float32"))

    def save(self, obj, path, **kwargs) -> None:
        self._saved[str(path)] = np.asarray(obj, dtype="float32").copy()
        np.save(f"{path}.npy", self._saved[str(path)])

    def load(self, path, map_location=None, weights_only=None) -> _T:
        return _T(np.load(f"{path}.npy", allow_pickle=False))

    def float(self):  # pragma: no cover - not used by this module's path
        return self


class _FakeConv:
    """Maps a window's samples to a deterministic unit vector.

    Deterministic *in the content*, which is the property the tests rely on: the
    same recording embeds the same way twice, and a different recording embeds
    differently. The feature is a coarse band-energy profile, because a scalar
    summary (loudness, a weighted sum) is nearly identical for two different
    pitches - a first cut of this fixture used one, and every speaker matched
    every other speaker.

    Real reference-encoder outputs are neither orthogonal nor band-limited, and
    these tests never assume they are.
    """

    device = "cpu"
    hps = None
    _N = 4096
    _BANDS = 12
    # ~1.5 kHz at 22.05 kHz (bin width 22050 / 2 / 2049 = 5.38 Hz). Speech
    # fundamentals and first harmonics live below this; spreading the bands over
    # the whole spectrum put most of them in the noise floor, where every speaker
    # looks identical (cos 0.95 between two different voices).
    _MAX_BIN = 279

    @property
    def model(self) -> _FakeConv:
        return self

    def ref_enc(self, spec) -> _T:
        a = np.asarray(spec, dtype="float64").ravel()
        seg = a[: self._N]
        if seg.size < 256:
            seg = np.pad(seg, (0, 256 - seg.size))
        mag = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
        edges = np.linspace(0, self._MAX_BIN, self._BANDS + 1).astype(int)
        prof = np.array([mag[edges[i] : edges[i + 1]].sum() for i in range(self._BANDS)])
        total = np.linalg.norm(prof)
        return _T((prof / total) if total else np.ones(self._BANDS))


@pytest.fixture
def conv_and_torch(monkeypatch):
    fake = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(vc, "_converter", lambda: _FakeConv())
    # The real `_spec` returns a (channels, frames) spectrogram and `_embed_windows`
    # transposes it to (frames, channels), so the stand-in must be 2-D too.
    monkeypatch.setattr(vc, "_spec", lambda conv, audio_22k: _T(np.asarray(audio_22k).reshape(1, 1, -1)))
    return fake


def _speech(seconds: float, f0: float, *, jitter: float = 0.0, formant: float = 1.0, seed: int = 0) -> np.ndarray:
    """Deterministic 'speech': a voiced harmonic stack at `f0` with a formant balance.

    The pitch is a *parameter*, not something derived from a seed. An earlier
    version computed it as `95 + 11 * (seed % 12)`, which quietly made seed 12
    and seed 13 collide with seed 1 (both wrap to 95 and 106) - so the two
    "speakers" in the fixture shared pitches and every identification test failed
    for a reason that had nothing to do with the code. Passing the pitch in makes
    the fixture's intent legible and its failure modes loud.

    `jitter` (a fraction of `f0`) and `formant` (the balance of the upper
    harmonics) let one speaker's takes differ from each other the way real
    re-records do. Without them every window of a take embeds identically, the
    within-speaker spread is exactly 0.0, and the derived threshold has nothing
    to be derived from.

    Level and pausing clear the real `analyze` gates, so enrolment runs end to end
    instead of being short-circuited: it separates speech from noise with a 10 dB
    gap between the 10th and 95th frame percentile, so a uniformly loud signal
    reads as noise and the clip is rejected as "0.0s of speech".
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * vc.SR)
    t = np.arange(n) / vc.SR
    # A slow intonation glide on top of the base pitch. Without it every window
    # of a take has the same spectrum, so the within-take spread is exactly 0.0
    # and the threshold has nothing to be derived from - a fixture artefact, not a
    # property of real speech, which moves in pitch as it is spoken.
    glide = 1.0 + 0.015 * np.sin(2 * np.pi * 0.35 * t)
    phase = 2 * np.pi * f0 * (1.0 + jitter) * np.cumsum(glide) / vc.SR
    tone = np.zeros(n)
    for k, amp in enumerate((1.0, 0.55 * formant, 0.30 * formant, 0.16 * formant), start=1):
        tone += amp * np.sin(k * phase + 0.3 * k)
    tone /= np.abs(tone).max() or 1.0
    burst, pause = 0.45, 0.25
    gate = np.zeros(n)
    pos = 0.0
    while pos < seconds:
        gate[int(pos * vc.SR) : int((pos + burst) * vc.SR)] = 1.0
        pos += burst + pause
    env = (0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 3.0 * t))) * gate
    out = 0.30 * tone * env + rng.normal(0, 0.002, n)
    return np.clip(out, -0.95, 0.95).astype("float32")


#: Two clearly separated voices. The gap is wider than a natural same-sex
#: difference because the point of the fixture is to make the *ranking* legible,
#: not to model population statistics.
VOICE_A_HZ = 110.0
VOICE_B_HZ = 190.0


def _takes(base_f0: float, seeds) -> list:
    """Three takes of one speaker: a couple of percent of pitch drift and a
    different formant balance each, so the cohort's own spread is a real number."""
    out = []
    for i, seed in enumerate(seeds):
        take = _speech(4.0, base_f0, jitter=0.02 * (i - 1), formant=(0.8, 1.0, 1.25)[i], seed=seed)
        out.append((vc.PROMPTS[i]["id"], _raw(take)))
    return out


def _raw(speech: np.ndarray) -> bytes:
    """A container the module can decode without ffmpeg."""
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(vc.SR)
        w.writeframes((np.clip(speech, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


@pytest.fixture
def voices_dir(tmp_path, monkeypatch):
    d = tmp_path / "voices"
    d.mkdir()
    monkeypatch.setattr(vc, "VOICES_DIR", str(d))
    return d


@pytest.fixture
def no_deps(monkeypatch):
    """Shorten the enrolment gates so a unit test can build a real profile.

    The passage length is not the subject here; the embedding, the spread and
    the ranking are. The prompt ids stay real, because the "read the whole
    passage" check keys on them.
    """
    monkeypatch.setattr(vc, "MIN_TOTAL_SPEECH_S", 1.0)
    monkeypatch.setattr(vc, "SE_WINDOW_S", 1.5)
    monkeypatch.setattr(vc, "PROMPTS", [{**p, "min_s": 1.0} for p in vc.PROMPTS])


def _enrol(name, takes):
    """Build a real profile through `create_profile`, bypassing soundfile.

    `takes` is a list of `(prompt_id, wav_bytes)`, from `_takes`. The ids stay
    real: `create_profile` looks the per-passage minimum up by id and falls back
    to 8 s for an id it does not know, so an invented id would be silently held to
    a stricter bar than a real recording faces.
    """
    written: dict[str, np.ndarray] = {}

    class _SF:
        @staticmethod
        def write(path, data, sr, subtype=None):
            written[str(path)] = np.asarray(data, dtype="float32").copy()
            Path(path).write_bytes(b"RIFFstub")

    sys.modules["soundfile"] = _SF
    return vc.create_profile(name, takes), written


# --------------------------------------------------------------------------
# The arithmetic
# --------------------------------------------------------------------------


def test_cosine_of_a_vector_with_itself_is_one():
    a = np.array([1.0, 2.0, 3.0])
    assert vc._cosine(a, a) == pytest.approx(1.0, abs=1e-6)


def test_cosine_ignores_magnitude():
    a = np.array([1.0, 0.0])
    assert vc._cosine(a, a * 1000.0) == pytest.approx(1.0, abs=1e-6)


def test_cosine_of_orthogonal_vectors_is_zero():
    assert vc._cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0, abs=1e-9)


def test_cosine_of_a_vector_with_its_negative_is_minus_one():
    a = np.array([0.3, 0.7])
    assert vc._cosine(a, -a) == pytest.approx(-1.0, abs=1e-6)


def test_cosine_of_a_zero_vector_is_zero_rather_than_a_division_error():
    assert vc._cosine(np.zeros(4), np.array([1.0, 2.0])) == 0.0


def test_cosine_of_mismatched_dimensions_is_zero_not_a_broadcast_error():
    """A truncated embedding must not be padded into a false match."""
    assert vc._cosine(np.zeros(3), np.zeros(8)) == 0.0
    assert vc._cosine(np.array([]), np.array([1.0])) == 0.0


def test_cosine_accepts_tensors_as_well_as_arrays():
    assert vc._cosine(_T([1.0, 0.0]), _T([1.0, 0.0])) == pytest.approx(1.0, abs=1e-6)


def test_spread_of_fewer_than_two_vectors_is_zero():
    assert vc._pairwise_spread([]) == 0.0
    assert vc._pairwise_spread([np.array([1.0, 0.0])]) == 0.0


def test_spread_of_identical_vectors_is_zero():
    v = [np.array([1.0, 0.0])] * 3
    assert vc._pairwise_spread(v) == pytest.approx(0.0, abs=1e-9)


def test_spread_is_the_mean_one_minus_cosine_over_every_pair():
    a, b, c = np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])
    expected = (1.0 - 0.0 + (1.0 - vc._cosine(a, c)) + (1.0 - vc._cosine(b, c))) / 3
    assert vc._pairwise_spread([a, b, c]) == pytest.approx(expected, abs=1e-9)


def test_spread_grows_as_vectors_move_apart():
    near = vc._pairwise_spread([np.array([1.0, 0.0]), np.array([0.99, 0.14])])
    far = vc._pairwise_spread([np.array([1.0, 0.0]), np.array([0.0, 1.0])])
    assert far > near > 0


# --------------------------------------------------------------------------
# The threshold is derived, not chosen
# --------------------------------------------------------------------------


def test_threshold_scales_with_the_worst_speakers_own_variation():
    tight = vc.identity_threshold([0.02, 0.03])
    loose = vc.identity_threshold([0.30, 0.40])
    assert loose > tight


def test_threshold_uses_the_worst_profile_not_the_average():
    """One inconsistent enrolment sets the floor; averaging would let the
    consistent majority talk it down and then reject their own re-records."""
    assert vc.identity_threshold([0.05, 0.05, 0.40]) == pytest.approx(vc.identity_threshold([0.40]))


def test_threshold_is_clamped_at_the_floor():
    assert vc.identity_threshold([0.0, 0.0]) == pytest.approx(vc.MIN_IDENTITY_THRESHOLD)
    assert vc.identity_threshold([0.001]) >= vc.MIN_IDENTITY_THRESHOLD


def test_threshold_is_clamped_at_the_ceiling():
    assert vc.identity_threshold([1.0]) == pytest.approx(vc.MAX_IDENTITY_THRESHOLD)


def test_threshold_ignores_profiles_that_predate_the_spread_record():
    """Every existing profile on disk lacks `intraspeaker_spread`; they must not
    silently become a spread of zero and make the floor absurdly low."""
    assert vc.identity_threshold([None, 0.2]) == pytest.approx(vc.identity_threshold([0.2]))


def test_threshold_with_no_usable_spreads_falls_back_to_a_default():
    assert 0.0 < vc.identity_threshold([]) < 1.0
    assert vc.identity_threshold([None, None]) == vc.identity_threshold([])


def test_a_single_take_profile_still_gets_a_usable_threshold():
    assert vc.identity_threshold([0.25]) > 0.0


# --------------------------------------------------------------------------
# Enrolment records its own spread
# --------------------------------------------------------------------------


def test_create_profile_records_the_spread_and_the_take_count(voices_dir, conv_and_torch, no_deps):
    meta, _ = _enrol("Ada", _takes(VOICE_A_HZ, (1, 2, 3)))
    assert meta["take_count"] == 3
    assert isinstance(meta["intraspeaker_spread"], float)
    assert 0.0 <= meta["intraspeaker_spread"] < 1.0


def test_create_profile_writes_one_embedding_per_take(voices_dir, conv_and_torch, no_deps):
    meta, _ = _enrol("Ada", _takes(VOICE_A_HZ, (1, 2, 3)))
    d = voices_dir / meta["id"]
    assert (d / "se.pt.npy").is_file(), "the centroid is what the cloner and the identifier both read"
    per_take = np.load(d / "takes.pt.npy", allow_pickle=False)
    assert per_take.shape[0] == 3


def test_the_spread_of_one_take_falls_back_to_its_windows(voices_dir, conv_and_torch, no_deps):
    """A single take has no take-pairs, so the within-take window variation is
    the only evidence available.

    Asserted against the windows the module itself derives, not against
    "greater than zero": with a synthetic voice every window of a take embeds
    identically, so a `> 0.0` assertion would be testing the fixture's realism
    rather than the code's choice of yardstick. The number may legitimately be
    0.0 here - what matters is that it came from the windows.
    """
    meta, _ = _enrol("Solo", _takes(VOICE_A_HZ, (7,)))
    speech = vc.analyze(vc.decode_to_22k(_raw(_speech(4.0, VOICE_A_HZ, seed=7))))["speech"]
    expected = vc._pairwise_spread(vc._embed_windows(vc._windows(speech)))
    assert meta["take_count"] == 1
    assert meta["intraspeaker_spread"] == pytest.approx(expected, abs=1e-4)


def test_the_recorded_spread_matches_a_recomputation(voices_dir, conv_and_torch, no_deps):
    """The number in meta.json must be derived, not typed in by hand."""
    meta, _ = _enrol("Ada", _takes(VOICE_A_HZ, (1, 2, 3)))
    per_take = np.load(voices_dir / meta["id"] / "takes.pt.npy", allow_pickle=False)
    assert vc._pairwise_spread(list(per_take)) == pytest.approx(meta["intraspeaker_spread"], abs=1e-4)


def test_two_identical_takes_record_a_near_zero_spread(voices_dir, conv_and_torch, no_deps):
    """A person who re-recorded the same passage identically is a stable
    enrolment, and the threshold should reflect that."""
    meta, _ = _enrol("Steady", _takes(VOICE_A_HZ, (5,)) * 3)
    assert meta["intraspeaker_spread"] == pytest.approx(0.0, abs=1e-3)


def test_profile_spread_reads_it_back_and_tolerates_an_older_profile(voices_dir, conv_and_torch, no_deps):
    meta, _ = _enrol("Ada", _takes(VOICE_A_HZ, (1, 2)))
    assert vc.profile_spread(meta["id"]) == pytest.approx(meta["intraspeaker_spread"])
    old = voices_dir / "legacy-abc123"
    (old).mkdir()
    (old / "meta.json").write_text(json.dumps({"id": "legacy-abc123", "name": "Legacy"}))
    assert vc.profile_spread("legacy-abc123") is None
    assert vc.profile_spread("does-not-exist") is None


def test_get_profile_still_reports_the_original_fields(voices_dir, conv_and_torch, no_deps):
    meta, _ = _enrol("Ada", _takes(VOICE_A_HZ, (1, 2)))
    got = vc.get_profile(meta["id"])
    for key in ("id", "name", "created", "speech_s", "recordings", "warnings", "engine"):
        assert key in got, key
    assert got["engine"] == "openvoice-v2"


# --------------------------------------------------------------------------
# Identification
# --------------------------------------------------------------------------


@pytest.fixture
def enrolled(voices_dir, conv_and_torch, no_deps):
    """Two distinct speakers, plus the raw audio each was enrolled from."""
    ada, _ = _enrol("Ada", _takes(VOICE_A_HZ, (1, 2, 3)))
    bob, _ = _enrol("Bob", _takes(VOICE_B_HZ, (11, 12, 13)))
    return {
        "ada": ada,
        "bob": bob,
        "ada_clip": _raw(_speech(4.0, VOICE_A_HZ, jitter=0.01, formant=1.1, seed=4)),
        "bob_clip": _raw(_speech(4.0, VOICE_B_HZ, jitter=-0.01, formant=0.9, seed=14)),
        "stranger_clip": _raw(_speech(4.0, 300.0, seed=99)),
    }


def test_identify_finds_the_speaker_it_was_enrolled_from(enrolled):
    result = vc.identify(enrolled["ada_clip"])
    assert result["ok"] is True
    assert result["matched"] == enrolled["ada"]["id"]
    assert result["matched_name"] == "Ada"


def test_identify_distinguishes_between_two_speakers(enrolled):
    assert vc.identify(enrolled["bob_clip"])["matched"] == enrolled["bob"]["id"]


def test_identify_returns_every_profile_ranked_by_score(enrolled):
    ranked = vc.identify(enrolled["ada_clip"])["candidates"]
    assert [c["id"] for c in ranked] == [enrolled["ada"]["id"], enrolled["bob"]["id"]]
    assert ranked[0]["score"] >= ranked[1]["score"]


def test_identify_reports_the_derived_threshold_and_a_rerun_margin(enrolled):
    """The threshold must come from the cohort's own spreads, not a constant.

    The expected value is computed from the two profiles that were really
    enrolled, so this asserts the wiring rather than a number that happens to
    match today.
    """
    expected = vc.identity_threshold([enrolled["ada"]["intraspeaker_spread"], enrolled["bob"]["intraspeaker_spread"]])
    r = vc.identify(enrolled["ada_clip"])
    assert r["threshold"] == pytest.approx(expected, abs=1e-4)
    assert r["margin"] == pytest.approx(r["candidates"][0]["score"] - r["candidates"][1]["score"], abs=1e-4)


def test_identify_still_ranks_candidates_below_the_threshold(enrolled):
    """Refusing to name someone must not discard the evidence - a SharedLLM
    agent needs the ranking to say "closest is X"."""
    r = vc.identify(enrolled["stranger_clip"], threshold=1.01)
    assert r["matched"] is None
    assert len(r["candidates"]) == 2


def test_identify_honours_an_explicit_threshold(enrolled):
    assert vc.identify(enrolled["ada_clip"], threshold=0.0)["matched"] == enrolled["ada"]["id"]
    assert vc.identify(enrolled["ada_clip"], threshold=1.0)["matched"] is None


def test_identify_reports_which_window_matched(enrolled):
    """A clip is several windows; the caller should be able to point at the
    evidence rather than take the verdict on faith."""
    r = vc.identify(enrolled["ada_clip"])
    assert 0 <= r["candidates"][0]["best_window"] < r["windows_scored"]


def test_identify_uses_the_best_window_not_the_average(enrolled):
    """Half a minute of speech contains breaths and doors; one clean window is
    evidence and the average of it with a door is not."""
    r = vc.identify(enrolled["ada_clip"])
    cohort = vc._centroids()
    centroid = next(c for p, c in cohort if p["id"] == enrolled["ada"]["id"])
    windows = vc._embed_windows(vc._windows(vc.analyze(vc.decode_to_22k(enrolled["ada_clip"]))["speech"]))
    best, _idx = vc._best_window(windows, centroid)
    assert r["candidates"][0]["score"] == pytest.approx(best, abs=1e-4)


def test_identify_carries_each_profile_spread_so_a_score_is_readable(enrolled):
    for c in vc.identify(enrolled["ada_clip"])["candidates"]:
        assert c["intraspeaker_spread"] is not None


def test_identify_with_no_enrolled_voices_says_so(enrolled, monkeypatch):
    monkeypatch.setattr(vc, "list_profiles", lambda: [])
    r = vc.identify(enrolled["ada_clip"])
    assert r["ok"] is False
    assert "no enrolled voices" in r["error"]


def test_identify_skips_a_corrupt_profile_instead_of_failing(voices_dir, enrolled):
    """One unreadable enrolment must not make every speaker unidentifiable."""
    broken = voices_dir / "broken-aaaaaa"
    broken.mkdir()
    (broken / "meta.json").write_text(json.dumps({"id": "broken-aaaaaa", "name": "Broken"}))
    r = vc.identify(enrolled["ada_clip"])
    assert r["matched"] == enrolled["ada"]["id"]
    assert [c["id"] for c in r["candidates"]] == [enrolled["ada"]["id"], enrolled["bob"]["id"]]


def test_identify_survives_a_profile_that_is_only_a_directory(enrolled):
    (Path(vc.VOICES_DIR) / "half-made-bbbbbb").mkdir()
    assert vc.identify(enrolled["ada_clip"])["ok"] is True


def test_identify_rejects_a_clip_with_no_usable_speech(enrolled):
    silence = np.zeros(vc.SR, dtype="float32")  # 1 s exactly: below the 1 s window floor
    with pytest.raises(ValueError):
        vc.identify(_raw(silence))


def test_identify_reports_the_engine_so_a_caller_can_temper_its_confidence(enrolled):
    assert vc.identify(enrolled["ada_clip"])["engine"] == "openvoice-v2-reference-encoder"


def test_a_legacy_profile_without_a_spread_still_identifies(voices_dir, enrolled):
    """Enrolments made before this change have no takes.pt and no spread field;
    they must keep working rather than raise."""
    meta = enrolled["ada"]
    d = voices_dir / meta["id"]
    (d / "takes.pt.npy").unlink()
    raw = json.loads((d / "meta.json").read_text())
    raw.pop("intraspeaker_spread", None)
    (d / "meta.json").write_text(json.dumps(raw))
    r = vc.identify(enrolled["ada_clip"])
    assert r["matched"] == meta["id"]
    assert r["candidates"][0]["intraspeaker_spread"] is None


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def test_calibrate_scores_a_clip_against_its_own_voice_and_the_impostors(enrolled):
    r = vc.calibrate([(enrolled["ada"]["id"], enrolled["ada_clip"])])
    row = r["rows"][0]
    assert row["label"] == enrolled["ada"]["id"]
    assert row["own_score"] is not None
    assert row["impostor_id"] == enrolled["bob"]["id"]
    assert row["margin"] == pytest.approx(row["own_score"] - row["impostor_score"], abs=1e-4)


def test_calibrate_owns_score_beat_the_impostor_score(enrolled):
    row = vc.calibrate([(enrolled["ada"]["id"], enrolled["ada_clip"])])["rows"][0]
    assert row["own_score"] > row["impostor_score"]


def test_calibrate_suggests_a_boundary_between_the_two_populations(enrolled):
    r = vc.calibrate([(enrolled["ada"]["id"], enrolled["ada_clip"]), (enrolled["bob"]["id"], enrolled["bob_clip"])])
    assert r["lowest_own_score"] == pytest.approx(min(row["own_score"] for row in r["rows"]), abs=1e-4)
    assert r["highest_impostor_score"] == pytest.approx(max(row["impostor_score"] for row in r["rows"]), abs=1e-4)
    assert r["suggested_threshold"] == pytest.approx(
        (r["lowest_own_score"] + r["highest_impostor_score"]) / 2, abs=1e-4
    )


def test_calibrate_names_the_weakest_clip_so_the_operator_knows_what_to_re_record(enrolled):
    r = vc.calibrate([(enrolled["ada"]["id"], enrolled["ada_clip"]), (enrolled["bob"]["id"], enrolled["bob_clip"])])
    assert r["weakest_clip"] in (enrolled["ada"]["id"], enrolled["bob"]["id"])


def test_calibrate_reports_the_cohort_threshold_beside_the_suggested_one(enrolled):
    r = vc.calibrate([(enrolled["ada"]["id"], enrolled["ada_clip"])])
    assert 0.0 < r["cohort_threshold"] < 1.0


def test_calibrate_with_one_enrolled_voice_has_no_impostor(enrolled, monkeypatch):
    only_ada = vc.get_profile(enrolled["ada"]["id"])
    monkeypatch.setattr(vc, "list_profiles", lambda: [only_ada])
    r = vc.calibrate([(enrolled["ada"]["id"], enrolled["ada_clip"])])
    assert r["rows"][0]["own_score"] is not None
    assert r["rows"][0]["impostor_score"] is None
    assert r["rows"][0]["margin"] is None
    assert r["suggested_threshold"] is None


def test_calibrate_with_a_label_that_names_nothing_still_reports_the_impostor(enrolled):
    """A clip labelled with an unknown id has no 'own' score, but the best
    wrong answer is exactly what the operator wants to see."""
    r = vc.calibrate([("not-a-real-voice", enrolled["ada_clip"])])
    assert r["rows"][0]["own_score"] is None
    assert r["rows"][0]["impostor_id"] == enrolled["ada"]["id"]


def test_calibrate_with_no_enrolled_voices_says_so(enrolled, monkeypatch):
    monkeypatch.setattr(vc, "list_profiles", lambda: [])
    r = vc.calibrate([(enrolled["ada"]["id"], enrolled["ada_clip"])])
    assert r["ok"] is False
    assert "no enrolled voices" in r["error"]


def test_calibrate_with_no_clips_reports_nothing_rather_than_inventing_a_threshold(enrolled):
    r = vc.calibrate([])
    assert r["rows"] == []
    assert r["clips"] == 0
    assert r["suggested_threshold"] is None


def test_calibrate_and_identify_agree_on_a_verdict_at_the_suggested_threshold(enrolled):
    """The two entry points must not disagree: calibrate proposes a boundary,
    identify is expected to accept exactly the clips that sit above it."""
    r = vc.calibrate([(enrolled["ada"]["id"], enrolled["ada_clip"])])
    thr = r["suggested_threshold"]
    assert vc.identify(enrolled["ada_clip"], threshold=thr)["matched"] == enrolled["ada"]["id"]
    assert vc.identify(enrolled["bob_clip"], threshold=thr)["matched"] != enrolled["ada"]["id"]
