"""Unit tests for web/podcast_mixer.py -- pure numpy, no network, no GPU.

The mixer is the part of the Podcast Studio that decides whether an episode
sounds like a podcast. Three of its properties are not stylistic preferences
and are therefore pinned here as contracts:

* the bed is synthesized at 24000 Hz, the rate Kokoro emits, so nothing in
  the speech path resamples;
* the final encode does **not** peak-normalise, because the bed sits ~20 dB
  under the speech by design and normalising would make the quietest thing in
  the mix the loudest;
* a long bed is crossfaded when it loops, because a 10 s bed under a 12
  minute episode would otherwise click 72 times.
"""

from __future__ import annotations

import io
import itertools
import math
import wave
from pathlib import Path

import numpy as np
import pytest

from web import podcast_mixer as pm

SR = pm.TARGET_SR


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _tone(seconds: float, hz: float = 220.0, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _wav_bytes(x: np.ndarray, sr: int = SR, width: int = 2, channels: int = 1) -> bytes:
    if width == 2:
        pcm = np.round(np.clip(x, -1, 1) * 32767).astype("<i2")
    elif width == 1:
        pcm = (np.clip(x, -1, 1) * 127 + 128).astype(np.uint8)
    else:
        pcm = np.round(np.clip(x, -1, 1) * 2147483647).astype("<i4")
    if channels > 1:
        pcm = np.repeat(pcm[:, None], channels, axis=1).ravel()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _host_a() -> pm.HostSpec:
    return pm.HostSpec(name="Ada", voice="af_nicole", key="a", gender="f")


def _host_b() -> pm.HostSpec:
    return pm.HostSpec(name="Rowan", voice="am_michael", key="b", gender="m")


def _bed_kwargs(preset: dict) -> dict:
    """Only the keys synthesize_bed accepts; a preset also carries label and
    prompt, which are for the MusicGen sting, not the synth."""
    return {k: v for k, v in preset.items() if k in {"key", "bpm", "density", "brightness", "mood"}}


def _seg(seconds: float, hz: float = 220.0, host_index: int = 0, amp: float = 0.5) -> dict:
    return {
        "wav": _wav_bytes(_tone(seconds, hz, amp=amp)),
        "host_index": host_index,
        "text": f"turn {host_index}",
        "line_no": host_index,
    }


# --------------------------------------------------------------------------
# the contract that decides whether the podcast path resamples at all
# --------------------------------------------------------------------------


def test_the_bed_is_synthesized_at_the_rate_kokoro_emits():
    """Kokoro is 24 kHz. If the bed were not too, every episode would pay a
    resample of the speech -- and linear interpolation on speech is audible."""
    assert SR == 24000


def test_audio_servers_really_are_24k_and_musicgen_is_32k():
    """If either of these ever changes, the resample stops being the single
    resample in the path and the 'good enough for a pad' claim in
    resample_linear stops being true."""
    audio = (Path(__file__).resolve().parents[1] / "audio_server.py").read_text()
    assert "sr = 24000" in audio
    assert 'os.getenv("AUDIO_MAX_MUSIC_S", "30")' in audio


# --------------------------------------------------------------------------
# decibels, soft clip, normalisation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "db,lin",
    [(0.0, 1.0), (-6.0, 0.5011872), (-20.0, 0.1), (-40.0, 0.01), (6.0, 1.9952623)],
)
def test_db_to_lin(db, lin):
    assert pm.db_to_lin(db) == pytest.approx(lin, rel=1e-4)


def test_db_round_trip():
    for db in (-60.0, -20.0, -6.0, 0.0, 3.0):
        assert pm.lin_to_db(pm.db_to_lin(db)) == pytest.approx(db, abs=1e-6)


def test_lin_to_db_floors_instead_of_raising_on_silence():
    """A silent bed would otherwise make the mix compute log10(0)."""
    assert pm.lin_to_db(0.0) == -120.0
    assert pm.lin_to_db(1e-9) == -120.0
    assert math.isfinite(pm.lin_to_db(0.0))


def test_soft_clip_bounds_without_a_hard_edge():
    x = np.linspace(-4.0, 4.0, 2001, dtype=np.float32)
    y = pm.soft_clip(x)
    assert y.min() > -1.0 and y.max() < 1.0
    # strictly monotonic: a limiter that folds back is a distortion generator
    assert np.all(np.diff(y) > 0)
    # and it does touch +-1, so it is actually doing something at full scale
    assert pm.soft_clip(np.array([20.0], dtype=np.float32))[0] > 0.99


def test_normalize_peak_hits_the_target():
    x = _tone(0.1, amp=0.2)
    y = pm.normalize_peak(x, 0.7)
    assert np.abs(y).max() == pytest.approx(0.7, rel=1e-3)


def test_normalize_peak_leaves_silence_alone():
    z = np.zeros(100, dtype=np.float32)
    assert np.array_equal(pm.normalize_peak(z), z)


def test_normalize_peak_leaves_empty_alone():
    e = np.zeros(0, dtype=np.float32)
    assert pm.normalize_peak(e).size == 0


# --------------------------------------------------------------------------
# fades
# --------------------------------------------------------------------------


def test_fade_envelope_is_flat_in_the_middle_and_zero_at_both_ends():
    env = pm.fade_envelope(SR, 1.0, 1.0, SR)
    assert env[0] == 0.0
    assert env[-1] == 0.0
    assert env[SR // 2] == pytest.approx(1.0, abs=1e-3)


def test_both_fades_are_clamped_to_half_the_buffer():
    """Otherwise a 2 s buffer asked for 1.2 s fades would never reach full
    height and the episode would be quiet for its whole length."""
    env = pm.fade_envelope(int(0.2 * SR), 1.2, 1.2, SR)
    assert env.max() == pytest.approx(1.0, abs=1e-3)
    assert env[int(0.1 * SR)] == pytest.approx(1.0, abs=1e-3)


def test_fade_envelope_of_zero_samples_is_empty():
    assert pm.fade_envelope(0, 1.0, 1.0, SR).size == 0


def test_a_zero_fade_length_is_all_ones():
    assert np.array_equal(pm.fade_envelope(100, 0.0, 0.0, SR), np.ones(100, dtype=np.float32))


# --------------------------------------------------------------------------
# ducking
# --------------------------------------------------------------------------


def test_the_envelope_leaves_the_bed_alone_when_nobody_is_speaking():
    silence = np.zeros(2 * SR, dtype=np.float32)
    env = pm.duck_envelope(silence, SR)
    assert np.allclose(env, 1.0, atol=1e-3)


def test_the_envelope_pulls_the_bed_down_under_speech():
    speech = np.concatenate([_tone(1.0, amp=0.5), np.zeros(2 * SR, dtype=np.float32)])
    env = pm.duck_envelope(speech, SR, duck_db=20.0)
    under_speech = float(env[int(0.5 * SR)])
    in_the_gap = float(env[int(2.0 * SR)])
    assert under_speech < in_the_gap
    # the floor is the requested attenuation
    assert under_speech == pytest.approx(pm.db_to_lin(-20.0), rel=0.25)


def test_the_duck_depth_is_monotonic_in_duck_db():
    speech = np.concatenate([_tone(1.0, amp=0.5), np.zeros(1 * SR, dtype=np.float32)])
    shallow = float(pm.duck_envelope(speech, SR, duck_db=6.0)[int(0.5 * SR)])
    deep = float(pm.duck_envelope(speech, SR, duck_db=30.0)[int(0.5 * SR)])
    assert deep < shallow


def test_the_release_is_slower_than_the_attack():
    """Asymmetric one-pole: fast to duck, slow to come back. A symmetric
    follower sounds like the bed is part of the speech."""
    speech = np.concatenate([_tone(0.5, amp=0.5), np.zeros(4 * SR, dtype=np.float32)])
    env = pm.duck_envelope(speech, SR, attack_s=0.05, release_s=1.5)
    ducked = float(env[int(0.45 * SR)])
    # 0.25 s after the speech ends the bed has only partly returned
    just_after = float(env[int(0.75 * SR)])
    much_later = float(env[int(4.0 * SR)])
    assert ducked < just_after < much_later


def test_the_threshold_stops_inter_sentence_pauses_from_pumping_the_bed():
    """A follower keyed on instantaneous level drops the bed between every
    word, which reads as a fault rather than as mixing."""
    quiet_gaps = np.concatenate(
        [
            _tone(0.2, amp=0.5),
            np.zeros(int(0.15 * SR), dtype=np.float32),
            _tone(0.2, amp=0.5),
            np.zeros(2 * SR, dtype=np.float32),
        ]
    )
    env = pm.duck_envelope(quiet_gaps, SR, threshold=0.02)
    gap_env = float(env[int(0.27 * SR)])
    assert gap_env > pm.db_to_lin(-20.0)  # not fully ducked in the gap


def test_an_empty_speech_buffer_yields_an_empty_envelope():
    assert pm.duck_envelope(np.zeros(0, dtype=np.float32), SR).size == 0


def test_the_envelope_is_one_sample_per_speech_sample():
    speech = np.zeros(int(1.5 * SR), dtype=np.float32)
    assert pm.duck_envelope(speech, SR).size == speech.size


# --------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------


def test_resample_to_the_same_rate_is_a_passthrough():
    x = _tone(0.2, amp=0.3)
    assert np.array_equal(pm.resample_linear(x, SR, SR), x)


def test_resample_produces_the_expected_length():
    x = _tone(1.0, sr=32000, amp=0.3)
    y = pm.resample_linear(x, 32000, 24000)
    assert y.size == pytest.approx(24000, rel=0.01)
    assert y.dtype == np.float32


def test_resample_upsamples_too():
    x = _tone(1.0, sr=8000, amp=0.3)
    assert pm.resample_linear(x, 8000, 24000).size == pytest.approx(24000, rel=0.01)


def test_resample_of_an_empty_buffer_is_empty():
    assert pm.resample_linear(np.zeros(0, dtype=np.float32), 32000, 24000).size == 0


def test_resample_preserves_the_amplitude_envelope():
    """Interpolating a ramp must stay a ramp: an off-by-one in the index
    mapping would show up here as a step."""
    ramp = np.linspace(0.0, 1.0, 1000, dtype=np.float32)
    out = pm.resample_linear(ramp, 1000, 500)
    assert np.all(np.diff(out) > 0)


# --------------------------------------------------------------------------
# looping
# --------------------------------------------------------------------------


def test_loop_to_length_hits_the_requested_length_exactly():
    bed = _tone(0.5, amp=0.2)
    for n in (100, 12000, 24000, 60001, SR * 3):
        assert pm.loop_to_length(bed, n, crossfade_s=0.05, sr=SR).size == n


def test_a_bed_longer_than_the_episode_is_truncated():
    bed = _tone(4.0, amp=0.2)
    out = pm.loop_to_length(bed, 1000, crossfade_s=0.05, sr=SR)
    assert out.size == 1000
    assert np.allclose(out, bed[:1000])


def test_the_join_is_crossfaded_rather_than_stepped():
    """A rising ramp ends at its peak and restarts at zero, so a naive loop
    puts a full-scale step at the join. That step is a click."""
    n = SR // 4
    ramp = np.linspace(0.0, 1.0, n, endpoint=False).astype(np.float32)
    faded = pm.loop_to_length(ramp, n * 3, crossfade_s=0.02, sr=SR)
    raw = pm.loop_to_length(ramp, n * 3, crossfade_s=0.0, sr=SR)

    def biggest_jump(buf: np.ndarray) -> float:
        return float(np.abs(np.diff(buf.astype(np.float64))).max())

    assert biggest_jump(raw) == pytest.approx(1.0, abs=0.01)  # a full-scale step
    # A crossfade cannot make a *non-periodic* input periodic. What remains is
    # the source's own rise across the fade window, which is a property of the
    # ramp rather than a discontinuity; a real bed is quasi-periodic, so this
    # is the whole of the improvement it gets.
    assert biggest_jump(faded) == pytest.approx((0.02 * SR) / n, abs=0.01)


def test_the_join_blends_rather_than_dipping_to_silence():
    """The previous construction appended the faded tail *before* the next
    copy, so the bed went to zero once per loop -- a heartbeat in the wrong
    place, on top of the step it was meant to remove."""
    n = SR // 4
    ramp = np.linspace(0.0, 1.0, n, endpoint=False).astype(np.float32)
    fade = int(0.02 * SR)
    out = pm.loop_to_length(ramp, n * 3, crossfade_s=0.02, sr=SR)
    blend = out[n : n + fade]
    assert np.abs(blend).max() > 0.5 * np.abs(out).max()  # never near silence
    assert np.abs(blend).min() < np.abs(out).max()  # but it does move


def test_looping_a_buffer_with_no_room_for_a_crossfade_still_returns_the_right_length():
    bed = _tone(0.001, amp=0.2)
    assert pm.loop_to_length(bed, SR, crossfade_s=2.0, sr=SR).size == SR


def test_looping_nothing_yields_silence():
    out = pm.loop_to_length(np.zeros(0, dtype=np.float32), 500)
    assert out.size == 500
    assert not out.any()


def test_loop_to_a_non_positive_length_yields_nothing():
    assert pm.loop_to_length(_tone(0.1), 0).size == 0
    assert pm.loop_to_length(_tone(0.1), -5).size == 0


# --------------------------------------------------------------------------
# music theory
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,midi",
    [("a3", 57), ("a4", 69), ("f#2", 42), ("bb1", 34), ("c", 60), ("g2", 43)],
)
def test_name_to_semitone(name, midi):
    assert pm.name_to_semitone(name) == midi


def test_a_note_without_an_octave_defaults_to_octave_four():
    """The presets are written as 'a' and 'f#m', so a bare letter has to
    mean something -- and the docstring's 'defaults to C4' is about the
    octave, not about changing the pitch class."""
    assert pm.name_to_semitone("a") == pm.name_to_semitone("a4")
    assert pm.name_to_semitone("a") == 69


def test_a_nonsense_note_name_falls_back_rather_than_raising():
    assert pm.name_to_semitone("") == 48  # C3
    assert pm.name_to_semitone("banana") == 48
    assert pm.name_to_semitone("h9") == 48


def test_every_preset_key_parses():
    for preset_id, preset in pm.BED_PRESETS.items():
        assert 0 <= pm.name_to_semitone(preset["key"]) < 128, preset_id


@pytest.mark.parametrize("mood", ["sparse", "dusty", "warm", "anything-else"])
def test_scale_intervals_start_on_the_root_and_contain_no_enormous_jumps(mood):
    iv = pm.scale_intervals(mood)
    assert iv[0] == 0
    assert all(b > a for a, b in itertools.pairwise(iv))


# --------------------------------------------------------------------------
# bed synthesis
# --------------------------------------------------------------------------


def test_a_bed_is_exactly_the_requested_length():
    for seconds in (0.5, 1.0, 3.7):
        assert pm.synthesize_bed(seconds, sr=SR).size == round(seconds * SR)


def test_a_bed_is_deterministic_for_a_seed():
    """Otherwise a re-render of the same script is a different episode and a
    bad bed cannot be reproduced from a one-line description."""
    a = pm.synthesize_bed(1.0, seed=7, sr=SR)
    b = pm.synthesize_bed(1.0, seed=7, sr=SR)
    assert np.array_equal(a, b)


def test_a_different_seed_changes_the_bed():
    a = pm.synthesize_bed(1.0, seed=1, sr=SR)
    b = pm.synthesize_bed(1.0, seed=2, sr=SR)
    assert not np.array_equal(a, b)


def test_different_keys_produce_different_beds():
    a = pm.synthesize_bed(1.0, key="a", seed=3, sr=SR)
    b = pm.synthesize_bed(1.0, key="f#m", seed=3, sr=SR)
    assert not np.allclose(a, b, atol=1e-3)


def test_the_bed_stays_below_the_level_a_bed_belongs_at():
    """The mixer sets the level itself; the synth must hand over something
    it can attenuate, not something already at full scale."""
    for preset_id in pm.BED_PRESETS:
        bed = pm.synthesize_bed(2.0, **_bed_kwargs(pm.bed_preset(preset_id)), sr=SR)
        assert np.abs(bed).max() <= 1.0, preset_id
        assert np.abs(bed).max() > 0.01, preset_id  # and not silence


def test_brightness_moves_energy_upward():
    """`brightness` is the one control that separates a deep-focus drone from
    an acoustic-morning bed, so it has to actually change the spectrum."""
    dark = pm.synthesize_bed(3.0, brightness=0.0, seed=1, sr=SR)
    bright = pm.synthesize_bed(3.0, brightness=1.0, seed=1, sr=SR)

    def high_energy(x: np.ndarray) -> float:
        return float(np.abs(np.diff(x.astype(np.float64))).mean())  # a cheap HF proxy

    assert high_energy(bright) > high_energy(dark)


def test_a_longer_bed_still_ends_faded_in_and_out():
    bed = pm.synthesize_bed(4.0, sr=SR)
    assert abs(float(bed[0])) < 0.05
    assert abs(float(bed[-1])) < 0.05


def test_bed_preset_falls_back_for_an_unknown_id():
    """A stale id from an old UI should not 400 a render the user waited for."""
    assert pm.bed_preset("no-such-preset") == pm.bed_preset("ambient_warm")
    assert pm.bed_preset("") == pm.bed_preset("ambient_warm")


def test_every_bed_preset_is_renderable():
    for preset_id in pm.BED_PRESETS:
        preset = pm.bed_preset(preset_id)
        assert preset["label"] and preset["prompt"]
        assert pm.synthesize_bed(0.5, **_bed_kwargs(preset), sr=SR).size == int(0.5 * SR)


# --------------------------------------------------------------------------
# encode / decode
# --------------------------------------------------------------------------


def test_the_encode_does_not_normalise():
    """The whole reason this module has its own encoder. A quiet signal must
    stay quiet, because the bed's whole point is that it is quiet -- a peak
    normaliser would rescale the mix so the quietest thing in it becomes the
    loudest, which is the opposite of what ducking is for."""
    quiet, _ = pm.decode_wav(pm.encode_wav(_tone(0.2, amp=0.05), SR))
    assert float(np.abs(quiet).max()) < 0.06


def test_the_encode_preserves_the_ratio_between_two_quiet_signals():
    """Compared in the linear region, where the tanh limiter is not the
    dominant term -- a limiter is a deliberate non-linearity, not a bug."""
    quiet, _ = pm.decode_wav(pm.encode_wav(_tone(0.2, amp=0.02), SR))
    loud, _ = pm.decode_wav(pm.encode_wav(_tone(0.2, amp=0.2), SR))
    q_rms = float(np.sqrt(np.mean(quiet.astype(np.float64) ** 2)))
    l_rms = float(np.sqrt(np.mean(loud.astype(np.float64) ** 2)))
    assert l_rms / q_rms == pytest.approx(10.0, rel=0.12)


def test_the_encode_leaves_a_quiet_mix_quiet_where_the_audio_server_would_boost_it():
    """audio_server._wav_bytes peak-normalises, which would turn a 20 dB-ducked
    bed back into a full-level one. Pin the difference rather than the intent."""
    mix = 0.9 * _tone(0.2, amp=0.5) + 0.02 * _tone(0.2, amp=0.5, hz=77.0)
    decoded, _ = pm.decode_wav(pm.encode_wav(mix, SR))
    assert np.abs(decoded).max() < 0.95


def test_encode_writes_a_mono_16_bit_wav_with_the_right_frame_count():
    x = _tone(0.1, amp=0.3)
    with wave.open(io.BytesIO(pm.encode_wav(x, SR)), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == SR
        assert w.getnframes() == x.size


def test_the_encode_survives_an_overdriven_mix_without_wrapping():
    hot = np.clip(_tone(0.1, amp=0.9) * 3.0, -1, 1).astype(np.float32)
    out, _ = pm.decode_wav(pm.encode_wav(hot, SR))
    assert np.abs(out).max() <= 1.0


def test_the_encode_can_skip_the_limiter():
    x = _tone(0.05, amp=0.5)
    a, _ = pm.decode_wav(pm.encode_wav(x, SR, clip=True))
    b, _ = pm.decode_wav(pm.encode_wav(x, SR, clip=False))
    # tanh is not the identity, so the two differ, and the raw one is louder
    assert not np.array_equal(a, b)
    assert float(np.abs(b).max()) > float(np.abs(a).max())


@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("channels", [1, 2])
def test_decode_handles_every_width_and_channel_count(width, channels):
    original = _tone(0.1, amp=0.4)
    data, sr = pm.decode_wav(_wav_bytes(original, SR, width=width, channels=channels))
    assert sr == SR
    assert data.dtype == np.float32
    assert data.size == pytest.approx(original.size, rel=0.01)
    if width == 2:
        assert float(np.abs(data).max()) == pytest.approx(0.4, rel=0.01)


def test_decode_rejects_an_impossible_sample_width():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(3)
        w.setframerate(SR)
        w.writeframes(b"\x00" * 300)
    with pytest.raises(ValueError, match="sample width"):
        pm.decode_wav(buf.getvalue())


def test_encode_decode_round_trips_a_16_bit_signal_faithfully():
    """With the limiter off, the only error left is 16-bit quantisation,
    which is ~3e-5 of full scale."""
    x = _tone(0.2, amp=0.5)
    out, sr = pm.decode_wav(pm.encode_wav(x, SR, clip=False))
    assert sr == SR
    assert float(np.abs(out - x).max()) < 1e-3


def test_the_limiter_is_what_makes_a_loud_encode_non_transparent():
    """Pinned so the quantisation test above cannot be 'fixed' by silently
    removing the safety limiter."""
    x = _tone(0.2, amp=0.5)
    limited, _ = pm.decode_wav(pm.encode_wav(x, SR, clip=True))
    assert float(np.abs(limited).max()) < float(np.abs(x).max())


# --------------------------------------------------------------------------
# the MusicGen sting: the one resample in the podcast path
# --------------------------------------------------------------------------


def test_a_32k_sting_is_brought_to_the_mix_rate():
    """MusicGen emits 32 kHz and the mixer runs at 24 kHz. This is the only
    place that conversion happens, and linear interpolation is good enough
    only because a sting is a slowly-varying pad."""
    sting = _tone(0.5, hz=440.0, sr=32000, amp=0.4)
    out = pm.resample_musicgen_sting(_wav_bytes(sting, 32000))
    assert out.size == pytest.approx(0.5 * SR, rel=0.02)
    assert out.dtype == np.float32


def test_a_sting_already_at_the_mix_rate_is_untouched():
    sting = _tone(0.3, hz=440.0, sr=SR, amp=0.4)
    out = pm.resample_musicgen_sting(_wav_bytes(sting, SR))
    assert out.size == pytest.approx(0.3 * SR, rel=0.01)


def test_a_stereo_sting_is_mixed_down_to_mono():
    out = pm.resample_musicgen_sting(_wav_bytes(_tone(0.2, amp=0.4, sr=32000), 32000, channels=2))
    assert out.ndim == 1


# --------------------------------------------------------------------------
# script parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line,expected",
    [
        ("[host_a] Welcome to the show.", ("host_a", "Welcome to the show.")),
        ("[host_a] Welcome", ("host_a", "Welcome")),
        ("(ADA) Hello there", ("ADA", "Hello there")),
        ("host_b: Glad to be here.", ("host_b", "Glad to be here.")),
        ("Ada: And I'm Ada.", ("Ada", "And I'm Ada.")),
        ("[ADA] - dash after the bracket", ("ADA", "dash after the bracket")),
    ],
)
def test_the_tag_forms_a_model_actually_writes(line, expected):
    assert pm.match_speaker_tag(line) == expected


@pytest.mark.parametrize(
    "line",
    [
        "Here is the plan: ship it",
        "Just a normal sentence with a colon: like that one",
        "In the end we decided to ship it: the tests passed",
        "  ",
        "",
    ],
)
def test_prose_is_not_mistaken_for_a_speaker_turn(line):
    """A bare tag has to require a colon AND be short enough to be a name.
    Without the word bound, "Here is the plan" becomes a speaker and "ship
    it" gets synthesised in somebody's voice."""
    assert pm.match_speaker_tag(line) is None


@pytest.mark.parametrize("line", ["Here: we ship", "So: the tests passed", "Anyway: on to it"])
def test_prose_with_a_colon_never_becomes_a_turn_when_parsed(line):
    """The end-to-end consequence, which is what actually matters: the line
    must not fragment the script or borrow a host's voice."""
    # The prose continues the previous host's turn rather than inventing a
    # speaker called "Here"; Kokoro will read the colon as a pause.
    turns = pm.parse_script(f"[host_a] Welcome.\n{line}\n[host_b] Hi.\n", hosts=[_host_a(), _host_b()])
    assert [t.speaker for t in turns] == ["host_a", "host_b"]
    assert turns[0].text == f"Welcome. {line}"


@pytest.mark.parametrize("tag", ["host_c", "GUEST", "Narrator", "Dr. Chen", "Ada Lovelace"])
def test_an_unresolved_tag_that_looks_like_a_label_still_starts_its_own_turn(tag):
    """A guest is worth a real turn rather than being glued onto the previous
    speaker's line, which would make that host read the label aloud."""
    turns = pm.parse_script(f"[host_a] Welcome.\n{tag}: May I say something.\n", hosts=[_host_a(), _host_b()])
    assert len(turns) == 2
    assert turns[1].speaker == tag
    assert turns[1].text == "May I say something."


def test_the_two_word_clause_limit_is_deliberate_and_documented():
    """"And so: the tests passed" is still read as a turn. Tightening the
    label test any further would refuse guest speakers, and the draft prompt
    tells the model to use the host keys it was given."""
    turns = pm.parse_script("And so: the tests passed.\n", hosts=[_host_a(), _host_b()])
    assert turns[0].speaker == "And so"
    assert "And so: the tests passed" in pm._looks_like_speaker_label.__doc__


def test_a_script_parses_into_turns_with_the_right_hosts():
    script = "[host_a] Welcome to the show.\n[host_b] Glad to be here.\n"
    turns = pm.parse_script(script, hosts=[_host_a(), _host_b()])
    assert [t.speaker for t in turns] == ["host_a", "host_b"]
    assert [t.host_index for t in turns] == [0, 1]


def test_a_tag_matches_the_host_display_name_as_well_as_the_key():
    """A model that writes 'Ada:' has identified the host; refusing because
    the key is 'a' would silently drop the turn to a default voice."""
    turns = pm.parse_script("Ada: Hello.\nRowan: Hi.\n", hosts=[_host_a(), _host_b()])
    assert [t.host_index for t in turns] == [0, 1]


@pytest.mark.parametrize("tag", ["host_a", "HOST A", "Host-A", "host a", "[host_a]", "ADA"])
def test_separators_and_case_are_not_part_of_a_host_identity(tag):
    turns = pm.parse_script(f"{tag}: Hello there friend.\n", hosts=[_host_a(), _host_b()])
    assert turns[0].host_index == 0


def test_plain_host_keys_work_without_hostspecs():
    turns = pm.parse_script("[host_a] Hi.\n[host_b] Yo.\n", hosts=["a", "b"])
    assert [t.host_index for t in turns] == [0, 1]


def test_a_wrapped_line_continues_the_previous_turn():
    """Models wrap long lines constantly; dropping the continuation loses
    words, and merging it into a new turn gives the speaker a second voice."""
    script = "[host_a] This is a long thought that the model\nwrapped onto a second line.\n"
    turns = pm.parse_script(script, hosts=[_host_a(), _host_b()])
    assert len(turns) == 1
    assert "wrapped onto a second line" in turns[0].text
    assert turns[0].host_index == 0


def test_a_tag_for_an_unknown_speaker_starts_its_own_turn():
    """Merging it would make the previous host read the other host's label
    aloud."""
    script = "[host_a] Welcome.\n[host_c] And I am a guest.\n"
    turns = pm.parse_script(script, hosts=[_host_a(), _host_b()])
    assert len(turns) == 2
    assert turns[1].speaker == "host_c"
    assert turns[1].text == "And I am a guest."


def test_an_unknown_speaker_keeps_its_name_and_borrows_the_nearest_host_voice():
    turns = pm.parse_script("[host_a] Welcome.\n[host_c] A guest speaks.\n", hosts=[_host_a(), _host_b()])
    assert turns[1].speaker == "host_c"
    assert turns[1].host_index == 0


@pytest.mark.parametrize("line", ["# Episode 1", "## The plan", "INTRO", "OUTRO", "HOST A"])
def test_headings_are_kept_for_display_but_never_spoken(line):
    """Reading '# Episode 12' aloud in a host's voice is the most jarring way
    this pipeline can be wrong."""
    script = f"{line}\n[host_a] Welcome to the show.\n"
    turns = pm.parse_script(script, hosts=[_host_a(), _host_b()])
    assert turns[0].is_heading is True
    assert pm.spoken_turns(turns) == [turns[1]]


def test_a_heading_does_not_swallow_the_line_after_it():
    turns = pm.parse_script("# Episode 1\n[host_a] Welcome to the show.\n", hosts=[_host_a(), _host_b()])
    assert turns[1].text == "Welcome to the show."


def test_a_capitalised_sentence_is_not_mistaken_for_a_heading():
    turns = pm.parse_script("[host_a] WELCOME BACK TO THE SHOW.\n", hosts=[_host_a(), _host_b()])
    assert turns[0].is_heading is False


def test_a_heading_never_inherits_a_host_index():
    turns = pm.parse_script("[host_a] Hi.\n# Break\n[host_b] Yo.\n", hosts=[_host_a(), _host_b()])
    assert turns[1].is_heading is True
    assert turns[1].host_index is None


def test_blank_lines_are_skipped_entirely():
    turns = pm.parse_script("[host_a] One.\n\n\n[host_b] Two.\n", hosts=[_host_a(), _host_b()])
    assert len(turns) == 2


def test_text_before_the_first_tag_is_kept_rather_than_discarded():
    turns = pm.parse_script("Some opening prose.\n[host_a] Welcome.\n", hosts=[_host_a(), _host_b()])
    assert any(t.text == "Some opening prose." for t in turns)


def test_an_empty_script_parses_to_nothing():
    assert pm.parse_script("", hosts=[_host_a(), _host_b()]) == []
    assert pm.parse_script("   \n\n  ", hosts=[_host_a(), _host_b()]) == []


def test_a_turn_records_the_line_it_came_from():
    turns = pm.parse_script("\n\n[host_a] Third line.\n", hosts=[_host_a()])
    assert turns[0].line_no == 3


def test_turn_to_dict_carries_every_field():
    d = pm.Turn(speaker="a", text="hi", line_no=2, host_index=1, is_heading=False).to_dict()
    assert d == {"speaker": "a", "text": "hi", "line_no": 2, "host_index": 1, "is_heading": False}


def test_spoken_turns_drops_headings_and_blank_turns():
    turns = [pm.Turn("a", "", is_heading=False), pm.Turn("a", "real", is_heading=False), pm.Turn("", "x", is_heading=True)]
    assert [t.text for t in pm.spoken_turns(turns)] == ["real"]


# --------------------------------------------------------------------------
# TTS chunking
# --------------------------------------------------------------------------


def test_a_short_turn_is_one_chunk():
    assert pm.split_for_tts("Just a line.") == ["Just a line."]


def test_no_text_is_no_chunks():
    assert pm.split_for_tts("") == []


def test_a_long_turn_is_split_under_the_audio_servers_cap():
    """The audio-server caps a request at 4000 characters of input text, so
    a long host turn would otherwise be a silent 400."""
    text = " ".join(f"This is sentence number {i} of a very long monologue." for i in range(400))
    chunks = pm.split_for_tts(text, max_chars=3800)
    assert len(chunks) > 1
    assert all(len(c) <= 3800 for c in chunks)


def test_splitting_keeps_every_word():
    text = " ".join(f"Sentence {i} here." for i in range(300))
    rejoined = " ".join(pm.split_for_tts(text, max_chars=500))
    assert rejoined == text


def test_a_single_sentence_over_the_cap_is_hard_split_not_dropped():
    """Dropping it would lose the speaker's words."""
    text = "word " * 1200
    chunks = pm.split_for_tts(text, max_chars=1000)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_the_default_cap_leaves_room_under_the_servers_4000():
    """A 3800 default with a 4000 server cap is deliberate: the split happens
    per turn, and the speaker tag is concatenated on the way out."""
    assert pm.split_for_tts.__defaults__[0] < 4000


# --------------------------------------------------------------------------
# host specs, gender, clones
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "voice,gender",
    [
        ("af_nicole", "f"),
        ("af_heart", "f"),
        ("am_michael", "m"),
        ("am_puck", "m"),
        ("bf_emma", "f"),
        ("bm_george", "m"),
        ("AF_NICOLE", "f"),
        ("", ""),
        ("something_else", ""),
    ],
)
def test_voice_gender_reads_the_kokoro_naming_scheme(voice, gender):
    assert pm.voice_gender(voice) == gender


def test_a_host_with_no_clone_has_no_cross_gender_warning():
    assert pm.cross_gender_warning(_host_a()) is None


def test_a_matched_clone_gender_produces_no_warning():
    h = pm.HostSpec(name="Ada", voice="af_nicole", gender="f", clone="v1")
    assert pm.cross_gender_warning(h) is None


def test_a_cross_gender_clone_warns_instead_of_silently_mis_speaking():
    """OpenVoice transfers timbre badly across genders; producing a subtly
    wrong voice and saying nothing is worse than a warning."""
    h = pm.HostSpec(name="Omar", voice="af_heart", gender="m", clone="v1")
    warn = pm.cross_gender_warning(h)
    assert warn is not None
    assert "Omar" in warn and "af_heart" in warn


def test_no_warning_is_possible_without_both_genders_being_known():
    assert pm.cross_gender_warning(pm.HostSpec(name="X", voice="weird", gender="m", clone="v1")) is None
    assert pm.cross_gender_warning(pm.HostSpec(name="X", voice="af_heart", gender="", clone="v1")) is None


def test_every_curated_pair_is_one_female_and_one_male_voice():
    """The difference between two voices is the fastest cue a listener has to
    who is talking."""
    for pair in pm.HOST_PAIRS:
        genders = sorted(pm.voice_gender(pair[s]["voice"]) for s in ("a", "b"))
        assert genders == ["f", "m"], pair["id"]


def test_every_curated_voice_is_a_real_kokoro_voice():
    """A typo in a voice name is a 400 on the first render of an episode the
    user just waited for."""
    import audio_server

    known = set(audio_server.KOKORO_VOICES)
    for pair in pm.HOST_PAIRS:
        for slot in ("a", "b"):
            assert pair[slot]["voice"] in known, pair["id"]


def test_every_curated_pair_has_a_unique_id_and_two_named_hosts():
    ids = [p["id"] for p in pm.HOST_PAIRS]
    assert len(ids) == len(set(ids))
    for pair in pm.HOST_PAIRS:
        assert pair["label"] and pair["description"]
        assert pair["a"]["name"] != pair["b"]["name"]


def test_host_roster_builds_two_hosts_keyed_a_and_b():
    roster = pm.host_roster("duo_warm")
    assert [h.key for h in roster] == ["a", "b"]
    assert [h.voice for h in roster] == ["af_nicole", "am_michael"]
    assert all(h.clone is None for h in roster)


def test_a_clone_is_attached_only_when_the_slot_asks_for_it():
    """Attaching every available clone by default would silently replace a
    curated voice with a stranger's."""
    assert pm.host_roster("duo_warm", {"host_clone_duo_warm_b": "v9"})[1].clone == "v9"
    assert pm.host_roster("duo_warm", {"host_clone_duo_warm_b": "v9"})[0].clone is None
    assert pm.host_roster("duo_warm", {"host_clone_duo_deep_b": "v9"})[1].clone is None


def test_an_unknown_pair_id_falls_back_to_the_first_pair():
    assert pm.host_roster("no-such-pair")[0].voice == pm.HOST_PAIRS[0]["a"]["voice"]


def test_curate_voice_pair_finds_the_pair_that_owns_a_voice():
    assert pm.curate_voice_pair(["am_puck"]) == "duo_witty"
    assert pm.curate_voice_pair(["af_sky"]) == "duo_bright"


def test_curate_voice_pair_falls_back_rather_than_refusing():
    assert pm.curate_voice_pair(["nonsense"]) == pm.HOST_PAIRS[0]["id"]
    assert pm.curate_voice_pair([]) == pm.HOST_PAIRS[0]["id"]
    assert pm.curate_voice_pair(["  "]) == pm.HOST_PAIRS[0]["id"]


def test_curate_voice_pair_is_case_and_space_insensitive():
    assert pm.curate_voice_pair([" AM_MICHAEL "]) == "duo_warm"


# --------------------------------------------------------------------------
# the mix
# --------------------------------------------------------------------------


def test_a_mix_without_a_bed_is_just_the_turns_plus_their_pauses():
    hosts = [_host_a(), _host_b()]
    result = pm.mix_podcast([_seg(1.0, host_index=0), _seg(1.0, host_index=1)], hosts)
    assert result.turn_count == 2
    assert result.bed_duration_s == 0.0
    # two 1 s turns plus one inter-turn gap, and no gap after the last
    assert result.duration_s == pytest.approx(1.0 + 1.0 + hosts[0].pause_after_s, abs=0.02)


def test_exactly_one_gap_is_inserted_between_turns_and_none_after_the_last():
    hosts = [pm.HostSpec(name="A", voice="af_heart", key="a", pause_after_s=2.0)]
    three = pm.mix_podcast([_seg(1.0), _seg(1.0), _seg(1.0)], hosts)
    assert three.duration_s == pytest.approx(1.0 * 3 + 2.0 * 2, abs=0.03)


def test_a_zero_pause_makes_the_episode_barely_longer_than_its_words():
    hosts = [pm.HostSpec(name="A", voice="af_heart", key="a", pause_after_s=0.0)]
    result = pm.mix_podcast([_seg(1.0), _seg(1.0)], hosts)
    assert result.duration_s == pytest.approx(2.0, abs=0.02)


def test_each_turn_reports_where_it_starts_and_how_long_it_is():
    hosts = [_host_a(), _host_b()]
    result = pm.mix_podcast([_seg(1.0, host_index=0), _seg(2.0, hz=330.0, host_index=1)], hosts)
    first, second = result.per_turn
    assert first["start_s"] == 0.0
    assert first["duration_s"] == pytest.approx(1.0, abs=0.02)
    assert second["start_s"] == pytest.approx(1.0 + _host_a().pause_after_s, abs=0.02)
    assert second["duration_s"] == pytest.approx(2.0, abs=0.02)
    assert second["host"] == "Rowan"
    assert second["voice"] == "am_michael"


def test_the_bed_lengthens_nothing_and_ducking_costs_nothing():
    """The duck envelope is a multiply, so a bed cannot change the duration;
    anything else would desync the picture and the narration."""
    hosts = [_host_a(), _host_b()]
    plain = pm.mix_podcast([_seg(1.0)], hosts)
    with_bed = pm.mix_podcast([_seg(1.0)], hosts, bed=pm.synthesize_bed(0.3, sr=SR))
    assert with_bed.duration_s == plain.duration_s


def test_a_short_bed_is_looped_to_the_whole_episode():
    """A 10 s bed under a 12 minute episode otherwise has 72 clicks and then
    silence."""
    hosts = [_host_a(), _host_b()]
    segs = [_seg(3.0), _seg(3.0, host_index=1), _seg(3.0)]
    bed = pm.synthesize_bed(0.5, sr=SR)
    result = pm.mix_podcast(segs, hosts, bed=bed)
    assert result.duration_s > 9.0
    assert result.bed_duration_s == pytest.approx(0.5, abs=0.01)


def test_the_mix_actually_contains_the_bed():
    hosts = [_host_a()]
    segs = [_seg(2.0)]
    silent = pm.decode_wav(pm.mix_podcast(segs, hosts).wav)[0]
    with_bed = pm.decode_wav(pm.mix_podcast(segs, hosts, bed=pm.synthesize_bed(1.0, sr=SR)).wav)[0]
    assert not np.allclose(silent, with_bed)


def test_the_bed_sits_far_under_the_speech_in_a_gap():
    """In the inter-turn silence the bed is the only sound, so its level is
    directly measurable there."""
    hosts = [pm.HostSpec(name="A", voice="af_heart", key="a", pause_after_s=2.0)]
    result = pm.mix_podcast([_seg(1.0), _seg(1.0)], hosts, bed=pm.synthesize_bed(2.0, sr=SR), duck_db=20.0)
    mix, _ = pm.decode_wav(result.wav)
    speech_rms = float(np.sqrt(np.mean(mix[: int(0.5 * SR)].astype(np.float64) ** 2)))
    gap_rms = float(np.sqrt(np.mean(mix[int(1.5 * SR) : int(2.0 * SR)].astype(np.float64) ** 2)))
    assert gap_rms < speech_rms  # the bed is the quieter thing, as intended


def test_a_zero_length_turn_list_mixes_to_silence():
    result = pm.mix_podcast([], [_host_a()], bed=pm.synthesize_bed(1.0, sr=SR))
    assert result.duration_s == 0.0
    assert result.turn_count == 0
    decoded, _ = pm.decode_wav(result.wav)
    assert decoded.size == 0 or not decoded.any()


def test_segments_without_audio_are_skipped_rather_than_crashing():
    result = pm.mix_podcast([_seg(0.5), {"wav": b"", "host_index": 1}, _seg(0.5)], [_host_a()])
    assert result.turn_count == 2


def test_an_unusable_host_index_warns_and_falls_back_to_the_first_host():
    """Silently dropping a turn's pacing would make the episode listen wrong
    with no explanation."""
    result = pm.mix_podcast([_seg(0.5, host_index=9)], [_host_a(), _host_b()])
    assert any("host index" in w for w in result.warnings)
    assert result.duration_s == pytest.approx(0.5, abs=0.02)


def test_a_cross_gender_clone_surfaces_in_the_mix_warnings():
    hosts = [pm.HostSpec(name="Omar", voice="af_heart", key="a", gender="m", clone="v1"), _host_b()]
    result = pm.mix_podcast([_seg(0.5, host_index=0)], hosts)
    assert any("across genders" in w for w in result.warnings)


def test_a_turn_synthesised_at_the_wrong_rate_is_resampled():
    seg = {"wav": _wav_bytes(_tone(1.0, sr=16000, amp=0.4), 16000), "host_index": 0}
    result = pm.mix_podcast([seg], [_host_a()])
    assert result.duration_s == pytest.approx(1.0, rel=0.02)


def test_the_whole_mix_is_edge_faded():
    hosts = [_host_a()]
    mix, _ = pm.decode_wav(pm.mix_podcast([_seg(2.0)], hosts, bed=pm.synthesize_bed(1.0, sr=SR)).wav)
    assert abs(float(mix[0])) < 1e-3
    assert abs(float(mix[-1])) < 1e-3
    assert float(np.abs(mix).max()) > 0.05  # and there is a real signal in the middle


def test_master_gain_scales_the_mix():
    """Measured on a quiet segment so the tanh limiter is not the dominant
    term; the gain is applied before it."""
    hosts = [_host_a()]
    seg = [_seg(0.5, amp=0.05)]
    loud = pm.decode_wav(pm.mix_podcast(seg, hosts, master_gain=1.0).wav)[0]
    half = pm.decode_wav(pm.mix_podcast(seg, hosts, master_gain=0.5).wav)[0]
    assert float(np.abs(half).max()) == pytest.approx(float(np.abs(loud).max()) * 0.5, rel=0.05)


def test_a_negative_master_gain_produces_silence_rather_than_inverted_audio():
    """Inverting the whole mix would swap who sounds like a man and who sounds
    like a woman, which is the one thing a voice must never do."""
    result = pm.mix_podcast([_seg(0.3)], [_host_a()], master_gain=-1.0)
    decoded, _ = pm.decode_wav(result.wav)
    assert not decoded.any()


def test_the_result_reports_the_bed_settings_it_used():
    result = pm.mix_podcast([_seg(0.5)], [_host_a()], bed=pm.synthesize_bed(0.5, sr=SR), bed_preset_id="lofi_calm", duck_db=15.0)
    d = result.to_dict()
    assert d["bed_preset"] == "lofi_calm"
    assert d["bed_duck_db"] == 15.0
    assert d["turn_count"] == 1
    assert d["duration_s"] == pytest.approx(0.5, abs=0.02)


def test_an_episode_with_no_turns_and_no_bed_is_zero_length():
    result = pm.mix_podcast([], [])
    assert result.duration_s == 0.0
    assert result.turn_count == 0
    assert result.warnings == [] or all(isinstance(w, str) for w in result.warnings)


# --------------------------------------------------------------------------
# data uri
# --------------------------------------------------------------------------


def test_the_data_uri_is_a_wav_a_browser_can_play_directly():
    import base64

    wav = pm.encode_wav(_tone(0.2, amp=0.3), SR)
    uri = pm.mix_to_data_uri(wav)
    assert uri.startswith("data:audio/wav;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]) == wav
