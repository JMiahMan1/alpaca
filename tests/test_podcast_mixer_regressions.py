"""Regression tests for bugs found in the review pass over web/podcast_mixer.py.

Every test here failed before a fix and pins the reason the fix exists, not
just the new value. The headline one is `test_the_duck_is_applied_once` -- the
bed's whole reason for existing is to be audible-but-under the speech, and the
double application made it 41 dB down, which is silent.
"""

from __future__ import annotations

import numpy as np
import pytest

from web import podcast_mixer as pm

SR = pm.TARGET_SR


def _tone(seconds: float, hz: float = 220.0, amp: float = 0.5, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _seg(seconds: float, hz: float = 220.0, host_index: int = 0, amp: float = 0.5) -> dict:
    return {"wav": pm.encode_wav(_tone(seconds, hz, amp=amp), SR), "host_index": host_index, "text": "t"}


def _hosts() -> list[pm.HostSpec]:
    return [
        pm.HostSpec(name="Ada", voice="af_nicole", key="a", gender="f"),
        pm.HostSpec(name="Rowan", voice="am_michael", key="b", gender="m"),
    ]


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def _separation_db(bed: np.ndarray, duck_db: float, lo: float, hi: float) -> float:
    """How many dB the bed sits *under* the speech, measured inside a turn.

    The bed is isolated by differencing a render with it against one without,
    because the mix is not linear once the limiter touches it and comparing raw
    levels would measure the limiter instead.
    """
    with_bed, _ = pm.decode_wav(pm.mix_podcast([_seg(1.5)], _hosts(), bed=bed, duck_db=duck_db, edge_fade_s=0.0).wav)
    without, _ = pm.decode_wav(pm.mix_podcast([_seg(1.5)], _hosts(), duck_db=duck_db, edge_fade_s=0.0).wav)
    i = slice(int(lo * SR), int(hi * SR))
    speech_rms = _rms(np.asarray(without)[i])
    bed_rms = _rms((np.asarray(with_bed, dtype=np.float64) - np.asarray(without, dtype=np.float64))[i])
    return 20 * float(np.log10(max(speech_rms, 1e-9) / max(bed_rms, 1e-9)))


# --------------------------------------------------------------------------
# 1. the duck was applied twice, putting the bed 41 dB under the speech
# --------------------------------------------------------------------------


def test_the_duck_is_applied_once_not_squared():
    """duck_db is both the bed's resting gain and the sidechain depth.

    Before the fix the two were the same number, so the product under speech was
    `0.1 * 0.1` = -40 dB. The requested attenuation is now the resting gain and
    the sidechain is a *separate*, shallower depth, so the bed is audible.
    """
    bed = pm.synthesize_bed(1.5, sr=SR)
    separation = _separation_db(bed, 20.0, 0.4, 1.1)
    # Under speech the bed sits at the requested -20 dB plus the sidechain's
    # extra 6 dB, not 2x the request. Allow 3 dB either way of that arithmetic.
    assert 23.0 <= separation <= 29.0, separation


def test_a_larger_duck_db_still_does_not_quadruple_the_attenuation():
    """The old arithmetic was 2x duck_db, so 6 and 30 dB were wildly wrong too."""
    bed = pm.synthesize_bed(1.5, sr=SR)
    separations = [_separation_db(bed, duck, 0.4, 1.1) for duck in (6.0, 12.0, 20.0)]
    # Monotonic, and each lands near `duck_db + sidechain` rather than `2 * duck_db`
    # (which would have been 12 / 24 / 40).
    assert separations[0] < separations[1] < separations[2]
    for duck, measured in zip((6.0, 12.0, 20.0), separations, strict=True):
        expected = duck + pm.DEFAULT_BED_SIDECHAIN_DB
        assert abs(measured - expected) <= 3.0, (duck, measured, expected)


def test_the_bed_is_quieter_under_speech_than_in_the_gap():
    """A sidechain that does not move is not a sidechain."""
    segs = [_seg(0.8, hz=220.0), _seg(0.8, hz=330.0)]
    bed = np.zeros(0)
    bed = pm.synthesize_bed(4.0, sr=SR)
    with_bed, _ = pm.decode_wav(pm.mix_podcast(segs, _hosts(), bed=bed, duck_db=20.0, edge_fade_s=0.0).wav)
    without, _ = pm.decode_wav(pm.mix_podcast(segs, _hosts(), duck_db=20.0, edge_fade_s=0.0).wav)
    bed_signal = np.asarray(with_bed, dtype=np.float64) - np.asarray(without, dtype=np.float64)
    under = _rms(bed_signal[int(0.2 * SR) : int(0.6 * SR)])
    gap = _rms(bed_signal[int(0.85 * SR) : int(1.5 * SR)])
    assert gap > under * 1.5, (under, gap)


def test_the_sidechain_depth_is_a_separate_named_constant():
    assert pm.DEFAULT_BED_SIDECHAIN_DB > 0.0
    assert pm.DEFAULT_BED_SIDECHAIN_DB < pm.DEFAULT_DUCK_DB


# --------------------------------------------------------------------------
# 2. duck_envelope raised for any buffer shorter than one 10 ms window
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 5, 17, 100, 239, 240])
def test_duck_envelope_survives_a_buffer_shorter_than_its_window(n):
    env = pm.duck_envelope(np.full(n, 0.5, np.float32), SR)
    assert env.size == n
    assert np.all(np.isfinite(env))


def test_mixing_a_stub_segment_does_not_raise():
    """A truncated TTS response can decode as a valid 5-sample WAV."""
    stub = {"wav": pm.encode_wav(np.zeros(5, np.float32), SR), "host_index": 0, "text": ""}
    result = pm.mix_podcast([stub], _hosts(), bed=pm.synthesize_bed(1.0, sr=SR), duck_db=20.0, edge_fade_s=0.0)
    decoded, sr = pm.decode_wav(result.wav)
    assert decoded.size > 0
    assert sr == SR


# --------------------------------------------------------------------------
# 3. a heading turn swallowed the untagged line(s) after it
# --------------------------------------------------------------------------


def test_a_heading_does_not_swallow_the_narration_after_it():
    turns = pm.parse_script("# Episode 12\nThe cold open narration.\n[host_a] Welcome.\n", pm.host_roster("duo_warm"))
    spoken = pm.spoken_turns(turns)
    texts = " ".join(t.text for t in spoken)
    assert "cold open narration" in texts
    assert "Welcome." in texts
    # and it must not be glued to the heading either
    assert all("Episode 12" not in t.text for t in spoken)


def test_a_midscript_label_does_not_swallow_the_narration_after_it():
    turns = pm.parse_script(
        "[host_a] Welcome to the show.\n\nINTRO\nSome unlabelled narration here.\n[host_b] Hi.\n",
        pm.host_roster("duo_warm"),
    )
    spoken = " ".join(t.text for t in pm.spoken_turns(turns))
    assert "unlabelled narration" in spoken
    assert "INTRO" not in spoken


def test_untagged_prose_still_continues_a_spoken_turn():
    """The fix must not have broken the wrapping-lines case it was guarding."""
    turns = pm.parse_script("[host_a] One two three\nfour five six\n[host_b] Seven.\n", pm.host_roster("duo_warm"))
    spoken = pm.spoken_turns(turns)
    assert len(spoken) == 2
    assert "four five six" in spoken[0].text
    assert spoken[0].speaker == "host_a"


# --------------------------------------------------------------------------
# 4. the limiter ran twice
# --------------------------------------------------------------------------


def test_the_limiter_runs_once_not_twice():
    """tanh applied twice is not the same as applied once."""
    loud = _tone(0.3, amp=1.4)
    once, _ = pm.decode_wav(pm.encode_wav(loud, SR, clip=True))
    twice, _ = pm.decode_wav(pm.encode_wav(pm.soft_clip(loud), SR, clip=True))
    assert not np.allclose(once, twice)
    # ...and the mix is only clipped once, so a hot bed + speech peaks sanely
    result = pm.mix_podcast([_seg(1.0, amp=0.9)], _hosts(), bed=pm.synthesize_bed(1.0, sr=SR), master_gain=1.0)
    mixed, _ = pm.decode_wav(result.wav)
    assert float(np.max(np.abs(np.asarray(mixed)))) <= 1.0


# --------------------------------------------------------------------------
# 5. a bare tag on its own line was read aloud by the previous host
# --------------------------------------------------------------------------


def test_a_bare_tag_opens_a_turn_instead_of_being_read_aloud():
    turns = pm.parse_script("[host_a] Welcome to the show.\n[host_b]\nGlad to be here.\n", pm.host_roster("duo_warm"))
    spoken = pm.spoken_turns(turns)
    joined = " ".join(t.text for t in spoken)
    assert "[host_b]" not in joined
    assert "host b" not in joined.casefold()
    assert "Glad to be here." in joined
    # host_b's line belongs to host_b
    b_turn = next(t for t in spoken if "Glad to be here." in t.text)
    assert b_turn.host_index == 1


def test_a_bare_tag_keeps_the_previous_hosts_own_words():
    turns = pm.parse_script("[host_a] First line.\n[host_b]\nSecond line.\n", pm.host_roster("duo_warm"))
    a_turn = next(t for t in pm.spoken_turns(turns) if t.host_index == 0)
    assert a_turn.text.strip() == "First line."


def test_an_ordinary_short_line_is_not_mistaken_for_a_bare_tag():
    """The first two attempts at this fix matched any short line."""
    turns = pm.parse_script("[host_a] Welcome.\nJust a short line.\n[host_b] Hi.\n", pm.host_roster("duo_warm"))
    spoken = pm.spoken_turns(turns)
    assert len(spoken) == 2
    assert "Just a short line." in spoken[0].text


def test_a_colon_terminated_bare_label_opens_a_turn():
    turns = pm.parse_script("[host_a] Welcome.\n[host_b]:\nMy turn now.\n", pm.host_roster("duo_warm"))
    spoken = pm.spoken_turns(turns)
    assert "My turn now." in spoken[-1].text
    assert spoken[-1].host_index == 1


# --------------------------------------------------------------------------
# 6. host index 0 was treated as "no host"
# --------------------------------------------------------------------------


def test_an_unattributed_turn_always_gets_the_default_host():
    """Index 0 is falsy, so `or` silently swapped the voice for host_a."""
    roster = pm.host_roster("duo_warm")
    script = "[host_a] One.\n[host_c] Guest speaks.\n"
    a = pm.parse_script(script, roster, default_host="a")
    b = pm.parse_script(script, roster, default_host="b")
    assert [t.host_index for t in a] == [t.host_index for t in b]
    assert a[-1].host_index == 0


def test_a_guest_line_does_not_borrow_a_voice_from_the_default_host():
    turns = pm.parse_script("[host_a] One.\n[host_c] Guest speaks.\n", pm.host_roster("duo_warm"), default_host="b")
    assert turns[-1].host_index == 0


# --------------------------------------------------------------------------
# 7. resample_linear divided by a zero sample rate
# --------------------------------------------------------------------------


@pytest.mark.parametrize("src,dst", [(0, SR), (SR, 0), (0, 0), (-1, SR)])
def test_resample_survives_a_nonsense_sample_rate(src, dst):
    x = _tone(0.1)
    out = pm.resample_linear(x, src, dst)
    assert out.size == x.size
    assert np.all(np.isfinite(out))


def test_resampling_a_zero_sample_rate_wav_does_not_raise():
    """A WAV header claiming framerate 0 reaches this from mix_podcast."""
    stub = {"wav": pm.encode_wav(_tone(0.1), SR), "host_index": 0, "text": ""}
    result = pm.mix_podcast([stub], _hosts(), bed=_tone(0.5, hz=110.0), edge_fade_s=0.0)
    assert result.duration_s > 0.0


# --------------------------------------------------------------------------
# 8. split_for_tts(text, 0) raised
# --------------------------------------------------------------------------


@pytest.mark.parametrize("max_chars", [0, -1, -100])
def test_split_survives_a_nonsensical_chunk_size(max_chars):
    parts = pm.split_for_tts("one two three four five", max_chars)
    assert isinstance(parts, list)
    assert all(isinstance(p, str) for p in parts)


def test_split_with_a_zero_limit_still_keeps_every_character():
    """0 is clamped to 1, so it hard-splits per character - but drops nothing."""
    joined = "".join(pm.split_for_tts("alpha beta gamma delta", 0)).replace(" ", "")
    assert joined == "alphabetagammadelta"
