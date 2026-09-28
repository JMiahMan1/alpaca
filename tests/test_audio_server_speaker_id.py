"""The two speaker-identification routes on the audio service.

`identify` and `calibrate` sit next to enrolment because the comparison needs
the OpenVoice reference encoder, which exists only inside the audio container -
and because Raven reaches them through the same tool surface. What matters here
is the boundary: a bad clip must be refused before the encoder is touched, an
explicit threshold must be honoured rather than silently replaced by the derived
one, and a cohort that cannot answer must say so with a 404 rather than an empty
200 that an agent would read as "nobody matched".
"""

from __future__ import annotations

import base64
import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("audio_server_under_test", REPO / "audio_server.py")
assert _spec and _spec.loader
audio = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audio)

#: A real WAV header so `_clip_from`'s decode check is satisfied; the encoder is
#: patched away, so the samples never have to be meaningful.
WAV = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00@\x1f\x00\x00\x80>\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
B64 = base64.b64encode(WAV).decode()


@pytest.fixture
def client():
    # Deliberately not a context manager: entering would fire the startup hook
    # and spawn the real idle-unloader task.
    return TestClient(audio.app)


@pytest.fixture(autouse=True)
def clean_state():
    snapshot = dict(audio._state)
    yield
    audio._state.clear()
    audio._state.update(snapshot)


@pytest.fixture
def identify(monkeypatch):
    """Capture what the route handed the module, and answer with `result`."""
    seen: dict = {}

    seen["calls"] = 0

    def _identify(raw, threshold=None):
        seen["calls"] += 1
        seen["raw"] = raw
        seen["threshold"] = threshold
        return seen["result"]

    monkeypatch.setattr(audio.voice_clone, "identify", _identify)
    return seen


@pytest.fixture
def calibrate(monkeypatch):
    seen: dict = {}

    seen["calls"] = 0

    def _calibrate(clips):
        seen["calls"] += 1
        seen["clips"] = clips
        return seen["result"]

    monkeypatch.setattr(audio.voice_clone, "calibrate", _calibrate)
    return seen


def _matched(score: float = 0.82) -> dict:
    return {
        "ok": True,
        "engine": "openvoice-v2-reference-encoder",
        "matched": "ada-e1d0bb",
        "matched_name": "Ada",
        "score": score,
        "threshold": 0.52,
        "margin": 0.3,
        "candidates": [{"id": "ada-e1d0bb", "score": score}, {"id": "bob-0255e2", "score": 0.4}],
    }


# --------------------------------------------------------------------------
# identify: the happy path
# --------------------------------------------------------------------------


def test_identify_returns_the_module_result_verbatim(client, identify):
    identify["result"] = _matched()
    r = client.post("/api/voices/identify", json={"audio_b64": B64})
    assert r.status_code == 200
    body = r.json()
    assert body["matched"] == "ada-e1d0bb"
    assert body["matched_name"] == "Ada"
    assert body["engine"] == "openvoice-v2-reference-encoder"


def test_identify_passes_the_decoded_audio_not_the_base64_string(client, identify):
    identify["result"] = _matched()
    client.post("/api/voices/identify", json={"audio_b64": B64})
    assert identify["raw"] == WAV, "the route must decode; the cloner takes bytes"


def test_identify_defaults_to_no_threshold_so_the_cohort_derives_one(client, identify):
    """Passing 0.0 would be worse than passing None: the module distinguishes
    "caller decided" from "derive it from the enrolled speakers"."""
    identify["result"] = _matched()
    client.post("/api/voices/identify", json={"audio_b64": B64})
    assert identify["threshold"] is None


def test_identify_forwards_an_explicit_threshold(client, identify):
    identify["result"] = _matched()
    client.post("/api/voices/identify", json={"audio_b64": B64, "threshold": 0.61})
    assert identify["threshold"] == pytest.approx(0.61)


def test_identify_accepts_a_numeric_string_threshold(client, identify):
    identify["result"] = _matched()
    r = client.post("/api/voices/identify", json={"audio_b64": B64, "threshold": "0.5"})
    assert r.status_code == 200
    assert identify["threshold"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# identify: rejected before the encoder is touched
# --------------------------------------------------------------------------


def test_identify_requires_audio(client, identify):
    r = client.post("/api/voices/identify", json={})
    assert r.status_code == 400
    assert identify["calls"] == 0, "a bad request must not load the reference encoder"


def test_identify_rejects_a_non_string_clip(client, identify):
    r = client.post("/api/voices/identify", json={"audio_b64": 12345})
    assert r.status_code == 400
    assert identify["calls"] == 0


def test_identify_rejects_a_clip_that_is_not_base64(client, identify):
    r = client.post("/api/voices/identify", json={"audio_b64": "not base64!!"})
    assert r.status_code == 400
    assert identify["calls"] == 0


def test_identify_rejects_an_empty_clip(client, identify):
    r = client.post("/api/voices/identify", json={"audio_b64": ""})
    assert r.status_code == 400
    assert identify["calls"] == 0


def test_identify_rejects_a_threshold_outside_zero_to_one(client, identify):
    identify["result"] = _matched()
    for bad in (-0.01, 1.01, 2, -1):
        r = client.post("/api/voices/identify", json={"audio_b64": B64, "threshold": bad})
        assert r.status_code == 400, bad
    assert identify["calls"] == 0, "an out-of-range threshold must not reach the module either"


def test_identify_rejects_a_threshold_that_is_not_a_number(client, identify):
    r = client.post("/api/voices/identify", json={"audio_b64": B64, "threshold": "high"})
    assert r.status_code == 400
    assert "number" in r.json()["error"]
    assert identify["calls"] == 0


def test_identify_accepts_the_threshold_bounds_themselves(client, identify):
    identify["result"] = _matched()
    for ok in (0.0, 1.0):
        assert client.post("/api/voices/identify", json={"audio_b64": B64, "threshold": ok}).status_code == 200


def test_identify_names_the_offending_threshold(client, identify):
    r = client.post("/api/voices/identify", json={"audio_b64": B64, "threshold": 1.5})
    assert "1.5" in r.json()["error"]


# --------------------------------------------------------------------------
# identify: outcomes that are not a match
# --------------------------------------------------------------------------


def test_identify_answers_404_when_nothing_is_enrolled(client, identify):
    """A 200 with no match would read as 'not this person'; the distinction
    between 'no cohort' and 'no match' is what lets an agent ask for enrolment."""
    identify["result"] = {"ok": False, "error": "no enrolled voices to compare against", "candidates": []}
    r = client.post("/api/voices/identify", json={"audio_b64": B64})
    assert r.status_code == 404
    assert "no enrolled voices" in r.json()["error"]


def test_identify_answers_422_when_the_clip_has_no_usable_speech(client, identify):
    identify["result"] = {"ok": False, "error": "not enough speech in the clip to identify", "candidates": []}
    r = client.post("/api/voices/identify", json={"audio_b64": B64})
    assert r.status_code == 422


def test_identify_still_returns_the_ranking_when_nothing_matches(client, identify):
    """Refusing to name someone must not discard the evidence: the agent needs
    'closest is Ada' even when that is below the floor."""
    result = _matched(score=0.31)
    result |= {"matched": None, "matched_name": None}
    identify["result"] = result
    body = client.post("/api/voices/identify", json={"audio_b64": B64}).json()
    assert body["matched"] is None
    assert [c["id"] for c in body["candidates"]] == ["ada-e1d0bb", "bob-0255e2"]


def test_identify_answers_422_when_the_cloner_raises_value_error(client, monkeypatch):
    def _boom(raw, threshold=None):
        raise ValueError("not enough speech to build a voice embedding")

    monkeypatch.setattr(audio.voice_clone, "identify", _boom)
    r = client.post("/api/voices/identify", json={"audio_b64": B64})
    assert r.status_code == 422
    assert "embedding" in r.json()["error"]


def test_identify_answers_500_for_an_unexpected_failure(client, monkeypatch):
    def _boom(raw, threshold=None):
        raise RuntimeError("the reference encoder fell over")

    monkeypatch.setattr(audio.voice_clone, "identify", _boom)
    r = client.post("/api/voices/identify", json={"audio_b64": B64})
    assert r.status_code == 500
    assert "fell over" in r.json()["error"]


# --------------------------------------------------------------------------
# calibrate
# --------------------------------------------------------------------------


def _cal_ok() -> dict:
    return {
        "ok": True,
        "suggested_threshold": 0.55,
        "clips": [{"label": "ada-e1d0bb", "own_score": 0.8, "impostor_score": 0.3, "margin": 0.5}],
        "weakest_clip": "ada-e1d0bb",
    }


def test_calibrate_returns_the_module_result(client, calibrate):
    calibrate["result"] = _cal_ok()
    r = client.post("/api/voices/calibrate", json={"clips": [{"voice_id": "ada-e1d0bb", "audio_b64": B64}]})
    assert r.status_code == 200
    assert r.json()["suggested_threshold"] == pytest.approx(0.55)


def test_calibrate_pairs_each_label_with_its_decoded_audio(client, calibrate):
    calibrate["result"] = _cal_ok()
    client.post(
        "/api/voices/calibrate",
        json={"clips": [{"voice_id": "ada-e1d0bb", "audio_b64": B64}, {"voice_id": "bob-0255e2", "audio_b64": B64}]},
    )
    assert calibrate["clips"] == [("ada-e1d0bb", WAV), ("bob-0255e2", WAV)]


def test_calibrate_requires_a_non_empty_list(client, calibrate):
    for body in ({}, {"clips": []}, {"clips": "nope"}, {"clips": {"voice_id": "x"}}):
        r = client.post("/api/voices/calibrate", json=body)
        assert r.status_code == 400, body
    assert calibrate["calls"] == 0


def test_calibrate_caps_the_batch(client, calibrate):
    """The encoder is loaded once and each clip is embedded; an unbounded list
    would let one call hold the shared card for minutes."""
    clips = [{"voice_id": f"v{i}", "audio_b64": B64} for i in range(21)]
    r = client.post("/api/voices/calibrate", json={"clips": clips})
    assert r.status_code == 400
    assert "20" in r.json()["error"]
    assert calibrate["calls"] == 0


def test_calibrate_accepts_exactly_twenty_clips(client, calibrate):
    calibrate["result"] = _cal_ok()
    clips = [{"voice_id": f"v{i}", "audio_b64": B64} for i in range(20)]
    assert client.post("/api/voices/calibrate", json={"clips": clips}).status_code == 200


def test_calibrate_rejects_a_non_object_element(client, calibrate):
    r = client.post("/api/voices/calibrate", json={"clips": [B64]})
    assert r.status_code == 400
    assert "clip 1" in r.json()["error"]
    assert calibrate["calls"] == 0


def test_calibrate_names_the_offending_clip(client, calibrate):
    """With a 20-clip batch, 'bad audio' is useless; the index is the whole
    message."""
    clips = [{"voice_id": "ok", "audio_b64": B64}, {"voice_id": "bad", "audio_b64": "???"}]
    r = client.post("/api/voices/calibrate", json={"clips": clips})
    assert r.status_code == 400
    assert r.json()["error"].startswith("clip 2:")


def test_calibrate_tolerates_a_clip_with_no_label(client, calibrate):
    """An unlabelled clip still measures something (its own best score and the
    impostor it beat), which is how you discover a new voice to enrol."""
    calibrate["result"] = _cal_ok()
    r = client.post("/api/voices/calibrate", json={"clips": [{"audio_b64": B64}]})
    assert r.status_code == 200
    assert calibrate["clips"] == [("", WAV)]


def test_calibrate_answers_422_on_value_error(client, monkeypatch):
    def _boom(clips):
        raise ValueError("at least one recording is required")

    monkeypatch.setattr(audio.voice_clone, "calibrate", _boom)
    r = client.post("/api/voices/calibrate", json={"clips": [{"voice_id": "a", "audio_b64": B64}]})
    assert r.status_code == 422


def test_calibrate_answers_500_on_an_unexpected_failure(client, monkeypatch):
    def _boom(clips):
        raise RuntimeError("disk gone")

    monkeypatch.setattr(audio.voice_clone, "calibrate", _boom)
    r = client.post("/api/voices/calibrate", json={"clips": [{"voice_id": "a", "audio_b64": B64}]})
    assert r.status_code == 500
    assert "disk gone" in r.json()["error"]


def test_calibrate_answers_404_when_nothing_is_enrolled(client, calibrate):
    calibrate["result"] = {"ok": False, "error": "no enrolled voices to compare against", "clips": []}
    r = client.post("/api/voices/calibrate", json={"clips": [{"voice_id": "a", "audio_b64": B64}]})
    assert r.status_code == 404


# --------------------------------------------------------------------------
# Boundaries shared with the rest of the service
# --------------------------------------------------------------------------


def test_a_get_on_either_route_is_405(client):
    """They take audio, so they are POST-only - a GET would be a URL with no
    meaning."""
    assert client.get("/api/voices/identify").status_code == 405
    assert client.get("/api/voices/calibrate").status_code == 405


def test_enrolment_still_works_alongside_them(client, monkeypatch):
    """The new routes must not have displaced POST /api/voices, which is what
    creates a profile in the first place."""
    assert any(getattr(r, "path", "") == "/api/voices" for r in audio.app.routes)
    assert any(getattr(r, "path", "") == "/api/voices/identify" for r in audio.app.routes)
    assert any(getattr(r, "path", "") == "/api/voices/calibrate" for r in audio.app.routes)


def test_the_two_routes_and_enrolment_all_take_the_same_lock(client, monkeypatch):
    """Identification reads the profiles enrolment is writing. If they can run
    concurrently a clip can be scored against a half-written centroid."""
    source = (REPO / "audio_server.py").read_text()
    for name in ("api_voices_create", "api_voices_identify", "api_voices_calibrate"):
        body = source.split(f"async def {name}", 1)[1].split("\n@app.", 1)[0]
        assert "async with _lock" in body, name
