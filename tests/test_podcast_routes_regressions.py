"""Regression tests for bugs found in the review pass over /api/podcast/render.

The mixer itself has its own regression file. What is pinned here is the
wiring: a bad number in the request body must become a 400 the panel can read,
not an HTML 500, and it must be caught *before* any speech is synthesised --
by the time a render is three minutes in, discovering the request was invalid
is the most expensive possible time to learn it.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
from unittest.mock import Mock, patch

import numpy as np
import pytest

from web import podcast_mixer as pm
from web.app import app

SR = pm.TARGET_SR


# --------------------------------------------------------------------------
# helpers (the httpx mock pattern used across the route test files)
# --------------------------------------------------------------------------


class _Rec:
    def __init__(self) -> None:
        self.timeout = None


class _Ctx:
    def __init__(self, http):
        self._http = http

    def __enter__(self):
        return self._http

    def __exit__(self, *exc):
        return False


def _upstream(payload, status_code=200):
    resp = Mock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.text = str(payload)
    http = Mock()
    for verb in ("get", "post", "patch", "delete", "put"):
        getattr(http, verb).return_value = resp
    return http


@contextmanager
def _patched(http):
    rec = _Rec()

    def _ctor(*args, **kwargs):
        rec.timeout = kwargs.get("timeout", args[0] if args else None)
        return _Ctx(http)

    with patch("web.app.httpx.Client", side_effect=_ctor):
        yield rec


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def _wav_b64(seconds: float = 0.4, hz: float = 220.0, amp: float = 0.4) -> str:
    t = np.arange(int(seconds * SR), dtype=np.float32) / SR
    samples = (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)
    return base64.b64encode(pm.encode_wav(samples, SR)).decode("ascii")


def _seg(**overrides) -> dict:
    seg = {"wav": _wav_b64(), "host_index": 0, "text": "already rendered"}
    seg.update(overrides)
    return seg


def _render(client, body: dict):
    """POST a pre-rendered render with the audio service mocked out."""
    with _patched(_upstream({})):
        return client.post("/api/podcast/render", json=body)


def _peak(body: dict) -> float:
    samples, _ = pm.decode_wav(base64.b64decode(body["wav_b64"]))
    return float(np.max(np.abs(np.asarray(samples))))


# --------------------------------------------------------------------------
# 9. an unparseable number produced an HTML 500, and only after synthesis
# --------------------------------------------------------------------------

BAD_NUMBERS = [
    ("bed_bpm", "fast"),
    ("bed_density", "high"),
    ("bed_brightness", "bright"),
    ("bed_seed", "yesterday"),
    ("duck_db", "deep"),
    ("edge_fade_s", "x"),
]


@pytest.mark.parametrize("key,bad", BAD_NUMBERS)
def test_a_non_numeric_field_is_a_400_not_a_500(client, key, bad):
    """A stale panel, a hand-edited curl or an agent that sends a string used
    to produce an HTML 500 - and the panel's own res.json() then threw, so the
    user saw a generic failure instead of which field was wrong."""
    resp = _render(client, {"segments": [_seg()], key: bad})
    assert resp.status_code == 400
    assert key in resp.get_json()["error"]
    assert resp.mimetype == "application/json"


@pytest.mark.parametrize("key,bad", BAD_NUMBERS)
def test_a_bad_number_is_rejected_before_anything_is_synthesised(client, key, bad):
    """A render is tens of minutes of TTS. Validating afterwards would burn all
    of it and then refuse."""
    http = _upstream({})
    with _patched(http):
        resp = client.post("/api/podcast/render", json={"script": "[host_a] One.", key: bad})
    assert resp.status_code == 400
    assert http.post.call_count == 0


def test_the_error_names_the_offending_value(client):
    body = _render(client, {"segments": [_seg()], "bed_bpm": "fast"}).get_json()
    assert "fast" in body["error"]


def test_a_numeric_string_is_accepted(client):
    """A JSON number and its string form mean the same thing to an agent."""
    body = _render(client, {"segments": [_seg()], "duck_db": "18"}).get_json()
    assert body["bed_duck_db"] == 18.0


def test_a_valid_number_still_renders(client):
    for key, value in (("bed_bpm", 92), ("bed_density", 0.4), ("bed_brightness", 0.6), ("bed_seed", 7), ("duck_db", 12.0), ("edge_fade_s", 0.5)):
        resp = _render(client, {"segments": [_seg()], key: value})
        assert resp.status_code == 200, (key, value, resp.get_json())
        assert resp.get_json()["wav_b64"]


def test_a_string_key_for_the_bed_note_is_not_a_number_field(client):
    """`bed_key` is a musical key, so 5 is legal and must not be coerced."""
    resp = _render(client, {"segments": [_seg()], "bed_key": 5})
    assert resp.status_code == 200


# --------------------------------------------------------------------------
# 10. master_gain=0 silently became 1.0
# --------------------------------------------------------------------------


def test_a_zero_master_gain_is_silence_not_full_volume(client):
    """`float(data.get("master_gain") or 1.0)` treats 0 as absent."""
    body = _render(client, {"segments": [_seg()], "master_gain": 0}).get_json()
    assert _peak(body) == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize("gain", [0.25, 0.5, 1.0])
def test_master_gain_scales_the_output(client, gain):
    body = _render(client, {"segments": [_seg()], "master_gain": gain}).get_json()
    peak = _peak(body)
    assert 0.0 < peak <= 1.0
    if gain < 1.0:
        assert peak < 0.35, peak


def test_master_gain_scales_monotonically(client):
    peaks = [_peak(_render(client, {"segments": [_seg()], "master_gain": g}).get_json()) for g in (0.25, 0.5, 1.0)]
    assert peaks[0] < peaks[1] < peaks[2]
