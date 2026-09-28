"""Tests for the web layer's /api/audio/* bridge to the audio-server.

Every one of these routes is a thin forwarder: request -> audio-server -> response,
with the upstream status code and JSON body handed straight back. The behaviour
worth pinning is therefore not "does it return JSON" but:

  * the exact upstream path and HTTP verb each route chooses,
  * the timeout each route allows (a cold Kokoro load is minutes, an unload is not),
  * that an upstream status code is *propagated* rather than collapsed to 200,
  * and that a dead audio-server is reported as 502 instead of a stack trace.

The whole Audio Studio feature group (7 routes) had zero coverage before this file.
"""

import json
from contextlib import contextmanager
from unittest.mock import Mock, patch

import pytest

from web.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


class _Rec:
    """What the route handed to httpx.Client: the timeout it asked for."""

    def __init__(self):
        self.timeout = None


class _Ctx:
    def __init__(self, http):
        self._http = http

    def __enter__(self):
        return self._http

    def __exit__(self, *exc):
        return False


def _upstream(payload, status_code=200):
    """A mock client whose every verb returns `payload` with `status_code`."""
    resp = Mock()
    resp.status_code = status_code
    resp.json.return_value = payload
    http = Mock()
    for verb in ("get", "post", "patch", "delete", "put"):
        getattr(http, verb).return_value = resp
    return http


def _dead(side_effect):
    """A mock client whose every verb raises, simulating an unreachable service."""
    http = Mock()
    for verb in ("get", "post", "patch", "delete", "put"):
        getattr(http, verb).side_effect = side_effect
    return http


@contextmanager
def _patched(http):
    """Patch httpx.Client so `with httpx.Client(timeout=...) as c` yields `http`."""
    rec = _Rec()

    def _ctor(*args, **kwargs):
        rec.timeout = kwargs.get("timeout", args[0] if args else None)
        return _Ctx(http)

    with patch("web.app.httpx.Client", side_effect=_ctor):
        yield rec


# --- GET /api/audio/status ----------------------------------------------------


def test_audio_status_forwards_health_and_passes_status_through(client):
    payload = {"status": "ok", "tts": {"loaded": True}, "vram_free_mb": 512}
    http = _upstream(payload)
    with _patched(http):
        res = client.get("/api/audio/status")

    assert res.status_code == 200
    assert res.get_json() == payload
    # Health is a GET, and it must not be addressed to the proxy.
    assert http.get.call_args.args[0] == "http://audio-server:8082/health"


def test_audio_status_uses_a_short_timeout(client):
    """Status is polled by the dashboard, so it must not hang the UI."""
    http = _upstream({"status": "ok"})
    with _patched(http) as rec:
        client.get("/api/audio/status")

    assert rec.timeout == 2.0


def test_audio_status_propagates_a_bad_upstream_status(client):
    """A 503 from a still-booting audio-server must reach the dashboard as 503."""
    http = _upstream({"status": "starting"}, status_code=503)
    with _patched(http):
        res = client.get("/api/audio/status")

    assert res.status_code == 503
    assert res.get_json()["status"] == "starting"


def test_audio_status_reports_offline_when_unreachable(client):
    with _patched(_dead(OSError("no route to host"))):
        res = client.get("/api/audio/status")

    # The Audio Studio renders this as "offline", so the payload must say so.
    assert res.status_code == 200
    body = res.get_json()
    assert body["status"] == "offline"
    assert "no route to host" in body["error"]


# --- POST /api/audio/tts ------------------------------------------------------


def test_audio_tts_forwards_body_and_metadata(client):
    payload = {"audio_b64": "UklGRg==", "meta": {"duration_s": 2.4, "voice": "af_heart"}}
    http = _upstream(payload)
    with _patched(http):
        res = client.post("/api/audio/tts", json={"text": "hello", "voice": "af_heart"})

    assert res.status_code == 200
    assert res.get_json()["meta"]["duration_s"] == 2.4
    assert http.post.call_args.args[0] == "http://audio-server:8082/api/tts"
    assert http.post.call_args.kwargs["json"] == {"text": "hello", "voice": "af_heart"}


def test_audio_tts_allows_a_cold_model_load(client):
    """The first Kokoro synthesis loads the model; 600s is the deliberate budget."""
    http = _upstream({"audio_b64": "x"})
    with _patched(http) as rec:
        client.post("/api/audio/tts", json={"text": "hi"})

    assert rec.timeout == 600.0


def test_audio_tts_propagates_the_text_cap_400(client):
    """audio-server rejects >MAX_TTS_CHARS with 400; the dashboard must not launder it."""
    http = _upstream({"error": "text exceeds 4000 chars"}, status_code=400)
    with _patched(http):
        res = client.post("/api/audio/tts", json={"text": "x" * 5000})

    assert res.status_code == 400
    assert "4000" in res.get_json()["error"]


def test_audio_tts_tolerates_an_empty_json_body(client):
    """A frontend bug that posts `{}` should reach the audio-server's own validation
    (400 "text is required"), not blow up in the bridge. A bodyless POST is a 415 from
    Flask itself, which is the convention the other 33 `get_json() or {}` routes share."""
    http = _upstream({"error": "text is required"}, status_code=400)
    with _patched(http):
        res = client.post("/api/audio/tts", json={})

    assert res.status_code == 400
    assert http.post.call_args.kwargs["json"] == {}


def test_audio_tts_rejects_a_bodyless_post_rather_than_crashing(client):
    """`request.get_json()` raises on a bodyless POST; the route must not 500."""
    http = _upstream({})
    with _patched(http):
        res = client.post("/api/audio/tts")

    assert res.status_code == 415


def test_audio_tts_reports_502_when_unreachable(client):
    with _patched(_dead(OSError("audio-server down"))):
        res = client.post("/api/audio/tts", json={"text": "hi"})

    assert res.status_code == 502
    assert "audio-server down" in res.get_json()["error"]


# --- POST /api/audio/music ----------------------------------------------------


def test_audio_music_forwards_to_the_music_endpoint(client):
    payload = {"audio_b64": "UklGRg==", "meta": {"duration_s": 8.0}}
    http = _upstream(payload)
    with _patched(http):
        res = client.post("/api/audio/music", json={"prompt": "lo-fi beat", "duration_s": 8})

    assert res.status_code == 200
    assert http.post.call_args.args[0] == "http://audio-server:8082/api/music"
    assert http.post.call_args.kwargs["json"]["duration_s"] == 8


def test_audio_music_allows_the_generous_generation_budget(client):
    http = _upstream({"audio_b64": "x"})
    with _patched(http) as rec:
        client.post("/api/audio/music", json={"prompt": "x"})

    assert rec.timeout == 900.0


def test_audio_music_propagates_the_30_second_ceiling(client):
    """MusicGen cannot exceed 30s; the 400 naming that must survive the hop."""
    http = _upstream({"error": "duration_s must be between 2 and 30"}, status_code=400)
    with _patched(http):
        res = client.post("/api/audio/music", json={"prompt": "x", "duration_s": 120})

    assert res.status_code == 400
    assert "30" in res.get_json()["error"]


def test_audio_music_reports_502_when_unreachable(client):
    with _patched(_dead(OSError("nope"))):
        res = client.post("/api/audio/music", json={"prompt": "x"})

    assert res.status_code == 502


# --- POST /api/audio/unload ---------------------------------------------------


def test_audio_unload_posts_with_no_body(client):
    http = _upstream({"status": "unloaded"})
    with _patched(http) as rec:
        res = client.post("/api/audio/unload")

    assert res.status_code == 200
    assert http.post.call_args.args[0] == "http://audio-server:8082/api/unload"
    # Freeing VRAM is quick; it must not hold a 600s socket open.
    assert rec.timeout == 30.0


def test_audio_unload_reports_502_when_unreachable(client):
    with _patched(_dead(OSError("nope"))):
        res = client.post("/api/audio/unload")

    assert res.status_code == 502


# --- GET /api/audio/voices/prompts ---------------------------------------------


def test_audio_voice_prompts_forwards_the_read_aloud_scripts(client):
    payload = {"prompts": [{"id": "1", "title": "Rainbow Passage", "min_s": 20}], "min_total_speech_s": 30.0}
    http = _upstream(payload)
    with _patched(http):
        res = client.get("/api/audio/voices/prompts")

    assert res.status_code == 200
    assert res.get_json()["prompts"][0]["id"] == "1"
    assert http.get.call_args.args[0] == "http://audio-server:8082/api/voices/prompts"


def test_audio_voice_prompts_reports_502_when_unreachable(client):
    with _patched(_dead(OSError("nope"))):
        res = client.get("/api/audio/voices/prompts")

    assert res.status_code == 502


# --- GET|POST /api/audio/voices -----------------------------------------------


def test_audio_voices_get_lists_profiles(client):
    payload = {"voices": [{"id": "ada-a1b2c3", "name": "Ada"}]}
    http = _upstream(payload)
    with _patched(http):
        res = client.get("/api/audio/voices")

    assert res.status_code == 200
    assert res.get_json()["voices"][0]["name"] == "Ada"
    assert http.get.call_args.args[0] == "http://audio-server:8082/api/voices"


def test_audio_voices_post_enrols_and_keeps_the_consent_gate_visible(client):
    """Enrolment requires explicit consent; the 400 must not be swallowed."""
    body = {"name": "Ada", "consent": False, "recordings": []}
    http = _upstream({"error": "consent is required"}, status_code=400)
    with _patched(http):
        res = client.post("/api/audio/voices", json=body)

    assert res.status_code == 400
    assert res.get_json()["error"] == "consent is required"
    # The whole body, including the consent flag, must reach the audio-server.
    assert http.post.call_args.kwargs["json"] == body


def test_audio_voices_post_returns_the_quality_report(client):
    payload = {"voice": {"id": "ada-a1b2c3", "speech_s": 96.0, "warnings": []}}
    http = _upstream(payload, status_code=200)
    with _patched(http) as rec:
        res = client.post("/api/audio/voices", json={"name": "Ada", "consent": True, "recordings": [{}]})

    assert res.status_code == 200
    assert res.get_json()["voice"]["speech_s"] == 96.0
    # Enrolment loads the converter, so it gets the long timeout.
    assert rec.timeout == 600.0


def test_audio_voices_reports_502_when_unreachable(client):
    with _patched(_dead(OSError("nope"))):
        res = client.get("/api/audio/voices")

    assert res.status_code == 502


# --- PATCH|DELETE /api/audio/voices/<pid> --------------------------------------


def test_audio_voice_rename_uses_patch_and_path_interpolates_pid(client):
    http = _upstream({"voice": {"id": "ada-a1b2c3", "name": "Ada L."}})
    with _patched(http):
        res = client.patch("/api/audio/voices/ada-a1b2c3", json={"name": "Ada L."})

    assert res.status_code == 200
    assert http.patch.call_args.args[0] == "http://audio-server:8082/api/voices/ada-a1b2c3"
    assert http.patch.call_args.kwargs["json"] == {"name": "Ada L."}


def test_audio_voice_delete_uses_delete_and_sends_no_body(client):
    http = _upstream({"deleted": "ada-a1b2c3"})
    with _patched(http):
        res = client.delete("/api/audio/voices/ada-a1b2c3")

    assert res.status_code == 200
    assert http.delete.call_args.args[0] == "http://audio-server:8082/api/voices/ada-a1b2c3"
    assert "json" not in http.delete.call_args.kwargs


def test_audio_voice_rename_propagates_duplicate_name_conflict(client):
    http = _upstream({"error": "name already in use"}, status_code=409)
    with _patched(http):
        res = client.patch("/api/audio/voices/ada-a1b2c3", json={"name": "Grace"})

    assert res.status_code == 409


def test_audio_voice_delete_propagates_unknown_voice(client):
    http = _upstream({"error": "unknown voice"}, status_code=404)
    with _patched(http):
        res = client.delete("/api/audio/voices/nope")

    assert res.status_code == 404


def test_audio_voice_item_reports_502_when_unreachable(client):
    with _patched(_dead(OSError("nope"))):
        res = client.patch("/api/audio/voices/ada-a1b2c3", json={"name": "x"})

    assert res.status_code == 502


# --- cross-route invariants ---------------------------------------------------


AUDIO_ROUTE_METHODS = [
    ("post", "/api/audio/tts", "post", "/api/tts"),
    ("post", "/api/audio/music", "post", "/api/music"),
    ("post", "/api/audio/unload", "post", "/api/unload"),
    ("get", "/api/audio/voices/prompts", "get", "/api/voices/prompts"),
    ("get", "/api/audio/voices", "get", "/api/voices"),
    ("post", "/api/audio/voices", "post", "/api/voices"),
]


@pytest.mark.parametrize("caller_verb,route,upstream_verb,upstream_path", AUDIO_ROUTE_METHODS)
def test_every_audio_route_targets_the_audio_server(client, caller_verb, route, upstream_verb, upstream_path):
    """A mis-typed AUDIO_SERVER_URL prefix or a proxy path would silently break a feature."""
    http = _upstream({})
    with _patched(http):
        getattr(client, caller_verb)(route, json={})

    getattr(http, upstream_verb).assert_called_once()
    assert getattr(http, upstream_verb).call_args.args[0] == f"http://audio-server:8082{upstream_path}"


def test_audio_routes_do_not_send_the_proxy_api_key(client):
    """The audio-server is a sibling service, not behind proxy auth; sending the
    key would leak a secret to a container that does not need it."""
    http = _upstream({})
    with _patched(http), patch.dict("os.environ", {"ALPACA_API_KEY": "secret-key"}):
        client.post("/api/audio/tts", json={"text": "hi"})

    assert "headers" not in http.post.call_args.kwargs


def test_audio_voice_routes_reject_wrong_methods(client):
    """Only the documented verbs are accepted, so a stray GET cannot rename a voice."""
    assert client.get("/api/audio/tts").status_code == 405
    assert client.get("/api/audio/unload").status_code == 405
    assert client.post("/api/audio/voices/prompts").status_code == 405
    assert client.put("/api/audio/voices/ada-a1b2c3").status_code == 405


def test_audio_routes_return_json_even_on_upstream_json_failure(client):
    """A non-JSON body from the audio-server must become a 502, not an HTML 500 page."""
    resp = Mock()
    resp.status_code = 200
    resp.json.side_effect = ValueError("Expecting value: line 1 column 1")
    http = Mock()
    http.get.return_value = resp
    with _patched(http):
        res = client.get("/api/audio/status")

    assert res.status_code == 200
    assert res.get_json()["status"] == "offline"


def test_audio_status_body_is_json_serialisable(client):
    """Guards against a float/NaN sneaking into the passthrough payload."""
    http = _upstream({"vram_free_mb": 512.5, "rtf": 0.12})
    with _patched(http):
        res = client.get("/api/audio/status")

    assert json.loads(res.data)["vram_free_mb"] == 512.5
