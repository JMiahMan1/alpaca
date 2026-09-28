"""Tests for the four /api/podcast/* routes (the Podcast Studio backend).

The mixer itself is covered by tests/test_podcast_mixer.py with no Flask and
no network. What is left to pin here is the wiring: which service each route
talks to, with which verb and timeout, what it does with a failure, and -- the
part that actually decides whether the button in the panel works -- that a
drafted script survives the round trip into a render.
"""

from __future__ import annotations

import base64
import io
import json
import wave
from contextlib import contextmanager
from unittest.mock import Mock, patch

import numpy as np
import pytest

from web import podcast_mixer as pm
from web.app import AUDIO_SERVER_URL, PROXY_URL, app

SR = pm.TARGET_SR


# --------------------------------------------------------------------------
# the httpx mock helpers, carried forward from the audio/SD/sandbox route tests
# --------------------------------------------------------------------------


class _Rec:
    """What the route handed to httpx.Client: the timeout it asked for."""

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
    """A client whose every verb answers with `payload`."""
    resp = Mock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.text = json.dumps(payload)
    http = Mock()
    for verb in ("get", "post", "patch", "delete", "put"):
        getattr(http, verb).return_value = resp
    return http


def _resp_only(payload, status_code=200):
    """A *response*, not a client -- for use as a side_effect return value."""
    resp = Mock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.text = json.dumps(payload)
    return resp


def _dead(side_effect):
    http = Mock()
    for verb in ("get", "post", "patch", "delete", "put"):
        getattr(http, verb).side_effect = side_effect
    return http


@contextmanager
def _patched(http):
    rec = _Rec()

    def _ctor(*args, **kwargs):
        rec.timeout = kwargs.get("timeout", args[0] if args else None)
        return _Ctx(http)

    with patch("web.app.httpx.Client", side_effect=_ctor):
        yield rec


@contextmanager
def _no_proxy():
    http = _dead(RuntimeError("connection refused"))
    with _patched(http), patch("web.app._find_proxy_url", return_value=None):
        yield http


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def _wav_b64(seconds: float = 0.4, hz: float = 220.0, amp: float = 0.4) -> str:
    t = np.arange(int(seconds * SR), dtype=np.float32) / SR
    samples = (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)
    return base64.b64encode(pm.encode_wav(samples, SR)).decode("ascii")


def _music_b64(seconds: float = 1.0, sr: int = 32000) -> str:
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    samples = (0.4 * np.sin(2 * np.pi * 330.0 * t)).astype(np.float32)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.round(samples * 32767).astype("<i2").tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------
# /api/podcast/status
# --------------------------------------------------------------------------


def test_status_answers_200_even_when_the_audio_server_is_down(client):
    """The panel has to be able to render this page and learn that audio is
    offline; a 500 would be a blank panel with no explanation."""
    with _no_proxy():
        resp = client.get("/api/podcast/status")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["audio"]["online"] is False
    assert body["audio"]["voices"] == []
    assert body["proxy"]["online"] is False


def test_status_reports_the_audio_server_and_its_voices(client):
    http = _upstream({"tts": {"voices": ["af_nicole", "am_michael"]}, "vram_free_mb": 5120})
    with _patched(http), patch("web.app._find_proxy_url", return_value=PROXY_URL):
        body = client.get("/api/podcast/status").get_json()
    assert http.get.call_args.args[0] == f"{AUDIO_SERVER_URL}/health"
    assert body["audio"]["online"] is True
    assert body["audio"]["voices"] == ["af_nicole", "am_michael"]
    assert body["audio"]["vram_free_mb"] == 5120
    assert body["proxy"] == {"online": True, "url": PROXY_URL}


def test_status_notes_a_non_200_from_the_audio_server(client):
    with _patched(_upstream({}, status_code=503)), patch("web.app._find_proxy_url", return_value=None):
        body = client.get("/api/podcast/status").get_json()
    assert body["audio"]["online"] is False
    assert "503" in body["audio"]["error"]


def test_status_offers_the_presets_pairs_and_rates_the_ui_needs(client):
    with _no_proxy():
        body = client.get("/api/podcast/status").get_json()
    assert {p["id"] for p in body["bed_presets"]} == set(pm.BED_PRESETS)
    for preset in body["bed_presets"]:
        assert preset["label"] and preset["prompt"] and preset["key"]
    assert {p["id"] for p in body["host_pairs"]} == {p["id"] for p in pm.HOST_PAIRS}
    assert body["target_sample_rate"] == SR
    assert body["max_tts_chars"] == 4000


def test_status_uses_a_short_timeout_because_it_gates_the_ui(client):
    with _patched(_upstream({})) as rec, patch("web.app._find_proxy_url", return_value=None):
        client.get("/api/podcast/status")
    assert rec.timeout == 2.0


def test_status_rejects_post(client):
    with _no_proxy():
        assert client.post("/api/podcast/status", json={}).status_code == 405


# --------------------------------------------------------------------------
# /api/podcast/voices
# --------------------------------------------------------------------------


def test_voices_lists_the_saved_profiles_and_the_roster(client):
    http = _upstream({"voices": [{"id": "ada-abc123", "name": "Ada"}, {"id": "row-def456", "name": "Rowan"}]})
    with _patched(http):
        body = client.get("/api/podcast/voices").get_json()
    assert http.get.call_args.args[0] == f"{AUDIO_SERVER_URL}/api/voices"
    assert [p["id"] for p in body["saved_profiles"]] == ["ada-abc123", "row-def456"]
    assert len(body["roster"]) == 2 * len(pm.HOST_PAIRS)
    assert {r["slot"] for r in body["roster"]} == {"a", "b"}


def test_the_roster_tells_the_ui_which_source_voice_is_which_gender(client):
    """This is what lets the panel refuse a cross-gender clone before the user
    waits through a render to find out."""
    with _patched(_upstream({"voices": []})):
        body = client.get("/api/podcast/voices").get_json()
    for row in body["roster"]:
        assert row["source_gender"] == pm.voice_gender(row["voice"])
        assert row["source_gender"] in {"f", "m"}


def test_voices_accepts_a_bare_list_from_the_audio_server(client):
    with _patched(_upstream([{"id": "v1", "name": "X"}])):
        body = client.get("/api/podcast/voices").get_json()
    assert body["saved_profiles"] == [{"id": "v1", "name": "X"}]


def test_voices_survives_an_audio_server_that_is_not_listing(client):
    with _patched(_upstream({}, status_code=500)):
        body = client.get("/api/podcast/voices").get_json()
    assert body["saved_profiles"] == []


def test_voices_reports_an_unreachable_audio_server(client):
    with _patched(_dead(RuntimeError("connection refused"))):
        resp = client.get("/api/podcast/voices")
    assert resp.status_code == 502
    assert "connection refused" in resp.get_json()["error"]


def test_voices_rejects_post(client):
    with _patched(_upstream({})):
        assert client.post("/api/podcast/voices", json={}).status_code == 405


# --------------------------------------------------------------------------
# /api/podcast/draft
# --------------------------------------------------------------------------


SCRIPT = """[host_a] Welcome back to the show.
[host_b] Glad to be here.
# Episode 1
[host_a] Today we are talking about caching.
[host_b] And why it keeps going wrong.
"""


def _chat_http(content: str):
    return _upstream({"message": {"content": content}})


def test_draft_rejects_a_missing_topic(client):
    with _no_proxy():
        assert client.post("/api/podcast/draft", json={}).status_code == 400
        assert client.post("/api/podcast/draft", json={"topic": "   "}).status_code == 400


def test_draft_rejects_an_absurdly_long_topic(client):
    with _no_proxy():
        resp = client.post("/api/podcast/draft", json={"topic": "x" * 2001})
    assert resp.status_code == 400
    assert "2000" in resp.get_json()["error"]


def test_draft_says_so_when_no_proxy_is_reachable(client):
    """Without a proxy there is no model to ask, and a 500 would look like a
    bug rather than a missing service."""
    with _no_proxy():
        resp = client.post("/api/podcast/draft", json={"topic": "caching"})
    assert resp.status_code == 503
    assert "not reachable" in resp.get_json()["error"]


def test_draft_asks_the_proxy_for_a_script(client):
    http = _chat_http(SCRIPT)
    with _patched(http), patch("web.app._find_proxy_url", return_value=PROXY_URL):
        resp = client.post("/api/podcast/draft", json={"topic": "caching"})
    assert resp.status_code == 200
    assert http.post.call_args.args[0] == f"{PROXY_URL}/api/chat"


def test_the_draft_prompt_disables_thinking_and_bounds_the_length(client):
    """A script is prose, not a deliverable document, so think costs a second
    of latency for nothing; and num_predict has to cover the word count."""
    http = _chat_http(SCRIPT)
    with _patched(http), patch("web.app._find_proxy_url", return_value=PROXY_URL):
        client.post("/api/podcast/draft", json={"topic": "caching", "target_words": 600})
    payload = http.post.call_args.kwargs["json"]
    assert payload["think"] is False
    assert payload["stream"] is False
    assert payload["options"]["num_predict"] == min(4000, 600 * 3)
    assert payload["options"]["temperature"] == 0.85


def test_the_num_predict_ceiling_is_enforced_for_a_long_target(client):
    http = _chat_http(SCRIPT)
    with _patched(http), patch("web.app._find_proxy_url", return_value=PROXY_URL):
        client.post("/api/podcast/draft", json={"topic": "caching", "target_words": 9999})
    assert http.post.call_args.kwargs["json"]["options"]["num_predict"] == 4000


def test_the_system_prompt_states_the_tag_format_and_the_word_target(client):
    http = _chat_http(SCRIPT)
    with _patched(http), patch("web.app._find_proxy_url", return_value=PROXY_URL):
        client.post("/api/podcast/draft", json={"topic": "caching", "target_words": 400})
    system = http.post.call_args.kwargs["json"]["messages"][0]["content"]
    assert "[host_a]" in system and "[host_b]" in system
    assert "400 words" in system
    assert "Alternate" in system
    assert "fences" in system


def test_notes_reach_the_model(client):
    http = _chat_http(SCRIPT)
    with _patched(http), patch("web.app._find_proxy_url", return_value=PROXY_URL):
        client.post("/api/podcast/draft", json={"topic": "caching", "notes": "mention Redis"})
    system = http.post.call_args.kwargs["json"]["messages"][0]["content"]
    assert "mention Redis" in system
    assert http.post.call_args.kwargs["json"]["messages"][1]["content"].startswith("Topic: caching")


def test_the_topic_reaches_the_model(client):
    http = _chat_http(SCRIPT)
    with _patched(http), patch("web.app._find_proxy_url", return_value=PROXY_URL):
        client.post("/api/podcast/draft", json={"topic": "why caching is hard"})
    assert "Topic: why caching is hard" in http.post.call_args.kwargs["json"]["messages"][1]["content"]


def test_draft_parses_the_script_into_turns_and_separates_headings(client):
    with _patched(_chat_http(SCRIPT)), patch("web.app._find_proxy_url", return_value=PROXY_URL):
        body = client.post("/api/podcast/draft", json={"topic": "caching"}).get_json()
    assert body["script"] == SCRIPT
    assert body["headings"] == ["# Episode 1"]
    assert body["turn_count"] == 4  # the heading is not a spoken turn
    assert [t["host_index"] for t in body["turns"]] == [0, 1, 0, 1]
    assert body["unattributed"] == 0
    assert body["word_count"] == len(SCRIPT.split())


def test_draft_returns_the_roster_it_parsed_against(client):
    with _patched(_chat_http(SCRIPT)), patch("web.app._find_proxy_url", return_value=PROXY_URL):
        body = client.post("/api/podcast/draft", json={"topic": "caching", "pair_id": "duo_bright"}).get_json()
    assert [h["voice"] for h in body["hosts"]] == ["af_sky", "am_fenrir"]


def test_draft_reports_a_proxy_error_with_the_upstream_detail(client):
    with _patched(_upstream({"error": "model not found"}, status_code=404)), patch(
        "web.app._find_proxy_url", return_value=PROXY_URL
    ):
        resp = client.post("/api/podcast/draft", json={"topic": "caching"})
    assert resp.status_code == 502
    assert "404" in resp.get_json()["error"]


def test_draft_reports_a_non_json_proxy_body(client):
    """An HTML error page from a reverse proxy must not become a JSONDecodeError
    traceback in the panel."""
    http = _upstream({})
    http.post.return_value.json.side_effect = ValueError("not json")
    http.post.return_value.text = "<html>502 Bad Gateway</html>"
    with _patched(http), patch("web.app._find_proxy_url", return_value=PROXY_URL):
        resp = client.post("/api/podcast/draft", json={"topic": "caching"})
    assert resp.status_code == 502
    assert "non-JSON" in resp.get_json()["error"]


def test_draft_reports_an_unreachable_proxy(client):
    http = _dead(RuntimeError("connection refused"))
    with _patched(http), patch("web.app._find_proxy_url", return_value=PROXY_URL):
        resp = client.post("/api/podcast/draft", json={"topic": "caching"})
    assert resp.status_code == 502
    assert "connection refused" in resp.get_json()["error"]


def test_draft_uses_a_generous_timeout_because_a_script_is_a_real_generation(client):
    http = _chat_http(SCRIPT)
    with _patched(http) as rec, patch("web.app._find_proxy_url", return_value=PROXY_URL):
        client.post("/api/podcast/draft", json={"topic": "caching"})
    assert rec.timeout == 600.0


def test_draft_survives_a_bodyless_post(client):
    """The route uses get_json(silent=True), so a panel bug that sends no body
    gets a clean 400 rather than a 415."""
    with _no_proxy():
        assert client.post("/api/podcast/draft").status_code == 400


def test_draft_rejects_get(client):
    with _no_proxy():
        assert client.get("/api/podcast/draft").status_code == 405


# --------------------------------------------------------------------------
# /api/podcast/render
# --------------------------------------------------------------------------


def _tts_http(wav_b64: str | None = None):
    return _upstream({"audio_b64": wav_b64 if wav_b64 is not None else _wav_b64()})


def test_render_needs_a_script_or_some_segments(client):
    with _no_proxy():
        assert client.post("/api/podcast/render", json={}).status_code == 400
        assert client.post("/api/podcast/render", json={"script": "   "}).status_code == 400


def test_render_rejects_a_script_with_no_spoken_turns(client):
    with _no_proxy():
        resp = client.post("/api/podcast/render", json={"script": "# Episode 1\n## Notes\n"})
    assert resp.status_code == 400
    assert "no spoken turns" in resp.get_json()["error"]


def test_render_synthesises_one_request_per_turn_and_mixes_the_result(client):
    http = _tts_http()
    with _patched(http):
        resp = client.post(
            "/api/podcast/render", json={"script": "[host_a] Hello there.\n[host_b] Hi to you too.\n"}
        )
    assert resp.status_code == 200
    assert http.post.call_count == 2
    assert all(call.args[0] == f"{AUDIO_SERVER_URL}/api/tts" for call in http.post.call_args_list)
    body = resp.get_json()
    assert body["ok"] is True
    assert body["turn_count"] == 2
    assert body["wav_b64"] and "data_uri" not in body
    assert body["duration_s"] > 0


def test_the_returned_wav_is_a_playable_file_at_the_mix_rate(client):
    with _patched(_tts_http()):
        body = client.post("/api/podcast/render", json={"script": "[host_a] Hello."}).get_json()
    with wave.open(io.BytesIO(base64.b64decode(body["wav_b64"])), "rb") as w:
        assert w.getframerate() == SR
        assert w.getsampwidth() == 2
        assert w.getnchannels() == 1
        assert w.getnframes() > 0


def test_each_turn_is_sent_with_its_own_hosts_voice(client):
    http = _tts_http()
    with _patched(http):
        client.post(
            "/api/podcast/render", json={"script": "[host_a] One.\n[host_b] Two.\n", "pair_id": "duo_deep"}
        )
    voices = [c.kwargs["json"]["voice"] for c in http.post.call_args_list]
    assert voices == ["af_heart", "am_adam"]


def test_a_clone_is_only_sent_for_the_slot_that_asked_for_one(client):
    http = _tts_http()
    with _patched(http):
        client.post(
            "/api/podcast/render",
            json={
                "script": "[host_a] One.\n[host_b] Two.\n",
                "voice_profiles": {"host_clone_duo_warm_b": "rowan-x1"},
            },
        )
    first, second = (c.kwargs["json"] for c in http.post.call_args_list)
    assert "clone" not in first
    assert second["clone"] == "rowan-x1"
    assert second["clone_tau"] == 0.3


def test_per_turn_speed_varies_but_is_deterministic(client):
    """A host who reads every line at exactly the same pace is the giveaway
    that nobody was in the room; but a re-render must be the same episode."""
    script = "\n".join(f"[host_a] Line {i}." for i in range(4))
    speeds = []
    for _ in range(2):
        http = _tts_http()
        with _patched(http):
            client.post("/api/podcast/render", json={"script": script})
        speeds.append([c.kwargs["json"]["speed"] for c in http.post.call_args_list])
    assert speeds[0] == speeds[1]
    assert len(set(speeds[0])) > 1


def test_a_rejected_turn_names_the_turn_and_how_far_it_got(client):
    http = _upstream({"error": "text exceeds 4000 chars"}, status_code=400)
    with _patched(http):
        resp = client.post(
            "/api/podcast/render", json={"script": "[host_a] One.\n[host_b] Two.\n[host_a] Three.\n"}
        )
    assert resp.status_code == 502
    body = resp.get_json()
    assert "turn 1" in body["error"]
    assert "4000" in body["error"]
    assert body["turns_done"] == 0
    assert body["turns_total"] == 3


def test_a_turn_that_produces_no_audio_is_an_error_not_a_silent_hole(client):
    with _patched(_upstream({})):
        resp = client.post("/api/podcast/render", json={"script": "[host_a] One."})
    assert resp.status_code == 502
    assert "produced no audio" in resp.get_json()["error"]


def test_an_unreachable_audio_server_reports_how_far_it_got(client):
    with _patched(_dead(RuntimeError("connection refused"))):
        resp = client.post("/api/podcast/render", json={"script": "[host_a] One."})
    assert resp.status_code == 502
    assert "unreachable" in resp.get_json()["error"]


def test_too_many_turns_is_refused_before_anything_is_synthesised(client):
    """200 turns is already tens of minutes of speech; starting the audio and
    then refusing would be the worst order."""
    http = _tts_http()
    with _patched(http):
        resp = client.post("/api/podcast/render", json={"segments": [{"text": f"t{i}"} for i in range(201)]})
    assert resp.status_code == 400
    assert "200" in resp.get_json()["error"]
    assert http.post.call_count == 0


def test_a_pre_rendered_segments_list_skips_the_tts_service_entirely(client):
    """A caller that already has the audio sends base64, because JSON has no
    bytes and JSON is the only way a browser can send this. The route decodes
    it, so the mixer always receives raw WAV."""
    seg = {"wav": _wav_b64(), "host_index": 0, "text": "already rendered"}
    http = _upstream({})
    with _patched(http):
        body = client.post("/api/podcast/render", json={"segments": [seg]}).get_json()
    assert http.post.call_count == 0
    assert body["turn_count"] == 1
    assert body["wav_b64"]


def test_the_default_bed_is_synthesized_under_the_speech(client):
    with _patched(_tts_http()):
        body = client.post("/api/podcast/render", json={"script": "[host_a] One."}).get_json()
    assert body["bed_preset"] == "ambient_warm"
    assert body["bed_duck_db"] == pm.DEFAULT_DUCK_DB
    assert body["bed_duration_s"] > 0


def test_a_bed_preset_comes_from_the_roster_and_an_unknown_one_falls_back(client):
    with _patched(_tts_http()):
        body = client.post(
            "/api/podcast/render", json={"script": "[host_a] One.", "bed_preset": "not-a-preset"}
        ).get_json()
    assert body["bed_preset"] == "not-a-preset"  # reported as asked, rendered as ambient_warm
    assert body["bed_duration_s"] > 0


def test_the_duck_depth_is_caller_controllable(client):
    with _patched(_tts_http()):
        body = client.post("/api/podcast/render", json={"script": "[host_a] One.", "duck_db": 6.0}).get_json()
    assert body["bed_duck_db"] == 6.0


def test_the_data_uri_is_returned_instead_of_the_raw_wav_on_request(client):
    with _patched(_tts_http()):
        body = client.post(
            "/api/podcast/render", json={"script": "[host_a] One.", "return_data_uri": True}
        ).get_json()
    assert body["data_uri"].startswith("data:audio/wav;base64,")
    assert "wav_b64" not in body


def test_a_sting_prompt_routes_to_musicgen_and_warns_when_it_is_truncated(client):
    """The 1500-token clamp is silent: a 180 s request comes back as 30 s with
    no error, and the only visible sign is the two meta fields disagreeing."""
    http = _upstream(
        {
            "audio_b64": _music_b64(),
            "meta": {"requested_duration_s": 120.0, "duration_s": 30.0},
        }
    )
    with _patched(http):
        body = client.post(
            "/api/podcast/render", json={"script": "[host_a] One.", "sting_prompt": "a warm theme"}
        ).get_json()
    assert [c.args[0] for c in http.post.call_args_list][-1] == f"{AUDIO_SERVER_URL}/api/music"
    assert body["bed_preset"].startswith("sting:")
    assert any("30 s training window" in w for w in body["warnings"])


def test_a_sting_that_came_back_at_the_requested_length_produces_no_warning(client):
    http = _upstream(
        {"audio_b64": _music_b64(), "meta": {"requested_duration_s": 10.0, "duration_s": 10.0}}
    )
    with _patched(http):
        body = client.post("/api/podcast/render", json={"script": "[host_a] One.", "sting_prompt": "a theme"}).get_json()
    assert body["warnings"] == []


def test_an_unavailable_musicgen_degrades_to_a_synthesized_bed(client):
    """Both services hang off the same audio-server, so the mock has to fail
    /api/music and nothing else."""
    http = _tts_http()
    real_post = http.post.return_value

    def post(url, **kwargs):
        if url.endswith("/api/music"):
            return _resp_only({"error": "no music model"}, status_code=503)
        return real_post

    http.post.side_effect = post
    with _patched(http):
        body = client.post(
            "/api/podcast/render", json={"script": "[host_a] One.", "sting_prompt": "a warm theme"}
        ).get_json()
    assert body.get("ok") is True and bool(body.get("wav_b64"))
    assert any("sting was skipped" in w for w in body["warnings"])
    # and it really is a synthesized bed, not a 30 s MusicGen loop
    assert not body["bed_preset"].startswith("sting:")


def test_the_synthesized_bed_is_built_at_the_mix_rate_not_musicgen_rate(client):
    """A sting comes back at 32 kHz; if it were placed into a 24 kHz buffer
    without conversion it would play an octave and a third fast."""
    with _patched(_upstream({"audio_b64": _music_b64(), "meta": {}})):
        body = client.post("/api/podcast/render", json={"script": "[host_a] One.", "sting_prompt": "a theme"}).get_json()
    with wave.open(io.BytesIO(base64.b64decode(body["wav_b64"])), "rb") as w:
        assert w.getframerate() == SR


def test_tts_gets_a_generous_timeout_because_a_turn_is_real_speech(client):
    http = _tts_http()
    with _patched(http) as rec:
        client.post("/api/podcast/render", json={"script": "[host_a] One."})
    assert rec.timeout == 600.0


def test_a_segment_whose_wav_is_not_base64_is_rejected_at_the_edge(client):
    """Otherwise the string reaches decode_wav and the failure surfaces as a
    500 from inside the mixer, with no mention of which segment was bad."""
    with _no_proxy():
        resp = client.post(
            "/api/podcast/render", json={"segments": [{"wav": "not base64!!", "host_index": 0}]}
        )
    assert resp.status_code == 400
    assert "segment 0" in resp.get_json()["error"]


def test_a_non_dict_segment_is_skipped_rather_than_crashing_the_mix(client):
    with _patched(_upstream({})):
        resp = client.post("/api/podcast/render", json={"segments": ["junk", {"wav": _wav_b64()}]})
    assert resp.status_code == 200
    assert resp.get_json()["turn_count"] == 1


def test_render_survives_a_bodyless_post(client):
    with _no_proxy():
        assert client.post("/api/podcast/render").status_code == 400


def test_render_rejects_get(client):
    with _no_proxy():
        assert client.get("/api/podcast/render").status_code == 405


# --------------------------------------------------------------------------
# the end-to-end property the panel actually depends on
# --------------------------------------------------------------------------


def test_a_drafted_script_renders_without_the_panel_touching_the_mixer(client):
    """Draft -> parse -> synthesise -> mix, with the mixer used only through
    its public API. This is the path the Podcast Studio button takes, and a
    regression in any one stage shows up here rather than in a browser."""
    with _patched(_chat_http(SCRIPT)), patch("web.app._find_proxy_url", return_value=PROXY_URL):
        draft = client.post("/api/podcast/draft", json={"topic": "caching"}).get_json()

    assert draft["turn_count"] == 4
    with _patched(_tts_http()):
        rendered = client.post("/api/podcast/render", json={"script": draft["script"]}).get_json()

    assert rendered["turn_count"] == 4
    assert rendered["duration_s"] > 0
    with wave.open(io.BytesIO(base64.b64decode(rendered["wav_b64"])), "rb") as w:
        assert w.getframerate() == SR
        assert w.getnframes() > 0


def test_a_script_written_by_hand_with_the_other_tag_spelling_also_renders(client):
    """[host_a] and Ada: are both what models emit; neither may silently lose
    a turn to a default voice."""
    script = "ADA: First thing.\nrowan: Second thing.\n"
    http = _tts_http()
    with _patched(http):
        body = client.post("/api/podcast/render", json={"script": script}).get_json()
    assert body["turn_count"] == 2
    assert [c.kwargs["json"]["voice"] for c in http.post.call_args_list] == ["af_nicole", "am_michael"]


def test_prose_with_a_colon_does_not_become_a_third_voice(client):
    http = _tts_http()
    with _patched(http):
        body = client.post(
            "/api/podcast/render",
            json={"script": "[host_a] Welcome back.\nAnyway: on to the news.\n[host_b] What news?\n"},
        ).get_json()
    assert body["turn_count"] == 2
    assert [c.kwargs["json"]["voice"] for c in http.post.call_args_list] == ["af_nicole", "am_michael"]


def test_a_heading_in_a_drafted_script_is_not_read_aloud(client):
    http = _tts_http()
    with _patched(http):
        body = client.post("/api/podcast/render", json={"script": SCRIPT}).get_json()
    assert body["turn_count"] == 4
    assert "Episode 1" not in " ".join(c.kwargs["json"]["text"] for c in http.post.call_args_list)


def test_a_very_long_turn_is_split_into_several_tts_requests(client):
    """The audio-server caps a request at 4000 characters of input text, so an
    unsplit long turn would be a silent 400."""
    http = _tts_http()
    long_turn = " ".join(f"That is sentence number {i} of a very long monologue." for i in range(300))
    with _patched(http):
        body = client.post("/api/podcast/render", json={"script": f"[host_a] {long_turn}"}).get_json()
    assert http.post.call_count > 1
    assert all(len(c.kwargs["json"]["text"]) < 4000 for c in http.post.call_args_list)
    assert body["turn_count"] == http.post.call_count
