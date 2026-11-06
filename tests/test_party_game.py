"""Tests for the 'Use Your Good Word' Biblical Worldview multiplayer party engine."""

from __future__ import annotations

import pytest

import arcade.app as arcade_app
from arcade import party


@pytest.fixture(autouse=True)
def clear_party_state():
    with party._lock:
        party._rooms.clear()
        party._tts_clips.clear()
    yield
    with party._lock:
        party._rooms.clear()
        party._tts_clips.clear()


@pytest.fixture
def client(tmp_path, monkeypatch):
    from arcade import family

    games = tmp_path / "games"
    monkeypatch.setattr(family, "GAMES_DIR", games)
    monkeypatch.setattr(family, "STATE_FILE", tmp_path / "family.json")
    monkeypatch.setattr(arcade_app, "GAMES_DIR", games)
    arcade_app.app.config["TESTING"] = True
    with arcade_app.app.test_client() as c:
        yield c


def test_create_and_join_room_lists_active_family_room():
    res1 = party.create_or_join_room(code="ZION", player_id="p_jer", name="Jeremiah", initials="JER")
    assert res1["room"]["code"] == "ZION"
    assert res1["room"]["phase"] == "lobby"

    res2 = party.create_or_join_room(code="ZION", player_id="p_mic", name="Michele", initials="MIC")
    assert len(res2["room"]["players"]) == 2

    rooms = party.list_active_rooms()
    assert len(rooms) == 1
    assert rooms[0]["code"] == "ZION"
    assert rooms[0]["player_count"] == 2


def test_seven_round_schedule_covers_all_five_modes():
    party.create_or_join_room(code="NOAH", player_id="p1", name="Jeremiah", initials="JER")
    party.create_or_join_room(code="NOAH", player_id="p2", name="Michele", initials="MIC")
    party.create_or_join_room(code="NOAH", player_id="p3", name="Kate", initials="KAT")

    seen_kinds = []
    state = party.start_match("NOAH")
    seen_kinds.append(state["prompt"]["kind"])

    for _ in range(6):
        # Advance writing -> voting -> reveal -> next round
        party.advance_phase("NOAH")
        party.advance_phase("NOAH")
        state = party.advance_phase("NOAH")
        seen_kinds.append(state["prompt"]["kind"])

    assert set(seen_kinds) == {
        "blank_o_matic",
        "extra_extra",
        "truth_or_bluff",
        "sub_the_scroll",
        "survey_says",
    }
    assert seen_kinds[-1] == "survey_says"


def test_scoring_original_answer_decoy_bait_and_house_trap():
    party.create_or_join_room(code="DOVE", player_id="p1", name="Jeremiah", initials="JER")
    party.create_or_join_room(code="DOVE", player_id="p2", name="Michele", initials="MIC")
    party.create_or_join_room(code="DOVE", player_id="p3", name="Kate", initials="KAT")
    party.start_match("DOVE")

    # p1 writes an original answer; p2 plays Scribe's Decoy; p3 writes an original answer
    party.submit_answer("DOVE", player_id="p1", text="My custom wise punchline")
    party.submit_answer("DOVE", player_id="p2", use_decoy=True)
    state = party.submit_answer("DOVE", player_id="p3", text="Another custom answer")

    # All 3 humans submitted -> auto-advances to voting!
    assert state["phase"] == "voting"

    # Find option IDs from internal room
    raw_room = party._rooms["DOVE"]
    p1_opt = next(o["option_id"] for o in raw_room["options"] if "p1" in o["authors"])
    decoy_opt = next(o["option_id"] for o in raw_room["options"] if o["kind"] == "decoy")

    # p2 votes for p1's original answer (+1000 to p1)
    # p3 votes for p1's original answer (+1000 to p1, plus +200 round winner bonus to p1 = 2200)
    # p1 votes for the Scribe's Decoy (-250 trap penalty to p1, +500 decoy bait to p2!)
    party.submit_vote("DOVE", player_id="p2", option_id=p1_opt)
    party.submit_vote("DOVE", player_id="p3", option_id=p1_opt)
    reveal_state = party.submit_vote("DOVE", player_id="p1", option_id=decoy_opt)

    assert reveal_state["phase"] == "reveal"
    scores = {p["id"]: p["score"] for p in reveal_state["players"]}
    assert scores["p1"] == 1000 + 1000 + 200 - 250  # 1950
    assert scores["p2"] == 500
    assert scores["p3"] == 0


def test_truth_or_bluff_blocks_real_truth_and_scores_truth_and_bluffs():
    party.create_or_join_room(code="LION", player_id="p1", name="Jeremiah", initials="JER")
    party.create_or_join_room(code="LION", player_id="p2", name="Michele", initials="MIC")
    party.create_or_join_room(code="LION", player_id="p3", name="Kate", initials="KAT")
    party.start_match("LION")

    # Jump to Round 3 (truth_or_bluff)
    with party._lock:
        party._start_round_unlocked(party._rooms["LION"], 3)
        truth_text = party._rooms["LION"]["current_prompt"]["truth"]

    # Submitting the actual truth is blocked so the player doesn't spoil the round
    with pytest.raises(ValueError, match="FAKE bluff"):
        party.submit_answer("LION", player_id="p1", text=truth_text)

    party.submit_answer("LION", player_id="p1", text="He lowered a golden magnet on a rope")
    party.submit_answer("LION", player_id="p2", text="He drained the brook with a trench")
    state = party.submit_answer("LION", player_id="p3", text="He called an eagle to dive for it")
    assert state["phase"] == "voting"

    raw_room = party._rooms["LION"]
    truth_opt = next(o["option_id"] for o in raw_room["options"] if o["kind"] == "truth")
    p1_bluff_opt = next(o["option_id"] for o in raw_room["options"] if "p1" in o["authors"])

    # p1 and p2 spot the real Scriptural truth (+1000 each); p3 is fooled by p1's bluff (+750 + 200 bonus to p1)
    party.submit_vote("LION", player_id="p1", option_id=truth_opt)
    party.submit_vote("LION", player_id="p2", option_id=truth_opt)
    reveal = party.submit_vote("LION", player_id="p3", option_id=p1_bluff_opt)

    assert reveal["phase"] == "reveal"
    scores = {p["id"]: p["score"] for p in reveal["players"]}
    assert scores["p1"] == 1000 + 750 + 200
    assert scores["p2"] == 1000
    assert scores["p3"] == 0


def test_flask_party_and_tv_routes(client):
    tv_resp = client.get("/tv/GRACE")
    assert tv_resp.status_code == 200
    assert b'var PINNED = "GRACE"' in tv_resp.data
    assert tv_resp.headers["Cache-Control"] == "no-store"

    create_resp = client.post("/api/party/rooms", json={"code": "GRACE", "name": "Jeremiah", "initials": "JER"})
    assert create_resp.status_code == 200
    payload = create_resp.get_json()
    assert payload["success"] is True
    assert payload["room"]["code"] == "GRACE"
    pid = payload["player_id"]

    start_resp = client.post("/api/party/rooms/GRACE/start", json={})
    assert start_resp.status_code == 200
    room = start_resp.get_json()["room"]
    assert room["phase"] == "writing"
    # AI companions Barnabas & Priscilla auto-join when only 1 human is in the room
    assert len(room["players"]) == 3

    sub_resp = client.post("/api/party/rooms/GRACE/submit", json={"player_id": pid, "text": "A joyful heart!"})
    assert sub_resp.status_code == 200
    assert sub_resp.get_json()["room"]["phase"] == "voting"


def test_alpaca_tts_decodes_audio_b64_and_supports_voice_clones(monkeypatch):
    import base64
    import json

    fake_wav = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 32
    captured_payloads = []

    class _FakeResp:
        def __init__(self, body: bytes):
            self._body = body
            self.status = 200

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def _fake_urlopen(req, timeout=8.0):
        body = json.loads(req.data.decode("utf-8"))
        captured_payloads.append(body)
        resp_json = json.dumps({"audio_b64": base64.b64encode(fake_wav).decode("ascii"), "mime": "audio/wav"}).encode(
            "utf-8"
        )
        return _FakeResp(resp_json)

    monkeypatch.setattr(party.urllib.request, "urlopen", _fake_urlopen)

    wav1 = party.synthesize_announcer_wav("Welcome to Use Your Good Word!", voice="am_fenrir,am_michael")
    assert wav1 == fake_wav
    assert captured_payloads[-1]["voice"] == "am_fenrir,am_michael"
    assert "clone" not in captured_payloads[-1]

    clip_id, wav2 = party.synthesize_announcer_clip("Jeremiah clone line", voice="clone:jeremiah-new-e572bb")
    assert wav2 == fake_wav
    assert party.get_cached_tts_wav(clip_id) == fake_wav
    assert captured_payloads[-1]["clone"] == "jeremiah-new-e572bb"
    # No base voice is named for a clone, so the audio server pairs the one that
    # needs no pitch correction (a forced default is what makes a clone sound sung).
    assert "voice" not in captured_payloads[-1]

    # The same line is served from the clip cache, not synthesized twice.
    calls_before = len(captured_payloads)
    assert party.synthesize_announcer_clip("Jeremiah clone line", voice="clone:jeremiah-new-e572bb")[0] == clip_id
    assert len(captured_payloads) == calls_before


class _JsonResp:
    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_party_voices_list_alpaca_kokoro_and_clones(client, monkeypatch):
    import json

    def _fake_urlopen(req, timeout=8.0):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if url.endswith("/api/voices"):
            return _JsonResp(json.dumps({"clones": [{"id": "jeremiah-new-e572bb", "name": "Jeremiah"}]}).encode())
        if url.endswith("/health"):
            return _JsonResp(json.dumps({"tts": {"voices": ["am_fenrir", "af_heart"]}}).encode())
        return _JsonResp(b"{}")

    monkeypatch.setattr(party.urllib.request, "urlopen", _fake_urlopen)
    vdata = client.get("/api/party/voices").get_json()
    assert vdata["success"] is True
    assert any(v["id"] == "clone:jeremiah-new-e572bb" for v in vdata["voices"])

    client.post("/api/party/rooms", json={"code": "CAST", "name": "Jeremiah", "initials": "JER"})
    voice_set = client.post("/api/party/rooms/CAST/voice", json={"voice": "clone:jeremiah-new-e572bb"})
    assert voice_set.get_json()["room"]["announcer_voice"] == "clone:jeremiah-new-e572bb"


def test_no_device_or_address_is_hardcoded_in_the_party_engine():
    from pathlib import Path

    src = Path("arcade/party.py").read_text(encoding="utf-8")
    assert "192.168." not in src
    assert "_chrome" not in src
    assert "SUPPORTED_TV_PLATFORMS" not in src


def test_room_snapshots_never_carry_tv_credentials():
    party.create_or_join_room(code="SAFE", player_id="p1", name="Ann")
    state = party.get_room_state("SAFE", viewer_id="p1")
    assert "cast_tv" not in state
    assert "token" not in str(state).lower()


def test_screens_are_discovered_by_the_execution_service(client, monkeypatch):
    import json

    seen = []

    def _fake_urlopen(req, timeout=8.0):
        seen.append((req.get_method(), req.full_url))
        return _JsonResp(json.dumps({
            "status": "SUCCESS",
            "home_assistant": True,
            "screens": [{"id": "ha:media_player.den", "name": "Den TV", "platform": "android_tv",
                         "state": "on", "display": "web", "methods": ["cast_web"], "method_labels": ["Google Cast"]}],
        }).encode())

    monkeypatch.setattr(party, "EXECUTION_SVC_URL", "http://exec:8003")
    monkeypatch.setattr(party.urllib.request, "urlopen", _fake_urlopen)
    data = client.get("/api/party/tvs?refresh=1").get_json()
    assert data["success"] is True and data["tvs"][0]["name"] == "Den TV"
    assert seen == [("GET", "http://exec:8003/execute/stage/screens?refresh=true")]


def test_execution_relay_sends_the_internal_secret_when_configured(monkeypatch):
    import json

    headers = {}

    def _fake_urlopen(req, timeout=8.0):
        headers.update({k.lower(): v for k, v in req.header_items()})
        return _JsonResp(json.dumps({"screens": []}).encode())

    monkeypatch.setattr(party, "EXECUTION_SVC_URL", "http://exec:8003")
    monkeypatch.setenv("JARVIS_INTERNAL_SECRET", "s3cret")
    monkeypatch.setattr(party.urllib.request, "urlopen", _fake_urlopen)
    party.discover_screens()
    assert headers["x-internal-secret"] == "s3cret"


def test_screen_discovery_reports_when_unconfigured(client, monkeypatch):
    monkeypatch.setattr(party, "EXECUTION_SVC_URL", "")
    data = client.get("/api/party/tvs").get_json()
    assert data["success"] is False and data["tvs"] == []
    assert "JARVIS_GATEWAY_URL" in data["error"]


def test_cast_sends_only_the_arcades_own_stage_page(client, monkeypatch):
    import json

    sent = []

    def _fake_urlopen(req, timeout=8.0):
        body = json.loads(req.data.decode())
        sent.append((req.full_url, body))
        return _JsonResp(json.dumps({"success": True, "method": "cast_web", "display": "web", "verified": True}).encode())

    monkeypatch.setattr(party, "EXECUTION_SVC_URL", "http://exec:8003")
    monkeypatch.setattr(party.urllib.request, "urlopen", _fake_urlopen)
    resp = client.post(
        "/api/party/cast",
        json={"screen_id": "ha:media_player.den", "room_code": "zion!", "url": "http://evil.example/x"},
        headers={"X-Forwarded-Proto": "https", "X-Forwarded-Host": "games.example.org"},
    )
    assert resp.status_code == 200 and resp.get_json()["success"] is True
    url, body = sent[0]
    assert url == "http://exec:8003/execute/stage/show"
    assert body["url"] == "https://games.example.org/tv/ZION"
    assert body["screen_id"] == "ha:media_player.den"

    client.post("/api/party/cast", json={"screen_id": "ha:media_player.den", "room_code": "ZION", "stop": True})
    assert sent[-1][0] == "http://exec:8003/execute/stage/stop"


def test_cast_without_a_screen_is_a_bad_request(client):
    resp = client.post("/api/party/cast", json={"room_code": "ZION"})
    assert resp.status_code == 400


def test_cast_reports_execution_errors_instead_of_claiming_success(client, monkeypatch):
    def _boom(req, timeout=8.0):
        raise OSError("connection refused")

    monkeypatch.setattr(party, "EXECUTION_SVC_URL", "http://exec:8003")
    monkeypatch.setattr(party.urllib.request, "urlopen", _boom)
    data = client.post("/api/party/cast", json={"screen_id": "ha:x", "room_code": "ZION"}).get_json()
    assert data["success"] is False and "unreachable" in data["error"]


def test_join_link_and_qr_code(client):
    client.post("/api/party/rooms", json={"code": "ZION", "name": "Ann"})
    resp = client.get("/join/zion")
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/game/family-good-word/index.html#room=ZION")
    qr = client.get("/api/party/rooms/ZION/join.svg")
    if qr.status_code == 404:
        pytest.skip("segno is not installed")
    assert qr.status_code == 200 and qr.mimetype == "image/svg+xml"
    assert b"<svg" in qr.data


def test_tv_stage_polling_marks_the_room_so_phones_stay_quiet(client):
    client.post("/api/party/rooms", json={"code": "LIVE", "player_id": "p1", "name": "Ann"})
    assert client.get("/api/party/rooms/LIVE?player_id=p1").get_json()["room"]["stage_connected"] is False
    client.get("/api/party/rooms/LIVE?stage=1")
    assert client.get("/api/party/rooms/LIVE?player_id=p1").get_json()["room"]["stage_connected"] is True


def test_tv_stage_page_is_hands_free_and_never_shows_draft_answers(client):
    html = client.get("/tv").get_data(as_text=True)
    assert 'var PINNED = ""' in html
    # Polls as the stage (marks presence) and follows the active room on its own.
    assert "?stage=1" in html and "/api/party/rooms" in html
    # Uses Alpaca TTS on the TV's speakers, never the browser's robotic voices.
    assert "/api/party/tts" in html and "speechSynthesis" not in html
    # It reads no viewer-specific state, so a player's draft can never reach it.
    assert "player_id" not in html and "my_submission" not in html


def test_good_word_game_never_uses_browser_speech_synthesis():
    from pathlib import Path

    html = Path("arcade/family_games/good-word/game.html").read_text(encoding="utf-8")
    assert "speechSynthesis" not in html
    assert "SpeechSynthesisUtterance" not in html
    assert "privacyShield" in html
    assert "hostPlaysOnTv" in html
    assert "Host Scroll Sealed & Locked In" in html


def test_host_device_can_play_without_projecting_answers_to_tv_stage():
    # Host device joins and creates room; TV stage joins as spectator
    party.create_or_join_room(code="HOST", player_id="p_host", name="Jeremiah", initials="JER", spectator=True)
    party.create_or_join_room(code="HOST", player_id="p_mic", name="Michele", initials="MIC")
    party.start_match("HOST")

    # Even if the Host Device initially opened TV stage mode, submitting an answer promotes Host to active player!
    host_state = party.submit_answer("HOST", player_id="p_host", text="My secret host punchline")
    host_player = next(p for p in host_state["players"] if p["id"] == "p_host")
    assert host_player["spectator"] is False
    assert host_player["submitted"] is True

    # What the TV Stage (no viewer_id) sees during writing: NEVER the Host's answer text!
    tv_stage_state = party.get_room_state("HOST", viewer_id=None)
    assert tv_stage_state["phase"] == "writing"
    assert tv_stage_state["my_submission"] is None
    assert "My secret host punchline" not in str(tv_stage_state)




def test_standalone_arcade_relays_through_the_jarvis_gateway(client, monkeypatch):
    import json

    seen = []

    def _fake_urlopen(req, timeout=8.0):
        body = json.loads(req.data.decode()) if req.data else None
        seen.append((req.get_method(), req.full_url, body, dict(req.header_items())))
        if req.full_url.endswith("/api/arcade/cast"):
            return _JsonResp(json.dumps({"success": True, "method": "cast_web", "display": "web"}).encode())
        return _JsonResp(json.dumps({"success": True, "tvs": [{"id": "ha:media_player.den", "name": "Den"}]}).encode())

    monkeypatch.setattr(party, "JARVIS_GATEWAY_URL", "http://jarvis:11435")
    monkeypatch.setattr(party, "EXECUTION_SVC_URL", "http://exec:8003")
    monkeypatch.setenv("JARVIS_INTERNAL_SECRET", "s3cret")
    monkeypatch.setattr(party.urllib.request, "urlopen", _fake_urlopen)

    tvs = client.get("/api/party/tvs").get_json()
    assert tvs["success"] is True and tvs["tvs"][0]["name"] == "Den"
    assert seen[0][1] == "http://jarvis:11435/api/arcade/tvs?refresh=false"
    assert seen[0][3]["X-internal-secret"] == "s3cret"

    out = client.post("/api/party/cast", json={"screen_id": "ha:media_player.den", "room_code": "zion"}).get_json()
    assert out["success"] is True
    assert seen[1][1] == "http://jarvis:11435/api/arcade/cast"
    assert seen[1][2] == {"screen_id": "ha:media_player.den", "room_code": "ZION", "method": "", "stop": False}


def test_a_refused_relay_explains_what_to_configure(client, monkeypatch):
    import io
    import urllib.error

    def _refuse(req, timeout=8.0):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(b"{}"))

    monkeypatch.setattr(party, "JARVIS_GATEWAY_URL", "http://jarvis:11435")
    monkeypatch.setattr(party.urllib.request, "urlopen", _refuse)
    data = client.get("/api/party/tvs").get_json()
    assert data["success"] is False and "JARVIS_INTERNAL_SECRET" in data["error"]
