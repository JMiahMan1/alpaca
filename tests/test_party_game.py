"""Tests for the 'Parable Party' Biblical Worldview multiplayer party engine."""

from __future__ import annotations

import contextlib
import time

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


def _by_real_id(code, players):
    """Snapshots carry public ids; map them back to the ids the test created."""
    room = party._rooms[code]
    real = {party.public_id(room, pid): pid for pid in room["players"]}
    return {real[p["id"]]: p["score"] for p in players}


def _pub(code, pid):
    return party.public_id(party._rooms[code], pid)


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


def _scores_seen(code):
    """Every round shows its scoreboard: let it have been up its time, so 'next' moves on."""
    party.get_room_state(code)  # the reveal goes on screen first (its clock starts when it does)
    with party._lock:
        room = party._rooms[code]
        if room["phase"] == "reveal":
            room["phase_started_at"] -= party.scores_at(room) + party.SCOREBOARD_MIN_SECONDS + 1


def test_seven_round_schedule_covers_all_five_modes():
    party.create_or_join_room(code="NOAH", player_id="p1", name="Jeremiah", initials="JER")
    party.create_or_join_room(code="NOAH", player_id="p2", name="Michele", initials="MIC")
    party.create_or_join_room(code="NOAH", player_id="p3", name="Kate", initials="KAT")

    seen_kinds = []
    state = party.start_match("NOAH", player_id="p1")
    seen_kinds.append(state["prompt"]["kind"])

    for _ in range(6):
        # Advance writing -> voting -> reveal -> next round
        party.advance_phase("NOAH", player_id="p1")
        party.advance_phase("NOAH", player_id="p1")
        _scores_seen("NOAH")
        state = party.advance_phase("NOAH", player_id="p1")
        seen_kinds.append(state["prompt"]["kind"])

    # Every match: an easy opener, the double-points finale, a Truth or Bluff,
    # and a different mix in between (drawn per match from every round type).
    assert seen_kinds[0] == party.MATCH_OPENER and seen_kinds[-1] == party.MATCH_FINALE
    assert "truth_or_bluff" in seen_kinds and len(set(seen_kinds)) >= 5
    assert all(k in party.ROUND_META for k in seen_kinds)
    ids = party._rooms["NOAH"]["used_prompt_ids"]
    assert len(ids) == len(set(ids)) >= 7  # no prompt twice in a match (a race round uses several clues)
    assert seen_kinds[-1] == "survey_says"


def test_scoring_original_answer_decoy_bait_and_house_trap():
    party.create_or_join_room(code="DOVE", player_id="p1", name="Jeremiah", initials="JER")
    party.create_or_join_room(code="DOVE", player_id="p2", name="Michele", initials="MIC")
    party.create_or_join_room(code="DOVE", player_id="p3", name="Kate", initials="KAT")
    party.start_match("DOVE", player_id="p1")

    # p1 writes an original answer; p2 plays Scribe's Decoy; p3 writes an original answer
    party.submit_answer("DOVE", player_id="p1", text="My custom wise punchline")
    party.submit_answer("DOVE", player_id="p2", use_decoy=True)
    state = party.submit_answer("DOVE", player_id="p3", text="Another custom answer")

    # All 3 humans submitted -> auto-advances to voting!
    assert state["phase"] == "voting"

    # Find option IDs from internal room
    raw_room = party._rooms["DOVE"]
    p1_opt = next(o["option_id"] for o in raw_room["options"] if "p1" in o["authors"])
    decoy_opt = "opt_house_decoy"

    # p2 votes for p1's original answer (+1000 to p1)
    # p3 votes for p1's original answer (+1000 to p1, plus +200 round winner bonus to p1 = 2200)
    # p1 votes for the Scribe's Decoy (-250 trap penalty to p1, +500 decoy bait to p2!)
    party.submit_vote("DOVE", player_id="p2", option_id=p1_opt)
    party.submit_vote("DOVE", player_id="p3", option_id=p1_opt)
    reveal_state = party.submit_vote("DOVE", player_id="p1", option_id=decoy_opt)

    assert reveal_state["phase"] == "reveal"
    scores = _by_real_id("DOVE", reveal_state["players"])
    assert scores["p1"] == 1000 + 1000 + 200 - 250  # 1950
    assert scores["p2"] == 500
    assert scores["p3"] == 0


def test_truth_or_bluff_blocks_real_truth_and_scores_truth_and_bluffs():
    party.create_or_join_room(code="LION", player_id="p1", name="Jeremiah", initials="JER")
    party.create_or_join_room(code="LION", player_id="p2", name="Michele", initials="MIC")
    party.create_or_join_room(code="LION", player_id="p3", name="Kate", initials="KAT")
    party.start_match("LION", player_id="p1")

    # Jump to a Truth or Bluff round (the mix between opener and finale is random)
    with party._lock:
        party._rooms["LION"]["schedule"][2] = "truth_or_bluff"
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
    scores = _by_real_id("LION", reveal["players"])
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

    start_resp = client.post("/api/party/rooms/GRACE/start", json={"player_id": pid})
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

    wav1 = party.synthesize_announcer_wav("Welcome to Parable Party!", voice="am_fenrir,am_michael")
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

    cast_host = client.post("/api/party/rooms", json={"code": "CAST", "name": "Jeremiah", "initials": "JER"}).get_json()["player_id"]
    voice_set = client.post("/api/party/rooms/CAST/voice", json={"voice": "clone:jeremiah-new-e572bb", "player_id": cast_host})
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


def test_jubilee_cast_sends_the_jubilee_board_page(client, monkeypatch):
    import json

    sent = []

    def _fake_urlopen(req, timeout=8.0):
        sent.append((req.full_url, json.loads(req.data.decode())))
        return _JsonResp(json.dumps({"success": True, "display": "web", "verified": True}).encode())

    monkeypatch.setattr(party, "JARVIS_GATEWAY_URL", "")
    monkeypatch.setattr(party, "EXECUTION_SVC_URL", "http://exec:8003")
    monkeypatch.setattr(party.urllib.request, "urlopen", _fake_urlopen)
    resp = client.post(
        "/api/jubilee/cast",
        json={"screen_id": "ha:media_player.den", "room_code": "noah", "url": "http://evil.example/x"},
        headers={"X-Forwarded-Proto": "https", "X-Forwarded-Host": "games.example.org"},
    )
    assert resp.status_code == 200 and resp.get_json()["success"] is True
    assert sent[0][0] == "http://exec:8003/execute/stage/show"
    assert sent[0][1]["url"] == "https://games.example.org/jubilee/tv/NOAH"


def test_jubilee_cast_through_jarvis_names_the_game(client, monkeypatch):
    """The gateway builds the TV address itself; the arcade says whose stage it is."""
    import json

    sent = []

    def _fake_urlopen(req, timeout=8.0):
        sent.append((req.full_url, json.loads(req.data.decode())))
        return _JsonResp(json.dumps({"success": True}).encode())

    monkeypatch.setattr(party, "JARVIS_GATEWAY_URL", "http://jarvis:11435")
    monkeypatch.setattr(party.urllib.request, "urlopen", _fake_urlopen)
    client.post("/api/jubilee/cast", json={"screen_id": "ha:media_player.den", "room_code": "NOAH"})
    assert sent[0][0] == "http://jarvis:11435/api/arcade/cast"
    assert sent[0][1]["game"] == "family-jubilee" and sent[0][1]["room_code"] == "NOAH"
    # Parable Party's cast stays unnamed (the gateway's default stage).
    client.post("/api/party/cast", json={"screen_id": "ha:media_player.den", "room_code": "ZION"})
    assert "game" not in sent[-1][1]


def test_jubilee_cast_without_a_screen_is_a_bad_request(client):
    assert client.post("/api/jubilee/cast", json={"room_code": "NOAH"}).status_code == 400


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


def test_good_word_controller_contract():
    from pathlib import Path

    html = Path("arcade/family_games/good-word/game.html").read_text(encoding="utf-8")
    # Alpaca TTS only, never the browser's robotic voices.
    assert "speechSynthesis" not in html and "SpeechSynthesisUtterance" not in html
    # What a player types is never masked: the TV never shows drafts.
    assert 'type="password"' not in html
    # When a TV stage has played for the room, phones stay silent; otherwise only the host speaks.
    assert "stageEverSeen" in html and "you_are_host" in html
    # A join lobby, and host-only controls sent with the player's id.
    assert 'id="gw-join"' in html and 'id="gw-host"' in html and 'id="gw-start"' in html
    assert "player_id: myId" in html


def test_good_word_page_meets_the_mobile_contract():
    from arcade import family, mobile

    assert mobile.contract_failures(family.build_html("good-word")) == []


def test_players_pick_unique_avatars():
    a = party.create_or_join_room(code="AVTR", player_id="p1", name="Ann", avatar="🦉")
    assert next(p for p in a["room"]["players"] if p["id"] == _pub("AVTR", "p1"))["avatar"] == "🦉"
    b = party.create_or_join_room(code="AVTR", player_id="p2", name="Bo", avatar="🦉")
    bo = next(p for p in b["room"]["players"] if p["id"] == _pub("AVTR", "p2"))
    assert bo["avatar"] != "🦉" and bo["avatar"] in party.PLAYER_AVATARS
    c = party.create_or_join_room(code="AVTR", player_id="p3", name="Cy", avatar="not-an-avatar")
    assert next(p for p in c["room"]["players"] if p["id"] == _pub("AVTR", "p3"))["avatar"] in party.PLAYER_AVATARS


def test_only_the_host_starts_and_advances(client):
    client.post("/api/party/rooms", json={"code": "HOSTY", "player_id": "h", "name": "Host"})
    client.post("/api/party/rooms", json={"code": "HOSTY", "player_id": "g", "name": "Guest"})
    assert client.post("/api/party/rooms/HOSTY/start", json={"player_id": "g"}).status_code == 403
    ok = client.post("/api/party/rooms/HOSTY/start", json={"player_id": "h"})
    assert ok.status_code == 200 and ok.get_json()["room"]["you_are_host"] is True
    assert client.post("/api/party/rooms/HOSTY/next", json={"player_id": "g"}).status_code == 403
    # The phase timer (no player id) may always advance.
    # A request that names no player is nobody: it cannot act as the host.
    assert client.post("/api/party/rooms/HOSTY/next", json={}).status_code == 403


def test_host_role_passes_on_when_the_host_leaves(monkeypatch):
    party.create_or_join_room(code="HAND", player_id="h", name="Host")
    party.create_or_join_room(code="HAND", player_id="g", name="Guest")
    with party._lock:
        party._rooms["HAND"]["players"]["h"]["last_seen"] -= party.HOST_HANDOFF_SECONDS + 5
    state = party.get_room_state("HAND", viewer_id="g")
    assert state["you_are_host"] is True
    assert next(p for p in state["players"] if p["id"] == _pub("HAND", "h"))["connected"] is False


def test_reveal_lasts_longer_with_more_answers():
    for i in range(4):
        party.create_or_join_room(code="REVL", player_id=f"p{i}", name=f"P{i}")
    party.start_match("REVL", player_id="p0")
    for i in range(4):
        party.submit_answer("REVL", player_id=f"p{i}", text=f"answer number {i}")
    room = party.get_room_state("REVL", viewer_id="p0")
    for i in range(4):
        mine = party.get_room_state("REVL", viewer_id=f"p{i}")["options"]
        party.submit_vote("REVL", player_id=f"p{i}", option_id=next(o for o in mine if not o["is_mine"])["option_id"])
    room = party.get_room_state("REVL")
    assert room["phase"] == "reveal"
    expected = (party.REVEAL_SECONDS + party.REVEAL_PER_ANSWER_SECONDS * len(room["options"])
                + party.REVEAL_LINGER_SECONDS)
    # At least that long; longer only while the result and its joke are still being said.
    assert room["phase_seconds"] >= expected - 0.2


def test_host_device_can_play_without_projecting_answers_to_tv_stage():
    # Host device joins and creates room; TV stage joins as spectator
    party.create_or_join_room(code="HOST", player_id="p_host", name="Jeremiah", initials="JER", spectator=True)
    party.create_or_join_room(code="HOST", player_id="p_mic", name="Michele", initials="MIC")
    party.start_match("HOST", player_id="p_host")

    # Even if the Host Device initially opened TV stage mode, submitting an answer promotes Host to active player!
    host_state = party.submit_answer("HOST", player_id="p_host", text="My secret host punchline")
    host_player = next(p for p in host_state["players"] if p["id"] == _pub("HOST", "p_host"))
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


def test_leaving_hands_off_host_and_the_last_human_closes_the_room(client):
    client.post("/api/party/rooms", json={"code": "BYE", "player_id": "h", "name": "Host"})
    client.post("/api/party/rooms", json={"code": "BYE", "player_id": "g", "name": "Guest"})
    out = client.post("/api/party/rooms/BYE/leave", json={"player_id": "h"}).get_json()
    assert out["closed"] is False
    assert client.get("/api/party/rooms/BYE?player_id=g").get_json()["room"]["you_are_host"] is True
    out = client.post("/api/party/rooms/BYE/leave", json={"player_id": "g"}).get_json()
    assert out["closed"] is True
    assert client.get("/api/party/rooms/BYE").status_code == 404


def test_a_leaver_mid_round_does_not_hold_up_the_round():
    for pid in ("a", "b", "c"):
        party.create_or_join_room(code="MID", player_id=pid, name=pid.upper())
    party.start_match("MID", player_id="a")
    party.submit_answer("MID", player_id="a", text="one")
    party.submit_answer("MID", player_id="b", text="two")
    out = party.leave_room("MID", player_id="c")
    assert out["room"]["phase"] == "voting"


def test_only_the_host_or_an_admin_ends_a_room(client, monkeypatch):
    client.post("/api/party/rooms", json={"code": "END", "player_id": "h", "name": "Host"})
    client.post("/api/party/rooms", json={"code": "END", "player_id": "g", "name": "Guest"})
    assert client.post("/api/party/rooms/END/end", json={"player_id": "g"}).status_code == 403
    assert client.post("/api/party/rooms/END/end", json={"player_id": "h"}).status_code == 200
    assert client.get("/api/party/rooms/END").status_code == 404

    client.post("/api/party/rooms", json={"code": "ADM", "player_id": "h", "name": "Host"})
    monkeypatch.setattr(arcade_app, "ADMIN_TOKEN", "tok")
    assert client.post("/api/party/rooms/ADM/end", json={}, headers={"X-Arcade-Token": "nope"}).status_code == 403
    assert client.post("/api/party/rooms/ADM/end", json={}, headers={"X-Arcade-Token": "tok"}).status_code == 200


def test_abandoned_rooms_close_by_themselves():
    party.create_or_join_room(code="IDLE", player_id="p", name="P")
    with party._lock:
        party._rooms["IDLE"]["players"]["p"]["last_seen"] -= party.ROOM_IDLE_SECONDS + 5
    assert all(r["code"] != "IDLE" for r in party.list_active_rooms())


def test_a_streamed_stage_delays_each_phase_so_phones_wait_for_the_tv(client):
    """A Roku sees the stage as video ~8 s late: every phase gets that time, phones hold."""
    client.post("/api/party/rooms", json={"code": "ROKU", "player_id": "p1", "name": "Ann"})
    client.post("/api/party/rooms", json={"code": "ROKU", "player_id": "p2", "name": "Ben"})
    client.get("/api/party/rooms/ROKU?stage=1&video=1")  # the headless renderer polling
    client.post("/api/party/rooms/ROKU/start", json={"player_id": "p1"})
    room = client.get("/api/party/rooms/ROKU?player_id=p2").get_json()["room"]
    assert room["phase"] == "writing"
    assert room["stage_delay_s"] == party.STREAM_STAGE_DELAY_S
    read = room["phase_seconds"] - party.WRITE_SECONDS - party.STREAM_STAGE_DELAY_S  # the story and prompt are read first
    assert party.PROMPT_READ_SECONDS - 0.2 <= read <= party.PROMPT_READ_MAX_S + 0.2
    assert 0 <= room["phase_age_s"] < 2


def test_a_live_page_stage_adds_no_delay(client):
    client.post("/api/party/rooms", json={"code": "LGTV", "player_id": "p1", "name": "Ann"})
    client.post("/api/party/rooms", json={"code": "LGTV", "player_id": "p2", "name": "Ben"})
    client.get("/api/party/rooms/LGTV?stage=1")
    client.post("/api/party/rooms/LGTV/start", json={"player_id": "p1"})
    room = client.get("/api/party/rooms/LGTV?player_id=p2").get_json()["room"]
    assert room["stage_delay_s"] == 0
    read = room["phase_seconds"] - party.WRITE_SECONDS  # the clock starts when the question has been asked
    assert -0.2 <= read <= party.PROMPT_READ_MAX_S + 0.2


def test_stream_delay_ends_when_the_streamed_stage_goes_away(monkeypatch):
    party.create_or_join_room(code="GONE", player_id="p1", name="Ann")
    party.get_room_state("GONE", stage=True, video=True)
    room = party._rooms["GONE"]
    assert party._stage_delay(room) == party.STREAM_STAGE_DELAY_S
    assert party._stage_delay(room, now=room["video_stage_seen_at"] + party.STAGE_PRESENCE_SECONDS + 1) == 0.0


def test_phone_and_tv_pages_know_about_the_stream_delay(client):
    tv = client.get("/tv").get_data(as_text=True)
    assert "HeadlessChrome" in tv and "&video=1" in tv and "stage_delay_s" in tv
    from arcade import family
    phone = family.build_html("good-word")
    assert "viewHold" in phone and "stage_delay_s" in phone and "phase_age_s" in phone


def test_host_can_tune_the_stream_lag_and_others_cannot(client):
    client.post("/api/party/rooms", json={"code": "SYNC", "player_id": "p1", "name": "Ann"})
    client.post("/api/party/rooms", json={"code": "SYNC", "player_id": "p2", "name": "Ben"})
    assert client.post("/api/party/rooms/SYNC/stream_delay", json={"player_id": "p2", "seconds": 3}).status_code == 403
    assert client.post("/api/party/rooms/SYNC/stream_delay", json={"player_id": "p1", "seconds": 11}).get_json()["stream_delay_s"] == 11
    # No streamed TV: still no delay at all.
    assert client.get("/api/party/rooms/SYNC?player_id=p1").get_json()["room"]["stage_delay_s"] == 0
    client.get("/api/party/rooms/SYNC?stage=1&video=1")
    assert client.get("/api/party/rooms/SYNC?player_id=p1").get_json()["room"]["stage_delay_s"] == 11


def test_reveal_moves_on_by_itself_and_never_waits_on_the_ai(monkeypatch):
    """The game must not stall on the results screen, and a slow AI must never hold the room lock."""
    party.create_or_join_room(code="FLOW", player_id="h", name="Host")
    party.create_or_join_room(code="FLOW", player_id="g", name="Guest")
    party.start_match("FLOW", player_id="h")
    with party._lock:
        room = party._rooms["FLOW"]
        room["schedule"] = [party.MATCH_OPENER, "sub_the_scroll", "extra_extra", "truth_or_bluff", "bible_bios",
                            "hymn_remix", party.MATCH_FINALE]  # writing rounds only: this is about the reveal
        party._build_voting_options_unlocked(room)
        party._score_round_unlocked(room)
        assert room["phase"] == "reveal"
        room["phase_deadline"] = 0.01  # the scores have been up long enough
    party.get_room_state("FLOW", "g")
    assert party._rooms["FLOW"]["phase"] == "writing" and party._rooms["FLOW"]["round"] == 2

    # The host asks for AI prompts: the model is called with the lock released.
    held = []

    def slow_model(kind):
        held.append(party._lock.locked())
        return None

    monkeypatch.setattr(party, "_generate_ai_prompt", slow_model)
    # "Is the lock held?" cannot tell whose thread holds it: keep the line-making helpers out of this one.
    monkeypatch.setattr(party, "_measure_reveal_line_async", lambda *a, **k: None)
    monkeypatch.setattr(party, "_premake_race_lines_async", lambda *a, **k: None)
    with party._lock:
        room = party._rooms["FLOW"]
        room["round"] = 4
        room["schedule"][4] = "blank_o_matic"  # next is round 5, made an AI-written round
        party._build_voting_options_unlocked(room)
        party._score_round_unlocked(room)
        room.pop("next_prompt", None)
    _scores_seen("FLOW")
    party.advance_phase("FLOW", use_ai_prompt=True, player_id="h")
    assert held and held[0] is False
    assert party._rooms["FLOW"]["round"] == 5


def test_a_prefetched_ai_prompt_is_used_for_the_next_round():
    party.create_or_join_room(code="PREF", player_id="h", name="Host")
    party.start_match("PREF", player_id="h")
    with party._lock:
        room = party._rooms["PREF"]
        room["round"] = 4
        room["schedule"][4] = "blank_o_matic"
        room["phase"] = "reveal"
        room["next_prompt"] = {"round": 5, "prompt": {"id": "ai_1", "prompt": "Noah forgot to pack ______.", "decoy": "the ark",
                                                      "bot_answers": ["snacks"]}}
    _scores_seen("PREF")
    party.advance_phase("PREF", player_id="h")
    assert party._rooms["PREF"]["current_prompt"]["prompt"] == "Noah forgot to pack ______."


def test_a_last_second_answer_is_not_replaced_by_the_house_answer():
    """Reported: an answer typed in the final seconds was swapped for the house answer."""
    party.create_or_join_room(code="LATE", player_id="a", name="Ann")
    party.create_or_join_room(code="LATE", player_id="b", name="Ben")
    party.start_match("LATE", player_id="a")
    party.submit_answer("LATE", player_id="b", text="A camel with opinions")
    with party._lock:
        party._rooms["LATE"]["phase_deadline"] = time.time() - 1.0  # the phone showed 0 a second ago
    room = party.submit_answer("LATE", player_id="a", text="Forty days of leftovers")
    assert room["my_submission"]["text"] == "Forty days of leftovers"


def test_an_answer_just_after_voting_opened_replaces_the_house_answer():
    party.create_or_join_room(code="LAT2", player_id="a", name="Ann")
    party.create_or_join_room(code="LAT2", player_id="b", name="Ben")
    party.start_match("LAT2", player_id="a")
    party.submit_answer("LAT2", player_id="b", text="A camel with opinions")
    with party._lock:
        room = party._rooms["LAT2"]
        room["phase_deadline"] = time.time() - party.SUBMIT_GRACE_SECONDS - 1
        party._check_auto_advance_unlocked(room)
        assert room["phase"] == "voting" and room["submissions"]["a"].get("auto_filled")
    snap = party.submit_answer("LAT2", player_id="a", text="Forty days of leftovers")
    texts = [o["text"] for o in snap["options"]]
    assert "Forty days of leftovers" in texts
    decoy = next(o for o in party._rooms["LAT2"]["options"] if o["option_id"] == "opt_house_decoy")
    assert "a" not in decoy["authors"]
    # Once anyone has voted, the round's options are settled.
    with party._lock:
        room = party._rooms["LAT2"]
        room["votes"] = {"b": "opt_house_decoy"}
        room["submissions"]["a"]["auto_filled"] = True
    with pytest.raises(ValueError):
        party.submit_answer("LAT2", player_id="a", text="Another try")


def test_bots_finish_at_random_moments_not_all_at_once():
    party.create_or_join_room(code="BOTS", player_id="a", name="Ann")
    party.start_match("BOTS", player_id="a")
    room = party._rooms["BOTS"]
    bots = [p for p in room["players"].values() if p.get("is_bot")]
    assert bots
    times = [room["submissions"][b["id"]]["submitted_at"] - room["phase_started_at"] for b in bots]
    assert all(party.PROMPT_READ_SECONDS + 6 <= t <= party.PROMPT_READ_SECONDS + party.WRITE_SECONDS for t in times)
    snap = party.serialize_room(room)
    assert not any(p["submitted"] for p in snap["players"] if p["is_bot"])  # nobody is "Sealed" at 0 s
    with party._lock:
        party._build_voting_options_unlocked(room)
    assert all(b["id"] in room["votes"] for b in bots)  # cast already, so the round never waits on them
    assert not any(p["voted"] for p in party.serialize_room(room)["players"] if p["is_bot"])
    with party._lock:
        for b in bots:
            room["bot_vote_at"][b["id"]] = 0
    assert all(p["voted"] for p in party.serialize_room(room)["players"] if p["is_bot"])


def test_review_fixes_hold(client):
    """Regression guard for the independent review's findings."""
    # Anonymous requests are nobody: no host powers.
    party.create_or_join_room(code="REV", player_id="h", name="Host")
    party.create_or_join_room(code="REV", player_id="g", name="Guest")
    with pytest.raises(PermissionError):
        party.start_match("REV")
    assert client.post("/api/party/rooms/REV/voice", json={"voice": "am_onyx", "player_id": "g"}).status_code == 403
    # Nobody's private id is on anyone's screen.
    snap = party.get_room_state("REV", "g")
    assert "h" not in {p["id"] for p in snap["players"]} and snap["host_id"] != "h"
    assert snap["you"] == party.public_id(party._rooms["REV"], "g")
    # A stale Next (the timer already moved on) does nothing.
    party.start_match("REV", player_id="h")
    state = party.advance_phase("REV", player_id="h", expect={"phase": "reveal", "round": 1})
    assert state["phase"] == "writing"
    # start_match cannot restart a running match.
    assert party.start_match("REV", player_id="h")["round"] == 1
    # An AI scribe's seat can't be taken.
    bot = next(p for p in party._rooms["REV"]["players"].values() if p.get("is_bot"))
    with pytest.raises(PermissionError):
        party.create_or_join_room(code="REV", player_id=bot["id"], name="Me")


def test_no_points_for_a_seat_that_never_answered():
    for pid in ("a", "b", "c"):
        party.create_or_join_room(code="NOAN", player_id=pid, name=pid.upper())
    party.start_match("NOAN", player_id="a")
    party.submit_answer("NOAN", player_id="a", text="first")
    party.submit_answer("NOAN", player_id="b", text="second")
    with party._lock:
        room = party._rooms["NOAN"]
        party._build_voting_options_unlocked(room)
    for pid in ("a", "b"):
        party.submit_vote("NOAN", player_id=pid, option_id="opt_house_decoy")
    with party._lock:
        party._score_round_unlocked(party._rooms["NOAN"])
    assert party._rooms["NOAN"]["players"]["c"]["score"] == 0


def test_spectators_cannot_vote_and_the_room_cap_holds():
    for i in range(8):
        party.create_or_join_room(code="FULL", player_id=f"p{i}", name=f"P{i}")
    late = party.create_or_join_room(code="FULL", player_id="late", name="Late")
    assert party._rooms["FULL"]["players"]["late"]["spectator"] is True
    party.create_or_join_room(code="FULL", player_id="late", name="Late")  # a refresh is not a seat
    assert party._rooms["FULL"]["players"]["late"]["spectator"] is True
    party.start_match("FULL", player_id="p0")
    with pytest.raises(ValueError):
        party.submit_answer("FULL", player_id="late", text="sneaky")
    with party._lock:
        party._build_voting_options_unlocked(party._rooms["FULL"])
    with pytest.raises(ValueError):
        party.submit_vote("FULL", player_id="late", option_id="opt_house_decoy")
    assert late["room"]["players"]


def test_whole_word_truth_check_lets_honest_bluffs_through():
    prompt = {"kind": "truth_or_bluff", "truth": "Because it was full of the contents of the ark",
              "truth_keywords": ["tents"]}
    assert party._matches_truth("They forgot the tent pegs, so no tents", prompt)  # the keyword, as a word
    assert party._matches_truth("because it was full of the contents of the ark", prompt)
    assert not party._matches_truth("Their contents were mostly snacks", prompt)  # "tents" inside "contents"
    assert not party._matches_truth("because it rained", prompt)


def test_ending_a_cast_room_frees_its_tv(monkeypatch):
    freed = []
    monkeypatch.setattr(party, "show_stage_on_screen", lambda screen, url, **kw: freed.append((screen, kw)))
    party.create_or_join_room(code="FREE", player_id="h", name="Host")
    party.remember_screen("FREE", "ha:media_player.master_bedroom_tv")
    party.end_room("FREE", player_id="h")
    for _ in range(50):
        if freed:
            break
        time.sleep(0.02)
    assert freed and freed[0][0] == "ha:media_player.master_bedroom_tv" and freed[0][1]["stop"] is True


def test_second_review_fixes_hold(client):
    # A wrong room code is a 404, not a crash.
    resp = client.post("/api/party/rooms", json={"code": "NOPE", "player_id": "x", "name": "X", "create_if_missing": False})
    assert resp.status_code == 404
    # Nobody can act as an AI scribe.
    party.create_or_join_room(code="SCRB", player_id="h", name="Host")
    party.start_match("SCRB", player_id="h")
    bot = next(p for p in party._rooms["SCRB"]["players"].values() if p.get("is_bot"))
    with pytest.raises(PermissionError):
        party.submit_answer("SCRB", player_id=bot["id"], text="I control you")
    with pytest.raises(PermissionError):
        party.leave_room("SCRB", player_id=bot["id"])
    assert client.post("/api/party/rooms/SCRB/submit", json={"player_id": bot["id"], "text": "x"}).status_code == 403
    # Joining doesn't change the host's announcer voice.
    before = party._rooms["SCRB"]["voice"]
    party.create_or_join_room(code="SCRB", player_id="g", name="G", voice="am_puck")
    assert party._rooms["SCRB"]["voice"] == before
    # A garbled "expect" is just a stale tap.
    assert party.advance_phase("SCRB", player_id="h", expect={"round": "two"})["phase"] == "writing"


def test_a_seat_that_never_answered_can_still_fall_for_the_trap():
    for pid in ("a", "b", "c"):
        party.create_or_join_room(code="TRAP", player_id=pid, name=pid.upper())
    party.start_match("TRAP", player_id="a")
    party.submit_answer("TRAP", player_id="a", text="first")
    party.submit_answer("TRAP", player_id="b", text="second")
    with party._lock:
        party._build_voting_options_unlocked(party._rooms["TRAP"])
    decoy = next(o for o in party.get_room_state("TRAP", "c")["options"] if o["option_id"] == "opt_house_decoy")
    assert decoy["is_mine"] is False


def test_one_room_per_tv():
    party.create_or_join_room(code="TVA", player_id="a", name="A")
    party.create_or_join_room(code="TVB", player_id="b", name="B")
    party.remember_screen("TVA", "cast:den")
    party.remember_screen("TVB", "cast:den")
    assert party._rooms["TVA"].get("cast_screen") is None and party._rooms["TVB"]["cast_screen"] == "cast:den"


def test_prompts_rotate_least_recently_used_across_rooms():
    party._recent_prompt_ids.clear()
    seen = []
    for i in range(len(party.PROMPT_BANK["blank_o_matic"]) * 2):
        room = {"code": f"R{i}", "used_prompt_ids": []}
        seen.append(party._select_prompt_unlocked(room, "blank_o_matic")["id"])
    n = len(party.PROMPT_BANK["blank_o_matic"])
    assert len(set(seen[:n])) == n  # every prompt once before any repeats


def test_the_reveal_ends_on_a_scoreboard_with_whats_next():
    for pid, name in (("a", "Ann"), ("b", "Ben"), ("c", "Cy")):
        party.create_or_join_room(code="SCOR", player_id=pid, name=name)
    party.start_match("SCOR", player_id="a")
    with party._lock:
        room = party._rooms["SCOR"]
        party._build_voting_options_unlocked(room)
        party._score_round_unlocked(room)
        snap = party.serialize_room(room, "a")
    assert snap["scores_at_s"] > party.reveal_result_at(room)  # after the result is called
    # Up long enough to read, and no longer than the joke being told over it needs.
    shown = snap["seconds_remaining"] - snap["scores_at_s"]
    assert shown >= 6
    assert shown <= max(10, party.SPEECH_START_SLACK_S + party.speech_seconds(party.reveal_result_line(room)) + 1.5)
    assert snap["up_next"]["round"] == 2 and snap["up_next"]["title"] == party.ROUND_META[party.round_kind(room, 2)]["title"]
    with party._lock:
        room["round"] = room["total_rounds"]
        assert party.serialize_room(room, "a")["up_next"] is None


def test_a_late_answer_matching_one_already_on_the_ballot_is_not_added_twice():
    party.create_or_join_room(code="LAT3", player_id="a", name="Ann")
    party.create_or_join_room(code="LAT3", player_id="b", name="Ben")
    party.start_match("LAT3", player_id="a")
    party.submit_answer("LAT3", player_id="b", text="A camel with opinions")
    with party._lock:
        room = party._rooms["LAT3"]
        room["phase_deadline"] = time.time() - party.SUBMIT_GRACE_SECONDS - 1
        party._check_auto_advance_unlocked(room)
        assert room["phase"] == "voting"
    with contextlib.suppress(ValueError):  # too late, and not a new answer: refused
        party.submit_answer("LAT3", player_id="a", text="a camel with OPINIONS!")
    texts = [party._normalize_compare(o["text"]) for o in party._rooms["LAT3"]["options"]]
    assert len(texts) == len(set(texts))


def test_the_reveal_waits_for_the_joke_to_be_told():
    """Reported: the next screen came before the joke's punchline."""
    for pid, name in (("a", "Ann"), ("b", "Ben"), ("c", "Cy")):
        party.create_or_join_room(code="JOKE", player_id=pid, name=name)
    party.start_match("JOKE", player_id="a")
    with party._lock:
        room = party._rooms["JOKE"]
        party._build_voting_options_unlocked(room)
        party._score_round_unlocked(room)
        line = party.reveal_result_line(room)
        assert "..." in line  # a joke follows the result between rounds
        said_by = room["phase_started_at"] + party.reveal_result_at(room) + party.speech_seconds(line)
        assert room["phase_deadline"] >= said_by + party.SPEECH_START_SLACK_S
    assert party.speech_seconds("one two three four five six seven eight nine ten eleven twelve thirteen") == pytest.approx(
        13 / party.SPOKEN_WORDS_PER_SECOND)


def _reveal_room(code):
    for pid, name in (("a", "Ann"), ("b", "Ben"), ("c", "Cy")):
        party.create_or_join_room(code=code, player_id=pid, name=name)
    party.start_match(code, player_id="a")
    with party._lock:
        room = party._rooms[code]
        party._build_voting_options_unlocked(room)
        party._score_round_unlocked(room)
    return room


def test_the_reveal_is_fitted_to_the_measured_line():
    room = _reveal_room("MEAS")
    with party._lock:
        start = room["phase_started_at"]
        party._apply_measured_line_unlocked(room, 30.0)  # a long line, timed from its audio
        assert room["phase_deadline"] == pytest.approx(
            start + party.reveal_result_at(room) + party.MEASURED_START_SLACK_S + 30.0 + 1.0)
        party._apply_measured_line_unlocked(room, 1.0)  # a short one never cuts the reveal or the scoreboard
        assert room["phase_deadline"] >= room["reveal_base_deadline"]
        assert room["phase_deadline"] >= start + party.scores_at(room) + 6.0


def test_a_line_once_said_is_timed_from_its_audio():
    from arcade import announcer
    wav = announcer.write_wav([0.0] * (24000 * 7), 24000)  # 7 s
    party._record_spoken_length("Seven seconds of wisdom.", "v", wav)
    assert party.speech_seconds("Seven seconds of wisdom.", "v") == pytest.approx(7.0)
    assert party.speech_seconds("Seven seconds of wisdom.", "other voice") != pytest.approx(7.0)  # estimated


def test_jokes_are_random_never_repeated_in_a_room_and_freshest_first():
    party._recent_jokes.clear()
    a, b = {"code": "J1", "match_id": "m1", "round": 1}, {"code": "J2", "match_id": "m2", "round": 1}
    first = party.joke_for(a, "writing")
    assert party.joke_for(a, "writing") == first  # chosen once, then kept for that moment
    told = {first}
    for rnd in range(1, 8):
        a["round"] = rnd
        told |= {party.joke_for(a, "writing"), party.joke_for(a, "between")}
    assert len(told) == 14  # no repeats within the room
    # Another room tonight hears jokes the first did not tell, while any are left.
    assert party.joke_for(b, "between") not in told


def test_a_writing_joke_only_where_it_fits_before_the_hurry_call():
    room = _reveal_room("FITS")
    with party._lock:
        party._advance_unlocked(room)  # to round 2 writing (or a race)
        if room["phase"] != "writing":
            return
        lines = party.chatter_for(room)
    joke = next(c for c in lines if c["text"].startswith(party.HOST_THINK_LEADIN))
    hurry_at = (room["phase_deadline"] - room["phase_started_at"]) - 10.0
    assert joke["at"] + party.speech_seconds(joke["text"], room.get("voice")) <= hurry_at


def test_the_host_never_repeats_a_line_in_a_room_until_the_list_runs_out():
    """Reported: the announcer was way too repetitive."""
    room = {"code": "SAY1", "match_id": "m", "round": 1, "players": {}}
    said = []
    for rnd in range(1, 8):
        room["round"] = rnd
        said.append(party._pick(room, party._pool(party.HOST_VOTING, party.MORE_VOTING), "vote"))
    assert len(said) == len(set(said))
    assert party._pick({**room}, ("only",), "x") == "only"
    # Chosen once per moment: the TV and phones asking again get the same line.
    assert party._pick(room, party._pool(party.HOST_VOTING, party.MORE_VOTING), "vote") == said[-1]


def test_standings_talk_about_the_scoreboard():
    players = {"a": {"name": "Ann", "score": 3000, "round_delta": 0}, "b": {"name": "Ben", "score": 2900, "round_delta": 0},
               "c": {"name": "Cy", "score": 100, "round_delta": 0}}
    room = {"code": "STD1", "match_id": "m", "round": 3, "total_rounds": 7, "players": players}
    line = party.standings_line(room)
    if party.STANDINGS.get("close"):
        assert "Ann" in line and "Ben" in line
    players["b"]["score"] = 3000
    room["round"] = 4
    if party.STANDINGS.get("tied"):
        assert "Ann" in party.standings_line(room) or "Ben" in party.standings_line(room)


def test_line_timings_make_room_for_new_ones():
    from arcade import announcer
    wav = announcer.write_wav([0.0] * 2400, 24000)
    old_max = party.SPOKEN_SECONDS_MAX
    try:
        party.SPOKEN_SECONDS_MAX = 3
        party._spoken_seconds.clear()
        for i in range(5):
            party._record_spoken_length(f"line {i}", "v", wav)
        assert list(k[1] for k in party._spoken_seconds) == ["line 2", "line 3", "line 4"]  # newest kept
    finally:
        party.SPOKEN_SECONDS_MAX = old_max
        party._spoken_seconds.clear()


def test_screens_ask_for_the_whole_line():
    import pathlib
    for f in ("arcade/templates/party_tv.html", "arcade/family_games/good-word/game.html"):
        src = pathlib.Path(f).read_text()
        assert ".slice(0, 320)" not in src and f".slice(0, {party.TTS_MAX_CHARS})" in src, f
