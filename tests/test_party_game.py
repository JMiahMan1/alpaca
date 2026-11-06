"""Tests for the 'Use Your Good Word' Biblical Worldview multiplayer party engine."""

from __future__ import annotations

import pytest

import arcade.app as arcade_app
from arcade import party


@pytest.fixture(autouse=True)
def clear_party_state():
    with party._lock:
        party._rooms.clear()
        party._tts_cache.clear()
        party._tts_clips.clear()
    yield
    with party._lock:
        party._rooms.clear()
        party._tts_cache.clear()
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
    assert b"family-good-word" in tv_resp.data
    assert b"tv=1&room=GRACE" in tv_resp.data

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


def test_party_voices_and_tv_cast_across_roku_lg_samsung_chrome_and_android_tv(client, monkeypatch):
    import base64
    import json

    fake_wav = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 32
    calls = []

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
        url = req.full_url if hasattr(req, "full_url") else str(req)
        method = req.get_method() if hasattr(req, "get_method") else "GET"
        raw_data = req.data.decode("utf-8") if getattr(req, "data", None) else ""
        calls.append((method, url, raw_data))
        if url.endswith("/api/tts"):
            return _FakeResp(
                json.dumps({"audio_b64": base64.b64encode(fake_wav).decode("ascii"), "mime": "audio/wav"}).encode(
                    "utf-8"
                )
            )
        if url.endswith("/api/voices"):
            return _FakeResp(
                json.dumps(
                    {
                        "voices": [{"id": "am_fenrir", "label": "Fenrir"}],
                        "clones": [{"id": "jeremiah-new-e572bb", "name": "Jeremiah"}],
                    }
                ).encode("utf-8")
            )
        if url.endswith("/health"):
            return _FakeResp(json.dumps({"tts_loaded": True, "openvoice_available": True}).encode("utf-8"))
        return _FakeResp(b"{}")

    monkeypatch.setattr(party.urllib.request, "urlopen", _fake_urlopen)

    voices_resp = client.get("/api/party/voices")
    assert voices_resp.status_code == 200
    vdata = voices_resp.get_json()
    assert vdata["success"] is True
    assert any(v["id"] == "clone:jeremiah-new-e572bb" for v in vdata["voices"])

    client.post("/api/party/rooms", json={"code": "CAST", "name": "Jeremiah", "initials": "JER"})
    voice_set = client.post("/api/party/rooms/CAST/voice", json={"voice": "clone:jeremiah-new-e572bb"})
    assert voice_set.status_code == 200
    assert voice_set.get_json()["room"]["announcer_voice"] == "clone:jeremiah-new-e572bb"

    tvs_resp = client.get("/api/party/tvs")
    assert tvs_resp.status_code == 200
    platforms = {p["id"] for p in tvs_resp.get_json()["platforms"]}
    assert platforms == {"roku", "webos", "samsung", "cast", "android_tv"}

    for platform in ("roku", "webos", "samsung", "cast", "android_tv"):
        cast_resp = client.post(
            "/api/party/cast",
            json={
                "room_code": "CAST",
                "platform": platform,
                "entity_id": f"media_player.{platform}_tv",
                "roku_ip": "192.168.2.156" if platform == "roku" else "",
                "ha_url": "http://ha.local:8123",
                "ha_token": "secret-token",
            },
        )
        assert cast_resp.status_code == 200
        cdata = cast_resp.get_json()
        assert cdata["success"] is True
        assert cdata["platform"] == platform
        assert "/tv/CAST" in cdata["stage_url"]


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


