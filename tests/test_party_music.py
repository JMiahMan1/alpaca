"""Game-show music: composed once by the audio server, looped cleanly, served as files."""
from __future__ import annotations

import array
import base64
import io
import json
import math
import wave

import pytest

import arcade.app as arcade_app
from arcade import party_music


def _tone_wav(seconds=6.0, rate=32000, channels=1) -> bytes:
    n = int(seconds * rate)
    s = array.array("h")
    for i in range(n):
        v = int(12000 * math.sin(2 * math.pi * 440 * i / rate))
        s.extend([v] * channels)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(s.tobytes())
    return buf.getvalue()


@pytest.fixture
def music_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(party_music, "MUSIC_DIR", tmp_path / "party_music")
    party_music._state.update({"thread": None, "last_attempt": 0.0, "errors": {}})
    return tmp_path / "party_music"


def test_a_bed_is_folded_into_a_seamless_loop():
    rate = 1000
    s = array.array("h", [i % 3000 for i in range(4000)])
    looped = party_music.make_loop(s, rate, crossfade_s=1.0)
    assert len(looped) == 3000  # the tail went into the head
    # Played back to back, the end runs straight on into what the tail was.
    assert looped[0] == s[3000]
    assert abs(looped[-1] - s[2999]) <= 1 and abs(s[3000] - s[2999]) <= 2999


def test_prepare_cue_is_mono_at_the_served_rate():
    out = party_music.prepare_cue(_tone_wav(channels=2), loop=True)
    with wave.open(io.BytesIO(out)) as w:
        assert w.getnchannels() == 1 and w.getframerate() == party_music.OUT_RATE
        assert w.getnframes() < 6.0 * party_music.OUT_RATE  # minus the crossfade


def test_missing_cues_are_composed_from_the_audio_server(music_dir, monkeypatch):
    sent = []

    def fake_compose(cue):
        sent.append(cue["prompt"])
        return _tone_wav()

    monkeypatch.setattr(party_music, "_compose", fake_compose)
    out = party_music.generate_missing()
    assert set(out) == set(party_music.CUES) and len(sent) == len(party_music.CUES)
    man = party_music.manifest()
    assert all(c["ready"] and c["url"].startswith(f"/api/party/music/{n}.wav") for n, c in man["cues"].items())
    # A second pass composes nothing: files are kept.
    assert party_music.generate_missing() == {}


def test_a_failing_audio_server_is_reported_not_fatal(music_dir, monkeypatch):
    def boom(cue):
        raise RuntimeError("audio server down")

    monkeypatch.setattr(party_music, "_compose", boom)
    out = party_music.generate_missing(["theme"])
    assert out["theme"].startswith("failed")
    assert party_music.manifest()["cues"]["theme"]["ready"] is False
    assert "theme" in party_music.manifest()["errors"]


def test_compose_reads_the_music_endpoint(monkeypatch):
    class Resp:
        def __init__(self, body):
            self.body = body

        def read(self):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    seen = {}

    def urlopen(req, timeout=None):
        seen["url"], seen["body"] = req.full_url, json.loads(req.data)
        return Resp(json.dumps({"audio_b64": base64.b64encode(b"RIFF....").decode()}).encode())

    monkeypatch.setenv("AUDIO_SERVER_URL", "http://audio:8082")
    monkeypatch.setattr(party_music.urllib.request, "urlopen", urlopen)
    assert party_music._compose(party_music.CUES["lobby"]) == b"RIFF...."
    assert seen["url"] == "http://audio:8082/api/music"
    assert seen["body"]["duration_s"] <= 30 and seen["body"]["seed"] == party_music.CUES["lobby"]["seed"]


def test_music_routes_serve_ready_cues_and_admin_regenerates(music_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(party_music, "ensure_background", lambda *a, **k: False)
    arcade_app.app.config["TESTING"] = True
    music_dir.mkdir(parents=True)
    (music_dir / "lobby.wav").write_bytes(party_music.prepare_cue(_tone_wav(), loop=True))
    with arcade_app.app.test_client() as c:
        man = c.get("/api/party/music").get_json()
        assert man["cues"]["lobby"]["ready"] and not man["cues"]["theme"]["ready"]
        assert c.get("/api/party/music/lobby.wav").mimetype == "audio/wav"
        assert c.get("/api/party/music/theme.wav").status_code == 404
        assert c.get("/api/party/music/../../etc.wav").status_code == 404
        assert c.post("/api/admin/party/music/regenerate").status_code == 403


def test_host_turns_music_off_for_the_room():
    from arcade import party
    with party._lock:
        party._rooms.clear()
    arcade_app.app.config["TESTING"] = True
    with arcade_app.app.test_client() as c:
        c.post("/api/party/rooms", json={"code": "TUNE", "player_id": "p1", "name": "Ann"})
        c.post("/api/party/rooms", json={"code": "TUNE", "player_id": "p2", "name": "Ben"})
        assert c.get("/api/party/rooms/TUNE?player_id=p2").get_json()["room"]["music"] is True
        assert c.post("/api/party/rooms/TUNE/music", json={"player_id": "p2", "on": False}).status_code == 403
        assert c.post("/api/party/rooms/TUNE/music", json={"player_id": "p1", "on": False}).get_json()["room"]["music"] is False
    with party._lock:
        party._rooms.clear()


def test_tv_and_phone_pages_play_the_cues_and_the_tv_audio_bug_is_gone():
    html = arcade_app.app.test_client().get("/tv").get_data(as_text=True)
    assert "/api/party/music" in html and "function setBed" in html and "duck(true)" in html
    # The WebAudio helper used to share its name with the announcer's <audio> variable.
    assert "function audio()" not in html and "function sfxCtx()" in html
    from arcade import family
    phone = family.build_html("good-word")
    assert "/api/party/music" in phone and 'id="gw-music"' in phone
