"""Game-show music: composed once by the audio server, looped cleanly, served as files."""
from __future__ import annotations

import array
import base64
import io
import json
import math
import wave

import numpy as np
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


def _pulse_wav(bpm=118.0, seconds=26.3, rate=32000, offset=0.21) -> bytes:
    """A click track, accent on each bar, as MusicGen-shaped 16-bit mono WAV."""
    n = int(seconds * rate)
    x = np.zeros(n)
    beat = 60.0 * rate / bpm
    k = 0
    while offset * rate + k * beat < n:
        i = round(offset * rate + k * beat)
        m = min(n - i, int(0.08 * rate))
        env = np.exp(-np.arange(m) / (0.015 * rate))
        x[i : i + m] += (0.9 if k % 4 == 0 else 0.55) * env * np.sin(2 * np.pi * 90 * np.arange(m) / rate)
        k += 1
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.round(x * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def _frames(wav_bytes: bytes) -> int:
    with wave.open(io.BytesIO(wav_bytes)) as w:
        return w.getnframes()


def test_a_bed_with_a_beat_loops_on_whole_bars():
    out = party_music.prepare_cue(_pulse_wav(bpm=118.0), loop=True)
    beats = _frames(out) / (60.0 * party_music.OUT_RATE / 118.0)
    assert abs(beats - round(beats)) < 0.03 and round(beats) % 4 == 0, beats
    with wave.open(io.BytesIO(out)) as w:
        peak = np.abs(np.frombuffer(w.readframes(w.getnframes()), "<i2")).max()
    assert peak < 32767  # the equal-power fold is rounded off, never clipped


def test_a_bed_with_no_beat_is_only_crossfaded():
    out = party_music.prepare_cue(_tone_wav(), loop=True)
    expected = int(6.0 * party_music.OUT_RATE) - int(party_music.OUT_RATE * party_music.LOOP_CROSSFADE_S)
    assert abs(_frames(out) - expected) <= 2


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
    # MusicGen's own output is kept, unlooped, beside each cue.
    for name in party_music.CUES:
        assert party_music.raw_path(name).read_bytes() == _tone_wav()


def test_a_new_shape_reshapes_beds_from_the_raw_clip_without_the_audio_server(music_dir, monkeypatch):
    monkeypatch.setattr(party_music, "_compose", lambda cue: _pulse_wav())
    party_music.generate_missing()
    lobby = party_music.cue_path("lobby").read_bytes()

    def no_gpu(cue):
        raise AssertionError("re-shaping must not compose again")

    monkeypatch.setattr(party_music, "_compose", no_gpu)
    monkeypatch.setattr(party_music, "SHAPE_VERSION", party_music.SHAPE_VERSION + 1)
    assert not party_music.is_current("lobby") and party_music.is_current("theme")
    out = party_music.generate_missing()
    assert set(out) == {n for n, c in party_music.CUES.items() if c["loop"]}
    assert all(v.startswith("re-shaped") for v in out.values())
    # Looped once, from the raw clip: the same bytes as before, not a loop of the loop.
    assert party_music.cue_path("lobby").read_bytes() == lobby
    assert party_music.is_current("lobby") and party_music.manifest()["needs_regenerate"] == []


def test_an_old_bed_without_a_raw_clip_waits_for_an_admin_regenerate(music_dir, monkeypatch):
    music_dir.mkdir(parents=True)
    old = party_music.CUES["lobby"]
    legacy = {"prompt": old["prompt"], "duration_s": old["duration_s"], "seed": old["seed"], "loop": True}
    (music_dir / "lobby.wav").write_bytes(_tone_wav())
    (music_dir / "lobby.json").write_text(json.dumps(legacy))
    composed = []
    monkeypatch.setattr(party_music, "_compose", lambda cue: composed.append(cue["seed"]) or _pulse_wav())
    party_music.generate_missing()
    assert old["seed"] not in composed  # it is not taken to the GPU on its own...
    assert party_music.manifest()["cues"]["lobby"]["ready"]  # ...it keeps playing...
    assert party_music.manifest()["needs_regenerate"] == ["lobby"]  # ...and says it is waiting
    composed.clear()
    party_music.generate_missing(["lobby"])  # what POST /api/admin/party/music/regenerate runs
    assert composed == [old["seed"]] and party_music.is_current("lobby")
    assert party_music.raw_path("lobby").is_file() and party_music.manifest()["needs_regenerate"] == []


def test_a_changed_prompt_still_recomposes_on_its_own(music_dir, monkeypatch):
    monkeypatch.setattr(party_music, "_compose", lambda cue: _pulse_wav())
    party_music.generate_missing()
    monkeypatch.setitem(party_music.CUES, "vote", {**party_music.CUES["vote"], "seed": 9999})
    seen = []
    monkeypatch.setattr(party_music, "_compose", lambda cue: seen.append(cue["seed"]) or _pulse_wav())
    out = party_music.generate_missing()
    assert list(out) == ["vote"] and out["vote"].startswith("composed") and seen == [9999]


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
