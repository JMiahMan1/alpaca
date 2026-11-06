"""The show's voices: a broadcast announcer, and a paper boy for "Extra! Extra!"."""
from __future__ import annotations

import base64
import io
import json
import math
import wave

import pytest

from arcade import announcer, party


def _speech_wav(seconds=1.5, rate=24000, f0=140.0) -> bytes:
    n = int(seconds * rate)
    # A voiced buzz with a slow amplitude contour: loud and quiet syllables.
    s = [0.6 * (0.3 + 0.7 * abs(math.sin(math.pi * i / (rate * 0.25)))) * math.sin(2 * math.pi * f0 * i / rate)
         for i in range(n)]
    return announcer.write_wav(s, rate)


def _peak_and_len(wav: bytes) -> tuple[float, int, int]:
    x, rate = announcer.read_wav(wav)
    return max(abs(v) for v in x), len(x), rate


@pytest.mark.parametrize("line, roles", [
    ("Extra! Extra! Look at the scene and write the headline!", ["paperboy", "announcer"]),
    ("EXTRA, EXTRA! Noah's neighbors called the HOA!", ["paperboy", "announcer"]),
    ("Extra! Extra! Read all about it! The walls fell!", ["paperboy", "announcer"]),
    ("Extra!", ["paperboy"]),
    ("Round time! Fill in the blank!", ["announcer"]),
    ("Fill in the blank! Even Solomon needed extra prayer!", ["announcer"]),  # "extra" mid-line is just a word
])
def test_only_an_opening_extra_extra_goes_to_the_paper_boy(line, roles):
    parts = announcer.split_show_line(line)
    assert [r for r, _ in parts] == roles
    if roles[0] == "paperboy":
        assert parts[0][1] == announcer.PAPERBOY_CALL
        if len(parts) > 1:
            assert not parts[1][1].lower().startswith(("extra", "read all"))


def test_the_announcer_is_compressed_and_loud_without_clipping():
    raw = _speech_wav()
    out = announcer.assemble([("announcer", raw)])
    peak, n, rate = _peak_and_len(out)
    assert rate == 24000 and 0.85 <= peak <= 0.9
    assert n > 1.5 * rate  # the room echo adds a short tail


def test_the_paper_boy_is_quicker_and_higher():
    raw = _speech_wav()
    boy = announcer.paperboy_fx(announcer.read_wav(raw)[0], 24000)
    # Pitched up by resampling, so the call itself is shorter (plus a street echo tail).
    assert len(boy) < 1.5 * 24000 / announcer.PAPERBOY_PITCH + 0.45 * 24000
    assert max(abs(v) for v in boy) <= 0.94


def test_show_line_asks_kokoro_for_the_paper_boy_then_the_announcer(monkeypatch):
    sent = []

    class Resp:
        status = 200

        def __init__(self, body):
            self.body = body

        def read(self):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout=None):
        sent.append(json.loads(req.data))
        return Resp(json.dumps({"audio_b64": base64.b64encode(_speech_wav(0.6)).decode()}).encode())

    with party._lock:
        party._tts_clips.clear()
    monkeypatch.setattr(party.urllib.request, "urlopen", urlopen)
    clip_id, wav = party.synthesize_show_line("Extra! Extra! Noah's neighbors called the inspector!", voice="am_onyx")
    assert [p.get("voice") for p in sent] == [announcer.PAPERBOY_VOICE, "am_onyx"]
    assert sent[0]["text"] == announcer.PAPERBOY_CALL and sent[1]["text"].startswith("Noah's")
    assert sent[1]["speed"] == pytest.approx(announcer.ANNOUNCER_SPEED)
    with wave.open(io.BytesIO(wav)) as w:
        assert w.getnchannels() == 1 and w.getnframes() > 0
    assert party.get_cached_tts_wav(clip_id) == wav
    # Served from the cache the second time.
    n = len(sent)
    assert party.synthesize_show_line("Extra! Extra! Noah's neighbors called the inspector!", voice="am_onyx")[0] == clip_id
    assert len(sent) == n


def test_without_the_paper_boy_the_announcer_still_reads_the_line(monkeypatch):
    def fake_clip(text, voice=party.DEFAULT_ANNOUNCER_VOICE, *, clone=None, speed=1.0):
        if voice == announcer.PAPERBOY_VOICE:
            return None
        return ("a" * 16, _speech_wav(0.5))

    with party._lock:
        party._tts_clips.clear()
    monkeypatch.setattr(party, "synthesize_announcer_clip", fake_clip)
    assert party.synthesize_show_line("Extra! Extra! Big news!") is not None


def test_the_host_has_patter_for_every_phase_and_everyone_hears_the_same_line():
    from arcade import party as p
    with p._lock:
        p._rooms.clear()
    p.create_or_join_room(code="HOST", player_id="a", name="Ann")
    p.create_or_join_room(code="HOST", player_id="b", name="Ben")
    lobby = p.get_room_state("HOST", "a")["announce"]
    assert "H O S T" in lobby
    p.start_match("HOST", player_id="a")
    tv, phone = p.get_room_state("HOST", stage=True)["announce"], p.get_room_state("HOST", "b")["announce"]
    assert tv == phone and tv  # one line, everywhere
    room = p._rooms["HOST"]
    prompt_line = room["current_prompt"]["announcer"]
    if prompt_line.lower().startswith("extra"):
        assert tv == prompt_line  # the paper boy opens it
    else:
        assert tv.endswith(prompt_line) and any(tv.startswith(o) for o in p.HOST_OPENERS)
    room["phase"] = "voting"
    assert p.announce_line(room) in p.HOST_VOTING
    room["phase"], room["reveal"] = "reveal", {"round_winners": [], "events": [], "announcer": "x"}
    line = p.announce_line(room)
    assert any(line.startswith(d) for d in p.HOST_DRUMROLL) and any(line.endswith(t) for t in p.HOST_TRAPPED)
    room["phase"] = "finished"
    finale = p.announce_line(room)
    assert any(finale == f.format(name="Ann", score=0) or finale == f.format(name="Ben", score=0) for f in p.HOST_FINALE)
    with p._lock:
        p._rooms.clear()
