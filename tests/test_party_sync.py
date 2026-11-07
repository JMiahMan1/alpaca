"""Every screen goes everywhere at once: no device, browser or TV sees a question first.

A new screen starts closed. Each device reports it is on it (``ready`` on its poll), a live TV
reports it has the host's line in hand, and only then does the screen go, on every screen
together. Until it goes nobody is sent what is on it, and answers and votes are refused.
"""

from __future__ import annotations

import time

import pytest

import arcade.app as arcade_app
from arcade import party

pytestmark = pytest.mark.party_gate


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    with party._lock:
        party._rooms.clear()
    # No audio server in tests: the room's own line is "made" at once.
    monkeypatch.setattr(party, "_make_line_for_the_gate",
                        lambda room, key: room["gate"].__setitem__("line_ready", True))
    yield
    with party._lock:
        party._rooms.clear()


def _room(code="SYNC", tv=False):
    party.create_or_join_room(code=code, player_id="a", name="Ann")
    party.create_or_join_room(code=code, player_id="b", name="Ben")
    room = party._rooms[code]
    if tv:
        room["stage_seen_at"] = time.time()
    for pid in ("a", "b"):  # both pages are current: they report from their first poll
        party.get_room_state(code, viewer_id=pid, ready="")
    party.start_match(code, player_id="a")
    return room


def _ready(code, pid):
    room = party._rooms[code]
    return party.get_room_state(code, viewer_id=pid, ready=party.step_key(room))


def _go(room):
    """Let the go moment pass."""
    gate = room["gate"]
    assert gate["go_at"] is not None
    gate["go_at"] = time.time() - 0.01


def test_without_a_tv_no_phone_sees_the_question_until_every_phone_is_ready():
    room = _room()
    snap = party.get_room_state("SYNC", viewer_id="a")
    assert snap["waiting"] and snap["poll_ms"] < 1000
    assert not snap["prompt"].get("prompt") and not snap["announce"]  # nothing to peek at
    with pytest.raises(ValueError):
        party.submit_answer("SYNC", player_id="a", text="early")
    _ready("SYNC", "a")
    assert room["gate"]["go_at"] is None  # Ben's phone has not said it is ready
    snap = _ready("SYNC", "b")
    assert snap["go_in"] is not None and snap["waiting"]  # going, together, in a moment
    _go(room)
    snap = party.get_room_state("SYNC", viewer_id="a")
    assert not snap["waiting"] and snap["prompt"]["prompt"]
    party.submit_answer("SYNC", player_id="a", text="now")


def test_the_clock_starts_when_the_screen_goes():
    room = _room()
    length = room["phase_deadline"] - room["phase_started_at"]
    time.sleep(0.05)
    _ready("SYNC", "a")
    _ready("SYNC", "b")
    assert room["phase_started_at"] == room["gate"]["go_at"]
    assert room["phase_deadline"] - room["phase_started_at"] == pytest.approx(length)


def test_a_device_that_has_gone_is_not_waited_for():
    room = _room()
    room["players"]["b"]["last_seen"] -= party.PLAYER_AWAY_SECONDS + 1
    _ready("SYNC", "a")
    assert room["gate"]["go_at"] is not None


def test_with_a_tv_the_screen_waits_for_the_tv_too_and_the_tv_sees_it_first_to_prepare():
    room = _room(tv=True)
    _ready("SYNC", "a")
    _ready("SYNC", "b")
    assert room["gate"]["go_at"] is None  # the TV has not got its line yet
    stage = party.get_room_state("SYNC", stage=True)
    assert stage["waiting"] and stage["prompt"]["prompt"] and stage["announce"]  # the TV prepares it
    assert party.stage_said("SYNC", "old screen", "ready") is False
    assert party.stage_said("SYNC", party.step_key(room), "ready") is True
    assert room["gate"]["go_at"] is not None


def test_a_race_clue_opens_after_the_tv_has_read_it_announced_to_all_in_advance():
    party.create_or_join_room(code="RACE", player_id="a", name="Ann")
    party.create_or_join_room(code="RACE", player_id="b", name="Ben")
    room = party._rooms["RACE"]
    room["schedule"] = ["blank_o_matic", "quick_draw", *["blank_o_matic"] * 5]
    room["stage_seen_at"] = time.time()
    with party._lock:
        party._start_round_unlocked(room, 2)
    key = party.step_key(room)
    for pid in ("a", "b"):
        _ready("RACE", pid)
    party.stage_said("RACE", key, "ready")
    _go(room)
    race = room["race"]
    assert race["await_tv"] == "clue" and race["open_at"] > time.time() + 60  # closed while it is read
    with pytest.raises(ValueError):
        party.race_answer("RACE", "a", 0, race["items"][0]["answer"])
    t0 = time.time()
    assert party.stage_said("RACE", key, "end") is True
    assert t0 + party.OPEN_LEAD_S - 0.1 <= race["open_at"] <= time.time() + party.OPEN_LEAD_S + 0.1
    snap = party.get_room_state("RACE", viewer_id="a")
    assert 0 < snap["race"]["opens_in"] <= party.OPEN_LEAD_S  # every phone opens at the same moment


def test_an_answer_flash_ends_when_the_tv_has_said_it():
    party.create_or_join_room(code="RACE", player_id="a", name="Ann")
    room = party._rooms["RACE"]
    room["schedule"] = ["blank_o_matic", "quick_draw", *["blank_o_matic"] * 5]
    room["stage_seen_at"] = time.time()
    with party._lock:
        party._start_round_unlocked(room, 2)
    for _ in range(2):  # the clue, then its flash
        key = party.step_key(room)
        _ready("RACE", "a")
        party.stage_said("RACE", key, "ready")
        _go(room)
        if not room["race"].get("flash_until"):
            party.stage_said("RACE", key, "end")
            room["race"]["open_at"] = 0
            party.race_answer("RACE", "a", 0, room["race"]["items"][0]["answer"])
    assert room["race"]["await_tv"] == "flash" and room["race"]["flash_until"] > time.time() + 60
    party.stage_said("RACE", party.step_key(room), "end")
    assert room["race"]["index"] == 1  # straight on to the next clue


def test_a_tv_that_goes_away_is_not_waited_for():
    room = _room(tv=True)
    _ready("SYNC", "a")
    _ready("SYNC", "b")
    room["stage_seen_at"] -= party.STAGE_PRESENCE_SECONDS + 1
    party.get_room_state("SYNC", viewer_id="a")
    assert room["gate"]["go_at"] is not None


def test_a_video_tv_keeps_its_own_delay_and_no_gate():
    room = _room()
    room["stage_seen_at"] = room["video_stage_seen_at"] = time.time()
    assert party._gate_unlocked(room, time.time()) is None


def test_the_said_route():
    room = _room(tv=True)
    client = arcade_app.app.test_client()
    res = client.post("/api/party/rooms/SYNC/said", json={"key": party.step_key(room), "event": "ready"})
    assert res.status_code == 200 and res.get_json()["applied"] is True
    assert client.post("/api/party/rooms/NOPE/said", json={"key": "x"}).status_code == 404
    assert client.post("/api/party/rooms/SYNC/said", json={"key": "x", "event": "soon"}).status_code == 400


def test_a_screen_that_has_not_gone_never_runs_out_behind_the_waiting_page():
    """Reported: writing, voting and the reveal each expired while everyone saw 'Get ready'."""
    room = _room()
    room["phase_started_at"] -= 120  # two minutes behind the waiting page: its whole clock has passed
    room["phase_deadline"] -= 120
    party.get_room_state("SYNC", viewer_id="a", ready=party.step_key(room))
    assert room["phase"] == "writing"  # not moved on to voting unseen
    party.get_room_state("SYNC", viewer_id="b", ready=party.step_key(room))
    assert room["gate"]["go_at"] is not None and room["phase_deadline"] > time.time()  # the clock starts at the go


def test_a_page_from_before_ready_reports_is_not_waited_for():
    """Reported: a browser left open across the deploy never said it was ready, and froze the room."""
    room = _room()
    room["players"]["b"].pop("reports_ready")  # an old page: it polls but has never reported
    party.get_room_state("SYNC", viewer_id="b")
    _ready("SYNC", "a")
    assert room["gate"]["go_at"] is not None


def test_snapshots_carry_the_build_so_open_pages_reload_after_a_deploy():
    _room()
    assert party.get_room_state("SYNC", viewer_id="a")["build"] == party.BUILD and len(party.BUILD) == 12
