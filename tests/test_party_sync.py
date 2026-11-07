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
        room["tv_reports"] = True  # a current TV page (it reports from its first poll)
    for pid in ("a", "b"):  # both pages are current: they report from their first poll
        party.get_room_state(code, viewer_id=pid, ready="")
    party.start_match(code, player_id="a")
    return room


def _ready(code, pid):
    room = party._rooms[code]
    return party.get_room_state(code, viewer_id=pid, ready=party.step_key(room))


def _asked(room):
    """The host has finished asking: writing / voting is open and its clock is running."""
    room.pop("await_tv_open", None)
    room["open_at"] = time.time() - 0.01
    room["phase_deadline"] = room["open_at"] + float(room.get("clock_s") or 0)


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
    _asked(room)
    party.submit_answer("SYNC", player_id="a", text="now")


def test_the_clock_starts_when_the_screen_goes():
    room = _room()
    time.sleep(0.05)
    _ready("SYNC", "a")
    _ready("SYNC", "b")
    assert room["phase_started_at"] == room["gate"]["go_at"]
    # The writing clock is the whole WRITE_SECONDS, counted from when the host has asked the question.
    assert room["open_at"] >= room["gate"]["go_at"]
    assert room["phase_deadline"] - room["open_at"] == pytest.approx(party.WRITE_SECONDS)


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
    room["tv_reports"] = True
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
    room["tv_reports"] = True
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
    race = room["race"]
    assert race["flash_until"] >= race["flash_at"] + party.FLASH_MIN_S - 0.01  # still seen, however short the line
    race["flash_until"] = time.time() - 0.01
    party.get_room_state("RACE", viewer_id="a")
    assert room["race"]["index"] == 1  # then on to the next clue


def test_a_tv_that_goes_away_is_not_waited_for():
    room = _room(tv=True)
    _ready("SYNC", "a")
    _ready("SYNC", "b")
    room["stage_seen_at"] -= party.TV_PRESENCE_SECONDS + 1
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


def test_each_rounds_opening_line_is_made_before_the_round_starts(monkeypatch):
    """Reported: the TV sat on 'Get ready' ~15 s while the long opening line was made."""
    made = []
    monkeypatch.setattr(party, "synthesize_show_line", lambda text, voice=None, **k: made.append(text))

    class Now:  # run the background work at once
        def __init__(self, target, **k):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(party.threading, "Thread", Now)
    party.create_or_join_room(code="PREP", player_id="a", name="Ann")
    room = party._rooms["PREP"]
    assert room["next_prompt"]["round"] == 1 and made  # made in the lobby
    chosen = room["next_prompt"]["prompt"]["id"]
    party.start_match("PREP", player_id="a")
    assert room["current_prompt"]["id"] == chosen
    assert party.announce_line(room) in made  # the line said is the line made
    # Later rounds: made while the previous round's scores are up.
    room["schedule"][1] = "blank_o_matic"
    with party._lock:
        party._prefetch_next_prompt_unlocked(room)
        nxt = room["next_prompt"]
        assert nxt["round"] == 2
        room["phase"] = "reveal"
        party._advance_unlocked(room)
    assert room["round"] == 2 and room["current_prompt"]["id"] == nxt["prompt"]["id"]
    assert party.announce_line(room) in made


def test_every_screen_counts_down_to_the_same_server_moment():
    """Reported: the TV's countdown ran behind the phones (it counted from when its slow reply arrived)."""
    room = _room()
    _ready("SYNC", "a")
    _ready("SYNC", "b")
    _go(room)
    _asked(room)
    snap = party.get_room_state("SYNC", viewer_id="a")
    assert abs(snap["server_time"] - time.time()) < 1
    assert snap["deadline_at"] == pytest.approx(room["phase_deadline"], abs=0.01)
    stage = party.get_room_state("SYNC", stage=True)
    assert stage["deadline_at"] == snap["deadline_at"]  # one instant for every screen


def test_a_race_clue_being_read_holds_its_clock_on_every_screen():
    party.create_or_join_room(code="RACE", player_id="a", name="Ann")
    room = party._rooms["RACE"]
    room["schedule"] = ["blank_o_matic", "quick_draw", *["blank_o_matic"] * 5]
    room["stage_seen_at"] = time.time()
    room["tv_reports"] = True
    with party._lock:
        party._start_round_unlocked(room, 2)
    key = party.step_key(room)
    _ready("RACE", "a")
    party.stage_said("RACE", key, "ready")
    _go(room)
    snap = party.get_room_state("RACE", viewer_id="a")
    assert snap["race"]["listening"] and snap["race"]["seconds"] == 15.0 and snap["deadline_at"] is None
    party.stage_said("RACE", key, "end")
    room["race"]["open_at"] = time.time() - 0.01
    snap = party.get_room_state("RACE", viewer_id="a")
    assert not snap["race"]["listening"] and snap["deadline_at"] > time.time()  # now it counts, everywhere


def test_a_slow_tv_that_reports_is_still_waited_for():
    """Measured: a 2020 LG polled only every 4-10 s; at 8 s presence the screens went without it."""
    room = _room(tv=True)
    party.get_room_state("SYNC", stage=True, rtt="900", late="4000")  # a current TV page
    _ready("SYNC", "a")
    _ready("SYNC", "b")
    room["stage_seen_at"] -= 12  # twelve seconds since its last poll: slow, not gone
    party.get_room_state("SYNC", viewer_id="a")
    assert room["gate"]["go_at"] is None  # still waiting for the TV
    room["stage_seen_at"] -= party.TV_PRESENCE_SECONDS  # really gone now
    party.get_room_state("SYNC", viewer_id="a")
    assert room["gate"]["go_at"] is not None


def test_the_reveal_says_when_the_host_calls_the_result():
    room = _room()
    with party._lock:
        room["submissions"] = {pid: {"player_id": pid, "text": f"a {pid}", "is_decoy": False, "submitted_at": 0}
                               for pid in ("a", "b")}
        room["tv_on"] = None
        party._build_voting_options_unlocked(room)
        party._score_round_unlocked(room)
    snap = party.serialize_room(room, "a", stage=True)
    assert snap["phase"] == "reveal" and snap["result_at_s"] == pytest.approx(party.reveal_result_at(room))


def test_every_round_shows_its_scoreboard_even_when_the_host_skips():
    """Reported: tapping 'Start next round now' during the answers skipped the scores every round."""
    room = _room()
    with party._lock:
        room["submissions"] = {pid: {"player_id": pid, "text": f"a {pid}", "is_decoy": False, "submitted_at": 0}
                               for pid in ("a", "b")}
        party._build_voting_options_unlocked(room)
        party._score_round_unlocked(room)
    rnd = room["round"]
    _ready("SYNC", "a")
    _ready("SYNC", "b")
    _go(room)  # the reveal is on screen
    snap = party.advance_phase("SYNC", player_id="a", expect={"phase": "reveal", "round": rnd})
    assert room["phase"] == "reveal" and room["round"] == rnd  # not on to the next round
    assert snap["phase_age_s"] >= snap["scores_at_s"]  # every screen is now on the scoreboard
    party.advance_phase("SYNC", player_id="a", expect={"phase": "reveal", "round": rnd})
    assert room["phase"] == "reveal"  # it stays up its minimum
    room["phase_started_at"] -= party.SCOREBOARD_MIN_SECONDS + 1  # the scores have been seen
    party.advance_phase("SYNC", player_id="a", expect={"phase": "reveal", "round": rnd})
    assert room["round"] == rnd + 1 or room["phase"] != "reveal"


def test_nobody_writes_or_votes_until_the_host_has_asked_the_question():
    """Reported: the box, the Send button and the clock came up while the host was still asking."""
    room = _room(tv=True)
    party.get_room_state("SYNC", stage=True, rtt="50", late="0")
    _ready("SYNC", "a")
    _ready("SYNC", "b")
    key = party.step_key(room)
    party.stage_said("SYNC", key, "ready")
    _go(room)
    snap = party.get_room_state("SYNC", viewer_id="a")
    assert snap["listening"] and snap["clock_s"] == party.WRITE_SECONDS and snap["deadline_at"] is None
    assert snap["prompt"]["prompt"]  # the question is on screen while it is read
    with pytest.raises(ValueError, match="Listen"):
        party.submit_answer("SYNC", player_id="a", text="too soon")
    t0 = time.time()
    assert party.stage_said("SYNC", key, "end")  # the host has finished asking
    assert room["open_at"] == pytest.approx(t0 + party.OPEN_LEAD_S, abs=0.2)
    assert room["phase_deadline"] == pytest.approx(room["open_at"] + party.WRITE_SECONDS)  # the whole clock, from then
    room["open_at"] = time.time() - 0.01
    party.submit_answer("SYNC", player_id="a", text="now")


def test_without_a_tv_writing_opens_when_the_line_has_been_said():
    room = _room()
    _ready("SYNC", "a")
    _ready("SYNC", "b")
    go = room["gate"]["go_at"]
    line = party.announce_line(room)
    expected = go + min(party.PROMPT_READ_MAX_S, party.speech_seconds(line, room.get("voice")))
    assert room["open_at"] == pytest.approx(expected, abs=0.05)
    assert room["phase_deadline"] == pytest.approx(room["open_at"] + party.WRITE_SECONDS)


def test_a_tap_during_the_reveals_get_ready_still_skips_to_the_scores():
    room = _room()
    with party._lock:
        room["submissions"] = {pid: {"player_id": pid, "text": f"a {pid}", "is_decoy": False, "submitted_at": 0}
                               for pid in ("a", "b")}
        party._build_voting_options_unlocked(room)
        party._score_round_unlocked(room)
    party.advance_phase("SYNC", player_id="a", expect={"phase": "reveal", "round": room["round"]})  # still waiting
    assert room.get("skip_to_scores")
    _ready("SYNC", "a")
    _ready("SYNC", "b")
    go = room["gate"]["go_at"]
    assert room["phase_started_at"] == pytest.approx(go - party.scores_at(room))  # it goes straight to the scores


def test_an_old_tv_page_that_never_reports_does_not_set_the_pace():
    room = _room()
    room["stage_seen_at"] = time.time()  # polling, but never reports (a page from before this version)
    _ready("SYNC", "a")
    _ready("SYNC", "b")
    assert room["gate"]["go_at"] is not None


def test_the_tv_is_told_what_the_room_still_needs_from_it():
    """One lost report must not stall the room: the TV re-sends what the room still waits for."""
    room = _room(tv=True)
    _ready("SYNC", "a")
    _ready("SYNC", "b")
    assert party.get_room_state("SYNC", stage=True)["tv_needs"] == "ready"
    party.stage_said("SYNC", party.step_key(room), "ready")
    _go(room)
    assert party.get_room_state("SYNC", stage=True)["tv_needs"] == "end"  # writing opens on the TV's "end"
    party.stage_said("SYNC", party.step_key(room), "end")
    assert party.get_room_state("SYNC", stage=True)["tv_needs"] == ""


def test_hidden_reveal_does_not_give_away_the_points():
    room = _room()
    with party._lock:
        room["submissions"] = {pid: {"player_id": pid, "text": f"a {pid}", "is_decoy": False, "submitted_at": 0}
                               for pid in ("a", "b")}
        party._build_voting_options_unlocked(room)
        room["votes"] = {"a": next(o["option_id"] for o in room["options"] if "b" in (o.get("authors") or []))}
        party._score_round_unlocked(room)
    snap = party.get_room_state("SYNC", viewer_id="b")
    assert snap["waiting"] and all(p["round_delta"] == 0 for p in snap["players"])
    assert next(p for p in snap["players"] if p["name"] == "Ben")["score"] == 0


def _live_race():
    party.create_or_join_room(code="RACE", player_id="a", name="Ann")
    room = party._rooms["RACE"]
    room["schedule"] = ["blank_o_matic", "quick_draw", *["blank_o_matic"] * 5]
    room["stage_seen_at"], room["tv_reports"] = time.time(), True
    with party._lock:
        party._start_round_unlocked(room, 2)
    return room


def test_a_race_clue_gets_its_full_time_however_long_the_gate_took():
    """Review: a 6 s wait at "Get ready" made a 15 s Quick Draw clue 21.5 s long."""
    room = _live_race()
    room["race"]["started_at"] -= 6  # the screen waited six seconds before it went
    _ready("RACE", "a")
    party.stage_said("RACE", party.step_key(room), "ready")
    _go(room)
    party.stage_said("RACE", party.step_key(room), "end")
    race = room["race"]
    assert race["deadline"] - race["open_at"] == pytest.approx(party.party_race.ITEM_SECONDS["quick_draw"])


def test_no_hour_long_countdown_during_an_answer_flash():
    """Review: phones showed ~3600 during a flash that waits on the TV."""
    room = _live_race()
    _ready("RACE", "a")
    party.stage_said("RACE", party.step_key(room), "ready")
    _go(room)
    party.stage_said("RACE", party.step_key(room), "end")
    room["race"]["open_at"] = time.time() - 0.01
    party.race_answer("RACE", "a", 0, room["race"]["items"][0]["answer"])
    assert room["race"]["await_tv"] == "flash"
    room["gate"]["go_at"] = time.time() - 0.01
    snap = party.get_room_state("RACE", viewer_id="a")
    assert snap["deadline_at"] is None and snap["seconds_remaining"] == 0


def test_the_opening_line_does_not_change_once_its_clip_is_measured():
    """Review: measuring the clip made ahead flipped the line to its shorter version."""
    item = max(party.PROMPT_BANK["parable_remix"], key=lambda p: len(p["announcer"]))
    room = {"phase": "writing", "match_id": "m", "round": 2, "lines_told": {},
            "current_prompt": {**item, "kind": "parable_remix", "context": party.party_scripture.story_context(item)}}
    before = party.announce_line(room)
    party._spoken_seconds[(party.DEFAULT_ANNOUNCER_VOICE, before)] = 99.0  # measured long
    assert party.announce_line(room) == before
