"""AI players only at a small table, and House answers that cost points either way."""

from __future__ import annotations

import pytest

from arcade import party


@pytest.fixture(autouse=True)
def clear_party_state():
    with party._lock:
        party._rooms.clear()
    yield
    with party._lock:
        party._rooms.clear()


def _room(code: str, humans: int) -> dict:
    for i in range(humans):
        party.create_or_join_room(code=code, player_id=f"p{i}", name=f"Player{i}", initials=f"P{i}")
    room = party._rooms[code]
    with party._lock:
        party._advance_unlocked(room)  # lobby -> round 1 writing
    return room


def _to_voting(room: dict) -> None:
    with party._lock:
        for pid, p in room["players"].items():
            if not p.get("is_bot"):
                room["submissions"][pid] = {"player_id": pid, "text": f"answer from {pid}",
                                            "is_decoy": False, "submitted_at": 0}
        party._advance_unlocked(room)
    assert room["phase"] == "voting"


@pytest.mark.parametrize(("humans", "bots"), [(1, 2), (2, 1), (3, 0), (5, 0)])
def test_ai_players_sit_in_only_when_two_or_fewer_people_play(humans, bots):
    room = _room("HOUSE", humans)
    assert sum(1 for p in room["players"].values() if p.get("is_bot")) == bots


def test_without_ai_players_the_house_plays_a_second_answer():
    room = _room("HOUSE", 3)
    _to_voting(room)
    house = [o for o in room["options"] if o["kind"] == "decoy"]
    assert {o["option_id"] for o in house} == {"opt_house_decoy", "opt_house_extra"}
    extra = next(o for o in house if o["option_id"] == "opt_house_extra")
    prompt = room["current_prompt"]
    assert extra["text"] in prompt["bot_answers"] and extra["text"] != prompt["decoy"]
    assert extra["authors"] == []


def test_with_ai_players_there_is_only_the_one_house_trap():
    room = _room("HOUSE", 2)
    _to_voting(room)
    assert [o["option_id"] for o in room["options"] if o["kind"] == "decoy"] == ["opt_house_decoy"]


def test_voting_for_the_extra_house_answer_costs_points():
    room = _room("HOUSE", 3)
    _to_voting(room)
    room["players"]["p0"]["score"] = 5000
    with party._lock:
        room["votes"]["p0"] = "opt_house_extra"
        party._advance_unlocked(room)
    mult = int(room["current_prompt"]["multiplier"])
    assert room["players"]["p0"]["score"] == 5000 - 250 * mult
    assert any(e["type"] == "trap_fell" and e["player_id"] == "p0" for e in room["reveal"]["events"])


def test_house_answers_never_reach_a_phone_before_voting():
    room = _room("HOUSE", 3)
    snap = party.serialize_room(room, "p0")
    assert "bot_answers" not in (snap.get("prompt") or {})
    assert room["current_prompt"]["decoy"] not in str(snap)


def test_house_extra_skips_what_a_player_already_wrote():
    prompt = {"decoy": "D", "bot_answers": ["Same as mine", "Fresh one"], "truth": "", "truth_keywords": []}
    subs = {"p0": {"text": "same as MINE!"}}
    for _ in range(20):
        assert party._house_extra_answer(prompt, subs) == "Fresh one"
    assert party._house_extra_answer({"decoy": "D", "bot_answers": []}, {}) == ""


def test_the_same_answer_twice_is_one_option_shared_by_both():
    """Reported in review: two people typing "Fish" put two identical buttons on the ballot."""
    for i in range(3):
        party.create_or_join_room(code="DUPE", player_id=f"d{i}", name=f"D{i}")
    party.start_match("DUPE", player_id="d0")
    party.submit_answer("DUPE", player_id="d0", text="Fish")
    party.submit_answer("DUPE", player_id="d1", text="fish!")
    party.submit_answer("DUPE", player_id="d2", text="Loaves")
    room = party._rooms["DUPE"]
    with party._lock:
        if room["phase"] == "writing":
            party._advance_unlocked(room)
    fish = [o for o in room["options"] if party._normalize_compare(o["text"]) == "fish"]
    assert len(fish) == 1 and sorted(fish[0]["authors"]) == ["d0", "d1"]
    party.submit_vote("DUPE", player_id="d2", option_id=fish[0]["option_id"])
    with party._lock:
        if room["phase"] == "voting":
            party._advance_unlocked(room)
    assert room["players"]["d0"]["score"] > 0 and room["players"]["d1"]["score"] > 0


def test_nobody_can_view_the_room_as_a_scribe():
    party.create_or_join_room(code="PEEK", player_id="k0", name="K0")
    party.start_match("PEEK", player_id="k0")
    bot = next(p for p in party._rooms["PEEK"]["players"].values() if p.get("is_bot"))
    snap = party.get_room_state("PEEK", viewer_id=bot["id"])
    assert not snap.get("you") and not snap.get("my_submission")
    with pytest.raises(PermissionError):
        party.create_or_join_room(code="PEEK2", player_id="bot-apollos", name="Sneaky")
