"""Race rounds: Quick Draw, Lightning, Pair Up and In Order."""

from __future__ import annotations

import random

import pytest

import arcade.app as arcade_app
from arcade import party, party_race
from arcade.party_race_content import IN_ORDER, LIGHTNING, PAIR_UP, QUICK_DRAW, RACE_ROUND_META


@pytest.fixture(autouse=True)
def clear_party_state():
    with party._lock:
        party._rooms.clear()
    yield
    with party._lock:
        party._rooms.clear()


# --- content ---------------------------------------------------------------

def test_content_is_well_formed():
    assert set(RACE_ROUND_META) == set(party_race.RACE_KINDS)
    for meta in RACE_ROUND_META.values():
        assert set(meta) == set(party.ROUND_META["blank_o_matic"]) and meta["multiplier"] == 1
    ids = [it["id"] for pool in (QUICK_DRAW, LIGHTNING, PAIR_UP, IN_ORDER) for it in pool]
    bank = {it["id"] for items in party.PROMPT_BANK.values() for it in items}
    assert len(ids) == len(set(ids)) and not set(ids) & bank
    for it in QUICK_DRAW:
        assert len(it["options"]) == 4 and len(set(it["options"])) == 4 and 0 <= it["answer"] < 4, it["id"]
    for it in LIGHTNING:
        assert isinstance(it["true"], bool) and it["statement"], it["id"]
    for it in PAIR_UP:
        assert len(it["pairs"]) == 5, it["id"]
        assert len({a for a, _ in it["pairs"]}) == 5 and len({b for _, b in it["pairs"]}) == 5, it["id"]
    for it in IN_ORDER:
        assert len(it["items"]) == 5 and len(set(it["items"])) == 5, it["id"]
    assert len(QUICK_DRAW) >= 60 and len(LIGHTNING) >= 60 and len(PAIR_UP) >= 25 and len(IN_ORDER) >= 25


# --- the engine on its own --------------------------------------------------

def _race(kind, seed=1):
    rng = random.Random(seed)
    race = party_race.new_race(kind, party_race.choose(kind, {}, set(), rng), rng)
    party_race.start_item(race, 100.0, 0.0, [], rng)
    return race, rng


def test_quick_draw_fastest_right_answer_scores_most_and_wrong_costs():
    race, rng = _race("quick_draw")
    item = race["items"][0]
    wrong = (item["answer"] + 1) % 4
    party_race.answer(race, "slow", 0, item["answer"], 105.0)
    party_race.answer(race, "fast", 0, item["answer"], 102.0)
    party_race.answer(race, "oops", 0, wrong, 101.0)
    party_race.answer(race, "fast", 0, wrong, 103.0)  # the first answer stands
    assert not party_race.tick(race, 106.0, ["slow", "fast", "oops"], 0.0, [], rng)
    assert race["flash_until"] == pytest.approx(106.0 + party_race.FLASH_SECONDS)
    deltas, events = party_race.score(race)
    assert deltas == {"fast": 1000, "slow": 750, "oops": -250}
    assert any(e["type"] == "race_first" and e["player_id"] == "fast" for e in events)


def test_race_runs_through_every_clue_then_ends():
    race, rng = _race("lightning")
    now = 100.0
    for _ in range(len(race["items"])):
        assert not race["flash_until"]
        now = race["deadline"] + party_race.GRACE_SECONDS
        done = party_race.tick(race, now, ["p"], 0.0, [], rng)
        if done:
            break
        now = race["flash_until"]
        assert not party_race.tick(race, now, ["p"], 0.0, [], rng)
    assert race["done"] and race["index"] == len(race["items"]) - 1


def test_a_closed_clue_takes_no_more_answers():
    race, rng = _race("quick_draw")
    party_race.tick(race, race["deadline"] + 5, ["p"], 0.0, [], rng)
    with pytest.raises(ValueError):
        party_race.answer(race, "p", 0, 0, race["deadline"] + 5)


@pytest.mark.parametrize("kind", ["pair_up", "in_order"])
def test_boards_score_each_part_and_a_bonus_for_perfect(kind):
    race, _ = _race(kind)
    right = list(race["items"][0]["answer"])
    half = list(right)
    half[0], half[1] = half[1], half[0]
    party_race.answer(race, "perfect", 0, right, 110.0)
    party_race.answer(race, "close", 0, half, 105.0)
    deltas, _ = party_race.score(race)
    assert deltas["perfect"] == 5 * party_race.PART_POINTS + 1000
    assert deltas["close"] == 3 * party_race.PART_POINTS


@pytest.mark.parametrize("kind", party_race.RACE_KINDS)
def test_boards_and_choices_are_shuffled_but_answer_is_kept(kind):
    rng = random.Random(7)
    raw = party_race.POOLS[kind][0]
    item = party_race._item(kind, raw, rng)
    if kind == "quick_draw":
        assert item["options"][item["answer"]] == raw["options"][raw["answer"]]
    elif kind == "pair_up":
        assert [item["right"][r] for r in item["answer"]] == [b for _, b in raw["pairs"]]
        assert item["answer"] != sorted(item["answer"])
    elif kind == "in_order":
        assert [item["items"][i] for i in item["answer"]] == raw["items"]
    pub = party_race.public_item(item, reveal=False)
    assert "answer" not in pub and "note" not in pub


def test_bad_choices_are_refused():
    race, _ = _race("pair_up")
    for bad in (None, 3, [0, 0, 1, 2, 3], [0, 1, 2], ["a"] * 5, [True, 0, 1, 2, 3]):
        with pytest.raises(ValueError):
            party_race.answer(race, "p", 0, bad, 101.0)


# --- in a room ---------------------------------------------------------------

def _room_in_race(kind, humans=3, monkeypatch=None):
    for i in range(humans):
        party.create_or_join_room(code="RACE", player_id=f"p{i}", name=f"P{i}", initials=f"P{i}")
    room = party._rooms["RACE"]
    room["schedule"] = ["blank_o_matic", kind, *["blank_o_matic"] * 5]
    with party._lock:
        party._ensure_bots_unlocked(room)
        party._start_round_unlocked(room, 2)
    return room


def test_room_plays_a_quick_draw_round_end_to_end():
    room = _room_in_race("quick_draw")
    assert room["phase"] == "racing"
    snap = party.serialize_room(room, "p0")
    assert snap["race"]["kind"] == "quick_draw" and "answer" not in snap["race"]["item"]
    assert snap["step"] == "0" and "Quick Draw" in snap["announce"]
    item = room["race"]["items"][0]
    party.race_answer("RACE", "p0", 0, item["answer"])
    party.race_answer("RACE", "p1", 0, item["answer"])
    snap = party.race_answer("RACE", "p2", 0, (item["answer"] + 1) % 4)
    assert snap["race"]["flash"] and snap["step"] == "0f"
    assert snap["race"]["item"]["answer"] == item["answer"]
    assert snap["race"]["right"] == ["P0", "P1"] and snap["race"]["mine"]["correct"] is False
    assert "P0 was fastest" in snap["announce"]
    # Run the clock out on the rest.
    with party._lock:
        for _ in range(40):
            race = room["race"]
            if room["phase"] != "racing":
                break
            party._tick_race_unlocked(room, (race["flash_until"] or race["deadline"] + 2) + 0.01)
    assert room["phase"] == "reveal"
    assert room["players"]["p0"]["score"] >= 1000 and room["players"]["p2"]["score"] == 0  # never below zero
    assert len(room["reveal"]["race"]) == 5 and room["reveal"]["race"][0]["right"][:2] == ["P0", "P1"]
    party.serialize_room(room, "p0")  # the reveal serializes


def test_bots_take_part_in_a_small_race():
    room = _room_in_race("lightning", humans=1)
    bots = [p for p in room["players"].values() if p.get("is_bot")]
    assert len(bots) == 2
    with party._lock:
        party._tick_race_unlocked(room, room["race"]["deadline"] + 2)
    assert all(b["id"] in room["race"]["answers"][0] for b in bots)


def test_race_route_and_permissions():
    room = _room_in_race("pair_up", humans=1)
    arcade_app.app.config["TESTING"] = True
    c = arcade_app.app.test_client()
    answer = room["race"]["items"][0]["answer"]
    bot = next(p["id"] for p in room["players"].values() if p.get("is_bot"))
    assert c.post("/api/party/rooms/RACE/race", json={"player_id": bot, "item": 0, "choice": answer}).status_code == 403
    assert c.post("/api/party/rooms/RACE/race", json={"player_id": "p0", "item": 0, "choice": [9]}).status_code == 400
    assert c.post("/api/party/rooms/NOPE/race", json={"player_id": "p0", "item": 0, "choice": answer}).status_code == 404
    r = c.post("/api/party/rooms/RACE/race", json={"player_id": "p0", "item": 0, "choice": answer})
    assert r.status_code == 200 and r.get_json()["room"]["phase"] == "reveal"  # one board: the round is over


def test_every_match_has_a_race_round():
    for _ in range(50):
        sched = party.new_match_schedule()
        races = [k for k in sched if party_race.is_race(k)]
        assert 1 <= len(races) <= 2 and len(set(races)) == len(races)
        assert sched[0] == party.MATCH_OPENER and sched[-1] == party.MATCH_FINALE and "truth_or_bluff" in sched
