"""Jubilee!'s table-wide rules: Feast Day, the Year of Jubilee, the Sword Drill,
Kingdom victory, the Herald's big moments, reactions, stand-ins for absent
phones, the smarter scribes and the awards ceremony.

Hermetic like ``test_jubilee.py``: rooms are driven through the engine's own
functions (holding ``_lock`` where a test reaches into a room), dice are
pinned through ``secrets.randbelow``.
"""

import json
import time

import pytest

from arcade import jubilee


@pytest.fixture(autouse=True)
def _clean_rooms(monkeypatch):
    jubilee._clear_rooms()
    # The tests play at once: no TV pacing between the scribes' steps.
    monkeypatch.setattr(jubilee, "AI_PACE_WHEN", "tv")
    yield
    jubilee._clear_rooms()


def _pin_dice(monkeypatch, *faces):
    seq = iter(faces)
    monkeypatch.setattr(jubilee.secrets, "randbelow", lambda n: next(seq, 2) - 1)


def _game(rules=None, humans=1):
    payload = {"name": "Mum", "token": "dove", **(rules or {})}
    made = jubilee.create_room(payload)
    code, pid = made["room"]["code"], made["player_id"]
    pids = [pid]
    for name in ("Dad", "Kid", "Gran")[: humans - 1]:
        pids.append(jubilee.join_room(code, {"name": name})["player_id"])
    jubilee.start_match(code, pid)
    return code, pids, jubilee._rooms[code]


def _own(room, seat, *indices):
    for i in indices:
        room["props"][i]["owner"] = seat


# --------------------------------------------------------------------------- #
# Rules: per-ruleset defaults, switchable either way
# --------------------------------------------------------------------------- #
def test_family_fun_starts_with_the_table_wide_rules():
    rules = jubilee._build_rules({"ruleset": "familyfun"})
    assert rules["jubilee_year"] and rules["feast"] and rules["sword_drill"]
    assert not rules["kingdom_win"]


def test_quick_game_starts_with_kingdom_victory_and_classic_with_none():
    assert jubilee._build_rules({"ruleset": "quick"})["kingdom_win"]
    classic = jubilee._build_rules({})
    assert not any(classic[k] for k in ("jubilee_year", "feast", "sword_drill", "kingdom_win"))


def test_a_ruleset_default_can_be_switched_off():
    rules = jubilee._build_rules({"ruleset": "familyfun", "sword_drill": False, "feast": "off"})
    assert not rules["sword_drill"] and not rules["feast"] and rules["jubilee_year"]


# --------------------------------------------------------------------------- #
# Feast Day
# --------------------------------------------------------------------------- #
def test_sabbath_rest_offers_a_feast_on_your_own_land(monkeypatch):
    code, (pid,), room = _game({"feast": True})
    with jubilee._lock:
        _own(room, 0, 1, 3)
        room["players"][0]["position"] = 16
    _pin_dice(monkeypatch, 1, 3)                    # 16 + 4 = 20, Sabbath Rest
    out = jubilee.roll(code, pid)["room"]
    assert out["pending"]["kind"] == "feast"
    assert {c["space"] for c in out["pending"]["choices"]} == {1, 3}
    with pytest.raises(ValueError):
        jubilee.end_turn(code, pid)                 # decide first
    out = jubilee.feast(code, pid, 3)["room"]
    assert out["feast"] == 3 and out["feast_name"] == "Nazareth" and out["pending"] is None
    assert any(e["kind"] == "feast" for e in out["events"])


def test_a_feast_doubles_the_rent_there_and_moves_with_the_next_one():
    _code, _pids, room = _game({"feast": True})
    with jubilee._lock:
        _own(room, 1, 1)
        base = jubilee._rent_for(room, 1, [3, 4])
        room["feast"] = 1
        assert jubilee._rent_for(room, 1, [3, 4]) == base * 2
        _own(room, 1, 3)
        jubilee._proclaim_feast(room, 1, 3)
        assert room["feast"] == 3 and jubilee._rent_for(room, 3, [3, 4]) == jubilee._base_rent(room, 3, [3, 4]) * 2
        assert jubilee._rent_for(room, 1, [3, 4]) == jubilee._base_rent(room, 1, [3, 4])


def test_a_feast_cannot_be_kept_on_someone_elses_land(monkeypatch):
    code, (pid,), room = _game({"feast": True})
    with jubilee._lock:
        _own(room, 0, 1)
        _own(room, 1, 3)
        room["players"][0]["position"] = 16
    _pin_dice(monkeypatch, 1, 3)
    jubilee.roll(code, pid)
    with pytest.raises(ValueError):
        jubilee.feast(code, pid, 3)
    out = jubilee.feast(code, pid, None)["room"]    # resting without one is allowed
    assert out["feast"] is None and out["pending"] is None


def test_no_feast_is_offered_to_someone_who_owns_nothing(monkeypatch):
    code, (pid,), room = _game({"feast": True})
    with jubilee._lock:
        room["players"][0]["position"] = 16
    _pin_dice(monkeypatch, 1, 3)
    assert jubilee.roll(code, pid)["room"]["pending"] is None


def test_a_scribe_keeps_the_feast_where_it_hurts_most():
    _code, _pids, room = _game({"feast": True})
    with jubilee._lock:
        _own(room, 1, 1, 3, 39)
        assert jubilee._ai_feast_choice(room, 1, [1, 3, 39]) == 39


# --------------------------------------------------------------------------- #
# The Year of Jubilee
# --------------------------------------------------------------------------- #
def _end_round(room):
    """Bring every seat to one turn short of a seventh round and pass the last one."""
    n = len(room["players"])
    room["turn_counts"] = [jubilee.JUBILEE_EVERY] * (n - 1) + [jubilee.JUBILEE_EVERY - 1]
    jubilee._advance_turn(room, n - 1)


def test_the_year_of_jubilee_frees_prisoners_and_restores_the_poorest():
    _code, _pids, room = _game({"jubilee_year": True})
    with jubilee._lock:
        rich, poor = room["players"][0], room["players"][1]
        rich["cash"], poor["cash"] = 3000, 40
        room["players"][2]["cash"] = 900
        _own(room, 1, 1, 3)
        room["props"][1]["mortgaged"] = room["props"][3]["mortgaged"] = True
        rich["jail"] = {"turns": 1}
        _end_round(room)
        assert rich["jail"] is None
        assert not room["props"][1]["mortgaged"] and not room["props"][3]["mortgaged"]
        assert poor["cash"] == 40 + jubilee.JUBILEE_GLEANING
        assert room["last_jubilee"]["poorest"] == 1
        snap = jubilee._snapshot_unlocked(room)
    assert snap["events"][-1]["kind"] == "jubilee"
    assert snap["jubilee"]["count"] == 1 and snap["jubilee"]["in_rounds"] == jubilee.JUBILEE_EVERY


def test_no_jubilee_without_the_rule():
    _code, _pids, room = _game()
    with jubilee._lock:
        room["players"][1]["cash"] = 40
        _end_round(room)
        assert room["players"][1]["cash"] == 40 and not room.get("jubilees")
        assert jubilee._snapshot_unlocked(room)["jubilee"] is None


def test_the_jubilee_countdown_counts_full_rounds():
    _code, _pids, room = _game({"jubilee_year": True})
    with jubilee._lock:
        room["turn_counts"] = [5, 5, 4]
        info = jubilee._snapshot_unlocked(room)["jubilee"]
    assert info == {"every": 7, "after_round": 7, "in_rounds": 3, "count": 0}


# --------------------------------------------------------------------------- #
# The Sword Drill
# --------------------------------------------------------------------------- #
def _land_on_scripture(monkeypatch, code, pid, room):
    with jubilee._lock:
        room["players"][0]["position"] = 13
    _pin_dice(monkeypatch, 1, 3)                    # 13 + 4 = 17, SCRIPTURE
    return jubilee.roll(code, pid)["room"]


def test_scripture_opens_a_drill_for_the_whole_table(monkeypatch):
    code, (mum, _dad), room = _game({"sword_drill": True}, humans=2)
    out = _land_on_scripture(monkeypatch, code, mum, room)
    d = out["drill"]
    assert d and d["clue"] and len(d["options"]) >= 2 and d["seat"] == 0
    assert "answer" not in d and "ai_plan" not in d and "qid" not in d   # never leaked
    assert out["pending"]["kind"] == "drill" and out["last_card"] is None
    with pytest.raises(ValueError):
        jubilee.end_turn(code, mum)                 # the table answers first


def test_the_fastest_right_answer_wins_and_then_the_card_is_drawn(monkeypatch):
    code, (mum, dad), room = _game({"sword_drill": True}, humans=2)
    _land_on_scripture(monkeypatch, code, mum, room)
    with jubilee._lock:
        answer = jubilee._DRILL_BY_ID[room["drill"]["qid"]]["answer"]
        room["drill"]["ai_plan"] = {}               # the scribe sits this one out
        cash = [p["cash"] for p in room["players"]]
    wrong = (answer + 1) % len(jubilee._DRILL_BY_ID[room["drill"]["qid"]]["options"])
    out = jubilee.drill_answer(code, dad, answer)["room"]
    assert out["drill"]["answered"] == [1] and out["drill"]["you_answered"] == answer
    with pytest.raises(ValueError):
        jubilee.drill_answer(code, dad, answer)     # once each
    out = jubilee.drill_answer(code, mum, wrong)["room"]   # the last one here: it closes
    assert out["drill"] is None and (out["pending"] is None or out["pending"]["kind"] != "drill")
    res = {r["seat"]: r for r in out["last_drill"]["results"]}
    assert res[1]["correct"] and res[1]["prize"] == jubilee.DRILL_PRIZE
    assert not res[0]["correct"] and res[0]["prize"] == 0
    assert out["last_drill"]["answer"] == answer
    assert out["last_card"] and out["last_card"]["deck"] == "scripture"   # the card follows
    assert out["players"][1]["cash"] >= cash[1] + jubilee.DRILL_PRIZE - 200  # (a card may cost)


def test_the_drill_closes_on_the_clock(monkeypatch):
    code, (mum, _dad), room = _game({"sword_drill": True}, humans=2)
    _land_on_scripture(monkeypatch, code, mum, room)
    with jubilee._lock:
        room["drill"]["deadline"] = time.time() - 1
        jubilee._maybe_ai_unlocked(room)
        assert room.get("drill") is None and room["last_drill"]


def test_scribes_answer_the_drill_too(monkeypatch):
    code, (mum,), room = _game({"sword_drill": True})
    _land_on_scripture(monkeypatch, code, mum, room)
    with jubilee._lock:
        answer = jubilee._DRILL_BY_ID[room["drill"]["qid"]]["answer"]
        for plan in room["drill"]["ai_plan"].values():
            plan["choice"], plan["at"] = answer, time.time()
    out = jubilee.drill_answer(code, mum, answer)["room"]
    seats = {r["seat"] for r in out["last_drill"]["results"] if r["correct"]}
    assert seats == {0, 1, 2}
    assert sum(r["prize"] for r in out["last_drill"]["results"]) == jubilee.DRILL_PRIZE + 2 * jubilee.DRILL_RIGHT


def test_drill_questions_come_from_the_checked_bank_without_repeats(monkeypatch):
    _code, (_mum,), room = _game({"sword_drill": True})
    seen = set()
    with jubilee._lock:
        for _ in range(30):
            jubilee._drill_open(room, 0, [1, 3])
            assert room["drill"]["qid"] not in seen
            seen.add(room["drill"]["qid"])
            room["drill"] = None
    assert len(jubilee._DRILL_BANK) > 150
    assert all(q["note"] for q in jubilee._DRILL_BANK)


# --------------------------------------------------------------------------- #
# Kingdom victory and the Herald
# --------------------------------------------------------------------------- #
def test_three_kingdoms_win_the_game():
    _code, (_pid,), room = _game({"kingdom_win": True})
    with jubilee._lock:
        _own(room, 0, *jubilee._GROUPS["brown"], *jubilee._GROUPS["lightblue"])
        assert not jubilee._maybe_finish(room)
        _own(room, 0, *jubilee._GROUPS["pink"])
        assert jubilee._maybe_finish(room)
        assert room["winner"] == 0 and room["result"] == "kingdom"


def test_completing_a_group_is_announced_once():
    _code, (_pid,), room = _game()
    with jubilee._lock:
        _own(room, 0, *jubilee._GROUPS["brown"])
        jubilee._note_kingdoms(room, 0)
        jubilee._note_kingdoms(room, 0)
        kingdoms = [e for e in room["events"] if e["kind"] == "kingdom"]
    assert len(kingdoms) == 1 and "the Hometowns" in kingdoms[0]["title"] and kingdoms[0]["text"]


def test_a_hotel_and_a_big_rent_are_big_moments():
    _code, (_pid,), room = _game()
    with jubilee._lock:
        _own(room, 1, 37, 39)
        room["props"][37]["houses"] = room["props"][39]["houses"] = 4
        room["players"][1]["cash"] = 5000
        jubilee._build_building(room, 1, 39)
        room["players"][0]["position"] = 39
        jubilee._resolve_landing(room, 0, [3, 4])
        kinds = [e["kind"] for e in room["events"]]
        assert "hotel" in kinds and "big_rent" in kinds
        assert room["stats"][1]["rent_in"] == 1500 or room["pending"]   # paid, or owed as a debt


def test_the_herald_does_not_repeat_itself_until_the_pool_is_spent():
    room = {"players": []}
    pool = jubilee.HERALD["big_rent"]
    said = [jubilee._herald(room, "big_rent") for _ in pool]
    assert sorted(said) == sorted(pool)


# --------------------------------------------------------------------------- #
# Reactions
# --------------------------------------------------------------------------- #
def test_reactions_reach_every_screen_and_are_rate_limited():
    code, (mum, dad), _room = _game(humans=2)
    out = jubilee.react(code, dad, "😂")["room"]
    assert out["reactions"][-1]["emoji"] == "😂" and out["reactions"][-1]["seat"] == 1
    with pytest.raises(ValueError):
        jubilee.react(code, dad, "🎉")              # too soon
    with pytest.raises(ValueError):
        jubilee.react(code, mum, "💩")              # not on the list
    assert jubilee.get_room(code, mum)["reactions"][-1]["emoji"] == "😂"


# --------------------------------------------------------------------------- #
# Stand-ins for absent phones
# --------------------------------------------------------------------------- #
def test_a_scribe_plays_the_turn_of_someone_whose_phone_went_away():
    _code, (mum, _dad), room = _game(humans=2)
    with jubilee._lock:
        room["turn"] = 1
        room["players"][1]["last_seen"] = time.time() - jubilee.STAND_IN_SECONDS - 5
        jubilee._maybe_ai_unlocked(room)
        assert room["turn"] != 1                    # the turn was played and passed
        assert any("away" in line for line in room["log"])
        snap = jubilee._snapshot_unlocked(room, mum)
    assert snap["players"][1]["away"] is True


def test_nobody_plays_for_the_absent_when_nobody_is_watching():
    _code, (_mum,), room = _game()
    with jubilee._lock:
        room["players"][0]["last_seen"] = time.time() - 600
        jubilee._maybe_ai_unlocked(room)
        assert room["turn"] == 0 and not room["rolled"]


def test_a_stand_in_never_declares_a_persons_bankruptcy_early():
    _code, (_mum, _dad), room = _game(humans=2)
    with jubilee._lock:
        room["turn"] = 1
        room["rolled"] = True
        room["players"][1]["cash"] = 0
        room["pending"] = {"kind": "debt", "amount": 900, "creditor": 0, "reason": "rent"}
        room["players"][1]["last_seen"] = time.time() - jubilee.STAND_IN_SECONDS - 5
        jubilee._maybe_ai_unlocked(room)
        assert not room["players"][1]["bankrupt"] and room["pending"]["kind"] == "debt"
        room["players"][1]["last_seen"] = time.time() - jubilee.STAND_IN_BANKRUPT_SECONDS - 5
        jubilee._maybe_ai_unlocked(room)
        assert room["players"][1]["bankrupt"]


# --------------------------------------------------------------------------- #
# Smarter scribes
# --------------------------------------------------------------------------- #
def _offer(room, frm, to, offer=None, ask=None):
    t = {"id": "t1", "from": frm, "to": to, "status": "pending",
         "offer": {"cash": 0, "spaces": [], "jail_cards": 0, **(offer or {})},
         "ask": {"cash": 0, "spaces": [], "jail_cards": 0, **(ask or {})}}
    room["trades"].append(t)
    jubilee._ai_respond_to_trade(room, t)
    return t["status"]


def test_a_scribe_will_not_hand_over_a_kingdom_at_face_value():
    _code, _pids, room = _game()
    with jubilee._lock:
        _own(room, 0, 6, 8)
        _own(room, 1, 9)                            # Cana completes Mum's light blues
        assert _offer(room, 0, 1, offer={"cash": 130}, ask={"spaces": [9]}) == "rejected"
        assert _offer(room, 0, 1, offer={"cash": 400}, ask={"spaces": [9]}) == "accepted"
        assert room["props"][9]["owner"] == 0


def test_a_scribe_takes_a_swap_that_crowns_both_sides():
    _code, _pids, room = _game()
    with jubilee._lock:
        _own(room, 0, 11, 13, 19)                   # Mum: two pinks, and an orange
        _own(room, 1, 14, 16, 18)                   # scribe: the last pink, two oranges
        status = _offer(room, 0, 1, offer={"spaces": [19]}, ask={"spaces": [14]})
        assert status == "accepted"
        assert room["props"][14]["owner"] == 0 and room["props"][19]["owner"] == 1


def test_a_scribe_will_not_swap_a_big_kingdom_for_a_small_one():
    _code, _pids, room = _game()
    with jubilee._lock:
        _own(room, 0, 1, 31, 32)                    # Mum: a brown, and two greens
        _own(room, 1, 3, 34)                        # scribe: the other brown, the last green
        assert _offer(room, 0, 1, offer={"spaces": [1]}, ask={"spaces": [34]}) == "rejected"


def test_scribes_swap_with_each_other_so_the_board_builds():
    _code, _pids, room = _game()
    with jubilee._lock:
        _own(room, 1, 11, 13, 19)
        _own(room, 2, 14, 16, 18)
        room["players"][1]["cash"] = room["players"][2]["cash"] = 1500
        assert jubilee._ai_seek_trade(room, 1)
        assert room["props"][14]["owner"] == 1 and room["props"][19]["owner"] == 2


def test_a_comfortable_scribe_lifts_its_mortgages():
    _code, _pids, room = _game()
    with jubilee._lock:
        _own(room, 1, 1, 3)
        room["props"][1]["mortgaged"] = room["props"][3]["mortgaged"] = True
        room["players"][1]["cash"] = 1000
        jubilee._ai_unmortgage(room, 1)
        assert not room["props"][1]["mortgaged"] and not room["props"][3]["mortgaged"]
        room["props"][1]["mortgaged"] = True
        room["players"][1]["cash"] = 300            # but never down to the bone
        jubilee._ai_unmortgage(room, 1)
        assert room["props"][1]["mortgaged"]


# --------------------------------------------------------------------------- #
# Awards, stats, and the room surviving a restart
# --------------------------------------------------------------------------- #
def test_the_finished_match_hands_out_awards():
    _code, (pid,), room = _game()
    with jubilee._lock:
        jubilee._rent_stats(room, 1, 0, 450)
        jubilee._stat(room, 2, "dungeon", 3)
        for s in (1, 2):
            jubilee._eliminate_player(room, s, None)
        jubilee._maybe_finish(room)
        snap = jubilee._snapshot_unlocked(room, pid)
    awards = {a["key"]: a for a in snap["awards"]}
    assert awards["rent_in"]["seat"] == 0 and "$450" in awards["rent_in"]["detail"]
    assert awards["dungeon"]["seat"] == 2
    assert snap["players"][0]["stats"]["rent_in"] == 450


def test_a_room_with_every_new_rule_in_play_still_saves_as_json(monkeypatch):
    code, (mum, dad), room = _game({"ruleset": "familyfun", "kingdom_win": True}, humans=2)
    _land_on_scripture(monkeypatch, code, mum, room)
    jubilee.react(code, dad, "🎉")
    with jubilee._lock:
        jubilee._proclaim_feast(room, 0, None)
        _end_round(room)
        assert json.loads(json.dumps(room))["code"] == code


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
@pytest.fixture
def client():
    import arcade.app as arcade_app

    arcade_app.app.config["TESTING"] = True
    with arcade_app.app.test_client() as c:
        yield c


def test_http_feast_drill_and_react(client, monkeypatch):
    r = client.post("/api/jubilee/rooms", json={"name": "Mum", "ruleset": "familyfun"})
    pid, code = r.get_json()["player_id"], r.get_json()["room"]["code"]
    dad = client.post("/api/jubilee/rooms", json={"name": "Dad", "code": code}).get_json()["player_id"]
    client.post(f"/api/jubilee/rooms/{code}/start", json={"player_id": pid})
    assert client.post(f"/api/jubilee/rooms/{code}/react", json={"player_id": dad, "emoji": "👏"}).status_code == 200
    assert client.post(f"/api/jubilee/rooms/{code}/react", json={"player_id": dad, "emoji": "x"}).status_code == 400
    assert client.post(f"/api/jubilee/rooms/{code}/drill", json={"player_id": dad, "choice": 0}).status_code == 400
    assert client.post(f"/api/jubilee/rooms/{code}/feast", json={"player_id": pid, "property": 1}).status_code == 400
    room = jubilee._rooms[code]
    _land_on_scripture(monkeypatch, code, pid, room)
    r = client.post(f"/api/jubilee/rooms/{code}/drill", json={"player_id": dad, "choice": 0})
    assert r.status_code == 200 and r.get_json()["room"]["drill"]["answered"] == [1]


@pytest.mark.parametrize("ruleset", ["classic", "familyfun", "quick"])
def test_every_rule_the_lobby_sends_reaches_the_room(client, ruleset):
    # The create route used to forward only the three house rules, so the
    # specials the lobby showed as on were silently replaced by the defaults.
    flags = {"sword_drill": True, "feast": False, "jubilee_year": True, "kingdom_win": True,
             "double_go": True}
    r = client.post("/api/jubilee/rooms", json={"name": "Mum", "ruleset": ruleset, **flags})
    rules = jubilee._rooms[r.get_json()["room"]["code"]]["rules"]
    assert rules["ruleset"] == ruleset
    for k, v in flags.items():
        assert rules[k] is v, k
    flipped = {k: not v for k, v in flags.items()}
    r = client.post("/api/jubilee/rooms", json={"name": "Dad", "ruleset": ruleset, **flipped})
    rules = jubilee._rooms[r.get_json()["room"]["code"]]["rules"]
    for k, v in flipped.items():
        assert rules[k] is v, k


@pytest.mark.parametrize("ruleset", ["classic", "familyfun", "quick"])
def test_every_ruleset_starts_and_the_host_can_roll(client, ruleset):
    r = client.post("/api/jubilee/rooms", json={"name": "Mum", "ruleset": ruleset}).get_json()
    pid, code = r["player_id"], r["room"]["code"]
    snap = client.post(f"/api/jubilee/rooms/{code}/start", json={"player_id": pid}).get_json()["room"]
    assert snap["status"] == "playing" and snap["turn"] == 0 and len(snap["players"]) == 3
    rolled = client.post(f"/api/jubilee/rooms/{code}/roll", json={"player_id": pid})
    assert rolled.status_code == 200 and rolled.get_json()["room"]["last_roll"]


def test_live_rooms_list_and_client_errors(client):
    r = client.post("/api/jubilee/rooms", json={"name": "Mum"}).get_json()
    live = client.get("/api/jubilee/rooms?live=1").get_json()["rooms"]
    assert any(x["code"] == r["room"]["code"] for x in live)
    for _ in range(40):
        assert client.post("/api/jubilee/client_error", json={"where": "t", "message": "boom"}).status_code == 200
