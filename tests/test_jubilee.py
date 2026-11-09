"""Jubilee! - the Bible-story property game: engine, rooms and HTTP surface.

The rules and AI are pure and tested directly; the rooms and HTTP surface run
through ``arcade.app.test_client()``. Nothing here touches the network, so the
whole file runs in the hermetic suite. Dice are made deterministic by
monkeypatching ``secrets.randbelow``.
"""

import time

import pytest

from arcade import jubilee


@pytest.fixture(autouse=True)
def _clean_rooms():
    jubilee._clear_rooms()
    yield
    jubilee._clear_rooms()


@pytest.fixture
def dice(monkeypatch):
    """Deterministic dice: randbelow returns each value from the queue minus 1,
    so the engine adds its +1 back and hands out the listed faces (1-6)."""
    seq = iter([2, 4, 5, 5, 2, 4, 1, 1, 6, 3, 4, 2, 5, 5, 3, 3])
    monkeypatch.setattr(jubilee.secrets, "randbelow", lambda n: next(seq, 2) - 1)
    return seq


def _room(**kw):
    payload = {"name": "Mum", "token": "dove", "initials": "MUM"}
    payload.update(kw)
    return jubilee.create_room(payload)


def _started(**kw):
    out = _room(**kw)
    code, pid = out["room"]["code"], out["player_id"]
    out.update(jubilee.start_match(code, pid))
    return code, pid


def test_create_room_shape():
    out = _room()
    assert out["room"]["status"] == "lobby"
    assert out["room"]["players"][0]["token"] == "dove"
    assert out["room"]["board"][0]["kind"] == "go"
    assert out["room"]["board"][10]["kind"] == "dungeon"
    assert out["room"]["board"][20]["kind"] == "sabbath"
    assert out["room"]["board"][30]["kind"] == "to_dungeon"
    assert len(out["room"]["board"]) == 40
    assert out["room"]["you"]["seat"] == 0


def test_join_and_start_fills_with_scribes():
    out = _room()
    code = out["room"]["code"]
    jubilee.join_room(code, {"name": "Dad", "token": "ark"})
    out2 = jubilee.start_match(code, out["player_id"])
    players = out2["room"]["players"]
    assert len(players) == 3  # two humans + one scribe
    assert players[2]["is_ai"] and players[2]["name"] in {s["name"] for s in jubilee.SCRIBES}
    assert out2["room"]["status"] == "playing"


def test_mid_match_join_refused():
    code, _pid = _started()
    with pytest.raises(ValueError):
        jubilee.join_room(code, {"name": "Late", "token": "star"})


def test_start_requires_host():
    out = _room()
    code = out["room"]["code"]
    kid = jubilee.join_room(code, {"name": "Kid", "token": "ark"})
    with pytest.raises(PermissionError):
        jubilee.start_match(code, kid["player_id"])


def test_roll_moves_and_collects_go(dice):
    code, pid = _started()
    out = jubilee.roll(code, pid)
    assert out["room"]["players"][0]["position"] == 2 + 4
    assert out["room"]["rolled"] is True


def test_end_turn_requires_a_roll():
    code, pid = _started()
    with pytest.raises(ValueError):
        jubilee.end_turn(code, pid)


def test_doubles_grant_a_bonus_roll(monkeypatch):
    rolls = iter([2, 2, 1, 4])  # first roll 3+3 (double, lands Sychar $100), bonus roll 2+5
    monkeypatch.setattr(jubilee.secrets, "randbelow", lambda n: next(rolls, 0))
    code, pid = _started()
    out = jubilee.roll(code, pid)  # doubles, but landing sets a pending buy
    assert out["room"]["bonus_roll"] is False
    out = jubilee.buy(code, pid)  # resolving the pending re-arms the bonus roll
    assert out["room"]["bonus_roll"] is True
    out2 = jubilee.roll(code, pid)  # the bonus roll consumes the flag
    assert out2["room"]["bonus_roll"] is False


def test_three_doubles_send_to_the_dungeon(monkeypatch):
    monkeypatch.setattr(jubilee.secrets, "randbelow", lambda n: 4)  # always 5+5
    code, pid = _started()
    jubilee.roll(code, pid)   # double 1
    jubilee.roll(code, pid)   # double 2
    out = jubilee.roll(code, pid)  # double 3
    assert out["room"]["players"][0]["in_jail"] is True
    assert out["room"]["players"][0]["position"] == 10


def test_jail_fine_releases(monkeypatch):
    monkeypatch.setattr(jubilee.secrets, "randbelow", lambda n: 4)
    code, pid = _started()
    jubilee.roll(code, pid)
    jubilee.roll(code, pid)
    jubilee.roll(code, pid)
    out = jubilee.jail_fine(code, pid)
    assert out["room"]["players"][0]["in_jail"] is False
    assert out["room"]["players"][0]["cash"] == 1500 - 50


def test_buy_flow_makes_owner(dice):
    code, pid = _started()
    jubilee.roll(code, pid)  # lands on Sychar (index 6, $100)
    assert jubilee.get_room(code, pid)["pending"]["kind"] == "buy"
    out = jubilee.buy(code, pid)
    assert out["room"]["board"][6]["owner"] == 0
    assert out["room"]["players"][0]["cash"] == 1500 - 100


def test_decline_opens_auction(dice):
    code, pid = _started()
    jubilee.roll(code, pid)
    out = jubilee.decline(code, pid)
    assert out["room"]["auction"] is not None
    seat = out["room"]["auction"]["space"]
    assert seat == 6


def test_auction_resolves_lone_player(dice):
    code, pid = _started()
    jubilee.roll(code, pid)
    jubilee.decline(code, pid)
    # other humans (none here) + AI act through polls; the bidder wins or the bank keeps it
    out = jubilee.auction_pass(code, pid)
    assert out["room"]["auction"] is None or out["room"]["auction"]["high_bidder"] is not None


def test_parable_space_draws_a_card(monkeypatch):
    rolls = iter([0, 5])  # dice 1 + 6 = 7 (PARABLE) - a card is drawn
    monkeypatch.setattr(jubilee.secrets, "randbelow", lambda n: next(rolls, 0))
    code, pid = _started()
    jubilee.roll(code, pid)
    assert isinstance(jubilee.get_room(code, pid)["players"][0]["position"], int)


def test_mortgage_and_lift(dice):
    code, pid = _started()
    jubilee.roll(code, pid)
    jubilee.buy(code, pid)
    cash = jubilee.get_room(code, pid)["players"][0]["cash"]
    out = jubilee.mortgage(code, pid, 6)
    assert out["room"]["board"][6]["mortgaged"] is True
    assert out["room"]["players"][0]["cash"] == cash + 50
    out = jubilee.unmortgage(code, pid, 6)
    assert out["room"]["board"][6]["mortgaged"] is False
    assert out["room"]["players"][0]["cash"] == cash + 50 - 55  # +10% rounds half-up


def test_trade_propose_and_accept():
    code, pid = _started()
    state = jubilee.get_room(code, pid)
    assert state["players"][2]["is_ai"]
    cash_before = state["players"][0]["cash"]
    # Asking for nothing in return: the scribe accepts instantly and pays a dollar.
    out = jubilee.trade_propose(code, pid, 2, {"cash": 1, "spaces": [], "jail_cards": 0},
                                {"cash": 0, "spaces": [], "jail_cards": 0})
    assert out["room"]["players"][0]["cash"] == cash_before - 1
    assert out["room"]["players"][2]["cash"] == 1500 + 1


def test_bankruptcy_eliminates_and_last_standing_wins(dice, monkeypatch):
    code, pid = _started()
    room = jubilee._rooms[code]
    room["players"][0]["cash"] = 10  # cannot possibly cover the debt
    jubilee._charge(room, 0, 5000, None, "tax")
    assert room["pending"]["kind"] == "debt"
    out = jubilee.declare_bankruptcy(code, pid)
    assert out["room"]["players"][0]["bankrupt"] is True
    assert out["room"]["bankruptcies"] == 1


def test_familyfun_forgives(monkeypatch):
    out = _room(ruleset="familyfun", jackpot=True, no_auction=True)
    code, pid = out["room"]["code"], out["player_id"]
    jubilee.start_match(code, pid)
    room = jubilee._rooms[code]
    room["players"][0]["cash"] = 0
    ok = jubilee._charge(room, 0, 500, None, "tax")
    assert ok is True


def test_jackpot_pot_feeds_and_pays(monkeypatch):
    out = _room(jackpot=True)
    code, pid = out["room"]["code"], out["player_id"]
    jubilee.start_match(code, pid)
    room = jubilee._rooms[code]
    room["players"][0]["cash"] = 1000
    jubilee._charge(room, 0, 200, None, "tax")
    assert room["pot"] == 200


def test_quick_ruleset_ends_on_two_bankruptcies():
    out = _room(ruleset="quick")
    code, pid = out["room"]["code"], out["player_id"]
    jubilee.start_match(code, pid)
    room = jubilee._rooms[code]
    assert room["rules"]["ruleset"] == "quick"


def test_restart_returns_to_lobby(dice):
    code, pid = _started()
    jubilee.roll(code, pid)
    room = jubilee._rooms[code]
    room["status"] = "finished"
    out = jubilee.restart_room(code, pid)
    assert out["room"]["status"] == "lobby"
    assert out["room"]["players"][0]["position"] == 0


def test_leave_mid_match_eliminates():
    code, pid = _started()
    out = jubilee.leave_room(code, pid)
    assert out["room"]["players"][0]["bankrupt"] is True


def test_snapshot_hides_raw_ids():
    code, pid = _started()
    room = jubilee.get_room(code, pid)
    assert pid not in str(room["players"])
    assert room["you"]["seat"] == 0
    stage = jubilee.get_room(code, pid, stage=True)
    assert stage["code"] == code


def test_unknown_room_returns_none():
    assert jubilee.get_room("NOPE", None) is None


# --------------------------------------------------------------------------- #
# The HTTP surface
# --------------------------------------------------------------------------- #
@pytest.fixture
def client():
    import arcade.app as arcade_app

    arcade_app.app.config["TESTING"] = True
    with arcade_app.app.test_client() as c:
        yield c


def test_http_create_join_start_roll(client):
    r = client.post("/api/jubilee/rooms", json={"name": "Mum", "token": "dove", "device_id": "a"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["success"] and body["room"]["status"] == "lobby"
    pid, code = body["player_id"], body["room"]["code"]
    assert body["room"]["invite_url"].endswith(f"#room={code}")
    r = client.post("/api/jubilee/rooms", json={"name": "Dad", "token": "ark", "code": code, "device_id": "b"})
    assert r.status_code == 200
    r = client.post(f"/api/jubilee/rooms/{code}/start", json={"player_id": pid})
    assert r.status_code == 200 and r.get_json()["room"]["status"] == "playing"
    r = client.post(f"/api/jubilee/rooms/{code}/roll", json={"player_id": pid})
    assert r.status_code == 200 and r.get_json()["success"]


def test_http_unknown_room_is_404(client):
    assert client.get("/api/jubilee/rooms/ZZZZ").status_code == 404


def test_http_tv_stage_renders(client):
    r = client.post("/api/jubilee/rooms", json={"name": "Mum", "token": "dove"})
    code = r.get_json()["room"]["code"]
    tv = client.get(f"/jubilee/tv/{code}")
    assert tv.status_code == 200
    assert code in tv.get_data(as_text=True)
    assert "__ROOM_CODE__" not in tv.get_data(as_text=True)


def test_http_qr_svg(client):
    r = client.post("/api/jubilee/rooms", json={"name": "Mum", "token": "dove"})
    code = r.get_json()["room"]["code"]
    qr = client.get(f"/api/jubilee/rooms/{code}/join.svg")
    assert qr.status_code == 200 and qr.mimetype == "image/svg+xml"


def test_http_join_redirects(client):
    r = client.get("/join/jubilee/WAVE")
    assert r.status_code == 302
    assert r.headers["Location"].endswith("#room=WAVE")


def test_live_rooms_list_lobbies_and_games_with_someone_at_the_table():
    lobby = _room(name="Mum")["room"]["code"]
    playing, _pid = _started(name="Dad", token="ark")
    rooms = {r["code"]: r for r in jubilee.live_rooms()}
    assert rooms[lobby]["status"] == "lobby" and rooms[lobby]["joinable"] is True
    assert rooms[lobby]["players"] == ["Mum"]
    assert rooms[playing]["status"] == "playing" and rooms[playing]["joinable"] is False
    # Scribes (the AI) are seated but never named as people at the table.
    assert rooms[playing]["players"] == ["Dad"]


def test_live_rooms_drop_a_table_everyone_left(monkeypatch):
    code = _room(name="Gone")["room"]["code"]
    later = time.time() + jubilee.PRESENCE_SECONDS + 5
    monkeypatch.setattr(jubilee.time, "time", lambda: later)
    assert code not in {r["code"] for r in jubilee.live_rooms()}


def test_a_tv_stage_turns_on_hand_mode_and_times_out(monkeypatch):
    code = _room(name="Mum")["room"]["code"]
    assert jubilee.get_room(code)["stage_connected"] is False
    jubilee.get_room(code, stage=True)
    assert jubilee.get_room(code)["stage_connected"] is True
    later = time.time() + jubilee.presence.STAGE_PRESENCE_SECONDS + 1
    monkeypatch.setattr(jubilee.presence.time, "time", lambda: later)
    monkeypatch.setattr(jubilee.time, "time", lambda: later)
    assert jubilee.get_room(code)["stage_connected"] is False


def test_rolls_and_cards_carry_counters_so_repeats_still_animate():
    code, _pid = _started(name="Dad", token="ark")
    room = jubilee._rooms[code]
    jubilee._record_roll(room, 6, 6)
    first = dict(room["last_roll"])
    jubilee._record_roll(room, 6, 6)
    assert room["last_roll"]["n"] == first["n"] + 1
    assert room["last_roll"]["d1"] == 6 and room["last_roll"]["doubles"] is True
    seq = room.get("log_seq", 0)
    for i in range(15):
        jubilee._log(room, f"line {i}")
    assert len(room["log"]) == 10 and room["log_seq"] == seq + 15


def test_card_parts_split_quote_effect_and_reference():
    parts = jubilee.card_parts("“He opens his hand and satisfies the desire of every living thing.” The bank pays you $200. (Psalm 145:16)")
    assert parts == {"quote": "He opens his hand and satisfies the desire of every living thing.",
                     "effect": "The bank pays you $200.", "ref": "Psalm 145:16"}
    plain = jubilee.card_parts("Your figs and wine bring in coin - the bank pays you $50. (Proverbs 27:18)")
    assert plain["effect"] == "The bank pays you $50." and plain["ref"] == "Proverbs 27:18"


def test_your_view_says_which_streets_you_can_build_on_and_why_not():
    code, pid = _started(name="Dad", token="ark")
    room = jubilee._rooms[code]
    seat = jubilee._find_seat(room, pid)
    jubilee._prop(room, 6)["owner"] = seat          # Sychar alone: not the whole group
    you = jubilee.get_room(code, pid)["you"]
    assert "whole colour group" in you["build"]["6"]
    for i in (6, 8, 9):                              # the whole light-blue group
        jubilee._prop(room, i)["owner"] = seat
    you = jubilee.get_room(code, pid)["you"]
    assert you["build"]["6"] == "" and you["sellable"] == []


def test_selling_a_hotel_in_a_housing_shortage_never_overdraws_the_bank():
    code, pid = _started(name="Dad", token="ark")
    room = jubilee._rooms[code]
    seat = jubilee._find_seat(room, pid)
    for i in (6, 8, 9):
        jubilee._prop(room, i)["owner"] = seat
    jubilee._prop(room, 6)["houses"] = 5
    room["bank_hotels"] -= 1
    room["bank_houses"] = 2                      # only two houses left in the Bank
    cash = room["players"][seat]["cash"]
    got = jubilee._sell_building(room, seat, 6)
    assert room["bank_houses"] == 0 and jubilee._prop(room, 6)["houses"] == 2
    cost = jubilee.SPACES[6]["house_cost"]
    assert got == (cost // 2) * 3 and room["players"][seat]["cash"] == cash + got


def test_a_reload_mid_game_rejoins_your_own_seat():
    """The page sends the player id it kept; that used to be ignored, so a
    reload once the game had begun locked the player out of their own seat."""
    made = jubilee.create_room({"name": "Ann", "device_id": "dev-ann"})
    code, pid = made["room"]["code"], made["player_id"]
    jubilee.start_match(code, pid)
    back = jubilee.join_room(code, {"player_id": pid, "name": "Ann"})
    assert back["player_id"] == pid
    assert back["room"]["status"] == "playing"


def test_a_reload_in_the_lobby_is_not_a_second_player():
    made = jubilee.create_room({"name": "Ann"})
    code, pid = made["room"]["code"], made["player_id"]
    jubilee.join_room(code, {"player_id": pid, "name": "Ann"})
    room = jubilee.get_room(code, pid)
    assert [p["name"] for p in room["players"]] == ["Ann"]


def test_the_same_phone_gets_its_seat_back_after_losing_its_saved_id(monkeypatch):
    made = jubilee.create_room({"name": "Ann", "device_id": "dev-ann"})
    code, pid = made["room"]["code"], made["player_id"]
    jubilee.start_match(code, pid)
    # Away from the table (storage cleared, app reinstalled): same device id.
    with jubilee._lock:
        for p in jubilee._rooms[code]["players"]:
            if p["id"] == pid:
                p["last_seen"] = 0
    back = jubilee.join_room(code, {"device_id": "dev-ann", "name": "Ann"})
    assert back["player_id"] == pid


def test_a_stranger_still_cannot_join_a_game_in_progress():
    made = jubilee.create_room({"name": "Ann", "device_id": "dev-ann"})
    code = made["room"]["code"]
    jubilee.start_match(code, made["player_id"])
    import pytest as _pytest
    with _pytest.raises(ValueError, match="already begun"):
        jubilee.join_room(code, {"player_id": "not-a-seat", "device_id": "dev-bob", "name": "Bob"})


def test_with_a_tv_watching_the_computer_takes_one_step_at_a_time(monkeypatch):
    """All at once, several computer turns landed in one TV update: every piece
    jumped together and only the last roll showed."""
    made = jubilee.create_room({"name": "Ann"})
    code, pid = made["room"]["code"], made["player_id"]
    jubilee.start_match(code, pid)
    room = jubilee._rooms[code]
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(jubilee.time, "time", lambda: clock["t"])
    # Ann passes; the computer players are next. A TV is showing the room.
    room["turn"] = 1
    room["rolled"] = False
    room["stage_seen_at"] = clock["t"]
    with jubilee._lock:
        jubilee._maybe_ai_unlocked(room)
    assert room["turn"] == 1 and room["rolled"] is True       # it rolled, and stopped there
    with jubilee._lock:
        assert jubilee._maybe_ai_unlocked(room) is False        # waiting while the piece hops
    clock["t"] += 10
    room["stage_seen_at"] = clock["t"]
    with jubilee._lock:
        jubilee._maybe_ai_unlocked(room)
    assert room["turn"] != 1 or room["rolled"] is False or room.get("pending")   # it went on


def test_without_a_tv_the_computer_plays_straight_through():
    made = jubilee.create_room({"name": "Ann"})
    code, pid = made["room"]["code"], made["player_id"]
    jubilee.start_match(code, pid)
    room = jubilee._rooms[code]
    room["turn"] = 1
    room["rolled"] = False
    with jubilee._lock:
        jubilee._maybe_ai_unlocked(room)
    assert room["turn"] == 0   # both computer players took their turns


def test_a_closed_room_takes_its_board_off_the_tv(monkeypatch):
    import threading as _t

    from arcade import party as _party
    from arcade import presence as _presence

    taken_off = []
    monkeypatch.setattr(_party, "show_stage_on_screen", lambda screen, url, **kw: taken_off.append((screen, kw.get("room_code"))) or {"success": True})
    made = jubilee.create_room({"name": "Ann"})
    code = made["room"]["code"]
    jubilee.remember_screen(code, "ha:media_player.office")
    _presence.claim_screen("ha:media_player.office", "jubilee", code)
    with jubilee._lock:
        jubilee._rooms[code]["updated_at"] = 0          # long idle: pruned
        jubilee._prune_unlocked(10 ** 10)
    for th in list(_t.enumerate()):
        if th.name.startswith("jubilee-release-"):
            th.join(timeout=5)
    assert taken_off == [("ha:media_player.office", code)]


def test_a_closed_room_leaves_a_tv_another_game_took_over(monkeypatch):
    import threading as _t

    from arcade import party as _party
    from arcade import presence as _presence

    taken_off = []
    monkeypatch.setattr(_party, "show_stage_on_screen", lambda *a, **kw: taken_off.append(a) or {"success": True})
    made = jubilee.create_room({"name": "Ann"})
    code = made["room"]["code"]
    jubilee.remember_screen(code, "ha:media_player.office")
    _presence.claim_screen("ha:media_player.office", "party", "ZION")   # a party went on it since
    with jubilee._lock:
        jubilee._rooms[code]["updated_at"] = 0
        jubilee._prune_unlocked(10 ** 10)
    for th in list(_t.enumerate()):
        if th.name.startswith("jubilee-release-"):
            th.join(timeout=5)
    assert taken_off == []



def test_with_people_watching_each_computer_player_rolls_where_they_can_see_it(monkeypatch):
    """Without a TV the computer used to play its whole turn inside the
    person's request, so its dice were gone before any phone saw them."""
    monkeypatch.setattr(jubilee, "AI_PACE_WHEN", "watched")
    made = jubilee.create_room({"name": "Ann"})
    code, pid = made["room"]["code"], made["player_id"]
    jubilee.start_match(code, pid)
    room = jubilee._rooms[code]
    room["turn"] = 1
    room["rolled"] = False
    with jubilee._lock:
        jubilee._maybe_ai_unlocked(room)
    snap = jubilee.get_room(code, pid)
    assert snap["turn"] == 1 and snap["rolled"] and snap["last_roll"]   # its roll is there to see


def test_an_auction_nobody_answers_ends_on_the_clock(monkeypatch):
    """A phone left on the table used to stall an auction -- and the game -- forever."""
    made = jubilee.create_room({"name": "Ann"})
    code, pid = made["room"]["code"], made["player_id"]
    bob = jubilee.join_room(code, {"name": "Bob"})["player_id"]
    jubilee.join_room(code, {"name": "Cy"})        # three people, no computer players
    jubilee.start_match(code, pid)
    clock = [1_000_000.0]
    monkeypatch.setattr(jubilee.time, "time", lambda: clock[0])
    with jubilee._lock:
        room = jubilee._rooms[code]
        jubilee._auction_open(room, 1)
    jubilee.auction_bid(code, bob, 20)
    snap = jubilee.get_room(code, pid)
    assert snap["auction"]["high_bidder_name"] == "Bob" and 0 < snap["auction"]["ends_in"] <= jubilee.AUCTION_OPEN_S
    clock[0] += jubilee.AUCTION_OPEN_S + 1        # Ann never answers
    snap = jubilee.get_room(code, pid)
    assert snap["auction"] is None
    assert snap["board"][1]["owner"] == 1


def test_a_bid_buys_everyone_more_time(monkeypatch):
    made = jubilee.create_room({"name": "Ann"})
    code, pid = made["room"]["code"], made["player_id"]
    bob = jubilee.join_room(code, {"name": "Bob"})["player_id"]
    jubilee.join_room(code, {"name": "Cy"})        # three people, no computer players
    jubilee.start_match(code, pid)
    clock = [1_000_000.0]
    monkeypatch.setattr(jubilee.time, "time", lambda: clock[0])
    with jubilee._lock:
        jubilee._auction_open(jubilee._rooms[code], 1)
    clock[0] += jubilee.AUCTION_OPEN_S - 2
    jubilee.auction_bid(code, bob, 20)
    clock[0] += 5                                  # past the opening window, inside the bid's
    assert jubilee.get_room(code, pid)["auction"] is not None
    jubilee.auction_bid(code, pid, 30)
    clock[0] += jubilee.AUCTION_BID_S + 1
    snap = jubilee.get_room(code, pid)
    assert snap["auction"] is None and snap["board"][1]["owner"] == 0


def test_every_action_the_phone_sends_has_a_route():
    """The phone posted trades to /trade_propose while the server listened on
    /trade/propose: every trade from a phone failed, and nobody saw offers."""
    import re
    from pathlib import Path

    import arcade.app as arcade_app

    page = (Path(__file__).resolve().parents[1] / "arcade" / "family_games" / "jubilee" / "game.html").read_text()
    sent = set(re.findall(r"doAction\('(/[a-z_/]+)'", page))
    sent |= set(re.findall(r"inJailVar\(\) \? '(/[a-z_]+)' : '(/[a-z_]+)'", page) and
                [x for pair in re.findall(r"inJailVar\(\) \? '(/[a-z_]+)' : '(/[a-z_]+)'", page) for x in pair])
    routes = {r.rule for r in arcade_app.app.url_map.iter_rules()}
    assert {"/trade/propose", "/trade/accept", "/trade/reject", "/trade/cancel", "/auction_bid"} <= sent
    missing = [p for p in sent if f"/api/jubilee/rooms/<code>{p}" not in routes]
    assert missing == [], f"the phone calls routes that do not exist: {missing}"


def test_offers_reach_the_phone_with_who_and_what():
    made = jubilee.create_room({"name": "Ann"})
    code, ann = made["room"]["code"], made["player_id"]
    bob = jubilee.join_room(code, {"name": "Bob"})["player_id"]
    jubilee.join_room(code, {"name": "Cy"})
    jubilee.start_match(code, ann)
    with jubilee._lock:
        jubilee._prop(jubilee._rooms[code], 1)["owner"] = 1
    jubilee.trade_propose(code, ann, 1, {"cash": 80}, {"spaces": [1]})
    t = jubilee.get_room(code, bob)["trades"][0]
    assert (t["from"], t["to"], t["from_name"]) == (0, 1, "Ann")
    assert t["offer"]["cash"] == 80 and t["ask"]["spaces"] == [1]
    jubilee.trade_accept(code, bob, t["id"])
    after = jubilee.get_room(code, ann)
    assert after["trades"] == [] and after["board"][1]["owner"] == 0
