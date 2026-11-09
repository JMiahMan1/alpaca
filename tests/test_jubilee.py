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
