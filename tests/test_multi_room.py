"""Several games at once: any number of Jubilee!, Parable Party and Checkers
rooms run side by side, each under its own code, without touching each other
-- including the TV one of them was cast to."""
import threading

import pytest

from arcade import checkers, jubilee, party, presence


@pytest.fixture(autouse=True)
def _clean_rooms():
    jubilee._clear_rooms()
    checkers._clear_rooms()
    with party._lock:
        party._rooms.clear()
    presence._screen_owner.clear()
    yield
    jubilee._clear_rooms()
    checkers._clear_rooms()
    with party._lock:
        party._rooms.clear()
    presence._screen_owner.clear()


def _wait_for_release_threads():
    for t in list(threading.enumerate()):
        if t.name.startswith("party-release-"):
            t.join(timeout=5)


def _party_room(name):
    out = party.create_or_join_room(name=name)
    return out["room"]["code"], out


def test_many_rooms_of_every_game_at_once_never_share_a_code():
    codes = []
    errors = []
    lock = threading.Lock()

    def make(kind, i):
        try:
            if kind == "jubilee":
                code = jubilee.create_room({"name": f"J{i}"})["room"]["code"]
            elif kind == "party":
                code, _ = _party_room(f"P{i}")
            else:
                code = checkers.create_room(name=f"C{i}")["room"]["code"]
            with lock:
                codes.append((kind, code))
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=make, args=(kind, i)) for i in range(12) for kind in ("jubilee", "party", "checkers")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(codes) == 36
    assert len({c for _, c in codes}) == 36, "two rooms (maybe of different games) got the same code"


def test_two_jubilee_games_play_independently():
    a = jubilee.create_room({"name": "Ann"})
    b = jubilee.create_room({"name": "Bob"})
    ca, cb = a["room"]["code"], b["room"]["code"]
    assert ca != cb
    jubilee.join_room(ca, {"name": "Cy"})
    jubilee.start_match(ca, a["player_id"])
    # Room A is playing; room B is still in its lobby with only its host.
    ra = jubilee.get_room(ca, a["player_id"])
    rb = jubilee.get_room(cb, b["player_id"])
    assert ra["status"] == "playing" and rb["status"] == "lobby"
    assert [p["name"] for p in rb["players"]] == ["Bob"]
    # A player of one room is a stranger to the other.
    with pytest.raises(PermissionError):
        jubilee.roll(cb, a["player_id"])


def test_a_jubilee_code_typed_into_parable_party_is_not_a_new_party_room():
    code = jubilee.create_room({"name": "Ann"})["room"]["code"]
    with pytest.raises(LookupError, match="different game"):
        party.create_or_join_room(code=code, name="Bob")
    assert code not in party._rooms


def test_ending_a_party_room_leaves_a_tv_another_game_was_cast_to_since(monkeypatch):
    taken_off = []
    monkeypatch.setattr(party, "show_stage_on_screen",
                        lambda screen, url, **kw: taken_off.append((screen, kw.get("room_code"))) or {"success": True})
    code, out = _party_room("Ann")
    party.remember_screen(code, "ha:media_player.office")
    # Then a Jubilee game goes on the same TV.
    presence.claim_screen("ha:media_player.office", "jubilee", "NOAH")
    party.end_room(code, player_id=out["player_id"])
    _wait_for_release_threads()
    assert taken_off == [], "ending the party took the Jubilee board off the TV"


def test_ending_a_party_room_takes_its_own_stage_off_its_tv(monkeypatch):
    taken_off = []
    monkeypatch.setattr(party, "show_stage_on_screen",
                        lambda screen, url, **kw: taken_off.append((screen, kw.get("room_code"))) or {"success": True})
    code, out = _party_room("Ann")
    party.remember_screen(code, "ha:media_player.office")
    party.end_room(code, player_id=out["player_id"])
    _wait_for_release_threads()
    assert taken_off == [("ha:media_player.office", code)]
    assert presence.screen_owner("ha:media_player.office") is None


def test_two_party_rooms_on_two_tvs_each_keep_their_own():
    a, _ = _party_room("Ann")
    b, _ = _party_room("Bob")
    party.remember_screen(a, "ha:media_player.office")
    party.remember_screen(b, "ha:media_player.living_room")
    assert presence.screen_owner("ha:media_player.office") == ("party", a)
    assert presence.screen_owner("ha:media_player.living_room") == ("party", b)
    assert party._rooms[a]["cast_screen"] == "ha:media_player.office"
    assert party._rooms[b]["cast_screen"] == "ha:media_player.living_room"


def test_games_in_progress_survive_a_restart(tmp_path):
    """Every deploy used to end every game: rooms lived only in memory."""
    import threading as _t

    made = jubilee.create_room({"name": "Ann"})
    code, pid = made["room"]["code"], made["player_id"]
    jubilee.start_match(code, pid)
    jubilee.roll(code, pid)
    before = jubilee.get_room(code, pid)
    assert presence.save_rooms_once(str(tmp_path), "jubilee", jubilee._rooms, jubilee._lock, {}) is True
    jubilee._clear_rooms()                                   # the restart
    assert presence.restore_rooms(str(tmp_path), "jubilee", jubilee._rooms, jubilee._lock) == 1
    after = jubilee.get_room(code, pid)
    assert after["status"] == "playing" and after["you"]["position"] == before["you"]["position"]
    assert after["last_roll"] == before["last_roll"] and after["log"] == before["log"]
    assert isinstance(_t.Lock(), type(jubilee._lock))


def test_an_unchanged_room_set_is_not_rewritten(tmp_path):
    jubilee.create_room({"name": "Ann"})
    last = {}
    assert presence.save_rooms_once(str(tmp_path), "jubilee", jubilee._rooms, jubilee._lock, last) is True
    assert presence.save_rooms_once(str(tmp_path), "jubilee", jubilee._rooms, jubilee._lock, last) is False


def test_a_damaged_or_missing_save_is_ignored(tmp_path):
    assert presence.restore_rooms(str(tmp_path), "jubilee", jubilee._rooms, jubilee._lock) == 0
    (tmp_path / "rooms-jubilee.json").write_text("{ not json")
    assert presence.restore_rooms(str(tmp_path), "jubilee", jubilee._rooms, jubilee._lock) == 0
    assert jubilee._rooms == {}
