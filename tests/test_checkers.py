"""Checkers: the rules, the computer opponent, and the two-player rooms.

The rules and the AI are pure and tested directly; the rooms and the HTTP
surface run through ``arcade.app.test_client()``. Nothing here touches the
network, so the whole file runs in the hermetic suite.
"""

import random
import time

import pytest

from arcade import checkers

RED = checkers.RED
BLACK = checkers.BLACK
MAN = checkers.MAN
KING = checkers.KING


@pytest.fixture(autouse=True)
def _clean_rooms():
    checkers._clear_rooms()
    yield
    checkers._clear_rooms()


def _empty():
    return [0] * 64


def _place(*pairs):
    """A board from (index, value) pairs."""
    b = _empty()
    for i, v in pairs:
        b[i] = v
    return b


# --------------------------------------------------------------------------- #
# The rules
# --------------------------------------------------------------------------- #
def test_initial_board_has_twelve_each_on_the_dark_squares():
    board = checkers.initial_board()
    assert len(board) == 64
    assert checkers.count(board, RED) == 12
    assert checkers.count(board, BLACK) == 12
    for i, p in enumerate(board):
        if p:
            assert checkers.playable(i), f"piece on a light square at {i}"
    # red is at the bottom, black at the top
    assert all(board[i] == MAN for i in range(64) if checkers.playable(i) and i // 8 >= 5)
    assert all(board[i] == -MAN for i in range(64) if checkers.playable(i) and i // 8 <= 2)


def test_red_opens_with_seven_simple_moves():
    moves = checkers.legal_moves(checkers.initial_board(), RED)
    assert len(moves) == 7
    assert all(m["captures"] == [] and not m["promotes"] for m in moves)
    assert all(m["path"][0] // 8 == 5 for m in moves)  # only the front row can move


def test_a_man_moves_forward_diagonally_only():
    board = _place((40, MAN))  # r5c0: only one forward diagonal is on the board
    moves = checkers.legal_moves(board, RED)
    assert [m["path"] for m in moves] == [[40, 33]]  # r4c1

    board2 = _place((42, MAN))  # r5c2 has both forward diagonals
    assert sorted(m["path"][1] for m in checkers.legal_moves(board2, RED)) == [33, 35]


def test_a_king_moves_in_all_four_directions():
    board = _place((27, KING))  # r3c3
    moves = checkers.legal_moves(board, RED)
    assert sorted(m["path"][1] for m in moves) == [18, 20, 34, 36]


def test_captures_are_compulsory():
    board = _place((40, MAN), (33, -MAN), (42, MAN))
    moves = checkers.legal_moves(board, RED)
    assert moves and all(m["captures"] for m in moves)
    assert {tuple(m["path"]) for m in moves} == {(40, 26), (42, 24)}


def test_a_jump_lands_beyond_the_captured_piece():
    board = _place((40, MAN), (33, -MAN))
    move = checkers.legal_moves(board, RED)[0]
    assert move["path"] == [40, 26]
    assert move["captures"] == [33]


def test_multi_jump_chains_are_one_move():
    board = _place((40, MAN), (33, -MAN), (19, -MAN))
    moves = checkers.legal_moves(board, RED)
    assert [m["path"] for m in moves] == [[40, 26, 12]]
    assert moves[0]["captures"] == [33, 19]


def test_a_chain_must_be_played_to_its_end():
    # Two jumps available; both must complete, so no single-jump move is legal.
    board = _place((40, MAN), (33, -MAN), (19, -MAN))
    for m in checkers.legal_moves(board, RED):
        assert len(m["path"]) == 3 and len(m["captures"]) == 2


def test_promotion_on_reaching_the_far_row():
    board = _place((10, MAN))  # r1c2 -> r0
    moves = checkers.legal_moves(board, RED)
    assert moves and all(m["promotes"] for m in moves)
    landed = {m["path"][1] for m in moves}
    assert landed == {1, 3}


def test_a_crowning_jump_is_a_single_hop():
    # Red at 19 (r2c3) can jump either black, landing on the back row; a crowning
    # jump is one hop (there is no square beyond it to continue from).
    board = _place((19, MAN), (10, -MAN), (12, -MAN))
    moves = checkers.legal_moves(board, RED)
    assert {tuple(m["path"]) for m in moves} == {(19, 1), (19, 5)}
    for m in moves:
        assert m["promotes"] is True
        assert len(m["path"]) == 2 and len(m["captures"]) == 1


def test_apply_move_moves_the_piece_and_removes_captures():
    board = _place((40, MAN), (33, -MAN))
    move = checkers.legal_moves(board, RED)[0]
    after = checkers.apply_move(board, move)
    assert after[40] == 0 and after[33] == 0
    assert after[26] == MAN
    # the original board is untouched
    assert board[40] == MAN and board[33] == -MAN


def test_apply_move_crowns_at_the_end_of_a_chain():
    board = _place((19, MAN), (10, -MAN))
    move = checkers.legal_moves(board, RED)[0]
    after = checkers.apply_move(board, move)
    assert after[1] == KING


def test_no_pieces_is_a_loss():
    board = _place((0, -MAN))
    assert checkers._result_for(board, RED) == "no-pieces"


def test_no_legal_move_is_a_loss():
    # A lone red man in the corner (r0c1) with black men blocking both exits.
    board = _place((1, MAN), (8, -MAN), (10, -MAN))
    assert checkers._result_for(board, RED) == "no-moves"


def test_evaluate_prefers_more_material_for_the_right_side():
    strong = _place((40, MAN), (42, MAN), (44, MAN), (33, -MAN))
    assert checkers.evaluate(strong, RED) > 0
    assert checkers.evaluate(strong, BLACK) < 0


# --------------------------------------------------------------------------- #
# The computer opponent
# --------------------------------------------------------------------------- #
def test_ai_opens_with_a_legal_move():
    board = checkers.initial_board()
    mv = checkers.choose_move(board, RED, max_depth=3, time_budget=0.5, rng=random.Random(1))
    assert mv in checkers.legal_moves(board, RED)


def test_ai_takes_a_forced_capture():
    board = _place((40, MAN), (33, -MAN), (42, MAN))
    mv = checkers.choose_move(board, RED, max_depth=3, time_budget=0.5, rng=random.Random(1))
    assert mv["captures"] == [33]


def test_ai_returns_none_when_there_is_no_move():
    board = _place((1, MAN), (8, -MAN), (10, -MAN))
    assert checkers.choose_move(board, RED, max_depth=2, time_budget=0.2) is None


def test_ai_prefers_the_capture_that_wins_a_king():
    # Red man at 19 can jump 10 to 1 (crowning), or it cannot otherwise capture.
    board = _place((19, MAN), (10, -MAN))
    mv = checkers.choose_move(board, RED, max_depth=3, time_budget=0.5, rng=random.Random(2))
    assert mv["path"] == [19, 1] and mv["promotes"] is True


def test_ai_finishes_a_chain_it_is_forced_into():
    # Red man at 40 jumps black at 33 (landing 26); black at 17 is then jumpable
    # (landing 8), and the chain must be completed to the end.
    board = _place((40, MAN), (33, -MAN), (17, -MAN))
    moves = checkers.legal_moves(board, RED)
    assert [m["path"] for m in moves] == [[40, 26, 8]]
    mv = checkers.choose_move(board, RED, max_depth=4, time_budget=0.5, rng=random.Random(3))
    assert mv["captures"] == [33, 17]


def test_ai_vs_ai_plays_a_whole_legal_game():
    board = checkers.initial_board()
    turn = RED
    rng = random.Random(11)
    plies = 0
    while plies < 400:
        moves = checkers.legal_moves(board, turn)
        if not moves:
            break
        mv = checkers.choose_move(board, turn, max_depth=2, time_budget=0.2, rng=rng)
        assert mv in moves
        board = checkers.apply_move(board, mv)
        turn = checkers.other(turn)
        plies += 1
    # Either someone ran out of moves or the cap was reached; both are "ended".
    assert plies < 400 or not checkers.legal_moves(board, turn)


# --------------------------------------------------------------------------- #
# Rooms
# --------------------------------------------------------------------------- #
def test_create_friend_room_waits_for_an_opponent():
    out = checkers.create_room(mode="friend", name="Ann", initials="ANN")
    room = out["room"]
    assert room["status"] == "waiting"
    assert room["players"][RED]["name"] == "Ann"
    assert room["players"][BLACK] is None
    assert room["you"]["color"] == RED
    assert out["player_id"]


def test_a_second_player_joining_starts_the_game():
    created = checkers.create_room(mode="friend", name="Ann")
    code = created["room"]["code"]
    joined = checkers.join_room(code, name="Bob", initials="BOB")
    assert joined["room"]["status"] == "playing"
    assert joined["room"]["you"]["color"] == BLACK
    # the creator sees the same board and the opponent
    seen = checkers.get_room(code, created["player_id"])
    assert seen["status"] == "playing"
    assert seen["players"][BLACK]["name"] == "Bob"


def test_quick_match_pairs_two_people():
    a = checkers.quick_match(name="One")
    assert a["room"]["status"] == "waiting"
    b = checkers.quick_match(name="Two")
    assert b["room"]["code"] == a["room"]["code"]
    assert b["room"]["status"] == "playing"


def test_adding_the_computer_starts_a_single_player_game():
    out = checkers.create_room(mode="friend", name="Solo")
    code = out["room"]["code"]
    res = checkers.add_ai(code, out["player_id"])
    room = res["room"]
    assert room["status"] == "playing"
    assert room["ai"] == BLACK
    assert room["players"][BLACK]["is_ai"] is True
    assert room["players"][BLACK]["name"] == "Alpaca AI"


def test_only_a_player_may_add_the_computer():
    out = checkers.create_room(mode="friend", name="Solo")
    with pytest.raises(PermissionError):
        checkers.add_ai(out["room"]["code"], "not-a-player")


def test_submit_move_changes_the_board_and_passes_the_turn():
    created = checkers.create_room(mode="friend", name="Ann")
    code = created["room"]["code"]
    checkers.join_room(code, name="Bob")
    before = checkers.get_room(code, created["player_id"])
    move = before["legal"][0]
    res = checkers.submit_move(code, created["player_id"], move["path"])
    room = res["room"]
    assert room["turn"] == BLACK
    assert room["board"][move["path"][0]] == 0
    assert room["history_len"] == 1


def test_you_cannot_move_on_your_opponents_turn():
    created = checkers.create_room(mode="friend", name="Ann")
    code = created["room"]["code"]
    joined = checkers.join_room(code, name="Bob")
    with pytest.raises(ValueError):
        checkers.submit_move(code, joined["player_id"], [17, 26])


def test_an_illegal_move_is_rejected():
    created = checkers.create_room(mode="friend", name="Ann")
    code = created["room"]["code"]
    with pytest.raises(ValueError):
        checkers.submit_move(code, created["player_id"], [40, 20])  # too far


def test_a_snapshot_never_carries_player_ids():
    created = checkers.create_room(mode="friend", name="Ann")
    room = checkers.get_room(created["room"]["code"], created["player_id"])
    blob = repr(room)
    assert created["player_id"] not in blob
    assert room["players"][RED]["public_id"].startswith("u")


def test_legal_moves_are_only_sent_to_the_player_to_move():
    created = checkers.create_room(mode="friend", name="Ann")
    code = created["room"]["code"]
    joined = checkers.join_room(code, name="Bob")
    for_ann = checkers.get_room(code, created["player_id"])
    for_bob = checkers.get_room(code, joined["player_id"])
    spectator = checkers.get_room(code)
    assert for_ann["legal"] and not for_bob["legal"]
    assert spectator["legal"] == []


def test_leaving_mid_game_forfeits_it():
    created = checkers.create_room(mode="friend", name="Ann")
    code = created["room"]["code"]
    joined = checkers.join_room(code, name="Bob")
    checkers.submit_move(code, created["player_id"], checkers.get_room(code, created["player_id"])["legal"][0]["path"])
    checkers.leave_room(code, joined["player_id"])
    room = checkers.get_room(code, created["player_id"])
    assert room["status"] == "finished"
    assert room["winner"] == RED and room["result"] == "forfeit"


def test_reset_starts_a_fresh_rematch():
    created = checkers.create_room(mode="friend", name="Ann")
    code = created["room"]["code"]
    checkers.join_room(code, name="Bob")
    checkers.submit_move(code, created["player_id"], checkers.get_room(code, created["player_id"])["legal"][0]["path"])
    res = checkers.reset_room(code, created["player_id"])
    room = res["room"]
    assert room["status"] == "playing"
    assert room["board"] == checkers.initial_board()
    assert room["turn"] == RED
    assert room["history_len"] == 0
    assert room["you"]["match_no"] == 2


def test_the_computer_replies_within_the_same_request():
    out = checkers.create_room(mode="ai", name="Solo")
    code = out["room"]["code"]
    assert out["room"]["turn"] == RED
    res = checkers.submit_move(code, out["player_id"], out["room"]["legal"][0]["path"])
    # the AI has already answered, so it is the person's turn again
    assert res["room"]["turn"] == RED
    assert res["room"]["history_len"] == 2


def test_list_open_rooms_shows_only_waiting_rooms():
    a = checkers.create_room(mode="friend", name="Ann")
    checkers.create_room(mode="ai", name="Solo")
    open_rooms = checkers.list_open_rooms()
    assert [r["code"] for r in open_rooms] == [a["room"]["code"]]
    assert open_rooms[0]["host"] == "Ann"


def test_an_idle_room_is_forgotten():
    created = checkers.create_room(mode="friend", name="Ann")
    code = created["room"]["code"]
    with checkers._lock:
        checkers._rooms[code]["updated_at"] = time.time() - checkers.ROOM_TTL_SECONDS - 10
    assert checkers.get_room(code) is None


# --------------------------------------------------------------------------- #
# The HTTP surface
# --------------------------------------------------------------------------- #
@pytest.fixture
def client():
    import arcade.app as arcade_app

    arcade_app.app.config["TESTING"] = True
    with arcade_app.app.test_client() as c:
        yield c


def test_http_create_join_and_move(client):
    r = client.post("/api/checkers/rooms", json={"mode": "friend", "name": "Ann", "initials": "ANN"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["success"] and body["room"]["status"] == "waiting"
    pid, code = body["player_id"], body["room"]["code"]

    r2 = client.post("/api/checkers/rooms", json={"code": code, "name": "Bob"})
    assert r2.status_code == 200 and r2.get_json()["room"]["status"] == "playing"

    state = client.get(f"/api/checkers/rooms/{code}?player_id={pid}").get_json()
    move = state["room"]["legal"][0]["path"]
    rm = client.post(f"/api/checkers/rooms/{code}/move", json={"player_id": pid, "path": move})
    assert rm.status_code == 200 and rm.get_json()["room"]["turn"] == "black"


def test_http_unknown_room_is_404(client):
    assert client.get("/api/checkers/rooms/ZZZZ").status_code == 404


def test_http_illegal_move_is_400(client):
    body = client.post("/api/checkers/rooms", json={"mode": "ai"}).get_json()
    code, pid = body["room"]["code"], body["player_id"]
    r = client.post(f"/api/checkers/rooms/{code}/move", json={"player_id": pid, "path": [40, 20]})
    assert r.status_code == 400


def test_http_move_by_a_stranger_is_403(client):
    body = client.post("/api/checkers/rooms", json={"mode": "friend"}).get_json()
    code = body["room"]["code"]
    r = client.post(f"/api/checkers/rooms/{code}/move", json={"player_id": "nobody", "path": [40, 32]})
    assert r.status_code == 403


def test_http_reset_by_a_stranger_is_403(client):
    body = client.post("/api/checkers/rooms", json={"mode": "friend"}).get_json()
    code = body["room"]["code"]
    assert client.post(f"/api/checkers/rooms/{code}/reset", json={"player_id": "nobody"}).status_code == 403


def test_http_list_rooms(client):
    client.post("/api/checkers/rooms", json={"mode": "friend", "name": "Ann"})
    r = client.get("/api/checkers/rooms")
    assert r.status_code == 200
    assert len(r.get_json()["rooms"]) == 1


# --------------------------------------------------------------------------- #
# The shipped page
# --------------------------------------------------------------------------- #
def test_checkers_source_passes_the_mobile_contract():
    from arcade import family
    from arcade.mobile import contract_failures

    html = family.build_html("checkers")
    assert contract_failures(html) == []
    # and it carries the score bridge the arcade reads
    assert "arcade-game" in html and "run_id" in html
    assert family.KIT_MARKER not in html  # the kit was inlined


def test_checkers_is_a_built_in_source():
    from arcade import family

    assert "checkers" in family.source_names()
    assert family.slug_for("checkers") == "family-checkers"


# --------------------------------------------------------------------------- #
# Reuse of Parable Party (the arcade's other room engine)
# --------------------------------------------------------------------------- #
def test_codes_prefer_parable_partys_friendly_words():
    from arcade import party

    code = checkers._pick_code_unlocked()
    assert code in party.ROOM_WORDS
    # and a taken word is not handed out twice
    checkers._rooms[code] = {}
    others = {checkers._pick_code_unlocked() for _ in range(50)}
    assert code not in others


def test_public_id_matches_parable_party():
    from arcade import party

    room = {"secret": "s3cret"}
    human = {"id": "pid-1", "is_ai": False}
    assert checkers._public_id(room, human) == party.public_id(room, human["id"])
    assert checkers._public_id(room, {"id": "ai", "is_ai": True}) == "ai"
    assert checkers._public_id(room, None) is None


def test_room_payload_carries_an_invite_link_and_qr(client):
    body = client.post("/api/checkers/rooms", json={"mode": "friend", "name": "Ann"}).get_json()
    room = body["room"]
    assert room["invite_url"].endswith(f"#room={room['code']}")
    assert room["qr_url"] == f"/api/checkers/rooms/{room['code']}/join.svg"


def test_join_qr_is_an_svg(client):
    body = client.post("/api/checkers/rooms", json={"mode": "friend", "name": "Ann"}).get_json()
    code = body["room"]["code"]
    r = client.get(f"/api/checkers/rooms/{code}/join.svg")
    assert r.status_code == 200
    assert r.mimetype == "image/svg+xml"
    assert b"<svg" in r.data


def test_the_lobby_swaps_in_the_waiting_screen():
    """A friend room used to dead-end on 'Starting…': the waiting screen (with
    the code and QR) was only rendered when the lobby was hidden, but it was
    already showing the starting screen. The guard must key on the code element.
    """
    from arcade import family

    html = family.build_html("checkers")
    assert '!lobbyEl.classList.contains("ck-show")' not in html  # the old guard
    assert 'else showWaiting(room);' in html
    assert 'room.invite_url || shareUrl(room.code)' in html
    assert 'room.qr_url' in html
