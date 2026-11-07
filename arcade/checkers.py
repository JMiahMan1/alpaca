"""Checkers - the rules, the computer opponent, and the rooms behind the game.

The family arcade's one game that two people play *against each other*: a
turn-based 8x8 board, so it does not fit the party engine's phase machine
(``party.py``). This module is therefore self-contained: English draughts
(American checkers) rules, a small alpha-beta opponent, and an in-memory room
store keyed by a short code.

English draughts, the variant the benchmarks grade ("Checkers"):

* 8x8 board, the twelve dark squares nearest each player. Dark squares are the
  ones where ``(row + col) % 2 == 1``.
* A man moves one square diagonally forward; a king moves one square diagonally
  in any direction (kings do not fly).
* Capture by jumping an adjacent enemy piece onto the empty square beyond;
  several jumps chain if the piece can keep capturing.
* A man that reaches the far row is crowned and its move ends there.
* **Captures are compulsory.** When a jump exists, only jumps are legal; the
  player may choose which one, and once a chain starts it must continue.
* A player with no pieces, or with no legal move, loses. Forty moves each with
  no capture and no man moving (here eighty plies) is a draw, as is a position
  repeating three times.

The board is a flat list of 64 ints, ``row * 8 + col`` from the top-left::

    0 empty    1 red man   2 red king   -1 black man   -2 black king

Red sits at the bottom (rows 5-7) and moves up; black sits at the top and
moves down. Red moves first.
"""

from __future__ import annotations

import hashlib
import random
import re
import secrets
import threading
import time
from typing import Any

RED = "red"
BLACK = "black"
COLORS = (RED, BLACK)

MAN = 1
KING = 2

# Rows that crown a man: red promotes at the top, black at the bottom.
_PROMO_ROW = {RED: 0, BLACK: 7}
# A room is forgotten two hours after its last activity, or after a player has
# been gone half an hour, so the in-memory store cannot grow without bound.
ROOM_TTL_SECONDS = 7200
ROOM_IDLE_SECONDS = 1800
ROOM_EMPTY_GRACE_SECONDS = 300
# A player counts as "present" if they polled within this window.
PRESENCE_SECONDS = 30
# Eighty plies (forty moves each) without a capture or a man advancing.
MAX_NO_PROGRESS = 80
# The AI never spends longer than this on one move.
AI_TIME_BUDGET = 1.6
AI_MAX_DEPTH = 6

_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

_lock = threading.Lock()
_rooms: dict[str, dict[str, Any]] = {}


# --------------------------------------------------------------------------- #
# Board geometry and the rules
# --------------------------------------------------------------------------- #
def _rc(i: int) -> tuple[int, int]:
    return divmod(i, 8)


def _idx(r: int, c: int) -> int:
    return r * 8 + c


def _on(r: int, c: int) -> bool:
    return 0 <= r < 8 and 0 <= c < 8


def playable(i: int) -> bool:
    r, c = _rc(i)
    return (r + c) % 2 == 1


def _sign(p: int) -> int:
    return 1 if p > 0 else -1 if p < 0 else 0


def sign_of(color: str) -> int:
    return 1 if color == RED else -1


def color_of_sign(sign: int) -> str:
    return RED if sign > 0 else BLACK


def other(color: str) -> str:
    return BLACK if color == RED else RED


def _dirs(sign: int, king: bool) -> tuple[tuple[int, int], ...]:
    if king:
        return ((-1, -1), (-1, 1), (1, -1), (1, 1))
    return ((-1, -1), (-1, 1)) if sign > 0 else ((1, -1), (1, 1))


def initial_board() -> list[int]:
    """The standard twelve-per-side opening position."""
    board = [0] * 64
    for r in range(3):
        for c in range(8):
            if playable(_idx(r, c)):
                board[_idx(r, c)] = -MAN  # black, top
    for r in range(5, 8):
        for c in range(8):
            if playable(_idx(r, c)):
                board[_idx(r, c)] = MAN  # red, bottom
    return board


def _simple_moves(board: list[int], i: int) -> list[dict[str, Any]]:
    p = board[i]
    sign = _sign(p)
    color = color_of_sign(sign)
    king = abs(p) == KING
    r0, c0 = _rc(i)
    out = []
    for dr, dc in _dirs(sign, king):
        r, c = r0 + dr, c0 + dc
        if not _on(r, c):
            continue
        land = _idx(r, c)
        if board[land] != 0:
            continue
        promotes = (not king) and r == _PROMO_ROW[color]
        out.append({"path": [i, land], "captures": [], "promotes": promotes})
    return out


def _jump_seqs(board: list[int], i: int) -> list[dict[str, Any]]:
    """Every maximal jump chain starting at ``i`` (the piece already there)."""
    p = board[i]
    if p == 0:
        return []
    sign = _sign(p)
    color = color_of_sign(sign)
    king = abs(p) == KING
    r0, c0 = _rc(i)
    out: list[dict[str, Any]] = []
    for dr, dc in _dirs(sign, king):
        mr, mc = r0 + dr, c0 + dc
        lr, lc = r0 + 2 * dr, c0 + 2 * dc
        if not _on(lr, lc):
            continue
        mid, land = _idx(mr, mc), _idx(lr, lc)
        if board[mid] == 0 or _sign(board[mid]) == sign or board[land] != 0:
            continue
        nb = list(board)
        nb[i] = 0
        nb[mid] = 0
        promotes = (not king) and lr == _PROMO_ROW[color]
        nb[land] = sign * KING if (king or promotes) else sign * MAN
        if promotes:
            # A crowning jump ends the move (English rule).
            out.append({"path": [i, land], "captures": [mid], "promotes": True})
            continue
        subs = _jump_seqs(nb, land)
        if subs:
            for s in subs:
                out.append(
                    {"path": [i] + s["path"], "captures": [mid] + s["captures"], "promotes": False}
                )
        else:
            out.append({"path": [i, land], "captures": [mid], "promotes": False})
    return out


def legal_moves(board: list[int], color: str) -> list[dict[str, Any]]:
    """Every legal move for ``color``. Captures are compulsory."""
    sign = sign_of(color)
    jumps: list[dict[str, Any]] = []
    for i, p in enumerate(board):
        if _sign(p) == sign:
            jumps.extend(_jump_seqs(board, i))
    if jumps:
        return jumps
    out: list[dict[str, Any]] = []
    for i, p in enumerate(board):
        if _sign(p) == sign:
            out.extend(_simple_moves(board, i))
    return out


def apply_move(board: list[int], move: dict[str, Any]) -> list[int]:
    """The board after ``move``: the piece walks its path, captured men leave."""
    b = list(board)
    start = move["path"][0]
    dest = move["path"][-1]
    p = b[start]
    sign = _sign(p)
    king = abs(p) == KING
    b[start] = 0
    for cap in move["captures"]:
        b[cap] = 0
    b[dest] = sign * KING if (king or move.get("promotes")) else sign * MAN
    return b


def count(board: list[int], color: str) -> int:
    sign = sign_of(color)
    return sum(1 for p in board if _sign(p) == sign)


def _result_for(board: list[int], color: str) -> str | None:
    """Whether ``color`` (to move) has lost on this board."""
    if count(board, color) == 0:
        return "no-pieces"
    if not legal_moves(board, color):
        return "no-moves"
    return None


# --------------------------------------------------------------------------- #
# The computer opponent
# --------------------------------------------------------------------------- #
def evaluate(board: list[int], color: str) -> int:
    """A static score from ``color``'s point of view (higher is better)."""
    sign = sign_of(color)
    total = 0
    for i, p in enumerate(board):
        if p == 0:
            continue
        r, c = _rc(i)
        v = 180 if abs(p) == KING else 100
        if abs(p) == MAN:
            advance = (7 - r) if p > 0 else r
            v += advance * 5
            if 2 <= c <= 5:
                v += 2
            # keeping men on your own back row slows the enemy down
            if (p > 0 and r == 7) or (p < 0 and r == 0):
                v += 8
        else:
            v += 4  # a king near the centre is worth a touch more
        total += v if p > 0 else -v
    return total * sign


def _ordered(moves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Try captures first: stronger pruning, and usually the right move."""
    return sorted(moves, key=lambda m: (len(m["captures"]), m["promotes"]), reverse=True)


def _search(board: list[int], color: str, depth: int, alpha: int, beta: int) -> int:
    moves = legal_moves(board, color)
    if not moves:
        # The side to move loses; a faster win is worth slightly more.
        return -100000 + depth
    if depth <= 0:
        return evaluate(board, color)
    best = -10**9
    for m in _ordered(moves):
        val = -_search(apply_move(board, m), other(color), depth - 1, -beta, -alpha)
        if val > best:
            best = val
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def choose_move(
    board: list[int],
    color: str,
    *,
    max_depth: int = AI_MAX_DEPTH,
    time_budget: float = AI_TIME_BUDGET,
    rng: random.Random | None = None,
) -> dict[str, Any] | None:
    """Pick a move with iterative-deepening alpha-beta. None if none is legal."""
    picker = rng or random
    moves = legal_moves(board, color)
    if not moves:
        return None
    if len(moves) == 1:
        return moves[0]
    deadline = time.monotonic() + time_budget
    best_move = moves[0]
    for depth in range(1, max_depth + 1):
        alpha = -10**9
        scored: list[tuple[int, dict[str, Any]]] = []
        for m in _ordered(moves):
            val = -_search(apply_move(board, m), other(color), depth - 1, -10**9, -alpha)
            scored.append((val, m))
            if val > alpha:
                alpha = val
            if time.monotonic() > deadline:
                break
        if scored:
            top = max(v for v, _ in scored)
            # Nudge among near-equal moves so the computer is not a metronome.
            pool = [m for v, m in scored if v >= top - 1]
            best_move = picker.choice(pool)
        if time.monotonic() > deadline:
            break
    return best_move


# --------------------------------------------------------------------------- #
# Rooms
# --------------------------------------------------------------------------- #
def _clean_code(raw: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(raw or "")).upper()[:8]


def _clean_text(raw: Any, max_len: int = 24) -> str:
    text = re.sub(r"\s+", " ", str(raw or "")).strip()
    return text[:max_len]


def _clean_initials(raw: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(raw or ""))[:3].upper() or "YOU"


def _pick_code_unlocked() -> str:
    for _ in range(60):
        candidate = "".join(random.choices(_CODE_ALPHABET, k=4))
        if candidate not in _rooms:
            return candidate
    return f"CK{random.randint(100, 999)}"


def _new_player(name: Any, initials: Any) -> dict[str, Any]:
    now = time.time()
    return {
        "id": secrets.token_hex(16),
        "name": _clean_text(name) or "Player",
        "initials": _clean_initials(initials),
        "joined_at": now,
        "last_seen": now,
    }


def _ai_player() -> dict[str, Any]:
    now = time.time()
    return {
        "id": "ai",
        "name": "Alpaca AI",
        "initials": "CPU",
        "is_ai": True,
        "joined_at": now,
        "last_seen": now,
    }


def _find_seat(room: dict[str, Any], player_id: str | None) -> str | None:
    if not player_id:
        return None
    for color in COLORS:
        p = room["seats"].get(color)
        if p and p.get("id") == player_id:
            return color
    return None


def _public_id(room: dict[str, Any], player: dict[str, Any] | None) -> str | None:
    if not player:
        return None
    if player.get("is_ai"):
        return "ai"
    secret = room.setdefault("secret", secrets.token_hex(16))
    return "u" + hashlib.sha256(f"{secret}:{player['id']}".encode()).hexdigest()[:12]


def _new_room(code: str, creator: dict[str, Any], *, ai: bool) -> dict[str, Any]:
    now = time.time()
    board = initial_board()
    room: dict[str, Any] = {
        "code": code,
        "created_at": now,
        "updated_at": now,
        "board": board,
        "turn": RED,
        "seats": {RED: creator, BLACK: None},
        "ai": None,
        "status": "waiting",
        "winner": None,
        "result": None,
        "history": [],
        "last_move": None,
        "no_progress": 0,
        "positions": {},
        "match_no": 1,
        "secret": secrets.token_hex(16),
    }
    _bump_position(room)
    if ai:
        room["seats"][BLACK] = _ai_player()
        room["ai"] = BLACK
        room["status"] = "playing"
    return room


def _position_key(room: dict[str, Any]) -> str:
    return hashlib.sha1(
        (",".join(str(p) for p in room["board"]) + "|" + room["turn"]).encode()
    ).hexdigest()


def _bump_position(room: dict[str, Any]) -> None:
    key = _position_key(room)
    room["positions"][key] = room["positions"].get(key, 0) + 1


def _finish(room: dict[str, Any], winner: str | None, result: str) -> None:
    room["status"] = "finished"
    room["winner"] = winner
    room["result"] = result


def _advance(room: dict[str, Any], move: dict[str, Any], by: str) -> None:
    """Apply ``move`` for ``by`` and settle turn, draws and the result."""
    room["board"] = apply_move(room["board"], move)
    room["history"].append(
        {"by": by, "path": list(move["path"]), "captures": list(move["captures"]),
         "promotes": bool(move.get("promotes"))}
    )
    room["last_move"] = {
        "path": list(move["path"]),
        "captures": list(move["captures"]),
        "by": by,
    }
    if move["captures"] or move.get("promotes"):
        room["no_progress"] = 0
    else:
        room["no_progress"] += 1
    room["turn"] = other(by)
    room["updated_at"] = time.time()
    _bump_position(room)
    nxt = room["turn"]
    if _result_for(room["board"], nxt):
        _finish(room, by, "no-piece-moves")
    elif room["no_progress"] >= MAX_NO_PROGRESS:
        _finish(room, None, "draw-no-progress")
    elif room["positions"].get(_position_key(room), 0) >= 3:
        _finish(room, None, "draw-repetition")


def _maybe_ai_unlocked(room: dict[str, Any]) -> bool:
    """Let the computer move if it is its turn. Returns True if it moved."""
    color = room.get("ai")
    moved = False
    guard = 0
    while (
        color
        and room["status"] == "playing"
        and room["turn"] == color
        and guard < 4
    ):
        guard += 1
        move = choose_move(room["board"], color)
        if not move:
            _finish(room, other(color), "no-piece-moves")
            break
        _advance(room, move, color)
        moved = True
    return moved


def _prune_unlocked(now: float) -> None:
    stale = []
    for code, room in _rooms.items():
        humans = [p for p in room["seats"].values() if p and not p.get("is_ai")]
        if now - float(room.get("updated_at") or 0) > ROOM_TTL_SECONDS:
            stale.append(code)
            continue
        if not humans:
            if now - float(room.get("created_at") or now) > ROOM_EMPTY_GRACE_SECONDS:
                stale.append(code)
            continue
        last = max(float(p.get("last_seen") or p.get("joined_at") or 0) for p in humans)
        if now - last > ROOM_IDLE_SECONDS:
            stale.append(code)
    for code in stale:
        _rooms.pop(code, None)


def _touch(room: dict[str, Any], color: str | None) -> None:
    if color:
        p = room["seats"].get(color)
        if p and not p.get("is_ai"):
            p["last_seen"] = time.time()


def create_room(
    *, mode: str = "friend", name: Any = None, initials: Any = None,
    player_id: Any = None,
) -> dict[str, Any]:
    """Open a room: ``friend`` waits for a second person, ``ai`` seats the computer."""
    with _lock:
        _prune_unlocked(time.time())
        creator = _new_player(name, initials)
        room = _new_room(_pick_code_unlocked(), creator, ai=(mode == "ai"))
        _rooms[room["code"]] = room
        return {"player_id": creator["id"], "room": snapshot_unlocked(room, creator["id"])}


def quick_match(*, name: Any = None, initials: Any = None) -> dict[str, Any]:
    """Join the oldest room waiting for a person, or open one if none exists."""
    with _lock:
        now = time.time()
        _prune_unlocked(now)
        waiting = [
            r for r in _rooms.values()
            if r["status"] == "waiting" and r["seats"].get(BLACK) is None
        ]
        waiting.sort(key=lambda r: float(r.get("created_at") or 0))
        player = _new_player(name, initials)
        if waiting:
            room = waiting[0]
            room["seats"][BLACK] = player
            room["status"] = "playing"
            room["updated_at"] = now
            return {"player_id": player["id"], "room": snapshot_unlocked(room, player["id"])}
        room = _new_room(_pick_code_unlocked(), player, ai=False)
        _rooms[room["code"]] = room
        return {"player_id": player["id"], "room": snapshot_unlocked(room, player["id"])}


def join_room(
    code: Any, *, name: Any = None, initials: Any = None, player_id: Any = None,
) -> dict[str, Any]:
    """Take the free seat in a room, or rejoin one this player already holds."""
    clean = _clean_code(code)
    with _lock:
        _prune_unlocked(time.time())
        room = _rooms.get(clean)
        if room is None:
            raise KeyError(clean)
        seat = _find_seat(room, player_id)
        if seat:
            _touch(room, seat)
            room["updated_at"] = time.time()
            return {"player_id": player_id, "room": snapshot_unlocked(room, player_id)}
        free = next((c for c in (BLACK, RED) if room["seats"].get(c) is None), None)
        if free is None:
            raise ValueError("room is full")
        player = _new_player(name, initials)
        room["seats"][free] = player
        room["updated_at"] = time.time()
        if all(room["seats"].get(c) is not None for c in COLORS):
            room["status"] = "playing"
        return {"player_id": player["id"], "room": snapshot_unlocked(room, player["id"])}


def add_ai(code: Any, player_id: Any) -> dict[str, Any]:
    """Seat the computer in a waiting room's empty seat."""
    clean = _clean_code(code)
    with _lock:
        room = _rooms.get(clean)
        if room is None:
            raise KeyError(clean)
        seat = _find_seat(room, player_id)
        if seat is None:
            raise PermissionError("only a player in the room may add the computer")
        empty = next((c for c in COLORS if room["seats"].get(c) is None), None)
        if empty is None:
            raise ValueError("the room is already full")
        room["seats"][empty] = _ai_player()
        room["ai"] = empty
        room["status"] = "playing"
        room["updated_at"] = time.time()
        _maybe_ai_unlocked(room)
        return {"player_id": player_id, "room": snapshot_unlocked(room, player_id)}


def get_room(code: Any, player_id: Any = None) -> dict[str, Any] | None:
    clean = _clean_code(code)
    with _lock:
        now = time.time()
        _prune_unlocked(now)
        room = _rooms.get(clean)
        if room is None:
            return None
        seat = _find_seat(room, player_id)
        if seat:
            _touch(room, seat)
        if room["status"] == "playing":
            _maybe_ai_unlocked(room)
        return snapshot_unlocked(room, player_id)


def submit_move(code: Any, player_id: Any, path: Any) -> dict[str, Any]:
    """Play ``path`` (a list of square indices) for the player's side."""
    clean = _clean_code(code)
    if not isinstance(path, (list, tuple)) or len(path) < 2:
        raise ValueError("a move needs at least two squares")
    try:
        wanted = [int(x) for x in path]
    except (TypeError, ValueError) as exc:
        raise ValueError("move squares must be integers") from exc
    with _lock:
        room = _rooms.get(clean)
        if room is None:
            raise KeyError(clean)
        seat = _find_seat(room, player_id)
        if seat is None:
            raise PermissionError("you are not a player in this room")
        if room["status"] != "playing":
            raise ValueError("the game is not in play")
        if room["turn"] != seat:
            raise ValueError("it is not your turn")
        legal = legal_moves(room["board"], seat)
        match = next((m for m in legal if m["path"] == wanted), None)
        if match is None:
            raise ValueError("that is not a legal move")
        _touch(room, seat)
        _advance(room, match, seat)
        _maybe_ai_unlocked(room)
        return {"player_id": player_id, "room": snapshot_unlocked(room, player_id)}


def reset_room(code: Any, player_id: Any) -> dict[str, Any]:
    """A rematch: fresh board, same seats, red to move again."""
    clean = _clean_code(code)
    with _lock:
        room = _rooms.get(clean)
        if room is None:
            raise KeyError(clean)
        seat = _find_seat(room, player_id)
        if seat is None:
            raise PermissionError("you are not a player in this room")
        board = initial_board()
        room["board"] = board
        room["turn"] = RED
        room["status"] = (
            "playing" if all(room["seats"].get(c) is not None for c in COLORS) else "waiting"
        )
        room["winner"] = None
        room["result"] = None
        room["history"] = []
        room["last_move"] = None
        room["no_progress"] = 0
        room["positions"] = {}
        room["match_no"] += 1
        room["updated_at"] = time.time()
        _bump_position(room)
        _touch(room, seat)
        _maybe_ai_unlocked(room)
        return {"player_id": player_id, "room": snapshot_unlocked(room, player_id)}


def leave_room(code: Any, player_id: Any) -> dict[str, Any] | None:
    """Leave a room. A player who leaves mid-game forfeits it."""
    clean = _clean_code(code)
    with _lock:
        room = _rooms.get(clean)
        if room is None:
            return None
        seat = _find_seat(room, player_id)
        if seat is None:
            return snapshot_unlocked(room, player_id)
        if room["status"] == "playing":
            _finish(room, other(seat), "forfeit")
            room["seats"][seat] = None
        else:
            room["seats"][seat] = None
        room["updated_at"] = time.time()
        return snapshot_unlocked(room, player_id)


def list_open_rooms() -> list[dict[str, Any]]:
    """Rooms still waiting for an opponent, newest activity first."""
    now = time.time()
    with _lock:
        _prune_unlocked(now)
        out = []
        for room in _rooms.values():
            if room["status"] != "waiting" or room["seats"].get(BLACK) is not None:
                continue
            host = room["seats"].get(RED)
            out.append(
                {
                    "code": room["code"],
                    "host": host["name"] if host else "Player",
                    "created_at": room.get("created_at"),
                }
            )
        out.sort(key=lambda r: -float(r.get("created_at") or 0))
        return out


def snapshot(room: dict[str, Any], player_id: Any = None) -> dict[str, Any]:
    with _lock:
        return snapshot_unlocked(room, player_id)


def snapshot_unlocked(room: dict[str, Any], player_id: Any = None) -> dict[str, Any]:
    """A room as one client should see it. Player ids are never included.

    ``legal`` is filled only for the player who is to move, so a spectator (or
    the other seat) never receives the moves on your turn.
    """
    now = time.time()
    seat = _find_seat(room, player_id)
    players: dict[str, Any] = {}
    for color in COLORS:
        p = room["seats"].get(color)
        if not p:
            players[color] = None
            continue
        players[color] = {
            "public_id": _public_id(room, p),
            "name": p["name"],
            "initials": p["initials"],
            "is_ai": bool(p.get("is_ai")),
            "present": bool(p.get("is_ai")) or (now - float(p.get("last_seen") or 0) < PRESENCE_SECONDS),
        }
    legal: list[dict[str, Any]] = []
    if room["status"] == "playing" and seat is not None and seat == room["turn"]:
        legal = [
            {"path": m["path"], "captures": m["captures"], "promotes": m["promotes"]}
            for m in legal_moves(room["board"], seat)
        ]
    you = None
    captured = {}
    for color in COLORS:
        captured[color] = 12 - count(room["board"], other(color))
    if seat:
        p = room["seats"][seat]
        you = {
            "color": seat,
            "public_id": _public_id(room, p),
            "name": p["name"],
            "initials": p["initials"],
            "captured": captured[seat],
            "match_no": room.get("match_no", 1),
        }
    return {
        "code": room["code"],
        "status": room["status"],
        "turn": room["turn"],
        "board": list(room["board"]),
        "winner": room["winner"],
        "result": room["result"],
        "players": players,
        "you": you,
        "legal": legal,
        "last_move": room["last_move"],
        "history_len": len(room["history"]),
        "captured": captured,
        "ai": room.get("ai"),
        "server_time": now,
        "updated_at": room.get("updated_at"),
    }


def _clear_rooms() -> None:
    """Test hook: forget every room."""
    with _lock:
        _rooms.clear()


if __name__ == "__main__":  # a quick self-check / demo
    board = initial_board()
    print("pieces:", count(board, RED), count(board, BLACK))
    mv = choose_move(board, RED, max_depth=4)
    print("ai opens with:", mv)
