"""Shared multiplayer presence for the arcade's room games.

Every room game (Jubilee!, Checkers, Parable Party) has the same two
questions: is a person still at this seat, and is a TV showing the stage? This
module answers both the same way for every game, so the phones of any game can
switch to "hand" mode while a TV shows the board, and back when it goes away.

A stage (the TV page) calls :func:`mark_stage` each time it polls; a phone asks
:func:`stage_connected`. Rooms are plain dicts owned by each game's module and
already guarded by that module's lock.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from typing import Any

#: A TV that has not polled for this long is gone (it polls about once a second).
STAGE_PRESENCE_SECONDS = 8.0
#: A player who has not polled for this long has left the table.
PLAYER_PRESENCE_SECONDS = 45.0


def mark_stage(room: dict[str, Any], now: float | None = None) -> None:
    """Record that a TV stage is showing ``room`` right now."""
    room["stage_seen_at"] = time.time() if now is None else now


def stage_connected(room: dict[str, Any], now: float | None = None) -> bool:
    """Whether a TV stage showed ``room`` within the last few seconds."""
    seen = float(room.get("stage_seen_at") or 0)
    current = time.time() if now is None else now
    return seen > 0 and current - seen < STAGE_PRESENCE_SECONDS


def player_present(player: dict[str, Any] | None, now: float | None = None,
                   window: float = PLAYER_PRESENCE_SECONDS) -> bool:
    """Whether a person (never the computer) still polls this seat."""
    if not player or player.get("is_ai"):
        return False
    current = time.time() if now is None else now
    return current - float(player.get("last_seen") or 0) < window


# --- Several games at once -------------------------------------------------
#
# Jubilee!, Parable Party and Checkers each keep their own rooms, and any
# number of each can run at the same time. Two things are shared between them:
#
# * Room codes. Every game draws friendly words from the same list, so one
#   game could pick NOAH while another already had it: two different games
#   under one code in the family's "playing now" list and in conversation.
#   Each game registers its rooms here and every picker avoids all of them.
# * TVs. A screen shows one room at a time. When a room ends it takes its
#   stage off the TV it was cast to -- but only if that TV still shows it, so
#   a Parable Party ending never blanks the Jubilee game cast there since.

_code_sources: list[Callable[[], Iterable[str]]] = []
_screen_lock = threading.Lock()
_screen_owner: dict[str, tuple[str, str]] = {}


def register_rooms(source: Callable[[], Iterable[str]]) -> None:
    """Add a game's live room codes (a callable returning them) to the shared pool."""
    if source not in _code_sources:
        _code_sources.append(source)


def codes_in_use() -> set[str]:
    """Every room code any game is using now."""
    codes: set[str] = set()
    for source in list(_code_sources):
        try:
            codes.update(str(c).upper() for c in tuple(source()))
        except Exception:
            continue
    return codes


#: A code picked but not yet in its game's rooms is held this long.
_RESERVE_SECONDS = 30.0
_code_lock = threading.Lock()
_reserved: dict[str, float] = {}


def reserve_code(code: str, now: float | None = None) -> bool:
    """Hold ``code`` for a new room if no game uses or holds it; True when it is
    now ours. Picking and holding happen under one lock, so two games opening
    rooms at the same instant never both take the same word."""
    clean = str(code).upper()
    current = time.time() if now is None else now
    with _code_lock:
        in_use = codes_in_use()
        for held, at in list(_reserved.items()):
            # Expired, or now a real room (which codes_in_use covers anyway).
            if current - at > _RESERVE_SECONDS or held in in_use:
                del _reserved[held]
        if clean in _reserved or clean in in_use:
            return False
        _reserved[clean] = current
        return True


def pick_code(words: Iterable[str], alphabet: str, fallback_prefix: str) -> str:
    """A free room code for any game: a friendly word if one is free, else a
    random four-character code, held for the caller (see :func:`reserve_code`)."""
    import random

    pool = list(words)
    random.shuffle(pool)
    for word in pool:
        if reserve_code(word):
            return word
    for _ in range(200):
        candidate = "".join(random.choices(alphabet, k=4))
        if reserve_code(candidate):
            return candidate
    while True:
        candidate = f"{fallback_prefix}{random.randint(100, 9999)}"
        if reserve_code(candidate):
            return candidate


def _reset_for_tests() -> None:
    with _code_lock:
        _reserved.clear()
    with _screen_lock:
        _screen_owner.clear()


def claim_screen(screen_id: str, game: str, code: str) -> None:
    """``game``'s room ``code`` is on ``screen_id`` now (whatever showed there before is not)."""
    if screen_id:
        with _screen_lock:
            _screen_owner[screen_id] = (game, str(code).upper())


def release_screen(screen_id: str, game: str, code: str) -> bool:
    """Forget the screen if this room still owns it; True when it did (so take it off)."""
    if not screen_id:
        return False
    with _screen_lock:
        if _screen_owner.get(screen_id) == (game, str(code).upper()):
            del _screen_owner[screen_id]
            return True
        return False


def screen_owner(screen_id: str) -> tuple[str, str] | None:
    with _screen_lock:
        return _screen_owner.get(screen_id)
