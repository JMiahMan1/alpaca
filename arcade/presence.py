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

import time
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
