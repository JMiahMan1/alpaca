"""Randomized soak of the party engine: hundreds of chaotic actions under a fake clock.

Players join, leave, refresh, spectate, submit at the last second, vote, tap
Next on stale screens and end games, while time jumps so deadlines, grace
windows and auto-advance all fire. After every step the room must still make
sense, and nothing anyone can see may carry a private player id.
"""
from __future__ import annotations

import json
import random

import pytest

from arcade import party as P

ALLOWED = (LookupError, ValueError, PermissionError)


@pytest.fixture
def fake_clock(monkeypatch):
    now = [1_000_000.0]
    monkeypatch.setattr(P.time, "time", lambda: now[0])
    monkeypatch.setattr(P, "_generate_ai_prompt", lambda kind: None)
    monkeypatch.setattr(P, "show_stage_on_screen", lambda *a, **k: {"success": True})
    with P._lock:
        P._rooms.clear()
    yield now
    with P._lock:
        P._rooms.clear()


def _check(code: str, snap: dict, viewer: str | None) -> None:
    room = P._rooms.get(code)
    if room is None:
        return
    text = json.dumps(snap)
    for pid in room["players"]:
        if pid.startswith("fz_"):
            assert f'"{pid}"' not in text, f"private id {pid} leaked in {room['phase']}"
    assert all(p["score"] >= 0 for p in snap["players"])
    seated = [p for p in room["players"].values() if not p.get("is_bot") and not p.get("spectator")]
    assert len(seated) <= P.MAX_CONTESTANTS
    assert snap["round"] <= snap["total_rounds"]
    ids = [o["option_id"] for o in snap["options"]]
    assert len(ids) == len(set(ids))
    if snap["phase"] == "voting":
        assert snap["options"]
    if viewer and viewer in room["players"]:
        assert snap["you"] is not None
    if room["phase"] in ("writing", "voting", "reveal") and room.get("schedule"):
        assert (room.get("current_prompt") or {}).get("kind") == P.round_kind(room, room["round"])


@pytest.mark.parametrize("seed", [11, 23, 37])
def test_the_engine_survives_chaos(fake_clock, seed):
    rng = random.Random(seed)
    finished = 0
    for g in range(60):
        code = f"S{seed}{g:03d}"
        people = [f"fz_{g}_{i}" for i in range(rng.randint(1, 10))]
        host = people[0]
        P.create_or_join_room(code=code, player_id=host, name="Host")
        for _ in range(rng.randint(80, 220)):
            fake_clock[0] += rng.choice([0.2, 0.5, 1, 2, 5, 15, 40])
            who, act = rng.choice(people), rng.random()
            try:
                if act < 0.12:
                    P.create_or_join_room(code=code, player_id=who, name=who[-3:], create_if_missing=False,
                                          avatar=rng.choice(P.PLAYER_AVATARS), spectator=rng.random() < 0.05)
                elif act < 0.17:
                    P.start_match(code, player_id=rng.choice([host, who, None]))
                elif act < 0.27:
                    r = P._rooms.get(code)
                    exp = {"phase": r["phase"], "round": r["round"] - (rng.random() < 0.3),
                           "match_id": r.get("match_id")} if r and rng.random() < 0.7 else None
                    P.advance_phase(code, player_id=rng.choice([host, who]), expect=exp)
                elif act < 0.47:
                    P.submit_answer(code, player_id=who, text=rng.choice(["", "a camel", "Noah's snacks"]),
                                    use_decoy=rng.random() < 0.15)
                elif act < 0.67:
                    snap = P.get_room_state(code, who)
                    opts = [o for o in snap["options"] if not o.get("is_mine")]
                    if opts:
                        P.submit_vote(code, player_id=who, option_id=rng.choice(opts)["option_id"])
                elif act < 0.72:
                    P.leave_room(code, player_id=who)
                elif act < 0.73:
                    P.end_room(code, player_id=rng.choice([host, who]))
                else:
                    stage = rng.random() < 0.3
                    viewer = None if stage else who
                    _check(code, P.get_room_state(code, viewer, stage=stage), viewer)
                r = P._rooms.get(code)
                if r and r.get("host_id"):
                    host = r["host_id"]
            except ALLOWED:
                pass
            r = P._rooms.get(code)
            if r is None:
                break
            finished += r["phase"] == "finished"
    assert finished, "the soak never reached the final scores; it is not exercising a whole match"
