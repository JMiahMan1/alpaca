"""Whole matches with the real sync gate: phones that report, old pages that do not, a TV that
loses reports, players who wander off and come back. No screen may ever be stuck."""
from __future__ import annotations

import random
import re
import types

import pytest

from arcade import party as P

pytestmark = pytest.mark.party_gate
ALLOWED = (LookupError, ValueError, PermissionError)
STUCK_S = 400.0  # no screen in the game lasts this long


class _Deferred:
    queue: list = []

    def __init__(self, target=None, name=None, daemon=None, args=(), kwargs=None):
        self.f = lambda: target(*args, **(kwargs or {}))

    def start(self):
        _Deferred.queue.append(self.f)


def _drain():
    while _Deferred.queue:
        f = _Deferred.queue.pop(0)
        f()


@pytest.fixture
def world(monkeypatch):
    now = [2_000_000.0]
    monkeypatch.setattr(P.time, "time", lambda: now[0])
    monkeypatch.setattr(P, "threading", types.SimpleNamespace(Thread=_Deferred, Lock=P.threading.Lock))
    monkeypatch.setattr(P, "_generate_ai_prompt", lambda kind: None)
    monkeypatch.setattr(P, "show_stage_on_screen", lambda *a, **k: {"success": True})

    def voice(text, voice=None, **k):
        secs = max(0.5, len(re.findall(r"[A-Za-z']+", text or "")) / 2.6)
        P._spoken_seconds[(voice or P.DEFAULT_ANNOUNCER_VOICE, text)] = secs
        return ("x", b"\0" * 200)

    monkeypatch.setattr(P, "synthesize_show_line", voice)
    with P._lock:
        P._rooms.clear()
    _Deferred.queue.clear()
    yield now
    with P._lock:
        P._rooms.clear()


def _key(s):
    return f"{s['match_id']}:{s['round']}:{s['phase']}" + (f":{s['step']}" if s.get("step") else "")


def _words_s(text):
    return max(0.5, len(re.findall(r"[A-Za-z']+", text or "")) / 2.6)


class TV:
    def __init__(self, rng, drop):
        self.rng, self.drop, self.ready, self.play, self.ended, self.sent = rng, drop, set(), {}, set(), {}

    def say(self, code, key, ev, now):
        self.sent[(key, ev)] = now
        if self.rng.random() < self.drop:
            return  # lost on the way
        P.stage_said(code, key, ev)

    def tick(self, code, now):
        s = P.get_room_state(code, None, stage=True, rtt="40", late="0")
        key = _key(s)
        if s.get("waiting") and s.get("go_in") is None and key not in self.ready:
            self.ready.add(key)
            self.say(code, key, "ready", now)
        if not s.get("waiting") and key not in self.play:
            self.play[key] = (now, _words_s(s.get("announce")))
        if key in self.play and key not in self.ended and now >= sum(self.play[key]):
            self.ended.add(key)
            self.say(code, key, "end", now)
        need = s.get("tv_needs")
        if need and (key, need) in self.sent and now - self.sent[(key, need)] >= 3:
            self.say(code, key, need, now)
        return s


def _act(code, pid, s, rng):
    if s.get("waiting") or s.get("listening"):
        return
    me = next((x for x in s["players"] if x.get("id") == s.get("you") or x.get("public_id") == s.get("you")), {})
    if s["phase"] == "writing" and not me.get("submitted"):
        P.submit_answer(code, player_id=pid, text=rng.choice(["a camel", "loaves", "Noah's snacks"]))
    elif s["phase"] == "voting":
        opts = [o for o in s["options"] if not o.get("is_mine")]
        if opts:
            P.submit_vote(code, player_id=pid, option_id=rng.choice(opts)["option_id"])
    elif s["phase"] == "racing" and s.get("race") and not s["race"].get("flash"):
        race = s["race"]
        if race.get("listening") or "item" not in race:
            return
        item = race["item"]
        n = len(item.get("options") or item.get("left") or item.get("items") or [])
        guess = rng.randrange(n) if item["type"] == "pick" else rng.sample(range(n), n)
        P.race_answer(code, pid, race["index"], guess)


@pytest.mark.parametrize("seed", range(12))
def test_matches_never_stall_behind_the_gate(world, seed):
    rng = random.Random(seed)
    code = f"G{seed:03d}"
    n = rng.randint(1, 6)
    people = [f"gp_{seed}_{i}" for i in range(n)]
    stale = {p for p in people[1:] if rng.random() < 0.25}  # old pages: poll, never report
    tv = TV(rng, drop=rng.choice([0.0, 0.2, 0.5])) if rng.random() < 0.7 else None
    P.create_or_join_room(code=code, player_id=people[0], name="Host")
    for p in people[1:]:
        P.create_or_join_room(code=code, player_id=p, name=p[-3:], create_if_missing=False)
    seen_key = {p: "" for p in people}
    away_until = {p: 0.0 for p in people}
    tv_away_until = 0.0
    P.start_match(code, player_id=people[0])
    _drain()
    last_key, since, matches = None, world[0], 0
    for _ in range(40000):
        world[0] += rng.choice([0.1, 0.2, 0.3, 0.5, 1.0])
        now = world[0]
        try:
            if tv and now >= tv_away_until:
                tv.tick(code, now)
                if rng.random() < 0.0005:
                    tv_away_until = now + rng.choice([5, 30, 60])  # the TV drops off Wi-Fi
            for p in people:
                if now < away_until[p] or rng.random() < 0.3:
                    continue
                if rng.random() < 0.0008:
                    away_until[p] = now + rng.choice([10, 45, 90])  # phone locked / walked off
                    continue
                s = P.get_room_state(code, p, ready=None if p in stale else seen_key[p])
                seen_key[p] = _key(s)
                if rng.random() < 0.5:
                    _act(code, p, s, rng)
                if p == P._rooms[code].get("host_id") and s["phase"] == "reveal" and rng.random() < 0.01:
                    P.advance_phase(code, player_id=p, expect={"phase": "reveal", "round": s["round"],
                                                               "match_id": s["match_id"]})
        except ALLOWED:
            pass
        _drain()
        room = P._rooms[code]
        key = P.step_key(room)
        if room["phase"] == "finished":
            matches += 1
            if matches >= 2:
                break
            world[0] += 5
            P.start_match(code, player_id=room["host_id"])  # play again
            _drain()
            key = P.step_key(room)
        if key != last_key:
            last_key, since = key, now
        assert now - since < STUCK_S, (f"stuck on {key} for {now - since:.0f}s; gate={room.get('gate')} "
                                       f"tv={bool(tv)} stale={len(stale)}/{n} race={room.get('race') and {k: v for k, v in room['race'].items() if k != 'items'}}")
    assert matches >= 2, f"only {matches} matches finished; on {last_key}"
