"""Race rounds for Parable Party: everyone answers at once and the fastest correct answer scores most.

Four kinds, all played on the same clock:

* ``quick_draw``: five clues, four choices each. First correct answer scores most;
  a wrong tap costs points.
* ``lightning``: eight rapid true-or-false statements, smaller stakes.
* ``pair_up``: one board of five pairs to connect. Points for every right pair, and a
  speed bonus for each player who gets the whole board right, fastest first.
* ``in_order``: one board of five things to put in order. Points per right place, the
  same speed bonus for a perfect order.

Multi-clue rounds flash the right answer (and who got it first) for a few seconds
between clues. Everything here is pure: the race is a dict on the room, the time is
passed in, and ``arcade.party`` holds the lock, the players and the scores.
"""

from __future__ import annotations

import random
from typing import Any

try:
    from arcade.party_race_content import IN_ORDER, LIGHTNING, PAIR_UP, QUICK_DRAW, RACE_ROUND_META
except ImportError:  # the arcade image runs with arcade/ on sys.path
    try:
        from party_race_content import (  # type: ignore[no-redef]
            IN_ORDER,
            LIGHTNING,
            PAIR_UP,
            QUICK_DRAW,
            RACE_ROUND_META,
        )
    except ImportError:  # no content: race rounds are simply never scheduled
        IN_ORDER, LIGHTNING, PAIR_UP, QUICK_DRAW, RACE_ROUND_META = [], [], [], [], {}

RACE_KINDS: tuple[str, ...] = ("quick_draw", "lightning", "pair_up", "in_order")
POOLS: dict[str, list[dict[str, Any]]] = {
    "quick_draw": QUICK_DRAW,
    "lightning": LIGHTNING,
    "pair_up": PAIR_UP,
    "in_order": IN_ORDER,
}
ITEMS_PER_ROUND = {"quick_draw": 5, "lightning": 8, "pair_up": 1, "in_order": 1}
#: How long each clue or board stays open (a streamed TV's lag is added on top).
ITEM_SECONDS = {"quick_draw": 15.0, "lightning": 9.0, "pair_up": 50.0, "in_order": 40.0}
#: The right answer stays up this long between clues.
FLASH_SECONDS = 4.0
#: A tap that left the phone as the clock hit zero still counts.
GRACE_SECONDS = 1.0

#: Points by finishing place among the correct answers (then the last value for everyone after).
RANK_POINTS = (1000, 750, 600, 500, 400, 350, 300, 250)
LIGHTNING_RANK_POINTS = (500, 400, 300, 250, 200, 175, 150, 125)
WRONG_PENALTY = {"quick_draw": 250, "lightning": 150}
#: Per right pair or right place on a board.
PART_POINTS = 150

#: How often a scribe gets a clue right, and how long it takes.
BOT_ACCURACY = {"quick_draw": 0.6, "lightning": 0.7, "pair_up": 0.45, "in_order": 0.4}


def is_race(kind: str | None) -> bool:
    return kind in RACE_KINDS


def _rank_points(kind: str, place: int) -> int:
    table = LIGHTNING_RANK_POINTS if kind == "lightning" else RANK_POINTS
    return table[min(place, len(table) - 1)]


def choose(kind: str, last_seen: dict[str, int], used: set[str], rng: random.Random) -> list[dict[str, Any]]:
    """This round's clues: never ones this room has had, then the least recently used anywhere."""
    pool = POOLS[kind]
    ranked = sorted(pool, key=lambda it: (it["id"] in used, last_seen.get(it["id"], -1), rng.random()))
    return ranked[: ITEMS_PER_ROUND[kind]]


def _item(kind: str, raw: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """The playable form of one content entry: choices shuffled, the answer kept on the server."""
    if kind == "lightning":
        return {"type": "pick", "id": raw["id"], "clue": raw["statement"], "options": ["True", "False"],
                "answer": 0 if raw["true"] else 1, "note": raw.get("note", "")}
    if kind == "quick_draw":
        order = list(range(len(raw["options"])))
        rng.shuffle(order)
        return {"type": "pick", "id": raw["id"], "clue": raw["clue"], "options": [raw["options"][i] for i in order],
                "answer": order.index(int(raw["answer"])), "note": raw.get("note", "")}
    if kind == "pair_up":
        lefts = [p[0] for p in raw["pairs"]]
        rights = [p[1] for p in raw["pairs"]]
        order = list(range(len(rights)))
        while len(order) > 1 and order == sorted(order):
            rng.shuffle(order)
        shuffled = [rights[i] for i in order]
        return {"type": "pairs", "id": raw["id"], "clue": raw["title"], "left": lefts, "right": shuffled,
                "left_label": raw.get("left_label", ""), "right_label": raw.get("right_label", ""),
                "answer": [order.index(i) for i in range(len(lefts))], "note": raw.get("note", "")}
    # in_order: content lists the items in the right order
    items = list(raw["items"])
    order = list(range(len(items)))
    while len(order) > 1 and order == sorted(order):
        rng.shuffle(order)
    shuffled = [items[i] for i in order]
    return {"type": "order", "id": raw["id"], "clue": raw["title"], "items": shuffled,
            "order_label": raw.get("order_label", "First → Last"),
            "answer": [order.index(i) for i in range(len(items))], "note": raw.get("note", "")}


def new_race(kind: str, raws: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    items = [_item(kind, raw, rng) for raw in raws]
    return {"kind": kind, "items": items, "index": 0, "started_at": 0.0, "deadline": 0.0, "flash_until": 0.0,
            "answers": [{} for _ in items], "bot_plan": [{} for _ in items], "done": False}


def open_after(race: dict[str, Any], now: float, read_s: float) -> None:
    """Hold the clue closed while it is read aloud: nobody answers before hearing it.

    The answer clock, and every scribe's plan, start when it opens.
    """
    read_s = max(0.0, read_s)
    race["open_at"] = now + read_s
    race["deadline"] = float(race["deadline"]) + read_s
    for plan in race["bot_plan"][race["index"]].values():
        plan["at"] = float(plan["at"]) + read_s


def start_item(race: dict[str, Any], now: float, delay: float, bot_ids: list[str], rng: random.Random) -> None:
    """Open the current clue: the clock starts now, plus the TV's lag, and the scribes plan their answers."""
    kind, idx = race["kind"], race["index"]
    seconds = ITEM_SECONDS[kind]
    race["started_at"] = now
    race["deadline"] = now + seconds + delay
    race["flash_until"] = 0.0
    race["open_at"] = now + delay  # until the host says how long reading it takes
    item = race["items"][idx]
    plan: dict[str, dict[str, Any]] = {}
    for pid in bot_ids:
        right = rng.random() < BOT_ACCURACY[kind]
        at = now + delay + rng.uniform(min(2.5, seconds * 0.3), seconds * 0.9)
        plan[pid] = {"at": at, "choice": _bot_choice(item, right, rng)}
    race["bot_plan"][idx] = plan


def _bot_choice(item: dict[str, Any], right: bool, rng: random.Random) -> Any:
    if item["type"] == "pick":
        if right:
            return item["answer"]
        return rng.choice([i for i in range(len(item["options"])) if i != item["answer"]])
    choice = list(item["answer"])
    if not right and len(choice) > 1:
        a, b = rng.sample(range(len(choice)), 2)
        choice[a], choice[b] = choice[b], choice[a]
    return choice


def _valid_choice(item: dict[str, Any], choice: Any) -> bool:
    if item["type"] == "pick":
        return isinstance(choice, int) and not isinstance(choice, bool) and 0 <= choice < len(item["options"])
    n = len(item["answer"])
    return (isinstance(choice, list) and len(choice) == n
            and all(isinstance(c, int) and not isinstance(c, bool) for c in choice) and sorted(choice) == list(range(n)))


def _grade(item: dict[str, Any], choice: Any) -> tuple[bool, int]:
    """(all right, how many parts right)."""
    if item["type"] == "pick":
        ok = choice == item["answer"]
        return ok, int(ok)
    parts = sum(1 for got, want in zip(choice, item["answer"], strict=True) if got == want)
    return parts == len(item["answer"]), parts


def answer(race: dict[str, Any], pid: str, index: int, choice: Any, now: float) -> dict[str, Any]:
    """Record one player's answer to the open clue; the first one stands."""
    if race.get("done") or index != race["index"]:
        raise ValueError("That question has already closed.")
    if race["flash_until"] or now > race["deadline"] + GRACE_SECONDS:
        raise ValueError("Time's up on that one!")
    if now < float(race.get("open_at") or 0) - 0.3:
        raise ValueError("Wait for the question!")
    item = race["items"][index]
    if not _valid_choice(item, choice):
        raise ValueError("That answer doesn't fit this question.")
    answers = race["answers"][index]
    if pid in answers:
        return answers[pid]
    correct, parts = _grade(item, choice)
    answers[pid] = {"choice": choice, "at": now, "correct": correct, "parts": parts}
    return answers[pid]


def _apply_bots(race: dict[str, Any], now: float, force: bool = False) -> None:
    idx = race["index"]
    item = race["items"][idx]
    for pid, plan in race["bot_plan"][idx].items():
        if pid in race["answers"][idx]:
            continue
        if plan["at"] <= now or force:
            correct, parts = _grade(item, plan["choice"])
            race["answers"][idx][pid] = {"choice": plan["choice"], "at": min(plan["at"], now),
                                         "correct": correct, "parts": parts}


def tick(race: dict[str, Any], now: float, human_ids: list[str], delay: float, bot_ids: list[str],
         rng: random.Random) -> bool:
    """Move the race along. True once the last clue has closed (time to score)."""
    if race.get("done"):
        return True
    if race["flash_until"]:
        if now < race["flash_until"]:
            return False
        race["index"] += 1
        start_item(race, now, delay, bot_ids, rng)
        return False
    _apply_bots(race, now)
    answers = race["answers"][race["index"]]
    everyone_in = bool(human_ids) and all(pid in answers for pid in human_ids)
    if not (everyone_in or now >= race["deadline"] + GRACE_SECONDS):
        return False
    # The clue closes. A scribe still thinking answers now (it was going to anyway).
    _apply_bots(race, now, force=True)
    if race["index"] + 1 < len(race["items"]):
        # A streamed TV shows the answer `delay` late; the phones wait for it, then
        # everyone gets the full flash.
        race["flash_at"] = now
        race["flash_until"] = now + FLASH_SECONDS + delay
        return False
    race["done"] = True
    return True


def item_result(race: dict[str, Any], index: int) -> dict[str, Any]:
    """Who got clue ``index`` right, fastest first, and who got it wrong."""
    answers = race["answers"][index]
    right = sorted((pid for pid, a in answers.items() if a["correct"]), key=lambda p: answers[p]["at"])
    wrong = [pid for pid, a in answers.items() if not a["correct"]]
    return {"right": right, "wrong": wrong}


def score(race: dict[str, Any], mult: int = 1) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Points per player for the whole round, and the moments worth announcing.

    Events name players by id; the caller fills in names.
    """
    kind = race["kind"]
    deltas: dict[str, int] = {}
    events: list[dict[str, Any]] = []
    for idx, item in enumerate(race["items"]):
        answers = race["answers"][idx]
        res = item_result(race, idx)
        if item["type"] == "pick":
            for place, pid in enumerate(res["right"]):
                gain = _rank_points(kind, place) * mult
                deltas[pid] = deltas.get(pid, 0) + gain
                if place == 0:
                    events.append({"type": "race_first", "player_id": pid, "delta": gain, "item": idx})
            penalty = WRONG_PENALTY.get(kind, 0) * mult
            for pid in res["wrong"]:
                deltas[pid] = deltas.get(pid, 0) - penalty
                events.append({"type": "race_wrong", "player_id": pid, "delta": -penalty, "item": idx})
        else:
            for pid, a in answers.items():
                deltas[pid] = deltas.get(pid, 0) + a["parts"] * PART_POINTS * mult
            for place, pid in enumerate(res["right"]):
                bonus = _rank_points(kind, place) * mult
                deltas[pid] = deltas.get(pid, 0) + bonus
                events.append({"type": "race_perfect", "player_id": pid, "delta": bonus, "item": idx, "place": place})
    return deltas, events


def correct_text(item: dict[str, Any]) -> str:
    if item["type"] == "pick":
        return str(item["options"][item["answer"]])
    if item["type"] == "pairs":
        return "; ".join(f"{item['left'][i]}: {item['right'][r]}" for i, r in enumerate(item["answer"]))
    return " → ".join(item["items"][i] for i in item["answer"])


def public_item(item: dict[str, Any], reveal: bool) -> dict[str, Any]:
    # No "id": the content ids name the answer ("qd_david_sling").
    out = {k: v for k, v in item.items() if k not in ("answer", "note", "id")}
    if reveal:
        out["answer"] = item["answer"]
        out["note"] = item.get("note", "")
        out["answer_text"] = correct_text(item)
    return out
