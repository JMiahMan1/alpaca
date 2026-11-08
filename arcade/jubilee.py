"""Jubilee! - a Bible-story property game for the family arcade.

A clone of the classic 1935 property-trading game (roll two dice, walk the
board, buy the streets, charge rent, build houses, go bankrupt trying) with
Hasbro's name, artwork, corners and Atlantic City street names left behind.
Everything here is the Bible's story: the board is the Holy Land, the decks
are Parable (the "chance" slot) and Scripture (the "community chest" slot),
and the play pieces are the Ark, the Dove, the Menorah, the Star, the Scroll,
the Fish, the Cup and the Trumpet.

Rules implemented (classic, the way the official rulebook plays):

* 2 to 8 players. $1500 to start. Jubilee! (the "go" corner) pays $200 for
  passing it. Roll two dice; doubles grant an extra roll; three doubles in a
  row send you to the Dungeon (the "jail").
* Unowned property may be bought at the marked price. If the player whose
  turn it is declines, the Bank auctions it to the room (any player may bid,
  from any opening price, including the player who declined).
* Streets come in colour groups; the full group must be owned to build, and
  houses must be built evenly (no property may have more than one house
  ahead of another in the group). Five houses become a hotel. The Bank only
  has 32 houses and 12 hotels; when it runs out, building waits.
* Rent on an unimproved street doubles while its colour group is complete.
  Railroads rent by how many the same owner holds ($25/$50/$100/$200).
  Utilities rent by the dice (4x, or 10x when both are owned).
* Mortgage unimproved property for half its price; lift the mortgage at
  mortgage + 10% interest. Mortgaged property collects no rent. Buildings
  must be sold (at half price, evenly) before their colour group can be
  mortgaged. You may not trade property that still has buildings on it.
* The Dungeon: sent there by the SENT TO THE DUNGEON corner, by a card, or by
  three doubles. $200 is not collected when sent to the Dungeon. Exit by
  doubles rolled within three turns, by paying $50 before rolling, or by
  playing a Get Out of the Dungeon Free card. While in the Dungeon you may
  still buy, build, sell, mortgage and collect rent. Rolling doubles to
  escape does not grant an extra turn.
* The Tribute (the "income tax" corner) is $200 - or 10% of your net worth,
  whichever the player prefers, chosen before the total is computed.
* Bankruptcy: a debt owed to a player hands your assets to that player
  (buildings are sold at half price to raise cash first). A debt owed to the
  Bank removes the property from the game (it returns to the Bank unmortgaged,
  buildings returned). The last player standing wins.
* Player-to-player loans are not allowed. The Bank lends only by mortgaging.
* Free Parking (SABBATH REST) does nothing in classic rules - the jackpot
  variant is an opt-in house rule.

Rulesets (chosen when the room is created, like "options at the start"):

* **classic** - the rules above, with three toggleable house rules:
  ``jackpot`` (Sabbath Rest collects every tax and fine paid into a pot),
  ``double_go`` (Jubilee! pays $400) and ``no_auction`` (declined property
  returns to the Bank instead of being auctioned).
* **familyfun** - forgiving: unpayable debts are forgiven (no bankruptcy
  elimination), no auctions, the Sabbath jackpot is on, and the game ends
  after 12 full rounds with the highest net worth winning.
* **quick** - two bankruptcies end the game (or the last player standing),
  and the highest net worth wins when it does.

The engine is pure Python and in-memory, like checkers: rooms live behind a
code, expire on their own, and are snapshotted per viewer (player ids never
leave the server). AI "scribes" (Barnabas, Priscilla, Apollos, Lydia, Silas)
take empty seats when fewer than two humans are seated, or when a seat is
left to the computer.
"""

from __future__ import annotations

import importlib
import random
import re
import secrets
import threading
import time
from typing import Any

# Parable Party is the arcade's room engine; Jubilee! reuses its friendly
# room words and one-way public ids. Dual-mode import, the same seam the
# checkers engine uses.
try:
    party = importlib.import_module("party")
except ImportError:  # tests import `arcade.jubilee` as a package from the repo root
    party = importlib.import_module("arcade.party")

# --------------------------------------------------------------------------- #
# The board
# --------------------------------------------------------------------------- #
# rents: [base, 1 house, 2 houses, 3 houses, 4 houses, hotel]. Mortgage is
# half the price. house_cost == price // 4 on the classic US board.
#
# Bible-world street names, classic economics:
#   brown   Bethlehem, Nazareth ($60) · light blue Sychar, Capernaum ($100), Cana ($120)
#   pink    Bethany, Emmaus ($140), Jericho ($160) · orange Gibeon, Hebron ($180), Shechem ($200)
#   red     Damascus, Antioch ($220), Ephesus ($240) · yellow Corinth, Philippi ($260), Thessalonica ($280)
#   green   Colossae, Smyrna ($300), Pergamos ($320) · blue Zion Hill ($350), The New Jerusalem ($400)

_STREETS: list[dict[str, Any]] = [
    # (index, name, group, price, rents, house_cost, mortgage)
    {"i": 1,  "kind": "street", "name": "Bethlehem",          "group": "brown",     "price": 60,  "rents": [2, 10, 30, 90, 160, 250],    "house_cost": 50,  "mortgage": 30},
    {"i": 3,  "kind": "street", "name": "Nazareth",           "group": "brown",     "price": 60,  "rents": [4, 20, 60, 180, 320, 450],    "house_cost": 50,  "mortgage": 30},
    {"i": 6,  "kind": "street", "name": "Sychar",             "group": "lightblue", "price": 100, "rents": [6, 30, 90, 270, 400, 550],    "house_cost": 50,  "mortgage": 50},
    {"i": 8,  "kind": "street", "name": "Capernaum",          "group": "lightblue", "price": 100, "rents": [6, 30, 90, 270, 400, 550],    "house_cost": 50,  "mortgage": 50},
    {"i": 9,  "kind": "street", "name": "Cana",               "group": "lightblue", "price": 120, "rents": [8, 40, 100, 300, 450, 600],   "house_cost": 50,  "mortgage": 60},
    {"i": 11, "kind": "street", "name": "Bethany",            "group": "pink",      "price": 140, "rents": [10, 50, 150, 450, 625, 750],  "house_cost": 100, "mortgage": 70},
    {"i": 13, "kind": "street", "name": "Emmaus",             "group": "pink",      "price": 140, "rents": [10, 50, 150, 450, 625, 750],  "house_cost": 100, "mortgage": 70},
    {"i": 14, "kind": "street", "name": "Jericho",            "group": "pink",      "price": 160, "rents": [12, 60, 180, 500, 700, 900],  "house_cost": 100, "mortgage": 80},
    {"i": 16, "kind": "street", "name": "Gibeon",             "group": "orange",    "price": 180, "rents": [14, 70, 200, 550, 750, 950],  "house_cost": 100, "mortgage": 90},
    {"i": 18, "kind": "street", "name": "Hebron",             "group": "orange",    "price": 180, "rents": [14, 70, 200, 550, 750, 950],  "house_cost": 100, "mortgage": 90},
    {"i": 19, "kind": "street", "name": "Shechem",            "group": "orange",    "price": 200, "rents": [16, 80, 220, 600, 800, 1000], "house_cost": 100, "mortgage": 100},
    {"i": 21, "kind": "street", "name": "Damascus",           "group": "red",       "price": 220, "rents": [18, 90, 250, 700, 875, 1050], "house_cost": 150, "mortgage": 110},
    {"i": 23, "kind": "street", "name": "Antioch",             "group": "red",       "price": 220, "rents": [18, 90, 250, 700, 875, 1050], "house_cost": 150, "mortgage": 110},
    {"i": 24, "kind": "street", "name": "Ephesus",            "group": "red",       "price": 240, "rents": [20, 100, 300, 750, 925, 1100],"house_cost": 150, "mortgage": 120},
    {"i": 26, "kind": "street", "name": "Corinth",            "group": "yellow",    "price": 260, "rents": [22, 110, 330, 800, 975, 1150],"house_cost": 150, "mortgage": 130},
    {"i": 27, "kind": "street", "name": "Philippi",           "group": "yellow",    "price": 260, "rents": [22, 110, 330, 800, 975, 1150],"house_cost": 150, "mortgage": 130},
    {"i": 29, "kind": "street", "name": "Thessalonica",       "group": "yellow",    "price": 280, "rents": [24, 120, 360, 850, 1025, 1200],"house_cost": 150, "mortgage": 140},
    {"i": 31, "kind": "street", "name": "Colossae",           "group": "green",     "price": 300, "rents": [26, 130, 390, 900, 1100, 1275],"house_cost": 200, "mortgage": 150},
    {"i": 32, "kind": "street", "name": "Smyrna",             "group": "green",     "price": 300, "rents": [26, 130, 390, 900, 1100, 1275],"house_cost": 200, "mortgage": 150},
    {"i": 34, "kind": "street", "name": "Pergamos",           "group": "green",     "price": 320, "rents": [28, 150, 450, 1000, 1200, 1400],"house_cost": 200, "mortgage": 160},
    {"i": 37, "kind": "street", "name": "Zion Hill",          "group": "blue",      "price": 350, "rents": [35, 175, 500, 1100, 1300, 1500],"house_cost": 200, "mortgage": 175},
    {"i": 39, "kind": "street", "name": "The New Jerusalem",  "group": "blue",      "price": 400, "rents": [50, 200, 600, 1400, 1700, 2000],"house_cost": 200, "mortgage": 200},
]

_RAILROADS = [
    {"i": 5,  "kind": "railroad", "name": "The King's Highway",  "price": 200, "mortgage": 100},
    {"i": 15, "kind": "railroad", "name": "The Via Maris",       "price": 200, "mortgage": 100},
    {"i": 25, "kind": "railroad", "name": "The Way",            "price": 200, "mortgage": 100},
    {"i": 35, "kind": "railroad", "name": "The Desert Caravan", "price": 200, "mortgage": 100},
]
_UTILITIES = [
    {"i": 12, "kind": "utility", "name": "The Lamp", "price": 150, "mortgage": 75},
    {"i": 28, "kind": "utility", "name": "The Well", "price": 150, "mortgage": 75},
]

SPACES: list[dict[str, Any]] = [None] * 40  # index → space dict
for _s in _STREETS + _RAILROADS + _UTILITIES:
    SPACES[_s["i"]] = dict(_s)
SPACES[0] = {"i": 0, "kind": "go", "name": "Jubilee!", "sub": "The year of release"}
_FIXED: list[tuple[int, str, str | None, Any]] = [
    (2,  "scripture", "SCRIPTURE", None),
    (4,  "tribute",   "The Tribute", {"flat": 200}),
    (7,  "parable",   "PARABLE", None),
    (10, "dungeon",   "The Dungeon", {"sub": "Paul & Silas"}),
    (17, "scripture", "SCRIPTURE", None),
    (20, "sabbath",  "Sabbath Rest", {"sub": "Free parking, or the jackpot"}),
    (22, "parable",   "PARABLE", None),
    (30, "to_dungeon","Sent to the Dungeon", None),
    (33, "scripture", "SCRIPTURE", None),
    (36, "parable",   "PARABLE", None),
    (38, "offering",  "The Offering", {"amount": 100}),
]
for _i, _kind, _name, _extra in _FIXED:
    SPACES[_i] = {"i": _i, "kind": _kind, "name": _name, **(_extra or {})}
assert all(SPACES[i] is not None for i in range(40))
_STREET_INDICES = [s["i"] for s in _STREETS]
_GROUPS: dict[str, list[int]] = {}
for _s in _STREETS:
    _GROUPS.setdefault(_s["group"], []).append(_s["i"])

TOKENS = ("ark", "dove", "menorah", "star", "scroll", "fish", "cup", "trumpet")
#: The scribes who sit in for empty seats.
SCRIBES = (
    {"name": "Barnabas",  "initials": "BAR", "avatar": "🧔🏽"},
    {"name": "Priscilla", "initials": "PRI", "avatar": "🧕🏽"},
    {"name": "Apollos",   "initials": "APO", "avatar": "📜"},
    {"name": "Lydia",     "initials": "LYD", "avatar": "🧶"},
    {"name": "Silas",     "initials": "SIL", "avatar": "⚒️"},
    {"name": "Mark",      "initials": "MRK", "avatar": "✍️"},
    {"name": "Phoebe",    "initials": "PHO", "avatar": "🕊️"},
    {"name": "Aquila",    "initials": "AQL", "avatar": "⛺"},
)

START_CASH = 1500
GO_SALARY = 200
BANK_HOUSES = 32
BANK_HOTELS = 12
JUBILEE_ROUND_LIMIT = 12
QUICK_END_BANKRUPTCIES = 2
MAX_PLAYERS = 8

# --------------------------------------------------------------------------- #
# The decks - PARABLE (chance) and SCRIPTURE (community chest)
# --------------------------------------------------------------------------- #
# Card kinds:
#   advance          -> move to space "to"; collect GO if passed (unless jail)
#   advance_go       -> move to 0, collect GO
#   advance_nearest  -> move to nearest "kind" (railroad|utility); then double
#                       rent (railroad, owned) or 10x dice (utility, owned);
#                       unowned -> the usual buy-or-auction flow
#   advance_street   -> move to space "to" (street); usual landing resolution
#   back             -> move back "spaces" spaces, resolve the new landing
#   money            -> bank pays/charges "amount" (negative = pay the bank);
#                       charge may exceed cash -> debt to the Bank
#   per_building     -> pay "per_house"/"per_hotel" per building you own
#   each_player      -> pay each living opponent "amount" (negative = receive)
#   birthday         -> receive "amount" from every living opponent
#   repairs          -> pay "per_house"/"per_hotel" per building you own
#   jail             -> go to the Dungeon (no GO salary)
#   jail_card        -> Get Out of the Dungeon Free card
#
# Economics are the classic decks'; the scripture is the wrapper.

PARABLE_DECK: list[dict[str, Any]] = [
    {"t": "advance_go", "text": "“The year of Jubilee: return to your property.” - Advance to Jubilee! (Leviticus 25:13)"},
    {"t": "money", "amount": 50, "text": "Your figs and wine bring in coin - the bank pays you $50. (Proverbs 27:18)"},
    {"t": "jail_card", "text": "A letter sealed with the jailer's ring - Get Out of the Dungeon Free. (Acts 16:26)"},
    {"t": "money", "amount": 150, "text": "Your building loan matures - collect $150. (Ecclesiastes 11:10)"},
    {"t": "back", "spaces": 3, "text": "“The adversary prowls like a roaring lion.” Go back three spaces. (1 Peter 5:8)"},
    {"t": "jail", "text": "Caught preaching without a permit in Philippi - go directly to the Dungeon. Do not pass Jubilee! (Acts 16:24)"},
    {"t": "per_building", "per_house": 25, "per_hotel": 100, "text": "Repair every roof and well in your lands - pay $25 per house, $100 per hotel. (Luke 14:28)"},
    {"t": "money", "amount": -15, "text": "Give to the poor, as the scribe did. Pay $15. (Matthew 19:21)"},
    {"t": "advance_street", "to": 5, "text": "Take the King's Highway to Damascus - advance to The King's Highway. (2 Kings 20:13)"},
    {"t": "advance_nearest", "kind": "railroad", "text": "Every road from Jerusalem runs through Rome - advance to the nearest toll road. (Acts 25:12)"},
    {"t": "advance_nearest", "kind": "utility", "text": "The oil and wine of the Good Samaritan - advance to The Well or The Lamp. If owned, throw the dice and pay ten times what you throw. (Luke 10:34)"},
    {"t": "advance_street", "to": 24, "text": "A certain man of the assembly at Ephesus summoned you - advance to Ephesus. (Acts 19:34)"},
    {"t": "each_player", "amount": 50, "text": "You are made an elder of the church - pay every player $50. (1 Timothy 3:1)"},
    {"t": "money", "amount": 100, "text": "Your psalm wins first prize at the festival - collect $100. (Psalm 149:3)"},
    {"t": "advance_street", "to": 37, "text": "Zion, fair and tall - advance to Zion Hill. (Psalm 48:2)"},
    {"t": "advance_street", "to": 14, "text": "The road down to Jericho is not safe - but you must go. Advance to Jericho. (Luke 10:30)"},
]

SCRIPTURE_DECK: list[dict[str, Any]] = [
    {"t": "advance_go", "text": "“Return in the year of Jubilee to your property.” - Advance to Jubilee! (Leviticus 25:13)"},
    {"t": "money", "amount": 200, "text": "“He opens his hand and satisfies the desire of every living thing.” The bank pays you $200. (Psalm 145:16)"},
    {"t": "money", "amount": -50, "text": "“A wounded man is carried by a donkey to the inn, and paid two denarii.” Pay $50. (Luke 10:35)"},
    {"t": "jail_card", "text": "“Paul and Silas prayed - and the prison doors flew open.” Get Out of the Dungeon Free. (Acts 16:26)"},
    {"t": "money", "amount": 50, "text": "“She sold her harvest and brought the silver.” Collect $50 from the sale of your flock. (Ruth 2:3)"},
    {"t": "jail", "text": "“And when they came out of the prison, the doors stood open.” You were not so lucky - go to the Dungeon. (Acts 16:26)"},
    {"t": "money", "amount": 100, "text": "“Come, eat my bread.” Your feast at the scribe's house is repaid - collect $100. (Luke 5:29)"},
    {"t": "money", "amount": 20, "text": "“Render to Caesar what is Caesar's.” Your tax comes due - and returns. Collect $20. (Mark 12:17)"},
    {"t": "birthday", "amount": 10, "text": "It is your birthday - “Everyone who calls on the Lord shall be saved.” Collect $10 from every player. (Romans 10:13)"},
    {"t": "money", "amount": 100, "text": "A gift from a distant relative's storehouse - collect $100. (Ruth 2:18)"},
    {"t": "money", "amount": -100, "text": "The wounded man stays one more day at the inn - pay $100. (Luke 10:35)"},
    {"t": "money", "amount": -50, "text": "“Train up a child in the way he should go.” Pay school fees $50. (Proverbs 22:6)"},
    {"t": "money", "amount": 25, "text": "“A word in due season - how good it is.” Your counsel is sought at the gate. Collect $25. (Proverbs 15:23)"},
    {"t": "per_building", "per_house": 40, "per_hotel": 115, "text": "“A wise man built his house upon the rock.” Repair every house you own - pay $40 per house, $115 per hotel. (Matthew 7:24)"},
    {"t": "money", "amount": 10, "text": "Your fig tree takes second prize at the harvest festival - collect $10. (Matthew 21:19)"},
    {"t": "money", "amount": 100, "text": "“He has filled the hungry with good things.” You inherit a cloak and a pot of oil. Collect $100. (Luke 1:53)"},
]

# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _clean_code(raw: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(raw or "")).upper()[:8]


def _clean_text(raw: Any, max_len: int = 24) -> str:
    return re.sub(r"\s+", " ", str(raw or "")).strip()[:max_len]


def _clean_initials(raw: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(raw or ""))[:3].upper() or "YOU"


def _clean_device(raw: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "", str(raw or ""))[:64]


_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def _pick_code_unlocked() -> str:
    words = [w for w in party.ROOM_WORDS if w not in _rooms]
    if words:
        return random.choice(words)
    for _ in range(60):
        candidate = "".join(random.choices(_CODE_ALPHABET, k=4))
        if candidate not in _rooms:
            return candidate
    return f"JU{random.randint(100, 999)}"


def _new_player(name: Any, initials: Any, token: Any, device_id: Any = None) -> dict[str, Any]:
    now = time.time()
    player: dict[str, Any] = {
        "id": secrets.token_hex(16),
        "name": _clean_text(name) or "Player",
        "initials": _clean_initials(initials),
        "token": _clean_token(token),
        "is_ai": False,
        "cash": START_CASH,
        "position": 0,
        "jail": None,  # {"turns": int} while in the Dungeon
        "jail_cards": 0,
        "bankrupt": False,
        "joined_at": now,
        "last_seen": now,
    }
    marker = _clean_device(device_id)
    if marker:
        player["device_id"] = marker
    return player


def _ai_player(name: str, initials: str, avatar: str, token: str) -> dict[str, Any]:
    now = time.time()
    return {
        "id": secrets.token_hex(16),
        "name": name,
        "initials": initials,
        "avatar": avatar,
        "token": token,
        "is_ai": True,
        "cash": START_CASH,
        "position": 0,
        "jail": None,
        "jail_cards": 0,
        "bankrupt": False,
        "joined_at": now,
        "last_seen": now,
    }


def _clean_token(raw: Any) -> str:
    t = re.sub(r"[^a-z]", "", str(raw or "").lower())
    return t if t in TOKENS else TOKENS[0]


def _find_seat(room: dict[str, Any], player_id: Any) -> int | None:
    if not player_id:
        return None
    for seat, p in enumerate(room["players"]):
        if p and p.get("id") == player_id:
            return seat
    return None


def _human_present(room: dict[str, Any], seat: int, now: float | None = None) -> bool:
    p = room["players"][seat] if 0 <= seat < len(room["players"]) else None
    if not p or p.get("is_ai") or p.get("bankrupt"):
        return False
    now = time.time() if now is None else now
    return now - float(p.get("last_seen") or 0) < PRESENCE_SECONDS


def _device_owns(room: dict[str, Any], device_id: str) -> bool:
    if not device_id:
        return False
    return any(
        p and not p.get("is_ai") and p.get("device_id") == device_id
        for p in room["players"]
    )


def _public_id(room: dict[str, Any], player: dict[str, Any] | None) -> str | None:
    if not player:
        return None
    return party.public_id(room, player["id"])


def _touch(room: dict[str, Any], seat: int | None) -> None:
    if seat is None:
        return
    p = room["players"][seat]
    if p and not p.get("is_ai"):
        p["last_seen"] = time.time()


def _next_seat(room: dict[str, Any], from_seat: int) -> int:
    n = len(room["players"])
    for step in range(1, n + 1):
        seat = (from_seat + step) % n
        if not room["players"][seat]["bankrupt"]:
            return seat
    return from_seat


def _active_seats(room: dict[str, Any]) -> list[int]:
    return [i for i, p in enumerate(room["players"]) if not p["bankrupt"]]


PRESENCE_SECONDS = 45.0
ROOM_TTL_SECONDS = 7200
ROOM_IDLE_SECONDS = 2700
ROOM_EMPTY_GRACE_SECONDS = 300


def _prune_unlocked(now: float) -> None:
    stale = []
    for code, room in _rooms.items():
        humans = [p for p in room["players"] if p and not p.get("is_ai")]
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

_rooms: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def _prop(room: dict[str, Any], index: int) -> dict[str, Any]:
    return room["props"][index]


def _group_owner(room: dict[str, Any], group: str, owner: int) -> bool:
    return all(_prop(room, i)["owner"] == owner for i in _GROUPS[group])


def _group_has_buildings(room: dict[str, Any], group: str) -> bool:
    return any(_prop(room, i)["houses"] > 0 for i in _GROUPS[group])


def _count_owned(room: dict[str, Any], owner: int, kind: str) -> int:
    total = 0
    for i, s in enumerate(SPACES):
        if s and s.get("kind") == kind and _prop(room, i)["owner"] == owner:
            total += 1
    return total


def _net_worth(room: dict[str, Any], seat: int) -> int:
    p = room["players"][seat]
    total = int(p["cash"])
    for i, s in enumerate(SPACES):
        if not s:
            continue
        prop = _prop(room, i)
        if prop["owner"] != seat:
            continue
        if prop["mortgaged"]:
            total += int(s.get("mortgage", s.get("price", 0)) // 1)
        else:
            total += int(s.get("price", 0))
        houses = int(prop["houses"])
        house_cost = int(s.get("house_cost", 0))
        total += houses * house_cost + (4 * house_cost if houses == 5 else 0)
    return total


def _building_count(room: dict[str, Any], seat: int) -> tuple[int, int]:
    houses = hotels = 0
    for i, s in enumerate(SPACES):
        if not s:
            continue
        prop = _prop(room, i)
        if prop["owner"] == seat:
            if prop["houses"] == 5:
                hotels += 1
            else:
                houses += prop["houses"]
    return houses, hotels


def _rent_for(room: dict[str, Any], index: int, dice: list[int] | None) -> int:
    s = SPACES[index]
    prop = _prop(room, index)
    owner = prop["owner"]
    if owner is None or prop["mortgaged"]:
        return 0
    kind = s["kind"]
    if kind == "railroad":
        n = _count_owned(room, owner, "railroad")
        return [0, 25, 50, 100, 200][n]
    if kind == "utility":
        n = _count_owned(room, owner, "utility")
        roll = sum(dice) if dice else 7
        return roll * (10 if n >= 2 else 4)
    group = s["group"]
    houses = prop["houses"]
    if houses:
        return s["rents"][houses]
    base = s["rents"][0]
    if _group_owner(room, group, owner):
        base *= 2
    return base


def _log(room: dict[str, Any], msg: str) -> None:
    room.setdefault("log", []).append(msg)
    del room["log"][:-10]


def _send_to_jail(room: dict[str, Any], seat: int) -> None:
    p = room["players"][seat]
    p["position"] = 10
    p["jail"] = {"turns": 0}
    room["doubles"] = 0
    _log(room, f"{p['name']} is sent to the Dungeon")


def _draw_card(room: dict[str, Any], deck_key: str) -> dict[str, Any]:
    deck = room["decks"][deck_key]
    if not deck:
        full = PARABLE_DECK if deck_key == "parable" else SCRIPTURE_DECK
        order = list(range(len(full)))
        random.shuffle(order)
        deck = [dict(full[i]) for i in order]
        room["decks"][deck_key] = deck
    return deck.pop()


def _credit(room: dict[str, Any], seat: int, amount: int) -> None:
    p = room["players"][seat]
    p["cash"] = int(p["cash"]) + int(amount)


def _charge(room: dict[str, Any], seat: int, amount: int, creditor: int | None,
            reason: str) -> bool:
    """Charge ``seat`` ``amount``. Returns True if paid in full.

    Feeds the Sabbath jackpot on taxes and fines. If ``creditor`` is a seat,
    the money goes to that player; ``None`` means the Bank (or the pot).
    """
    amount = int(amount)
    p = room["players"][seat]
    rules = room["rules"]
    feed_pot = rules["jackpot"] and creditor is None and reason in ("tax", "fine", "tribute")
    if int(p["cash"]) >= amount:
        if feed_pot:
            room["pot"] += amount
        p["cash"] -= amount
        if creditor is not None:
            _credit(room, creditor, amount)
        return True
    if rules["ruleset"] in ("familyfun",):
        # The forgiving ruleset: pay what you can, forgive the rest.
        paid = int(p["cash"])
        if feed_pot:
            room["pot"] += paid
        p["cash"] = 0
        if creditor is not None:
            _credit(room, creditor, paid)
        _log(room, f"{p['name']} could not cover ${amount} - the scribes forgive the rest")
        return True
    # Not enough cash: the player must raise money (mortgage, sell buildings,
    # trade) and then confirm, or declare bankruptcy.
    room["pending"] = {
        "kind": "debt",
        "amount": amount,
        "creditor": creditor,
        "reason": reason,
    }
    return False


def _advance(room: dict[str, Any], seat: int, to_index: int, collect_go: bool = True) -> None:
    p = room["players"][seat]
    old = int(p["position"])
    new = to_index % 40
    if collect_go and (new < old or to_index >= 40):
        salary = GO_SALARY * (2 if room["rules"]["double_go"] else 1)
        p["cash"] += salary
        _log(room, f"{p['name']} passes Jubilee! and collects ${salary}")
    p["position"] = new


def _maybe_finish(room: dict[str, Any]) -> bool:
    alive = [i for i, p in enumerate(room["players"]) if not p["bankrupt"]]
    n = len(alive)
    rules = room["rules"]
    if n <= 1:
        room["status"] = "finished"
        room["phase"] = "finished"
        room["winner"] = alive[0] if alive else None
        room["result"] = "last-standing"
        return True
    if rules["ruleset"] == "quick" and room.get("bankruptcies", 0) >= QUICK_END_BANKRUPTCIES:
        room["status"] = "finished"
        room["phase"] = "finished"
        room["winner"] = max(alive, key=lambda s: _net_worth(room, s))
        room["result"] = "quick-game"
        return True
    if rules["ruleset"] == "familyfun":
        mins = min(room.get("turn_counts", [0] * len(room["players"]))[s] for s in alive)
        if mins >= JUBILEE_ROUND_LIMIT:
            room["status"] = "finished"
            room["phase"] = "finished"
            room["winner"] = max(alive, key=lambda s: _net_worth(room, s))
            room["result"] = "family-fun-full-rounds"
            return True
    return False


# --------------------------------------------------------------------------- #
# Cards and landing resolution
# --------------------------------------------------------------------------- #

def _refresh_bonus(room: dict[str, Any], seat: int) -> None:
    """Re-evaluate whether the seat may roll again (doubles), once a pending
    state has cleared."""
    if room["pending"] or room["auction"] or room["status"] != "playing":
        room["bonus_roll"] = False
        return
    p = room["players"][seat]
    room["bonus_roll"] = bool(room["doubles"] > 0 and p["jail"] is None)


def _advance_nearest(room: dict[str, Any], seat: int, kind: str) -> int:
    pos = int(room["players"][seat]["position"])
    indices = [s["i"] for s in SPACES if s and s.get("kind") == kind]
    target = min(indices, key=lambda i: (i - pos) % 40)
    wrapped = target <= pos - 1
    _advance(room, seat, target + (40 if wrapped else 0), collect_go=True)
    return target


def _apply_card(room: dict[str, Any], seat: int, card: dict[str, Any], dice: list[int] | None) -> None:
    p = room["players"][seat]
    t = card["t"]

    if t == "advance_go":
        old = int(p["position"])
        wrapped_to = 40  # always collects Jubilee!
        _advance(room, seat, wrapped_to, collect_go=True)
        _log(room, f"{p['name']} advances to Jubilee!")
        _resolve_landing(room, seat, dice)
        return
    if t in ("advance", "advance_street"):
        to = int(card["to"])
        old = int(p["position"])
        wrapped = to <= old
        _advance(room, seat, to + (40 if wrapped else 0), collect_go=True)
        _resolve_landing(room, seat, dice)
        return
    if t == "back":
        _advance(room, seat, int(p["position"]) - int(card["spaces"]), collect_go=False)
        _resolve_landing(room, seat, dice)
        return
    if t == "advance_nearest":
        target = _advance_nearest(room, seat, card["kind"])
        prop = _prop(room, target)
        if card["kind"] == "railroad":
            if prop["owner"] is None:
                room["pending"] = {"kind": "buy", "space": target}
            elif prop["owner"] != seat:
                rent = _rent_for(room, target, dice) * 2
                if rent > 0:
                    _charge(room, seat, rent, creditor=prop["owner"], reason="rent")
        else:  # utility - classic card says 10x the dice, not the usual 4x
            if prop["owner"] is None:
                room["pending"] = {"kind": "buy", "space": target}
            elif prop["owner"] != seat:
                roll_sum = sum(dice) if dice else 7
                _charge(room, seat, roll_sum * 10, creditor=prop["owner"], reason="rent")
        return
    if t == "money":
        amount = int(card["amount"])
        if amount >= 0:
            _credit(room, seat, amount)
        else:
            _charge(room, seat, -amount, None, "card")
        return
    if t in ("per_building", "repairs"):
        houses, hotels = _building_count(room, seat)
        total = houses * int(card["per_house"]) + hotels * int(card["per_hotel"])
        _charge(room, seat, total, None, "card")
        return
    if t == "each_player":
        amount = int(card["amount"])
        opponents = [s for s in _active_seats(room) if s != seat]
        if amount >= 0:
            remaining = int(p["cash"])
            for other in opponents:
                pay = min(amount, remaining)
                if pay <= 0:
                    break
                p["cash"] -= pay
                _credit(room, other, pay)
                remaining -= pay
        else:
            for other in opponents:
                take = min(-amount, int(room["players"][other]["cash"]))
                if take > 0:
                    room["players"][other]["cash"] -= take
                    _credit(room, seat, take)
        return
    if t == "birthday":
        amount = int(card["amount"])
        for other in _active_seats(room):
            if other == seat:
                continue
            take = min(amount, int(room["players"][other]["cash"]))
            if take > 0:
                room["players"][other]["cash"] -= take
                _credit(room, seat, take)
        return
    if t == "jail":
        _send_to_jail(room, seat)
        return
    if t == "jail_card":
        p["jail_cards"] = int(p["jail_cards"]) + 1
        _log(room, f"{p['name']} holds a Dungeon-free card")
        return


def _resolve_landing(room: dict[str, Any], seat: int, dice: list[int] | None) -> None:
    pos = int(room["players"][seat]["position"])
    s = SPACES[pos]
    kind = s["kind"]
    p = room["players"][seat]

    if kind in ("street", "railroad", "utility"):
        prop = _prop(room, pos)
        owner = prop["owner"]
        if owner is None:
            room["pending"] = {"kind": "buy", "space": pos}
        elif owner != seat:
            rent = _rent_for(room, pos, dice)
            if rent > 0:
                _charge(room, seat, rent, creditor=owner, reason="rent")
        return
    if kind == "tribute":
        nw = _net_worth(room, seat)
        room["pending"] = {"kind": "tribute", "flat": int(s.get("flat", 200)), "percent": round(nw * 0.1)}
        return
    if kind == "offering":
        _charge(room, seat, int(s.get("amount", 100)), None, "tax")
        return
    if kind in ("parable", "scripture"):
        card = _draw_card(room, kind)
        _log(room, f"{p['name']} draws - {card['text']}")
        _apply_card(room, seat, card, dice)
        return
    if kind == "sabbath":
        if room["rules"]["jackpot"] and int(room["pot"]) > 0:
            take = int(room["pot"])
            p["cash"] = int(p["cash"]) + take
            room["pot"] = 0
            _log(room, f"{p['name']} rests - and gathers the Sabbath pot of ${take}")
        return
    if kind == "to_dungeon":
        _send_to_jail(room, seat)
        return
    # go / dungeon: nothing further to do


# --------------------------------------------------------------------------- #
# Property management - building, selling, mortgaging
# --------------------------------------------------------------------------- #

def _sellable_building_counts(room: dict[str, Any], seat: int) -> list[int]:
    """Even-build rule reversed: the prop with the most houses may shed one."""
    by_group: dict[str, list[int]] = {}
    for i in _STREET_INDICES:
        if _prop(room, i)["owner"] == seat:
            by_group.setdefault(SPACES[i]["group"], []).append(i)
    sellable: list[int] = []
    for members in by_group.values():
        houses = [int(_prop(room, i)["houses"]) for i in members]
        if max(houses) <= 0:
            continue
        top = max(houses)
        for i, h in zip(members, houses, strict=True):
            if h == top:
                sellable.append(i)
    return sellable


def _build_house_cost(room: dict[str, Any], seat: int, index: int) -> int:
    s = SPACES[index]
    prop = _prop(room, index)
    if prop["houses"] >= 4:
        return int(s["house_cost"])  # 4 houses become a hotel, same price
    return int(s["house_cost"])


def _can_build_on(room: dict[str, Any], seat: int, index: int) -> tuple[bool, str]:
    s = SPACES[index]
    if s.get("kind") != "street":
        return False, "only streets can be built on"
    prop = _prop(room, index)
    if prop["owner"] != seat:
        return False, "you do not own that street"
    if prop["mortgaged"]:
        return False, "unmortgage the street before building"
    group = s["group"]
    if not _group_owner(room, group, seat):
        return False, "you need the whole colour group before building"
    members = _GROUPS[group]
    mine = int(prop["houses"])
    others = min(int(_prop(room, i)["houses"]) for i in members if i != index)
    if mine > others:
        return False, "buildings must go up evenly - this street is already ahead"
    if mine == 5:
        return False, "the hotel is complete"
    if mine == 4:
        if int(room["bank_hotels"]) < 1:
            return False, "the Bank is out of hotels"
    else:
        if int(room["bank_houses"]) < 1:
            return False, "the Bank is out of houses"
    if int(room["players"][seat]["cash"]) < int(s["house_cost"]):
        return False, "not enough cash to build"
    return True, ""


def _build_building(room: dict[str, Any], seat: int, index: int) -> str:
    s = SPACES[index]
    prop = _prop(room, index)
    cost = int(s["house_cost"])
    p = room["players"][seat]
    if int(prop["houses"]) == 4:
        # 4 houses become 1 hotel; the 4 houses return to the Bank
        prop["houses"] = 5
        room["bank_houses"] = int(room["bank_houses"]) + 4
        room["bank_hotels"] = int(room["bank_hotels"]) - 1
    else:
        prop["houses"] = int(prop["houses"]) + 1
        room["bank_houses"] = int(room["bank_houses"]) - 1
    p["cash"] = int(p["cash"]) - cost
    return SPACES[index]["name"]


def _sell_building(room: dict[str, Any], seat: int, index: int, count: int = 1) -> int:
    s = SPACES[index]
    prop = _prop(room, index)
    p = room["players"][seat]
    received = 0
    for _ in range(max(1, int(count))):
        h = int(prop["houses"])
        if h == 5:
            # Sell hotel: houses drop back to 4, hotel returns to the Bank
            prop["houses"] = 4
            room["bank_hotels"] = int(room["bank_hotels"]) + 1
            room["bank_houses"] = int(room["bank_houses"]) - 4
            received += int(s["house_cost"]) // 2
        elif h > 0:
            prop["houses"] = h - 1
            room["bank_houses"] = int(room["bank_houses"]) + 1
            received += int(s["house_cost"]) // 2
        else:
            break
        p["cash"] = int(p["cash"]) + int(s["house_cost"]) // 2
    return received


def _can_mortgage(room: dict[str, Any], seat: int, index: int) -> tuple[bool, str]:
    s = SPACES[index]
    prop = _prop(room, index)
    if prop["owner"] != seat:
        return False, "you do not own it"
    if prop["mortgaged"]:
        return False, "it is already mortgaged"
    if s.get("kind") == "street" and _group_has_buildings(room, s["group"]):
        return False, "sell the buildings on that colour group first"
    return True, ""


def _mortgage(room: dict[str, Any], seat: int, index: int) -> None:
    s = SPACES[index]
    prop = _prop(room, index)
    prop["mortgaged"] = True
    _credit(room, seat, int(s["mortgage"]))


def _unmortgage(room: dict[str, Any], seat: int, index: int) -> None:
    s = SPACES[index]
    prop = _prop(room, index)
    cost = int(s["mortgage"]) + int(int(s["mortgage"]) * 0.1 + 0.5)
    p = room["players"][seat]
    if int(p["cash"]) < cost:
        raise ValueError(f"lifting the mortgage costs ${cost}")
    p["cash"] = int(p["cash"]) - cost
    prop["mortgaged"] = False


# --------------------------------------------------------------------------- #
# The auction - any player may bid on a declined property
# --------------------------------------------------------------------------- #

def _auction_open(room: dict[str, Any], space: int) -> None:
    room["auction"] = {
        "space": int(space),
        "high_bid": 0,
        "high_bidder": None,
        "passed": [],
        "started_at": time.time(),
    }
    _log(room, f"The Bank auctions {SPACES[space]['name']}")


def _auction_remaining(room: dict[str, Any]) -> list[int]:
    a = room["auction"]
    passed = set(a["passed"])
    return [s for s in _active_seats(room) if s not in passed]


def _auction_check_done(room: dict[str, Any]) -> bool:
    """Close the auction when nobody can outbid. Returns True if it closed."""
    a = room["auction"]
    if not a:
        return True
    remaining = _auction_remaining(room)
    space = int(a["space"])
    high = a["high_bidder"]
    # Done when the high bidder is the only one left, or nobody wanted it.
    done = (high is not None and remaining == [high]) or (high is None and not remaining)
    if not done:
        return False
    if high is not None:
        # Award to the high bidder, even if they are the last one standing
        prop = _prop(room, space)
        prop["owner"] = int(high)
        prop["mortgaged"] = False
        prop["houses"] = 0
        p = room["players"][high]
        p["cash"] = int(p["cash"]) - int(a["high_bid"])
        _log(room, f"{p['name']} wins the auction - {SPACES[space]['name']} for ${a['high_bid']}")
    else:
        _log(room, f"Nobody bid on {SPACES[space]['name']} - it returns to the Bank")
    room["auction"] = None
    return True


def _auction_bid(room: dict[str, Any], seat: int, amount: int) -> None:
    a = room["auction"]
    if not a:
        raise ValueError("no auction is running")
    amount = int(amount)
    p = room["players"][seat]
    if int(p["cash"]) < amount:
        raise ValueError("not enough cash")
    if amount <= int(a["high_bid"]):
        raise ValueError(f"a bid must beat ${a['high_bid']}")
    if a["high_bidder"] == seat:
        raise ValueError("you are already the high bidder")
    a["high_bid"] = amount
    a["high_bidder"] = seat
    a["passed"] = [s for s in a["passed"] if s != seat]
    _log(room, f"{p['name']} bids ${amount}")


def _auction_pass(room: dict[str, Any], seat: int) -> None:
    a = room["auction"]
    if not a:
        raise ValueError("no auction is running")
    if a["high_bidder"] == seat:
        raise ValueError("the high bidder cannot pass")
    if seat not in a["passed"]:
        a["passed"].append(seat)
        _log(room, f"{room['players'][seat]['name']} passes")


# --------------------------------------------------------------------------- #
# Bankruptcy and debts
# --------------------------------------------------------------------------- #

def _sell_all_buildings_for_cash(room: dict[str, Any], seat: int) -> None:
    """Every building the seat owns is sold at half price, most built first,
    in even order, until none remain."""
    for _ in range(200):
        sellable = _sellable_building_counts(room, seat)
        if not sellable:
            return
        # Sell from the property with the most houses (classic even selling)
        target = max(sellable, key=lambda i: int(_prop(room, i)["houses"]))
        _sell_building(room, seat, target, 1)


def _eliminate_player(room: dict[str, Any], seat: int, creditor: int | None) -> None:
    """The bankrupt player's assets change hands; they leave the game."""
    p = room["players"][seat]
    _sell_all_buildings_for_cash(room, seat)
    cash = int(p["cash"])
    jail_cards = int(p["jail_cards"])
    if creditor is not None and not room["players"][creditor]["bankrupt"]:
        room["players"][creditor]["cash"] = int(room["players"][creditor]["cash"]) + cash
        room["players"][creditor]["jail_cards"] = int(room["players"][creditor]["jail_cards"]) + jail_cards
        for i in _STREET_INDICES + [s["i"] for s in _RAILROADS + _UTILITIES]:
            prop = _prop(room, i)
            if prop["owner"] == seat:
                prop["owner"] = creditor
                if prop["mortgaged"]:
                    # The new owner may lift immediately for mortgage + 10%;
                    # skip the fee when they cannot cover it (documented).
                    s = SPACES[i]
                    fee = int(int(s["mortgage"]) * 0.1 + 0.5)
                    if int(room["players"][creditor]["cash"]) >= fee:
                        room["players"][creditor]["cash"] -= fee
                        prop["mortgaged"] = False
    else:
        # Owed to the Bank: property returns unmortgaged, buildings to the Bank.
        for i, s in enumerate(SPACES):
            if not s:
                continue
            prop = _prop(room, i)
            if prop["owner"] == seat:
                prop["owner"] = None
                prop["mortgaged"] = False
                prop["houses"] = 0
    p["cash"] = 0
    p["jail_cards"] = 0
    p["jail"] = None
    p["bankrupt"] = True
    room["bankruptcies"] = int(room.get("bankruptcies", 0)) + 1
    _log(room, f"{p['name']} is bankrupt and leaves the game")


def _pay_debt(room: dict[str, Any], seat: int) -> bool:
    """Try to settle the pending debt with available cash. Returns paid."""
    pending = room["pending"]
    if not pending or pending.get("kind") != "debt":
        return True
    amount = int(pending["amount"])
    p = room["players"][seat]
    if int(p["cash"]) < amount:
        return False
    p["cash"] -= amount
    creditor = pending.get("creditor")
    if creditor is not None:
        _credit(room, int(creditor), amount)
    elif (room["rules"]["jackpot"] and pending.get("reason") in ("tax", "fine", "tribute")):
        # The pot is fed now (not when the debt is raised) so money raised by
        # mortgaging and selling buildings counts toward the Sabbath pot.
        room["pot"] = int(room["pot"]) + amount
    room["pending"] = None
    _refresh_bonus(room, seat)
    _maybe_finish(room)
    return True


def _declare_bankruptcy(room: dict[str, Any], seat: int, creditor: int | None) -> None:
    _eliminate_player(room, seat, creditor)
    room["pending"] = None
    if not _maybe_finish(room):
        # The game goes on: hand the dice to the next living player.
        _advance_turn(room, seat)


# --------------------------------------------------------------------------- #
# Trading
# --------------------------------------------------------------------------- #

_TRADEABLE_KINDS = ("street", "railroad", "utility")


def _clean_trade_side(raw: Any, seat: int, room: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"cash": 0, "spaces": [], "jail_cards": 0}
    try:
        cash = max(0, int(raw.get("cash") or 0))
    except (TypeError, ValueError):
        cash = 0
    spaces: list[int] = []
    for i in raw.get("spaces") or []:
        try:
            i = int(i)
        except (TypeError, ValueError):
            continue
        if 0 <= i < 40 and i not in spaces:
            spaces.append(i)
    try:
        jail_cards = max(0, int(raw.get("jail_cards") or 0))
    except (TypeError, ValueError):
        jail_cards = 0
    p = room["players"][seat]
    if cash > int(p["cash"]):
        raise ValueError("not enough cash in that offer")
    if jail_cards > int(p["jail_cards"]):
        raise ValueError("not enough Dungeon-free cards in that offer")
    for i in spaces:
        s = SPACES[i]
        if not s or s.get("kind") not in _TRADEABLE_KINDS:
            raise ValueError(f"{s['name'] if s else 'That space'} cannot be traded")
        prop = _prop(room, i)
        if prop["owner"] != seat:
            raise ValueError(f"you do not own {SPACES[i]['name']}")
        if s.get("kind") == "street" and _group_has_buildings(room, s["group"]):
            raise ValueError(f"sell the buildings on {SPACES[i]['name']}'s colour group first")
    return {"cash": cash, "spaces": spaces, "jail_cards": jail_cards}


def _trade_value(room: dict[str, Any], side: dict[str, Any]) -> int:
    total = int(side["cash"])
    for i in side["spaces"]:
        s = SPACES[i]
        total += int(s["price"])
    total += int(side.get("jail_cards", 0)) * 50
    return total


def _execute_trade(room: dict[str, Any], trade: dict[str, Any]) -> None:
    a, b = int(trade["from"]), int(trade["to"])
    pa, pb = room["players"][a], room["players"][b]
    # Cash flows both ways
    pa["cash"] = int(pa["cash"]) - int(trade["offer"]["cash"]) + int(trade["ask"]["cash"])
    pb["cash"] = int(pb["cash"]) - int(trade["ask"]["cash"]) + int(trade["offer"]["cash"])
    for i in trade["offer"]["spaces"]:
        _prop(room, i)["owner"] = b
    for i in trade["ask"]["spaces"]:
        _prop(room, i)["owner"] = a
    pa["jail_cards"] = int(pa["jail_cards"]) - int(trade["offer"]["jail_cards"]) + int(trade["ask"]["jail_cards"])
    pb["jail_cards"] = int(pb["jail_cards"]) - int(trade["ask"]["jail_cards"]) + int(trade["offer"]["jail_cards"])
    _log(room, f"{pa['name']} and {pb['name']} strike a deal")


def _trade_side_label(room: dict[str, Any], side: dict[str, Any]) -> str:
    bits = []
    if side["cash"]:
        bits.append(f"${side['cash']}")
    for i in side["spaces"]:
        bits.append(SPACES[i]["name"])
    if side.get("jail_cards"):
        bits.append(f"{side['jail_cards']} Dungeon-free card" + ("s" if side["jail_cards"] > 1 else ""))
    return " + ".join(bits) if bits else "nothing"


def _new_trade_id() -> str:
    return secrets.token_hex(6)


# --------------------------------------------------------------------------- #
# Rooms - creation, lobby, lifecycle
# --------------------------------------------------------------------------- #

def _fresh_props() -> list[dict[str, Any]]:
    return [{"owner": None, "houses": 0, "mortgaged": False} for _ in range(40)]


def _shuffled_deck(full: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = list(range(len(full)))
    random.shuffle(order)
    return [dict(full[i]) for i in order]


def _build_rules(payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = payload or {}
    ruleset = str(payload.get("ruleset") or "classic").lower()
    if ruleset not in ("classic", "familyfun", "quick"):
        ruleset = "classic"

    def flag(name: str, default: bool = False) -> bool:
        v = payload.get(name)
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("1", "true", "yes", "on")
        return bool(v) if v is not None else default

    rules = {
        "ruleset": ruleset,
        "jackpot": flag("jackpot"),
        "double_go": flag("double_go"),
        "no_auction": flag("no_auction"),
    }
    if ruleset == "familyfun":
        rules["jackpot"] = True
        rules["no_auction"] = True
    return rules


def _new_room_state(code: str, host: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    now = time.time()
    return {
        "code": code,
        "status": "lobby",
        "phase": "lobby",
        "rules": rules,
        "players": [host],
        "turn": 0,
        "rolled": False,
        "bonus_roll": False,
        "doubles": 0,
        "last_roll": None,
        "props": _fresh_props(),
        "decks": {"parable": _shuffled_deck(PARABLE_DECK), "scripture": _shuffled_deck(SCRIPTURE_DECK)},
        "bank_houses": BANK_HOUSES,
        "bank_hotels": BANK_HOTELS,
        "pot": 0,
        "turn_counts": [0],
        "log": [f"{host['name']} is preparing the ledgers"],
        "pending": None,
        "auction": None,
        "trades": [],
        "bankruptcies": 0,
        "winner": None,
        "result": None,
        "match_no": 1,
        "created_at": now,
        "updated_at": now,
    }


def _host_seat(room: dict[str, Any]) -> int | None:
    for i, p in enumerate(room["players"]):
        if p and not p.get("is_ai") and not p.get("bankrupt"):
            return i
    return None


def _free_token(room: dict[str, Any], wanted: str | None) -> str | None:
    taken = {p.get("token") for p in room["players"] if p}
    if wanted and wanted in TOKENS and wanted not in taken:
        return wanted
    for t in TOKENS:
        if t not in taken:
            return t
    return None


def _reset_match(room: dict[str, Any]) -> None:
    """Fresh ledgers with the same adventurers - the rematch button."""
    for p in room["players"]:
        p["cash"] = START_CASH
        p["position"] = 0
        p["jail"] = None
        p["jail_cards"] = 0
        p["bankrupt"] = False
        p["last_seen"] = time.time()
    room["status"] = "lobby"
    room["phase"] = "lobby"
    room["turn"] = 0
    room["rolled"] = False
    room["bonus_roll"] = False
    room["doubles"] = 0
    room["last_roll"] = None
    room["props"] = _fresh_props()
    room["decks"] = {"parable": _shuffled_deck(PARABLE_DECK), "scripture": _shuffled_deck(SCRIPTURE_DECK)}
    room["bank_houses"] = BANK_HOUSES
    room["bank_hotels"] = BANK_HOTELS
    room["pot"] = 0
    room["turn_counts"] = [0 for _ in room["players"]]
    room["log"] = ["The ledgers are fresh - a new game begins"]
    room["pending"] = None
    room["auction"] = None
    room["trades"] = []
    room["bankruptcies"] = 0
    room["winner"] = None
    room["result"] = None
    room["match_no"] = int(room.get("match_no", 1)) + 1
    room["updated_at"] = time.time()


def _cancel_trades_for(room: dict[str, Any], seat: int) -> None:
    for t in room["trades"]:
        if t["status"] == "pending" and seat in (t["from"], t["to"]):
            t["status"] = "cancelled"
            _log(room, "A trade on the table is struck off")
    room["trades"] = [t for t in room["trades"] if t["status"] == "pending"]


def create_room(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    with _lock:
        _prune_unlocked(time.time())
        code = _pick_code_unlocked()
        rules = _build_rules(payload)
        token = _clean_token(payload.get("token")) if payload.get("token") else None
        host = _new_player(payload.get("name"), payload.get("initials"), token or TOKENS[0],
                           payload.get("device_id"))
        if token:
            host["token"] = token
        room = _new_room_state(code, host, rules)
        _rooms[code] = room
        return {"player_id": host["id"], "room": _snapshot_unlocked(room, host["id"])}


def join_room(code: Any, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    clean = _clean_code(code)
    with _lock:
        _prune_unlocked(time.time())
        room = _rooms.get(clean)
        if room is None:
            raise KeyError(f"no such room: {clean}")
        if room["status"] != "lobby":
            raise ValueError("that game has already begun")
        token = _clean_token(payload.get("token")) if payload.get("token") else None
        free = _free_token(room, token)
        if free is None:
            raise ValueError("the room is full")
        player = _new_player(payload.get("name"), payload.get("initials"), free,
                             payload.get("device_id"))
        room["players"].append(player)
        room["turn_counts"].append(0)
        room["updated_at"] = time.time()
        _log(room, f"{player['name']} joins the ledger ({len(room['players'])} at the table)")
        return {"player_id": player["id"], "room": _snapshot_unlocked(room, player["id"])}


def list_open_rooms() -> list[dict[str, Any]]:
    with _lock:
        _prune_unlocked(time.time())
        now = time.time()
        out = []
        for room in _rooms.values():
            if room["status"] != "lobby":
                continue
            humans = [p for p in room["players"] if p and not p.get("is_ai")]
            if not humans or now - max(float(p.get("last_seen") or 0) for p in humans) > PRESENCE_SECONDS:
                continue
            out.append({"code": room["code"], "players": len(room["players"]),
                        "ruleset": room["rules"]["ruleset"], "created_at": room["created_at"]})
        return out


def start_match(code: Any, player_id: Any) -> dict[str, Any]:
    """Seat the scribes, and deal the first hand - only the host may call it."""
    clean = _clean_code(code)
    with _lock:
        _prune_unlocked(time.time())
        room = _rooms.get(clean)
        if room is None:
            raise KeyError(clean)
        seat = _find_seat(room, player_id)
        if seat is None:
            raise PermissionError("you are not in this room")
        if seat != _host_seat(room):
            raise PermissionError("only the host may start the game")
        if room["status"] != "lobby":
            raise ValueError("the match has already begun")
        # Fill with scribes until at least three sit at the table
        for scribe in SCRIBES:
            if len(room["players"]) >= 3:
                break
            token = _free_token(room, None) or TOKENS[len(room["players"]) % len(TOKENS)]
            room["players"].append(_ai_player(scribe["name"], scribe["initials"], scribe["avatar"], token))
            room["turn_counts"].append(0)
        if len(room["players"]) < 2:
            raise ValueError("at least two players are needed")
        room["status"] = "playing"
        room["phase"] = "turn"
        room["turn"] = 0
        room["turn_counts"] = [0 for _ in room["players"]]
        _log(room, "The game begins - " + ", ".join(p["name"] for p in room["players"]))
        room["updated_at"] = time.time()
        return {"room": _snapshot_unlocked(room, player_id)}


def update_settings(code: Any, player_id: Any, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    clean = _clean_code(code)
    with _lock:
        room = _rooms.get(clean)
        if room is None:
            raise KeyError(clean)
        seat = _find_seat(room, player_id)
        if seat is None:
            raise PermissionError("you are not in this room")
        if seat != _host_seat(room):
            raise PermissionError("only the host may set the rules")
        if room["status"] != "lobby":
            raise ValueError("the rules are set once the game begins")
        room["rules"] = _build_rules(payload)
        room["updated_at"] = time.time()
        _log(room, "The host amends the ledger's rules")
        return {"room": _snapshot_unlocked(room, player_id)}


def leave_room(code: Any, player_id: Any) -> dict[str, Any]:
    clean = _clean_code(code)
    with _lock:
        _prune_unlocked(time.time())
        room = _rooms.get(clean)
        if room is None:
            raise KeyError(clean)
        seat = _find_seat(room, player_id)
        if seat is None:
            return {"success": True}
        if room["status"] == "lobby":
            room["players"].pop(seat)
            room["turn_counts"].pop(seat)
            if not room["players"]:
                del _rooms[clean]
                return {"success": True, "gone": True}
        else:
            # Mid-match: a resignation. Their assets fall to the Bank.
            if not room["players"][seat]["bankrupt"]:
                _eliminate_player(room, seat, None)
            _cancel_trades_for(room, seat)
            if room["turn"] == seat and room["status"] == "playing":
                room["turn"] = _next_seat(room, seat)
                room["rolled"] = False
                room["bonus_roll"] = False
                room["doubles"] = 0
            _maybe_finish(room)
        room["updated_at"] = time.time()
        return {"room": _snapshot_unlocked(room, player_id)}


def _do_roll(room: dict[str, Any], seat: int) -> tuple[int, int]:
    d1, d2 = secrets.randbelow(6) + 1, secrets.randbelow(6) + 1
    is_doubles = d1 == d2
    room["last_roll"] = {"d1": d1, "d2": d2, "doubles": is_doubles}
    room["rolled"] = True
    room["doubles"] = int(room["doubles"]) + 1 if is_doubles else 0
    p = room["players"][seat]
    if is_doubles and room["doubles"] >= 3:
        _log(room, f"{p['name']} throws three doubles in a row - to the Dungeon!")
        _send_to_jail(room, seat)
        room["bonus_roll"] = False
        return d1, d2
    _advance(room, seat, int(p["position"]) + d1 + d2, collect_go=True)
    _resolve_landing(room, seat, [d1, d2])
    _refresh_bonus(room, seat)
    return d1, d2


def roll(code: Any, player_id: Any) -> dict[str, Any]:
    clean = _clean_code(code)
    with _lock:
        room = _rooms.get(clean)
        if room is None:
            raise KeyError(clean)
        seat = _find_seat(room, player_id)
        if seat is None:
            raise PermissionError("you are not in this room")
        if room["status"] != "playing":
            raise ValueError("the match is not in play")
        if room["players"][seat]["jail"] is not None:
            raise ValueError("you are in the Dungeon - roll for doubles, pay $50, or play a Dungeon-free card")
        if room["turn"] != seat:
            raise ValueError("it is not your turn")
        if room["pending"] or room["auction"]:
            raise ValueError("settle what is in front of you first")
        if room["rolled"] and not room["bonus_roll"]:
            raise ValueError("you already rolled - end your turn")
        _touch(room, seat)
        _do_roll(room, seat)
        _maybe_ai_unlocked(room)
        _maybe_finish(room)
        return {"room": _snapshot_unlocked(room, player_id)}


def jail_roll(code: Any, player_id: Any) -> dict[str, Any]:
    clean = _clean_code(code)
    with _lock:
        room = _rooms.get(clean)
        if room is None:
            raise KeyError(clean)
        seat = _find_seat(room, player_id)
        if seat is None:
            raise PermissionError("you are not in this room")
        p = room["players"][seat]
        if p["jail"] is None:
            raise ValueError("you are not in the Dungeon")
        if room["status"] != "playing":
            raise ValueError("the match is not in play")
        if room["turn"] != seat:
            raise ValueError("it is not your turn")
        if room["pending"] or room["auction"]:
            raise ValueError("settle what is in front of you first")
        if room["rolled"]:
            raise ValueError("you already made your move")
        _touch(room, seat)
        d1, d2 = secrets.randbelow(6) + 1, secrets.randbelow(6) + 1
        is_doubles = d1 == d2
        room["last_roll"] = {"d1": d1, "d2": d2, "doubles": is_doubles}
        room["rolled"] = True
        room["doubles"] = 0
        if is_doubles:
            p["jail"] = None
            _log(room, f"{p['name']} throws doubles - out of the Dungeon!")
            _advance(room, seat, int(p["position"]) + d1 + d2, collect_go=True)
            _resolve_landing(room, seat, [d1, d2])
            room["bonus_roll"] = False
        else:
            p["jail"]["turns"] = int(p["jail"]["turns"]) + 1
            _log(room, f"{p['name']} throws {d1}-{d2} - no doubles, one turn passes in the Dungeon")
            if p["jail"]["turns"] >= 3:
                # Third failed attempt: $50 is mandatory; move with the dice.
                _charge(room, seat, 50, None, "fine")
                p["jail"] = None
                _log(room, f"{p['name']} pays $50 and leaves the Dungeon")
                _advance(room, seat, int(p["position"]) + d1 + d2, collect_go=True)
                _resolve_landing(room, seat, [d1, d2])
        _maybe_ai_unlocked(room)
        _maybe_finish(room)
        return {"room": _snapshot_unlocked(room, player_id)}


def jail_fine(code: Any, player_id: Any) -> dict[str, Any]:
    clean = _clean_code(code)
    with _lock:
        room = _rooms.get(clean)
        if room is None:
            raise KeyError(clean)
        seat = _find_seat(room, player_id)
        if seat is None:
            raise PermissionError("you are not in this room")
        p = room["players"][seat]
        if p["jail"] is None:
            raise ValueError("you are not in the Dungeon")
        if room["status"] != "playing":
            raise ValueError("the match is not in play")
        if room["turn"] != seat:
            raise ValueError("it is not your turn")
        _touch(room, seat)
        paid = _charge(room, seat, 50, None, "fine")
        if paid:
            p["jail"] = None
            room["rolled"] = False
            room["bonus_roll"] = False
            room["doubles"] = 0
            _log(room, f"{p['name']} pays $50 to leave the Dungeon")
        _maybe_ai_unlocked(room)
        return {"room": _snapshot_unlocked(room, player_id)}


def jail_card(code: Any, player_id: Any) -> dict[str, Any]:
    clean = _clean_code(code)
    with _lock:
        room = _rooms.get(clean)
        if room is None:
            raise KeyError(clean)
        seat = _find_seat(room, player_id)
        if seat is None:
            raise PermissionError("you are not in this room")
        p = room["players"][seat]
        if p["jail"] is None:
            raise ValueError("you are not in the Dungeon")
        if room["status"] != "playing":
            raise ValueError("the match is not in play")
        if room["turn"] != seat:
            raise ValueError("it is not your turn")
        if int(p["jail_cards"]) < 1:
            raise ValueError("no Dungeon-free card in hand")
        _touch(room, seat)
        p["jail_cards"] = int(p["jail_cards"]) - 1
        p["jail"] = None
        room["rolled"] = False
        room["bonus_roll"] = False
        room["doubles"] = 0
        _log(room, f"{p['name']} plays a Dungeon-free card")
        _maybe_ai_unlocked(room)
        return {"room": _snapshot_unlocked(room, player_id)}


def end_turn(code: Any, player_id: Any) -> dict[str, Any]:
    clean = _clean_code(code)
    with _lock:
        room = _rooms.get(clean)
        if room is None:
            raise KeyError(clean)
        seat = _find_seat(room, player_id)
        if seat is None:
            raise PermissionError("you are not in this room")
        if room["status"] != "playing":
            raise ValueError("the match is not in play")
        if room["turn"] != seat:
            raise ValueError("it is not your turn")
        if room["pending"]:
            raise ValueError("settle what is in front of you first")
        if room["auction"]:
            raise ValueError("an auction is on - wait for it to close")
        if not room["rolled"]:
            raise ValueError("roll the dice first")
        if room["bonus_roll"]:
            raise ValueError("you rolled doubles - roll again before ending your turn")
        _touch(room, seat)
        _advance_turn(room, seat)
        _maybe_finish(room)
        _maybe_ai_unlocked(room)
        _maybe_finish(room)
        return {"room": _snapshot_unlocked(room, player_id)}


# --------------------------------------------------------------------------- #
# Buying, auctions, building - the actions a player can take
# --------------------------------------------------------------------------- #

def buy(code: Any, player_id: Any) -> dict[str, Any]:
    clean = _clean_code(code)
    with _lock:
        room = _rooms.get(clean)
        if room is None:
            raise KeyError(clean)
        seat = _find_seat(room, player_id)
        if seat is None:
            raise PermissionError("you are not in this room")
        pending = room["pending"]
        if not pending or pending.get("kind") != "buy":
            raise ValueError("there is nothing to buy right now")
        if seat != room["turn"]:
            raise PermissionError("only the player whose turn it is may buy")
        index = int(pending["space"])
        s = SPACES[index]
        price = int(s["price"])
        p = room["players"][seat]
        if int(p["cash"]) < price:
            raise ValueError(f"not enough cash to buy {s['name']}")
        p["cash"] = int(p["cash"]) - price
        _prop(room, index)["owner"] = seat
        room["pending"] = None
        _log(room, f"{p['name']} buys {s['name']} for ${price}")
        _refresh_bonus(room, seat)
        _maybe_ai_unlocked(room)
        return {"room": _snapshot_unlocked(room, player_id)}


def decline(code: Any, player_id: Any) -> dict[str, Any]:
    clean = _clean_code(code)
    with _lock:
        room = _rooms.get(clean)
        if room is None:
            raise KeyError(clean)
        seat = _find_seat(room, player_id)
        if seat is None:
            raise PermissionError("you are not in this room")
        pending = room["pending"]
        if not pending or pending.get("kind") != "buy":
            raise ValueError("there is nothing to decline right now")
        if seat != room["turn"]:
            raise PermissionError("only the player whose turn it is may decline")
        index = int(pending["space"])
        room["pending"] = None
        if room["rules"]["no_auction"]:
            _log(room, f"{SPACES[index]['name']} returns to the Bank - auctions are off")
        else:
            _auction_open(room, index)
        _maybe_ai_unlocked(room)
        _auction_check_done(room)
        _maybe_ai_unlocked(room)
        _auction_check_done(room)
        return {"room": _snapshot_unlocked(room, player_id)}


def auction_bid(code: Any, player_id: Any, amount: Any) -> dict[str, Any]:
    clean = _clean_code(code)
    with _lock:
        room = _rooms.get(clean)
        if room is None:
            raise KeyError(clean)
        if not room["auction"]:
            raise ValueError("no auction is running")
        seat = _find_seat(room, player_id)
        if seat is None:
            raise PermissionError("you are not in this room")
        if room["players"][seat]["bankrupt"]:
            raise PermissionError("a bankrupt player cannot bid")
        try:
            amt = int(amount)
        except (TypeError, ValueError):
            raise ValueError("a bid must be a number") from None
        _touch(room, seat)
        _auction_bid(room, seat, amt)
        _maybe_ai_unlocked(room)
        _auction_check_done(room)
        _maybe_ai_unlocked(room)
        _auction_check_done(room)
        return {"room": _snapshot_unlocked(room, player_id)}


def auction_pass(code: Any, player_id: Any) -> dict[str, Any]:
    clean = _clean_code(code)
    with _lock:
        room = _rooms.get(clean)
        if room is None:
            raise KeyError(clean)
        if not room["auction"]:
            raise ValueError("no auction is running")
        seat = _find_seat(room, player_id)
        if seat is None:
            raise PermissionError("you are not in this room")
        _touch(room, seat)
        _auction_pass(room, seat)
        _maybe_ai_unlocked(room)
        _auction_check_done(room)
        _maybe_ai_unlocked(room)
        _auction_check_done(room)
        return {"room": _snapshot_unlocked(room, player_id)}


def tribute_choice(code: Any, player_id: Any, choice: Any) -> dict[str, Any]:
    """The Tribute: pay the flat $200, or 10% of your net worth."""
    clean = _clean_code(code)
    with _lock:
        room = _rooms.get(clean)
        if room is None:
            raise KeyError(clean)
        seat = _find_seat(room, player_id)
        if seat is None:
            raise PermissionError("you are not in this room")
        pending = room["pending"]
        if not pending or pending.get("kind") != "tribute":
            raise ValueError("the Tribute is not on the table")
        if seat != room["turn"]:
            raise PermissionError("only the player whose turn it is may choose")
        choice = str(choice).lower()
        if choice not in ("flat", "percent"):
            raise ValueError("choose 'flat' or 'percent'")
        amount = int(pending["flat"]) if choice == "flat" else int(pending["percent"])
        room["pending"] = None
        _charge(room, seat, amount, None, "tribute")
        _log(room, f"{room['players'][seat]['name']} renders unto Caesar ${amount}")
        _refresh_bonus(room, seat)
        _maybe_ai_unlocked(room)
        _maybe_finish(room)
        return {"room": _snapshot_unlocked(room, player_id)}


def pay_debt(code: Any, player_id: Any) -> dict[str, Any]:
    """Settle a pending debt with cash in hand (raise more first if needed)."""
    clean = _clean_code(code)
    with _lock:
        room = _rooms.get(clean)
        if room is None:
            raise KeyError(clean)
        seat = _find_seat(room, player_id)
        if seat is None:
            raise PermissionError("you are not in this room")
        pending = room["pending"]
        if not pending or pending.get("kind") != "debt":
            raise ValueError("no debt is outstanding")
        if seat != room["turn"]:
            raise PermissionError("only the player with the debt may pay it")
        _touch(room, seat)
        paid = _pay_debt(room, seat)
        if not paid:
            raise ValueError("not enough cash yet - mortgage or sell buildings first")
        _maybe_ai_unlocked(room)
        _maybe_finish(room)
        return {"room": _snapshot_unlocked(room, player_id)}


def declare_bankruptcy(code: Any, player_id: Any) -> dict[str, Any]:
    clean = _clean_code(code)
    with _lock:
        room = _rooms.get(clean)
        if room is None:
            raise KeyError(clean)
        seat = _find_seat(room, player_id)
        if seat is None:
            raise PermissionError("you are not in this room")
        pending = room["pending"]
        creditor = None
        if pending and pending.get("kind") == "debt":
            if seat != room["turn"]:
                raise PermissionError("only the player with the debt may declare bankruptcy")
            creditor = pending.get("creditor")
        else:
            raise ValueError("bankruptcy is only declared against an outstanding debt")
        _touch(room, seat)
        _declare_bankruptcy(room, seat, int(creditor) if creditor is not None else None)
        _maybe_ai_unlocked(room)
        _maybe_finish(room)
        return {"room": _snapshot_unlocked(room, player_id)}


def build(code: Any, player_id: Any, property_index: Any, count: Any = 1) -> dict[str, Any]:
    clean = _clean_code(code)
    with _lock:
        room = _rooms.get(clean)
        if room is None:
            raise KeyError(clean)
        seat = _find_seat(room, player_id)
        if seat is None:
            raise PermissionError("you are not in this room")
        if room["status"] != "playing":
            raise ValueError("the match is not in play")
        if not _can_act_now(room, seat):
            raise ValueError("it is not your turn")
        try:
            index = int(property_index)
            n = int(count) if count is not None else 1
        except (TypeError, ValueError):
            raise ValueError("property and count must be numbers") from None
        if index not in _STREET_INDICES:
            raise ValueError("only streets can be built on")
        n = max(1, min(n, 5))
        built = []
        for _ in range(n):
            ok, why = _can_build_on(room, seat, index)
            if not ok:
                if built:
                    break
                raise ValueError(why)
            built.append(_build_building(room, seat, index))
        _log(room, f"{room['players'][seat]['name']} builds on {SPACES[index]['name']}")
        return {"room": _snapshot_unlocked(room, player_id)}


def sell_building(code: Any, player_id: Any, property_index: Any, count: Any = 1) -> dict[str, Any]:
    clean = _clean_code(code)
    with _lock:
        room = _rooms.get(clean)
        if room is None:
            raise KeyError(clean)
        seat = _find_seat(room, player_id)
        if seat is None:
            raise PermissionError("you are not in this room")
        if room["status"] != "playing":
            raise ValueError("the match is not in play")
        if not _can_act_now(room, seat):
            raise ValueError("it is not your turn")
        try:
            index = int(property_index)
            n = max(1, min(int(count) if count is not None else 1, 10))
        except (TypeError, ValueError):
            raise ValueError("property and count must be numbers") from None
        s = SPACES[index] if 0 <= index < 40 else None
        if not s or s.get("kind") != "street":
            raise ValueError("only streets hold buildings")
        if _prop(room, index)["owner"] != seat:
            raise PermissionError("you do not own that street")
        received = 0
        for _ in range(n):
            sellable = _sellable_building_counts(room, seat)
            if index not in sellable:
                if received:
                    break
                raise ValueError("buildings must come down evenly - sell from the most built street first")
            received += _sell_building(room, seat, index, 1)
        _log(room, f"{room['players'][seat]['name']} sells buildings on {SPACES[index]['name']} for ${received}")
        _refresh_bonus(room, seat)
        return {"room": _snapshot_unlocked(room, player_id)}


def _can_act_now(room: dict[str, Any], seat: int) -> bool:
    # Property management happens on your turn (and against your own debt).
    return seat == room["turn"]


def mortgage(code: Any, player_id: Any, property_index: Any) -> dict[str, Any]:
    clean = _clean_code(code)
    with _lock:
        room = _rooms.get(clean)
        if room is None:
            raise KeyError(clean)
        seat = _find_seat(room, player_id)
        if seat is None:
            raise PermissionError("you are not in this room")
        if room["status"] != "playing":
            raise ValueError("the match is not in play")
        if not _can_act_now(room, seat):
            raise ValueError("it is not your turn")
        try:
            index = int(property_index)
        except (TypeError, ValueError):
            raise ValueError("property must be a number") from None
        ok, why = _can_mortgage(room, seat, index)
        if not ok:
            raise ValueError(why)
        _mortgage(room, seat, index)
        _log(room, f"{room['players'][seat]['name']} mortgages {SPACES[index]['name']}")
        _refresh_bonus(room, seat)
        return {"room": _snapshot_unlocked(room, player_id)}


def unmortgage(code: Any, player_id: Any, property_index: Any) -> dict[str, Any]:
    clean = _clean_code(code)
    with _lock:
        room = _rooms.get(clean)
        if room is None:
            raise KeyError(clean)
        seat = _find_seat(room, player_id)
        if seat is None:
            raise PermissionError("you are not in this room")
        if room["status"] != "playing":
            raise ValueError("the match is not in play")
        if not _can_act_now(room, seat):
            raise ValueError("it is not your turn")
        try:
            index = int(property_index)
        except (TypeError, ValueError):
            raise ValueError("property must be a number") from None
        prop = _prop(room, index)
        if prop["owner"] != seat:
            raise PermissionError("you do not own it")
        if not prop["mortgaged"]:
            raise ValueError("it is not mortgaged")
        _unmortgage(room, seat, index)
        _log(room, f"{room['players'][seat]['name']} lifts the mortgage on {SPACES[index]['name']}")
        return {"room": _snapshot_unlocked(room, player_id)}


# --------------------------------------------------------------------------- #
# Trading - propose, accept, reject
# --------------------------------------------------------------------------- #

def trade_propose(code: Any, player_id: Any, to_seat: Any, offer: Any, ask: Any) -> dict[str, Any]:
    clean = _clean_code(code)
    with _lock:
        room = _rooms.get(clean)
        if room is None:
            raise KeyError(clean)
        seat = _find_seat(room, player_id)
        if seat is None:
            raise PermissionError("you are not in this room")
        if room["status"] != "playing":
            raise ValueError("the match is not in play")
        try:
            to = int(to_seat)
        except (TypeError, ValueError):
            raise ValueError("name a seat to trade with") from None
        if to == seat:
            raise ValueError("you can hardly trade with yourself")
        if not (0 <= to < len(room["players"])):
            raise ValueError("no such seat")
        if room["players"][to]["bankrupt"]:
            raise ValueError("they have left the game")
        _touch(room, seat)
        offer_clean = _clean_trade_side(offer, seat, room)
        ask_clean = _clean_trade_side(ask, to, room)
        if not (offer_clean["cash"] or offer_clean["spaces"] or offer_clean["jail_cards"]
                or ask_clean["cash"] or ask_clean["spaces"] or ask_clean["jail_cards"]):
            raise ValueError("a trade must carry something both ways")
        trade = {
            "id": _new_trade_id(),
            "from": seat,
            "to": to,
            "offer": offer_clean,
            "ask": ask_clean,
            "status": "pending",
            "created_at": time.time(),
        }
        room["trades"].append(trade)
        room["trades"] = [t for t in room["trades"] if t["status"] == "pending"][-12:]
        _log(room, f"{room['players'][seat]['name']} offers {room['players'][to]['name']}: "
                   f"{_trade_side_label(room, offer_clean)} for {_trade_side_label(room, ask_clean)}")
        # Scribes answer at once; people think.
        if room["players"][to]["is_ai"]:
            _ai_respond_to_trade(room, trade)
        return {"room": _snapshot_unlocked(room, player_id)}


def _ai_respond_to_trade(room: dict[str, Any], trade: dict[str, Any]) -> None:
    if trade["status"] != "pending":
        return
    gain = _trade_value(room, trade["offer"])
    cost = _trade_value(room, trade["ask"])
    if gain >= cost:
        try:
            _clean_trade_side(trade["offer"], trade["from"], room)
            _clean_trade_side(trade["ask"], trade["to"], room)
        except ValueError:
            trade["status"] = "rejected"
            _log(room, f"{room['players'][trade['to']]['name']} declines the deal")
            return
        trade["status"] = "accepted"
        _execute_trade(room, trade)
        _log(room, f"{room['players'][trade['to']]['name']} accepts the deal")
    else:
        trade["status"] = "rejected"
        _log(room, f"{room['players'][trade['to']]['name']} declines the deal")
    room["trades"] = [t for t in room["trades"] if t["status"] == "pending"]


def trade_accept(code: Any, player_id: Any, trade_id: Any) -> dict[str, Any]:
    clean = _clean_code(code)
    trade_id = str(trade_id or "")
    with _lock:
        room = _rooms.get(clean)
        if room is None:
            raise KeyError(clean)
        seat = _find_seat(room, player_id)
        if seat is None:
            raise PermissionError("you are not in this room")
        trade = next((t for t in room["trades"] if t["id"] == trade_id), None)
        if trade is None:
            raise ValueError("that offer is gone")
        if trade["to"] != seat:
            raise PermissionError("that offer is not for you")
        if room["status"] != "playing":
            raise ValueError("the match is not in play")
        _touch(room, seat)
        # Re-validate both sides before the handshake (the offer may be stale)
        try:
            offer_clean = _clean_trade_side(trade["offer"], trade["from"], room)
            ask_clean = _clean_trade_side(trade["ask"], trade["to"], room)
        except ValueError as exc:
            trade["status"] = "cancelled"
            room["trades"] = [t for t in room["trades"] if t["status"] == "pending"]
            raise ValueError(f"the offer no longer stands - {exc}") from None
        trade["offer"], trade["ask"] = offer_clean, ask_clean
        trade["status"] = "accepted"
        _execute_trade(room, trade)
        room["trades"] = [t for t in room["trades"] if t["status"] == "pending"]
        _maybe_ai_unlocked(room)
        return {"room": _snapshot_unlocked(room, player_id)}


def trade_reject(code: Any, player_id: Any, trade_id: Any) -> dict[str, Any]:
    clean = _clean_code(code)
    trade_id = str(trade_id or "")
    with _lock:
        room = _rooms.get(clean)
        if room is None:
            raise KeyError(clean)
        seat = _find_seat(room, player_id)
        if seat is None:
            raise PermissionError("you are not in this room")
        trade = next((t for t in room["trades"] if t["id"] == trade_id), None)
        if trade is None:
            raise ValueError("that offer is gone")
        if trade["to"] != seat:
            raise PermissionError("that offer is not for you")
        _touch(room, seat)
        trade["status"] = "rejected"
        _log(room, f"{room['players'][seat]['name']} declines the deal")
        room["trades"] = [t for t in room["trades"] if t["status"] == "pending"]
        return {"room": _snapshot_unlocked(room, player_id)}


def trade_cancel(code: Any, player_id: Any, trade_id: Any) -> dict[str, Any]:
    clean = _clean_code(code)
    trade_id = str(trade_id or "")
    with _lock:
        room = _rooms.get(clean)
        if room is None:
            raise KeyError(clean)
        seat = _find_seat(room, player_id)
        if seat is None:
            raise PermissionError("you are not in this room")
        trade = next((t for t in room["trades"] if t["id"] == trade_id), None)
        if trade is None:
            raise ValueError("that offer is gone")
        if trade["from"] != seat:
            raise PermissionError("only the one who offered may take it back")
        _touch(room, seat)
        trade["status"] = "cancelled"
        _log(room, f"{room['players'][seat]['name']} takes back the offer")
        room["trades"] = [t for t in room["trades"] if t["status"] == "pending"]
        return {"room": _snapshot_unlocked(room, player_id)}


# --------------------------------------------------------------------------- #
# The scribes - simple heuristic AI opponents
# --------------------------------------------------------------------------- #

def _ai_max_bid(room: dict[str, Any], seat: int, space: int) -> int:
    s = SPACES[space]
    price = int(s.get("price", 0))
    base = float(price)
    if s["kind"] == "street":
        group = s["group"]
        owned = sum(1 for i in _GROUPS[group] if _prop(room, i)["owner"] == seat)
        if owned >= len(_GROUPS[group]) - 1:
            base *= 1.6  # completes the set - build on it
        elif owned:
            base *= 1.25  # stitching the set together
    elif s["kind"] == "railroad":
        base *= 1.1 + 0.6 * _count_owned(room, seat, "railroad")
    elif s["kind"] == "utility":
        base *= 0.6
    affordable = int(room["players"][seat]["cash"]) - 100
    return max(0, min(int(base), affordable))


def _ai_auction(room: dict[str, Any], seat: int) -> bool:
    a = room["auction"]
    if not a:
        return False
    if seat in a["passed"] or a["high_bidder"] == seat:
        return False
    cap = _ai_max_bid(room, seat, int(a["space"]))
    if cap <= int(a["high_bid"]):
        _auction_pass(room, seat)
        return True
    bid = min(cap, int(a["high_bid"]) + max(10, (cap - int(a["high_bid"])) // 3 + 10))
    try:
        _auction_bid(room, seat, bid)
    except ValueError:
        _auction_pass(room, seat)
    return True


def _ai_decide_buy(room: dict[str, Any], seat: int, space: int) -> None:
    s = SPACES[space]
    price = int(s["price"])
    p = room["players"][seat]
    cash_after = int(p["cash"]) - price
    wants = cash_after >= 0
    if wants and s["kind"] == "street":
        group = s["group"]
        owned = sum(1 for i in _GROUPS[group] if _prop(room, i)["owner"] == seat)
        wants = owned >= len(_GROUPS[group]) - 1 or owned > 0 or cash_after >= 150 or price <= 120
    if wants:
        p["cash"] = cash_after
        _prop(room, space)["owner"] = seat
        _log(room, f"{p['name']} buys {s['name']} for ${price}")
    else:
        if room["rules"]["no_auction"]:
            _log(room, f"{p['name']} passes on {s['name']} - it returns to the Bank")
        else:
            _auction_open(room, space)
    room["pending"] = None
    _refresh_bonus(room, seat)


def _ai_decide_tribute(room: dict[str, Any], seat: int) -> None:
    pending = room["pending"]
    choice = "flat" if int(pending["flat"]) <= int(pending["percent"]) else "percent"
    amount = int(pending["flat"]) if choice == "flat" else int(pending["percent"])
    room["pending"] = None
    _charge(room, seat, amount, None, "tribute")
    _log(room, f"{room['players'][seat]['name']} renders unto Caesar ${amount}")
    _refresh_bonus(room, seat)


def _ai_raise_for_debt(room: dict[str, Any], seat: int) -> None:
    pending = room["pending"]
    amount = int(pending["amount"])
    p = room["players"][seat]
    # 1. Sell buildings (most built first) until the debt is covered
    for _ in range(60):
        if int(p["cash"]) >= amount:
            break
        sellable = _sellable_building_counts(room, seat)
        if not sellable:
            break
        target = max(sellable, key=lambda i: int(_prop(room, i)["houses"]))
        _sell_building(room, seat, target, 1)
    # 2. Mortgage the cheapest unmortgaged property without buildings
    while int(p["cash"]) < amount:
        options = []
        for i, s in enumerate(SPACES):
            if not s or s.get("kind") not in _TRADEABLE_KINDS:
                continue
            ok, _ = _can_mortgage(room, seat, i)
            if ok:
                options.append((int(s["mortgage"]), i))
        if not options:
            break
        options.sort(key=lambda x: x[0])
        _mortgage(room, seat, options[0][1])
        _log(room, f"{p['name']} mortgages {SPACES[options[0][1]]['name']} to raise cash")
    if int(p["cash"]) >= amount:
        _pay_debt(room, seat)
        return
    # 3. The scribes admit defeat
    creditor = pending.get("creditor")
    _declare_bankruptcy(room, seat, int(creditor) if creditor is not None else None)


def _ai_build_phase(room: dict[str, Any], seat: int) -> bool:
    p = room["players"][seat]
    built = False
    for _ in range(8):
        best = None
        for i in _STREET_INDICES:
            if _prop(room, i)["owner"] != seat:
                continue
            ok, _ = _can_build_on(room, seat, i)
            if not ok:
                continue
            houses = int(_prop(room, i)["houses"])
            cost = int(SPACES[i]["house_cost"])
            key = (houses, cost)
            if best is None or key < best[0]:
                best = (key, i)
        if best is None:
            break
        i = best[1]
        cost = int(SPACES[i]["house_cost"])
        if int(p["cash"]) - cost < 150:
            break
        _build_building(room, seat, i)
        built = True
    if built:
        _log(room, f"{p['name']} raises buildings on the Holy Land")
    return built


def _advance_turn(room: dict[str, Any], seat: int) -> None:
    nxt = _next_seat(room, seat)
    room["turn"] = nxt
    room["turn_counts"][seat] = int(room["turn_counts"][seat]) + 1
    room["rolled"] = False
    room["bonus_roll"] = False
    room["doubles"] = 0
    room["last_roll"] = None
    room["updated_at"] = time.time()


def _maybe_ai_unlocked(room: dict[str, Any]) -> bool:
    """Let the scribes take their turns synchronously (bounded loop)."""
    if room["status"] != "playing":
        return False
    moved = False
    for _ in range(60):
        progressed = False
        if room["auction"]:
            for seat in _active_seats(room):
                if room["players"][seat]["is_ai"] and _ai_auction(room, seat):
                    progressed = True
            if progressed:
                _auction_check_done(room)
                continue
            break
        cur = room["turn"]
        p = room["players"][cur]
        if p["is_ai"] and not p["bankrupt"]:
            pending = room["pending"]
            if pending:
                kind = pending["kind"]
                if kind == "buy":
                    _ai_decide_buy(room, cur, int(pending["space"]))
                elif kind == "tribute":
                    _ai_decide_tribute(room, cur)
                elif kind == "debt":
                    _ai_raise_for_debt(room, cur)
                else:
                    break
                progressed = True
            elif not room["rolled"] or room["bonus_roll"]:
                _do_roll(room, cur)
                progressed = True
            else:
                if not _ai_build_phase(room, cur):
                    _advance_turn(room, cur)
                progressed = True
        if not progressed:
            break
        moved = True
    return moved


# --------------------------------------------------------------------------- #
# The snapshot - what a client sees. Player ids never leave the server.
# --------------------------------------------------------------------------- #

def _space_snapshot(room: dict[str, Any], index: int) -> dict[str, Any]:
    s = SPACES[index]
    prop = _prop(room, index)
    out: dict[str, Any] = {
        "i": index,
        "kind": s["kind"],
        "name": s["name"],
    }
    for key in ("sub", "group", "price", "rents", "mortgage", "house_cost", "amount", "flat"):
        if key in s:
            out[key] = s[key]
    kind = s["kind"]
    if kind in ("street", "railroad", "utility"):
        out["owner"] = prop["owner"]
        out["houses"] = int(prop["houses"])
        out["mortgaged"] = bool(prop["mortgaged"])
        out["color"] = s.get("group") if kind == "street" else kind
    return out


def _snapshot_unlocked(room: dict[str, Any], player_id: Any = None, stage: bool = False) -> dict[str, Any]:
    now = time.time()
    seat = _find_seat(room, player_id)
    host = _host_seat(room)

    players = []
    for i, p in enumerate(room["players"]):
        is_you = seat is not None and i == seat
        players.append({
            "seat": i,
            "public_id": _public_id(room, p),
            "name": p["name"],
            "initials": p["initials"],
            "avatar": p.get("avatar"),
            "token": p["token"],
            "is_ai": bool(p.get("is_ai")),
            "bankrupt": bool(p["bankrupt"]),
            "cash": int(p["cash"]),
            "position": int(p["position"]),
            "in_jail": p["jail"] is not None,
            "jail_turns": int(p["jail"]["turns"]) if p["jail"] else 0,
            "jail_cards": int(p["jail_cards"]),
            "net_worth": _net_worth(room, i),
            "houses": _building_count(room, i)[0],
            "hotels": _building_count(room, i)[1],
            "present": bool(p.get("is_ai")) or (not p["bankrupt"]
                    and now - float(p.get("last_seen") or 0) < PRESENCE_SECONDS),
            "you": is_you,
            "is_host": i == host,
            "is_turn": i == room["turn"],
        })

    you = None
    if seat is not None:
        you = players[seat]

    alive = _active_seats(room)
    min_count = min((room["turn_counts"][s] for s in alive), default=0)
    round_no = 1 + int(min_count)

    trades = []
    for t in room["trades"]:
        if t["status"] != "pending":
            continue
        trades.append({
            "id": t["id"],
            "from": t["from"],
            "to": t["to"],
            "offer": t["offer"],
            "ask": t["ask"],
            "offer_label": _trade_side_label(room, t["offer"]),
            "ask_label": _trade_side_label(room, t["ask"]),
            "from_name": room["players"][t["from"]]["name"],
            "to_name": room["players"][t["to"]]["name"],
        })

    auction = None
    if room["auction"]:
        a = room["auction"]
        space = int(a["space"])
        auction = {
            "space": space,
            "name": SPACES[space]["name"],
            "high_bid": int(a["high_bid"]),
            "high_bidder": a["high_bidder"],
            "high_bidder_name": room["players"][a["high_bidder"]]["name"] if a["high_bidder"] is not None else None,
            "passed": list(a["passed"]),
            "you_passed": seat is not None and seat in a["passed"],
        }

    pending = None
    if room["pending"]:
        pending = dict(room["pending"])
        if pending["kind"] == "buy":
            pending["price"] = int(SPACES[int(pending["space"])]["price"])
            pending["name"] = SPACES[int(pending["space"])]["name"]

    return {
        "code": room["code"],
        "status": room["status"],
        "phase": room["phase"],
        "rules": dict(room["rules"]),
        "ruleset": room["rules"]["ruleset"],
        "pot": int(room["pot"]),
        "round": round_no,
        "match_no": room["match_no"],
        "turn": room["turn"],
        "rolled": room["rolled"],
        "bonus_roll": room["bonus_roll"],
        "doubles": room["doubles"],
        "last_roll": room["last_roll"],
        "board": [_space_snapshot(room, i) for i in range(40)],
        "players": players,
        "you": you,
        "pending": pending,
        "auction": auction,
        "trades": trades,
        "decks": {"parable": len(room["decks"]["parable"]), "scripture": len(room["decks"]["scripture"])},
        "bank_houses": int(room["bank_houses"]),
        "bank_hotels": int(room["bank_hotels"]),
        "log": list(room["log"]),
        "winner": room["winner"],
        "result": room["result"],
        "bankruptcies": int(room["bankruptcies"]),
        "server_time": now,
        "updated_at": room["updated_at"],
    }


def snapshot(room: dict[str, Any], player_id: Any = None, stage: bool = False) -> dict[str, Any]:
    with _lock:
        return _snapshot_unlocked(room, player_id, stage)


def get_room(code: Any, player_id: Any = None, stage: bool = False) -> dict[str, Any] | None:
    clean = _clean_code(code)
    with _lock:
        _prune_unlocked(time.time())
        room = _rooms.get(clean)
        if room is None:
            return None
        seat = _find_seat(room, player_id)
        if seat is not None:
            _touch(room, seat)
        if room["status"] == "playing":
            _maybe_ai_unlocked(room)
        return _snapshot_unlocked(room, player_id, stage)


def restart_room(code: Any, player_id: Any) -> dict[str, Any]:
    """A rematch: fresh ledgers, same adventurers, back to the lobby."""
    clean = _clean_code(code)
    with _lock:
        room = _rooms.get(clean)
        if room is None:
            raise KeyError(clean)
        seat = _find_seat(room, player_id)
        if seat is None:
            raise PermissionError("you are not in this room")
        if room["status"] not in ("finished", "lobby"):
            raise ValueError("finish the game before the rematch")
        _reset_match(room)
        room["updated_at"] = time.time()
        return {"room": _snapshot_unlocked(room, player_id)}


def _clear_rooms() -> None:
    """Test hook: forget every room."""
    with _lock:
        _rooms.clear()
