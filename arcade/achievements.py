#!/usr/bin/env python3
"""Alpaca Arcade — achievement engine.

Pure logic (no Flask): given a player's aggregated cross-game stats, decide
which achievements are unlocked. Identity is arcade-true: a player's 3-letter
initials ARE the account, tracked across every published machine.

Each achievement: id, icon, name, tagline + a check(stats) returning
(unlocked: bool, progress: float|None, goal: int|None). Count-based
achievements expose progress/goal so the UI can render progress bars.
"""

NIGHT_OWL_START = 0
NIGHT_OWL_END = 5
HIGH_ROLLER_SCORE = 100_000


def _count(unlocked: bool, have: int, goal: int) -> tuple:
    return (unlocked or have >= goal, (min(have, goal) / goal) if goal else None, goal)


def check_first_blood(s):
    return _count(s["submits"] >= 1, s["submits"], 1)


def check_on_board(s):
    return _count(s["boards"] >= 1, s["boards"], 1)


def check_podium(s):
    return _count(s["podiums"] >= 1, s["podiums"], 1)


def check_champion(s):
    return _count(s["crowns"] >= 1, s["crowns"], 1)


def check_dynasty(s):
    # "Hold #1 on 3 machines at once" — counted live from today's top-5
    # boards (current crowns), never a never-decreasing lifetime tally.
    return _count(s["crowns"] >= 3, s["crowns"], 3)


def check_globetrotter(s):
    return _count(s["games_scored"] >= 3, s["games_scored"], 3)


def check_completionist(s):
    total = s["total_games"]
    done = total > 0 and s["games_scored"] >= total
    return (done, (min(s["games_scored"], total) / total) if total else None, total)


def check_grinder(s):
    return _count(s["submits"] >= 10, s["submits"], 10)


def check_hot_streak(s):
    return _count(s["personal_bests"] >= 3, s["personal_bests"], 3)


def check_pioneer(s):
    return _count(s["pioneered"] >= 1, s["pioneered"], 1)


def check_night_owl(s):
    return (s["night_owl"], None, None)


def check_high_roller(s):
    return (s["high_roller"], None, None)


def check_critic(s):
    return _count(s["games_rated"] >= 3, s["games_rated"], 3)


def check_standing_ovation(s):
    return (s["gave_five_stars"], None, None)


ACHIEVEMENTS = [
    {
        "id": "first-blood",
        "icon": "🩸",
        "name": "First Blood",
        "tagline": "Submit your first score on any machine.",
        "check": check_first_blood,
    },
    {
        "id": "on-board",
        "icon": "📋",
        "name": "On the Board",
        "tagline": "Crack a top-5 scoreboard.",
        "check": check_on_board,
    },
    {
        "id": "podium",
        "icon": "🥉",
        "name": "Podium Finish",
        "tagline": "Finish top-3 on any machine.",
        "check": check_podium,
    },
    {
        "id": "champion",
        "icon": "👑",
        "name": "Champion",
        "tagline": "Take the #1 spot on any machine.",
        "check": check_champion,
    },
    {
        "id": "pioneer",
        "icon": "🚩",
        "name": "Pioneer",
        "tagline": "Be the first soul to score on a machine.",
        "check": check_pioneer,
    },
    {
        "id": "personal-record",
        "icon": "🚀",
        "name": "Personal Record",
        "tagline": "Beat your own best on a machine.",
        "check": lambda s: (s["personal_bests"] >= 1, None, None),
    },
    {
        "id": "hot-streak",
        "icon": "🔥",
        "name": "Hot Streak",
        "tagline": "Set 3 personal records.",
        "check": check_hot_streak,
    },
    {
        "id": "high-roller",
        "icon": "🎲",
        "name": "High Roller",
        "tagline": f"Post a single score over {HIGH_ROLLER_SCORE:,}.",
        "check": check_high_roller,
    },
    {
        "id": "globetrotter",
        "icon": "🌍",
        "name": "Globetrotter",
        "tagline": "Score on 3 different machines.",
        "check": check_globetrotter,
    },
    {
        "id": "dynasty",
        "icon": "🏆",
        "name": "Dynasty",
        "tagline": "Hold #1 on 3 different machines at once.",
        "check": check_dynasty,
    },
    {
        "id": "completionist",
        "icon": "💫",
        "name": "Completionist",
        "tagline": "Score on every machine in the arcade.",
        "check": check_completionist,
    },
    {
        "id": "grinder",
        "icon": "🎰",
        "name": "Grinder",
        "tagline": "Submit 10 scores. The machines know your name.",
        "check": check_grinder,
    },
    {
        "id": "night-owl",
        "icon": "🦉",
        "name": "Night Owl",
        "tagline": "Score between midnight and 5am. Sleep is for high scores.",
        "check": check_night_owl,
    },
    {
        "id": "critic",
        "icon": "🎬",
        "name": "Critic",
        "tagline": "Rate 3 different machines.",
        "check": check_critic,
    },
    {
        "id": "standing-ovation",
        "icon": "⭐",
        "name": "Standing Ovation",
        "tagline": "Award a machine the full 5 stars.",
        "check": check_standing_ovation,
    },
]


def evaluate(stats: dict) -> list:
    """Return achievement dicts with unlocked/progress/goal filled in."""
    out = []
    for a in ACHIEVEMENTS:
        try:
            unlocked, progress, goal = a["check"](stats)
        except Exception:
            unlocked, progress, goal = False, None, None
        out.append(
            {
                "id": a["id"],
                "icon": a["icon"],
                "name": a["name"],
                "tagline": a["tagline"],
                "unlocked": bool(unlocked),
                "progress": progress,
                "goal": goal,
            }
        )
    return out


def unlocked_ids(stats: dict) -> set:
    return {a["id"] for a in evaluate(stats) if a["unlocked"]}
