"""Family Games: hand-built, touch-first games for the Jarvis family hub.

Benchmark games are whatever a model wrote; many are desktop-shaped and only
fit a phone after the mobile shim. Family games are the other collection: made
for a phone first, built on one shared kit (``family_games/_kit``), and shipped
with the arcade itself so a deploy brings them up to date.

Source layout (in the repo, copied into the image)::

    arcade/family_games/<name>/game.html   page with a ``<!-- @family-kit -->`` marker
    arcade/family_games/<name>/meta.json   title, tagline, emoji, art, genre, version...

At startup each source is built (the kit is inlined, so the served page has no
network dependency) and synced into ``data/arcade/games/family-<name>/`` next
to the benchmark games. Only ``game.html`` and the descriptive meta fields are
replaced; scores, ratings, bests and the play count are never touched, so a new
version of a game keeps its family scoreboard.

Lifecycle, stored in ``data/arcade/family.json``::

    testing      playable by the family, can be voted on, not on the scoreboard yet
    published    on the Family Games shelf and the family scoreboard
    unpublished  hidden from players; scores and votes are kept for a re-publish

A game seen for the first time starts in ``testing``; an admin decides the rest.

Family Games are mobile games only. A game joins the collection, and can be
published, only when its page passes the mobile contract
(``mobile.contract_report``: touch controls, fits any screen, safe areas,
pauses when hidden, the score bridge...). A built-in game that fails it is
not installed; a benchmark game that fails it stays an ordinary arcade game.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

try:  # container runs arcade/ as the script dir; tests import arcade.* from the repo root
    from mobile import contract_failures
except ImportError:
    from arcade.mobile import contract_failures

ARCADE_DIR = Path(os.getenv("ARCADE_DIR", "data/arcade"))
GAMES_DIR = ARCADE_DIR / "games"
STATE_FILE = ARCADE_DIR / "family.json"
SOURCE_DIR = Path(os.getenv("FAMILY_GAMES_DIR", str(Path(__file__).resolve().parent / "family_games")))
KIT_DIR = SOURCE_DIR / "_kit"

SLUG_PREFIX = "family-"
STATES = ("testing", "published", "unpublished")
PLAYABLE_STATES = ("testing", "published")
DEFAULT_STATE = "testing"
KIT_MARKER = "<!-- @family-kit -->"
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}$")

# Descriptive fields owned by the source meta.json; everything else in the
# game's runtime meta (plays, published_at...) belongs to the arcade.
SOURCE_FIELDS = (
    "title",
    "tagline",
    "description",
    "emoji",
    "art",
    "genre",
    "ages",
    "players",
    "party",
    "tv_path",
    "version",
    "metric",
    "higher_is_better",
    "how_to",
)

_lock = threading.Lock()
_synced = False


def _read_json(path: Path, default):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default
    return data if isinstance(data, type(default)) else default


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def slug_for(name: str) -> str:
    return f"{SLUG_PREFIX}{name}"


def is_family_slug(slug: str) -> bool:
    """A built-in family game (shipped in family_games/)."""
    return bool(slug) and slug.startswith(SLUG_PREFIX)


def is_family(slug: str) -> bool:
    """Any game in the Family collection: built-in, or a mobile benchmark game
    whose publish tagged it ``collection: family``."""
    if not slug or "/" in slug or slug.startswith("."):
        return False
    d = GAMES_DIR / slug
    if not (d / "game.html").is_file():
        return False
    meta = _read_json(d / "meta.json", {})
    if meta.get("mobile_compatible") is not True:
        return False
    return is_family_slug(slug) or meta.get("collection") == "family"


def source(slug: str) -> str:
    """Where a family game came from: ``built-in`` or ``benchmark``."""
    return "built-in" if is_family_slug(slug) else "benchmark"


def source_names() -> list[str]:
    """Every family game in the source tree (directories with a game.html)."""
    if not SOURCE_DIR.is_dir():
        return []
    return sorted(
        child.name
        for child in SOURCE_DIR.iterdir()
        if child.is_dir() and NAME_RE.match(child.name) and (child / "game.html").is_file()
    )


def build_html(name: str) -> str:
    """The served page for one game: its source with the kit inlined."""
    src = (SOURCE_DIR / name / "game.html").read_text(encoding="utf-8")
    if KIT_MARKER not in src:
        raise ValueError(f"family game {name!r} has no {KIT_MARKER} marker")
    css = (KIT_DIR / "kit.css").read_text(encoding="utf-8")
    js = (KIT_DIR / "kit.js").read_text(encoding="utf-8")
    # A literal "</script>" inside the kit would end the inline block early.
    if "</script" in js.lower() or "</style" in css.lower():
        raise ValueError("family kit must not contain closing script/style tags")
    kit = f'<style id="family-kit-css">\n{css}\n</style>\n<script id="family-kit">\n{js}\n</script>'
    return src.replace(KIT_MARKER, kit, 1)


def source_meta(name: str) -> dict:
    meta = _read_json(SOURCE_DIR / name / "meta.json", {})
    return {k: meta[k] for k in SOURCE_FIELDS if k in meta}


def load_states() -> dict[str, dict]:
    raw = _read_json(STATE_FILE, {})
    games = raw.get("games", {}) if isinstance(raw.get("games"), dict) else {}
    clean: dict[str, dict] = {}
    for slug, entry in games.items():
        if not isinstance(entry, dict) or entry.get("state") not in STATES:
            continue
        clean[str(slug)] = entry
    return clean


def state_of(slug: str) -> str | None:
    """The lifecycle state of a family game, or None for any other slug."""
    if not is_family(slug):
        return None
    entry = load_states().get(slug)
    if entry:
        return str(entry["state"])
    return DEFAULT_STATE


def set_state(slug: str, state: str, by: str | None = None) -> dict:
    """Move a family game to a lifecycle state. Raises ValueError on bad input."""
    if state not in STATES:
        raise ValueError(f"state must be one of {', '.join(STATES)}")
    if not is_family(slug):
        raise LookupError("family game not found")
    if state == "published":
        failures = contract_failures((GAMES_DIR / slug / "game.html").read_text(encoding="utf-8", errors="replace"))
        if failures:
            raise ValueError("not mobile compatible, so it cannot be published: missing " + ", ".join(failures))
    with _lock:
        raw = _read_json(STATE_FILE, {})
        games = raw.get("games", {}) if isinstance(raw.get("games"), dict) else {}
        entry = dict(games.get(slug) or {})
        previous = entry.get("state", DEFAULT_STATE)
        entry["state"] = state
        entry["changed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        if by:
            entry["changed_by"] = str(by)[:64]
        if state == "published" and not entry.get("first_published_at"):
            entry["first_published_at"] = entry["changed_at"]
        games[slug] = entry
        _write_json(STATE_FILE, {"games": games})
    return {"slug": slug, "state": state, "previous": previous}


def all_slugs() -> set[str]:
    """Every game in the Family collection, whatever its state."""
    return slugs_in(STATES)


def slugs_in(states: tuple[str, ...]) -> set[str]:
    """Family game slugs currently in any of the given states."""
    if not GAMES_DIR.is_dir():
        return set()
    found = set()
    for child in GAMES_DIR.iterdir():
        if child.is_dir() and is_family(child.name) and state_of(child.name) in states:
            found.add(child.name)
    return found


def sync(force: bool = False) -> dict:
    """Install or update every family game from source into the games dir.

    Idempotent: a game whose built page and descriptive meta are unchanged is
    left alone. Player data files are never written here.
    """
    global _synced
    report = {"installed": [], "updated": [], "unchanged": [], "errors": {}}
    with _lock:
        raw = _read_json(STATE_FILE, {})
        games_state = raw.get("games", {}) if isinstance(raw.get("games"), dict) else {}
        states_changed = False
        for name in source_names():
            slug = slug_for(name)
            try:
                html = build_html(name)
            except (OSError, ValueError) as exc:
                report["errors"][slug] = str(exc)
                continue
            failures = contract_failures(html)
            if failures:
                report["errors"][slug] = "not mobile compatible: missing " + ", ".join(failures)
                continue
            digest = hashlib.sha256(html.encode("utf-8")).hexdigest()
            d = GAMES_DIR / slug
            meta_path = d / "meta.json"
            meta = _read_json(meta_path, {})
            fresh = not (d / "game.html").is_file()
            desc = source_meta(name)
            wanted = {
                **desc,
                "title": desc.get("title") or name.replace("-", " ").title(),
                "collection": "family",
                "category": "family",
                "kind": "playable",
                "type": desc.get("genre", ""),
                "lang": "html",
                "model": "Family Games",
                "score_bridge": True,
                "mobile_native": True,
                "mobile_compatible": True,
                "source_hash": digest,
            }
            if not force and not fresh and all(meta.get(k) == v for k, v in wanted.items()):
                report["unchanged"].append(slug)
            else:
                d.mkdir(parents=True, exist_ok=True)
                tmp = d / "game.html.tmp"
                tmp.write_text(html, encoding="utf-8")
                os.replace(tmp, d / "game.html")
                merged = {**meta, **wanted}
                merged.setdefault("plays", 0)
                merged.setdefault("published_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
                _write_json(meta_path, merged)
                report["installed" if fresh else "updated"].append(slug)
            if slug not in games_state:
                games_state[slug] = {"state": DEFAULT_STATE, "changed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                states_changed = True
        if states_changed:
            _write_json(STATE_FILE, {"games": games_state})
        _synced = True
    return report


def ensure_synced() -> None:
    if not _synced:
        sync()


if __name__ == "__main__":  # python arcade/family.py [build <name> <out.html>]
    import sys

    if len(sys.argv) == 4 and sys.argv[1] == "build":
        Path(sys.argv[3]).write_text(build_html(sys.argv[2]), encoding="utf-8")
        print(f"built {sys.argv[2]} -> {sys.argv[3]}")
    else:
        print(json.dumps(sync(force="--force" in sys.argv), indent=2))
