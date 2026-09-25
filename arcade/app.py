#!/usr/bin/env python3
"""Alpaca Arcade — public scoreboard web service for top-rated benchmark games.

Standalone Flask service (own port, default 5001). Each published game is a
fully self-contained directory under data/arcade/games/<slug>/:

    game.html    frozen playable HTML artifact (served same-origin so
                 the arcade page can read the game's localStorage for
                 score auto-capture); code-kind games (pygame/desktop apps)
                 freeze game.py + screenshot.png instead, served with a
                 code viewer and download
    meta.json    title, test id/label, model, kind (playable/code), benchmark
                 validation breakdown, publish date, play count
    scores.json  persistent server-side top-5 scoreboard
    ratings.json player 1-5 star votes

Because everything lives under data/arcade, a published game survives
deletion of its benchmark data and model in Alpaca. It stays until it is
explicitly removed (DELETE /api/admin/games/<slug> with ARCADE_ADMIN_TOKEN,
or the Unpublish button in the Alpaca dashboard).
"""

import fcntl
import hashlib
import json
import os
import re
import shlex
import threading
import time
import urllib.request
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

try:  # container runs `python arcade/app.py` (script dir on sys.path)
    from achievements import HIGH_ROLLER_SCORE, NIGHT_OWL_END, NIGHT_OWL_START, evaluate, unlocked_ids
except ImportError:  # tests import `arcade.app` as a package from the repo root
    from arcade.achievements import HIGH_ROLLER_SCORE, NIGHT_OWL_END, NIGHT_OWL_START, evaluate, unlocked_ids

ARCADE_DIR = Path(os.getenv("ARCADE_DIR", "data/arcade"))
GAMES_DIR = ARCADE_DIR / "games"
ADMIN_TOKEN = os.getenv("ARCADE_ADMIN_TOKEN", "")
PORT = int(os.getenv("ARCADE_PORT", "5001"))
MAX_SCORES = 5
# Alpaca web backend, which owns the UI sandbox (Xvfb + x11vnc + noVNC).
# Reachable as a compose service name in production; override for local dev.
WEB_BASE = os.getenv("ARCADE_WEB_URL", "http://alpaca-web:5000").rstrip("/")

SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")

app = Flask(__name__, template_folder="templates", static_folder="static")
_lock = threading.Lock()


def _game_dir(slug: str) -> Path | None:
    if not SLUG_RE.match(slug or ""):
        return None
    d = (GAMES_DIR / slug).resolve()
    try:
        d.relative_to(GAMES_DIR.resolve())
    except ValueError:
        return None
    return d


def _playable_file(d: Path) -> Path | None:
    """The frozen game payload: playable game.html or a code-kind game.py."""
    for name in ("game.html", "game.py"):
        p = d / name
        if p.exists():
            return p
    return None


def _read_json(path: Path, default):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, type(default)) else default
    except Exception:
        return default


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _rating_stats(game_dir: Path) -> dict:
    votes = _read_json(game_dir / "ratings.json", {}).get("votes", [])
    stars = [v.get("stars") for v in votes if isinstance(v, dict) and 1 <= int(v.get("stars", 0) or 0) <= 5]
    count = len(stars)
    return {"count": count, "average": round(sum(stars) / count, 2) if count else 0.0}


def _clean_initials(raw) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(raw or ""))[:3].upper()


def _hour_of(stamp: str) -> int | None:
    try:
        return int(str(stamp or "")[11:13])
    except (TypeError, ValueError, IndexError):
        return None


def _player_rank(scores: list, initials: str) -> int | None:
    """Best (lowest) 1-based rank of this player in a top-5 board."""
    best = None
    for i, s in enumerate(scores):
        if isinstance(s, dict) and str(s.get("initials", "")).upper() == initials:
            best = i + 1 if best is None else min(best, i + 1)
    return best


def _empty_stats(initials: str) -> dict:
    return {
        "initials": initials,
        "games_scored": 0,
        "submits": 0,
        "boards": 0,
        "podiums": 0,
        "crowns": 0,
        "crown_games": 0,
        "personal_bests": 0,
        "pioneered": 0,
        "night_owl": False,
        "high_roller": False,
        "best_score": 0,
        "games_rated": 0,
        "gave_five_stars": False,
        "total_games": 0,
    }


def _accumulate_game(stats: dict, scores: list, bests: dict, mine: dict, votes: list) -> None:
    """Fold one published game into a player's aggregate stats.

    Night Owl / High Roller are evaluated from the durable per-player ledger
    (max_ever, night_owl_at) so they survive board turnover; the current
    top-5 scan stays as a fallback for ledgers written before those fields
    existed. Board position (boards/podiums/crowns) is always read live from
    today's top-five — crowns are current, never a lifetime tally.
    """
    initials = stats["initials"]
    stats["total_games"] += 1
    my_count = int(mine.get("count", 0) or 0)
    if my_count:
        stats["games_scored"] += 1
        stats["submits"] += my_count
        stats["personal_bests"] += int(mine.get("pbs", 0) or 0)
        stats["best_score"] = max(stats["best_score"], int(mine.get("best", 0) or 0))
    if bests.get("first_by") == initials:
        stats["pioneered"] += 1
    if mine.get("night_owl_at"):
        stats["night_owl"] = True
    if int(mine.get("max_ever", 0) or 0) >= HIGH_ROLLER_SCORE:
        stats["high_roller"] = True
    for s in scores:
        if not isinstance(s, dict) or str(s.get("initials", "")).upper() != initials:
            continue
        try:
            hour = _hour_of(s.get("at", ""))
            if hour is not None and NIGHT_OWL_START <= hour < NIGHT_OWL_END:
                stats["night_owl"] = True
            if int(s.get("score", 0) or 0) >= HIGH_ROLLER_SCORE:
                stats["high_roller"] = True
        except (TypeError, ValueError):
            pass
    rank = _player_rank(scores, initials)
    if rank is not None:
        stats["boards"] += 1
        if rank <= 3:
            stats["podiums"] += 1
        if rank == 1:
            stats["crowns"] += 1
            stats["crown_games"] += 1
    my_votes = [
        v for v in votes if isinstance(v, dict) and str(v.get("by", "")).upper() == initials and v.get("stars")
    ]
    if my_votes:
        stats["games_rated"] += 1
        if any(int(v.get("stars", 0) or 0) == 5 for v in my_votes):
            stats["gave_five_stars"] = True


def player_stats(initials: str) -> dict:
    """Aggregate one player's cross-game stats + achievements.

    Reads every published game's scores/bests/ratings; purely derived, so it
    stays correct no matter when games were published or removed.
    """
    initials = (initials or "").upper()
    stats = _empty_stats(initials)
    stats["per_game"] = []
    if not GAMES_DIR.exists():
        stats["achievements"] = evaluate(stats)
        return stats
    for child in sorted(GAMES_DIR.iterdir()):
        if not child.is_dir() or _playable_file(child) is None:
            continue
        meta = _read_json(child / "meta.json", {})
        score_state = _read_json(child / "scores.json", {})
        scores = score_state.get("scores", [])
        bests = score_state.get("bests")
        if not isinstance(bests, dict):
            bests = _read_json(child / "bests.json", {})
        players = bests.get("players", {}) if isinstance(bests.get("players"), dict) else {}
        mine = players.get(initials, {}) if isinstance(players.get(initials), dict) else {}
        votes = _read_json(child / "ratings.json", {}).get("votes", [])

        _accumulate_game(stats, scores, bests, mine, votes)
        my_count = int(mine.get("count", 0) or 0)
        rank = _player_rank(scores, initials)
        stats["per_game"].append(
            {
                "slug": child.name,
                "title": meta.get("title") or child.name,
                "my_best": int(mine.get("best", 0) or 0) if my_count else None,
                "my_rank": rank,
                "my_plays": my_count,
                "top_score": scores[0].get("score") if scores else None,
                "top_initials": scores[0].get("initials") if scores else None,
            }
        )
    stats["per_game"].sort(key=lambda r: (-(r["my_best"] or 0), r["title"]))
    stats["achievements"] = evaluate(stats)
    stats["unlocked_count"] = sum(1 for a in stats["achievements"] if a["unlocked"])
    return stats


def list_players() -> list:
    """Enumerate every known callsign with cross-game standings.

    Standings rank by achievements, #1 finishes, machines played and scores
    submitted — never by summed raw scores, since every machine's score scale
    is its own. Callsigns come from each game's durable bests.players ledger
    (falling back to legacy bests.json), the current top-five, and ratings.
    """
    if not GAMES_DIR.exists():
        return []
    games = []
    total_games = 0
    for child in sorted(GAMES_DIR.iterdir()):
        if not child.is_dir() or _playable_file(child) is None:
            continue
        total_games += 1
        score_state = _read_json(child / "scores.json", {})
        scores = score_state.get("scores", [])
        bests = score_state.get("bests")
        if not isinstance(bests, dict):
            bests = _read_json(child / "bests.json", {})
        players = bests.get("players", {}) if isinstance(bests.get("players"), dict) else {}
        votes = _read_json(child / "ratings.json", {}).get("votes", [])
        games.append((scores, bests, players, votes))

    agg: dict[str, dict] = {}
    for scores, bests, players, votes in games:
        seen: set[str] = set()
        seen.update(str(k) for k in players)
        for s in scores:
            if isinstance(s, dict) and s.get("initials"):
                seen.add(str(s["initials"]))
        for v in votes:
            if isinstance(v, dict) and v.get("by"):
                seen.add(str(v["by"]))
        if bests.get("first_by"):
            seen.add(str(bests["first_by"]))
        for raw in sorted(seen):
            initials = _clean_initials(raw)
            if not initials:
                continue
            stats = agg.get(initials)
            if stats is None:
                stats = _empty_stats(initials)
                agg[initials] = stats
            mine = players.get(initials, {}) if isinstance(players.get(initials), dict) else {}
            _accumulate_game(stats, scores, bests, mine, votes)

    rows = list(agg.values())
    for stats in rows:
        stats["total_games"] = total_games
        stats["achievements"] = evaluate(stats)
        stats["unlocked_count"] = sum(1 for a in stats["achievements"] if a["unlocked"])
    rows.sort(
        key=lambda p: (
            -p["unlocked_count"],
            -p["crowns"],
            -p["podiums"],
            -p["games_scored"],
            -p["submits"],
            p["initials"],
        )
    )
    return rows


def _higher_is_better(d: Path) -> bool:
    """Whether this game's gameplay scoreboard ranks higher scores first.

    Kept separate from the benchmark score (always higher-is-better): a
    lower-is-better game (e.g. a time attack) stores raw scores but sorts
    and reports its board lowest-first. Defaults to True.
    """
    meta = _read_json(d / "meta.json", {})
    return meta.get("higher_is_better") is not False


def _game_card(slug: str) -> dict | None:
    d = _game_dir(slug)
    if d is None or _playable_file(d) is None:
        return None
    meta = _read_json(d / "meta.json", {})
    scores = _read_json(d / "scores.json", {}).get("scores", [])
    card = {
        "slug": slug,
        "title": meta.get("title") or slug,
        "test_id": meta.get("test_id", ""),
        "test_label": meta.get("test_label", meta.get("label", "")),
        "label": meta.get("label", meta.get("test_label", "")),
        "category": meta.get("category", ""),
        "type": meta.get("type", ""),
        "kind": meta.get("kind", ""),
        "model": meta.get("model", ""),
        "benchmark_score": meta.get("benchmark_score"),
        "max_score": meta.get("max_score"),
        "higher_is_better": meta.get("higher_is_better") is not False,
        "metric": meta.get("metric", "score"),
        "run_id": meta.get("run_id", ""),
        "lang": meta.get("lang", ""),
        "benchmark_date": meta.get("benchmark_date", ""),
        "published_at": meta.get("published_at", ""),
        "plays": int(meta.get("plays", 0) or 0),
        "top_score": scores[0] if scores else None,
        "score_count": len(scores),
        "rating": _rating_stats(d),
    }
    return card


def list_games(category: str | None = None) -> list:
    if not GAMES_DIR.exists():
        return []
    cards = []
    for child in sorted(GAMES_DIR.iterdir()):
        if child.is_dir():
            card = _game_card(child.name)
            if card:
                cards.append(card)
    if category:
        want = category.strip().lower()
        cards = [c for c in cards if (c.get("category") or "").lower() == want]
    # Group by category server-side (order by category, then benchmark quality
    # within it) so the index can render grouped sections without reshuffling.
    cards.sort(key=lambda c: ((c.get("category") or "").lower(), -(c["benchmark_score"] or 0), c["title"] or ""))
    return cards


@app.after_request
def _no_store_static(resp):
    if request.path.startswith("/static/") or resp.mimetype == "text/html":
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "games": len(list_games())})


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", games=list_games())


def _launch_lang(meta: dict) -> str:
    """Language tag for the sandbox, from the benchmark record's own meta.

    Empty means 'unstated' — the key is then omitted from the launch payload
    so the sandbox applies its own default instead of us hardcoding one.
    """
    return (meta.get("lang") or "").strip().lower()


def detect_game_keys(code: str) -> dict:
    """Smart keys: which controls a code-kind game actually uses.

    Scans the frozen source for pygame ``K_*`` constants and groups them
    into overlay clusters (arrows D-pad, WASD, space/enter/esc, extras).
    The play page renders only the clusters the game uses, plus the
    keyboard (high-score entry) and exit buttons. Empty/unreadable code
    returns every cluster (safe fallback = current full key set).
    """
    tokens = set(re.findall(r"K_([A-Z0-9]+)", code or ""))
    if not tokens:
        return {"arrows": True, "wasd": True, "space": True, "enter": True, "esc": True, "other": []}
    arrows = bool(tokens & {"UP", "DOWN", "LEFT", "RIGHT"})
    wasd = bool(tokens & {"W", "A", "S", "D"})
    # Extra single-letter keys (pause, mute, restart, ...). WASD ride
    # in their own cluster; everything else becomes compact buttons.
    other = sorted({t.lower() for t in tokens if len(t) == 1 and t.isalpha()} - {"w", "a", "s", "d"})[:6]
    return {
        "arrows": arrows,
        "wasd": wasd,
        "space": "SPACE" in tokens,
        "enter": bool(tokens & {"RETURN", "KP_ENTER"}),
        "esc": "ESCAPE" in tokens,
        "other": other,
    }


def _run_hint(lang: str) -> dict:
    """Human run instructions for a code-kind game, keyed by its language.

    Unknown/unstated languages get a generic sandbox-first hint; only
    python mentions pip/pygame (its own ecosystem facts).
    """
    name = f" ({lang})" if lang else ""
    short = f"🖥️ Desktop app{name} — play it live above, right in your browser, or download it below"
    if lang == "python":
        long = (
            "🎮 Press ▶ Play above to run it live in your browser, "
            "or download it below and run locally (pip install pygame), "
            "then enter your score by hand"
        )
    else:
        long = (
            "🎮 Press ▶ Play above to run it live in your browser, "
            "or download it below to run it yourself, then enter your score by hand"
        )
    return {"short": short, "long": long}


@app.route("/play/<slug>", methods=["GET"])
def play(slug):
    d = _game_dir(slug)
    if d is None or _playable_file(d) is None:
        return render_template("index.html", games=list_games(), error=f"Game '{slug}' not found."), 404
    card = _game_card(slug)
    meta = _read_json(d / "meta.json", {})
    with _lock:
        meta["plays"] = int(meta.get("plays", 0) or 0) + 1
        _write_json(d / "meta.json", meta)
        card["plays"] = meta["plays"]
    scores = _read_json(d / "scores.json", {}).get("scores", [])
    kind = meta.get("kind") or ("code" if (d / "game.py").exists() else "playable")
    code_text = (d / "game.py").read_text(encoding="utf-8", errors="replace") if (d / "game.py").exists() else ""
    code_lang = _launch_lang(meta)
    return render_template(
        "play.html",
        card=card,
        meta=meta,
        scores=scores,
        prompt=meta.get("prompt", ""),
        validation=(meta.get("validation") or {}).get("breakdown", {}),
        kind=kind,
        code_text=code_text,
        code_lang=code_lang,
        run_hint=_run_hint(code_lang),
        smart_keys=detect_game_keys(code_text) if kind == "code" else None,
        has_screenshot=(d / "screenshot.png").exists(),
    )


@app.route("/game/<slug>/index.html", methods=["GET"])
def serve_game(slug):
    d = _game_dir(slug)
    if d is None or not (d / "game.html").exists():
        return jsonify({"error": "game not found"}), 404
    return send_file(d / "game.html", mimetype="text/html")


@app.route("/game/<slug>/three.min.js", methods=["GET"])
def serve_game_three(slug):
    d = _game_dir(slug)
    if d is None or not (d / "game.html").is_file():
        return jsonify({"error": "game not found"}), 404
    bundle = Path(__file__).resolve().parent.parent / "three.min.js"
    if not bundle.is_file():
        return jsonify({"error": "Three.js bundle not installed"}), 404
    return send_file(bundle, mimetype="application/javascript")


@app.route("/game/<slug>/game.py", methods=["GET"])
def serve_game_code(slug):
    d = _game_dir(slug)
    if d is None or not (d / "game.py").exists():
        return jsonify({"error": "game not found"}), 404
    return send_file(d / "game.py", mimetype="text/plain", as_attachment=True, download_name=f"{slug}.py")


@app.route("/api/games/<slug>/launch", methods=["POST"])
def launch_game(slug):
    """Run a code-kind game in the Alpaca UI sandbox (same one the Tests
    page uses) and return the browser-facing noVNC launcher URL.

    The sandbox lives in the web backend, so this proxies server-side
    (the browser can't reach across origins). Spinning up the container
    takes a while — callers should show a loading state (up to ~2 min).
    """
    import uuid

    d = _game_dir(slug)
    if d is None or not (d / "game.py").exists():
        return jsonify({"error": "no runnable code game found for this slug"}), 404
    meta = _read_json(d / "meta.json", {})
    code = (d / "game.py").read_text(encoding="utf-8", errors="replace")
    # 2 h session lifetime: the sandbox container's PID1 sleep expires at
    # timeout+60 s and reaps the session, so the 10 min default would kill
    # long play sessions mid-game with no message.
    payload_obj: dict = {
        "code": code,
        "name": f"alpaca-arcade-{uuid.uuid4().hex}",
        "exclusive": False,
        "timeout": 7200,
    }
    if _launch_lang(meta):
        payload_obj["lang"] = _launch_lang(meta)
    payload = json.dumps(payload_obj).encode("utf-8")
    req = urllib.request.Request(
        f"{WEB_BASE}/api/sandbox/serve_ui", data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=150) as resp:
            res = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return jsonify({"error": f"sandbox launch failed: {str(e)[:200]}"}), 502
    if res.get("error") or not res.get("container_id"):
        return jsonify({"error": str(res.get("error") or "sandbox refused the launch")[:200]}), 502
    launcher_url = _browser_launcher_url(res["container_id"])
    return jsonify({"success": True, "launcher_url": launcher_url, "container_id": res["container_id"]})


@app.route("/api/games/<slug>/stop", methods=["POST"])
def stop_game(slug):
    """Stop a sandbox container started by launch_game.

    The embedded launcher hides its own Stop button, and the arcade page
    never called stop_serve — so every Play leaked a container. The
    play page calls this when relaunching (kills the previous session)
    and on pagehide (fire-and-forget, so exiting stops the sandbox).
    """
    data = request.get_json(silent=True) or {}
    cid = data.get("container_id")
    if not cid:
        return jsonify({"error": "No container_id provided"}), 400
    score_sync = None
    d = _game_dir(slug)
    if d is not None and _playable_file(d) is not None:
        try:
            score_sync, _ = _capture_container_score(d, cid)
        except Exception as e:
            score_sync = {"success": False, "error": f"final score sync failed: {str(e)[:200]}"}
    payload = json.dumps({"container_id": cid}).encode("utf-8")
    req = urllib.request.Request(
        f"{WEB_BASE}/api/sandbox/stop_serve", data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            res = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return jsonify({"error": f"sandbox stop failed: {str(e)[:200]}", "score_sync": score_sync}), 502
    if score_sync is not None:
        res["score_sync"] = score_sync
    return jsonify(res)


@app.route("/api/sandbox/ui_inner_status", methods=["POST"])
def sandbox_ui_inner_status():
    """Relay a launcher noVNC-state beacon to the web backend.

    The public proxy maps /api/* to the arcade, not the web backend —
    so the launcher page's relative beacon would 404. Proxy it
    server-side (same pattern as stop_game).
    """
    data = request.get_json(silent=True) or {}
    payload = json.dumps(
        {
            "container_id": data.get("container_id"),
            "state": data.get("state"),
            "detail": data.get("detail", ""),
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{WEB_BASE}/api/sandbox/ui_inner_status", data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            res = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return jsonify({"error": f"inner-status relay failed: {str(e)[:200]}"}), 502
    return jsonify(res)


def _proxy_sandbox(path: str):
    """Proxy a sandbox endpoint from the web backend (server-side relay).

    The public proxy maps /api/* to the arcade, not the web backend —
    so the launcher page's relative /api/sandbox/* calls would 404.
    Relay them server-side so the key bar, log, screenshot, restart
    and stop controls all work from the arcade domain.
    """
    data = request.get_json(silent=True) or {}
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(f"{WEB_BASE}{path}", data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            res = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return jsonify({"error": f"sandbox {path} failed: {str(e)[:200]}"}), 502
    return jsonify(res)


@app.route("/api/sandbox/ui/exec", methods=["POST"])
def sandbox_ui_exec():
    """Relay launcher terminal exec to the web backend."""
    return _proxy_sandbox("/api/sandbox/ui/exec")


@app.route("/api/sandbox/ui/status", methods=["POST"])
def sandbox_ui_status():
    """Relay launcher status to the web backend."""
    return _proxy_sandbox("/api/sandbox/ui/status")


@app.route("/api/sandbox/ui/screenshot", methods=["POST"])
def sandbox_ui_screenshot():
    """Relay launcher screenshot to the web backend."""
    return _proxy_sandbox("/api/sandbox/ui/screenshot")


@app.route("/api/sandbox/ui/restart", methods=["POST"])
def sandbox_ui_restart():
    """Relay launcher restart to the web backend."""
    return _proxy_sandbox("/api/sandbox/ui/restart")


@app.route("/api/sandbox/stop_serve", methods=["POST"])
def sandbox_stop_serve():
    """Relay stop_serve to the web backend."""
    return _proxy_sandbox("/api/sandbox/stop_serve")


def _browser_launcher_url(container_id: str) -> str:
    """Browser-facing noVNC launcher URL for this request's origin.

    Direct LAN access reaches the web backend on :5000, but behind the
    public TLS proxy only :5001 is open (and browsers block https→http
    mixed content) — so a forwarded-https request gets the same public
    origin and relies on the proxy mapping /ui/launcher/* + /serve/*
    through to the backend.
    """
    host = (request.host or "").split(":")[0] or "localhost"
    if (request.headers.get("X-Forwarded-Proto", "") or "").split(",")[0].strip().lower() == "https":
        return f"https://{host}/ui/launcher/{container_id}?embed=1"
    return f"http://{host}:5000/ui/launcher/{container_id}?embed=1"


@app.route("/game/<slug>/screenshot.png", methods=["GET"])
def serve_game_screenshot(slug):
    d = _game_dir(slug)
    if d is None or not (d / "screenshot.png").exists():
        return jsonify({"error": "screenshot not found"}), 404
    return send_file(d / "screenshot.png", mimetype="image/png")


@app.route("/api/games", methods=["GET"])
def api_games():
    category = request.args.get("category") or None
    return jsonify({"success": True, "games": list_games(category=category)})


@app.route("/api/games/<slug>", methods=["GET"])
def api_game(slug):
    card = _game_card(slug)
    if card is None:
        return jsonify({"success": False, "error": "game not found"}), 404
    d = _game_dir(slug)
    card["scores"] = _read_json(d / "scores.json", {}).get("scores", [])
    card["prompt"] = _read_json(d / "meta.json", {}).get("prompt", "")
    return jsonify({"success": True, "game": card})


def _store_score(d: Path, initials: str, score: int, capture_id: str | None = None) -> dict:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    with _lock, open(d / ".scores.lock", "a", encoding="utf-8") as score_lock:
        fcntl.flock(score_lock, fcntl.LOCK_EX)
        data = _read_json(d / "scores.json", {})
        scores = data.get("scores", []) if isinstance(data.get("scores"), list) else []
        captures = data.get("captures", [])
        if not isinstance(captures, list):
            captures = []
        bests = data.get("bests")
        if not isinstance(bests, dict):
            bests = _read_json(d / "bests.json", {})
        if capture_id is not None and capture_id in captures:
            if _read_json(d / "bests.json", {}) != bests:
                _write_json(d / "bests.json", bests)
            return {
                "scores": scores,
                "status": "unchanged",
                "duplicate": True,
                "new_unlocks": [],
                "personal_best": False,
                "pioneer": False,
            }
        before = unlocked_ids(player_stats(initials))
        pioneer = not scores
        scores.append({"initials": initials, "score": score, "at": stamp})
        desc = _higher_is_better(d)
        scores.sort(key=lambda s: int(s.get("score", 0) or 0), reverse=desc)
        scores = scores[:MAX_SCORES]
        if not isinstance(bests.get("players"), dict):
            bests = {"submits": 0, "first_by": None, "players": {}}
        bests["submits"] = int(bests.get("submits", 0) or 0) + 1
        if pioneer and not bests.get("first_by"):
            bests["first_by"] = initials
        entry = bests["players"].get(initials, {})
        had_best = entry.get("best") is not None
        prev_best = int(entry.get("best", 0) or 0)
        if desc:
            personal_best = had_best and score > prev_best
            entry["best"] = max(prev_best, score) if had_best else score
        else:
            personal_best = had_best and score < prev_best
            entry["best"] = min(prev_best, score) if had_best else score
        entry["count"] = int(entry.get("count", 0) or 0) + 1
        entry["pbs"] = int(entry.get("pbs", 0) or 0) + (1 if personal_best else 0)
        entry["max_ever"] = max(int(entry.get("max_ever", 0) or 0), score)
        hour = _hour_of(stamp)
        if hour is not None and NIGHT_OWL_START <= hour < NIGHT_OWL_END:
            entry.setdefault("night_owl_at", stamp)
        entry.setdefault("first_at", stamp)
        bests["players"][initials] = entry
        if capture_id is not None:
            captures.append(capture_id)
        _write_json(d / "scores.json", {**data, "scores": scores, "bests": bests, "captures": captures})
        _write_json(d / "bests.json", bests)
        after_stats = player_stats(initials)
    new_unlocks = [a for a in after_stats["achievements"] if a["unlocked"] and a["id"] not in before]
    rank = next((i + 1 for i, s in enumerate(scores) if s["initials"] == initials and s["score"] == score), None)
    return {
        "scores": scores,
        "rank": rank,
        "made_board": rank is not None,
        "personal_best": personal_best,
        "pioneer": pioneer,
        "new_unlocks": new_unlocks,
        "player_url": f"/player/{initials}",
        "duplicate": False,
    }


def _sync_score_from_container(container_id: str) -> dict | None:
    """Read the latest run score from a running UI sandbox container.

    Tries multiple files in order (first hit wins):
    1. /tmp/alpaca_score.json — canonical per benchmark prompt,
       format {initials, score}
    2. /tmp/high_scores.json — some games write only this
       (array of {name, score}); we use the latest entry.

    Returns the parsed score dict, None if nothing exists yet,
    or {"error": ...} on failure.
    """
    candidates = [
        ("/tmp/alpaca_score.json", "initials", "score"),
        ("/tmp/high_scores.json", "name", "score"),
    ]
    for path, name_key, score_key in candidates:
        script = (
            "import hashlib,json,os\n"
            "try:\n"
            f" with open({path!r}, 'rb') as f:\n"
            "  before = os.fstat(f.fileno())\n"
            "  raw = f.read(1048577)\n"
            "  after = os.fstat(f.fileno())\n"
            f" current = os.stat({path!r})\n"
            " revision = lambda s: [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]\n"
            " if revision(before) == revision(after) == revision(current) and len(raw) <= 1048576:\n"
            "  print(json.dumps({'data': json.loads(raw), 'revision': revision(after), "
            "'digest': hashlib.sha256(raw).hexdigest()}))\n"
            "except (OSError,ValueError):\n"
            " pass\n"
        )
        payload = json.dumps({"container_id": container_id, "command": f"python3 -c {shlex.quote(script)}"}).encode(
            "utf-8"
        )
        req = urllib.request.Request(
            f"{WEB_BASE}/api/sandbox/ui/exec", data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                res = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"error": f"score read failed: {str(e)[:200]}"}
        if not isinstance(res, dict):
            return {"error": "invalid score read response"}
        if res.get("error"):
            return {"error": res["error"]}
        if res.get("exit_code") not in (None, 0):
            return {"error": "score reader failed"}
        raw = (res.get("output") or "").strip()
        if not raw or raw == "":
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        revision = None
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if isinstance(data, dict) and "data" in data and "revision" in data:
            revision = data["revision"]
            digest = data.get("digest", digest)
            data = data["data"]
        if isinstance(data, list) and data:
            data = data[-1]
        if isinstance(data, dict) and name_key in data and score_key in data:
            return {
                "initials": data[name_key],
                "score": data[score_key],
                "source": path,
                "revision": revision,
                "digest": digest,
            }
    return None


def _capture_container_score(d: Path, container_id: str) -> tuple[dict, int]:
    score_data = _sync_score_from_container(container_id)
    if score_data is None:
        return {"success": True, "status": "no score yet"}, 200
    if "error" in score_data:
        return {"success": False, "error": score_data["error"]}, 502
    initials = _clean_initials(score_data.get("initials")) or "YOU"
    try:
        score = int(score_data.get("score", 0))
    except (TypeError, ValueError, OverflowError):
        return {"success": False, "error": "score must be an integer"}, 400
    if score < 0 or score > 999_999_999:
        return {"success": False, "error": "score out of range"}, 400
    identity = [
        container_id,
        score_data.get("source", "/tmp/alpaca_score.json"),
        score_data.get("revision"),
        score_data.get("digest"),
        initials,
        score,
    ]
    capture_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    try:
        stored = _store_score(d, initials, score, capture_id=capture_id)
    except OSError as e:
        return {"success": False, "error": f"score store failed: {str(e)[:200]}"}, 502
    return {"success": True, "initials": initials, "score": score, **stored, "auto": True}, 200


@app.route("/api/games/<slug>/sync_score", methods=["POST"])
def api_sync_score(slug):
    """Pick up a score auto-written by the sandbox game.

    Code-kind games write {initials, score} to /tmp/alpaca_score.json
    (working dir of the sandbox container) when a run finishes. This
    endpoint reads that file via the web backend (which owns the
    container), stores the score through the same ledger as manual
    submission, and reports rank + achievements. Returns a marker when
    no score file exists yet so the client can keep polling.
    """
    d = _game_dir(slug)
    if d is None or _playable_file(d) is None:
        return jsonify({"success": False, "error": "game not found"}), 404
    data = request.get_json(silent=True) or {}
    container_id = data.get("container_id")
    if not container_id:
        return jsonify({"success": False, "error": "No container_id provided"}), 400
    result, status = _capture_container_score(d, container_id)
    return jsonify(result), status


@app.route("/api/games/<slug>/scores", methods=["POST"])
def api_submit_score(slug):
    d = _game_dir(slug)
    if d is None or _playable_file(d) is None:
        return jsonify({"success": False, "error": "game not found"}), 404
    body = request.get_json(force=True, silent=True) or {}
    initials = _clean_initials(body.get("initials")) or "YOU"
    try:
        score = int(body.get("score", 0))
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "score must be an integer"}), 400
    if score < 0 or score > 999_999_999:
        return jsonify({"success": False, "error": "score out of range"}), 400
    return jsonify({"success": True, **_store_score(d, initials, score)})


@app.route("/api/games/<slug>/rate", methods=["POST"])
def api_rate(slug):
    d = _game_dir(slug)
    if d is None or _playable_file(d) is None:
        return jsonify({"success": False, "error": "game not found"}), 404
    body = request.get_json(force=True, silent=True) or {}
    try:
        stars = int(body.get("stars", 0))
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "stars must be 1-5"}), 400
    if stars < 1 or stars > 5:
        return jsonify({"success": False, "error": "stars must be 1-5"}), 400
    voter = _clean_initials(body.get("initials")) or None
    before = unlocked_ids(player_stats(voter)) if voter else set()
    with _lock:
        data = _read_json(d / "ratings.json", {})
        votes = data.get("votes", []) if isinstance(data.get("votes"), list) else []
        votes.append(
            {
                "stars": stars,
                "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "ip": request.remote_addr,
                **({"by": voter} if voter else {}),
            }
        )
        _write_json(d / "ratings.json", {"votes": votes})
    new_unlocks: list = []
    player_url: str | None = None
    if voter:
        after = player_stats(voter)
        new_unlocks = [a for a in after["achievements"] if a["unlocked"] and a["id"] not in before]
        player_url = f"/player/{voter}"
    return jsonify(
        {"success": True, "rating": _rating_stats(d), "new_unlocks": new_unlocks, "player_url": player_url}
    )


@app.route("/player/<initials>", methods=["GET"])
def player_page(initials):
    clean = _clean_initials(initials)
    if not clean:
        return render_template("index.html", games=list_games(), error="Pick a 3-letter callsign."), 404
    stats = player_stats(clean)
    return render_template("player.html", stats=stats)


@app.route("/api/players/<initials>", methods=["GET"])
def api_player(initials):
    clean = _clean_initials(initials)
    if not clean:
        return jsonify({"success": False, "error": "invalid callsign"}), 404
    return jsonify({"success": True, "player": player_stats(clean)})


@app.route("/players", methods=["GET"])
def players_page():
    return render_template("players.html", players=list_players())


@app.route("/api/players", methods=["GET"])
def api_players():
    q = (request.args.get("q") or "").strip().lower()
    players = [p for p in list_players() if not q or q in p["initials"].lower()]
    return jsonify({"success": True, "players": players, "count": len(players)})


@app.route("/api/admin/games/<slug>", methods=["DELETE"])
def api_unpublish(slug):
    if not ADMIN_TOKEN or request.headers.get("X-Arcade-Token") != ADMIN_TOKEN:
        return jsonify({"success": False, "error": "unauthorized"}), 401
    d = _game_dir(slug)
    if d is None or not d.exists():
        return jsonify({"success": False, "error": "game not found"}), 404
    import shutil

    shutil.rmtree(d, ignore_errors=True)
    return jsonify({"success": True, "removed": slug})


if __name__ == "__main__":
    GAMES_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="0.0.0.0", port=PORT, threaded=True)
