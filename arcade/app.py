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

import json
import os
import re
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


def player_stats(initials: str) -> dict:
    """Aggregate one player's cross-game stats + achievements.

    Reads every published game's scores/bests/ratings; purely derived, so it
    stays correct no matter when games were published or removed.
    """
    initials = (initials or "").upper()
    stats = {
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
        "per_game": [],
    }
    if not GAMES_DIR.exists():
        stats["achievements"] = evaluate(stats)
        return stats
    for child in sorted(GAMES_DIR.iterdir()):
        if not child.is_dir() or _playable_file(child) is None:
            continue
        meta = _read_json(child / "meta.json", {})
        scores = _read_json(child / "scores.json", {}).get("scores", [])
        bests = _read_json(child / "bests.json", {})
        players = bests.get("players", {}) if isinstance(bests.get("players"), dict) else {}
        mine = players.get(initials, {}) if isinstance(players.get(initials), dict) else {}
        votes = _read_json(child / "ratings.json", {}).get("votes", [])

        stats["total_games"] += 1
        my_count = int(mine.get("count", 0) or 0)
        if my_count:
            stats["games_scored"] += 1
            stats["submits"] += my_count
            stats["personal_bests"] += int(mine.get("pbs", 0) or 0)
            stats["best_score"] = max(stats["best_score"], int(mine.get("best", 0) or 0))
        if bests.get("first_by") == initials:
            stats["pioneered"] += 1
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
        "test_label": meta.get("test_label", ""),
        "model": meta.get("model", ""),
        "benchmark_score": meta.get("benchmark_score"),
        "max_score": meta.get("max_score"),
        "lang": meta.get("lang", ""),
        "benchmark_date": meta.get("benchmark_date", ""),
        "published_at": meta.get("published_at", ""),
        "plays": int(meta.get("plays", 0) or 0),
        "top_score": scores[0] if scores else None,
        "score_count": len(scores),
        "rating": _rating_stats(d),
    }
    return card


def list_games() -> list:
    if not GAMES_DIR.exists():
        return []
    cards = []
    for child in sorted(GAMES_DIR.iterdir()):
        if child.is_dir():
            card = _game_card(child.name)
            if card:
                cards.append(card)
    cards.sort(key=lambda c: (-(c["benchmark_score"] or 0), c["title"]))
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
        has_screenshot=(d / "screenshot.png").exists(),
    )


@app.route("/game/<slug>/index.html", methods=["GET"])
def serve_game(slug):
    d = _game_dir(slug)
    if d is None or not (d / "game.html").exists():
        return jsonify({"error": "game not found"}), 404
    return send_file(d / "game.html", mimetype="text/html")


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
    d = _game_dir(slug)
    if d is None or not (d / "game.py").exists():
        return jsonify({"error": "no runnable code game found for this slug"}), 404
    meta = _read_json(d / "meta.json", {})
    code = (d / "game.py").read_text(encoding="utf-8", errors="replace")
    payload_obj: dict = {"code": code}
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
    host = (request.host or "").split(":")[0] or "localhost"
    launcher_url = f"http://{host}:5000/ui/launcher/{res['container_id']}?embed=1"
    return jsonify({"success": True, "launcher_url": launcher_url, "container_id": res["container_id"]})


@app.route("/game/<slug>/screenshot.png", methods=["GET"])
def serve_game_screenshot(slug):
    d = _game_dir(slug)
    if d is None or not (d / "screenshot.png").exists():
        return jsonify({"error": "screenshot not found"}), 404
    return send_file(d / "screenshot.png", mimetype="image/png")


@app.route("/api/games", methods=["GET"])
def api_games():
    return jsonify({"success": True, "games": list_games()})


@app.route("/api/games/<slug>", methods=["GET"])
def api_game(slug):
    card = _game_card(slug)
    if card is None:
        return jsonify({"success": False, "error": "game not found"}), 404
    d = _game_dir(slug)
    card["scores"] = _read_json(d / "scores.json", {}).get("scores", [])
    card["prompt"] = _read_json(d / "meta.json", {}).get("prompt", "")
    return jsonify({"success": True, "game": card})


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
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    with _lock:
        before = unlocked_ids(player_stats(initials))
        data = _read_json(d / "scores.json", {})
        scores = data.get("scores", []) if isinstance(data.get("scores"), list) else []
        pioneer = not scores
        scores.append({"initials": initials, "score": score, "at": stamp})
        scores.sort(key=lambda s: -int(s.get("score", 0) or 0))
        scores = scores[:MAX_SCORES]
        _write_json(d / "scores.json", {"scores": scores})
        # Cross-game personal-best ledger (top-5 alone forgets old bests).
        bests = _read_json(d / "bests.json", {})
        if not isinstance(bests.get("players"), dict):
            bests = {"submits": 0, "first_by": None, "players": {}}
        bests["submits"] = int(bests.get("submits", 0) or 0) + 1
        if pioneer and not bests.get("first_by"):
            bests["first_by"] = initials
        entry = bests["players"].get(initials, {})
        prev_best = int(entry.get("best", 0) or 0)
        personal_best = bool(prev_best) and score > prev_best
        entry["best"] = max(prev_best, score)
        entry["count"] = int(entry.get("count", 0) or 0) + 1
        entry["pbs"] = int(entry.get("pbs", 0) or 0) + (1 if personal_best else 0)
        entry.setdefault("first_at", stamp)
        bests["players"][initials] = entry
        _write_json(d / "bests.json", bests)
        after_stats = player_stats(initials)
    new_unlocks = [a for a in after_stats["achievements"] if a["unlocked"] and a["id"] not in before]
    rank = next((i + 1 for i, s in enumerate(scores) if s["initials"] == initials and s["score"] == score), None)
    return jsonify(
        {
            "success": True,
            "scores": scores,
            "rank": rank,
            "made_board": rank is not None,
            "personal_best": personal_best,
            "pioneer": pioneer,
            "new_unlocks": new_unlocks,
            "player_url": f"/player/{initials}",
        }
    )


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
    return jsonify({"success": True, "rating": _rating_stats(d)})


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
