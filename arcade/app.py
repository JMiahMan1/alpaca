#!/usr/bin/env python3
"""Alpaca Arcade — public scoreboard web service for top-rated benchmark games.

Standalone Flask service (own port, default 5001). Each published game is a
fully self-contained directory under data/arcade/games/<slug>/:

    game.html    frozen copy of the benchmark artifact (served same-origin so
                 the arcade page can read the game's localStorage for
                 score auto-capture)
    meta.json    title, test id/label, model, benchmark score + date, prompt,
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
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

ARCADE_DIR = Path(os.getenv("ARCADE_DIR", "data/arcade"))
GAMES_DIR = ARCADE_DIR / "games"
ADMIN_TOKEN = os.getenv("ARCADE_ADMIN_TOKEN", "")
PORT = int(os.getenv("ARCADE_PORT", "5001"))
MAX_SCORES = 5

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


def _game_card(slug: str) -> dict | None:
    d = _game_dir(slug)
    if d is None or not (d / "game.html").exists():
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
    if request.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "games": len(list_games())})


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", games=list_games())


@app.route("/play/<slug>", methods=["GET"])
def play(slug):
    d = _game_dir(slug)
    if d is None or not (d / "game.html").exists():
        return render_template("index.html", games=list_games(), error=f"Game '{slug}' not found."), 404
    card = _game_card(slug)
    meta = _read_json(d / "meta.json", {})
    with _lock:
        meta["plays"] = int(meta.get("plays", 0) or 0) + 1
        _write_json(d / "meta.json", meta)
        card["plays"] = meta["plays"]
    scores = _read_json(d / "scores.json", {}).get("scores", [])
    return render_template(
        "play.html",
        card=card,
        meta=meta,
        scores=scores,
        prompt=meta.get("prompt", ""),
        validation=(meta.get("validation") or {}).get("breakdown", {}),
    )


@app.route("/game/<slug>/index.html", methods=["GET"])
def serve_game(slug):
    d = _game_dir(slug)
    if d is None or not (d / "game.html").exists():
        return jsonify({"error": "game not found"}), 404
    return send_file(d / "game.html", mimetype="text/html")


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
    if d is None or not (d / "game.html").exists():
        return jsonify({"success": False, "error": "game not found"}), 404
    body = request.get_json(force=True, silent=True) or {}
    initials = re.sub(r"[^A-Za-z0-9]", "", str(body.get("initials", "")))[:3].upper() or "YOU"
    try:
        score = int(body.get("score", 0))
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "score must be an integer"}), 400
    if score < 0 or score > 999_999_999:
        return jsonify({"success": False, "error": "score out of range"}), 400
    with _lock:
        data = _read_json(d / "scores.json", {})
        scores = data.get("scores", []) if isinstance(data.get("scores"), list) else []
        scores.append({"initials": initials, "score": score, "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        scores.sort(key=lambda s: -int(s.get("score", 0) or 0))
        scores = scores[:MAX_SCORES]
        _write_json(d / "scores.json", {"scores": scores})
    rank = next((i + 1 for i, s in enumerate(scores) if s["initials"] == initials and s["score"] == score), None)
    return jsonify({"success": True, "scores": scores, "rank": rank, "made_board": rank is not None})


@app.route("/api/games/<slug>/rate", methods=["POST"])
def api_rate(slug):
    d = _game_dir(slug)
    if d is None or not (d / "game.html").exists():
        return jsonify({"success": False, "error": "game not found"}), 404
    body = request.get_json(force=True, silent=True) or {}
    try:
        stars = int(body.get("stars", 0))
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "stars must be 1-5"}), 400
    if stars < 1 or stars > 5:
        return jsonify({"success": False, "error": "stars must be 1-5"}), 400
    with _lock:
        data = _read_json(d / "ratings.json", {})
        votes = data.get("votes", []) if isinstance(data.get("votes"), list) else []
        votes.append({"stars": stars, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "ip": request.remote_addr})
        _write_json(d / "ratings.json", {"votes": votes})
    return jsonify({"success": True, "rating": _rating_stats(d)})


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
