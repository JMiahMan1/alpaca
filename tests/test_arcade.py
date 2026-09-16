"""Tests for the arcade service (port 5001), the publish helper, and the
Alpaca publish/unpublish API endpoints.

All filesystem state is redirected to tmp dirs via monkeypatched GAMES_DIR;
nothing touches the real data/arcade.
"""

import json

import pytest

import web.arcade_publish as ap


@pytest.fixture
def games_dir(tmp_path, monkeypatch):
    d = tmp_path / "games"
    monkeypatch.setattr(ap, "GAMES_DIR", d)
    return d


@pytest.fixture
def source_html(tmp_path):
    p = tmp_path / "demo.html"
    p.write_text("<html><body>demo game</body></html>")
    return p


def _publish(games_dir, source_html, **kw):
    params = {
        "model": "demo-model",
        "test_id": "demo_breakout",
        "benchmark_score": 92.5,
        "prompt": "build breakout",
        "run_date": "2026-09-11",
        "source_file": source_html,
    }
    params.update(kw)
    return ap.publish_game(**params)


# --- slugify ---


def test_slugify_basic():
    assert ap.slugify("qwen3:35b", "space_invaders") == "qwen3-35b_space-invaders"


def test_slugify_empty_fallback():
    assert ap.slugify("", "") == "game_game"
    assert ap.slugify("!!!", "???") == "game_game"


# --- publish_game ---


def test_publish_copies_game_and_meta(games_dir, source_html):
    res = _publish(games_dir, source_html)
    assert res["slug"] == "demo-model_demo-breakout"
    assert res["url"] == "/play/demo-model_demo-breakout"
    assert res["republished"] is False
    game_dir = games_dir / res["slug"]
    assert (game_dir / "game.html").read_text() == "<html><body>demo game</body></html>"
    meta = json.loads((game_dir / "meta.json").read_text())
    assert meta["model"] == "demo-model"
    assert meta["benchmark_score"] == 92.5
    assert meta["prompt"] == "build breakout"
    # Both date keys are written (arcade cards/templates read benchmark_date).
    assert meta["benchmark_date"] == "2026-09-11"
    assert meta["run_date"] == "2026-09-11"
    # Player files are initialised.
    assert json.loads((game_dir / "scores.json").read_text()) == []
    assert json.loads((game_dir / "ratings.json").read_text()) == {}


def test_publish_leaves_files_writable_by_container_user(games_dir, source_html):
    """Publishes often happen as root while the arcade serves as non-root uid 1000:
    everything must stay world-writable or score/plays writes 500."""
    import os
    import stat

    res = _publish(games_dir, source_html)
    game_dir = games_dir / res["slug"]
    for name in ("game.html", "meta.json", "scores.json", "ratings.json"):
        mode = stat.S_IMODE(os.stat(game_dir / name).st_mode)
        assert mode & stat.S_IWOTH, f"{name} not world-writable ({oct(mode)})"
    assert stat.S_IMODE(os.stat(game_dir).st_mode) & 0o777 == 0o777


def test_publish_requires_model_and_test(games_dir, source_html):
    with pytest.raises(ValueError):
        ap.publish_game(model="", test_id="x", source_file=source_html)
    with pytest.raises(ValueError):
        ap.publish_game(model="m", test_id="", source_file=source_html)


def test_publish_missing_artifact(games_dir, tmp_path):
    with pytest.raises(FileNotFoundError):
        ap.publish_game(model="m", test_id="nope", source_file=tmp_path / "missing.html")


def test_republish_preserves_player_data(games_dir, source_html, tmp_path):
    res = _publish(games_dir, source_html)
    game_dir = games_dir / res["slug"]
    (game_dir / "scores.json").write_text(json.dumps({"scores": [{"initials": "ABC", "score": 10}]}))
    (game_dir / "ratings.json").write_text(json.dumps({"votes": [{"stars": 5}]}))

    src2 = tmp_path / "demo2.html"
    src2.write_text("<html><body>v2</body></html>")
    res2 = _publish(games_dir, src2, benchmark_score=95.0)
    assert res2["republished"] is True
    assert (game_dir / "game.html").read_text() == "<html><body>v2</body></html>"
    meta = json.loads((game_dir / "meta.json").read_text())
    assert meta["benchmark_score"] == 95.0
    scores = json.loads((game_dir / "scores.json").read_text())
    assert scores["scores"][0]["initials"] == "ABC"
    assert json.loads((game_dir / "ratings.json").read_text())["votes"][0]["stars"] == 5


def test_unpublish_and_slugs(games_dir, source_html):
    res = _publish(games_dir, source_html)
    assert ap.published_slugs() == {res["slug"]}
    assert ap.is_published(res["slug"]) is True
    assert ap.unpublish_game(res["slug"]) is True
    assert ap.is_published(res["slug"]) is False
    assert ap.published_slugs() == set()
    assert ap.unpublish_game("never-existed") is False


# --- arcade service ---


@pytest.fixture
def arcade_client(games_dir, source_html, monkeypatch):
    import arcade.app as arcade_app

    monkeypatch.setattr(arcade_app, "GAMES_DIR", games_dir)
    monkeypatch.setattr(arcade_app, "ADMIN_TOKEN", "test-token")
    _publish(games_dir, source_html)
    arcade_app.app.config["TESTING"] = True
    with arcade_app.app.test_client() as client:
        yield client


def test_arcade_index_and_play(arcade_client):
    res = arcade_client.get("/")
    assert res.status_code == 200
    assert b"demo-model_demo-breakout" in res.data
    # Benchmark date renders on the index card (regression: run_date/benchmark_date key mismatch).
    assert b"2026-09-11" in res.data

    res = arcade_client.get("/play/demo-model_demo-breakout")
    assert res.status_code == 200
    assert b"2026-09-11" in res.data
    assert b"build breakout" in res.data
    # Fullscreen play mode: toggle, overlay key bar + exit present.
    assert b"btn-fullscreen" in res.data
    assert b"play-overlay-bar" in res.data
    assert b"play-keys" in res.data
    assert b"btn-exit-full" in res.data
    # Mobile refactor: D-pad + action clusters, high-score keyboard button.
    assert b"pad-cluster" in res.data
    assert b"action-cluster" in res.data
    assert b"btn-keyboard" in res.data
    assert b"score-initials" in res.data
    # ⌨ types into the game via a focus-proxy input: stays in fullscreen,
    # never scrolls to the score form (regression: old handler exited to it).
    assert b'id="kbd-proxy"' in res.data
    assert b"open the initials form" not in res.data


def test_arcade_play_missing(arcade_client):
    res = arcade_client.get("/play/nope-not-here")
    assert res.status_code == 404


def test_arcade_score_submit_and_top5(arcade_client):
    for i, pts in enumerate([50, 200, 30, 400, 10, 300]):
        res = arcade_client.post(
            "/api/games/demo-model_demo-breakout/scores",
            json={"initials": f"P{i}", "score": pts},
        )
        assert res.status_code == 200
    data = json.loads(res.data.decode())
    assert data["success"] is True
    assert len(data["scores"]) == 5  # capped at MAX_SCORES
    assert [s["score"] for s in data["scores"]] == [400, 300, 200, 50, 30]
    assert data["rank"] == 2  # P5's 300 is second
    assert data["made_board"] is True


def test_arcade_score_validation(arcade_client):
    res = arcade_client.post("/api/games/demo-model_demo-breakout/scores", json={"initials": "X", "score": "abc"})
    assert res.status_code == 400
    res = arcade_client.post("/api/games/demo-model_demo-breakout/scores", json={"initials": "X", "score": -5})
    assert res.status_code == 400
    res = arcade_client.post("/api/games/missing-game/scores", json={"initials": "X", "score": 5})
    assert res.status_code == 404


def test_arcade_rating_average(arcade_client):
    for stars in (5, 3, 4):
        res = arcade_client.post("/api/games/demo-model_demo-breakout/rate", json={"stars": stars})
        assert res.status_code == 200
    data = json.loads(res.data.decode())
    assert data["rating"] == {"count": 3, "average": 4.0}

    res = arcade_client.post("/api/games/demo-model_demo-breakout/rate", json={"stars": 6})
    assert res.status_code == 400


def test_arcade_admin_delete_requires_token(arcade_client):
    res = arcade_client.delete("/api/admin/games/demo-model_demo-breakout")
    assert res.status_code == 401
    res = arcade_client.delete("/api/admin/games/demo-model_demo-breakout", headers={"X-Arcade-Token": "wrong"})
    assert res.status_code == 401
    res = arcade_client.delete("/api/admin/games/demo-model_demo-breakout", headers={"X-Arcade-Token": "test-token"})
    assert res.status_code == 200
    assert arcade_client.get("/play/demo-model_demo-breakout").status_code == 404


def test_arcade_slug_traversal_blocked(arcade_client):
    res = arcade_client.get("/play/..%2F..%2Fapp")
    assert res.status_code in (404, 400)


# --- Alpaca publish/unpublish endpoints ---


@pytest.fixture
def web_client(games_dir, monkeypatch):
    from web.app import app as web_app

    monkeypatch.setattr(ap, "GAMES_DIR", games_dir)
    web_app.config["TESTING"] = True
    with web_app.test_client() as client:
        yield client


def test_web_publish_endpoint(web_client, source_html):
    res = web_client.post(
        "/api/arcade/publish",
        json={"model": "demo-model", "test_id": "demo_breakout", "benchmark_score": 92.5},
    )
    # No source_file param and no data/artifacts entry -> 404 FileNotFoundError.
    assert res.status_code in (200, 404)
    if res.status_code == 404:
        assert json.loads(res.data.decode())["success"] is False


def test_web_publish_missing_params(web_client):
    res = web_client.post("/api/arcade/publish", json={"model": "", "test_id": ""})
    assert res.status_code == 404
    assert json.loads(res.data.decode())["success"] is False


def test_web_unpublish_endpoint(web_client, games_dir, source_html):
    slug = _publish(games_dir, source_html)["slug"]
    res = web_client.post("/api/arcade/unpublish", json={"slug": slug})
    assert res.status_code == 200
    data = json.loads(res.data.decode())
    assert data == {"success": True, "slug": slug, "removed": True}

    res = web_client.post("/api/arcade/unpublish", json={"slug": slug})
    assert res.status_code == 200
    assert json.loads(res.data.decode())["removed"] is False


def test_web_unpublish_requires_slug(web_client):
    res = web_client.post("/api/arcade/unpublish", json={})
    assert res.status_code == 400


def test_web_published_list(web_client, games_dir, source_html):
    res = web_client.get("/api/arcade/published")
    assert res.status_code == 200
    assert json.loads(res.data.decode()) == {"slugs": [], "games": []}
    slug = _publish(games_dir, source_html)["slug"]
    res = web_client.get("/api/arcade/published")
    data = json.loads(res.data.decode())
    assert data["slugs"] == [slug]
    assert len(data["games"]) == 1
    game = data["games"][0]
    assert game["slug"] == slug
    assert game["title"] == "Demo Breakout"
    assert game["model"] == "demo-model"
    assert game["benchmark_score"] == 92.5
    assert game["run_date"] == "2026-09-11"
    assert game["plays"] == 0
    assert game["auto_published"] is False


def test_published_games_skips_broken_entries(games_dir, source_html):
    _publish(games_dir, source_html)
    # Directory without meta.json is ignored.
    (games_dir / "junk").mkdir()
    # Corrupt meta is ignored.
    bad = games_dir / "bad-game"
    bad.mkdir()
    (bad / "meta.json").write_text("{not json")
    games = ap.published_games()
    assert [g["slug"] for g in games] == ["demo-model_demo-breakout"]


def test_get_auto_publish_score_default(monkeypatch):
    monkeypatch.delenv("ARCADE_AUTO_PUBLISH_SCORE", raising=False)
    assert ap.get_auto_publish_score() == 80.0
    monkeypatch.setenv("ARCADE_AUTO_PUBLISH_SCORE", "70")
    assert ap.get_auto_publish_score() == 70.0
    monkeypatch.setenv("ARCADE_AUTO_PUBLISH_SCORE", "junk")
    assert ap.get_auto_publish_score() == 80.0


def test_web_arcade_settings_get(web_client, monkeypatch):
    monkeypatch.setenv("ARCADE_AUTO_PUBLISH_SCORE", "75")
    res = web_client.get("/api/arcade/settings")
    assert res.status_code == 200
    data = json.loads(res.data.decode())
    assert data["live_auto_publish_score"] == 75.0


def test_web_arcade_settings_post_validates(web_client):
    res = web_client.post("/api/arcade/settings", json={"auto_publish_score": "high"})
    assert res.status_code == 400
    res = web_client.post("/api/arcade/settings", json={"auto_publish_score": 150})
    assert res.status_code == 400
    assert "between 0 and 100" in json.loads(res.data.decode())["error"]


def test_web_arcade_settings_post_saves_and_applies(web_client, monkeypatch):
    saved = {}
    monkeypatch.setattr(
        "online_providers.online_model_provider.save_credentials",
        lambda keys: saved.update(keys) or {"success": True},
    )
    res = web_client.post("/api/arcade/settings", json={"auto_publish_score": 70})
    assert res.status_code == 200
    data = json.loads(res.data.decode())
    assert data == {"success": True, "auto_publish_score": 70.0}
    assert saved == {"ARCADE_AUTO_PUBLISH_SCORE": "70.0"}
    # Live getter picks it up with no restart.
    assert ap.get_auto_publish_score() == 70.0


# --- players + achievements ---


def _player(arcade_client, initials):
    res = arcade_client.get(f"/api/players/{initials}")
    assert res.status_code == 200
    data = json.loads(res.data.decode())
    assert data["success"] is True
    return data["player"]


def test_submit_tracks_bests_and_first_unlocks(arcade_client):
    res = arcade_client.post("/api/games/demo-model_demo-breakout/scores", json={"initials": "abc", "score": 500})
    assert res.status_code == 200
    data = json.loads(res.data.decode())
    assert data["success"] is True
    assert data["pioneer"] is True
    assert data["personal_best"] is False
    assert data["player_url"] == "/player/ABC"
    ids = {u["id"] for u in data["new_unlocks"]}
    assert {"first-blood", "on-board", "podium", "champion", "pioneer"} <= ids

    # A better score sets a personal record and unlocks exactly that.
    res = arcade_client.post("/api/games/demo-model_demo-breakout/scores", json={"initials": "ABC", "score": 900})
    data = json.loads(res.data.decode())
    assert data["personal_best"] is True
    assert [u["id"] for u in data["new_unlocks"]] == ["personal-record"]

    p = _player(arcade_client, "abc")  # case-insensitive identity
    assert p["initials"] == "ABC"
    assert p["submits"] == 2
    assert p["best_score"] == 900
    assert p["boards"] == 1 and p["crowns"] == 1
    assert p["personal_bests"] == 1 and p["pioneered"] == 1


def test_cross_game_globetrotter_and_completionist(arcade_client, games_dir, source_html):
    slugs = [_publish(games_dir, source_html, test_id=f"demo_{n}")["slug"] for n in ("two", "three")]
    slugs.append("demo-model_demo-breakout")
    for slug in slugs:
        res = arcade_client.post(f"/api/games/{slug}/scores", json={"initials": "XYZ", "score": 100})
        assert res.status_code == 200
    p = _player(arcade_client, "XYZ")
    assert p["games_scored"] == 3
    assert p["total_games"] == 3
    unlocked = {a["id"] for a in p["achievements"] if a["unlocked"]}
    assert {"globetrotter", "completionist", "grinder"} - unlocked == {"grinder"}  # only 3 submits
    assert p["achievements"] and all("progress" in a and "goal" in a for a in p["achievements"])


def test_critic_and_standing_ovation(arcade_client, games_dir, source_html):
    slugs = [_publish(games_dir, source_html, test_id=f"demo_{n}")["slug"] for n in ("two", "three")]
    slugs.append("demo-model_demo-breakout")
    for i, slug in enumerate(slugs):
        res = arcade_client.post(f"/api/games/{slug}/rate", json={"stars": 5 if i == 0 else 4, "initials": "crt"})
        assert res.status_code == 200
    p = _player(arcade_client, "CRT")
    assert p["games_rated"] == 3
    assert p["gave_five_stars"] is True
    unlocked = {a["id"] for a in p["achievements"] if a["unlocked"]}
    assert {"critic", "standing-ovation"} <= unlocked


def test_anonymous_votes_do_not_credit_critic(arcade_client):
    res = arcade_client.post("/api/games/demo-model_demo-breakout/rate", json={"stars": 5})
    assert res.status_code == 200
    p = _player(arcade_client, "ZZZ")
    assert p["games_rated"] == 0
    assert p["submits"] == 0


def test_player_page_renders(arcade_client):
    arcade_client.post("/api/games/demo-model_demo-breakout/scores", json={"initials": "QWE", "score": 42})
    res = arcade_client.get("/player/qwe")
    assert res.status_code == 200
    assert b"QWE" in res.data
    assert b"Trophy Room" in res.data
    assert b"First Blood" in res.data

    res = arcade_client.get("/player/!!")
    assert res.status_code == 404


def test_achievements_evaluate_all_locked_and_all_unlocked():
    import arcade.achievements as ach

    bare = {
        "submits": 0,
        "boards": 0,
        "podiums": 0,
        "crowns": 0,
        "crown_games": 0,
        "games_scored": 0,
        "personal_bests": 0,
        "pioneered": 0,
        "night_owl": False,
        "high_roller": False,
        "best_score": 0,
        "games_rated": 0,
        "gave_five_stars": False,
        "total_games": 4,
    }
    assert ach.unlocked_ids(bare) == set()

    hero = dict(
        bare,
        submits=25,
        boards=5,
        podiums=4,
        crowns=4,
        crown_games=4,
        games_scored=4,
        personal_bests=5,
        pioneered=2,
        night_owl=True,
        high_roller=True,
        best_score=200000,
        games_rated=4,
        gave_five_stars=True,
    )
    assert ach.unlocked_ids(hero) == {a["id"] for a in ach.ACHIEVEMENTS}


# --- response-fallback publishing (pygame / desktop apps) ---

PYGAME_RESP = """Here is your game:

```python
import pygame

pygame.init()
screen = pygame.display.set_mode((640, 480))
print("space invaders")
```

Enjoy!"""

HTML_RESP = """Here it is:

```html
<!DOCTYPE html><html><body><canvas id="g"></canvas></body></html>
```"""


def _write_general_result(root, model, test_id, response, screenshot=None, score=0, extra=()):
    models_dir = root / "data" / "llm_benchmarks" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    rec = {"test_id": test_id, "response": response, "score": score}
    if screenshot is not None:
        rec["screenshot"] = screenshot
    models_dir.joinpath(f"general_{model}.json").write_text(
        json.dumps({"model": model, "results": [{"retro": {"tests": [rec, *extra]}}]})
    )


def test_find_model_response_prefers_longest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_general_result(
        tmp_path,
        "m",
        "t1",
        "x" * 10 + "much longer response here",
        extra=[{"test_id": "t1", "response": "short", "score": 1}],
    )
    rec = ap.find_model_response("m", "t1")
    assert rec is not None and len(rec["response"]) > 20
    assert ap.find_model_response("other-model", "t1") is None
    assert ap.find_model_response("m", "nope") is None


def test_publish_from_python_response(games_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_general_result(tmp_path, "py-model", "retro_space_invaders", PYGAME_RESP, score=61.0)
    res = ap.publish_game(model="py-model", test_id="retro_space_invaders")
    game_dir = games_dir / res["slug"]
    assert (game_dir / "game.py").exists()
    assert not (game_dir / "game.html").exists()
    code = (game_dir / "game.py").read_text()
    assert "import pygame" in code and "Enjoy!" not in code
    meta = json.loads((game_dir / "meta.json").read_text())
    assert meta["kind"] == "code" and meta["lang"] == "python"
    assert meta["benchmark_score"] == 61.0
    assert ap.is_published(res["slug"]) is True


def test_publish_from_response_saves_screenshot(games_dir, tmp_path, monkeypatch):
    import base64

    monkeypatch.chdir(tmp_path)
    png = base64.b64encode(b"\x89PNG-fake-bytes").decode()
    _write_general_result(tmp_path, "shot-model", "retro_game", PYGAME_RESP, screenshot=png)
    res = ap.publish_game(model="shot-model", test_id="retro_game")
    assert (games_dir / res["slug"] / "screenshot.png").exists()
    meta = json.loads((games_dir / res["slug"] / "meta.json").read_text())
    assert meta["has_screenshot"] is True


def test_publish_from_html_response_is_playable(games_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_general_result(tmp_path, "html-model", "web_game", HTML_RESP)
    res = ap.publish_game(model="html-model", test_id="web_game")
    game_dir = games_dir / res["slug"]
    assert (game_dir / "game.html").exists()
    assert not (game_dir / "game.py").exists()
    assert "<canvas" in (game_dir / "game.html").read_text()
    meta = json.loads((game_dir / "meta.json").read_text())
    assert meta["kind"] == "playable"


def test_publish_nothing_found_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="no saved game found"):
        ap.publish_game(model="ghost", test_id="nope")


def test_arcade_code_game_page(arcade_client):
    import arcade.app as arcade_app

    games_dir = arcade_app.GAMES_DIR
    slug = "pymod_retro"
    d = games_dir / slug
    d.mkdir()
    (d / "game.py").write_text("import pygame\n")
    (d / "meta.json").write_text(
        json.dumps({"slug": slug, "title": "Retro", "model": "pymod", "kind": "code", "lang": "python"})
    )
    res = arcade_client.get(f"/play/{slug}")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "import pygame" in html
    assert "Download" in html
    assert "game-frame" not in html
    # Code + screenshot file routes.
    res = arcade_client.get(f"/game/{slug}/game.py")
    assert res.status_code == 200
    res = arcade_client.get(f"/game/{slug}/screenshot.png")
    assert res.status_code == 404


def _code_slug(arcade_client):
    import arcade.app as arcade_app

    slug = "pymod_launch"
    d = arcade_app.GAMES_DIR / slug
    d.mkdir(exist_ok=True)
    (d / "game.py").write_text("import pygame\n")
    (d / "meta.json").write_text(
        json.dumps({"slug": slug, "title": "Launch", "model": "pymod", "kind": "code", "lang": "python"})
    )
    return slug


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_arcade_launch_code_game(arcade_client, monkeypatch):
    import arcade.app as arcade_app

    slug = _code_slug(arcade_client)
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode())
        return _FakeResp({"container_id": "abc123", "host_port": 6901})

    monkeypatch.setattr(arcade_app.urllib.request, "urlopen", fake_urlopen)
    res = arcade_client.post(f"/api/games/{slug}/launch")
    assert res.status_code == 200
    data = json.loads(res.data.decode())
    assert data["success"] is True
    assert data["launcher_url"].endswith("/ui/launcher/abc123?embed=1")
    assert data["launcher_url"].startswith("http://")
    assert ":5000/ui/launcher/" in data["launcher_url"]
    # The frozen code + lang are forwarded to the web sandbox.
    assert seen["body"]["code"] == "import pygame\n"
    assert seen["body"]["lang"] == "python"
    # Arcade launches are exclusive: one game session at a time.
    assert seen["body"]["exclusive"] is True
    # Arcade sessions get a 2 h lifetime: the default 10 min container
    # sleep would reap long play sessions mid-game.
    assert seen["body"]["timeout"] == 7200


def test_arcade_launch_url_same_origin_behind_https_proxy(arcade_client, monkeypatch):
    import arcade.app as arcade_app

    slug = _code_slug(arcade_client)

    def fake_urlopen(req, timeout=None):
        return _FakeResp({"container_id": "abc123", "host_port": 6901})

    monkeypatch.setattr(arcade_app.urllib.request, "urlopen", fake_urlopen)
    res = arcade_client.post(
        f"/api/games/{slug}/launch",
        base_url="https://games.sumemail.com",
        headers=[("X-Forwarded-Proto", "https")],
    )
    assert res.status_code == 200
    data = json.loads(res.data.decode())
    assert data["launcher_url"] == "https://games.sumemail.com/ui/launcher/abc123?embed=1"


def test_arcade_launch_url_direct_lan_uses_backend_port(arcade_client, monkeypatch):
    import arcade.app as arcade_app

    slug = _code_slug(arcade_client)

    def fake_urlopen(req, timeout=None):
        return _FakeResp({"container_id": "abc123", "host_port": 6901})

    monkeypatch.setattr(arcade_app.urllib.request, "urlopen", fake_urlopen)
    res = arcade_client.post(f"/api/games/{slug}/launch", base_url="http://192.168.2.43:5001")
    assert res.status_code == 200
    data = json.loads(res.data.decode())
    assert data["launcher_url"] == "http://192.168.2.43:5000/ui/launcher/abc123?embed=1"


def test_arcade_launch_missing_game(arcade_client):
    res = arcade_client.post("/api/games/nope-not-here/launch")
    assert res.status_code == 404


def test_arcade_stop_proxies_container_id(arcade_client, monkeypatch):
    import arcade.app as arcade_app

    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode())
        return _FakeResp({"stopped": True})

    monkeypatch.setattr(arcade_app.urllib.request, "urlopen", fake_urlopen)
    res = arcade_client.post("/api/games/any-slug/stop", json={"container_id": "abc123"})
    assert res.status_code == 200
    assert json.loads(res.data.decode()) == {"stopped": True}
    assert seen["url"].endswith("/api/sandbox/stop_serve")
    assert seen["body"] == {"container_id": "abc123"}


def test_arcade_stop_requires_container_id(arcade_client):
    res = arcade_client.post("/api/games/any-slug/stop", json={})
    assert res.status_code == 400


def test_arcade_inner_status_relays_to_web(arcade_client, monkeypatch):
    import arcade.app as arcade_app

    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode())
        return _FakeResp({"ok": True})

    monkeypatch.setattr(arcade_app.urllib.request, "urlopen", fake_urlopen)
    res = arcade_client.post(
        "/api/sandbox/ui_inner_status",
        json={"container_id": "abc123", "state": "inner-status", "detail": "Connecting..."},
    )
    assert res.status_code == 200
    assert json.loads(res.data.decode()) == {"ok": True}
    assert seen["url"].endswith("/api/sandbox/ui_inner_status")
    assert seen["body"]["state"] == "inner-status"


def test_arcade_launch_playable_has_no_code(arcade_client):
    res = arcade_client.post("/api/games/demo-model_demo-breakout/launch")
    assert res.status_code == 404


def test_arcade_launch_sandbox_error(arcade_client, monkeypatch):
    import arcade.app as arcade_app

    slug = _code_slug(arcade_client)

    def fake_urlopen(req, timeout=None):
        return _FakeResp({"error": "docker unavailable"})

    monkeypatch.setattr(arcade_app.urllib.request, "urlopen", fake_urlopen)
    res = arcade_client.post(f"/api/games/{slug}/launch")
    assert res.status_code == 502


def test_arcade_launch_unreachable(arcade_client, monkeypatch):
    import arcade.app as arcade_app

    slug = _code_slug(arcade_client)

    def fake_urlopen(req, timeout=None):
        raise ConnectionError("refused")

    monkeypatch.setattr(arcade_app.urllib.request, "urlopen", fake_urlopen)
    res = arcade_client.post(f"/api/games/{slug}/launch")
    assert res.status_code == 502


def test_arcade_code_page_has_live_button(arcade_client):
    slug = _code_slug(arcade_client)
    res = arcade_client.get(f"/play/{slug}")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "btn-play-live" in html
    assert "live-frame" in html
    # Regression: arcade.js reads window.ARCADE_SLUG, so the page must set it
    # on window (a top-level `const` never lands on window, leaving every
    # /api/games/<slug>/... call pointed at "undefined").
    assert f'window.ARCADE_SLUG = "{slug}"' in html
    assert ">▶ Play<" in html
    # Under-screen info is minimized behind a click-here expander.
    assert 'class="under-screen"' in html
    assert "Game code, download" in html


def test_arcade_js_no_commented_out_widget():
    """Regression: `const widget = ...` was once merged into a `//` comment,
    leaving `widget` undefined and killing the entire play-page script
    (launch, scores, ratings — dead on mobile and desktop)."""
    with open("arcade/static/arcade.js", encoding="utf-8") as f:
        src = f.read()
    assert 'const widget = $("rating-widget");' in src
    for line in src.splitlines():
        stripped = line.strip()
        if "const widget" in stripped:
            assert not stripped.startswith("//"), f"widget declaration commented out: {line!r}"


def test_publish_backfills_prompt_and_date_from_catalog_and_file(games_dir, tmp_path, monkeypatch):
    """Minimal run records (no prompt/date, like the real deepseek general
    file) still publish with a prompt (static test catalog) and a run date
    (source result-file mtime) instead of blank scorecard fields."""
    import os
    import time

    src_file = tmp_path / "general_fake.json"
    src_file.write_text("{}")
    mtime = time.mktime(time.strptime("2026-09-10", "%Y-%m-%d"))
    os.utime(src_file, (mtime, mtime))
    monkeypatch.setattr(
        ap,
        "find_model_response",
        lambda model, test_id: {
            "response": "```python\nimport pygame\npygame.init()\n```",
            "screenshot": None,
            "score": 92.0,
            "model": model,
            "prompt": "",
            "run_date": "",
            "source_file": str(src_file),
        },
    )
    res = ap.publish_game(model="demo-model", test_id="retro_space_invaders")
    meta = json.loads((games_dir / res["slug"] / "meta.json").read_text())
    assert meta["kind"] == "code"
    assert len(meta["prompt"]) > 100  # from benchmark_tests.json catalog
    assert meta["run_date"] == "2026-09-10"
    assert meta["benchmark_date"] == "2026-09-10"
    assert meta["benchmark_score"] == 92.0


def test_response_lang_uses_model_fence_tag():
    """The extraction language comes from the model's own fenced block tag —
    never a hardcoded per-kind default."""
    assert ap._response_lang("```html\n<canvas></canvas>\n```") == "html"
    assert ap._response_lang("```python\nimport pygame\n```") == "python"
    assert ap._response_lang("```js\nconsole.log(1)\n```") == "javascript"
    assert ap._response_lang("no fences here, just code") == "python"  # documented last resort


def test_publish_meta_lang_follows_response(games_dir, monkeypatch):
    """A JS-fenced response publishes with lang javascript, not python."""
    monkeypatch.setattr(
        ap,
        "find_model_response",
        lambda model, test_id: {
            "response": "```js\nconsole.log('hi')\n```",
            "screenshot": None,
            "score": 70.0,
            "model": model,
            "prompt": "p",
            "run_date": "2026-09-10",
            "source_file": None,
        },
    )
    res = ap.publish_game(model="demo-model", test_id="demo_js")
    meta = json.loads((games_dir / res["slug"] / "meta.json").read_text())
    assert meta["lang"] == "javascript"


def test_publish_max_score_from_record(games_dir, monkeypatch):
    """max_score falls back to the run record's own scale; None when unknown."""
    monkeypatch.setattr(
        ap,
        "find_model_response",
        lambda model, test_id: {
            "response": "```python\nprint('hi')\n```",
            "screenshot": None,
            "score": 70.0,
            "max_score": 200.0,
            "model": model,
            "prompt": "p",
            "run_date": "2026-09-10",
            "source_file": None,
        },
    )
    res = ap.publish_game(model="demo-model", test_id="demo_max")
    meta = json.loads((games_dir / res["slug"] / "meta.json").read_text())
    assert meta["max_score"] == 200.0

    monkeypatch.setattr(
        ap,
        "find_model_response",
        lambda model, test_id: {
            "response": "```python\nprint('hi')\n```",
            "screenshot": None,
            "score": 70.0,
            "model": model,
            "prompt": "p",
            "run_date": "2026-09-10",
            "source_file": None,
        },
    )
    res = ap.publish_game(model="demo-model", test_id="demo_nomax")
    meta = json.loads((games_dir / res["slug"] / "meta.json").read_text())
    assert meta["max_score"] is None


def test_arcade_html_responses_are_no_store():
    """Play/index pages must not be cached (stale-phone-cache insurance)."""
    import arcade.app as arcade_app

    client = arcade_app.app.test_client()
    for path in ("/", "/health"):
        resp = client.get(path)
        if path == "/" and resp.status_code == 200:
            assert resp.headers.get("Cache-Control") == "no-store"


JS_RESP = """Here is your game:

```javascript
const canvas = document.getElementById('g');
console.log('breakout ready');
```
"""


def test_publish_from_js_response_is_playable_in_hero(games_dir, tmp_path, monkeypatch):
    """JavaScript responses publish as playable game.html (hero iframe)."""
    monkeypatch.chdir(tmp_path)
    _write_general_result(tmp_path, "js-model", "web_breakout", JS_RESP)
    res = ap.publish_game(model="js-model", test_id="web_breakout")
    game_dir = games_dir / res["slug"]
    assert (game_dir / "game.html").exists()
    assert not (game_dir / "game.py").exists()
    html = (game_dir / "game.html").read_text()
    assert "breakout ready" in html
    meta = json.loads((game_dir / "meta.json").read_text())
    assert meta["kind"] == "playable"
    assert meta["lang"] == "javascript"


def test_launch_lang_derivation_and_omit_when_unstated(arcade_client, monkeypatch):
    """Sandbox lang comes from stored meta; unstated omits the key."""
    import arcade.app as arcade_app

    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["body"] = json.loads(req.data.decode())
        return _FakeResp({"container_id": "xyz1", "host_port": 6901})

    monkeypatch.setattr(arcade_app.urllib.request, "urlopen", fake_urlopen)

    d = arcade_app.GAMES_DIR / "jslaunch"
    d.mkdir(exist_ok=True)
    (d / "game.py").write_text("console.log(1);\n")
    (d / "meta.json").write_text(json.dumps({"slug": "jslaunch", "kind": "code", "lang": "javascript"}))
    res = arcade_client.post("/api/games/jslaunch/launch")
    assert res.status_code == 200
    assert seen["body"]["lang"] == "javascript"

    d2 = arcade_app.GAMES_DIR / "nolang"
    d2.mkdir(exist_ok=True)
    (d2 / "game.py").write_text("print(1)\n")
    (d2 / "meta.json").write_text(json.dumps({"slug": "nolang", "kind": "code"}))
    res = arcade_client.post("/api/games/nolang/launch")
    assert res.status_code == 200
    assert "lang" not in seen["body"]


def test_run_hint_generic_for_unknown_languages(arcade_client):
    """Non-python code games get a generic sandbox-first hint."""
    import arcade.app as arcade_app

    assert "pygame" in arcade_app._run_hint("python")["long"]
    generic = arcade_app._run_hint("rust")
    assert "pygame" not in generic["long"]
    assert "browser" in generic["short"]
    unstated = arcade_app._run_hint("")
    assert "pygame" not in unstated["long"]


def test_launcher_keybar_markup_and_js():
    """VNC launcher has arrow/space/enter/esc bar posting xdotool keys."""
    from pathlib import Path

    html = Path("web/templates/ui_launcher.html").read_text()
    assert 'id="key-bar"' in html
    for label, xkey in [("▲", "Up"), ("▼", "Down"), ("◀", "Left"), ("▶", "Right")]:
        assert label in html
        assert f'data-xkey="{xkey}"' in html
    for xkey in ("space", "Return", "Escape"):
        assert f'data-xkey="{xkey}"' in html
    assert "xdotool ${verb} ${b.dataset.xkey}" in html
    assert "xdotool ${verb} ${m.xkey}" in html
    assert "pointerdown" in html
    assert "DISPLAY=:99" in html


def test_launcher_embed_keeps_only_keybar():
    """Embed mode trims toolbar chrome (brand/restart/shot/close) but the
    on-screen key bar — a phone's only input — must stay visible."""
    from pathlib import Path

    html = Path("web/templates/ui_launcher.html").read_text()
    for sel in ("#launcher-toolbar .brand", "#btn-restart", "#btn-shot", "#btn-close"):
        assert f"body.embed {sel}" in html
    assert "body.embed #key-bar" not in html


def test_sandbox_image_includes_xdotool():
    """Sandbox image must ship xdotool (key-bar key injection)."""
    from pathlib import Path

    dockerfile = Path("Dockerfile.sandbox").read_text()
    assert "xdotool" in dockerfile


def test_code_game_restores_line_height():
    """Regression: .screen zeroes line-height for game frames; the code-game
    branch must restore it or under-screen text paints over itself."""
    from pathlib import Path

    css = Path("arcade/static/arcade.css").read_text()
    assert ".screen .code-game" in css
    assert "line-height" in css.split(".screen .code-game", 1)[1].split("}", 1)[0]


def test_play_layout_grid_blowout_guard():
    """Regression: on mobile the collapsed 1fr .play-layout track blew out
    to 929px (grid items default min-width:auto), pushing the live game
    frame off-screen so the phone showed a dark sliver. Grid children must
    be allowed to shrink to the track."""
    from pathlib import Path

    css = Path("arcade/static/arcade.css").read_text()
    assert ".play-layout > *" in css
    assert "min-width: 0" in css.split(".play-layout > *", 1)[1].split("}", 1)[0]


def test_launcher_frame_fits_canvas():
    """The vnc frame must be exactly the 4:3 desktop — no fixed tall frame
    (black gap), no JS measure-and-shrink loop (feedback collapse), no
    min-height floor (letterbox). CSS aspect-ratio owns the height; embed
    launcher hides its own toolbar so noVNC fills the frame."""
    import re
    from pathlib import Path

    css = Path("arcade/static/arcade.css").read_text()
    # Every #live-frame rule block sizes by aspect-ratio (never a fixed
    # viewport height — that caused the black gap / measure-shrink loop).
    # Generic .screen iframe dvh heights still exist for playable games.
    blocks = re.findall(r"^#live-frame\s*\{([^}]*)\}", css, re.MULTILINE)
    assert blocks, "no #live-frame rules found"
    for b in blocks:
        assert "aspect-ratio" in b
        assert "dvh" not in b
        assert re.search(r"(?<!min-)height\s*:", b) is None or "auto" in b
    html = Path("web/templates/ui_launcher.html").read_text()
    assert "fitFrameToCanvas" not in html
    assert "min-height: 240px" not in html
    assert "#launcher-toolbar" in html


def test_launcher_stream_telemetry_markup():
    """Launcher reports inner-stream state to the arcade parent so a phone
    can say what's failing (Live! status alone proved nothing)."""
    from pathlib import Path

    html = Path("web/templates/ui_launcher.html").read_text()
    assert "arcade-vnc" in html
    assert "watchInnerStream" in html
    assert "noVNC_status" in html
    assert "postMessage" in html


def test_launcher_game_audio_sidechannel():
    """Launcher streams game sound via <audio> + /serve/audio/<id> (noVNC is
    video-only): toggle button, embed muted-autoplay + self-unmute (taps
    inside the nested VNC iframe never reach the launcher document, so a
    gesture-gated start would never fire), arcade overlay control channel,
    and states on the report() telemetry channel."""
    from pathlib import Path

    html = Path("web/templates/ui_launcher.html").read_text()
    assert 'id="game-audio"' in html
    assert "/serve/audio/${CONTAINER_ID}" in html
    assert 'id="btn-sound"' in html
    assert "arcade-audio" in html
    assert "audio-playing" in html
    assert "audio-blocked" in html
    # Embed autostart: muted play (always allowed) then unmute — only
    # play() is autoplay-gated, mute is a volume control.
    assert "audioEl.muted = true" in html
    assert "audioEl.muted = false" in html
    # Sync guards: redundant on() must not reset src (each reset reboots
    # the encoder + forces browser rebuffer = minute-long start delays),
    # and a stale in-flight play() must not unmute after a mute (gen guard).
    assert "audioGen" in html
    assert "if (!audioEl.src) audioEl.src = audioURL" in html
    assert "if (gen !== audioGen) return" in html


def test_arcade_js_live_sound_control():
    """Arcade play page drives the launcher's arcade-audio channel: unmute
    on live-frame load plus a Sound toggle (the launcher toolbar, with its
    own Sound button, is hidden in embeds)."""
    from pathlib import Path

    js = Path("arcade/static/arcade.js").read_text()
    assert "postLiveAudio" in js
    assert "arcade-audio" in js
    assert "btn-sound-live" in js


def test_arcade_play_page_has_sound_button():
    """Code-game branch offers a Sound toggle next to Play (embed has no
    launcher toolbar of its own)."""
    from pathlib import Path

    html = Path("arcade/templates/play.html").read_text()
    assert 'id="btn-sound-live"' in html


def test_sandbox_image_includes_audio_chain():
    """Sandbox image must ship the virtual-audio toolchain: parec/pactl
    (pulseaudio-utils), ALSA->Pulse redirect (libasound2-plugins), and the
    MP3 encoder/streamer (ffmpeg)."""
    from pathlib import Path

    dockerfile = Path("Dockerfile.sandbox").read_text()
    for pkg in ("pulseaudio-utils", "libasound2-plugins", "ffmpeg"):
        assert pkg in dockerfile


def test_arcade_js_stream_status_handler():
    """Arcade play page listens for the launcher's stream telemetry and
    surfaces it on the live-status line."""
    from pathlib import Path

    js = Path("arcade/static/arcade.js").read_text()
    assert "arcade-vnc" in js
    assert "Stream:" in js


def test_arcade_js_gated_auto_fullscreen():
    """Auto-fullscreen waits for the noVNC connected signal instead of a
    blind 3 s timer that reshapes the stream iframe mid-handshake."""
    from pathlib import Path

    js = Path("arcade/static/arcade.js").read_text()
    assert "setTimeout(enterFull, 3000)" not in js
    assert "__arcadeFullDone" in js
    assert "inner-status" in js
    assert "45000" in js


def test_arcade_sync_score_success(arcade_client, monkeypatch):
    """Game wrote {initials, score} to /tmp/alpaca_score.json — stored
    through the same ledger as manual submit."""
    import arcade.app as arcade_app

    slug = _code_slug(arcade_client)

    def fake_urlopen(req, timeout=None):
        if "/api/sandbox/ui/exec" in req.full_url:
            return _FakeResp({"output": '{"initials": "XYZ", "score": 350}'})
        return _FakeResp({})

    monkeypatch.setattr(arcade_app.urllib.request, "urlopen", fake_urlopen)
    res = arcade_client.post(
        f"/api/games/{slug}/sync_score",
        json={"container_id": "abc123"},
    )
    assert res.status_code == 200
    data = json.loads(res.data.decode())
    assert data["success"] is True
    assert data["initials"] == "XYZ"
    assert data["score"] == 350
    assert data["auto"] is True
    assert data["rank"] == 1
    assert data["made_board"] is True


def test_arcade_sync_score_no_score_yet(arcade_client, monkeypatch):
    """No score file yet — polling marker, not an error."""
    import arcade.app as arcade_app

    slug = _code_slug(arcade_client)

    def fake_urlopen(req, timeout=None):
        if "/api/sandbox/ui/exec" in req.full_url:
            return _FakeResp({"output": ""})
        return _FakeResp({})

    monkeypatch.setattr(arcade_app.urllib.request, "urlopen", fake_urlopen)
    res = arcade_client.post(
        f"/api/games/{slug}/sync_score",
        json={"container_id": "abc123"},
    )
    assert res.status_code == 200
    data = json.loads(res.data.decode())
    assert data["success"] is True
    assert data["status"] == "no score yet"


def test_arcade_sync_score_missing_container_id(arcade_client):
    res = arcade_client.post(
        "/api/games/demo-model_demo-breakout/sync_score",
        json={},
    )
    assert res.status_code == 400
    assert "container_id" in json.loads(res.data.decode())["error"]


def test_arcade_sync_score_missing_game(arcade_client):
    res = arcade_client.post(
        "/api/games/nope-not-here/sync_score",
        json={"container_id": "abc"},
    )
    assert res.status_code == 404


def test_arcade_sync_score_web_backend_error(arcade_client, monkeypatch):
    import arcade.app as arcade_app

    slug = _code_slug(arcade_client)

    def fake_urlopen(req, timeout=None):
        if "/api/sandbox/ui/exec" in req.full_url:
            return _FakeResp({"error": "container not found"})
        return _FakeResp({})

    monkeypatch.setattr(arcade_app.urllib.request, "urlopen", fake_urlopen)
    res = arcade_client.post(
        f"/api/games/{slug}/sync_score",
        json={"container_id": "abc123"},
    )
    assert res.status_code == 502
    assert "error" in json.loads(res.data.decode())


def test_arcade_sync_score_invalid_json(arcade_client, monkeypatch):
    import arcade.app as arcade_app

    slug = _code_slug(arcade_client)

    def fake_urlopen(req, timeout=None):
        if "/api/sandbox/ui/exec" in req.full_url:
            return _FakeResp({"output": "not json at all"})
        return _FakeResp({})

    monkeypatch.setattr(arcade_app.urllib.request, "urlopen", fake_urlopen)
    res = arcade_client.post(
        f"/api/games/{slug}/sync_score",
        json={"container_id": "abc123"},
    )
    # Invalid JSON in all candidate files → keep polling, not an error.
    assert res.status_code == 200
    data = json.loads(res.data.decode())
    assert data["success"] is True
    assert data["status"] == "no score yet"


def test_arcade_sync_score_bad_payload(arcade_client, monkeypatch):
    import arcade.app as arcade_app

    slug = _code_slug(arcade_client)

    def fake_urlopen(req, timeout=None):
        if "/api/sandbox/ui/exec" in req.full_url:
            return _FakeResp({"output": '{"name": "no score key"}'})
        return _FakeResp({})

    monkeypatch.setattr(arcade_app.urllib.request, "urlopen", fake_urlopen)
    res = arcade_client.post(
        f"/api/games/{slug}/sync_score",
        json={"container_id": "abc123"},
    )
    assert res.status_code == 200
    data = json.loads(res.data.decode())
    assert data["success"] is True
    assert data["status"] == "no score yet"


def test_arcade_sync_score_high_scores_fallback(arcade_client, monkeypatch):
    """Game only wrote /tmp/high_scores.json (array of {name, score}).

    Some games don't write /tmp/alpaca_score.json but use
    high_scores.json. We should fall back and extract the latest.
    """
    import arcade.app as arcade_app

    slug = _code_slug(arcade_client)
    call_count = {"n": 0}

    def fake_urlopen(req, timeout=None):
        call_count["n"] += 1
        if "/api/sandbox/ui/exec" in req.full_url:
            # First call (alpaca_score.json) → empty, second call (high_scores.json) → data
            if call_count["n"] == 1:
                return _FakeResp({"output": ""})
            return _FakeResp({"output": '[{"name": "FOO", "score": 42}]'})
        return _FakeResp({})

    monkeypatch.setattr(arcade_app.urllib.request, "urlopen", fake_urlopen)
    res = arcade_client.post(
        f"/api/games/{slug}/sync_score",
        json={"container_id": "abc123"},
    )
    assert res.status_code == 200
    data = json.loads(res.data.decode())
    assert data["success"] is True
    assert data["initials"] == "FOO"
    assert data["score"] == 42
    assert data["auto"] is True
    assert call_count["n"] == 2
