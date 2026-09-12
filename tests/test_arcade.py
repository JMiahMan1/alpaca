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
    res = arcade_client.post(
        "/api/games/demo-model_demo-breakout/scores", json={"initials": "X", "score": "abc"}
    )
    assert res.status_code == 400
    res = arcade_client.post(
        "/api/games/demo-model_demo-breakout/scores", json={"initials": "X", "score": -5}
    )
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
    res = arcade_client.delete(
        "/api/admin/games/demo-model_demo-breakout", headers={"X-Arcade-Token": "wrong"}
    )
    assert res.status_code == 401
    res = arcade_client.delete(
        "/api/admin/games/demo-model_demo-breakout", headers={"X-Arcade-Token": "test-token"}
    )
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
