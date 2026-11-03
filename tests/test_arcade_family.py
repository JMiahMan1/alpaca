"""Family Games: built-in sync, lifecycle, scoreboard scoping, mobile benchmarks.

Everything runs against tmp dirs: the source tree, the games dir, the curated
list and the lifecycle file are all monkeypatched.
"""

import json

import pytest

import web.arcade_publish as ap
from llm_benchmark_suite import LLMModelBenchmark

KIT_CSS = ":root{--fk-accent:#000}"
KIT_JS = "window.FamilyKit={create(){}};"
GAME = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<!-- @family-kit -->
<style>html,body{overflow:hidden;touch-action:none}#hud{top:env(safe-area-inset-top)}</style>
</head><body><div id="fk-stage"></div><script>
const c=document.createElement('canvas');function size(){c.width=innerWidth*devicePixelRatio}
addEventListener('resize',size);size();localStorage.getItem('b');
addEventListener('pointerdown',e=>e.preventDefault());
document.addEventListener('visibilitychange',()=>{});requestAnimationFrame(()=>{});
function over(score){parent.postMessage({source:'arcade-game',type:'score',score,run_id:'r'},'*')}
</script></body></html>"""
# A desktop-shaped page: keyboard only, fixed 800x600 canvas, no bridge.
DESKTOP_GAME = """<!doctype html><html><head><!-- @family-kit --></head><body>
<canvas width="800" height="600"></canvas><script>addEventListener('keydown',()=>{})</script></body></html>"""


@pytest.fixture
def env(tmp_path, monkeypatch):
    import arcade.app as arcade_app
    from arcade import curation, family

    games = tmp_path / "games"
    src = tmp_path / "family_games"
    (src / "_kit").mkdir(parents=True)
    (src / "_kit" / "kit.css").write_text(KIT_CSS)
    (src / "_kit" / "kit.js").write_text(KIT_JS)
    for name, title in (("cloud-hop", "Cloud Hop"), ("petal-swap", "Petal Swap")):
        (src / name).mkdir()
        (src / name / "game.html").write_text(GAME)
        (src / name / "meta.json").write_text(
            json.dumps({"title": title, "emoji": "☁️", "art": {"from": "#111", "to": "#222"}, "version": 1})
        )
    monkeypatch.setattr(family, "SOURCE_DIR", src)
    monkeypatch.setattr(family, "KIT_DIR", src / "_kit")
    monkeypatch.setattr(family, "GAMES_DIR", games)
    monkeypatch.setattr(family, "STATE_FILE", tmp_path / "family.json")
    monkeypatch.setattr(arcade_app, "GAMES_DIR", games)
    monkeypatch.setattr(arcade_app, "ADMIN_TOKEN", "tok")
    monkeypatch.setattr(ap, "GAMES_DIR", games)
    monkeypatch.setattr(curation, "CURATED_FILE", tmp_path / "curated.json")
    arcade_app.app.config["TESTING"] = True
    with arcade_app.app.test_client() as client:
        yield {"family": family, "app": arcade_app, "client": client, "games": games, "src": src}


def _set(client, slug, state, token="tok"):
    return client.post(f"/api/admin/family/{slug}/state", json={"state": state}, headers={"X-Arcade-Token": token})


# --- sync ---------------------------------------------------------------


def test_sync_installs_games_with_kit_inlined_and_starts_them_in_testing(env):
    report = env["family"].sync()
    assert sorted(report["installed"]) == ["family-cloud-hop", "family-petal-swap"]
    html = (env["games"] / "family-cloud-hop" / "game.html").read_text()
    assert "<!-- @family-kit -->" not in html
    assert KIT_JS in html and KIT_CSS in html
    meta = json.loads((env["games"] / "family-cloud-hop" / "meta.json").read_text())
    assert meta["collection"] == "family" and meta["score_bridge"] is True and meta["mobile_native"] is True
    assert env["family"].state_of("family-cloud-hop") == "testing"


def test_resync_keeps_scores_ratings_and_plays_and_updates_the_game(env):
    fam = env["family"]
    fam.sync()
    d = env["games"] / "family-cloud-hop"
    (d / "scores.json").write_text(json.dumps({"scores": [{"initials": "JER", "score": 9}]}))
    (d / "ratings.json").write_text(json.dumps({"votes": [{"stars": 5}]}))
    meta = json.loads((d / "meta.json").read_text())
    meta["plays"] = 7
    (d / "meta.json").write_text(json.dumps(meta))
    fam.set_state("family-cloud-hop", "published")

    assert "family-cloud-hop" in fam.sync()["unchanged"]
    (env["src"] / "cloud-hop" / "game.html").write_text(GAME.replace("fk-stage", "fk-stage2"))
    assert "family-cloud-hop" in fam.sync()["updated"]

    assert "fk-stage2" in (d / "game.html").read_text()
    assert json.loads((d / "scores.json").read_text())["scores"][0]["score"] == 9
    assert json.loads((d / "ratings.json").read_text())["votes"] == [{"stars": 5}]
    assert json.loads((d / "meta.json").read_text())["plays"] == 7
    assert fam.state_of("family-cloud-hop") == "published"


def test_a_built_in_game_that_is_not_mobile_compatible_is_not_installed(env):
    (env["src"] / "cloud-hop" / "game.html").write_text(DESKTOP_GAME)
    report = env["family"].sync()
    assert "not mobile compatible" in report["errors"]["family-cloud-hop"]
    assert "touch/pointer controls" in report["errors"]["family-cloud-hop"]
    assert not (env["games"] / "family-cloud-hop").exists()
    assert "family-petal-swap" in report["installed"]


def test_a_game_that_stops_being_mobile_compatible_cannot_be_published(env):
    env["family"].sync()
    (env["games"] / "family-cloud-hop" / "game.html").write_text(DESKTOP_GAME)
    resp = _set(env["client"], "family-cloud-hop", "published")
    assert resp.status_code == 400
    assert "not mobile compatible" in resp.get_json()["error"]
    assert env["family"].state_of("family-cloud-hop") == "testing"


def test_a_game_without_the_kit_marker_is_reported_not_installed(env):
    (env["src"] / "cloud-hop" / "game.html").write_text("<html></html>")
    report = env["family"].sync()
    assert "family-cloud-hop" in report["errors"]
    assert not (env["games"] / "family-cloud-hop").exists()


# --- lifecycle and visibility -------------------------------------------


def test_family_list_reports_every_state_and_benchmark_shelf_excludes_family_games(env):
    env["family"].sync()
    client = env["client"]
    _set(client, "family-petal-swap", "unpublished")
    games = {g["slug"]: g for g in client.get("/api/family/games").get_json()["games"]}
    assert games["family-cloud-hop"]["family_state"] == "testing"
    assert games["family-petal-swap"]["family_state"] == "unpublished"
    assert games["family-cloud-hop"]["source"] == "built-in"
    assert games["family-cloud-hop"]["mobile_compatible"] is True
    assert games["family-cloud-hop"]["art"] == {"from": "#111", "to": "#222"}
    assert client.get("/api/games").get_json()["games"] == []


def test_testing_and_published_games_play_but_unpublished_do_not(env):
    env["family"].sync()
    client = env["client"]
    assert client.get("/play/family-cloud-hop?embed=1").status_code == 200
    _set(client, "family-cloud-hop", "unpublished")
    assert client.get("/play/family-cloud-hop?embed=1").status_code == 404
    assert client.get("/api/games/family-cloud-hop").status_code == 404
    _set(client, "family-cloud-hop", "published")
    page = client.get("/play/family-cloud-hop?embed=1").get_data(as_text=True)
    assert "window.ARCADE_BRIDGE = true" in page


def test_state_changes_need_the_admin_token(env):
    env["family"].sync()
    client = env["client"]
    assert _set(client, "family-cloud-hop", "published", token="nope").status_code == 401
    assert _set(client, "family-cloud-hop", "bogus").status_code == 400
    assert _set(client, "family-nothing", "published").status_code == 404
    ok = _set(client, "family-cloud-hop", "published").get_json()
    assert ok["state"] == "published" and ok["previous"] == "testing"


def test_state_changes_refuse_clearly_when_the_token_is_not_configured(env, monkeypatch):
    env["family"].sync()
    monkeypatch.setattr(env["app"], "ADMIN_TOKEN", "")
    resp = _set(env["client"], "family-cloud-hop", "published")
    assert resp.status_code == 503
    assert "ARCADE_ADMIN_TOKEN" in resp.get_json()["error"]


def test_native_mobile_games_are_served_without_the_fit_shim(env):
    env["family"].sync()
    html = env["client"].get("/game/family-cloud-hop/index.html").get_data(as_text=True)
    assert "arcade-mobile-fit" not in html


# --- scores and the family scoreboard ------------------------------------


def test_family_scoreboard_counts_only_published_family_games(env):
    env["family"].sync()
    client = env["client"]
    for slug in ("family-cloud-hop", "family-petal-swap"):
        r = client.post(
            f"/api/games/{slug}/scores",
            json={"initials": "JER", "score": 120, "player_id": "jeremiah", "capture_id": f"{slug}:run1"},
        )
        assert r.get_json()["success"]
    _set(client, "family-cloud-hop", "published")

    fam = client.get("/api/highscores?collection=family").get_json()
    assert [g["slug"] for g in fam["games"]] == ["family-cloud-hop"]
    assert fam["players"][0]["games_scored"] == 1
    # The benchmark arcade board never sees family games.
    assert client.get("/api/highscores").get_json()["games"] == []
    player = client.get("/api/players/JER?player_id=jeremiah&collection=family").get_json()["player"]
    assert player["total_games"] == 1 and player["best_score"] == 120


def test_a_bridged_run_is_saved_once(env):
    env["family"].sync()
    client = env["client"]
    body = {"initials": "JER", "score": 50, "player_id": "jeremiah", "capture_id": "family-cloud-hop:abc"}
    assert client.post("/api/games/family-cloud-hop/scores", json=body).get_json()["duplicate"] is False
    assert client.post("/api/games/family-cloud-hop/scores", json=body).get_json()["duplicate"] is True


# --- mobile benchmark games ---------------------------------------------


def test_a_published_mobile_benchmark_game_joins_the_family_collection_in_testing(env, tmp_path):
    src = tmp_path / "flyer.html"
    src.write_text(GAME.replace("<!-- @family-kit -->", ""))
    out = ap.publish_game(
        model="qwen3:35b", test_id="mobile_one_tap_flyer", source_file=src, category="mobilegames", benchmark_score=90
    )
    slug = out["slug"]
    meta = json.loads((env["games"] / slug / "meta.json").read_text())
    assert meta["collection"] == "family" and meta["score_bridge"] is True
    fam = env["family"]
    assert fam.is_family(slug) and fam.source(slug) == "benchmark"
    assert fam.state_of(slug) == "testing"
    games = {g["slug"]: g for g in env["client"].get("/api/family/games").get_json()["games"]}
    assert games[slug]["source"] == "benchmark"
    # Curated for the benchmark shelf by the publish, but still kept off it.
    assert env["client"].get("/api/games").get_json()["games"] == []


def test_a_mobile_benchmark_game_that_fails_the_contract_stays_out_of_family_games(env, tmp_path):
    src = tmp_path / "desktop.html"
    src.write_text(DESKTOP_GAME.replace("<!-- @family-kit -->", ""))
    out = ap.publish_game(model="m", test_id="mobile_bubble_pop", source_file=src, category="mobilegames")
    meta = json.loads((env["games"] / out["slug"] / "meta.json").read_text())
    assert "collection" not in meta
    assert "touch/pointer controls" in meta["mobile_contract_failures"]
    assert not env["family"].is_family(out["slug"])
    assert env["client"].get("/api/family/games").get_json()["games"] == []


def test_an_ordinary_benchmark_game_is_not_a_family_game(env, tmp_path):
    src = tmp_path / "g.html"
    src.write_text("<html><body>x</body></html>")
    out = ap.publish_game(model="m", test_id="game_snake_canvas", source_file=src, category="gamedev_alt")
    assert not env["family"].is_family(out["slug"])


GOOD_MOBILE = """```html
<!DOCTYPE html><html><head><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>html,body{overflow:hidden;touch-action:none;overscroll-behavior:none}
.hud{top:env(safe-area-inset-top)}</style></head><body><canvas id="c"></canvas><script>
const c=document.getElementById('c');function size(){c.width=innerWidth*devicePixelRatio;}
addEventListener('resize',size);size();let best=+localStorage.getItem('best')||0;
c.addEventListener('pointerdown',e=>{e.preventDefault();});
document.addEventListener('visibilitychange',()=>{});
parent.postMessage({source:'arcade-game',type:'ready',bridge:1},'*');
function over(score){parent.postMessage({source:'arcade-game',type:'score',score,run_id:Date.now()+''},'*');}
requestAnimationFrame(function loop(){requestAnimationFrame(loop)});
// bubble pop burst spiky combo
</script></body></html>
```"""


def test_mobile_contract_report_names_each_requirement():
    report = LLMModelBenchmark.mobile_contract_report(GOOD_MOBILE.lower())
    assert all(report.values()), report
    fixed = GOOD_MOBILE.replace('<canvas id="c">', '<canvas id="c" width="800" height="600">').lower()
    assert LLMModelBenchmark.mobile_contract_report(fixed)["no_fixed_canvas"] is False
    no_bridge = GOOD_MOBILE.replace("arcade-game", "game").lower()
    assert LLMModelBenchmark.mobile_contract_report(no_bridge)["score_bridge"] is False
    keyboard_only = GOOD_MOBILE.replace("pointerdown", "keydown").lower()
    assert LLMModelBenchmark.mobile_contract_report(keyboard_only)["touch_input"] is False


def test_mobile_benchmark_grader_requires_the_contract_and_the_game():
    bench = LLMModelBenchmark.__new__(LLMModelBenchmark)
    assert bench._verify_functional_response({"id": "mobile_bubble_pop"}, GOOD_MOBILE)
    assert not bench._verify_functional_response(
        {"id": "mobile_bubble_pop"}, GOOD_MOBILE.replace("visibilitychange", "focus")
    )
    assert not bench._verify_functional_response({"id": "mobile_one_tap_flyer"}, GOOD_MOBILE)


def test_mobile_benchmarks_are_registered_ui_html_tests():
    with open("benchmark_tests.json", encoding="utf-8") as f:
        tests = json.load(f)["mobilegames"]
    assert {t["id"] for t in tests} == {
        "mobile_one_tap_flyer",
        "mobile_swipe_match3",
        "mobile_memory_pairs",
        "mobile_drag_catcher",
        "mobile_bubble_pop",
        "mobile_block_blast",
    }
    for t in tests:
        assert t["type"] == "ui" and t["lang"] == "html"
        assert "arcade-game" in t["prompt"] and "visibilitychange" in t["prompt"]
    assert "mobilegames" in ap.GAME_CATEGORIES
    assert "mobilegames" in LLMModelBenchmark.CODE_CATEGORIES


def test_a_member_voting_again_changes_their_vote_instead_of_adding_one(env):
    env["family"].sync()
    client = env["client"]
    for stars in (5, 5, 5, 2):
        client.post("/api/games/family-cloud-hop/rate", json={"stars": stars, "initials": "JER"})
    rating = client.post("/api/games/family-cloud-hop/rate", json={"stars": 4, "initials": "KAT"}).get_json()["rating"]
    assert rating == {"count": 2, "average": 3.0}


def test_block_blast_benchmark_needs_the_complete_game():
    bench = LLMModelBenchmark.__new__(LLMModelBenchmark)
    game = GOOD_MOBILE.replace(
        "// bubble pop burst spiky combo",
        "// 8x8 board grid, tray of pieces (polyomino shapes), pointermove drag with ghost preview, "
        "canPlace(), clear full row and column, combo x2, gameOver when no piece fits",
    )
    assert bench._verify_functional_response({"id": "mobile_block_blast"}, game)
    assert not bench._verify_functional_response({"id": "mobile_block_blast"}, game.replace("combo", "streak"))
    assert not bench._verify_functional_response({"id": "mobile_block_blast"}, game.replace("pointermove", "click"))
