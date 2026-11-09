"""The Jubilee TV stage in a real browser: it fits the screen, every space has
the same look, the pieces move along the board, a landing card comes up, and
a dropped connection says "Reconnecting" instead of ending the game.

Skips cleanly where Playwright or Chrome/Chromium is not installed (like
test_smoke_playwright); run it with ``pip install playwright`` and either
``playwright install chromium`` or a local Google Chrome.
"""
from __future__ import annotations

import threading
import time

import pytest

sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

pytestmark = pytest.mark.live  # opens a local socket (the arcade under test)

import arcade.app as arcade_app  # noqa: E402
from arcade import jubilee  # noqa: E402


@pytest.fixture
def arcade_server():
    from werkzeug.serving import make_server

    server = make_server("127.0.0.1", 0, arcade_app.app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    jubilee._clear_rooms()


@pytest.fixture
def browser():
    with sync_playwright() as p:
        b = None
        for kwargs in ({"channel": "chrome"}, {}):
            try:
                b = p.chromium.launch(headless=True, **kwargs)
                break
            except Exception:
                continue
        if b is None:
            pytest.skip("No Chrome or Chromium for Playwright")
        yield b
        b.close()


def _room():
    made = jubilee.create_room({"name": "Ann"})
    code, pid = made["room"]["code"], made["player_id"]
    jubilee.start_match(code, pid)
    return code, pid


@pytest.mark.parametrize("size", [(960, 540), (1920, 1080)])
def test_the_tv_board_fits_and_every_space_looks_the_same(arcade_server, browser, size):
    _, base = arcade_server
    code, _ = _room()
    page = browser.new_page(viewport={"width": size[0], "height": size[1]})
    page.goto(f"{base}/jubilee/tv/{code}")
    page.wait_for_selector(".tvBoard svg text", timeout=8000)
    page.wait_for_timeout(600)
    fit = page.evaluate("""() => {
        const r = document.scrollingElement;
        const rect = document.querySelector('.tvPlay').getBoundingClientRect();
        return { sh: r.scrollHeight, ih: innerHeight, sw: r.scrollWidth, iw: innerWidth,
                 left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom };
    }""")
    assert fit["sh"] <= fit["ih"] + 1 and fit["sw"] <= fit["iw"] + 1
    assert fit["left"] >= 0 and fit["top"] >= 0 and fit["right"] <= fit["iw"] + 1 and fit["bottom"] <= fit["ih"] + 1
    tiles = page.evaluate("() => document.querySelectorAll('.tvBoard svg > g[transform]').length")
    assert tiles >= 40
    sizes = page.evaluate("""() => [...document.querySelectorAll('.tvBoard svg text.tl')].map(t => +t.getAttribute('font-size'))""")
    assert len(sizes) >= 50 and len(set(sizes)) == 1, f"one name/price size on every space, got {sorted(set(sizes))}"
    assert "'s turn" in page.inner_text(".tvCenter")
    assert "Ann" in page.inner_text(".tvFrame.left") and code in page.inner_text(".tvFrame.right")


def test_a_roll_moves_the_piece_along_and_shows_where_it_landed(arcade_server, browser):
    _, base = arcade_server
    code, pid = _room()
    page = browser.new_page(viewport={"width": 960, "height": 540})
    page.goto(f"{base}/jubilee/tv/{code}")
    page.wait_for_selector(".tvBoard svg text", timeout=8000)
    page.evaluate("""() => {
        window.__seen = new Set();
        const tok = () => document.querySelector('.tvBoard svg > g:last-of-type > g');
        (function loop() { const t = tok(); if (t) window.__seen.add(t.getAttribute('transform')); requestAnimationFrame(loop); })();
    }""")
    out = jubilee.roll(code, pid)
    steps = out["room"]["last_roll"]["d1"] + out["room"]["last_roll"]["d2"]
    page.wait_for_timeout(1500 + steps * 300)
    positions = page.evaluate("() => window.__seen.size")
    assert positions > steps * 3, f"the piece hopped through {positions} positions for {steps} spaces"
    landed_on = jubilee.SPACES[out["room"]["you"]["position"]]
    if landed_on.get("price"):
        page.wait_for_selector("#tv-land.on", timeout=4000)
        assert landed_on["name"] in page.inner_text("#tv-land")


def test_a_lost_connection_says_reconnecting_instead_of_game_over(arcade_server, browser):
    server, base = arcade_server
    code, _ = _room()
    page = browser.new_page(viewport={"width": 960, "height": 540})
    page.goto(f"{base}/jubilee/tv/{code}")
    page.wait_for_selector(".tvBoard svg text", timeout=8000)
    server.shutdown()
    page.wait_for_selector("#tv-reconnect.on", timeout=12000)
    assert "GAME OVER" not in page.inner_text("body").upper()
    time.sleep(0.1)
