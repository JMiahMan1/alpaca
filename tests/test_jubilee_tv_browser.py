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
    page.wait_for_selector(".tvZoom > svg text", timeout=8000)
    page.wait_for_timeout(600)
    fit = page.evaluate("""() => {
        const r = document.scrollingElement;
        const rect = document.querySelector('.tvPlay').getBoundingClientRect();
        return { sh: r.scrollHeight, ih: innerHeight, sw: r.scrollWidth, iw: innerWidth,
                 left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom };
    }""")
    assert fit["sh"] <= fit["ih"] + 1 and fit["sw"] <= fit["iw"] + 1
    assert fit["left"] >= 0 and fit["top"] >= 0 and fit["right"] <= fit["iw"] + 1 and fit["bottom"] <= fit["ih"] + 1
    tiles = page.evaluate("() => document.querySelectorAll('.tvZoom > svg > g[transform]').length")
    assert tiles >= 40
    sizes = page.evaluate("""() => [...document.querySelectorAll('.tvZoom > svg text.tl')].map(t => +t.getAttribute('font-size'))""")
    assert len(sizes) >= 50 and len(set(sizes)) == 1, f"one name/price size on every space, got {sorted(set(sizes))}"
    assert "'s turn" in page.inner_text(".tvCenter")
    assert "Ann" in page.inner_text(".tvFrame.left") and code in page.inner_text(".tvFrame.right")


def test_a_roll_moves_the_piece_along_and_shows_where_it_landed(arcade_server, browser):
    _, base = arcade_server
    code, pid = _room()
    page = browser.new_page(viewport={"width": 960, "height": 540})
    page.goto(f"{base}/jubilee/tv/{code}")
    page.wait_for_selector(".tvZoom > svg text", timeout=8000)
    page.evaluate("""() => {
        window.__seen = new Set();
        const tok = () => document.querySelector('.tvZoom > svg > g:last-of-type > g');
        (function loop() { const t = tok(); if (t) window.__seen.add(t.getAttribute('transform')); requestAnimationFrame(loop); })();
    }""")
    out = jubilee.roll(code, pid)
    steps = out["room"]["last_roll"]["d1"] + out["room"]["last_roll"]["d2"]
    page.wait_for_timeout(1500 + steps * 300)
    positions = page.evaluate("() => window.__seen.size")
    # A card on the space it reached can send it on (to the Dungeon, or more
    # than 12 spaces, which the page jumps straight to), so count hops only
    # when the piece ended where the dice put it.
    if out["room"]["you"]["position"] == steps:
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
    page.wait_for_selector(".tvZoom > svg text", timeout=8000)
    server.shutdown()
    page.wait_for_selector("#tv-reconnect.on", timeout=12000)
    assert "GAME OVER" not in page.inner_text("body").upper()
    time.sleep(0.1)


_FIT = """(sel) => {
  const out = [];
  document.querySelectorAll(sel).forEach((g, idx) => {
    const rect = g.querySelector(':scope > rect'); if (!rect) return;
    const x = +rect.getAttribute('x'), y = +rect.getAttribute('y'), w = +rect.getAttribute('width'), h = +rect.getAttribute('height');
    g.querySelectorAll(':scope > text, :scope > circle, :scope > ellipse, :scope > g').forEach(el => {
      if (el.getAttribute('display') === 'none' || el.getAttribute('fill') === 'none') return;
      let b; try { b = el.getBBox(); } catch (e) { return; }
      if (!b.width && !b.height) return;
      const m = el.transform && el.transform.baseVal.consolidate ? el.transform.baseVal.consolidate() : null;
      if (m) { const k = m.matrix; b = { x: k.e + b.x * k.a, y: k.f + b.y * k.d, width: b.width * k.a, height: b.height * k.d }; }
      if (b.x < x + 1.5 || b.x + b.width > x + w - 1.5 || b.y < y + 1.5 || b.y + b.height > y + h - 1.5)
        out.push(idx + ': ' + el.tagName + ' ' + (el.textContent || '').slice(0, 20));
    });
  });
  return out;
}"""


def test_every_space_keeps_its_words_and_picture_inside_its_box(arcade_server, browser):
    _, base = arcade_server
    code, _ = _room()
    page = browser.new_page(viewport={"width": 960, "height": 540})
    page.goto(f"{base}/jubilee/tv/{code}")
    page.wait_for_selector(".tvZoom > svg text", timeout=8000)
    page.wait_for_timeout(500)
    spill = page.evaluate(_FIT, ".tvZoom > svg > g[transform]")
    assert spill == [], f"spilling out of their spaces: {spill}"


_LG_UA = ("Mozilla/5.0 (Web0S; Linux/SmartTV) AppleWebKit/537.36 (KHTML, like Gecko) "
          "Chrome/68.0.3440.106 Safari/537.36 WebAppManager")


@pytest.mark.parametrize("ua,query,expect", [(_LG_UA, "", True), (_LG_UA, "?lowfx=0", False), (None, "?lowfx=1", True), (None, "", False)])
def test_low_power_mode_on_webos_flat_card_and_the_loop_sleeps(arcade_server, browser, ua, query, expect):
    _, base = arcade_server
    code, pid = _room()
    ctx = browser.new_context(viewport={"width": 960, "height": 540}, **({"user_agent": ua} if ua else {}))
    page = ctx.new_page()
    page.add_init_script("""(() => { window.__frames = 0; const raf = window.requestAnimationFrame.bind(window);
        window.requestAnimationFrame = (f) => raf((t) => { window.__frames++; f(t); }); })()""")
    page.goto(f"{base}/jubilee/tv/{code}{query}")
    page.wait_for_selector(".tvZoom > svg text", timeout=8000)
    assert page.evaluate("() => document.documentElement.classList.contains('lowfx')") is expect
    # Idle, the frame loop stops instead of running for the whole game.
    page.wait_for_timeout(1500)
    before = page.evaluate("() => window.__frames")
    page.wait_for_timeout(700)
    assert page.evaluate("() => window.__frames") == before, "the frame loop kept running with nothing moving"
    page.evaluate("""() => window.__jbStage.flyCard({deck: 'parable', player: 'Ann', quote: 'Seek and ye shall find', ref: 'Matthew 7:7', effect: 'Collect $50'})""")
    page.wait_for_selector("#tv-card.on .face.front", timeout=2000)
    assert (page.locator("#tv-card .face.back").count() == 0) is expect
    if expect:
        flat = page.evaluate("""() => [getComputedStyle(document.querySelector('.flyCard')).perspective,
            getComputedStyle(document.querySelector('.flyCard .face.front')).transform,
            getComputedStyle(document.querySelector('.cSign')).animationName]""")
        assert flat == ["none", "none", "none"]
    # A roll still moves the piece (and wakes the loop).
    jubilee.roll(code, pid)
    page.wait_for_timeout(1500)
    assert page.evaluate("() => window.__frames") > before
    ctx.close()
