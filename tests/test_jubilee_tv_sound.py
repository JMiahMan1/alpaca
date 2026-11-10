"""Jubilee's sound goes through the TV while a TV shows the board.

A room of phones each playing the same dice rattle and rent "bad" over a TV
showing the game is an echo chamber. While a live TV stage polls the room
(``stage_connected`` and ``stage_sound``), every phone is silent -- board
events and the player's own taps alike, haptics and visuals kept -- and the TV
page (``templates/jubilee_tv.html``) plays the board's sounds, synthesized
with WebAudio the way ``party_tv.html``'s effects are. A streamed stage (a
Roku gets headless Chromium as video, which carries no sound) says so when it
polls, and the phones keep their sound.

The JavaScript is run in node where node is installed: the kit's gate, the TV
page's sound block with a stand-in AudioContext, and its reading of the
server's real log lines.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from arcade import jubilee
from tests.test_jubilee_tv_lowfx import _strip_js

ROOT = Path(__file__).resolve().parent.parent
TV = ROOT / "arcade" / "templates" / "jubilee_tv.html"
GAME = ROOT / "arcade" / "family_games" / "jubilee" / "game.html"
KIT = ROOT / "arcade" / "family_games" / "_kit" / "kit.js"

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _tv_js() -> str:
    return re.findall(r"<script>(.*?)</script>", TV.read_text(encoding="utf-8"), re.S)[1]


def _game_js() -> str:
    return "\n".join(re.findall(r"<script>(.*?)</script>", GAME.read_text(encoding="utf-8"), re.S))


def _run_node(tmp_path: Path, src: str) -> dict:
    f = tmp_path / "harness.js"
    f.write_text(src, encoding="utf-8")
    out = subprocess.run([NODE, str(f)], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


# --- the server: is a TV showing the room, and does it play sound? -----------

@pytest.fixture
def room():
    made = jubilee.create_room({"name": "Ann"})
    code, pid = made["room"]["code"], made["player_id"]
    yield code, pid
    jubilee._clear_rooms()


def test_a_live_tv_stage_plays_the_sound(room):
    code, pid = room
    assert jubilee.get_room(code, pid)["stage_sound"] is False
    jubilee.get_room(code, stage=True)
    seen = jubilee.get_room(code, pid)
    assert seen["stage_connected"] is True and seen["stage_sound"] is True


def test_a_streamed_stage_shows_the_board_but_leaves_the_sound_on_the_phones(room):
    code, pid = room
    jubilee.get_room(code, stage=True, video=True)
    seen = jubilee.get_room(code, pid)
    assert seen["stage_connected"] is True and seen["stage_sound"] is False


def test_the_route_passes_the_video_flag(room):
    import arcade.app as arcade_app

    code, _pid = room
    arcade_app.app.config["TESTING"] = True
    with arcade_app.app.test_client() as c:
        r = c.get(f"/api/jubilee/rooms/{code}?stage=1&video=1").get_json()["room"]
        assert r["stage_connected"] and not r["stage_sound"]
        r = c.get(f"/api/jubilee/rooms/{code}?stage=1").get_json()["room"]
        assert r["stage_connected"] and r["stage_sound"]


def test_the_tv_page_reports_a_streamed_stage():
    js = _tv_js()
    assert "var STREAMED = /HeadlessChrome/.test(navigator.userAgent" in js
    assert '"?stage=1" + (STREAMED ? "&video=1" : "")' in js


# --- the phone: silent while the TV plays ------------------------------------

def test_every_phone_sound_goes_through_the_kit():
    """The gate is in the kit's synth, so it must be the only way a sound is made."""
    js = _strip_js(_game_js())
    assert not re.search(r"AudioContext|new Audio\b|\.play\(\)", js)
    assert "kit.sound.tone(" not in js
    # The player's own taps too: buying, bidding, accepting a deal.
    src = _game_js()
    for where in ("doAction('/buy', {}); kit.sound.play('coin')",
                  "doAction('/auction_bid', { amount: +b.dataset.bid }); kit.sound.play('coin')",
                  "kit.toast('🤝 Deal done!'); kit.sound.play('coin')"):
        assert where in src


def test_the_phone_is_silenced_while_a_tv_stage_plays_the_sound():
    js = _game_js()
    assert "function tvHasSound(room) { return !!(room && room.stage_connected && room.stage_sound !== false); }" in js
    sync = js[js.index("function syncTvSound()"):js.index("tap(soundBtn")]
    assert "kit.sound.setSilenced(tv)" in sync
    apply = js[js.index("function applyState(next)"):js.index("function syncTabs()")]
    assert "syncTvSound();" in apply, "every room update re-decides, so the sound comes back when the TV goes"
    # Not by muting: that is the player's choice, saved, and must survive the TV.
    assert "setMuted(tv" not in js and "setMuted(true" not in js


def test_the_speaker_button_keeps_the_players_choice_and_shows_the_tv():
    js = _game_js()
    handler = js[js.index("tap(soundBtn"):js.index("syncSoundBtn();", js.index("tap(soundBtn"))]
    assert "state.muted = !state.muted" in handler and "store.set('muted'" in handler
    assert "soundBtn.classList.toggle('tv', tv)" in js
    html = GAME.read_text(encoding="utf-8")
    assert '<span id="jb-tv-sound" hidden>' in html
    assert ".jb-icon.tv::after" in html


def test_the_kit_silence_is_separate_from_mute():
    kit = KIT.read_text(encoding="utf-8")
    assert kit.count("if (!ctx || muted || silenced) return;") == 2, "both tone and noise obey it"
    setter = kit[kit.index("setSilenced(on)"):kit.index("_slug: slug")]
    assert "store(" not in setter, "never saved: it is the TV's, not the player's"


@needs_node
def test_the_kit_makes_no_sound_while_silenced(tmp_path):
    kit = KIT.read_text(encoding="utf-8")
    start = kit.index("function createSound(slug)")
    body = kit[start:kit.index("/* ---------- Particles", start)]
    src = """
var made = 0, saved = {};
function store(k, v) { if (v === undefined) return saved[k] || null; saved[k] = v; }
function P() { return { value: 0, setValueAtTime() {}, exponentialRampToValueAtTime() {} }; }
class AC {
  constructor() { this.state = "running"; this.currentTime = 0; this.sampleRate = 8000; this.destination = {}; }
  createGain() { return { gain: P(), connect() {} }; }
  createOscillator() { return { frequency: P(), connect() {}, start() { made++; }, stop() {} }; }
  createBuffer(c, n) { return { getChannelData() { return new Float32Array(n); } }; }
  createBufferSource() { return { connect() {}, start() { made++; } }; }
  createBiquadFilter() { return { frequency: { value: 0 }, connect() {} }; }
  resume() { return Promise.resolve(); }
}
var window = { AudioContext: AC };
""" + body + """
var s = createSound("x"); s.unlock();
var out = {};
s.play("coin"); s.noise(0.2); out.normal = made; made = 0;
s.setSilenced(true); s.play("coin"); s.play("win"); s.noise(0.2); out.silenced = made; made = 0;
out.savedWhileSilenced = JSON.stringify(saved);
s.setSilenced(false); s.play("pop"); out.back = made; made = 0;
s.setMuted(true); s.play("pop"); out.muted = made;
console.log(JSON.stringify(out));
"""
    out = _run_node(tmp_path, src)
    assert out["normal"] > 0 and out["back"] > 0
    assert out["silenced"] == 0 and out["muted"] == 0
    assert out["savedWhileSilenced"] == "{}"


# --- the TV: plays every board event -----------------------------------------

EVENTS = ("dice", "hop", "land", "coin", "card", "pay", "dungeon", "build", "bankrupt", "win")


def test_the_tv_has_a_sound_for_every_board_event():
    js = _tv_js()
    block = js[js.index("var SFX = {"):js.index("function playSfx")]
    for name in EVENTS:
        assert re.search(r"\b" + name + r": function \(d\)", block), f"no TV sound for {name}"


def test_each_sound_is_hooked_where_the_tv_animates_it():
    js = _tv_js()
    moments = js[js.index("function bigMoments(room)"):js.index("function poll()")]
    assert "flyCard(c);\n      playSfx(\"card\");" in moments
    assert "showDice(r, true);\n      playSfx(\"dice\");" in moments
    tick = js[js.index("function tick(now)"):js.index("/* ---------- the middle")]
    assert 'playSfx(t.to === 0 ? "coin" : t.queue.length ? "hop" : "land")' in tick
    poll = js[js.index("function poll()"):js.index("function renderNoCode()")]
    assert "logSounds(room);" in poll
    assert 'if (room.status === "finished" && was === "playing") playSfx("win"' in poll


def test_no_sound_without_a_game_on_screen():
    js = _tv_js()
    poll = js[js.index("function poll()"):js.index("function renderNoCode()")]
    assert 'soundOn = !STREAMED && (room.status === "playing" || room.status === "finished");' in poll
    assert poll.index("soundOn = ") < poll.index('if (room.status === "lobby")')
    for fn in ("function renderNoCode()", "function renderGone()"):
        assert js[js.index(fn):js.index(fn) + 80].count("soundOn = false;") == 1
    play = js[js.index("function playSfx"):js.index("window.__jbSound")]
    assert "if (!soundOn || STREAMED) return;" in play


def test_autoplay_is_handled_like_the_party_stage():
    js = _tv_js()
    for ev in ("keydown", "click", "touchstart"):
        assert f'document.addEventListener("{ev}", unlock);' in js
    assert '<div id="tv-sound">🔊 Press OK for sound</div>' in TV.read_text(encoding="utf-8")
    assert 'if (!c || c.state === "suspended") return;' in js, "a blocked context does not bank sounds"


SOUND_HARNESS = """
var calls = [], STATE = "running", UA = %(ua)s;
function P() { return { value: 0, setValueAtTime() {}, exponentialRampToValueAtTime() {} }; }
class AC {
  constructor() { this.state = STATE; this.currentTime = 0; this.sampleRate = 8000; this.destination = {}; }
  createGain() { return { gain: P(), connect() {} }; }
  createOscillator() { var o = { type: "", frequency: P(), connect() {}, start(t) { calls.push(["tone", t]); }, stop() {} }; return o; }
  createBuffer(c, n) { return { getChannelData() { return new Float32Array(n); } }; }
  createBufferSource() { return { connect() {}, start(t) { calls.push(["noise", t]); } }; }
  createBiquadFilter() { return { frequency: { value: 0 }, Q: { value: 0 }, connect() {} }; }
  resume() { if (STATE === "running") this.state = "running"; return Promise.resolve(); }
}
var els = {};
var window = { AudioContext: AC };
var navigator = { userAgent: UA };
var document = { addEventListener() {}, getElementById(id) { return els[id] || (els[id] = { className: "" }); } };
function $(id) { return document.getElementById(id); }
var RM = false, LOWFX = false, board = null, HOP_MS = 280;
%(block)s
var out = {};
%(body)s
console.log(JSON.stringify(out));
"""


def _sound_block() -> str:
    js = _tv_js()
    return js[js.index("/* ---------- sound: the TV is the room's speaker"):js.index('var lastSig = "";')]


def _harness(tmp_path: Path, body: str, ua: str = "Mozilla/5.0 (Web0S; Linux/SmartTV) Chrome/68") -> dict:
    src = SOUND_HARNESS % {"ua": json.dumps(ua), "block": _sound_block(), "body": body}
    return _run_node(tmp_path, src)


@needs_node
def test_every_tv_sound_makes_sound_while_a_game_is_on(tmp_path):
    out = _harness(tmp_path, """
playSfx("dice"); out.lobby = calls.length;
soundOn = true;
Object.keys(SFX).forEach(function (k) { calls = []; playSfx(k); out[k] = calls.length; });
""")
    assert out["lobby"] == 0, "no sound in the lobby"
    for name in EVENTS:
        assert out[name] > 0, f"{name} made no sound"


@needs_node
def test_a_streamed_stage_makes_no_sound(tmp_path):
    out = _harness(tmp_path, 'soundOn = true; playSfx("dice"); playSfx("win"); out.n = calls.length;',
                   ua="Mozilla/5.0 HeadlessChrome/120.0")
    assert out["n"] == 0


@needs_node
def test_a_blocked_context_drops_sounds_and_asks_for_a_key(tmp_path):
    out = _harness(tmp_path, """
STATE = "suspended"; soundOn = true;
playSfx("dice"); playSfx("coin"); out.n = calls.length;
setTimeout(function () { out.hint = $("tv-sound").className; console.log(JSON.stringify(out)); process.exit(0); }, 700);
""")
    assert out["n"] == 0 and out["hint"] == "on"


def _server_lines() -> dict[str, str]:
    """One real line per log message jubilee.py writes that should (or must not) sound."""
    return {
        "Ann is sent to the Dungeon": "dungeon",
        "Ann throws three doubles in a row - to the Dungeon!": "dungeon",
        "Ann throws doubles - out of the Dungeon!": "",
        "Ann throws 2-5 - no doubles, one turn passes in the Dungeon": "",
        "Ann holds a Dungeon-free card": "",
        "Ann passes Jubilee! and collects $200": "coin",
        "Ann pays $50 and leaves the Dungeon": "pay",
        "Ann renders unto Caesar $200": "pay",
        "Ann is bankrupt and leaves the game": "bankrupt",
        "Ann builds on Jerusalem": "build",
        "Scribe raises buildings on the Holy Land": "build",
        "Ann sells buildings on Jerusalem for $50": "",
        "Ann buys Jerusalem for $400": "",
        "Ann passes": "",
        "Ann passes on Jerusalem - it returns to the Bank": "",
    }


def test_the_sample_lines_are_the_servers():
    src = (ROOT / "arcade" / "jubilee.py").read_text(encoding="utf-8")
    for fragment in ("is sent to the Dungeon", "passes Jubilee! and collects", "renders unto Caesar",
                     "is bankrupt and leaves the game", "builds on", "raises buildings on the Holy Land",
                     "sells buildings on", "out of the Dungeon!", "holds a Dungeon-free card",
                     "- to the Dungeon!", "pays $50 and leaves the Dungeon"):
        assert fragment in src, f"jubilee.py no longer logs {fragment!r}: update the TV's lineSound"


@needs_node
def test_the_tv_reads_the_servers_log_lines(tmp_path):
    lines = _server_lines()
    out = _harness(tmp_path, "var L = " + json.dumps(list(lines))
                   + "; L.forEach(function (l) { out[l] = lineSound(l); });")
    assert out == lines


@needs_node
def test_rent_is_heard_from_the_cash_and_a_purchase_is_not(tmp_path):
    out = _harness(tmp_path, """
soundOn = true;
var played = [];
playSfx = function (k, d) { played.push(k); };
function room(seq, cash, log) { return { log_seq: seq, log: log, players: [{ name: "Ann", cash: cash[0] }, { name: "Bob", cash: cash[1] }] }; }
logSounds(room(3, [1500, 1500], ["a", "b", "c"])); out.primed = played.slice(); played = [];
logSounds(room(3, [1400, 1600], ["a", "b", "c"])); out.rent = played.slice(); played = [];
logSounds(room(4, [1000, 1600], ["b", "c", "Ann buys Jerusalem for $400"])); out.buy = played.slice(); played = [];
logSounds(room(6, [1000, 1800], ["c", "Ann buys Jerusalem for $400", "Bob passes Jubilee! and collects $200", "Bob is sent to the Dungeon"]));
out.two = played.slice(); played = [];
logSounds(room(1, [1500, 1500], ["x"])); out.rematch = played.slice();
""")
    assert out["primed"] == [], "the first look at a room plays nothing old"
    assert out["rent"] == ["pay"]
    assert out["buy"] == []
    assert out["two"] == ["coin", "dungeon"]
    assert out["rematch"] == [], "a fresh ledger primes again"


def test_the_tv_page_stays_es5():
    from tests.test_jubilee_tv_lowfx import test_the_page_is_es5

    test_the_page_is_es5()
