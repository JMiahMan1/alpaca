import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "arcade/static/arcade.js").read_text()
LIFECYCLE = JS[JS.index("  let scorePollToken = 0;") : JS.index("  // Score auto-capture:")]
BOARD = JS[JS.index("  function watchBoard(") : JS.index("  // Remember the player's callsign")]
CONTROLS = JS[JS.index('  const controlsBtn = $("btn-controls");') : JS.index("  function isFull()")]

PRELUDE = """
const assert = require('node:assert/strict');
const listeners = {}, docListeners = {}, timers = new Map(), intervals = [];
let nextTimer = 0, refreshed = 0, celebrated = 0;
const elements = new Map();
const $ = id => {
  if (!elements.has(id)) elements.set(id, {
    value: '', textContent: '', style: {}, disabled: false, events: {}, attrs: {},
    addEventListener(event, fn) { this.events[event] = fn; },
    removeAttribute(name) { delete this.attrs[name]; },
    setAttribute(name, value) { this.attrs[name] = value; },
  });
  return elements.get(id);
};
const window = { __arcadeCid: 'old', location: { pathname: '/player/ABC' },
  addEventListener: (event, fn) => { listeners[event] = fn; } };
const document = { hidden: false,
  addEventListener: (event, fn) => { docListeners[event] = fn; },
  querySelector: () => null, querySelectorAll: () => [] };
const navigator = { sendBeacon: () => true };
const setTimeout = (fn, delay) => { const id = ++nextTimer; timers.set(id, {fn, delay}); return id; };
const clearTimeout = id => timers.delete(id);
const setInterval = (fn, delay) => intervals.push({fn, delay});
const refreshScores = async () => { ++refreshed; };
const celebrate = unlocks => { if (unlocks?.length) ++celebrated; };
const exitFull = () => {};
const flush = async () => { for (let i = 0; i < 20; i++) await Promise.resolve(); };
const advance = async () => {
  const [id, timer] = timers.entries().next().value;
  timers.delete(id);
  assert.equal(timer.delay, 3000);
  await timer.fn();
};
const score = n => ({success: true, score: n, initials: 'ABC', new_unlocks: ['win']});
const localStorage = { getItem: () => null, setItem: () => {} };
let fetch;
"""


def run_node(code):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required for Arcade frontend behavior tests")
    script = PRELUDE + "\n(async () => {\n" + code + "\n})().catch(e => { console.error(e); process.exit(1); });"
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr


def test_repeated_score_poll_dedup_errors_and_new_runs():
    run_node(
        "const slug = 'game';\n"
        + LIFECYCLE
        + """
const responses = [
  {success: true, status: 'no score yet'}, score(10), {...score(10), duplicate: true},
  {success: false, error: 'temporary'}, new Error('offline'), score(20), score(20),
];
let calls = 0;
fetch = async (url, options) => {
  assert.equal(url, '/api/games/game/sync_score');
  assert.equal(JSON.parse(options.body).container_id, 'old');
  ++calls;
  const data = responses.shift();
  if (data instanceof Error) throw data;
  return {json: async () => data};
};
startScorePoll($('live-status'));
await flush();
await advance();
assert.equal($('score-value').value, 10);
assert.equal(celebrated, 1);
$('score-value').value = 99;
await advance();
assert.equal($('score-value').value, 99);
assert.equal(celebrated, 1);
assert.equal(refreshed, 1);
await advance();
assert.match($('live-status').textContent, /temporary/);
await advance();
await advance();
assert.equal($('score-value').value, 20);
await advance();
assert.equal(calls, 7);
assert.equal(celebrated, 3);
assert.equal(timers.size, 1);
stopScorePoll();
assert.equal(timers.size, 0);
"""
    )


def test_old_poll_response_cannot_update_new_session():
    run_node(
        "const slug = 'game';\n"
        + LIFECYCLE
        + """
let resolveOld;
fetch = () => new Promise(resolve => { resolveOld = resolve; });
startScorePoll($('live-status'));
stopScorePoll();
window.__arcadeCid = 'new';
resolveOld({json: async () => score(100)});
await flush();
assert.equal(refreshed, 0);
assert.equal(timers.size, 0);
fetch = async () => ({json: async () => score(200)});
startScorePoll($('live-status'));
await flush();
assert.equal($('score-value').value, 200);
assert.equal(timers.size, 1);
"""
    )


@pytest.mark.parametrize("sync_fails", [False, True])
def test_stop_waits_for_final_sync_and_consumes_stop_capture(sync_fails):
    run_node(
        "const slug = 'game';\n"
        + LIFECYCLE
        + f"\nconst syncFails = {json.dumps(sync_fails)};\n"
        + """
const calls = [];
let finishSync;
fetch = (url, options) => {
  calls.push(url);
  assert.equal(JSON.parse(options.body).container_id, 'old');
  if (url.endsWith('/sync_score')) return new Promise((resolve, reject) => {
    finishSync = () => syncFails ? reject(new Error('offline')) : resolve({json: async () => score(10)});
  });
  return Promise.resolve({ok: true, json: async () => ({success: true, score_sync: score(30)})});
};
const pending = stopLiveSession();
assert.equal(stopLiveSession(), pending);
assert.deepEqual(calls, ['/api/games/game/sync_score']);
finishSync();
await pending;
assert.deepEqual(calls, ['/api/games/game/sync_score', '/api/games/game/stop']);
assert.equal(window.__arcadeCid, null);
assert.equal($('score-value').value, 30);
assert.equal(timers.size, 0);
"""
    )


def test_stop_button_and_restart_use_final_sync():
    launch = JS[JS.index('  if ($("btn-play-live"))') : JS.index("  let scorePollToken = 0;")]
    run_node(
        "const slug = 'game', COARSE = false, soundBtn = null;\n"
        + LIFECYCLE
        + launch
        + """
const calls = [];
fetch = async url => {
  calls.push(url.split('/').pop());
  const data = url.endsWith('/launch') ? {success: true, container_id: 'new', launcher_url: '/live'} :
    url.endsWith('/stop') ? {success: true, score_sync: {...score(10), duplicate: true}} : score(10);
  return {ok: true, json: async () => data};
};
await $('btn-play-live').events.click();
await flush();
assert.deepEqual(calls, ['sync_score', 'stop', 'launch', 'sync_score']);
assert.equal(window.__arcadeCid, 'new');
await $('btn-stop-live').events.click();
assert.deepEqual(calls.slice(-2), ['sync_score', 'stop']);
assert.equal(window.__arcadeCid, null);
assert.equal($('live-frame').style.display, 'none');
assert.equal($('btn-play-live').disabled, false);
assert.equal(timers.size, 0);
"""
    )


@pytest.mark.parametrize("beacon", ["true", "false", "throw"])
def test_pagehide_uses_atomic_capture_and_stop_with_keepalive_fallback(beacon):
    run_node(
        "const slug = 'game';\n"
        + LIFECYCLE
        + f"\nconst beacon = {json.dumps(beacon)};\n"
        + """
let sent, fallback;
navigator.sendBeacon = (url, body) => {
  sent = {url, body};
  if (beacon === 'throw') throw new Error('unavailable');
  return beacon === 'true';
};
fetch = async (url, options) => { fallback = {url, options}; };
listeners.pagehide();
assert.equal(sent.url, '/api/games/game/stop');
assert.deepEqual(JSON.parse(await sent.body.text()), {container_id: 'old'});
assert.equal(window.__arcadeCid, null);
if (beacon === 'true') assert.equal(fallback, undefined);
else {
  assert.equal(fallback.url, sent.url);
  assert.equal(fallback.options.keepalive, true);
  assert.deepEqual(JSON.parse(fallback.options.body), {container_id: 'old'});
}
listeners.pagehide();
listeners.pageshow({persisted: true});
assert.equal($('btn-stop-live').disabled, true);
assert.equal($('live-frame').style.display, 'none');
"""
    )


def test_board_refresh_timing_visibility_focus_and_overlap():
    run_node(
        "const slug = 'game';\n"
        + BOARD
        + """
let resolve;
watchBoard(() => { ++refreshed; return new Promise(r => { resolve = r; }); });
assert.equal(intervals[0].delay, 15000);
const pending = intervals[0].fn();
await listeners.focus();
assert.equal(refreshed, 1);
resolve(); await pending;
document.hidden = true;
await intervals[0].fn();
assert.equal(refreshed, 1);
document.hidden = false;
const visible = docListeners.visibilitychange();
assert.equal(refreshed, 2);
resolve(); await visible;
"""
    )


@pytest.mark.parametrize("player", [False, True])
def test_index_and_player_board_refresh(player):
    run_node(
        f"const slug = null, player = {json.dumps(player)};\n"
        + """
const top = {textContent: ''};
let replaced = false;
const card = {dataset: {gameSlug: 'game'}, querySelector: () => top, classList: null};
document.querySelector = () => player ? {replaceWith: () => { replaced = true; }} : null;
document.querySelectorAll = () => [card];
const DOMParser = class { parseFromString() { return {querySelector: () => ({})}; } };
fetch = async (url, options) => {
  assert.equal(options.cache, 'no-store');
  if (player) {
    assert.equal(url, '/player/ABC');
    return {ok: true, text: async () => '<main></main>', json: async () => ({})};
  }
  assert.equal(url, '/api/games');
  return {ok: true, json: async () => ({success: true, games: [{slug: 'game', top_score: {initials: 'XYZ', score: 42}}]})};
};
(function () {
"""
        + BOARD
        + """
})();
await intervals[0].fn();
if (player) assert.equal(replaced, true);
else assert.equal(top.textContent, 'XYZ — 42');
"""
    )


@pytest.mark.parametrize("saved", ["0", "1"])
def test_desktop_controls_toggle_persists(saved):
    run_node(
        f"let stored = {json.dumps(saved)};\n"
        + """
const localStorage = {
  getItem: key => { assert.equal(key, 'arcade_show_controls'); return stored; },
  setItem: (key, value) => { assert.equal(key, 'arcade_show_controls'); stored = value; },
};
let visible;
const screenEl = {classList: {toggle: (name, on) => { assert.equal(name, 'show-controls'); visible = on; }}};
"""
        + CONTROLS
        + """
assert.equal(visible, stored === '1');
const previous = visible;
$('btn-controls').events.click();
assert.equal(visible, !previous);
assert.equal(stored, visible ? '1' : '0');
assert.equal($('btn-controls').textContent, visible ? 'Hide Controls' : 'Show Controls');
assert.equal($('btn-controls').attrs['aria-pressed'], String(visible));
"""
    )


def test_sound_toggle_labels_show_state():
    sound = JS[JS.index("  const SOUND_KEY =") : JS.index("  function mapXkey(key) {")]
    run_node(
        """
let stored = '1';
const localStorage = {
  getItem: key => { assert.equal(key, 'arcade_sound'); return stored; },
  setItem: (key, value) => { assert.equal(key, 'arcade_sound'); stored = value; },
};
"""
        + sound
        + """
const posted = [];
const fr = $('live-frame');
fr.style.display = 'block';
fr.contentWindow = { postMessage: (msg, origin) => posted.push([msg, origin]) };
const btn = $('btn-sound-live');
assert.equal(btn.textContent, '🔊 Sound');
btn.events['click']();
assert.equal(btn.textContent, '🔇 Muted');
assert.equal(stored, '0');
assert.deepEqual(posted[0][0], { source: 'arcade-audio', on: false });
btn.events['click']();
assert.equal(btn.textContent, '🔊 Sound');
assert.equal(stored, '1');
assert.deepEqual(posted[1][0], { source: 'arcade-audio', on: true });
"""
    )


@pytest.mark.parametrize("saved", ["1", "0"])
def test_sound_preference_is_reapplied_on_load(saved):
    """The persisted sound preference (arcade_sound) re-applies on page load:
    the button label and the postLiveAudio channel both honor it."""
    sound = JS[JS.index("  const SOUND_KEY =") : JS.index("  function mapXkey(key) {")]
    run_node(
        f"let stored = {json.dumps(saved)};\n"
        + """
const localStorage = {
  getItem: key => { assert.equal(key, 'arcade_sound'); return stored; },
  setItem: (key, value) => { assert.equal(key, 'arcade_sound'); stored = value; },
};
"""
        + sound
        + """
const posted = [];
const fr = $('live-frame');
fr.style.display = 'block';
fr.contentWindow = { postMessage: (msg, origin) => posted.push([msg, origin]) };
const btn = $('btn-sound-live');
assert.equal(btn.textContent, stored === '1' ? '🔊 Sound' : '🔇 Muted');
assert.equal(btn.title, stored === '1' ? 'Mute game sound' : 'Unmute game sound');
"""
    )


def test_failed_stop_keeps_session_and_resumes_polling():
    run_node(
        "const slug = 'game';\n"
        + LIFECYCLE
        + """
fetch = async url => url.endsWith('/stop') ?
  {ok: false, json: async () => ({error: 'busy', score_sync: score(20)})} :
  {ok: true, json: async () => ({success: true, duplicate: true})};
await $('btn-stop-live').events.click();
await flush();
assert.equal(window.__arcadeCid, 'old');
assert.equal($('btn-stop-live').disabled, false);
assert.equal($('btn-play-live').disabled, false);
assert.equal($('score-value').value, 20);
assert.equal(timers.size, 1);
assert.match($('live-status').textContent, /busy/);
"""
    )


def test_play_board_reloads_concurrent_scores_without_caching():
    refresh = JS[JS.index("  async function refreshScores()") : JS.index('  $("btn-submit-score").addEventListener')]
    run_node(
        """
const slug = 'game';
const watchBoard = callback => intervals.push(callback);
const list = $('score-list');
list.children = [];
list.appendChild = child => list.children.push(child);
Object.defineProperty(list, 'innerHTML', {set: () => { list.children = []; }});
document.createElement = tag => ({tag, children: [], appendChild(child) { this.children.push(child); }});
let scores = [{initials: 'ABC', score: 10}];
fetch = async (url, options) => {
  assert.equal(url, '/api/games/game');
  assert.equal(options.cache, 'no-store');
  return {json: async () => ({success: true, game: {scores}})};
};
"""
        + refresh
        + """
await intervals[0]();
assert.equal(list.children[0].children[1].textContent, 10);
scores = [{initials: 'XYZ', score: 100}, ...scores];
await intervals[0]();
assert.equal(list.children.length, 2);
assert.equal(list.children[0].children[0].children[0].textContent, 'XYZ');
assert.equal(list.children[0].children[1].textContent, 100);
"""
    )


def test_templates_wire_refresh_and_preserve_mobile_controls():
    templates = ROOT / "arcade/templates"
    for name in ("index", "player", "players", "play"):
        assert '<script src="/static/arcade.js"></script>' in (templates / f"{name}.html").read_text()
    play = (templates / "play.html").read_text()
    assert 'id="btn-stop-live"' in play
    assert 'aria-controls="play-keys">Show Controls</button>' in play
    assert ".screen.show-controls #play-overlay-bar { display: flex;" in play
    assert "@media (pointer: fine) { .screen:not(.show-controls) #play-keys { display: none; } }" in play
    assert 'id="play-keys" role="group"' in play
    # Every arcade page carries the Players + My player card navigation.
    for name in ("index", "player", "players", "play"):
        html = (templates / f"{name}.html").read_text()
        assert 'href="/players"' in html
        assert "data-my-card" in html


def test_player_directory_template_wires_standings_and_unverified_copy():
    players = (ROOT / "arcade/templates/players.html").read_text()
    assert 'id="players-search"' in players
    assert 'data-player-initials' in players
    assert "unverified" in players
    assert 'href="/player/{{ p.initials }}"' in players


def test_dpad_hold_release_and_cancel_unchanged():
    keys = JS[JS.index('  document.querySelectorAll("#play-keys button")') : JS.index("  if (COARSE) {")]
    run_node(
        """
const button = $('arrow'), sent = [];
button.setPointerCapture = id => assert.equal(id, 7);
document.querySelectorAll = () => [button];
const sendArcadeKey = (b, down) => { assert.equal(b, button); sent.push(down); };
"""
        + keys
        + """
let prevented = false;
button.events.pointerdown({pointerId: 7, preventDefault: () => { prevented = true; }});
assert.equal(prevented, true);
for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) button.events[name]();
button.events.keydown({key: 'Enter'});
button.events.keyup({key: 'Enter'});
assert.deepEqual(sent, [true, false, false, false, true, false]);
"""
    )


def test_my_player_card_link_uses_saved_callsign():
    """The shared "My player card" header link follows the saved callsign
    (localStorage) and stays on /players until one is remembered."""
    my_card = JS[JS.index('  const CALLSIGN_KEY = "arcade_callsign";') : JS.index("  function watchBoard(")]
    run_node(
        my_card
        + """
function run() {
  const link = { textContent: '', attrs: {}, setAttribute(k, v) { this.attrs[k] = v; } };
  document.querySelectorAll = selector => {
    assert.equal(selector, '[data-my-card]');
    return [link];
  };
  let stored = null;
  localStorage.getItem = () => stored;
  localStorage.setItem = (k, v) => { stored = v; };
  paintMyCardLinks();
  assert.equal(link.textContent, 'My player card');
  assert.equal(link.attrs.href, undefined);
  assert.equal(rememberCallsign('  j-Q!x '), 'JQX');
  assert.equal(stored, 'JQX');
  paintMyCardLinks();
  assert.equal(link.textContent, 'My player card · JQX');
  assert.equal(link.attrs.href, '/player/JQX');
  rememberCallsign('');
  assert.equal(link.textContent, 'My player card · JQX');
}
run();
"""
    )


def test_players_directory_search_filters_rows():
    """The /players search box hides non-matching standings rows live."""
    run_node(
        "const slug = null;\n"
        + BOARD
        + """
function run() {
  const search = $('players-search');
  const rowA = { style: {}, dataset: { playerInitials: 'ACE' }, classList: null };
  const rowB = { style: {}, dataset: { playerInitials: 'BEE' }, classList: null };
  document.querySelector = selector => selector === '#players-search' ? search : null;
  document.querySelectorAll = selector => selector === '[data-player-initials]' ? [rowA, rowB] : [];
  search.value = 'be';
  search.events.input();
  assert.equal(rowA.style.display, 'none');
  assert.equal(rowB.style.display, '');
  search.value = '';
  search.events.input();
  assert.equal(rowA.style.display, '');
  assert.equal(rowB.style.display, '');
}
run();
"""
    )
