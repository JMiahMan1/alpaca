"""Behavioural tests for the Podcast Studio panel in dashboard.js + index.html.

The panel is the only place the two podcast routes are actually driven: it
decides which host pair is active, which voice each host speaks with, which
saved voice (if any) it is cloned from, and how deep the bed is ducked. A
mismatch between what this sends and what the routes accept is invisible until
a user clicks Render and gets a 400 or, worse, a mix with the wrong voices.

So these tests do the same thing tests/test_dashboard_frontend.py does — slice
the shipped source, run it in `node`, and assert on the request it builds —
plus structural assertions tying the ids the JS reads to the ids the template
declares, which is the other way this wiring silently rots.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JS_PATH = ROOT / "web" / "static" / "js" / "dashboard.js"
HTML_PATH = ROOT / "web" / "templates" / "index.html"
JS = JS_PATH.read_text()
HTML = HTML_PATH.read_text()


def _slice(start_marker: str, end_marker: str) -> str:
    start = JS.index(start_marker)
    end = JS.index(end_marker)
    assert start < end, f"{start_marker!r} must precede {end_marker!r}"
    return JS[start:end]


SLICE_PODCAST = _slice(
    "// ═══════════════════════════ PODCAST STUDIO ═══════════════════════════",
    "function wirePodcastStudio() {",
)


def run_js(code: str) -> str:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required for dashboard frontend behavior tests")
    script = """
const assert = require('node:assert/strict');
const log = [];
""" + code + """
process.stdout.write(log.join('\\n'));
"""
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


# --------------------------------------------------------------------------
# The panel is wired into the SPA
# --------------------------------------------------------------------------


def test_dashboard_js_parses():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required for dashboard frontend behavior tests")
    assert subprocess.run([node, "--check", str(JS_PATH)], capture_output=True, text=True).returncode == 0


def test_the_tab_button_and_view_exist():
    assert 'id="tab-btn-podcast"' in HTML
    assert 'id="view-podcast"' in HTML


def test_the_tab_button_comes_after_audio_and_before_docs():
    nav = HTML[HTML.index('id="tab-btn-audio"') : HTML.index('id="tab-btn-docs"')]
    assert 'id="tab-btn-podcast"' in nav, "the Podcast tab should sit between Audio Studio and API Docs"


def test_switch_tab_deactivates_hides_and_activates_the_podcast_tab():
    body = JS[JS.index("function switchTab(tabName)") : JS.index("tabBtnMonitor.addEventListener")]
    assert "tabBtnPodcast.classList.remove('active')" in body
    assert "viewPodcast.classList.add('d-none')" in body
    assert "tabName === 'podcast'" in body
    assert "viewPodcast.classList.remove('d-none')" in body
    assert "loadPodcastStatus()" in body


def test_the_tab_button_has_a_click_handler():
    assert "tabBtnPodcast.addEventListener('click', () => switchTab('podcast'));" in JS


def test_every_id_the_podcast_js_reads_is_declared_in_the_template():
    """The failure mode this catches: a renamed id makes the panel silently
    do nothing rather than error, because every lookup goes through
    `getElementById` and a null is handled."""
    declared = {m for m in _IDS_IN_TEMPLATE}
    referenced = {m for m in _IDS_IN_JS}
    assert referenced <= declared, f"dashboard.js reads ids the template never declares: {sorted(referenced - declared)}"


_IDS_IN_TEMPLATE = set()
for _token in HTML.split('id="')[1:]:
    _IDS_IN_TEMPLATE.add(_token.split('"')[0])

_IDS_IN_JS = set()
for _line in JS.splitlines():
    for _quoted in ("podcastEl('", "getElementById('podcast"):
        if _quoted in _line:
            _IDS_IN_JS.add(_line.split(_quoted)[1].split("'")[0])
            break


def test_the_ids_check_actually_found_something():
    assert len(_IDS_IN_JS) >= 15
    assert "podcast-script" in _IDS_IN_JS
    assert "view-podcast" in _IDS_IN_JS


# --------------------------------------------------------------------------
# collectPodcastRequest — the payload the routes actually accept
# --------------------------------------------------------------------------


# A minimal DOM stand-in. happy-dom/jsdom are not dependencies, and pulling one
# in to assert a dozen ids would be a heavier test than the code it covers.
_DOM = """
const _els = {};
const mk = (id) => ({ id, value: '', textContent: '', style: {}, dataset: {}, classList: { contains: () => false },
    innerHTML: '', children: [], options: [], disabled: false,
    appendChild(c) { this.children.push(c); return c; },
    append(...cs) { for (const c of cs) this.children.push(c); },
    addEventListener() {} });
const IDS = %s;
for (const id of IDS) _els[id] = mk(id);
const CLONE_SELECTS = %s;
globalThis.document = {
    hidden: false,
    getElementById: id => _els[id] || null,
    querySelector: () => _els['view-podcast'],
    querySelectorAll: sel => (sel === '#podcast-hosts select.js-podcast-clone' ? CLONE_SELECTS : []),
    createElement: () => mk('tmp'),
};
"""

_PODCAST_IDS = [
    "podcast-status-chip", "podcast-pair", "podcast-bed", "podcast-duck", "podcast-duck-val",
    "podcast-hosts", "podcast-hosts-meta", "podcast-sting", "podcast-topic", "podcast-words",
    "podcast-script", "podcast-script-meta", "podcast-player", "podcast-download",
    "podcast-player-box", "podcast-output-empty", "podcast-render-meta", "podcast-warnings",
    "view-podcast", "btn-podcast-draft", "btn-podcast-render",
]


def podcast_js(body: str, clone_selects: list[dict] | None = None) -> str:
    """The shipped podcast source over a scripted DOM, then `body`."""
    return SLICE_PODCAST + (_DOM % (json.dumps(_PODCAST_IDS), json.dumps(clone_selects or []))) + body


def _collect(*scripted) -> str:
    """Run collectPodcastRequest() and return its payload as JSON text.

    Accepts either ``_collect(clones, body)`` or the same two as one tuple.
    """
    if len(scripted) == 1 and isinstance(scripted[0], tuple):
        scripted = scripted[0]
    clone_selects, body = scripted
    return run_js(podcast_js(body, clone_selects) + "\nlog.push(JSON.stringify(__RESULT__));\n")


def _render_hosts(body: str) -> str:
    return run_js(podcast_js(body) + "\nlog.push(JSON.stringify(__OUT__));\n")


def test_the_request_carries_the_script_the_pair_and_the_bed():
    out = _collect((
            [
                {"dataset": {"slot": "a"}, "value": ""},
                {"dataset": {"slot": "b"}, "value": ""},
            ],
            """
_els['podcast-script'].value = 'ADA: hi\\nROWAN: hello';
_els['podcast-pair'].value = 'duo_warm';
_els['podcast-bed'].value = 'lofi_calm';
_els['podcast-duck'].value = '18';
_els['podcast-sting'].value = '';
const __RESULT__ = collectPodcastRequest();
""",
        )
    )
    body = json.loads(out)
    assert body == {
        "script": "ADA: hi\nROWAN: hello",
        "pair_id": "duo_warm",
        "bed_preset": "lofi_calm",
        "duck_db": 18,
        "voice_profiles": {},
        "return_data_uri": True,
    }


def test_a_chosen_clone_is_keyed_the_way_host_roster_reads_it():
    """host_roster() looks up host_clone_<pair>_<slot> where slot is "a" or
    "b". Sending an index, or a bare profile id, attaches no clone at all --
    silently, because the host still speaks, just in the wrong voice."""
    out = _collect((
            [
                {"dataset": {"slot": "a"}, "value": "ada-a1b2c3"},
                {"dataset": {"slot": "b"}, "value": "rowan-d4e5f6"},
            ],
            """
_els['podcast-script'].value = 'x';
_els['podcast-pair'].value = 'duo_bright';
const __RESULT__ = collectPodcastRequest();
""",
        )
    )
    assert json.loads(out)["voice_profiles"] == {
        "host_clone_duo_bright_a": "ada-a1b2c3",
        "host_clone_duo_bright_b": "rowan-d4e5f6",
    }


def test_an_empty_clone_select_sends_nothing_rather_than_an_empty_id():
    out = _collect((
            [{"dataset": {"slot": "a"}, "value": ""}, {"dataset": {"slot": "b"}, "value": "only-b"}],
            "_els['podcast-script'].value='x';_els['podcast-pair'].value='duo_deep';const __RESULT__=collectPodcastRequest();",
        )
    )
    assert json.loads(out)["voice_profiles"] == {"host_clone_duo_deep_b": "only-b"}


def test_a_sting_prompt_is_included_only_when_typed():
    blank = _collect((
            [],
            "_els['podcast-script'].value='x';_els['podcast-sting'].value='   ';const __RESULT__=collectPodcastRequest();",
        )
    )
    assert "sting_prompt" not in json.loads(blank)

    typed = _collect((
            [],
            "_els['podcast-script'].value='x';_els['podcast-sting'].value=' warm motif ';"
            "const __RESULT__=collectPodcastRequest();",
        )
    )
    assert json.loads(typed)["sting_prompt"] == "warm motif"


def test_the_panel_asks_for_a_data_uri_so_the_player_never_carries_a_megabyte_of_base64_in_json():
    out = _collect([], "_els['podcast-script'].value='x';const __RESULT__=collectPodcastRequest();")
    assert json.loads(out)["return_data_uri"] is True


def test_the_duck_depth_is_a_number_not_a_string():
    """Flask does not coerce a form field here; a string would reach
    float() fine but would also make duck_db=0 indistinguishable from
    duck_db="" in the JSON the route stores."""
    out = _collect(([], "_els['podcast-script'].value='x';_els['podcast-duck'].value='7';const __RESULT__=collectPodcastRequest();"))
    assert json.loads(out)["duck_db"] == 7


# --------------------------------------------------------------------------
# podcastVoiceGender — the cross-gender clone guard
# --------------------------------------------------------------------------


def test_kokoro_prefixes_carry_the_gender_openvoice_needs():
    out = run_js(
        SLICE_PODCAST
        + """
const out = {};
for (const v of ['af_heart','am_michael','bf_emma','bm_george','af_sky'])
    out[v] = podcastVoiceGender(v);
log.push(JSON.stringify(out));
"""
    )
    assert json.loads(out) == {"af_heart": "f", "am_michael": "m", "bf_emma": "f", "bm_george": "m", "af_sky": "f"}


def test_an_unknown_or_blank_voice_has_no_gender_rather_than_guessing_one():
    out = run_js(
        SLICE_PODCAST
        + """
log.push(JSON.stringify([podcastVoiceGender(''), podcastVoiceGender('custom_profile'),
                         podcastVoiceGender(null), podcastVoiceGender('xx_thing')]));
"""
    )
    assert json.loads(out) == ["", "", "", ""]


def test_the_voice_gender_helper_matches_the_mixers_own_prefix_table():
    """A typo in either copy means the panel offers a cross-gender clone the
    mixer will then warn about at render time, which is a wasted render."""
    mixer = (ROOT / "web" / "podcast_mixer.py").read_text()
    body = mixer[matcher.start() : matcher.start() + 900] if (matcher := __import__("re").search(r"def voice_gender\(", mixer)) else ""
    for prefix in ("am", "bm", "af", "bf"):
        assert f'v.startswith("{prefix}")' in body, f"voice_gender no longer recognises {prefix}_*"
    assert 'return "m"' in body and 'return "f"' in body

    # And the JS agrees with it on the same four families.
    out = run_js(
        SLICE_PODCAST
        + """
const out = {};
for (const v of ['af_heart','am_michael','bf_emma','bm_george','a_f','xaf_heart'])
    out[v] = podcastVoiceGender(v);
log.push(JSON.stringify(out));
"""
    )
    assert json.loads(out) == {"af_heart": "f", "am_michael": "m", "bf_emma": "f", "bm_george": "m", "a_f": "", "xaf_heart": ""}


# --------------------------------------------------------------------------
# renderPodcastHosts — what the roster actually feeds
# --------------------------------------------------------------------------


def test_the_roster_is_filtered_by_pair_id_from_the_voices_route():
    """The roster is a flat list on /api/podcast/voices, not a dict keyed by
    pair on /api/podcast/status."""
    out = run_js(
        SLICE_PODCAST
        + """
_podcastVoices = { roster: [
    { pair_id: 'duo_warm', slot: 'a', name: 'Ada', voice: 'af_nicole', role: 'host' },
    { pair_id: 'duo_warm', slot: 'b', name: 'Rowan', voice: 'am_michael', role: 'cohost' },
    { pair_id: 'duo_deep', slot: 'a', name: 'Vera', voice: 'af_heart', role: 'host' },
]};
const r = podcastRosterFor('duo_warm');
log.push(JSON.stringify(r.map(h => h.name)));
"""
    )
    assert json.loads(out) == ["Ada", "Rowan"]


def test_an_unknown_pair_yields_an_empty_roster_rather_than_every_host():
    out = run_js(
        SLICE_PODCAST
        + """
_podcastVoices = { roster: [{ pair_id: 'duo_warm', slot: 'a', name: 'Ada' }] };
log.push(JSON.stringify(podcastRosterFor('nope')));
"""
    )
    assert json.loads(out) == []


def test_a_roster_with_no_voices_route_loaded_is_empty_and_does_not_throw():
    out = run_js(
        SLICE_PODCAST
        + """
_podcastVoices = null;
log.push(JSON.stringify(podcastRosterFor('duo_warm')));
"""
    )
    assert json.loads(out) == []


def test_a_cross_gender_clone_option_is_labelled_rather_than_silently_offered():
    """OpenVoice transfers timbre badly across genders, so the label is the
    only thing standing between the user and a two-minute render that sounds
    wrong."""
    out = _render_hosts(
        """
_podcastVoices = { saved_profiles: [{ id: 'p1', name: 'Mira', gender: 'f' }, { id: 'p2', name: 'Tom', gender: 'm' }],
                       roster: [
        { pair_id: 'duo_warm', slot: 'a', name: 'Ada', voice: 'af_nicole', role: 'host', source_gender: 'f' },
        { pair_id: 'duo_warm', slot: 'b', name: 'Rowan', voice: 'am_michael', role: 'cohost', source_gender: 'm' },
] };
_podcastStatus = { audio: { voices: ['af_nicole', 'am_michael'] } };
_els['podcast-pair'].value = 'duo_warm';
_els['podcast-duck'].value = '20';
renderPodcastHosts();
const __OUT__ = _els['podcast-hosts'].children[0].children[2].children.map(o => o.textContent);
"""
    )
    labels = json.loads(out)
    assert labels[0] == "No clone (Kokoro)"
    # Ada speaks af_nicole (f): Mira is same-gender, Tom is not.
    assert labels[1] == "Mira"
    assert "cross-gender" in labels[2] and "Tom" in labels[2]


def test_the_duck_slider_label_tracks_the_slider():
    out = _render_hosts(
        """
_podcastVoices = { roster: [] };
_podcastStatus = null;
_els['podcast-pair'].value = 'duo_warm';
_els['podcast-duck'].value = '9';
renderPodcastHosts();
const __OUT__ = _els['podcast-duck-val'].textContent;
"""
    )
    assert json.loads(out) == "9 dB"


def test_a_missing_roster_says_so_instead_of_leaving_an_empty_box():
    out = _render_hosts(
        """
_podcastVoices = { roster: [], saved_profiles: [] };
_podcastStatus = null;
_els['podcast-pair'].value = 'duo_warm';
_els['podcast-duck'].value = '20';
renderPodcastHosts();
const __OUT__ = _els['podcast-hosts'].children[0].textContent;
"""
    )
    assert "roster" in json.loads(out).lower()


# --------------------------------------------------------------------------
# No inline handlers, and the panel is a real form of input
# --------------------------------------------------------------------------


def test_the_podcast_panel_adds_no_inline_event_handlers():
    """The applyAnalysisRec ReferenceError (Phase 2) came from an inline
    onclick reaching for a closure-scoped function; nothing new should be
    wired that way."""
    start = HTML.index('id="view-podcast"')
    end = HTML.index('id="view-requests"')
    panel = HTML[start:end]
    assert "onclick=" not in panel
    assert "onchange=" not in panel


def test_the_panel_has_a_topic_box_a_script_box_and_both_buttons():
    start = HTML.index('id="view-podcast"')
    end = HTML.index('id="view-requests"')
    panel = HTML[start:end]
    for needed in (
        'id="podcast-topic"',
        'id="podcast-words"',
        'id="podcast-script"',
        'id="btn-podcast-draft"',
        'id="btn-podcast-render"',
        'id="podcast-pair"',
        'id="podcast-bed"',
        'id="podcast-duck"',
        'id="podcast-sting"',
        'id="podcast-player"',
        'id="podcast-download"',
        'id="podcast-warnings"',
    ):
        assert needed in panel, f"missing {needed}"


def test_the_script_placeholder_shows_the_tag_syntax_the_parser_resolves():
    panel = HTML[HTML.index('id="view-podcast"') : HTML.index('id="view-requests"')]
    assert "HOST A:" in panel and "HOST B:" in panel


def test_the_duck_slider_bounds_match_the_mixers_duck_range():
    mixer = (ROOT / "web" / "podcast_mixer.py").read_text()
    start = mixer.index("DEFAULT_DUCK_DB")
    assert '20.0' in mixer[start : start + 40]
    panel = HTML[HTML.index('id="view-podcast"') : HTML.index('id="view-requests"')]
    assert 'id="podcast-duck" min="6" max="30"' in panel
    assert 'value="20"' in panel
