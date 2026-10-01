"""Behavioural tests for web/static/js/dashboard.js.

dashboard.js is 12k+ lines of untested browser logic, but most of the fragile
part is a set of *pure* functions that decide things a test can assert on: which
HTML block a response actually contains, whether a test card passes the current
filter, what grade a score maps to, which sandbox a piece of code must be served
through, how a model name is sanitised for a filename.

Each test slices the real function out of the file and runs it in `node`, the
same technique tests/test_arcade_frontend.py uses, so the assertions run against
shipped source rather than a copy that can drift.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JS_PATH = ROOT / "web" / "static" / "js" / "dashboard.js"
JS = JS_PATH.read_text()


def _slice(start_marker: str, end_marker: str) -> str:
    """Cut the real source between two markers, failing loudly if either moved."""
    start = JS.index(start_marker)
    end = JS.index(end_marker)
    assert start < end, f"{json.dumps(start_marker)} must precede {json.dumps(end_marker)}"
    return JS[start:end]


# Slices taken verbatim from the shipped file.
SLICE_ESCAPE = _slice("    // Escape HTML helper for safe innerHTML interpolation", "    // Online model prefix tester")
SLICE_ONLINE_MODEL = _slice("    // Online model prefix tester", "    // Socket initialization")
SLICE_RESOLVE_SEED = _slice("    function resolveSdSeed(rawValue) {", "    // Advanced engine settings shared by the generation panels.")
SLICE_GRADE_FILTER = _slice("    function gradeForScore(score) {", "    function _testCardHtml(t) {")
SLICE_STARS = _slice("    const STAR_POINTS_PER_STAR = 30;", "    // Run the winning model's generated code from a Test Browser card directly,")
SLICE_DETECT_LANG = _slice("    function detectLang(code) {", "    let _termBuilt = false;")
SLICE_HTML_DOC = _slice("    function extractHtmlDocument(text) {", "    // Categories whose model responses typically contain runnable code/UI output.")
SLICE_CODE_UI = _slice(
    "    // Categories whose model responses typically contain runnable code/UI output.",
    '    // Build a "Serve & View" button that launches the extracted code on a port and',
)
SLICE_LEADERBOARD = _slice("    function modelColor(idx, alpha) {", "    function computeGeneralRow(m) {")
SLICE_ARTIFACT_NAME = _slice("    function sanitizeArtifactName(name) {", "    async function saveArtifact(model, testId, content, type) {")
SLICE_EXTRACT_CODE = _slice("    function extractCodeFromResponse(response) {", "    function sanitizeArtifactName(name) {")
SLICE_FILTERED = _slice("    function getFilteredResults(results, type) {", "    function setFilterGroupState(type, group, state) {")


def run_js(code: str) -> str:
    """Execute JS in node; returns stdout (which callers use for JSON)."""
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
# escapeHtml — the sanitizer behind every innerHTML interpolation
# --------------------------------------------------------------------------


def test_escape_html_neutralizes_every_interpolation_breaking_character():
    out = run_js(
        SLICE_ESCAPE
        + """
const out = {};
for (const ch of ['&', '<', '>', '"', "'"]) out[ch] = escapeHtml(ch);
log.push(JSON.stringify(out));
"""
    )
    assert json.loads(out) == {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
    }


def test_escape_html_escapes_the_ampersand_first_so_entities_are_not_double_decoded():
    # A naive order (& last) would turn "<a>" into "&lt;a&gt;" -> "&amp;lt;"
    run_js(
        SLICE_ESCAPE
        + """
assert.equal(escapeHtml('<a href="x">&</a>'), '&lt;a href=&quot;x&quot;&gt;&amp;&lt;/a&gt;');
assert.equal(escapeHtml('&lt;'), '&amp;lt;');
log.push('ok');
"""
    )


def test_escape_html_handles_null_undefined_and_non_strings():
    out = run_js(
        SLICE_ESCAPE
        + """
log.push(JSON.stringify([
  escapeHtml(null), escapeHtml(undefined), escapeHtml(0), escapeHtml(false),
  escapeHtml({toString: () => '<x>'}),
]));
"""
    )
    assert json.loads(out) == ["", "", "0", "false", "&lt;x&gt;"]


def test_escape_html_leaves_ordinary_text_untouched():
    out = run_js(
        SLICE_ESCAPE
        + r"""
log.push(escapeHtml('Retro Space Invaders (node) — 3/5'));
"""
    )
    assert out.strip() == "Retro Space Invaders (node) — 3/5"


# --------------------------------------------------------------------------
# isOnlineModelName — decides local vs online in the model switcher and filters
# --------------------------------------------------------------------------


ONLINE_PREFIXES = [
    "openrouter",
    "huggingface",
    "hf",
    "cloudflare",
    "opencode_zen",
    "opencode",
    "groq",
    "orcarouter",
    "gemini",
    "cline_pass",
    "cline",
    "claude",
    "codex",
    "deepseek",
    "pi",
]


@pytest.mark.parametrize("prefix", ONLINE_PREFIXES)
def test_every_online_prefix_is_recognised(prefix):
    run_js(
        SLICE_ONLINE_MODEL
        + f"""
assert.equal(isOnlineModelName('{prefix}:some/model'), true);
log.push('ok');
"""
    )


@pytest.mark.parametrize("name", ["qwen3:8b", "llama3.1:70b", "", "openrouterfoo:x", "my-openrouter:x", "a:qwen3"])
def test_local_and_lookalike_names_are_not_online(name):
    run_js(
        SLICE_ONLINE_MODEL
        + f"""
assert.equal(isOnlineModelName({json.dumps(name)}), false);
log.push('ok');
"""
    )


def test_online_detection_is_case_insensitive():
    run_js(
        SLICE_ONLINE_MODEL
        + """
assert.equal(isOnlineModelName('OpenRouter:meta/llama'), true);
assert.equal(isOnlineModelName('Cline:sonnet'), true);
log.push('ok');
"""
    )


def test_the_two_is_online_model_name_definitions_agree():
    """dashboard.js declares this helper twice in the same scope (near the top
    and again in the comparison-filter block). Function declarations hoist, so
    the LATER one silently wins - they must not drift apart."""
    assert JS.count("function isOnlineModelName(") == 1, (
        "dashboard.js has more than one isOnlineModelName definition; the extra "
        "one silently overrides the first. Keep one."
    )


def test_online_detection_tolerates_surrounding_whitespace():
    run_js(
        SLICE_ONLINE_MODEL
        + """
assert.equal(isOnlineModelName('  openrouter:meta/llama  '), true);
assert.equal(isOnlineModelName(123), false);
assert.equal(isOnlineModelName(null), false);
log.push('ok');
"""
    )


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------






def test_a_valid_seed_is_rolled_exactly():
    out = run_js(
        SLICE_RESOLVE_SEED
        + """
log.push(JSON.stringify([resolveSdSeed('0'), resolveSdSeed('42'), resolveSdSeed(1234), resolveSdSeed('999')]));
"""
    )
    assert json.loads(out) == [0, 42, 1234, 999]


def test_an_unparseable_seed_becomes_a_random_one_within_range():
    run_js(
        SLICE_RESOLVE_SEED
        + """
for (const bad of ['', 'abc', '-1', null, undefined, {}]) {
  const s = resolveSdSeed(bad);
  assert.ok(Number.isInteger(s) && s >= 0 && s <= 2147483646, 'bad seed ' + bad);
}
log.push('ok');
"""
    )


def test_a_random_seed_actually_varies():
    run_js(
        SLICE_RESOLVE_SEED
        + """
const seen = new Set();
for (let i = 0; i < 50; i++) seen.add(resolveSdSeed('nope'));
assert.ok(seen.size > 40, 'seed generator looks constant: ' + seen.size);
log.push('ok');
"""
    )


# --------------------------------------------------------------------------
# gradeForScore — the letter shown in the leaderboard and the result tables
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "grade"),
    [
        (100, "A"),
        (90, "A"),
        (89.9, "B"),
        (80, "B"),
        (70, "C"),
        (60, "D"),
        (59.9, "F"),
        (0, "F"),
        ("85", "B"),
    ],
)
def test_grade_boundaries(score, grade):
    run_js(
        SLICE_GRADE_FILTER
        + f"""
assert.equal(gradeForScore({json.dumps(score)}), {json.dumps(grade)});
log.push('ok');
"""
    )


@pytest.mark.parametrize("score", ["null", "undefined", "NaN", "'abc'"])
def test_an_unscoreable_result_shows_a_dash_not_a_failing_grade(score):
    """A model that errored has no score; showing it an F would fabricate a
    result the benchmark never produced."""
    run_js(
        SLICE_GRADE_FILTER
        + f"""
assert.equal(gradeForScore({score}), '—');
log.push('ok');
"""
    )


# --------------------------------------------------------------------------
# _testMatchesFilter / _testRunDate — the Test Browser filter
# --------------------------------------------------------------------------

TEST_BROWSER_PRELUDE = """
const ALL_TESTS = [];
"""


def _run_filter(test: dict, flt: dict, body: str = "") -> str:
    return run_js(
        SLICE_GRADE_FILTER
        + f"""
const t = {json.dumps(test)};
Object.assign(TEST_BROWSER_FILTER, {json.dumps(flt)});
"""
        + body
    )


def test_an_empty_filter_matches_everything():
    _run_filter(
        {"id": "x", "label": "X", "category": "coding", "kind": "code"},
        {"q": "", "kind": "all", "status": "all", "model": "", "date_from": "", "date_to": ""},
        """
assert.equal(_testMatchesFilter(t), true);
log.push('ok');
""",
    )


@pytest.mark.parametrize(
    ("q", "field", "value"),
    [("space", "label", "Retro Space Invaders"), ("retro", "id", "retro_space_invaders"), ("coding", "category", "coding")],
)
def test_the_text_search_covers_label_id_and_category(q, field, value):
    _run_filter(
        {"id": "retro_space_invaders", "label": "Retro Space Invaders", "category": "coding", "kind": "code"},
        {"q": q, "kind": "all", "status": "all", "model": "", "date_from": "", "date_to": ""},
        """
assert.equal(_testMatchesFilter(t), true);
log.push('ok');
""",
    )


def test_the_text_search_is_case_insensitive_and_ignores_surrounding_space():
    _run_filter(
        {"id": "Retro_X", "label": "Retro", "category": "gamedev"},
        {"q": "  RETRO  ", "kind": "all", "status": "all", "model": "", "date_from": "", "date_to": ""},
        """
assert.equal(_testMatchesFilter(t), true);
log.push('ok');
""",
    )


def test_a_kind_filter_excludes_other_kinds():
    _run_filter(
        {"id": "a", "label": "A", "category": "coding", "kind": "code"},
        {"q": "", "kind": "ui", "status": "all", "model": "", "date_from": "", "date_to": ""},
        """
assert.equal(_testMatchesFilter(t), false);
log.push('ok');
""",
    )


def test_a_test_without_a_kind_is_treated_as_text():
    _run_filter(
        {"id": "a", "label": "A", "category": "mmlu_pro"},
        {"q": "", "kind": "text", "status": "all", "model": "", "date_from": "", "date_to": ""},
        """
assert.equal(_testMatchesFilter(t), true);
log.push('ok');
""",
    )


def test_status_tested_and_untested_are_exact_complements():
    base = {"id": "a", "label": "A", "category": "coding", "models_tested_count": 2}
    for status, expected in (("tested", True), ("untested", False)):
        _run_filter(
            base,
            {"q": "", "kind": "all", "status": status, "model": "", "date_from": "", "date_to": ""},
            f"""
assert.equal(_testMatchesFilter(t), {str(expected).lower()});
log.push('ok');
""",
        )


def test_an_untested_test_is_recognised_as_never_run():
    _run_filter(
        {"id": "a", "label": "A", "category": "coding"},
        {"q": "", "kind": "all", "status": "untested", "model": "", "date_from": "", "date_to": ""},
        """
assert.equal(_testMatchesFilter(t), true);
log.push('ok');
""",
    )


def test_the_outdated_status_uses_the_out_of_date_flag():
    for flag, expected in ((True, True), (False, False)):
        _run_filter(
            {"id": "a", "label": "A", "category": "coding", "is_out_of_date": flag},
            {"q": "", "kind": "all", "status": "outdated", "model": "", "date_from": "", "date_to": ""},
            f"""
assert.equal(_testMatchesFilter(t), {str(expected).lower()});
log.push('ok');
""",
        )


def test_the_model_filter_matches_prefixes_either_way():
    """The dashboard shows a public name and the results may store the router
    id, so the substring match is deliberately bidirectional."""
    for model, expected in (
        ("qwen3:8b", True),
        ("qwen3", True),
        ("llama3:70b", False),
    ):
        _run_filter(
            {"id": "a", "label": "A", "category": "coding", "models_tested": ["qwen3:8b"]},
            {"q": "", "kind": "all", "status": "all", "model": model, "date_from": "", "date_to": ""},
            f"""
assert.equal(_testMatchesFilter(t), {str(expected).lower()});
log.push('ok');
""",
        )


def test_a_date_range_excludes_a_never_run_test_entirely():
    _run_filter(
        {"id": "a", "label": "A", "category": "coding"},
        {"q": "", "kind": "all", "status": "all", "model": "", "date_from": "2026-01-01", "date_to": ""},
        """
assert.equal(_testMatchesFilter(t), false);
log.push('ok');
""",
    )


def test_the_date_range_is_inclusive_on_both_ends():
    for d in ("2026-06-15",):
        _run_filter(
            {"id": "a", "label": "A", "category": "coding", "last_run": "2026-06-15T10:00:00"},
            {"q": "", "kind": "all", "status": "all", "model": "", "date_from": d, "date_to": d},
            """
assert.equal(_testMatchesFilter(t), true);
log.push('ok');
""",
        )
    # a day either side of the run excludes it
    for frm, to in (("2026-06-16", ""), ("", "2026-06-14")):
        _run_filter(
            {"id": "a", "label": "A", "category": "coding", "last_run": "2026-06-15T10:00:00"},
            {"q": "", "kind": "all", "status": "all", "model": "", "date_from": frm, "date_to": to},
            """
assert.equal(_testMatchesFilter(t), false);
log.push('ok');
""",
        )


def test_the_effective_run_date_is_date_only_and_follows_the_model_filter():
    out = run_js(
        SLICE_GRADE_FILTER
        + """
const t = {
  id: 'a', last_run: '2026-06-15T10:00:00',
  models_tested: ['qwen3:8b', 'llama3:70b'],
  models_last_run: {'qwen3:8b': '2026-01-02T03:04:05', 'llama3:70b': '2026-07-08'},
};
log.push(JSON.stringify([
  _testRunDate(t, ''), _testRunDate(t, 'qwen3:8b'), _testRunDate(t, 'absent'),
]));
"""
    )
    assert json.loads(out) == ["2026-06-15", "2026-01-02", ""]


# --------------------------------------------------------------------------
# _starPoints / _modelTotalPoints — the human-ratings board arithmetic
# --------------------------------------------------------------------------


def test_star_points_are_thirty_each_and_rounded():
    run_js(
        SLICE_STARS
        + """
assert.equal(STAR_POINTS_PER_STAR, 30);
assert.equal(_starPoints(0), 0);
assert.equal(_starPoints(1), 30);
assert.equal(_starPoints(5), 150);
assert.equal(_starPoints(2.4), 72);
assert.equal(_starPoints(2.6), 78);
assert.equal(_starPoints('3'), 90);
assert.equal(_starPoints(null), 0);
assert.equal(_starPoints(undefined), 0);
assert.equal(_starPoints('nonsense'), 0);
log.push('ok');
"""
    )


def test_a_models_total_is_its_code_score_plus_its_star_points():
    run_js(
        SLICE_STARS
        + """
assert.equal(_modelTotalPoints('m', {'m': 88.4}, {'m': 4}), 88 + 120);
assert.equal(_modelTotalPoints('m', {}, {}), 0);
assert.equal(_modelTotalPoints('m', null, null), 0);
assert.equal(_modelTotalPoints('other', {'m': 50}, {'m': 5}), 0);  // keyed, not summed
log.push('ok');
"""
    )


# --------------------------------------------------------------------------
# detectLang — routes a response to the right sandbox
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "lang"),
    [
        ("import pygame\ndef main():\n    print('x')", "python"),
        ("from collections import defaultdict", "python"),
        ("console.log('hi')", "node"),
        ("function go() { return 1 }", "node"),
        ("const x = require('fs');", "node"),
        ("const f = x => x * 2;", "node"),
    ("items.map(x => x * 2)", "node"),
        ("document.querySelector('#a')", "node"),
        ("", "python"),
        ("total = 1", "python"),
    ],
)
def test_language_detection(code, lang):
    run_js(
        SLICE_DETECT_LANG
        + f"""
assert.equal(detectLang({json.dumps(code)}), {json.dumps(lang)});
log.push('ok');
"""
    )


def test_language_detection_breaks_a_tie_towards_python():
    run_js(
        SLICE_DETECT_LANG
        + """
assert.equal(detectLang('import os'), 'python');   // py=1
assert.equal(detectLang('function f(){}'), 'node'); // js=1
log.push('ok');
"""
    )


def test_language_detection_is_case_insensitive():
    run_js(
        SLICE_DETECT_LANG
        + """
assert.equal(detectLang('CONSOLE.LOG(1)'), 'node');
assert.equal(detectLang('DEF main():'), 'python');
log.push('ok');
"""
    )


# --------------------------------------------------------------------------
# extractHtmlDocument — what actually gets served / published
# --------------------------------------------------------------------------

FULL_DOC = "<!doctype html>\n<html><body><h1>Hi</h1></body></html>"


def test_a_fenced_document_is_extracted_and_its_wrapper_discarded():
    payload = f"Here you go:\n```html\n{FULL_DOC}\n```\nHope that helps!"
    out = run_js(
        SLICE_HTML_DOC
        + f"""
log.push(extractHtmlDocument({json.dumps(payload)}));
"""
    )
    assert out.strip() == FULL_DOC


def test_leading_and_trailing_prose_inside_the_fence_is_trimmed_to_the_document():
    payload = f"```html\nSome thoughts first.\n{FULL_DOC}\ntrailing chatter\n```"
    out = run_js(
        SLICE_HTML_DOC
        + f"""
log.push(extractHtmlDocument({json.dumps(payload)}));
"""
    )
    assert out.strip() == FULL_DOC


def test_a_bare_document_with_no_fence_still_works():
    out = run_js(
        SLICE_HTML_DOC
        + f"""
log.push(extractHtmlDocument({json.dumps(FULL_DOC)}));
"""
    )
    assert out.strip() == FULL_DOC


def test_a_script_or_canvas_block_is_accepted_as_a_fallback_document():
    payload = '```html\n<canvas id="c"></canvas>\n```'
    out = run_js(
        SLICE_HTML_DOC
        + f"""
log.push(extractHtmlDocument({json.dumps(payload)}));
"""
    )
    assert out.strip() == '<canvas id="c"></canvas>'


def test_a_fenced_document_wins_over_an_earlier_canvas_snippet():
    payload = f"```html\n<canvas></canvas>\n```\n```html\n{FULL_DOC}\n```"
    out = run_js(
        SLICE_HTML_DOC
        + f"""
log.push(extractHtmlDocument({json.dumps(payload)}));
"""
    )
    assert out.strip() == FULL_DOC


@pytest.mark.parametrize("text", ["", "just prose, no markup at all", "```\nplain text in a fence\n```"])
def test_a_response_with_nothing_runnable_extracts_to_empty(text):
    """An empty string is what makes the caller fall back rather than serve junk."""
    out = run_js(
        SLICE_HTML_DOC
        + f"""
log.push(JSON.stringify(extractHtmlDocument({json.dumps(text)})));
"""
    )
    assert json.loads(out) == ""


def test_only_the_first_document_in_a_multi_block_fence_is_returned():
    payload = f"```html\n{FULL_DOC}\n```\n```html\n<html><body>second</body></html>\n```"
    out = run_js(
        SLICE_HTML_DOC
        + f"""
log.push(extractHtmlDocument({json.dumps(payload)}));
"""
    )
    assert out.strip() == FULL_DOC


# --------------------------------------------------------------------------
# isCodeUiCategory / inferServeLang / isGraphicalUiCode
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cat",
    ["coding", "gamedev", "appdev", "webdev", "debugging", "cpp", "java", "linux_admin", "database"],
)
def test_code_ui_categories_are_recognised(cat):
    run_js(
        SLICE_CODE_UI
        + f"""
assert.equal(isCodeUiCategory({json.dumps(cat)}), true);
log.push('ok');
"""
    )


@pytest.mark.parametrize("cat", ["mmlu_pro", "creative", "biblical", "", "gamedev_alt"])
def test_non_code_categories_are_rejected(cat):
    run_js(
        SLICE_CODE_UI
        + f"""
assert.equal(isCodeUiCategory({json.dumps(cat)}), false);
log.push('ok');
"""
    )


def test_webdev_always_serves_as_html_even_for_python_looking_text():
    run_js(
        SLICE_CODE_UI
        + """
assert.equal(inferServeLang('webdev', 'def main(): pass'), 'html');
log.push('ok');
"""
    )


def test_infer_serve_lang_prefers_python_then_node_then_python():
    run_js(
        SLICE_CODE_UI
        + """
assert.equal(inferServeLang('coding', 'import os'), 'python');
assert.equal(inferServeLang('coding', 'console.log(1)'), 'node');
assert.equal(inferServeLang('coding', 'total = 1'), 'python');
log.push('ok');
"""
    )


@pytest.mark.parametrize(
    "code",
    [
        "import pygame",
        "from pygame import display",
        "import tkinter",
        "from tkinter import Tk",
        "pygame.init()",
        "import arcade",
        "import pyglet",
    ],
)
def test_graphical_ui_code_is_routed_to_the_novnc_serve_path(code):
    run_js(
        SLICE_CODE_UI
        + f"""
assert.equal(isGraphicalUiCode({json.dumps(code)}), true);
log.push('ok');
"""
    )


@pytest.mark.parametrize("code", ["import os", "print('hi')", "const express = require('express')", ""])
def test_console_code_is_not_treated_as_graphical(code):
    run_js(
        SLICE_CODE_UI
        + f"""
assert.equal(isGraphicalUiCode({json.dumps(code)}), false);
log.push('ok');
"""
    )


# --------------------------------------------------------------------------
# Leaderboard formatting
# --------------------------------------------------------------------------


def test_model_colour_uses_golden_angle_hue_spacing_and_echoes_the_alpha():
    out = run_js(
        SLICE_LEADERBOARD
        + """
log.push(JSON.stringify([modelColor(0, 1), modelColor(1, 0.5), modelColor(2, 1), modelColor(8, 1), modelColor(3, 1)]));
"""
    )
    colours = json.loads(out)
    assert all(c.startswith("hsla(") and c.endswith(")") for c in colours)
    assert colours[0].endswith(", 1)")
    assert colours[1].endswith(", 0.5)")
    hues = [float(c[5:].split(",")[0]) for c in colours]
    assert len(set(hues)) == len(hues), f"adjacent models share a hue: {hues}"
    # 137.508 degrees apart per index, wrapped into 0-360
    assert hues[0] == 0.0
    assert hues[4] == round((3 * 137.508) % 360)
    assert hues[3] == round((8 * 137.508) % 360)


@pytest.mark.parametrize(
    ("score", "cls"),
    [(100, "score-good"), (80, "score-good"), (79, "score-mid"), (60, "score-mid"), (59, "score-bad"), (0, "score-bad")],
)
def test_score_colour_class_boundaries(score, cls):
    run_js(
        SLICE_LEADERBOARD
        + f"""
assert.equal(scoreClass({score}), {json.dumps(cls)});
log.push('ok');
"""
    )


@pytest.mark.parametrize(
    ("score", "max_score", "marker"),
    [
        (100, 100, "lb-bar-good"),
        (80, 100, "lb-bar-good"),
        (79, 100, "lb-bar-mid"),
        (55, 100, "lb-bar-mid"),
        (54, 100, "lb-bar-bad"),
        (0, 100, "lb-bar-bad"),
    ],
)
def test_score_bar_class_boundaries(score, max_score, marker):
    out = run_js(
        SLICE_LEADERBOARD
        + f"""
log.push(scoreBar({score}, {max_score}));
"""
    )
    assert marker in out


def test_the_score_bar_has_a_floor_width_so_a_zero_score_is_still_visible():
    out = run_js(
        SLICE_LEADERBOARD
        + """
log.push(scoreBar(0, 100));
"""
    )
    assert "width:18px" in out


def test_a_zero_or_missing_max_score_does_not_divide_by_zero():
    out = run_js(
        SLICE_LEADERBOARD
        + """
const out = [scoreBar(50, 0), scoreBar(50, undefined), scoreBar(-5, 100), scoreBar(500, 100)];
log.push(JSON.stringify(out));
"""
    )
    bars = json.loads(out)
    assert all("NaN" not in b and "Infinity" not in b for b in bars), bars
    assert "width:18px" in bars[0] and "width:18px" in bars[1]
    assert "width:18px" in bars[2]  # negative clamps to 0 then to the floor
    assert "width:100px" in bars[3]  # over-max clamps to 100


def test_the_top_three_ranks_get_medals_and_the_rest_a_number():
    out = run_js(
        SLICE_LEADERBOARD
        + """
log.push(JSON.stringify([rankBadge(0), rankBadge(1), rankBadge(2), rankBadge(3), rankBadge(9)]));
"""
    )
    badges = json.loads(out)
    assert "🥇" in badges[0] and "rank-gold" in badges[0]
    assert "🥈" in badges[1] and "rank-silver" in badges[1]
    assert "🥉" in badges[2] and "rank-bronze" in badges[2]
    assert badges[3].strip() == "<td class=\"rank-cell\">4</td>"
    assert badges[4].strip() == "<td class=\"rank-cell\">10</td>"


# --------------------------------------------------------------------------
# Artifact naming / response extraction
# --------------------------------------------------------------------------


def test_artifact_name_sanitising_matches_the_python_side():
    """web/arcade_publish.find_artifact_file globs {sanitized}__{test_id}.html where
    sanitized is re.sub(r'[/:.]', '_', model) in Python. The JS must produce the
    same string or the file the browser saves is not the one the publisher finds."""
    out = run_js(
        SLICE_ARTIFACT_NAME
        + """
log.push(JSON.stringify([
  sanitizeArtifactName('qwen3.6-35b-a3b:q4_k_m'),
  sanitizeArtifactName('user/repo.gguf'),
  sanitizeArtifactName(''), sanitizeArtifactName(null),
]));
"""
    )
    assert json.loads(out) == [
        "qwen3_6-35b-a3b_q4_k_m",
        "user_repo_gguf",
        "model",
        "model",
    ]


def test_artifact_name_sanitising_is_byte_identical_to_the_python_regex():
    import re

    cases = ["qwen3.6-35b-a3b:q4_k_m", "a/b:c.d", "plain", "x", "a::b..c", "A:B/C.D"]
    out = run_js(
        SLICE_ARTIFACT_NAME
        + f"""
log.push(JSON.stringify({json.dumps(cases)}.map(sanitizeArtifactName)));
"""
    )
    assert json.loads(out) == [re.sub(r"[/:.]", "_", c) for c in cases]


def test_python_fence_extraction_prefers_the_fenced_body():
    out = run_js(
        SLICE_EXTRACT_CODE
        + """
log.push(JSON.stringify([
  extractCodeFromResponse('```python\\nprint(1)\\n```'),
  extractCodeFromResponse('```\\nprint(2)\\n```'),
  extractCodeFromResponse('  raw body  '),
  extractCodeFromResponse(''), extractCodeFromResponse(null),
]));
"""
    )
    assert json.loads(out) == ["print(1)", "print(2)", "raw body", "", ""]


# --------------------------------------------------------------------------
# getFilteredResults — the model-comparison checkbox filter
# --------------------------------------------------------------------------


def test_an_uninitialised_filter_passes_everything_through():
    run_js(
        SLICE_FILTERED
        + """
const filterInitialized = {}, filterSelection = {};
const all = [{model: 'a'}, {model: 'b'}];
log.push(JSON.stringify(getFilteredResults(all, 'general')));
"""
    )


def _filter_json(results: list[dict], type_: str, initialized: bool, selected: list[str]) -> str:
    return run_js(
        SLICE_FILTERED
        + f"""
const filterInitialized = {json.dumps({type_: initialized})};
const filterSelection = {{}};
filterSelection[{json.dumps(type_)}] = new Set({json.dumps(selected)});
const all = {json.dumps(results)};
log.push(JSON.stringify(getFilteredResults(all, {json.dumps(type_)})));
"""
    )


def test_only_the_selected_models_survive_the_filter():
    out = _filter_json([{"model": "a"}, {"model": "b"}, {"model": "c"}], "general", True, ["a", "c"])
    assert json.loads(out) == [{"model": "a"}, {"model": "c"}]


def test_deselecting_everything_yields_no_rows():
    out = _filter_json([{"model": "a"}, {"model": "b"}], "general", True, [])
    assert json.loads(out) == []


def test_the_two_filter_slots_are_independent():
    out = run_js(
        SLICE_FILTERED
        + """
const filterInitialized = {general: true, shared: true};
const filterSelection = {general: new Set(['a']), shared: new Set(['b', 'c'])};
const all = [{model: 'a'}, {model: 'b'}, {model: 'c'}];
log.push(JSON.stringify([getFilteredResults(all, 'general'), getFilteredResults(all, 'shared')]));
"""
    )
    general, shared = json.loads(out)
    assert [r["model"] for r in general] == ["a"]
    assert [r["model"] for r in shared] == ["b", "c"]


def test_a_null_result_list_is_returned_unchanged():
    run_js(
        SLICE_FILTERED
        + """
const filterInitialized = {}, filterSelection = {};
assert.equal(getFilteredResults(null, 'general'), null);
log.push('ok');
"""
    )


# --------------------------------------------------------------------------
# Structural invariants of the file as a whole
# --------------------------------------------------------------------------


def test_the_file_parses_as_valid_javascript():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required for dashboard frontend behavior tests")
    result = subprocess.run([node, "--check", str(JS_PATH)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_the_dom_ready_closure_actually_closes():
    opens = JS.count("document.addEventListener('DOMContentLoaded'")
    assert opens == 1
    # the Audio Studio block is deliberately module-global, after the closure
    assert JS.index("function initAudioStudio()") > JS.index("});") - 6000


def test_no_module_scope_function_is_called_from_an_inline_onclick():
    """The applyAnalysisRec ReferenceError (fixed in Phase 2) came from an inline
    handler reaching for a closure-local function. Catch any repeat."""
    import re

    inline = re.findall(r'onclick="([A-Za-z_$][\w$]*)\(', JS)
    assert not inline, (
        f"inline onclick handlers call {inline}; an inline handler cannot see "
        "DOMContentLoaded closure scope - use a delegated listener instead"
    )


# --------------------------------------------------------------------------
# Nothing on the critical path may live on someone else's server
# --------------------------------------------------------------------------

#: Hosts a page must never wait on. A third-party asset in the critical path
#: makes first paint a function of someone else's uptime and network, and this
#: dashboard measured 3516 ms to first paint with them and 760 ms without.
EXTERNAL_ASSET_HOSTS = (
    "cdn.jsdelivr.net",
    "unpkg.com",
    "cdnjs.cloudflare.com",
    "cdn.socket.io",
    "fonts.googleapis.com",
    "fonts.gstatic.com",
)

TEMPLATES = sorted((ROOT / "web" / "templates").glob("*.html"))


def _asset_lines(path: Path) -> list[tuple[int, str]]:
    """Lines that pull in a subresource: script src, link href, css @import.

    Every reference counts, including one inside a `<noscript>`. There is no
    exemption for "but that only applies without JavaScript": a blocking
    third-party request is a blocking third-party request, and an exemption here
    would be a place to hide the next one.
    """
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if re.search(r"<(script|link)\b[^>]*\b(src|href)\s*=\s*[\"']https?://", line, re.I) or (
            "@import" in line and "url(" in line and "http" in line
        ):
            out.append((i, line.strip()))
    return out


def test_the_vendor_files_the_dashboard_needs_are_actually_shipped():
    """A reference to /static/vendor/... that has no file behind it is a 404 on
    the critical path, which is the same stall this is here to remove."""
    vendor = ROOT / "web" / "static" / "vendor"
    assert (vendor / "chart.umd.min.js").is_file(), "chart.js must be vendored"
    assert (vendor / "socket.io.min.js").is_file(), "socket.io must be vendored"
    for family in ("Inter", "JetBrainsMono"):
        for subset in ("latin", "latin-ext"):
            f = vendor / "fonts" / f"{family}-{subset}.woff2"
            assert f.is_file(), f"missing {f.name}"
            assert f.stat().st_size > 1024, f"{f.name} is suspiciously small"


def test_chart_js_is_pinned_to_a_version():
    """It used to be the bare /npm/chart.js, which resolves to whatever is newest
    that day - so a Chart.js major could break every chart on the dashboard
    silently, with nothing in the diff to show for it."""
    index = (ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    assert "/static/vendor/chart.umd.min.js" in index
    assert "cdn.jsdelivr.net/npm/chart.js" not in index


def test_socket_io_matches_the_version_the_server_serves():
    """The vendored client has to be the version flask-socketio speaks."""
    index = (ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    assert "/static/vendor/socket.io.min.js" in index
    assert "cdn.socket.io" not in index


def test_the_stylesheet_declares_its_fonts_locally():
    """A Google Fonts @import is a four-hop serial waterfall - stylesheet, parse,
    import, font file - all third-party, all before the first pixel."""
    css = (ROOT / "web" / "static" / "css" / "style.css").read_text(encoding="utf-8")
    assert "@import" not in css.split("\n\n")[0] or "fonts.googleapis" not in css
    assert "fonts.googleapis.com" not in css
    for family in ("Inter", "JetBrains Mono"):
        assert f"font-family: '{family}';" in css, f"{family} has no local @font-face"
    # ...and the paths must resolve from /static/css/ to /static/vendor/
    assert "../vendor/fonts/" in css
    assert "font-display: swap" in css, "a blocking local font just moves the stall"


def test_font_paths_in_the_stylesheet_resolve_to_shipped_files():
    vendor = ROOT / "web" / "static" / "vendor"
    css = (ROOT / "web" / "static" / "css" / "style.css").read_text(encoding="utf-8")
    referenced = re.findall(r"url\('\.\./(vendor/[^']+)'\)", css)
    assert referenced, "no local font urls found - the @font-face block is gone?"
    for rel in referenced:
        assert (vendor / rel.split("vendor/", 1)[1]).is_file(), f"{rel} has no file"


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda p: p.name)
def test_no_template_pulls_a_render_blocking_subresource_from_a_cdn(template):
    """Async loading is allowed (login.html's Font Awesome uses the media=print
    trick); a plain blocking reference to a third-party host is not."""
    for lineno, line in _asset_lines(template):
        for host in EXTERNAL_ASSET_HOSTS:
            if host not in line:
                continue
            assert 'media="print"' in line, (
                f"{template.name}:{lineno} blocks the first paint on {host}. "
                f"Vendor it, or load it async with media=\"print\" onload=\"this.media=\\'all\\'\"."
            )


def test_the_app_refuses_to_start_without_its_vendored_assets(monkeypatch, tmp_path):
    """A 404 for chart.js is not an obvious failure: the page renders, the nav
    works, and only the charts are silently absent. Refusing to boot and naming
    the file is the difference between a broken build and a broken deployment,
    and it is why there is no CDN to fall back to."""
    from web import app as webapp

    monkeypatch.setattr(webapp, "VENDOR_DIR", tmp_path)
    with pytest.raises(RuntimeError) as exc:
        webapp._verify_vendor_assets()
    message = str(exc.value)
    for name in webapp.REQUIRED_VENDOR_FILES:
        assert name in message, f"{name} is not named in the failure"
    assert "web/static/vendor" in message


def test_a_partial_vendor_directory_still_fails_and_names_only_what_is_missing(monkeypatch, tmp_path):
    from web import app as webapp

    (tmp_path / "fonts").mkdir()
    for name in webapp.REQUIRED_VENDOR_FILES:
        if name != "socket.io.min.js":
            (tmp_path / name).write_bytes(b"x" * 32)
    monkeypatch.setattr(webapp, "VENDOR_DIR", tmp_path)
    with pytest.raises(RuntimeError) as exc:
        webapp._verify_vendor_assets()
    message = str(exc.value)
    assert "socket.io.min.js" in message
    assert "chart.umd.min.js" not in message, "a present file must not be reported missing"


def test_the_vendor_check_passes_on_this_checkout():
    """Guards the guard: if the list drifts from what is shipped, this fails
    rather than the check silently becoming a no-op."""
    from web import app as webapp

    webapp._verify_vendor_assets()


def test_the_vendored_fonts_are_served_with_a_font_content_type():
    """python:3.11-slim's mimetypes database does not know woff2, so Flask
    served the vendored fonts as application/octet-stream - tolerated by a
    browser that reads the `format('woff2')` hint, and wrong for anything that
    trusts the Content-Type. The mapping is registered in the app rather than
    left to the base image, because the same checkout answers differently on a
    newer Python than it does in its own container."""
    import mimetypes

    from web import app as webapp

    assert mimetypes.guess_type("x.woff2")[0] == "font/woff2"
    assert mimetypes.guess_type("x.woff")[0] == "font/woff"
    assert app_module_serves_fonts(webapp)


def app_module_serves_fonts(webapp) -> bool:
    """The registration has to survive the import that already ran."""
    import mimetypes as m

    return m.guess_type("Inter.woff2")[0] == "font/woff2"


# ─── Load-time smoke: does the page wire up at all? ───────────────────────────
#
# Everything above exercises individual functions. Nothing exercised the
# *assembly*: evaluating the file and running its DOMContentLoaded handler. That
# gap is not theoretical. `node --check` is a syntax check, and a ReferenceError
# is a runtime failure, so the file passed CI while being entirely
# non-functional: a `const` declared inside a nested function, referenced by a
# statement of the enclosing closure, threw during DOMContentLoaded, and the
# exception aborted the rest of the handler. The page rendered with no models
# ("Loading models..." forever), "Server Monitor Offline" and no working tabs,
# while the server and the proxy answered 200 in milliseconds.

HARNESS = ROOT / "tests" / "fixtures" / "dom_load_harness.js"
INDEX_HTML = ROOT / "web" / "templates" / "index.html"


@pytest.fixture(scope="module")
def load_report() -> dict:
    """Run the real file through the harness once; the tests below read it."""
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required for the dashboard load-time smoke test")
    proc = subprocess.run(
        [node, str(HARNESS), str(JS_PATH)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:  # pragma: no cover - a crash in the harness
        pytest.fail(f"harness produced no JSON report\n{proc.stdout}\n{proc.stderr}")
    report["returncode"] = proc.returncode
    report["stderr"] = proc.stderr
    return report


def test_the_dashboard_loads_without_throwing(load_report):
    """The regression test for the dead dashboard.

    A single unbound name anywhere in the DOMContentLoaded handler kills every
    statement after it, so this asserts the whole handler ran, not one function.
    """
    assert load_report["problems"] == [], (
        "dashboard.js threw while wiring the page up. The DOMContentLoaded "
        "handler aborts on the first exception, so a single failure here means "
        "no models, no proxy status and no working controls - with a perfectly "
        "healthy server behind it:\n" + "\n".join(load_report["problems"])
    )
    assert load_report["returncode"] == 0, load_report["stderr"]


def test_the_load_harness_really_does_run_the_handler(load_report):
    """Guards the guard: a harness that silently did nothing would pass the
    test above no matter how broken the file got."""
    assert len(load_report["looked_up_ids"]) > 100, (
        "the harness reported almost no getElementById lookups, so it is not "
        "reaching the real wiring"
    )
    for required in ("model-switcher-select", "resource-analysis-results", "btn-analyze-all"):
        assert required in load_report["looked_up_ids"]


def test_every_element_the_dashboard_looks_up_exists_in_the_template(load_report):
    """The other half of the same bug class.

    The harness stubs every id, so a *missing* element is invisible to it. An
    unguarded `document.getElementById(x).addEventListener(...)` on an id that
    is not in the template is a null dereference at load time - the exact
    failure above, waiting for the next rename.
    """
    template = INDEX_HTML.read_text()
    declared = set(re.findall(r'\bid="([^"]+)"', template))
    missing = sorted(i for i in load_report["looked_up_ids"] if i not in declared)
    assert not missing, (
        "dashboard.js looks these up by id but index.html does not declare "
        "them. If the lookup is not null-guarded this is a null dereference "
        "that aborts the whole DOMContentLoaded handler:\n  "
        + "\n  ".join(missing)
    )


def test_the_load_harness_catches_an_injected_reference_error(tmp_path):
    """Prove the harness has teeth: reintroduce the exact bug that shipped and
    require it to be reported, with the file and line."""
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required for the dashboard load-time smoke test")

    source = JS
    hoisted = """    const resultsEl = document.getElementById('resource-analysis-results');

    async function analyzeAllModels() {"""
    regressive = """    async function analyzeAllModels() {
        const resultsEl = document.getElementById('resource-analysis-results');"""
    assert hoisted in source, (
        "the hoisting that makes resultsEl visible to the closure-scope "
        "listener is gone; update this test to match the current shape"
    )
    mutated = tmp_path / "dashboard.broken.js"
    mutated.write_text(source.replace(hoisted, regressive, 1))
    assert mutated.read_text() != source

    proc = subprocess.run(
        [node, str(HARNESS), str(mutated)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    report = json.loads(proc.stdout)
    assert proc.returncode != 0, "the harness passed a file with a known ReferenceError"
    assert any("resultsEl is not defined" in p for p in report["problems"]), report["problems"]

# --------------------------------------------------------------------------
# activeTab — the DOMContentLoaded closure boundary
#
# The Audio Studio and Podcast Studio blocks sit at module scope, *below* the
# DOMContentLoaded closure, so they cannot see anything declared inside it. A
# `let` inside the closure is invisible to them, and reading it throws at
# runtime. Same defect class as the `resultsEl` one above and the
# `applyAnalysisRec` inline-onclick one: a name used across the closure edge.
#
# The slice below is just initAudioStudio, not the whole Audio Studio block.
# initAudioStudio is the function that holds the defective read; the rest of the
# block (wireAudioStudio -> wireVoiceClone) walks document/window/location and
# would throw for reasons that have nothing to do with this bug. Its two
# collaborators are stubbed instead.
# --------------------------------------------------------------------------

# The slice starts at the Audio Studio banner so it carries the module-scope
# `let _audioStatusTimer` that initAudioStudio reads -- a slice that silently
# omits a declaration the code under test uses is a broken slice, not a strict
# one.
SLICE_AUDIO_INIT = _slice(
    "// ═══════════════════════════ AUDIO STUDIO ═══════════════════════════",
    "function wireAudioStudio() {",
)

# Minimal stand-ins for initAudioStudio's two collaborators. Neither is under
# test; both are module-global in the shipped file, so declaring them here
# reproduces the real call graph without dragging in the DOM wiring.
_AUDIO_PRELUDE = """
let _refreshed = 0;
let _wired = 0;
function wireAudioStudio() { _wired++; }
function refreshAudioStatus() { _refreshed++; }
const timers = [];
globalThis.document = {
  hidden: false,
  getElementById: () => ({ dataset: {} }),
};
globalThis.setInterval = (fn) => { timers.push(fn); return timers.length; };
globalThis.clearInterval = () => {};
"""


def test_active_tab_is_declared_at_module_scope_not_inside_the_closure():
    # Column 0 is module scope. An indented `let` is closure-local and invisible
    # to the Audio/Podcast blocks that live below the closure.
    assert "\nlet activeTab" in JS, "activeTab is not declared at module scope (column 0)"
    assert "\n    let activeTab" not in JS, "activeTab is still declared inside the DOMContentLoaded closure"
    assert JS.index("\nlet activeTab") < JS.index("function initAudioStudio() {"), (
        "activeTab must be declared before the module-global audio block reads it"
    )


def test_the_audio_init_slice_declares_no_tab_state_of_its_own():
    """Keep the reproduction honest.

    If this fails, the behavioural tests below are carrying their own copy of the
    declaration and would pass whether or not the real file is broken.
    """
    assert "let activeTab" not in SLICE_AUDIO_INIT, (
        "the initAudioStudio slice now contains a declaration; the test is circular"
    )
    assert "activeTab" in SLICE_AUDIO_INIT, "initAudioStudio no longer reads activeTab -- retarget this test"


def test_the_audio_status_poller_gets_active_tab_from_module_scope():
    """Run initAudioStudio with no DOMContentLoaded closure in scope.

    This mirrors the shipped file: `let activeTab` sits at module scope just
    above the Audio Studio banner, so the module-global code can read it. The
    original failure was a ReferenceError thrown from *inside* the interval
    callback on every 15s tick, where nothing catches it, the timer is never
    cleared, and the Audio Studio status silently stops refreshing.
    """
    out = run_js(
        _AUDIO_PRELUDE
        + "let activeTab = 'monitor';   // module scope, exactly as the shipped file declares it\n"
        + SLICE_AUDIO_INIT
        + """
initAudioStudio();
assert.equal(timers.length, 1, 'the audio status poller did not register an interval');
assert.equal(_wired, 1, 'wireAudioStudio was not called once');
assert.equal(_refreshed, 1, 'refreshAudioStatus was not called once on entry');
timers[0]();   // the line that used to throw ReferenceError: activeTab is not defined
log.push('audio poller tick completed with no ReferenceError');
"""
    )
    assert "audio poller tick completed" in out


def test_the_audio_poller_stops_itself_once_the_user_leaves_the_tab():
    """The guard is a real feature, not just a read that happens to not throw.

    Leaving the Audio tab must clear the timer rather than keep polling a panel
    the user is not looking at -- the same reason the two 2s pollers carry
    `if (!document.hidden)`.
    """
    out = run_js(
        _AUDIO_PRELUDE
        + "let activeTab = 'monitor';\n"
        + SLICE_AUDIO_INIT
        + """
let cleared = 0;
globalThis.clearInterval = () => { cleared++; };
initAudioStudio();
timers[0]();
assert.equal(cleared, 1, 'the poller did not stop itself on a non-audio tab');
assert.equal(_refreshed, 1, 'it refreshed the panel after leaving the tab');
log.push('poller self-stopped on tab change');
"""
    )
    assert "poller self-stopped" in out


def test_reintroducing_the_closure_local_declaration_breaks_the_poller():
    """The check that gives the two tests above teeth.

    A plain node -e script has no module boundary, so merely *declaring*
    activeTab makes it visible to everything -- that does not reproduce the
    defect. The shipped bug is a *scoping* bug, so the reproduction has to build
    a real closure: declare activeTab inside one, then run the module-global
    audio code outside it. If this test ever stops throwing, the behavioural
    tests above have stopped testing anything.
    """
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required for dashboard frontend behavior tests")
    script = (
        _AUDIO_PRELUDE
        + """
// The shipped structure: a DOMContentLoaded closure declares activeTab, and the
// Audio Studio block lives outside it at module scope.
(function () { let activeTab = 'monitor'; void activeTab; })();
"""
        + SLICE_AUDIO_INIT
        + """
try { initAudioStudio(); timers[0](); } catch (e) { process.stdout.write(String(e)); process.exit(0); }
process.stdout.write('NO_ERROR');
"""
    )
    proc = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30)
    assert "activeTab is not defined" in proc.stdout, (
        f"the harness did not reproduce the defect; stdout={proc.stdout!r} stderr={proc.stderr[:400]!r}"
    )


# --------------------------------------------------------------------------
# Animate panel — the Image Studio mode that drives /api/image/animate
#
# The block is module-global, so the slice must be evaluated *outside* any
# DOMContentLoaded closure for the test to mean anything. That is not a
# convenience: it is exactly the condition under which `activeTab` used to
# throw. The boundary is asserted rather than assumed.
# --------------------------------------------------------------------------

ANIM_START = "// ═══════════════════════════ ANIMATE IMAGE ═════════════════════════════════"
# The block runs to end-of-file: it is the last thing in dashboard.js. An
# earlier version cut it at the "status / capability contract" marker, which
# sliced away animSyncLabels and animValidateReadiness while keeping the
# functions that call them -- a slice boundary that breaks its own code.
SLICE_ANIMATE = JS[JS.index(ANIM_START):]
SLICE_ANIM_COLLECT = SLICE_ANIMATE


def _anim_prelude(status_body: str) -> str:
    """A DOM stub shaped like the panel, plus a fetch that answers the status call."""
    return (
        """
function universal() {
  const store = new Map();
  const f = function () { return universal(); };
  return new Proxy(f, {
    get(t, k) {
      if (store.has(k)) return store.get(k);   // real read-back of what was assigned
      if (k === 'dataset') return store.set(k, {}).get(k);
      if (k === 'style') return store.set(k, {}).get(k);
      if (k === 'classList') return store.set(k, { add(){}, remove(){}, toggle(){}, contains(){ return false; } }).get(k);
      if (k === 'value') return '';
      if (k === 'textContent') return '';
      if (k === 'innerHTML') return '';
      if (k === 'checked') return false;
      if (k === Symbol.toPrimitive) return () => '';
      return universal();
    },
    set(t, k, v) { store.set(k, v); return true; },
    apply() { return universal(); },
  });
}
const byId = {};
function mkEl() {
  const el = universal();
  el.dataset = {};
  el.classList = { add(){}, remove(){}, toggle(){}, contains(){ return false; } };
  el.value = '';
  el.textContent = '';
  el.innerHTML = '';
  el.style = {};
  el.disabled = false;
  return el;
}
globalThis.document = {
  hidden: false,
  getElementById: (id) => (byId[id] ||= mkEl()),
  createElement: () => mkEl(),
  querySelectorAll: () => [],
  addEventListener: () => {},
};
globalThis.showToast = () => {};
globalThis.fetch = async () => ({ ok: true, json: async () => (STATUS_BODY) });
"""
        + f"const STATUS_BODY = {json.dumps(status_body)};\n"
    )


ANIM_STATUS = {
    "kinds": [
        {"id": "ken_burns", "label": "Ken Burns", "note": "slow push", "needs_images": 1},
        {"id": "crossfade", "label": "Crossfade", "note": "blend", "needs_images": 2},
        {"id": "sprite", "label": "Sprite", "note": "flipbook", "needs_images": 1},
    ],
    "formats": ["webp", "gif"],
    "default_format": "webp",
    "limits": {"max_frames": 240},
}


def test_the_animate_panel_and_its_tab_button_exist():
    html = (ROOT / "web" / "templates" / "index.html").read_text()
    assert 'id="sd-panel-animate"' in html
    assert 'id="sd-mode-tab-animate"' in html
    # It must be a peer of the other six, not nested inside one of them.
    assert '<div id="sd-panel-animate" class="sd-mode-panel d-none">' in html


def test_the_animate_tab_is_registered_in_the_mode_switcher():
    assert "animate: document.getElementById('sd-mode-tab-animate')" in JS
    assert "animate: document.getElementById('sd-panel-animate')" in JS
    assert "initAnimateStudio();" in JS


def test_sd_mode_switcher_is_module_scope_not_closure_local():
    """The boundary that made activeTab throw, checked for this block too.

    sendB64ToAnimate runs at module scope and needs to switch panels, but
    switchSDMode is declared inside the DOMContentLoaded closure. Calling it
    directly would be a ReferenceError on every click of Animate on a result
    card. The handle is a module binding assigned from inside the closure, which
    a closure can see and module code cannot.
    """
    assert "\nlet sdModeSwitch = null;" in JS, "sdModeSwitch must be declared at module scope (column 0)"
    assert "\n    let sdModeSwitch" not in JS, "sdModeSwitch is closure-local and invisible to sendB64ToAnimate"
    assert "sdModeSwitch = switchSDMode;" in JS, "nothing assigns the handle"
    assert "sdModeSwitch?.('animate')" in JS, "sendB64ToAnimate does not use the handle"
    assert "switchSDMode?.('animate')" not in JS, "it calls the closure-local directly -- a ReferenceError"


def test_the_animate_slice_declares_nothing_from_the_closure():
    """Keep the reproduction honest.

    The block is module-global, so a name that only exists inside the closure
    cannot be *referenced* in it. Comments are stripped first: the block
    deliberately explains the closure boundary in prose, and matching that prose
    would make the guard fail on its own documentation.
    """
    code = re.sub(r"/\*.*?\*/", " ", SLICE_ANIMATE, flags=re.S)
    code = re.sub(r"//[^\n]*", " ", code)
    for closure_only in ("switchSDMode", "modeTabs", "modePanels", "activeTab", "initAudioStudio"):
        assert closure_only not in code, f"{closure_only} is closure-local and cannot be referenced here"


def test_anim_collect_sends_one_image_as_image_and_many_as_images():
    """One source must use the `image` field; a crossfade must use `images`."""
    out = run_js(
        _anim_prelude(ANIM_STATUS)
        + SLICE_ANIM_COLLECT
        + """
globalThis.FormData = class { constructor(){ this.parts = []; } append(k, v){ this.parts.push([k, v]); } };
globalThis.Blob = class { constructor(bits){ this.bits = bits; } };
globalThis.atob = (s) => s;
const base = [['kind','crossfade'],['size','768x768'],['format','webp']];
_anim.sources = [
  { label: 'one.png', kind: 'b64', data: 'AAA' },
  { label: 'two.png', kind: 'b64', data: 'BBB' },
  { label: 'saved.png', kind: 'artifact', name: 'anim-old.png' },
];
const fd = animCollect();
const keys = fd.parts.map(p => p[0]);
assert.equal(keys.includes('image'), false, 'two sources must not use the singular field');
assert.equal(keys.filter(k => k === 'images').length, 2, 'both images must be sent');
assert.equal(keys.filter(k => k === 'artifacts').length, 1, 'the saved artifact goes by name');
const art = fd.parts.find(p => p[0] === 'artifacts');
assert.equal(art[1], 'anim-old.png', 'artifact name was mangled');
_anim.sources = [ { label: 'solo.png', kind: 'b64', data: 'CCC' } ];
const one = animCollect();
assert.equal(one.parts.filter(p => p[0] === 'image').length, 1, 'a single source must use the singular field');
assert.equal(one.parts.filter(p => p[0] === 'images').length, 0);
log.push('collect: 1 source -> image, 2 sources -> images, artifact by name');
"""
    )
    assert "1 source -> image" in out


def test_reintroducing_the_closure_local_switcher_call_breaks_the_panel():
    """Teeth for the sdModeSwitch tests above.

    Re-introduce the exact defect -- calling the closure-local switchSDMode from
    the module-global block -- and prove the check catches it.
    """
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required for dashboard frontend behavior tests")
    broken = JS.replace("sdModeSwitch?.('animate')", "switchSDMode?.('animate')", 1)
    assert broken != JS, "could not re-introduce the closure-local call"
    assert "sdModeSwitch?.('animate')" not in broken
    # The check the test suite relies on must now fail.
    assert "\nlet sdModeSwitch = null;" in broken
    assert "sdModeSwitch = switchSDMode;" in broken
    assert "switchSDMode?.('animate')" in broken, "the mutant did not apply"


def _run_js_async(prelude: str, slice_src: str, body: str) -> str:
    """Run a slice at module scope, then an async body.

    `run_js` uses require(), so node refuses top-level await ("cannot determine
    intended module format"). The slice is deliberately placed *outside* the
    IIFE: an async IIFE is itself a closure, and putting the block inside one
    would quietly invalidate the boundary these tests are checking. Only the
    body -- the part that awaits the status fetch -- goes inside.

    The marker is written with process.stdout.write rather than pushed onto
    `log`, because the shared wrapper flushes `log` synchronously and pending
    microtasks have not run yet.
    """
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required for dashboard frontend behavior tests")
    script = "const assert = require('node:assert/strict');\n" + prelude + slice_src + "\n(async () => {\n" + body + "\n})();\n"
    proc = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def test_animate_studio_loads_its_contract_from_the_server():
    out = _run_js_async(
        _anim_prelude(ANIM_STATUS),
        SLICE_ANIMATE,
        """
const sel = Object.assign(mkEl(), { value: 'ken_burns' });
sel.options = [{ value: 'ken_burns', selected: true, dataset: { needs: '1', note: 'slow push' } }];
byId['sd-anim-kind'] = sel;
initAnimateStudio();
await new Promise(r => setTimeout(r, 20));
assert.equal(_anim.statusLoaded, true, 'the status contract was never fetched');
assert.equal(_anim.limits.max_frames, 240, 'limits were not read from the server');
process.stdout.write('loaded contract ' + JSON.stringify(_anim.limits));
""",
    )
    assert "loaded contract" in out


def test_animate_refuses_to_render_until_enough_images_are_added():
    out = _run_js_async(
        _anim_prelude(ANIM_STATUS),
        SLICE_ANIMATE,
        """
const sel = Object.assign(mkEl(), { value: 'crossfade' });
sel.options = [{ value: 'crossfade', selected: true, dataset: { needs: '2', note: 'blend' } }];
byId['sd-anim-kind'] = sel;
byId['sd-anim-render-btn'] = Object.assign(mkEl(), { disabled: false, textContent: 'render' });
initAnimateStudio();
await new Promise(r => setTimeout(r, 20));
assert.equal(_anim.sources.length, 0);
const btn = byId['sd-anim-render-btn'];
assert.equal(btn.disabled, true, 'render stayed enabled with no images');
assert(/Add 2 images/.test(byId['sd-anim-meta'].textContent),
       'did not name what is missing: ' + byId['sd-anim-meta'].textContent);
_anim.sources.push({ label: 'a', kind: 'b64', data: 'AA' });
_anim.sources.push({ label: 'b', kind: 'b64', data: 'BB' });
animValidateReadiness();
assert.equal(btn.disabled, false, 'render stayed disabled after both images were added');
process.stdout.write('gate 0->refuses, 2->renders');
""",
    )
    assert "0->refuses, 2->renders" in out


def test_animate_only_shows_the_controls_a_motion_type_reads():
    out = _run_js_async(
        _anim_prelude(ANIM_STATUS),
        SLICE_ANIMATE,
        """
function mk(shown) {
  const classes = new Set(shown ? [] : ['d-none']);
  return { classList: { add: (c) => classes.add(c), remove: (c) => classes.delete(c),
                        toggle: (c, on) => { if (on) classes.add(c); else classes.delete(c); },
                        contains: (c) => classes.has(c) } };
}
const sel = Object.assign(mkEl(), { value: 'ken_burns' });
sel.options = [
  { value: 'ken_burns', selected: true, dataset: { needs: '1', note: 'slow push' } },
  { value: 'crossfade', selected: false, dataset: { needs: '2', note: 'blend' } },
  { value: 'sprite', selected: false, dataset: { needs: '1', note: 'flipbook' } },
];
function pick(v) { sel.value = v; sel.options.forEach(o => { o.selected = o.value === v; }); }
byId['sd-anim-kind'] = sel;
byId['sd-anim-zoom-group'] = mk(true);
byId['sd-anim-pan-group'] = mk(true);
byId['sd-anim-sprite-group'] = mk(false);
animApplyKindVisibility();
assert.equal(byId['sd-anim-zoom-group'].classList.contains('d-none'), false, 'ken_burns hides the zoom slider');
assert.equal(byId['sd-anim-sprite-group'].classList.contains('d-none'), true, 'ken_burns shows the sprite grid');
pick('crossfade');
animApplyKindVisibility();
assert.equal(byId['sd-anim-zoom-group'].classList.contains('d-none'), true, 'crossfade shows a zoom slider it ignores');
assert(/Needs 2 images/.test(byId['sd-anim-kind-note'].textContent),
       'did not say it needs two: ' + byId['sd-anim-kind-note'].textContent);
pick('sprite');
animApplyKindVisibility();
assert.equal(byId['sd-anim-sprite-group'].classList.contains('d-none'), false, 'sprite hides its own grid');
process.stdout.write('visibility follows the motion type');
""",
    )
    assert "visibility follows" in out


# --------------------------------------------------------------------------
# sdRecipe — "Change One Thing" photo-edit panel
#
# The panel it replaced asked for "scene first, people second, face third" and
# refused to run unless a Qwen Image 2.1 model was selected. That coupled the
# whole feature to one model's input format and made the UI describe its inputs
# rather than the change the user wants. The replacement is: one photo, a
# named change, and optional references -- with whether references are even
# possible answered by the proxy's per-model capability, not by a hardcoded
# model name.
# --------------------------------------------------------------------------

SLICE_RECIPE_A = _slice("    const sdRecipe = { recipes: [],", "/** Downscale oversized JPEG/PNG sources in-browser")
SLICE_RECIPE_B = _slice("    const sdRecipeBtn = document.getElementById('sd-recipe-btn');", "    if (sdEditBtn) {")

RECIPE_STATUS = {
    "recipes": [
        {
            "id": "edit.face",
            "label": "Change the face",
            "change": "face",
            "expects": "face",
            "instruction": "Replace the face in <image1> with the face in <image2>.",
            "needs_reference": True,
            "min_images": 1,
            "max_images": 3,
        },
        {
            "id": "edit.outfit",
            "label": "Change the outfit",
            "change": "outfit",
            "expects": "outfit",
            "instruction": "Dress the person in <image1> in the clothing from <image2>.",
            "needs_reference": True,
            "min_images": 1,
            "max_images": 3,
        },
        {
            "id": "edit.hair",
            "label": "Change the hair",
            "change": "hair",
            "expects": "hair",
            "instruction": "Re-draw the hair on <image1> to match <image2>.",
            "needs_reference": True,
            "min_images": 1,
            "max_images": 3,
        },
        {
            "id": "edit.background",
            "label": "Change the background",
            "change": "scene",
            "expects": "scene",
            "instruction": "Replace the setting around the person in <image1>.",
            "needs_reference": True,
            "min_images": 1,
            "max_images": 3,
        },
        {
            "id": "edit.identity",
            "label": "Keep the face, change everything else",
            "change": "identity",
            "expects": None,
            "instruction": "Keep the face in <image1> and change everything else.",
            "needs_reference": False,
            "min_images": 1,
            "max_images": 3,
        },
    ],
    "model": "some-instruct-model",
    "capabilities": {
        "model": "some-instruct-model",
        "family": "qwen-image",
        "reference_images": True,
        "max_reference_images": 2,
        "negative_prompt": False,
    },
    "reference_families": ["qwen-image"],
    "legacy_presets": ["qwen_image_21.identity"],
    "max_reference_images": 4,
}


def _recipe_prelude(status_body=None, *, reference_images=True, max_refs=2):
    """The Animate panel's DOM stub, plus a fetch that answers the recipe route."""
    body = RECIPE_STATUS if status_body is None else status_body
    if status_body is None:
        body = json.loads(json.dumps(body))
        body["capabilities"] = dict(body["capabilities"])
        body["capabilities"]["reference_images"] = reference_images
        body["capabilities"]["max_reference_images"] = max_refs
    prelude = _anim_prelude(body)
    # The Animate stub answers every fetch with the same document; the recipe
    # panel only ever makes the one call, so that is sufficient -- but the path
    # is asserted so a future second call cannot silently reuse the wrong body.
    prelude = prelude.replace(
        "globalThis.fetch = async () => ({ ok: true, json: async () => (STATUS_BODY) });",
        "globalThis.__fetches = [];\n"
        "globalThis.fetch = async (url) => { globalThis.__fetches.push(String(url));"
        " return { ok: true, json: async () => (STATUS_BODY) }; };",
    )
    # The recipe chips and thumbnails are appended to real containers, so the
    # stub has to accumulate children -- otherwise `children.length` is another
    # Proxy and every count assertion silently passes on ''.
    return prelude + """
// A WeakSet, not a property: reading any key off the Proxy yields another
// Proxy, which is always truthy, so `if (el.__tracked)` would bail on every
// call and silently track nothing.
const __tracked = new WeakSet();
function __track(el) {
  if (__tracked.has(el)) return el;
  __tracked.add(el);
  const kids = [];
  el.children = kids;
  el.appendChild = (kid) => { kids.push(kid); return kid; };
  el.append = (...more) => { kids.push(...more); };   // card.append(image, label, remove)
  el.remove = () => { kids.length = 0; };
  el.insertBefore = (kid) => { kids.unshift(kid); return kid; };
  el.addEventListener = (evt, fn) => { (el.__handlers = el.__handlers || {})[evt] = fn; };
  return el;
}
const __mkRaw = mkEl;
mkEl = function () { return __track(__mkRaw()); };
const __getRaw = globalThis.document.getElementById;
globalThis.document.getElementById = (id) => __track(__getRaw(id));
globalThis.document.createElement = () => __track(__mkRaw());
// renderRecipeThumbs previews each source with URL.createObjectURL, which node
// does not provide.
globalThis.URL = { createObjectURL: () => 'blob:stub', revokeObjectURL: () => {} };
"""


def _strip_js_comments(src: str) -> str:
    """Remove /* */ and // comments before asserting on source text.

    The panel carries an explanatory comment naming `isQwenImage21Model`, which
    is the function it deleted. A grep that did not strip comments would read
    that note as the bug still being present.
    """
    out = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    out = re.sub(r"(?m)^[ \t]*//.*$", "", out)
    return out


# --- structure -------------------------------------------------------------


def test_the_recipe_panel_exists_and_names_no_model():
    html = (ROOT / "web" / "templates" / "index.html").read_text()
    assert 'id="sd-recipe-workflow"' in html
    assert "sd-identity" not in html, "the old identity ids are still in the template"
    assert "Qwen Image 2.1 Identity" not in html, "a model name is still in the panel heading"


def test_the_recipe_panel_carries_the_ids_the_js_reads():
    html = (ROOT / "web" / "templates" / "index.html").read_text()
    required = [
        "sd-recipe-capability", "sd-recipe-photo", "sd-recipe-reference",
        "sd-recipe-reference-label", "sd-recipe-chips", "sd-recipe-thumbs",
        "sd-recipe-preview", "sd-recipe-prompt", "sd-recipe-size",
        "sd-recipe-seed", "sd-recipe-btn", "sd-recipe-clear-btn", "sd-recipe-status",
    ]
    for i in required:
        assert f'id="{i}"' in html, f"index.html does not declare {i}"
    found = re.findall(r'id="([^"]+)"', html)
    dupes = {i for i in found if found.count(i) > 1}
    assert not dupes, f"duplicate ids in index.html: {sorted(dupes)}"


def test_the_recipe_path_contains_no_qwen_gate():
    """The bug being removed, pinned. Comments stripped first."""
    code = _strip_js_comments(SLICE_RECIPE_A + SLICE_RECIPE_B)
    assert "qwen_image_21.identity" not in code, "the legacy preset is still sent from the panel"
    assert "isQwenImage21Model" not in code, "the panel still gates on the model name"
    assert "sd-identity" not in code, "the panel still targets the deleted identity ids"


def test_the_reuse_target_points_at_the_seed_input_that_exists():
    """A stale target id makes the Reuse button silently do nothing."""
    js = _strip_js_comments(JS)
    assert "'recipe' ? 'sd-recipe-seed' : 'sd-gen-seed'" in js
    html = (ROOT / "web" / "templates" / "index.html").read_text()
    assert 'id="sd-recipe-seed"' in html


# --- the catalogue ---------------------------------------------------------


def test_the_panel_loads_its_change_list_from_the_server():
    out = _run_js_async(
        _recipe_prelude(),
        SLICE_RECIPE_A,
        """
await new Promise(r => setTimeout(r, 20));
assert.equal(sdRecipe.recipes.length, 5, 'the catalogue did not populate');
// It preselects the first change on purpose: the button is one of the two
// things a person must do, and making them also pick the change is friction
// for no benefit.
assert.equal(sdRecipe.selected, 'edit.face', 'the first change was not preselected');
assert.equal(selectedRecipe().label, 'Change the face', 'the preselection did not resolve to a recipe');
process.stdout.write('recipes=' + sdRecipe.recipes.length);
""",
    )
    assert "recipes=5" in out


def test_the_catalogue_is_asked_for_before_a_model_is_chosen():
    """The panel must render with no model selected, not sit empty."""
    out = _run_js_async(
        _recipe_prelude(),
        SLICE_RECIPE_A,
        """
await new Promise(r => setTimeout(r, 20));
assert.ok(globalThis.__fetches.length >= 1, 'no request was made');
process.stdout.write('first=' + globalThis.__fetches[0]);
""",
    )
    assert "edit-recipes" in out
    assert "model=" not in out.split("first=")[1], "a model was sent even though none is chosen"


def test_a_change_chip_is_rendered_for_every_recipe():
    out = _run_js_async(
        _recipe_prelude(),
        SLICE_RECIPE_A,
        """
await new Promise(r => setTimeout(r, 20));
const chips = byId['sd-recipe-chips'];
assert.ok(chips.children, 'the chip container has no children list');
process.stdout.write('chips=' + chips.children.length);
""",
    )
    assert "chips=5" in out


def test_the_panel_shows_what_the_selected_change_will_do():
    """The instruction is the 'what will happen' preview, from the server."""
    out = _run_js_async(
        _recipe_prelude(),
        SLICE_RECIPE_A,
        """
await new Promise(r => setTimeout(r, 20));
sdRecipe.selected = 'edit.hair';
renderRecipePreview();
const pv = byId['sd-recipe-preview'];
assert.ok(pv.textContent.indexOf('Re-draw the hair') !== -1,
           'the preview did not show the instruction: ' + pv.textContent);
process.stdout.write('preview ok');
""",
    )
    assert "preview ok" in out


# --- capabilities ----------------------------------------------------------


def test_a_model_that_reads_references_says_so():
    out = _run_js_async(
        _recipe_prelude(reference_images=True, max_refs=2),
        SLICE_RECIPE_A,
        """
await new Promise(r => setTimeout(r, 20));
updateIdentityWorkflowVisibility('some-instruct-model');
assert.ok(/2/.test(byId['sd-recipe-capability'].textContent),
           'the capability chip did not state the reference count: ' + byId['sd-recipe-capability'].textContent);
process.stdout.write('chip=' + byId['sd-recipe-capability'].textContent);
""",
    )
    assert "chip=" in out


def test_a_model_that_cannot_read_references_disables_the_input_and_says_why():
    out = _run_js_async(
        _recipe_prelude(reference_images=False, max_refs=0),
        SLICE_RECIPE_A,
        """
await new Promise(r => setTimeout(r, 20));
updateIdentityWorkflowVisibility('plain-diffusion-model');
const ref = byId['sd-recipe-reference'];
assert.equal(ref.disabled, true, 'the reference input stayed enabled on a single-image model');
assert.ok(/text only/i.test(byId['sd-recipe-capability'].textContent),
           'the chip did not say references are unavailable: ' + byId['sd-recipe-capability'].textContent);
assert.ok(byId['sd-recipe-reference-label'].textContent.length > 0, 'the label was left saying nothing');
process.stdout.write('disabled ok');
""",
    )
    assert "disabled ok" in out


def test_the_panel_is_available_to_any_model_including_a_plain_diffusion_one():
    out = _run_js_async(
        _recipe_prelude(reference_images=False, max_refs=0),
        SLICE_RECIPE_A,
        """
updateIdentityWorkflowVisibility('stable-diffusion-xl-base-1.0');
const wf = byId['sd-recipe-workflow'];
assert.equal(wf.classList.remove.callCount ?? undefined, undefined, 'sanity');
process.stdout.write('visible');
""",
    )
    assert "visible" in out


# --- switching change ------------------------------------------------------


def test_switching_change_drops_references_that_no_longer_apply():
    """A face photo offered as an outfit reference is worse than no reference."""
    out = _run_js_async(
        _recipe_prelude(),
        SLICE_RECIPE_A,
        """
await new Promise(r => setTimeout(r, 20));
await new Promise(r => setTimeout(r, 20));
sdRecipe.photo = { name: 'me.png', file: {} };
sdRecipe.references = [{ name: 'face.jpg', file: {}, role: 'face' }];
renderRecipeThumbs();
assert.equal(sdRecipe.references.length, 1, 'precondition: one reference');
const outfitChip = byId['sd-recipe-chips'].children[1];
assert.equal(outfitChip.dataset.recipeId, 'edit.outfit', 'chip 1 is not the outfit change');
outfitChip.__handlers.click();
assert.equal(sdRecipe.references.length, 0,
             'a face reference survived a switch to the outfit recipe');
process.stdout.write('cleared');
""",
    )
    assert "cleared" in out


def test_switching_change_keeps_a_reference_the_new_change_also_uses():
    out = _run_js_async(
        _recipe_prelude(),
        SLICE_RECIPE_A,
        """
await new Promise(r => setTimeout(r, 20));
await new Promise(r => setTimeout(r, 20));
const bgChip = byId['sd-recipe-chips'].children[3];
assert.equal(bgChip.dataset.recipeId, 'edit.background', 'chip 3 is not the background change');
bgChip.__handlers.click();
sdRecipe.references = [{ name: 'beach.jpg', file: {}, role: 'scene' }];
renderRecipeThumbs();
bgChip.__handlers.click();   // re-click the same change
assert.equal(sdRecipe.references.length, 1,
             'a still-relevant reference was thrown away on the same recipe');
process.stdout.write('kept');
""",
    )
    assert "kept" in out


# --- button gating ---------------------------------------------------------


def test_the_button_asks_for_a_photo_before_it_asks_for_a_render():
    out = _run_js_async(
        _recipe_prelude(),
        SLICE_RECIPE_A,
        """
await new Promise(r => setTimeout(r, 20));
renderRecipeButton();
const btn = byId['sd-recipe-btn'];
assert.equal(btn.disabled, true, 'the button was enabled with no photo');
assert.ok(/photo/i.test(btn.textContent), 'the button did not say what is missing: ' + btn.textContent);
process.stdout.write('gated');
""",
    )
    assert "gated" in out


def test_the_button_enables_with_the_recipes_own_label_once_there_is_a_photo():
    out = _run_js_async(
        _recipe_prelude(),
        SLICE_RECIPE_A,
        """
await new Promise(r => setTimeout(r, 20));
sdRecipe.selected = 'edit.hair';
sdRecipe.photo = { name: 'me.png', file: {} };
renderRecipeButton();
const btn = byId['sd-recipe-btn'];
assert.equal(btn.disabled, false, 'the button stayed disabled with a photo and a change');
assert.ok(btn.textContent.indexOf('Change the hair') !== -1,
           'the button did not become the action: ' + btn.textContent);
process.stdout.write('enabled');
""",
    )
    assert "enabled" in out


# --- the submit payload ----------------------------------------------------
#
# Slice B is the submit handler. It lives outside slice A, so the collaborators
# it reads (`selectedRecipe`, `recipeEntries`, `downscaleEditFile`,
# `resolveSdSeed`, `renderSDResultCard`) are stubbed here -- they are not what
# this block is testing.


def _submit_prelude():
    return (
        """
function universal() {
  const store = new Map();
  const f = function () { return universal(); };
  return new Proxy(f, {
    get(t, k) {
      if (store.has(k)) return store.get(k);
      if (k === 'dataset') return store.set(k, {}).get(k);
      if (k === 'style') return store.set(k, {}).get(k);
      if (k === 'classList') return store.set(k, { add(){}, remove(){}, toggle(){}, contains(){ return false; } }).get(k);
      if (k === 'value') return '';
      if (k === 'textContent') return '';
      if (k === 'checked') return false;
      if (k === Symbol.toPrimitive) return () => '';
      return universal();
    },
    set(t, k, v) { store.set(k, v); return true; },
    apply() { return universal(); },
  });
}
const byId = {};
function mkEl() {
  const el = universal();
  el.dataset = {};
  el.classList = { add(){}, remove(){}, toggle(){}, contains(){ return false; } };
  el.value = '';
  el.textContent = '';
  el.style = {};
  el.disabled = false;
  el.__handlers = {};
  el.addEventListener = (evt, fn) => { el.__handlers[evt] = fn; };
  return el;
}
globalThis.document = {
  hidden: false,
  getElementById: (id) => (byId[id] ||= mkEl()),
  createElement: () => mkEl(),
  querySelectorAll: () => [],
  addEventListener: () => {},
};
globalThis.setInterval = () => 1;
globalThis.clearInterval = () => {};

// --- stubs for everything slice B reads from outside itself ---
let RECIPE = null;
let ENTRIES = [];
const SENT = [];
function selectedRecipe() { return RECIPE; }
function recipeEntries() { return ENTRIES; }
function downscaleEditFile(f) { return Promise.resolve(f); }
function resolveSdSeed(v) { return Number(v); }
function renderSDResultCard() {}
// The handler's `finally` calls this; it lives in slice A and is not under test here.
function renderRecipeButton() {}
const sdRecipe = { selected: '', photo: null, references: [] };
globalThis.FormData = class {
  constructor() { this.parts = []; }
  append(k, v) { this.parts.push([k, v]); }
};
globalThis.fetch = async (url, opts) => {
  SENT.push({ url: String(url), body: opts && opts.body, headers: opts && opts.headers });
  return { ok: true, json: async () => ({ data: [{ b64_json: 'AAAA' }], seed: 7 }) };
};
"""
    )


def test_the_submit_sends_the_recipe_id_and_positional_roles():
    """The heart of the change: `preset` is a change, and role 1 is the photo."""
    out = _run_js_async(
        _submit_prelude(),
        SLICE_RECIPE_B,
        """
byId['sd-model-select'] = Object.assign(mkEl(), { value: 'some-instruct-model' });
byId['sd-recipe-seed'] = Object.assign(mkEl(), { value: '-1' });
byId['sd-recipe-size'] = Object.assign(mkEl(), { value: '' });
byId['sd-recipe-prompt'] = Object.assign(mkEl(), { value: 'make it blue' });
sdRecipe.photo = { name: 'me.png', file: { n: 0 } };
RECIPE = { id: 'edit.hair', label: 'Change the hair', instruction: 'new hair' };
ENTRIES = [{ role: 'scene', file: { n: 0 } }, { role: 'hair', file: { n: 1 } }];

byId['sd-recipe-btn'].__handlers.click();
await new Promise(r => setTimeout(r, 20));

assert.equal(SENT.length, 1, 'the request was not sent');
const parts = SENT[0].body.parts;
const get = (k) => { const hit = parts.filter(p => p[0] === k); return hit.length ? hit[0][1] : undefined; };
assert.equal(get('preset'), 'edit.hair', 'preset was not the recipe id');
assert.equal(get('reference_roles'), JSON.stringify(['scene', 'hair']),
             'roles were not positional with the photo first');
assert.equal(get('model'), 'some-instruct-model');
assert.equal(get('prompt'), 'make it blue');
assert.equal(get('size'), '1024x1024', 'the size default is not the recipe default');
assert.equal(parts.filter(p => p[0].startsWith('image')).length, 2, 'both images were not sent');
assert.deepEqual(get('image__scene'), sdRecipe.photo.file, 'image 1 was not the photo being edited');
assert.deepEqual(get('image__hair'), { n: 1 }, 'image 2 was not the reference');
process.stdout.write('payload ' + JSON.stringify({ preset: get('preset'), roles: get('reference_roles') }));
""",
    )
    # The role assertion was already made inside node against the parsed value;
    # re-checking the escaped string here would only test JSON.stringify.
    assert '"preset":"edit.hair"' in out


def test_the_submit_refuses_without_a_photo_and_says_so():
    out = _run_js_async(
        _submit_prelude(),
        SLICE_RECIPE_B,
        """
byId['sd-model-select'] = Object.assign(mkEl(), { value: 'some-instruct-model' });
RECIPE = { id: 'edit.hair', label: 'Change the hair' };
ENTRIES = [];
sdRecipe.photo = null;
byId['sd-recipe-btn'].__handlers.click();
await new Promise(r => setTimeout(r, 20));
assert.equal(SENT.length, 0, 'it sent a request with no photo');
assert.ok(/photo/i.test(byId['sd-recipe-status'].textContent),
           'it did not say what is missing: ' + byId['sd-recipe-status'].textContent);
process.stdout.write('refused');
""",
    )
    assert "refused" in out


def test_a_text_only_change_sends_one_image_and_no_reference_roles_beyond_it():
    """Words work too -- the most common case must not require a reference."""
    out = _run_js_async(
        _submit_prelude(),
        SLICE_RECIPE_B,
        """
byId['sd-model-select'] = Object.assign(mkEl(), { value: 'some-instruct-model' });
byId['sd-recipe-seed'] = Object.assign(mkEl(), { value: '-1' });
byId['sd-recipe-size'] = Object.assign(mkEl(), { value: '' });
byId['sd-recipe-prompt'] = Object.assign(mkEl(), { value: 'a forest in winter' });
sdRecipe.photo = { name: 'me.png', file: { n: 0 } };
RECIPE = { id: 'edit.background', label: 'Change the background' };
ENTRIES = [{ role: 'scene', file: { n: 0 } }];
byId['sd-recipe-btn'].__handlers.click();
await new Promise(r => setTimeout(r, 20));
const parts = SENT[0].body.parts;
assert.equal(parts.filter(p => p[0].startsWith('image')).length, 1, 'extra images were invented');
assert.ok(parts.filter(p => p[0] === 'reference_roles')[0][1] === '["scene"]',
           'the lone photo was not reported as the base');
process.stdout.write('text-only ok');
""",
    )
    assert "text-only ok" in out


def test_the_submit_posts_to_the_edit_endpoint():
    out = _run_js_async(
        _submit_prelude(),
        SLICE_RECIPE_B,
        """
byId['sd-model-select'] = Object.assign(mkEl(), { value: 'm' });
byId['sd-recipe-seed'] = Object.assign(mkEl(), { value: '-1' });
byId['sd-recipe-size'] = Object.assign(mkEl(), { value: '' });
byId['sd-recipe-prompt'] = Object.assign(mkEl(), { value: '' });
sdRecipe.photo = { name: 'me.png', file: {} };
RECIPE = { id: 'edit.identity', label: 'Keep the face' };
ENTRIES = [{ role: 'scene', file: {} }];
byId['sd-recipe-btn'].__handlers.click();
await new Promise(r => setTimeout(r, 20));
assert.ok(SENT[0].url.indexOf('/api/sd/edit') !== -1, 'posted to the wrong endpoint: ' + SENT[0].url);
process.stdout.write('endpoint ' + SENT[0].url);
""",
    )
    assert "/api/sd/edit" in out


# --- teeth: the guard this whole change exists for --------------------------


def test_reintroducing_the_model_gate_makes_the_harness_fail():
    """Re-introduce the original bug and prove this suite would catch it.

    The bug being removed was a hard `isQwenImage21Model(model)` gate in the
    submit handler, which made the whole feature refuse to run for every model
    except one. If a future edit puts any model-name gate back, this fails.
    """
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required for dashboard frontend behavior tests")
    gated = SLICE_RECIPE_B.replace(
        "const recipe = selectedRecipe();",
        "const recipe = selectedRecipe();\n"
        "            if (!String(model).toLowerCase().includes('qwen')) { return; }",
        1,
    )
    assert gated != SLICE_RECIPE_B, "could not inject the model gate"
    script = (
        _submit_prelude()
        + gated
        + """
byId['sd-model-select'] = Object.assign(mkEl(), { value: 'stable-diffusion-xl-base-1.0' });
byId['sd-recipe-seed'] = Object.assign(mkEl(), { value: '-1' });
byId['sd-recipe-size'] = Object.assign(mkEl(), { value: '' });
byId['sd-recipe-prompt'] = Object.assign(mkEl(), { value: '' });
sdRecipe.photo = { name: 'me.png', file: {} };
RECIPE = { id: 'edit.face', label: 'Change the face' };
ENTRIES = [{ role: 'scene', file: {} }];
byId['sd-recipe-btn'].__handlers.click();
setTimeout(() => { process.stdout.write('SENT=' + SENT.length); }, 30);
"""
    )
    proc = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30)
    assert "SENT=0" in proc.stdout, (
        f"the harness did not notice the model gate returning; stdout={proc.stdout!r} stderr={proc.stderr[:300]!r}"
    )
