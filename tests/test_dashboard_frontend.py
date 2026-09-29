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
SLICE_QWEN_SEED = _slice("    function isQwenImage21Model(modelName) {", "    async function loadSdPresets() {")
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
# isQwenImage21Model / resolveSdSeed — Image Studio gating
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "Qwen-Image-2.1-GGUF/qwen_image_2.1-Q4_K",
        "qwen_image_2.1_q4_k",
        "QWEN-IMAGE-2.1",
        "qwen-image-2.1:q8",
    ],
)
def test_qwen_image_21_variants_are_detected(name):
    run_js(
        SLICE_QWEN_SEED
        + f"""
assert.equal(isQwenImage21Model({json.dumps(name)}), true);
log.push('ok');
"""
    )


@pytest.mark.parametrize(
    "name",
    ["", None, "stable-diffusion-xl-base-1.0", "qwen3-vl-8b", "qwen-image-2.0", "image-2.1", "qwen"],
)
def test_non_qwen_21_image_models_are_rejected(name):
    run_js(
        SLICE_QWEN_SEED
        + f"""
assert.equal(isQwenImage21Model({json.dumps(name)}), false);
log.push('ok');
"""
    )


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
