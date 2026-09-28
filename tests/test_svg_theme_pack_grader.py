"""Tests for the creative_svg_theme_pack grader (llm_benchmark_suite._grade_svg_theme_pack).

This grader is the only one in the suite that checks a *shape* rather than a set
of keywords, because the thing it grades has exactly one consumer: SharedLLM's
motifPattern(), whose output is handed to CSS as
``url("data:image/svg+xml,...")`` and repeated as a ``background-image``. A tile
that is not square-with-matching-viewBox seams at every edge; a tile with baked
hex colours is wrong on every theme but one; a tile with <text> depends on fonts
the host does not ship.

The fixtures below are the shipped motifs from
SharedLLM/services/ui/src/themes/siteTheme.ts, so the "good" answers are real
assets that the production code renders today, and the "bad" answers are each
one specific way to be wrong.
"""

from __future__ import annotations

import pytest

from llm_benchmark_suite import (
    LLMModelBenchmark,
    _grade_svg_theme_pack,
    _svg_has_glyphs,
    _svg_is_seamless_tile,
    _svg_roots,
    _svg_sprite_symbols,
    _svg_uses_theme_placeholders,
    _svg_wireframe_is_valid,
)

# --- A correct tile, in the exact shape motifPattern('petal') emits ----------
GOOD_TILE = """
<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 120 120">
  <g fill="none" stroke-width="1.5">
    <path d="M30 22c8 6 8 16 0 22-8-6-8-16 0-22z" stroke="{{accentAlt}}" stroke-opacity="0.18"/>
    <path d="M92 66c6 5 6 13 0 18-6-5-6-13 0-18z" stroke="{{accent}}" stroke-opacity="0.14"/>
    <circle cx="74" cy="20" r="3" stroke="{{accentAlt}}" stroke-opacity="0.12"/>
    <circle cx="18" cy="78" r="2.5" stroke="{{accent}}" stroke-opacity="0.12"/>
    <path d="M58 96c7 5 7 14 0 19-7-5-7-14 0-19z" stroke="{{accentAlt}}" stroke-opacity="0.16"/>
  </g>
</svg>
"""

# --- A correct sprite sheet, following the repo's own public/icons.svg shape -
GOOD_SPRITE = """
<svg xmlns="http://www.w3.org/2000/svg">
  <symbol id="pulse-icon" viewBox="0 0 24 24">
    <g fill="none" stroke-width="1.4">
      <path d="M2 12h5l2-6 3 12 3-9 2 3h5" stroke="{{accent}}" stroke-opacity="0.16"/>
    </g>
  </symbol>
  <symbol id="shield-icon" viewBox="0 0 24 24">
    <path fill="none" stroke="{{accentAlt}}" stroke-width="1.4" stroke-opacity="0.14"
          d="M12 2l8 3v6c0 5-4 9-8 11-4-2-8-6-8-11V5z"/>
  </symbol>
  <symbol id="bolt-icon" viewBox="0 0 24 24">
    <path fill="none" stroke="{{accent}}" stroke-width="1.5" stroke-opacity="0.18" d="M13 2L4 14h6l-1 8 9-12h-6z"/>
  </symbol>
  <symbol id="droplet-icon" viewBox="0 0 24 24">
    <path fill="none" stroke="{{accentAlt}}" stroke-width="1.4" stroke-opacity="0.12" d="M12 2c4 5 6 8 6 11a6 6 0 11-12 0c0-3 2-6 6-11z"/>
  </symbol>
  <symbol id="moon-icon" viewBox="0 0 24 24">
    <path fill="none" stroke="{{accent}}" stroke-width="1.4" stroke-opacity="0.16" d="M20 14A8 8 0 1110 4a7 7 0 0010 10z"/>
  </symbol>
  <symbol id="spark-icon" viewBox="0 0 24 24">
    <path fill="none" stroke="{{accentAlt}}" stroke-width="1.5" stroke-opacity="0.20" d="M12 3l2 7 7 2-7 2-2 7-2-7-7-2 7-2z"/>
  </symbol>
</svg>
"""

GOOD_ANSWER = f"```svg\n{GOOD_TILE}\n```\n\nAnd the sprite sheet:\n\n```svg\n{GOOD_SPRITE}\n```"


# ==========================================================================
# _svg_roots - locating the <svg> blocks
# ==========================================================================


def test_svg_roots_finds_every_block_in_a_two_asset_answer():
    roots = _svg_roots(GOOD_ANSWER)
    assert len(roots) == 2
    assert roots[0].startswith("<svg")
    assert roots[0].endswith("</svg>")


def test_svg_roots_handles_a_self_closing_root():
    roots = _svg_roots('<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8" viewBox="0 0 8 8"/>')
    assert roots == ['<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8" viewBox="0 0 8 8"/>']


def test_svg_roots_ignores_an_unterminated_block():
    # A truncated answer (the num_predict cap hit) must yield nothing rather
    # than a half-parsed block that could accidentally satisfy a check.
    assert _svg_roots('<svg width="120" height="120" viewBox="0 0 120 120"><g fill="none">') == []


def test_svg_roots_ignores_a_missing_close_bracket():
    assert _svg_roots("<svg width=\"10\"") == []


# ==========================================================================
# _svg_is_seamless_tile - the square-viewBox contract
# ==========================================================================


def test_a_square_tile_with_a_matching_viewbox_is_seamless():
    assert _svg_is_seamless_tile(GOOD_TILE) is True


@pytest.mark.parametrize(
    ("head", "why"),
    [
        ('<svg width="120" height="200" viewBox="0 0 120 120">', "height differs from width"),
        ('<svg width="120" height="120" viewBox="0 0 200 200">', "viewBox extent differs from the pixel size"),
        ('<svg width="120" height="120">', "no viewBox at all"),
        ('<svg viewBox="0 0 120 120">', "no pixel size"),
        ('<svg width="120" height="120" viewBox="10 10 120 120">', "viewBox is not anchored at the origin"),
        ('<svg width="0" height="0" viewBox="0 0 0 0">', "degenerate zero-size tile"),
        ('<svg width="wide" height="wide" viewBox="0 0 120 120">', "non-numeric size"),
    ],
)
def test_a_tile_that_would_seam_is_rejected(head, why):
    assert _svg_is_seamless_tile(head + "</svg>") is False, why


def test_a_sprite_sheet_root_is_not_mistaken_for_a_tile():
    """The sprite root has no viewBox of its own, so it must not be eligible."""
    assert _svg_is_seamless_tile(GOOD_SPRITE) is False


def test_decimal_extents_are_accepted():
    assert _svg_is_seamless_tile('<svg width="48.0" height="48.0" viewBox="0 0 48 48"></svg>') is True


# ==========================================================================
# _svg_uses_theme_placeholders - the colour contract
# ==========================================================================


def test_both_placeholders_are_required():
    assert _svg_uses_theme_placeholders(GOOD_TILE) is True
    only_accent = '<svg><path stroke="{{accent}}" stroke-opacity="0.1"/></svg>'
    assert _svg_uses_theme_placeholders(only_accent) is False


def test_whitespace_inside_the_placeholder_is_tolerated():
    assert _svg_uses_theme_placeholders('<svg><path stroke="{{ accentAlt }}" stroke="{{accent}}"/></svg>') is True


def test_the_placeholder_match_is_case_insensitive_but_preserves_accentAlt():
    # motifPattern() interpolates the exact literal {{accentAlt}}; a model that
    # writes {{ACCENTALT}} still works at runtime only if the consumer is
    # case-insensitive, so accept either casing here.
    assert _svg_uses_theme_placeholders('<svg><path stroke="{{accentalt}}" stroke="{{accent}}"/></svg>') is True


def test_a_baked_hex_colour_rejects_the_tile():
    baked = GOOD_TILE.replace('stroke="{{accentAlt}}"', 'stroke="#863BFF"')
    assert _svg_uses_theme_placeholders(baked) is False


def test_a_hex_in_any_attribute_rejects_the_tile_even_alongside_placeholders():
    sneaky = GOOD_TILE.replace('fill="none"', 'fill="none" style="color:#863bff"')
    assert _svg_uses_theme_placeholders(sneaky) is False


def test_a_three_digit_hex_also_rejects():
    assert _svg_uses_theme_placeholders('<svg><path stroke="{{accent}}" fill="#abc" stroke-opacity="0.1"/></svg>') is False


# ==========================================================================
# _svg_wireframe_is_valid - the hairline/whisper contract
# ==========================================================================


def test_the_shipped_petal_motif_passes_the_wireframe_check():
    assert _svg_wireframe_is_valid(GOOD_TILE) is True


def test_fill_none_is_required():
    assert _svg_wireframe_is_valid('<svg><g stroke-width="1.2"><path stroke="{{accent}}" stroke-opacity="0.1"/></g></svg>') is False


def test_a_missing_stroke_width_is_rejected():
    body = '<svg><g fill="none"><path stroke="{{accent}}" stroke-opacity="0.1"/></g></svg>'
    assert _svg_wireframe_is_valid(body) is False


def test_a_missing_stroke_opacity_is_rejected():
    body = '<svg><g fill="none" stroke-width="1.2"><path stroke="{{accent}}"/></g></svg>'
    assert _svg_wireframe_is_valid(body) is False


@pytest.mark.parametrize("width", ["0.5", "4", "8"])
def test_a_stroke_width_outside_the_hairline_range_is_rejected(width):
    body = f'<svg><g fill="none" stroke-width="{width}"><path stroke="{{{{accent}}}}" stroke-opacity="0.1"/></g></svg>'
    assert _svg_wireframe_is_valid(body) is False


@pytest.mark.parametrize("opacity", ["0.6", "1.0", "0.02"])
def test_a_stroke_opacity_outside_the_whisper_range_is_rejected(opacity):
    body = f'<svg><g fill="none" stroke-width="1.2"><path stroke="{{{{accent}}}}" stroke-opacity="{opacity}"/></g></svg>'
    assert _svg_wireframe_is_valid(body) is False


def test_one_out_of_range_shape_rejects_the_whole_tile():
    """A tile is only as subtle as its darkest element, so a single bold shape
    makes the pattern compete with body copy."""
    bold = GOOD_TILE.replace('stroke-opacity="0.18"', 'stroke-opacity="0.55"')
    assert _svg_wireframe_is_valid(bold) is False


def test_stroke_width_may_be_set_per_shape_not_only_on_the_group():
    per_shape = (
        '<svg><g fill="none">'
        '<path d="M0 0h10" stroke="{{accent}}" stroke-width="1.2" stroke-opacity="0.1"/>'
        '<circle cx="5" cy="5" r="3" stroke="{{accentAlt}}" stroke-width="1.5" stroke-opacity="0.2"/>'
        "</g></svg>"
    )
    assert _svg_wireframe_is_valid(per_shape) is True


# ==========================================================================
# _svg_has_glyphs - why the repo has no font files matters here
# ==========================================================================


@pytest.mark.parametrize("body", ['<text x="1" y="2">hi</text>', "<tspan>hi</tspan>", "<textPath>hi</textPath>"])
def test_glyph_elements_are_detected(body):
    assert _svg_has_glyphs(f"<svg>{body}</svg>") is True


def test_a_font_family_is_detected_even_without_text():
    assert _svg_has_glyphs('<svg style="font-family: Inter"><path d="M0 0"/></svg>') is True


def test_clean_line_art_has_no_glyphs():
    assert _svg_has_glyphs(GOOD_TILE) is False


# ==========================================================================
# _svg_sprite_symbols - the repo's own icon-sheet convention
# ==========================================================================


def test_all_shipped_symbols_are_recognised():
    assert _svg_sprite_symbols(GOOD_SPRITE) == [
        "pulse-icon",
        "shield-icon",
        "bolt-icon",
        "droplet-icon",
        "moon-icon",
        "spark-icon",
    ]


def test_a_symbol_without_a_viewbox_is_not_counted():
    body = '<svg><symbol id="pulse-icon"><path d="M0 0"/></symbol></svg>'
    assert _svg_sprite_symbols(body) == []


def test_an_id_outside_the_kebab_icon_convention_is_not_counted():
    body = '<svg><symbol id="pulse" viewBox="0 0 24 24"/></svg>'
    assert _svg_sprite_symbols(body) == []


def test_a_camel_case_id_is_not_counted():
    body = '<svg><symbol id="pulseIcon" viewBox="0 0 24 24"/></svg>'
    assert _svg_sprite_symbols(body) == []


# ==========================================================================
# _grade_svg_theme_pack - the end-to-end verdict
# ==========================================================================


def test_a_correct_two_asset_answer_passes():
    assert _grade_svg_theme_pack(GOOD_ANSWER) is True


def test_the_prompt_example_itself_would_not_pass():
    """Guard against a grader that accepts the prompt. The example tile in the
    prompt has one shape, no sprite sheet, and no stroke-opacity sweep."""
    example = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 120 120">\n'
        '  <g fill="none" stroke-width="1.4">\n'
        '    <circle cx="60" cy="60" r="30" stroke="{{accent}}" stroke-opacity="0.12"/>\n'
        "  </g>\n"
        "</svg>"
    )
    assert _grade_svg_theme_pack(example) is False


@pytest.mark.parametrize(
    "response",
    [
        pytest.param("", id="empty"),
        pytest.param("I cannot do that.", id="refusal"),
        pytest.param("```svg\n<svg></svg>\n```", id="empty-root"),
        pytest.param("```\n<svg width='10' height='10' viewBox='0 0 10 10'><g fill='none' stroke-width='1'/>"
                     "<path stroke='{{accentAlt}}' stroke-opacity='0.1'/><path stroke='{{accent}}' stroke-opacity='0.1'/>"
                     "</g></svg>\n```", id="unclosed"),
    ],
)
def test_a_non_answer_fails(response):
    assert _grade_svg_theme_pack(response) is False


def test_a_tile_without_a_sprite_sheet_fails():
    assert _grade_svg_theme_pack(f"```svg\n{GOOD_TILE}\n```") is False


def test_a_sprite_sheet_without_a_tile_fails():
    assert _grade_svg_theme_pack(f"```svg\n{GOOD_SPRITE}\n```") is False


def test_a_seaming_tile_fails_even_when_everything_else_is_right():
    rect = GOOD_TILE.replace('width="120" height="120"', 'width="120" height="240"')
    assert _grade_svg_theme_pack(f"{rect}\n{GOOD_SPRITE}") is False


def test_a_hex_baked_tile_fails():
    baked = GOOD_TILE.replace('stroke="{{accentAlt}}"', 'stroke="#E06A9A"')
    assert _grade_svg_theme_pack(f"{baked}\n{GOOD_SPRITE}") is False


def test_a_tile_with_text_fails():
    with_text = GOOD_TILE.replace("</g>", '<text x="10" y="60" stroke="{{accentAlt}}" stroke-opacity="0.2">JARVIS</text></g>')
    assert _grade_svg_theme_pack(f"{with_text}\n{GOOD_SPRITE}") is False


def test_five_symbols_is_one_short():
    trimmed = GOOD_SPRITE.replace(
        '<symbol id="spark-icon" viewBox="0 0 24 24">\n'
        '    <path fill="none" stroke="{{accentAlt}}" stroke-width="1.5" stroke-opacity="0.20" d="M12 3l2 7 7 2-7 2-2 7-2-7-7-2 7-2z"/>\n'
        "  </symbol>\n",
        "",
    )
    assert len(_svg_sprite_symbols(trimmed)) == 5
    assert _grade_svg_theme_pack(f"{GOOD_TILE}\n{trimmed}") is False


def test_six_symbols_with_the_wrong_id_shape_fail():
    renamed = GOOD_SPRITE.replace("-icon", "Icon")
    assert _grade_svg_theme_pack(f"{GOOD_TILE}\n{renamed}") is False


def test_a_tile_with_only_the_svg_wrapper_and_no_shapes_fails():
    empty = (
        '<svg width="120" height="120" viewBox="0 0 120 120">'
        '<g fill="none" stroke-width="1.2" stroke="{{accentAlt}}" stroke-opacity="0.1"/></svg>'
    )
    assert _grade_svg_theme_pack(f"{empty}\n{GOOD_SPRITE}") is False


def test_a_sprite_sheet_with_text_fails():
    texty = GOOD_SPRITE.replace("</svg>", '<text x="0" y="0">pulse</text></svg>')
    assert _grade_svg_theme_pack(f"{GOOD_TILE}\n{texty}") is False


def test_both_assets_inside_one_document_pass():
    """One <svg> holding the tile's shapes and six symbols is a valid answer;
    the two deliverables do not have to arrive in separate documents."""
    merged = GOOD_TILE.replace("</svg>", GOOD_SPRITE[GOOD_SPRITE.index("<symbol") :])
    assert _svg_sprite_symbols(merged) != []  # the merged root has no viewBox, so:
    # ...which means the tile's own root is gone. A merged document is therefore
    # not accepted, and that is deliberate: the consumer needs a square root for
    # the tile and a viewBox-less root for the sheet.
    assert _grade_svg_theme_pack(merged) is False


# ==========================================================================
# Registration in the benchmark
# ==========================================================================


def test_the_grader_is_reachable_from_the_dispatch_chain():
    bench = LLMModelBenchmark.__new__(LLMModelBenchmark)
    assert bench._verify_functional_response({"id": "creative_svg_theme_pack"}, GOOD_ANSWER) is True
    assert bench._verify_functional_response({"id": "creative_svg_theme_pack"}, "no") is False


def test_the_grader_is_reachable_from_a_legacy_test_id_string():
    bench = LLMModelBenchmark.__new__(LLMModelBenchmark)
    assert bench._verify_functional_response("creative_svg_theme_pack", GOOD_ANSWER) is True


def test_a_think_block_around_the_answer_does_not_break_it():
    """A reasoning-tuned model wraps its answer in <think>; the surrounding
    think prose must not stop the grader finding the assets."""
    noisy = f"<think>\nThe user wants two blocks. Let me plan.\n</think>\n\n{GOOD_ANSWER}"
    assert _grade_svg_theme_pack(noisy) is True


def test_the_test_carries_a_functional_grader_version():
    version = LLMModelBenchmark.FUNCTIONAL_GRADER_VERSIONS.get("creative_svg_theme_pack")
    assert version == "v1", "bump this when the grader changes materially, so outdated_only re-runs it"


def test_the_test_is_registered_in_the_benchmark_corpus():
    import json
    from pathlib import Path

    data = json.loads((Path(__file__).resolve().parents[1] / "benchmark_tests.json").read_text())
    tests = [t for bucket in data.values() for t in bucket if t["id"] == "creative_svg_theme_pack"]
    assert len(tests) == 1
    entry = tests[0]
    assert entry["type"] == "functional"
    assert entry["num_predict"] >= 3000, "two full SVG documents do not fit in 800 tokens"
    # The prompt must actually state the three structural rules the grader
    # enforces, or the test measures whether the model guessed them.
    assert "{{accent}}" in entry["prompt"] and "{{accentAlt}}" in entry["prompt"]
    assert "fill=\"none\"" in entry["prompt"]
    assert "stroke-opacity" in entry["prompt"]
    assert "viewBox" in entry["prompt"] and "width=\"N\"" in entry["prompt"]
    assert "font-family" in entry["prompt"]
    assert "-icon" in entry["prompt"]
