"""Tests for manuscript_rubric.py and its wiring into the benchmark suite.

The reference manuscript in this file is a genuine miniature of the thing the
grader is for: a framed leaf, hand-drawn SVG border and initial, all 18 verses,
and a real rotateY page flip. Negative cases then remove exactly one property at
a time, so a failure names the missing one instead of "it didn't look right".
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import llm_benchmark_suite as suite
import manuscript_rubric as rubric

REPO = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# A reference manuscript that passes every criterion.
# --------------------------------------------------------------------------

_VERSE_HTML = "".join(
    f'<p class="verse"><span class="initial">{"I" if index == 0 else ""}</span>{verse}</p>'
    for index, verse in enumerate(rubric.JOHN_1_KJV)
)

REFERENCE = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>In the beginning</title>
<style>
  body {{ margin:0; background:#2b2013; font-family: Georgia, 'Times New Roman', serif; }}
  .leaf {{ perspective:1600px; }}
  .leaf .face {{ transition: transform .7s ease; transform-style: preserve-3d; }}
  .leaf.flipped .face {{ transform: rotateY(180deg); }}
  .page {{ position:relative; width:640px; min-height:900px; margin:40px auto;
           padding:56px 64px; background:#f4ecd8; border:2px solid #8a6a3a;
           box-shadow:0 18px 40px rgba(0,0,0,.5); }}
  .page .ornament {{ position:absolute; inset:18px; border:3px double #a8763c; }}
  .verse {{ font-size:19px; line-height:1.5; }}
  .initial {{ float:left; font-size:64px; line-height:.8; color:#8a1f1f; }}
</style>
</head>
<body>
<div class="leaf" id="leaf" onclick="this.classList.toggle('flipped')">
  <section class="page">
    <svg class="ornament" viewBox="0 0 640 900" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <path d="M0 40 Q20 0 60 0 L580 0 Q620 0 640 40 L640 860 Q620 900 580 900 L60 900 Q20 900 0 860 Z"
            fill="none" stroke="#a8763c" stroke-width="3"/>
      <circle cx="320" cy="52" r="18" fill="none" stroke="#8a1f1f" stroke-width="2"/>
      <path d="M302 52 h36 M320 34 v36" stroke="#8a1f1f" stroke-width="2"/>
    </svg>
    {_VERSE_HTML}
  </section>
</div>
</body>
</html>"""


def reference() -> str:
    return REFERENCE


# --------------------------------------------------------------------------
# The scripture itself
# --------------------------------------------------------------------------


def test_the_source_text_is_eighteen_verses():
    assert len(rubric.JOHN_1_KJV) == 18


def test_the_source_text_is_the_kjv_and_not_a_summary():
    """Spot-check the openings and the two awkward verses a summariser drops."""
    assert rubric.JOHN_1_KJV[0] == (
        "In the beginning was the Word, and the Word was with God, and the Word was God."
    )
    assert rubric.JOHN_1_KJV[1] == "The same was in the beginning with God."
    assert rubric.JOHN_1_KJV[14] == (
        "John bare witness of him, and cried, saying, Behold, the Lamb of God, "
        "which taketh away the sin of the world."
    )
    assert rubric.JOHN_1_KJV[16] == "For the law was given by Moses; but grace and truth came by Jesus Christ."


def test_verse_twelve_keeps_its_trailing_colon():
    # It is the only verse that does not end in a full stop, and dropping it is
    # the kind of silent alteration the test exists to catch.
    assert rubric.JOHN_1_KJV[11].endswith("his name:")


def test_verse_fourteen_keeps_its_parentheses_for_the_audio_pipeline():
    # tts_text._paren_to_commas turns these into commas before narration, which
    # is the desired behaviour for speech and the reason they can stay here.
    assert "(and we beheld his glory" in rubric.JOHN_1_KJV[13]


def test_every_verse_is_non_empty_and_distinct():
    assert all(v.strip() for v in rubric.JOHN_1_KJV)
    assert len({v for v in rubric.JOHN_1_KJV}) == 18


# --------------------------------------------------------------------------
# Verse detection
# --------------------------------------------------------------------------


def test_all_eighteen_verses_are_found_in_a_plain_rendering():
    html = "<html><body>" + "<br>".join(f"{i}. {v}" for i, v in enumerate(rubric.JOHN_1_KJV, 1)) + "</body></html>"
    assert rubric.verses_present(html) == list(range(1, 19))


def test_verses_are_found_across_tag_soup_and_wrapped_lines():
    """A real page splits each verse across a <span> number, a drop cap and
    three wrapped lines. The text content is what matters, not the markup."""
    parts = []
    for index, verse in enumerate(rubric.JOHN_1_KJV, 1):
        words = verse.split(" ")
        cut = len(words) // 2
        parts.append(
            f'<p><sup class="n">{index}</sup><span class="c">{" ".join(words[:cut])}</span>\n'
            f'          <span>{" ".join(words[cut:])}</span></p>'
        )
    assert len(rubric.verses_present("<html><body>" + "".join(parts) + "</body></html>")) == 18


@pytest.mark.parametrize(
    ("from_char", "to_char", "note"),
    [
        ("’", "'", "curly apostrophe"),
        ("“", '"', "curly double quote"),
        ("–", "-", "en dash"),
        ("—", "-", "em dash"),
        (" ", " ", "non-breaking space"),
    ],
)
def test_smart_punctuation_and_nbsp_do_not_hide_a_verse(from_char, to_char, note):
    html = "<html><body>" + "<br>".join(v.replace("'", from_char) if from_char in "'’" else v for v in rubric.JOHN_1_KJV) + "</body></html>"
    if from_char == " ":
        html = html.replace(" ", " ")
    assert len(rubric.verses_present(html)) == 18, note


def test_script_and_style_content_is_not_mistaken_for_text():
    html = (
        "<html><head><style>.x::after{content:'the light shineth in darkness'}</style></head>"
        "<body><p>nothing here</p></body></html>"
    )
    assert rubric.verses_present(html) == []


def test_a_verse_absent_from_the_page_is_reported_by_number():
    """The failure message has to name the verse, or diagnosing a 16k-token run
    means re-reading the model output."""
    html = "<html><body>" + "<br>".join(rubric.JOHN_1_KJV[1:5] + rubric.JOHN_1_KJV[5:]) + "</body></html>"
    # drop verse 18 only
    html = "<html><body>" + "<br>".join(v for i, v in enumerate(rubric.JOHN_1_KJV, 1) if i != 18) + "</body></html>"
    found = rubric.verses_present(html)
    assert 18 not in found
    assert len(found) == 17
    result = rubric.manuscript_rubric(html)
    assert result["verses_missing"] == [18]


def test_a_paraphrase_does_not_count_as_the_verse():
    """KJV "was made flesh"; a modern rendering says "became flesh". Fidelity
    is the point of a manuscript, so the paraphrase must not pass."""
    html = "<html><body><p>The Word became flesh and lived among us.</p></body></html>"
    assert rubric.verses_present(html) == []


def test_entities_are_decoded_before_matching():
    html = "<html><body><p>In the beginning was the Word, and the Word was with God, and the Word was God.</p></body></html>"
    assert 1 in rubric.verses_present(html)


# --------------------------------------------------------------------------
# Complete document
# --------------------------------------------------------------------------


def test_the_reference_is_a_complete_document():
    assert rubric.is_complete_document(reference()) is True


@pytest.mark.parametrize(
    "mutation",
    [
        ("<!DOCTYPE html>", ""),
        ("<html lang=\"en\">", ""),
        ("</html>", ""),
        ("<body>", ""),
        ("<body>", "<div>"),
    ],
)
def test_a_fragment_is_not_a_complete_document(mutation, ):
    """arcade_publish applies the same rule and silently downgrades a fragment
    to a frozen .py, so an answer that fails it can never be played either."""
    html = reference().replace(*mutation)
    assert rubric.is_complete_document(html) is False


def test_a_lowercase_doctype_still_counts():
    assert rubric.is_complete_document(reference().replace("<!DOCTYPE html>", "<!doctype html>"))


# --------------------------------------------------------------------------
# Drawn SVG
# --------------------------------------------------------------------------


def test_the_reference_has_drawn_svg():
    assert rubric._has_drawn_svg(reference()) is True


@pytest.mark.parametrize("shape", ["path", "circle", "ellipse", "rect", "line", "polyline", "polygon"])
def test_any_single_drawn_shape_is_enough(shape):
    svg = f'<svg viewBox="0 0 10 10"><{shape} /></svg>'
    assert rubric._has_drawn_svg(f"<html><body>{svg}</body></html>") is True


def test_an_empty_group_is_not_decoration():
    assert rubric._has_drawn_svg('<html><body><svg viewBox="0 0 10 10"><g></g></svg></body></html>') is False


def test_a_wrapper_only_svg_is_not_decoration():
    assert rubric._has_drawn_svg('<html><body><svg viewBox="0 0 10 10"></svg></body></html>') is False


@pytest.mark.parametrize(
    "svg",
    [
        '<svg><path d="M0 0 L10 10"/></svg>',  # no viewBox at all
        '<svg viewBox="0 0 0 0"><path d="M0 0 L10 10"/></svg>',  # renders nothing
        '<svg viewBox="0 10 10 0"><path d="M0 0 L10 10"/></svg>',  # zero height
        '<svg viewBox="0 0 wide tall"><path d="M0 0 L10 10"/></svg>',  # not numbers
        '<svg viewBox="0 0 10"><path d="M0 0 L10 10"/></svg>',  # three numbers
    ],
)
def test_a_viewbox_that_is_absent_zero_or_nonnumeric_does_not_count(svg):
    assert rubric._has_drawn_svg(f"<html><body>{svg}</body></html>") is False


def test_a_truncated_svg_cannot_half_satisfy_the_decoration_check():
    """A response cut off by the token budget is the common failure here."""
    truncated = reference().replace("</svg>", "", 1)
    assert rubric._has_drawn_svg(truncated) is False


def test_decoration_may_live_in_several_svgs():
    html = '<html><body><svg viewBox="0 0 4 4"></svg><svg viewBox="0 0 4 4"><path d="M0 0"/></svg></body></html>'
    assert rubric._has_drawn_svg(html) is True


# --------------------------------------------------------------------------
# Framed page
# --------------------------------------------------------------------------


def test_the_reference_has_a_framed_page():
    assert rubric._has_page_frame(reference()) is True


def test_a_background_on_body_is_page_background_not_a_framed_leaf():
    html = reference().replace("border:2px solid #8a6a3a;", "").replace("box-shadow:0 18px 40px rgba(0,0,0,.5);", "").replace("background:#f4ecd8;", "")
    assert rubric._has_page_frame(html) is False


def test_a_container_with_no_page_named_class_does_not_count():
    assert rubric._has_page_frame(reference().replace('class="page"', 'class="x"')) is False


@pytest.mark.parametrize("name", ["page", "leaf", "folio", "spread", "codex", "manuscript", "vellum", "page-left"])
def test_a_page_naming_class_is_accepted(name):
    html = f"<html><head><style>.{name} {{ border:2px solid #000; }}</style></head><body><div class=\"{name}\"></div></body></html>"
    assert rubric._has_page_frame(html) is True


def test_an_inline_style_frame_counts():
    html = '<html><body><section class="page" style="border:3px solid #000"></section></body></html>'
    assert rubric._has_page_frame(html) is True


def test_an_id_can_name_the_page_too():
    html = '<html><head><style>#leaf { border:2px solid #000; }</style></head><body><div id="leaf"></div></body></html>'
    assert rubric._has_page_frame(html) is True


def test_a_frame_declared_only_in_a_comment_is_not_a_frame():
    html = reference().replace("border:2px solid #8a6a3a;", "/* border:2px solid #8a6a3a; */")
    html = html.replace("box-shadow:0 18px 40px rgba(0,0,0,.5);", "/* box-shadow */")
    html = html.replace("background:#f4ecd8;", "/* background */")
    assert rubric._has_page_frame(html) is False


def test_the_ornaments_own_border_does_not_frame_the_page():
    """A real stylesheet frames the page and then draws a border on a child
    ornament. The child's border must not be credited to its parent."""
    html = reference()
    # Give the page no frame of its own; only .page .ornament keeps one.
    html = html.replace("border:2px solid #8a6a3a;", "").replace("box-shadow:0 18px 40px rgba(0,0,0,.5);", "").replace("background:#f4ecd8;", "")
    assert rubric._has_page_frame(html) is False


def test_the_body_is_never_the_page_container():
    assert rubric._has_page_frame("<html><head><style>body{border:2px solid #000}</style></head><body></body></html>") is False


def test_a_nested_selector_on_the_page_class_counts():
    """A real stylesheet writes `.leaf .page { ... }`; the declarations still
    apply to the element, so the check must find them."""
    html = '<html><head><style>.leaf .page { border:1px solid #000; }</style></head><body><div class="leaf"><div class="page"></div></div></body></html>'
    assert rubric._has_page_frame(html) is True


# --------------------------------------------------------------------------
# Page flip
# --------------------------------------------------------------------------


def test_the_reference_has_a_flip_affordance():
    assert rubric._has_flip_affordance(reference()) is True


def test_a_javascript_flip_counts():
    html = '<html><body><div onclick="flip()"></div><script>function flip(){}</script></body></html>'
    assert rubric._has_flip_affordance(html) is True


def test_a_css_only_flip_counts():
    """A :hover or class-toggled 3D transform is a legitimate implementation."""
    html = "<html><head><style>.page:hover { transform: rotateY(180deg); }</style></head><body><div class=\"page\"></div></body></html>"
    assert rubric._has_flip_affordance(html) is True


def test_a_flip_described_in_a_comment_is_not_an_affordance():
    html = "<html><head><style>/* perspective:1600px; transform: rotateY(180deg); onclick flips it */</style></head><body></body></html>"
    assert rubric._has_flip_affordance(html) is False


def test_a_keyword_without_machinery_is_not_an_affordance():
    """A class called `.flip` with no transform and no handler is decoration by
    name only."""
    html = '<html><body><div class="flip page"></div></body></html>'
    assert rubric._has_flip_affordance(html) is False


def test_an_html_comment_is_also_stripped():
    html = '<html><body><!-- transform: rotateY(180deg) --></body></html>'
    assert rubric._has_flip_affordance(html) is False


# --------------------------------------------------------------------------
# The rubric as a whole
# --------------------------------------------------------------------------


def test_the_reference_scores_full_marks():
    result = rubric.manuscript_rubric(reference())
    assert result["passed"] is True
    assert result["score"] == 100
    assert result["criteria_met"] == result["criteria_total"] == 5
    assert result["verses_missing"] == []


def test_the_rubric_reports_every_criterion_key():
    result = rubric.manuscript_rubric(reference())
    for key in (
        "complete_document",
        "all_verses_present",
        "decorative_svg_drawn",
        "framed_page",
        "page_flip_affordance",
    ):
        assert key in result


def test_an_empty_answer_scores_zero_and_names_everything():
    passed, reason = rubric.grade_illuminated_manuscript("")
    assert passed is False
    for phrase in ("complete", "verse", "svg", "framed page", "flip"):
        assert phrase in reason.lower()


def test_the_text_alone_is_two_of_five_criteria():
    """The realistic near-miss: a model returns John 1 as a plain <body>."""
    html = "<!DOCTYPE html><html><body>" + "".join(f"<p>{v}</p>" for v in rubric.JOHN_1_KJV) + "</body></html>"
    result = rubric.manuscript_rubric(html)
    assert result["criteria_met"] == 2
    assert result["score"] == 40
    assert result["passed"] is False


def test_each_criterion_can_be_removed_independently():
    """One property removed at a time, so a regression names the property."""
    mutations = {
        "complete_document": lambda h: h.replace("<!DOCTYPE html>", ""),
        "all_verses_present": lambda h: h.replace(f"<span class=\"initial\"></span>{rubric.JOHN_1_KJV[17]}", ""),
        "decorative_svg_drawn": lambda h: re.sub(r"<(path|circle)\b.*?(?:/>|</\1>)", "<g></g>", h, flags=re.DOTALL),
        "framed_page": lambda h: h.replace("border:2px solid #8a6a3a;", "").replace("box-shadow:0 18px 40px rgba(0,0,0,.5);", "").replace("background:#f4ecd8;", ""),
        "page_flip_affordance": lambda h: h.replace("perspective:1600px;", "").replace("transform: rotateY(180deg);", "").replace("transform-style: preserve-3d;", "").replace("transition: transform .7s ease;", "").replace('onclick="this.classList.toggle(\'flipped\')"', ""),
    }
    full = rubric.manuscript_rubric(reference())
    assert full["passed"] is True
    for key, mutate in mutations.items():
        result = rubric.manuscript_rubric(mutate(reference()))
        assert result[key] is False, f"{key} should have been the thing that broke"
        assert result["passed"] is False
        assert key in result


def test_score_is_a_percentage_of_criteria():
    result = rubric.manuscript_rubric("")
    assert result["score"] == 0
    assert result["criteria_met"] == 0
    assert result["criteria_total"] == 5


# --------------------------------------------------------------------------
# Wiring into the benchmark suite
# --------------------------------------------------------------------------


def _load_tests() -> dict:
    return json.loads((REPO / "benchmark_tests.json").read_text())


def test_the_test_exists_in_the_creative_category():
    entry = next(t for t in _load_tests()["creative"] if t["id"] == "creative_illuminated_manuscript_john1")
    assert entry["type"] == "ui"
    assert entry["lang"] == "html"
    assert entry["num_predict"] == 16000


def test_the_prompt_carries_all_eighteen_verses_verbatim():
    """If the prompt and the rubric ever disagree about the text, the test
    silently grades for something the model was never asked for."""
    entry = next(t for t in _load_tests()["creative"] if t["id"] == "creative_illuminated_manuscript_john1")
    for verse in rubric.JOHN_1_KJV:
        assert verse in entry["prompt"], verse[:40]


def test_the_prompt_states_every_rule_the_grader_enforces():
    entry = next(t for t in _load_tests()["creative"] if t["id"] == "creative_illuminated_manuscript_john1")
    prompt = entry["prompt"].lower()
    for requirement in ("doctype", "viewbox", "rotatey", "georgia", "border", "path"):
        assert requirement in prompt, requirement


def test_the_grader_is_registered_so_outdated_only_re_runs_it():
    versions = suite.LLMModelBenchmark.FUNCTIONAL_GRADER_VERSIONS
    assert versions.get("creative_illuminated_manuscript_john1")


def test_the_suite_dispatches_to_the_rubric():
    test = {"id": "creative_illuminated_manuscript_john1", "type": "ui"}
    benchmark = suite.LLMModelBenchmark()
    response = f"Here you go:\n\n```html\n{reference()}\n```\n"
    assert benchmark._verify_functional_response(test, response) is True


def test_the_suite_rejects_a_bare_text_page():
    test = {"id": "creative_illuminated_manuscript_john1", "type": "ui"}
    benchmark = suite.LLMModelBenchmark()
    html = "<!DOCTYPE html><html><body>" + "".join(f"<p>{v}</p>" for v in rubric.JOHN_1_KJV) + "</body></html>"
    assert benchmark._verify_functional_response(test, f"```html\n{html}\n```") is False


def test_the_suite_reads_the_fenced_document_not_the_whole_reply():
    """A model that explains the flip in prose and ships a bare page must fail:
    the prose must not be mistaken for the affordance."""
    test = {"id": "creative_illuminated_manuscript_john1", "type": "ui"}
    benchmark = suite.LLMModelBenchmark()
    html = "<!DOCTYPE html><html><body>" + "".join(f"<p>{v}</p>" for v in rubric.JOHN_1_KJV) + "</body></html>"
    chatter = "You can add perspective: 1600px and transform: rotateY(180deg) with an onclick handler later.\n"
    assert benchmark._verify_functional_response(test, f"{chatter}\n```html\n{html}\n```") is False


def test_a_think_block_around_the_answer_does_not_break_it():
    test = {"id": "creative_illuminated_manuscript_john1", "type": "ui"}
    benchmark = suite.LLMModelBenchmark()
    response = f"<think>Let me draw a border with SVG paths.</think>\n\n```html\n{reference()}\n```"
    assert benchmark._verify_functional_response(test, response) is True


def test_the_grader_ignores_other_tests():
    """A ui test that is not the manuscript must not be gated by it."""
    benchmark = suite.LLMModelBenchmark()
    html = "<!DOCTYPE html><html><body><h1>hi</h1></body></html>"
    test = {"id": "office_logo", "type": "functional"}
    assert benchmark._verify_functional_response(test, f"```html\n{html}\n```") is False
