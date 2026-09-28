"""Schema and coherence rules for benchmark_tests.json.

This file is data, not code, so nothing stops someone from adding a test with
`"num_predict": 0` (which asks the backend to generate nothing, so the test can
never pass) or an answer key pointing at a letter the prompt never offers. Each
rule below is a defect that has actually been present in this file at some
point, or a property the rest of the pipeline silently depends on.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TESTS_PATH = REPO / "benchmark_tests.json"

ALL_TESTS: list[tuple[str, dict]] = [
    (category, test) for category, tests in json.loads(TESTS_PATH.read_text()).items() for test in tests
]
CATEGORIES = sorted({category for category, _ in ALL_TESTS})
BY_ID = {test["id"]: (category, test) for category, test in ALL_TESTS}

VALID_TYPES = {"functional", "code", "ui", "open", "performance", "review", "image", "tts", "music", "composite"}
VALID_LANGS = {
    "python", "py", "node", "js", "javascript", "html", "htm", "web", "c", "cpp", "c++", "cxx",
    "go", "rust", "java", "sql", "bash", "sh", "basic", "bas", "pascal", "pas", "typescript", "ts",
    "yaml", "yml", "json", "terraform", "rpm", "text", "lua", "ruby", "php", "csharp", "swift", "kotlin",
}
VALID_FRAMEWORKS = {"lvgl"}

# Tests whose prompt says "the attached X" but which carry no attachment. The
# benchmark runner never sends attachments to a model - `_resolve_attachment`
# in web/app.py is only reached from the Test Browser preview route - so these
# three prompts are asking about something the model cannot see. They have
# never produced a result file. This set is a shrink-only allow-list: a fourth
# attachment-dependent test fails the suite until the runner is wired.
KNOWN_UNWIRED_ATTACHMENT_TESTS = {"mm_image_light_switch", "mm_html_form_render", "mm_node_fib"}

# `performance` ships two rows with an EMPTY prompt; `web/app.py` substitutes the
# real ~800/~1000-token load prompt when it serves `/api/tests`. They are the
# only tests allowed to have no prompt.
SYNTHESIZED_PROMPT_TESTS = {"perf_medium", "perf_long"}

# The values `scripts/gen_reasoning_estimates.py` is allowed to emit: the four
# task-shape tiers plus the two hand-set values the retrogames use.
REASONING_ESTIMATE_TIERS = {1024, 2048, 3072, 4096, 6000, 8000}

# Tests with no `type` key, so `_infer_type` guesses from the response.
UNTYPED_TESTS = {"instruction_adherence", "logic_bridge_torch", "logic_twins_birthday"}


# --------------------------------------------------------------------------
# file shape
# --------------------------------------------------------------------------


def test_the_file_is_a_dict_of_categories_to_test_lists():
    data = json.loads(TESTS_PATH.read_text())
    assert isinstance(data, dict)
    for category, tests in data.items():
        assert isinstance(category, str) and category
        assert isinstance(tests, list) and tests, f"category {category!r} has no tests"
        assert all(isinstance(t, dict) for t in tests), f"category {category!r} has a non-dict test"


def test_a_json_round_trip_preserves_every_prompt_and_every_hash():
    """A re-serialisation of this file must not change a single graded field.

    The file is written with a mix of literal and `\\uXXXX`-escaped non-ASCII,
    so a maintainer who reformats it with different `json.dumps` settings can
    silently rewrite prompts. A rewritten prompt is a different
    `_compute_test_hash`, which marks the whole stored corpus outdated. This is
    the guard that makes a formatting-only change safe to detect.
    """
    from web.app import _compute_test_hash

    original = json.loads(TESTS_PATH.read_text())
    for settings in ({"ensure_ascii": True}, {"ensure_ascii": False}):
        round_tripped = json.loads(json.dumps(original, **settings))
        assert round_tripped == original, f"round trip with {settings} changed the data"
        before = [(_compute_test_hash(t), t["id"]) for ts in original.values() for t in ts]
        after = [
            (_compute_test_hash(t), t["id"]) for tests in round_tripped.values() for t in tests
        ]
        assert after == before, f"round trip with {settings} moved a test hash"


def test_the_file_ends_with_exactly_one_newline():
    raw = TESTS_PATH.read_text()
    assert raw.endswith("\n")
    assert not raw.endswith("\n\n")


# --------------------------------------------------------------------------
# per-test required fields
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("category", "test"), ALL_TESTS, ids=[t["id"] for _, t in ALL_TESTS])
def test_required_fields_are_present_and_typed(category: str, test: dict):
    for field in ("id", "label", "prompt", "num_predict", "reasoning_estimate"):
        assert field in test, f"{test.get('id')} is missing {field}"
    assert isinstance(test["id"], str) and test["id"].strip() == test["id"] and test["id"]
    assert isinstance(test["label"], str) and test["label"].strip()
    assert isinstance(test["prompt"], str)
    if test["id"] in SYNTHESIZED_PROMPT_TESTS:
        assert test["prompt"] == "", f"{test['id']} is a placeholder and must stay empty"
    else:
        assert test["prompt"].strip(), f"{test['id']} has an empty prompt"
        assert "\n\n\n" not in test["prompt"], f"{test['id']} has a triple blank line"
    assert isinstance(test["num_predict"], int) and not isinstance(test["num_predict"], bool)
    assert isinstance(test["reasoning_estimate"], int) and not isinstance(test["reasoning_estimate"], bool)


@pytest.mark.parametrize(("category", "test"), ALL_TESTS, ids=[t["id"] for _, t in ALL_TESTS])
def test_generation_budget_is_at_least_one_token(category: str, test: dict):
    """`num_predict: 0` asks the backend for an empty completion, so the answer
    can never match and the test fails for every model forever.

    `llm_benchmark_suite._test_num_predict` returns the value verbatim and
    `context_awareness.turn_budget` clamps to `max(0, ...)`, so 0 survives all
    the way to the request payload. Two tests shipped this way
    (mm_html_form_render, mm_node_fib) and neither had ever been run."""
    assert test["num_predict"] >= 1, f"{test['id']} can never produce an answer with num_predict=0"


@pytest.mark.parametrize(("category", "test"), ALL_TESTS, ids=[t["id"] for _, t in ALL_TESTS])
def test_type_is_known_when_present(category: str, test: dict):
    if "type" not in test:
        return
    assert test["type"] in VALID_TYPES, f"{test['id']} has unknown type {test['type']!r}"


def test_the_untyped_tests_are_exactly_the_three_known_ones():
    """A missing `type` makes `_infer_type` guess from the response, which is
    what silently routes a test to the wrong grader. The count is pinned so a
    fourth one is a decision, not a drift."""
    untyped = sorted(t["id"] for _, t in ALL_TESTS if "type" not in t)
    assert untyped == sorted(UNTYPED_TESTS)
    assert len(untyped) == 3, "a fourth untyped test needs a deliberate decision"


@pytest.mark.parametrize(("category", "test"), ALL_TESTS, ids=[t["id"] for _, t in ALL_TESTS])
def test_lang_and_framework_are_known(category: str, test: dict):
    if "lang" in test:
        assert test["lang"] in VALID_LANGS, f"{test['id']} has unknown lang {test['lang']!r}"
    if "framework" in test:
        assert test["framework"] in VALID_FRAMEWORKS, f"{test['id']} has unknown framework {test['framework']!r}"


@pytest.mark.parametrize(("category", "test"), ALL_TESTS, ids=[t["id"] for _, t in ALL_TESTS])
def test_reasoning_estimate_is_one_of_the_generated_tiers(category: str, test: dict):
    """`scripts/gen_reasoning_estimates.py` emits these; an off-tier value means
    the doubling maths in `_test_num_predict` produces a budget nobody chose."""
    assert test["reasoning_estimate"] in REASONING_ESTIMATE_TIERS, (
        f"{test['id']} estimate {test['reasoning_estimate']} is not a generated tier"
    )


@pytest.mark.parametrize(("category", "test"), ALL_TESTS, ids=[t["id"] for _, t in ALL_TESTS])
def test_reasoning_budget_is_a_sane_tier(category: str, test: dict):
    if "reasoning_budget" not in test:
        return
    budget = test["reasoning_budget"]
    assert isinstance(budget, int) and budget > 0
    # Only the documented tiers, plus the two deliberately-heavy retrogames.
    assert budget in {512, 1024, 2048, 6000, 8000}, f"{test['id']} budget {budget} is off-tier"


def test_a_heavy_reasoning_budget_is_only_used_on_expensive_tests():
    """A 6000/8000 budget doubles the output cap (`base + 2*estimate`); spending
    it on a 300-token multiple-choice question would cost 10x for nothing."""
    for _category, test in ALL_TESTS:
        if test.get("reasoning_budget", 0) >= 6000:
            assert test["num_predict"] >= 4000, f"{test['id']} pairs a huge budget with a tiny cap"


# --------------------------------------------------------------------------
# identity and uniqueness
# --------------------------------------------------------------------------


def test_test_ids_are_globally_unique():
    """Result records are keyed by test id, so a duplicate silently merges two
    different tests' scores."""
    dupes = [tid for tid, n in Counter(t["id"] for _, t in ALL_TESTS).items() if n > 1]
    assert dupes == []


def test_test_ids_are_slug_shaped():
    for _, test in ALL_TESTS:
        assert re.fullmatch(r"[a-z0-9_]+", test["id"]), f"{test['id']} is not a snake_case id"


def test_no_two_tests_share_a_prompt():
    real = [t["prompt"] for _, t in ALL_TESTS if t["id"] not in SYNTHESIZED_PROMPT_TESTS]
    dupes = [prompt[:60] for prompt, n in Counter(real).items() if n > 1]
    assert dupes == []


def test_the_synthesized_prompt_placeholders_are_exactly_two():
    empty = sorted(t["id"] for _, t in ALL_TESTS if not t["prompt"])
    assert empty == sorted(SYNTHESIZED_PROMPT_TESTS)


def test_category_keys_are_snake_case_and_have_no_tests_in_the_category_field():
    for category, test in ALL_TESTS:
        assert re.fullmatch(r"[a-z0-9_]+", category), f"category {category!r} is not snake_case"
        # 19 tests carry a redundant `category` key; if it ever disagrees with
        # the parent key, `arcade_publish.iter_eligible_game_results` reads the
        # wrong one.
        if "category" in test:
            assert test["category"] == category, f"{test['id']} disagrees with its bucket"


def test_every_category_has_at_least_one_test_that_actually_runs():
    for category in CATEGORIES:
        runnable = [t for c, t in ALL_TESTS if c == category and t["num_predict"] >= 1]
        assert runnable, f"category {category} has nothing runnable"


# --------------------------------------------------------------------------
# answers: keys must be offered, and must be balanced
# --------------------------------------------------------------------------


def _letter_keyed() -> list[tuple[str, dict]]:
    return [
        (c, t)
        for c, t in ALL_TESTS
        if isinstance(t.get("expected"), str) and re.fullmatch(r"[A-E]", t["expected"].strip())
    ]


def test_every_letter_answer_is_one_of_the_options_the_prompt_offers():
    """A key pointing at a letter the prompt never lists is unpassable, and
    `scripts/rebalance_choice_keys.py` shuffles the options without checking
    the prompt still offers every letter."""
    for _category, test in _letter_keyed():
        offered = set(re.findall(r"^\s*([A-E])[\.\)]\s", test["prompt"], re.M))
        assert test["expected"].strip() in offered, (
            f"{test['id']} keys {test['expected']!r} but the prompt offers {sorted(offered)}"
        )


def test_no_single_answer_letter_dominates_the_keyed_tests():
    """AGENTS.md claims rebalancing spread the keys so nothing beats 25%."""
    counts = Counter(t["expected"].strip() for _, t in _letter_keyed())
    total = sum(counts.values())
    assert total >= 30, "the keyed sample shrank; re-check the balance rule"
    worst = max(counts.values()) / total
    assert worst <= 0.30, f"letter distribution is skewed: {dict(counts)}"
    assert len(counts) >= 4, f"too few distinct letters: {dict(counts)}"


def test_numeric_answers_are_real_finite_numbers():
    # `json.loads` accepts the non-standard NaN / Infinity literals by default,
    # and `float("nan") == float("nan")` is False, so a NaN answer would be
    # graded as "never correct" with no error anywhere.
    for _category, test in ALL_TESTS:
        expected = test.get("expected")
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            assert math.isfinite(expected), f"{test['id']} has a non-finite answer"
        if isinstance(expected, str) and re.fullmatch(r"-?\d+(\.\d+)?", expected.strip()):
            assert expected.strip() == expected, f"{test['id']} has a padded numeric answer"


def test_expected_output_values_are_non_empty_strings():
    for _category, test in ALL_TESTS:
        if "expected_output" in test:
            out = test["expected_output"]
            assert isinstance(out, str) and out.strip(), f"{test['id']} has an empty expected_output"


def test_a_functional_test_with_an_expected_answer_states_a_type():
    """`expected` is only consulted by the functional grader; a test typed as
    `ui` or `code` with an `expected` is checked by a grader that ignores it."""
    for _category, test in ALL_TESTS:
        if "expected" in test and "type" in test:
            assert test["type"] == "functional", (
                f"{test['id']} is typed {test['type']!r} but carries an `expected` answer "
                "the objective grader will not read"
            )


# --------------------------------------------------------------------------
# attachments
# --------------------------------------------------------------------------


def test_a_prompt_that_refers_to_an_attachment_carries_one():
    """The runner never forwards attachments, so a prompt that says "the
    attached X" is asking about something the model cannot see."""
    offenders = sorted(
        test["id"]
        for _, test in ALL_TESTS
        if re.search(r"\battached\b", test["prompt"], re.I)
        and not test.get("attachments")
        and test["id"] not in KNOWN_UNWIRED_ATTACHMENT_TESTS
    )
    assert offenders == [], f"these prompts need an attachment the runner does not send: {offenders}"


def test_the_unwired_attachment_debt_has_not_grown():
    """Shrink-only allow-list. Wire the runner, then delete entries here."""
    still = {
        test["id"]
        for _, test in ALL_TESTS
        if re.search(r"\battached\b", test["prompt"], re.I) and not test.get("attachments")
    }
    assert still == KNOWN_UNWIRED_ATTACHMENT_TESTS, (
        f"the set of attachment-dependent-but-unattached tests changed: {sorted(still)}"
    )


def test_attachment_entries_are_well_formed():
    for _category, test in ALL_TESTS:
        for att in test.get("attachments", []):
            assert isinstance(att, dict) and att.get("name")
            assert {"data_base64", "text", "data", "path", "url"} & set(att), f"{test['id']}: no payload"


# --------------------------------------------------------------------------
# relationships with the code that consumes the file
# --------------------------------------------------------------------------


def test_every_arcade_game_category_exists_in_the_test_file():
    """`web/arcade_publish.GAME_CATEGORIES` gates bulk publishing. A category
    that is renamed here but not there stops games being published at all."""
    import web.arcade_publish as ap

    missing = sorted(set(ap.GAME_CATEGORIES) - set(CATEGORIES))
    assert missing == []


def test_the_tool_test_types_are_declared_even_though_they_were_unused():
    """`llm_benchmark_suite.TOOL_TEST_TYPES` is a live dispatch branch. It sat
    dead for the whole life of the suite because no test used those types, so
    the enum is asserted here to stay reachable."""
    from llm_benchmark_suite import LLMModelBenchmark

    tool_types = LLMModelBenchmark.TOOL_TEST_TYPES
    assert tuple(tool_types) == ("image", "tts", "music", "composite")
    assert set(tool_types) <= VALID_TYPES


def test_the_computed_hash_is_stable_and_only_depends_on_graded_fields():
    from web.app import _compute_test_hash

    category, test = next((c, t) for c, t in ALL_TESTS if t["id"] == "story_start")
    once = _compute_test_hash(test)
    assert once == _compute_test_hash(test)
    assert isinstance(once, str) and once

    # A change the grader reads must move the hash, or `outdated_only` will
    # never re-run the test after its rubric changed.
    for field, value in (("prompt", test["prompt"] + " Now also answer twice."), ("expected", "zzz"), ("type", "review")):
        mutated = dict(test)
        mutated[field] = value
        assert _compute_test_hash(mutated) != once, f"mutating {field} did not move the hash"

    # A change no grader reads must NOT move it, or every run re-runs 289 tests.
    for field, value in (("num_predict", test["num_predict"] + 1), ("label", test["label"] + "!")):
        mutated = dict(test)
        mutated[field] = value
        assert _compute_test_hash(mutated) == once, f"mutating {field} moved the hash for no reason"
    assert category == "creative"


def test_the_hash_is_whitespace_insensitive_but_content_sensitive():
    """`_compute_test_hash` strips prompt/expected. Padding a prompt must not
    invalidate a stored result, but changing a word must."""
    from web.app import _compute_test_hash

    _, test = BY_ID["story_start"]
    base = _compute_test_hash(test)
    assert _compute_test_hash({**test, "prompt": f"  {test['prompt']}  "}) == base
    assert _compute_test_hash({**test, "prompt": test["prompt"].replace("sci-fi", "cyberpunk")}) != base


def test_the_hash_tracks_every_grader_version():
    """A grader change with an unchanged prompt must move the hash, or
    `outdated_only` would keep serving results scored by retired rules."""
    import llm_benchmark_suite
    import web.app as wa

    # per-test functional grader versions
    graded_id = next(iter(wa.FUNCTIONAL_GRADER_VERSIONS))
    _, graded = BY_ID[graded_id]
    with pytest.MonkeyPatch.context() as mp:
        before = wa._compute_test_hash(graded)
        mp.setitem(wa.FUNCTIONAL_GRADER_VERSIONS, graded_id, "v999")
        assert wa._compute_test_hash(graded) != before

    # the objective grader, applied to every test carrying an `expected`
    _, keyed = next((c, t) for c, t in ALL_TESTS if t.get("expected"))
    with pytest.MonkeyPatch.context() as mp:
        before = wa._compute_test_hash(keyed)
        mp.setattr(wa, "OBJECTIVE_GRADER_VERSION", "v999")
        assert wa._compute_test_hash(keyed) != before

    # the code grader + the appended directive, applied to code/ui tests only
    _, code = next((c, t) for c, t in ALL_TESTS if t.get("type") == "code")
    with pytest.MonkeyPatch.context() as mp:
        before = wa._compute_test_hash(code)
        mp.setattr(llm_benchmark_suite.LLMModelBenchmark, "CODE_GRADER_VERSION", "v999")
        assert wa._compute_test_hash(code) != before
    with pytest.MonkeyPatch.context() as mp:
        before = wa._compute_test_hash(code)
        mp.setattr(llm_benchmark_suite.LLMModelBenchmark, "GRADER_DIRECTIVE_VERSION", "v999")
        assert wa._compute_test_hash(code) != before

    # ...and a functional test must be untouched by the code grader
    _, functional = next((c, t) for c, t in ALL_TESTS if t.get("type") == "functional" and t.get("expected"))
    with pytest.MonkeyPatch.context() as mp:
        before = wa._compute_test_hash(functional)
        mp.setattr(llm_benchmark_suite.LLMModelBenchmark, "CODE_GRADER_VERSION", "v999")
        mp.setattr(llm_benchmark_suite.LLMModelBenchmark, "GRADER_DIRECTIVE_VERSION", "v999")
        assert wa._compute_test_hash(functional) == before


def test_the_hash_tracks_the_review_only_flag():
    from web.app import _compute_test_hash

    review = next(t for _, t in ALL_TESTS if t.get("review_only"))
    assert _compute_test_hash({**review, "review_only": False}) != _compute_test_hash(review)
    plain = next(t for _, t in ALL_TESTS if not t.get("review_only"))
    assert _compute_test_hash({**plain, "review_only": True}) != _compute_test_hash(plain)


def test_the_hash_tracks_attachment_names_only_not_their_payloads():
    """A test's attachment identity is its filename; swapping the bytes behind
    a filename is invisible to `outdated_only`, which is intentional (the
    runner does not send attachments at all - see the allow-list above)."""
    from web.app import _compute_test_hash

    probe = {"id": "probe", "prompt": "p", "type": "functional", "attachments": [{"name": "b.txt", "text": "1"}]}
    assert _compute_test_hash(probe) == _compute_test_hash(
        {**probe, "attachments": [{"name": "b.txt", "text": "2"}]}
    )
    assert _compute_test_hash(probe) != _compute_test_hash(
        {**probe, "attachments": [{"name": "a.txt", "text": "1"}]}
    )
    # attachment order is irrelevant
    two = {"id": "probe", "prompt": "p", "attachments": [{"name": "a"}, {"name": "b"}]}
    assert _compute_test_hash(two) == _compute_test_hash({**two, "attachments": [{"name": "b"}, {"name": "a"}]})


def test_every_versioned_test_id_still_exists():
    """`FUNCTIONAL_GRADER_VERSIONS` is keyed by test id. A renamed or deleted
    test leaves a dead entry that no longer versions anything."""
    from web.app import FUNCTIONAL_GRADER_VERSIONS

    dead = sorted(set(FUNCTIONAL_GRADER_VERSIONS) - set(BY_ID))
    assert dead == [], f"stale grader versions for tests that no longer exist: {dead}"


def test_the_current_grader_versions_are_pinned():
    """A silent bump of any of these re-runs part of the corpus, so it must be
    a visible edit rather than an accident."""
    from llm_benchmark_suite import LLMModelBenchmark
    from web.app import FUNCTIONAL_GRADER_VERSIONS, OBJECTIVE_GRADER_VERSION

    assert OBJECTIVE_GRADER_VERSION == "v2"
    assert LLMModelBenchmark.CODE_GRADER_VERSION == "v2"
    assert LLMModelBenchmark.GRADER_DIRECTIVE_VERSION == "v4"
    assert all(v.startswith("v") for v in FUNCTIONAL_GRADER_VERSIONS.values())


def test_adding_a_test_does_not_disturb_the_existing_corpus():
    """The property Phase 5-7 depend on: new tests are free, changed tests are
    not. If this ever fails, every stored result is invalidated at once."""
    from web.app import _compute_test_hash

    before = {t["id"]: _compute_test_hash(t) for _, t in ALL_TESTS}
    data = json.loads(TESTS_PATH.read_text())
    data["creative"].append(
        {
            "id": "schema_probe_added_by_a_test",
            "label": "probe",
            "prompt": "probe",
            "num_predict": 100,
            "reasoning_estimate": 1024,
            "type": "functional",
        }
    )
    after = {t["id"]: _compute_test_hash(t) for tests in data.values() for t in tests if t["id"] in before}
    assert after == before


def test_no_duplicate_prompt_between_the_shipped_file_and_a_new_test():
    """Belt and braces: the probe test above must not collide with a real one."""
    prompts = {t["prompt"] for _, t in ALL_TESTS}
    assert "probe" not in prompts
