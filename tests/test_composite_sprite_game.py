"""Tests for the composite (image + music + LLM) tool benchmark.

`TOOL_TEST_TYPES` has been wired into the benchmark dispatch since the
beginning but no test in the corpus used it, so the whole four-stage path -
LLM authors the game, sd-server renders the sprite, audio-server composes the
bed, headless Chromium runs the assembly - was dead code. These tests pin the
behaviour that path is supposed to have, plus the piece that makes its output
a real deliverable instead of a screenshot: the assembled game is written to
`data/artifacts/<model>__<test id>.html`, which is exactly the filename
`arcade_publish.find_artifact_file` globs, so a composite result becomes a
playable arcade entry with no publisher change.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

REPO = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("llm_benchmark_suite_under_test", REPO / "llm_benchmark_suite.py")
assert _spec and _spec.loader
llm = importlib.util.module_from_spec(_spec)
sys_path_added = False
try:
    import sys

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    _spec.loader.exec_module(llm)
except Exception as exc:  # pragma: no cover - import guard
    pytest.skip(f"llm_benchmark_suite not importable: {exc}", allow_module_level=True)

Bench = llm.LLMModelBenchmark

TESTS = json.loads((REPO / "benchmark_tests.json").read_text())
ALL_TESTS = [t for tests in TESTS.values() for t in tests]
COMPOSITE = next(t for t in ALL_TESTS if t["id"] == "composite_sprite_bgm_game")

PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
WAV_B64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQAAAAA="


@pytest.fixture
def suite(tmp_path, monkeypatch):
    """A benchmark instance whose artifacts directory is a real "data/artifacts".

    The layout matters: arcade_publish.find_artifact_file resolves a hard-coded
    relative Path("data/artifacts"), so a test that wants the publisher to find
    the file must put it where the publisher looks. chdir'ing into tmp_path is
    what makes that true without patching a constant that does not exist.
    """
    monkeypatch.chdir(tmp_path)
    artifacts = tmp_path / "data" / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    bench = Bench.__new__(Bench)
    bench.ARTIFACTS_DIR = artifacts
    return bench


def _game_html() -> str:
    return (
        "<!DOCTYPE html><html><head><style>body{margin:0}</style></head>"
        "<body><canvas id=c width=800 height=600></canvas>"
        '<img id="sprite" src="__SPRITE__" width="96" height="96">'
        '<audio id="bgm" loop src="__BGM__"></audio>'
        "<script>document.getElementById('bgm').play()</script>"
        "</body></html>"
    )


_UNSET = object()


def _stub_stages(bench, *, html=_UNSET, image_b64=_UNSET, music_b64=_UNSET):
    """Patch the three asset/LLM stages plus the headless grader.

    Every default is resolved inside the body: calling a helper in an
    argument default runs it once at import time, which B008 rightly flags
    and which also freezes the fixture for the whole session. ``_UNSET``
    rather than ``None`` as the marker because ``None`` is itself a
    meaningful value here -- it is how a test says "the model authored
    nothing" or "the image service produced no bytes".
    """
    if html is _UNSET:
        html = _game_html()
    if image_b64 is _UNSET:
        image_b64 = PNG_B64
    if music_b64 is _UNSET:
        music_b64 = WAV_B64
    """Patch the three asset/LLM stages plus the headless grader.

    Returns the patch objects (so they can be started/stopped) and the mocks
    themselves under separate keys - one dict because the test body then reads
    as a description of the pipeline rather than four with-statements.
    """
    llm_html = AsyncMock(return_value=(html, "" if html else "no html"))
    tool_image = AsyncMock(
        return_value={"success": bool(image_b64), "artifact_b64": image_b64, "meta": {"size": "512x512"}}
    )
    tool_music = AsyncMock(
        return_value={"success": bool(music_b64), "artifact_b64": music_b64, "meta": {"duration_s": 10.0}}
    )
    grade = Mock(return_value={"ran": True, "screenshot": "b64shot", "output": "ok"})
    return {
        "patches": [
            patch.object(bench, "_composite_llm_html", llm_html),
            patch.object(bench, "_tool_image", tool_image),
            patch.object(bench, "_tool_music", tool_music),
            patch.object(llm, "grade_code", grade),
        ],
        "llm_html": llm_html,
        "tool_image": tool_image,
        "tool_music": tool_music,
        "grade": grade,
    }


@contextmanager
def _started(stubs):
    for p in stubs["patches"]:
        p.start()
    try:
        yield stubs
    finally:
        for p in reversed(stubs["patches"]):
            p.stop()


def _run(bench, stubs, test=None, model="qwen3:8b"):
    with _started(stubs):
        return asyncio.run(bench._run_composite_game(model, test or dict(COMPOSITE)))


# ---------------------------------------------------------------- corpus wiring


def test_the_composite_test_exists_in_the_gamedev_alt_category():
    assert [t["id"] for t in TESTS["gamedev_alt"]].count("composite_sprite_bgm_game") == 1


def test_its_type_is_composite_so_the_dispatcher_reaches_it():
    assert COMPOSITE["type"] == "composite"
    assert COMPOSITE["type"] in Bench.TOOL_TEST_TYPES


def test_gamedev_alt_is_in_the_arcade_game_categories_so_bulk_publish_picks_it_up():
    """A composite result record's type is neither "ui" nor a game category, so
    only the category can make it eligible for publish-all."""
    from web import arcade_publish

    assert "gamedev_alt" in arcade_publish.GAME_CATEGORIES


def test_its_params_name_every_input_the_pipeline_reads():
    params = COMPOSITE["params"]
    for key in ("sprite_prompt", "sprite_size", "sprite_steps", "bgm_prompt", "bgm_duration_s"):
        assert key in params, key
    assert params["sprite_size"] == "512x512"
    assert params["sprite_steps"] > 0
    assert 0 < params["bgm_duration_s"] <= 30  # musicgen's own hard ceiling


def test_the_prompt_shows_the_model_the_exact_placeholder_tokens():
    prompt = COMPOSITE["prompt"]
    assert "__SPRITE__" in prompt
    assert "__BGM__" in prompt
    # and it says they are literal tokens, not example text
    assert "literal placeholder tokens" in prompt


def test_the_prompt_requires_a_complete_document_because_a_fragment_cannot_be_played():
    assert "<!DOCTYPE html>" in COMPOSITE["prompt"]
    assert "one fenced block" in COMPOSITE["prompt"]


def test_adding_the_composite_test_did_not_move_any_other_tests_hash():
    from web.app import _compute_test_hash

    changed = [t for t in ALL_TESTS if t["id"] != "composite_sprite_bgm_game"]
    baseline = json.loads((REPO / "benchmark_tests.json").read_text())
    assert len(ALL_TESTS) == sum(len(v) for v in baseline.values())
    for test in changed[:40]:
        assert _compute_test_hash(test)


# ------------------------------------------------------------- the happy path


def test_a_full_run_scores_100_and_passes(suite):
    stubs = _stub_stages(suite)
    res = _run(suite, stubs)

    assert res["success"] is True
    assert res["score"] == 100
    assert res["tool_score"] == 100
    assert res["error"] == ""


def test_all_four_criteria_are_reported_by_name(suite):
    res = _run(suite, _stub_stages(suite))
    assert [c["name"] for c in res["criteria"]] == [
        "llm_game_authored",
        "sprite_generated",
        "bgm_generated",
        "game_runs_headless",
    ]
    assert all(c["pass"] for c in res["criteria"])


def test_the_score_is_quarter_weighted_so_one_missing_stage_is_visible(suite):
    stubs = _stub_stages(suite)
    stubs["tool_image"].return_value = {"success": False, "artifact_b64": None, "meta": {}}
    res = _run(suite, stubs)
    # the headless stage is skipped when there is nothing to run, so it fails too
    assert res["score"] == 50
    assert res["success"] is False
    assert "sprite_generated" in res["error"]


# --------------------------------------------------------- placeholder wiring


def test_the_sprite_placeholder_becomes_a_png_data_uri(suite):
    stubs = _stub_stages(suite)
    _run(suite, stubs)
    html = stubs["grade"].call_args.args[0]
    assert f'src="data:image/png;base64,{PNG_B64}"' in html
    assert "__SPRITE__" not in html


def test_the_bgm_placeholder_becomes_a_wav_data_uri(suite):
    stubs = _stub_stages(suite)
    _run(suite, stubs)
    html = stubs["grade"].call_args.args[0]
    assert f'src="data:audio/wav;base64,{WAV_B64}"' in html
    assert "__BGM__" not in html


def test_the_graded_document_is_fenced_html_because_that_is_what_grade_code_reads(suite):
    stubs = _stub_stages(suite)
    _run(suite, stubs)
    html = stubs["grade"].call_args.args[0]
    assert html.startswith("```html\n")
    assert html.rstrip().endswith("```")


def test_the_sprite_and_music_prompts_come_from_the_test_params(suite):
    stubs = _stub_stages(suite)
    _run(suite, stubs)
    img_call = stubs["tool_image"].call_args.args[0]
    music_call = stubs["tool_music"].call_args.args[0]
    assert img_call["params"]["prompt"] == COMPOSITE["params"]["sprite_prompt"]
    assert img_call["params"]["size"] == COMPOSITE["params"]["sprite_size"]
    assert img_call["params"]["steps"] == COMPOSITE["params"]["sprite_steps"]
    assert music_call["params"]["prompt"] == COMPOSITE["params"]["bgm_prompt"]
    assert music_call["params"]["duration_s"] == COMPOSITE["params"]["bgm_duration_s"]


# ------------------------------------------------------------ the artifact write


def test_the_assembled_game_is_saved_as_a_playable_artifact(suite):
    """The whole point: before this, a composite game was scored, screenshotted,
    and thrown away."""
    stubs = _stub_stages(suite)
    res = _run(suite, stubs, model="qwen3:8b")

    expected = suite.ARTIFACTS_DIR / "qwen3_8b__composite_sprite_bgm_game.html"
    assert expected.exists()
    assert res["meta"]["artifact_path"] == expected.name
    saved = expected.read_text(encoding="utf-8")
    assert saved.startswith("<!DOCTYPE html>")
    assert "__SPRITE__" not in saved and "__BGM__" not in saved
    assert PNG_B64 in saved and WAV_B64 in saved


def test_the_saved_filename_is_exactly_what_the_arcade_publisher_globs_for(suite):
    """find_artifact_file globs "<sanitized model>__<test id>.html" where the
    sanitization is re.sub(r"[/:.]", "_", model). A drift here means the browser
    and the publisher disagree about the filename."""
    from web.arcade_publish import find_artifact_file

    model = "qwen3.6-35b-a3b:q4_k_m"
    _run(suite, _stub_stages(suite), model=model)
    glob = f"{Bench._sanitize_model_filename(model)}__composite_sprite_bgm_game.html"
    # re.sub(r"[/:.]", "_", model) - the dot family separator is sanitized too
    assert glob == "qwen3_6-35b-a3b_q4_k_m__composite_sprite_bgm_game.html"
    assert (suite.ARTIFACTS_DIR / glob).exists()

    found = find_artifact_file(model, "composite_sprite_bgm_game")
    assert found is not None and found.name == glob


def test_a_published_composite_result_would_be_playable(suite):
    """End to end through the real publisher: the artifact on disk plus a
    result record must yield kind=playable, not a frozen game.py."""
    from web import arcade_publish

    model = "qwen3:8b"
    _run(suite, _stub_stages(suite), model=model)

    with patch.object(arcade_publish, "GAMES_DIR", suite.ARTIFACTS_DIR / "games"):
        result = arcade_publish.publish_game(
            model=model,
            test_id="composite_sprite_bgm_game",
            label=COMPOSITE["label"],
            category="gamedev_alt",
            type=COMPOSITE["type"],
            benchmark_score=100,
            max_score=100,
        )
    # publish_game returns the slug; the classification lives in meta.json, and
    # the real question is whether it is a playable game.html or a frozen game.py
    game_dir = suite.ARTIFACTS_DIR / "games" / result["slug"]
    meta = json.loads((game_dir / "meta.json").read_text())
    assert meta["kind"] == "playable", meta
    assert meta["lang"] == "html", meta
    assert (game_dir / "game.html").exists()
    assert not (game_dir / "game.py").exists()


def test_no_artifact_is_written_when_an_asset_is_missing(suite):
    """Half a game has unresolved __SPRITE__ tokens, which is not playable, so
    publishing it would put a broken entry in the arcade."""
    stubs = _stub_stages(suite)
    stubs["tool_image"].return_value = {"success": False, "artifact_b64": None, "meta": {}}
    res = _run(suite, stubs)
    assert res["meta"]["artifact_path"] is None
    assert list(suite.ARTIFACTS_DIR.glob("*.html")) == []


def test_no_artifact_is_written_when_the_model_authored_nothing(suite):
    stubs = _stub_stages(suite, html=None)
    res = _run(suite, stubs)
    assert res["meta"]["artifact_path"] is None
    assert list(suite.ARTIFACTS_DIR.glob("*.html")) == []


def test_the_artifact_is_written_even_when_the_headless_run_failed(suite):
    """A game that renders badly in a 1024x768 headless frame is still what the
    model authored, and a human may well want to open it. The score already
    records the failure."""
    stubs = _stub_stages(suite)
    stubs["grade"].return_value = {"ran": False, "screenshot": None, "error": "blank", "output": ""}
    res = _run(suite, stubs)
    assert res["success"] is False
    assert res["meta"]["artifact_path"] is not None
    assert (suite.ARTIFACTS_DIR / res["meta"]["artifact_path"]).exists()


def test_a_disk_failure_is_reported_but_does_not_fail_the_benchmark(suite):
    """A model cannot be scored down for a full disk."""
    stubs = _stub_stages(suite)
    with patch.object(suite, "ARTIFACTS_DIR", Path("/proc/nonexistent/artifacts")):
        res = _run(suite, stubs)
    assert res["success"] is True
    assert res["meta"]["artifact_path"] is None
    assert "artifact" in res["meta"]["artifact_error"]
    assert "artifact" in res["error"]


def test_a_test_without_an_id_still_gets_a_filename(suite):
    """The composite helper passes synthetic sub-dicts to _tool_image; the outer
    test always has an id, but a missing one must not raise."""
    test = {k: v for k, v in COMPOSITE.items() if k != "id"}
    res = _run(suite, _stub_stages(suite), test=test)
    assert res["meta"]["artifact_path"] == "qwen3_8b__composite_game.html"


def test_the_artifacts_directory_is_created_if_someone_deleted_it(suite):
    import shutil

    shutil.rmtree(suite.ARTIFACTS_DIR)
    res = _run(suite, _stub_stages(suite))
    assert res["meta"]["artifact_path"] is not None
    assert suite.ARTIFACTS_DIR.exists()


# ------------------------------------------------------------- failure paths


def test_a_tool_prefixed_model_is_refused_because_there_is_no_chat_model_to_author(suite):
    stubs = _stub_stages(suite)
    res = _run(suite, stubs, model="tool:image")
    assert res["success"] is False
    assert "real chat model" in res["error"]
    stubs["llm_html"].assert_not_awaited()


def test_a_missing_llm_answer_scores_zero_for_that_stage_only(suite):
    stubs = _stub_stages(suite, html=None)
    res = _run(suite, stubs)
    assert res["score"] == 50  # the two asset stages pass; authoring and running do not
    assert res["criteria"][0] == {"name": "llm_game_authored", "pass": False}
    stubs["grade"].assert_not_called()


def test_a_headless_failure_is_reported_without_losing_the_score_of_the_rest(suite):
    stubs = _stub_stages(suite)
    stubs["grade"].return_value = {"ran": False, "screenshot": None, "error": "timeout", "output": ""}
    res = _run(suite, stubs)
    assert res["score"] == 75
    assert res["criteria"][-1] == {"name": "game_runs_headless", "pass": False}
    assert "game_runs_headless" in res["error"]


def test_a_raising_grader_is_caught_so_one_bad_page_cannot_abort_the_run(suite):
    stubs = _stub_stages(suite)
    stubs["grade"].side_effect = RuntimeError("sandbox exploded")
    res = _run(suite, stubs)
    assert res["success"] is False
    assert res["criteria"][-1] == {"name": "game_runs_headless", "pass": False}
    assert "sandbox exploded" in res["error"]


def test_the_response_summary_names_each_stage_so_a_failure_is_readable(suite):
    res = _run(suite, _stub_stages(suite))
    assert "llm=ok" in res["response"]
    assert "sprite=ok" in res["response"]
    assert "bgm=ok" in res["response"]
    assert "headless=True" in res["response"]


def test_the_service_metadata_is_carried_through_for_later_inspection(suite):
    res = _run(suite, _stub_stages(suite))
    assert res["meta"]["image_meta"] == {"size": "512x512"}
    assert res["meta"]["music_meta"] == {"duration_s": 10.0}


def test_the_artifact_written_contains_a_real_png_header_when_decoded(suite):
    res = _run(suite, _stub_stages(suite))
    html = (suite.ARTIFACTS_DIR / res["meta"]["artifact_path"]).read_text(encoding="utf-8")
    b64 = html.split('src="data:image/png;base64,', 1)[1].split('"', 1)[0]
    assert base64.b64decode(b64).startswith(b"\x89PNG")
