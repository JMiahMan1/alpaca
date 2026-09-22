import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from llm_benchmark_suite import LLMModelBenchmark


@pytest.mark.parametrize("test_id", ["life_dad_joke", "uiux_wireframe", "debug_offbyone", "debug_infinite_loop"])
def test_corrected_functional_grades_are_outdated(test_id):
    import hashlib

    from web.app import _compute_test_hash

    test = {"id": test_id, "prompt": "unchanged prompt", "type": "functional"}
    old = {
        "id": test_id,
        "prompt": "unchanged prompt",
        "expected": "",
        "expected_output": "",
        "type": "functional",
        "kind": "text",
        "attachments": [],
        "grader_directive": "",
    }
    old_hash = hashlib.sha256(json.dumps(old, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
    assert _compute_test_hash(test) != old_hash


@pytest.fixture
def benchmark():
    return LLMModelBenchmark()


def criterion(benchmark, response, label, code=""):
    result = benchmark._evaluate_rubric({}, response, code)
    return next(item["passed"] for item in result["criteria"] if item["label"] == label)


@pytest.mark.parametrize(
    ("response", "complete"),
    [
        ("```html\n<html><script>newGame();</script></html>\n```", True),
        ("```js\nrender();\n```\n\n", True),
        ("```js\ninit();\n```\n```js\nrender();\n```", True),
        ("render();", True),
        ("```js\nrender();", False),
        ("```js\ninit();\n```\n```js\nrender();", False),
        ("render();...", False),
        ("render();\u2026", False),
        ("```js\nrender();\n...\n```", False),
    ],
)
def test_complete_fences_are_not_truncation(benchmark, response, complete):
    assert criterion(benchmark, response, "Code is complete, not truncated") is complete
    quality = benchmark._score_code_quality(response, {})
    assert ("appears truncated" not in quality["notes"]) is complete


@pytest.mark.parametrize(
    "code",
    [
        '<input placeholder="Initials">',
        '<textarea placeholder="Your notes"></textarea>',
        "input::placeholder {color: gray;}",
        ".drop-placeholder {height: 20px;}",
        'const placeholder = document.createElement("div");',
        'placeholder.className = "drop-placeholder";',
        'const placeholderElement = document.querySelector(".drop-placeholder");',
        "<!-- To Do column headers --><h2>To Do</h2>",
        'const todoItems = [{title: "To Do"}];',
        "def handle(value):\n    try:\n        return int(value)\n    except ValueError:\n        pass\n    return 0",
        "def handle(value):\n    pass\n    return value",
        "class InvalidMove(Exception):\n    pass",
        'class InvalidMove(Exception):\n    """An invalid board move."""\n    pass',
        "class Board:\n    size = 8\n\nclass ChessBoard(Board):\n    pass",
    ],
)
def test_legitimate_placeholder_uses(benchmark, code):
    response = f"```\n{code}\n```"
    assert criterion(benchmark, response, "No placeholder/stub text")
    assert "contains placeholder text" not in benchmark._score_code_quality(response, {})["notes"]


@pytest.mark.parametrize(
    "code",
    [
        "// TODO: implement storage",
        "function save() { /* FIXME: add persistence */ }",
        "<!-- TODO: build the board -->",
        "// placeholder for persistence",
        "your code here",
        "implement this",
        "insert code",
        "def save():\n    pass",
        "async def save(): pass",
        'def save():\n    """Save the board."""\n    pass',
        "class Board:\n    pass",
        'class Board:\n    """A board."""\n    pass',
        "class Board(object):\n    def save(self):\n        pass",
        "def save():\n    raise NotImplementedError()",
    ],
)
def test_real_stubs_still_lose_points(benchmark, code):
    response = f"```\n{code}\n```"
    assert not criterion(benchmark, response, "No placeholder/stub text")
    assert "contains placeholder text" in benchmark._score_code_quality(response, {})["notes"]


@pytest.mark.parametrize(
    "code",
    [
        'document.addEventListener("DOMContentLoaded", init);',
        'button.addEventListener("click", () => render());',
        'window.addEventListener("keydown", handleKey);',
        "window.onload = init;",
        "window.onload = () => render();",
        'function init() { board.textContent = "Ready"; }\ninit();',
        'function render() { board.textContent = "Ready"; }\nrender();',
        "function newGame() { score = 0; }\nnewGame();",
        "http.createServer(handler).listen(3000);",
        'const http = require("http"); const server = http.createServer(handler); server.listen(3000);',
        "requestAnimationFrame(frame);",
        "setInterval(tick, 1000);",
        'print("hello")',
    ],
)
def test_event_driven_and_immediate_entry_points(benchmark, code):
    assert criterion(benchmark, f"```\n{code}\n```", "Has a self-contained entry point / demo")


@pytest.mark.parametrize(
    "code",
    [
        '// document.addEventListener("DOMContentLoaded", init);',
        '/* button.addEventListener("click", render); */',
        '<!-- window.addEventListener("keydown", handleKey); -->',
        "// window.onload = init;\n// init();\n// render();\n// newGame();",
        "// http.createServer(handler).listen(3000);",
        "// requestAnimationFrame(frame);\n// setInterval(tick, 1000);",
        '# print("hello")',
        'const instructions = "newGame(); console.log(1);";',
        'const instructions = `document.addEventListener("click", render);`;',
        "function init() {}\nfunction render() {}\nfunction newGame() {}",
        'const eventName = "DOMContentLoaded";',
        "document.addEventListener;",
        "window.onload;",
        "const server = http.createServer(handler);",
    ],
)
def test_comments_and_definitions_are_not_startup(benchmark, code):
    assert not criterion(benchmark, f"```\n{code}\n```", "Has a self-contained entry point / demo")


@pytest.mark.parametrize(
    ("test_id", "snippet"),
    [
        ("game_snake_canvas", '<input placeholder="Initials"><script>requestAnimationFrame(frame);</script>'),
        ("game_checkers_web", '<input placeholder="Name"><script>newGame();</script>'),
        (
            "app_fake_desktop",
            "<style>input::placeholder{color:#637995}</style><script>tick(); setInterval(tick,1000);</script>",
        ),
        (
            "app_expense_tracker",
            '<input placeholder="Amount"><script>applyTheme();fillFilters();render();save();</script>',
        ),
        (
            "app_kanban_board",
            '<!-- To Do column headers --><style>.drop-placeholder{height:68px}</style><script>const placeholder = document.createElement("div"); button.addEventListener("click", render);</script>',
        ),
    ],
)
def test_audited_false_deductions_minimized(benchmark, test_id, snippet):
    response = f"```html\n<!doctype html><html>{snippet}</html>\n```"
    test = {"id": test_id, "type": "ui"}
    assert benchmark._evaluate_rubric(test, response)["score"] == 100
    assert "contains placeholder text" not in benchmark._score_code_quality(response, test)["notes"]


@pytest.fixture(scope="module")
def audited_results():
    path = Path(
        os.environ.get(
            "GRADING_AUDIT_SNAPSHOT", "/var/folders/h4/7cqby_zj7sjfm9v0pp2vwv9h0000gn/T/opencode/union-results.json"
        )
    )
    if not path.is_file():
        pytest.skip("Optional grading audit snapshot is not available")
    data = json.loads(path.read_text())
    return {
        test["test_id"]: test
        for model in data["results"]
        for key, category in model.items()
        if key.startswith("category_") and isinstance(category, dict)
        for test in category.get("tests", [])
    }


@pytest.mark.parametrize(
    ("test_id", "old_score"),
    [
        ("game_snake_canvas", 72),
        ("game_checkers_web", 58),
        ("app_fake_desktop", 72),
        ("app_expense_tracker", 72),
        ("app_kanban_board", 0),
    ],
)
def test_audited_snapshot_grading(benchmark, audited_results, test_id, old_score):
    original = audited_results[test_id]
    assert original["score"] == old_score
    test = {"id": test_id, "category": original["test_category"], "type": "ui"}
    response = original["response"]
    rubric = benchmark._evaluate_rubric(test, response)
    quality = benchmark._score_code_quality(response, test)
    assert rubric["score"] == 100
    assert "contains placeholder text" not in quality["notes"]
    result = {**original, "rubric": rubric, "code_quality": quality}
    score = benchmark._score_test(test, result)
    if test_id == "app_kanban_board":
        assert original["code_ran"] is False
        assert score == 0
    else:
        assert score > old_score


def test_kanban_syntax_failure_is_not_placeholder_failure(benchmark):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the syntax regression")
    code = 'const placeholder = "drop-placeholder";\n$("labelFilter").innerHTML+ = "";'
    response = f"```js\n{code}\n```"
    assert criterion(benchmark, response, "No placeholder/stub text")
    result = subprocess.run([node, "--check"], input=code, text=True, capture_output=True, timeout=10)
    assert result.returncode != 0
    assert "SyntaxError" in result.stderr
    assert benchmark._score_test({"type": "ui"}, {"response": response, "code_ran": False}) == 0


def test_kanban_snapshot_keeps_real_syntax_failure(audited_results):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the syntax regression")
    response = audited_results["app_kanban_board"]["response"]
    scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", response, re.S)
    result = subprocess.run([node, "--check"], input="\n".join(scripts), text=True, capture_output=True, timeout=10)
    assert result.returncode != 0
    assert "SyntaxError" in result.stderr


def test_rubric_does_not_override_execution_or_functional_failure(benchmark):
    test = {"id": "app_kanban_board", "type": "ui"}
    response = "```html\n<html><script>render();</script></html>\n```"
    result = {
        "response": response,
        "rubric": benchmark._evaluate_rubric(test, response),
        "code_quality": benchmark._score_code_quality(response, test),
        "code_ran": False,
        "functional_pass": True,
        "code_score": 100,
    }
    assert result["rubric"]["score"] == 100
    assert benchmark._score_test(test, result) == 0
    result.update(code_ran=True, functional_pass=False)
    assert benchmark._score_test(test, result) <= 45


@pytest.mark.parametrize("lang", ["python", "cpp"])
@pytest.mark.parametrize(
    "tag,code",
    [
        ("c", '#include <stdio.h>\nint main(void) { puts("ok"); }'),
        (
            "kotlin",
            'import retrofit2.http.GET\ninterface ApiService { @GET("users") suspend fun getUsers(): List<User> }',
        ),
    ],
)
def test_c_kotlin_fences_never_extract_interstitial_prose(lang, tag, code):
    from sandbox_exec import extract_clean_code

    response = f"```{tag}\n{code}\n```\nNotes on the pieces\n```{tag}\nshort\n```"
    assert extract_clean_code(response, lang) == code


def test_review_only_config_matches_explicit_prompts():
    config = json.loads((Path(__file__).parents[1] / "benchmark_tests.json").read_text())
    tests = [test for group in config.values() for test in group]
    explicit = {test["id"] for test in tests if "not compiled in the sandbox" in test["prompt"]}
    opted_in = {test["id"] for test in tests if test.get("review_only")}
    assert opted_in == explicit
    assert len(opted_in) == 12
    assert not any(test.get("review_only") for group in ("pascal", "iac") for test in config[group])
    usb = next(test for test in tests if test["id"] == "usb_interface_claim")
    assert "USBDEVFS_BULK ioctl with struct usbdevfs_bulktransfer" in usb["prompt"]
    assert "USBDEVFS_BULKTRANSFER" not in usb["prompt"]


def test_changed_functional_graders_are_recorded_for_rerun():
    """A grader can change while its prompt does not; the version table is what
    makes outdated_only re-run those tests instead of trusting a stale score."""
    from web.app import FUNCTIONAL_GRADER_VERSIONS, _compute_test_hash

    for test_id in ("logic_knights", "logic_river", "logic_modus", "logic_weigh"):
        assert test_id in FUNCTIONAL_GRADER_VERSIONS
        test = {"id": test_id, "prompt": "unchanged", "type": "functional"}
        ungraded = dict(test, id=f"{test_id}__unversioned")
        assert _compute_test_hash(test) != _compute_test_hash(ungraded)


def test_objectively_keyed_tests_rerun_after_the_answer_grader_change():
    from web.app import _compute_test_hash

    keyed = {"id": "math_hard_x", "prompt": "p", "type": "functional", "expected": "7"}
    unkeyed = {"id": "math_hard_x", "prompt": "p", "type": "functional"}
    assert _compute_test_hash(keyed) != _compute_test_hash(unkeyed)


def test_grading_version_invalidates_code_hash_only(monkeypatch):
    from web.app import _compute_test_hash

    assert LLMModelBenchmark.GRADER_DIRECTIVE_VERSION == "v4"
    code = {"id": "game_checkers_web", "type": "ui", "prompt": "Build checkers"}
    text = {"id": "story", "type": "knowledge", "prompt": "Tell a story"}
    current_code = _compute_test_hash(code)
    current_text = _compute_test_hash(text)
    monkeypatch.setattr(LLMModelBenchmark, "GRADER_DIRECTIVE_VERSION", "v2")
    assert _compute_test_hash(code) != current_code
    assert _compute_test_hash(text) == current_text
