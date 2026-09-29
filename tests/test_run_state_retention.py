"""The benchmark run state must not keep its result corpus after the run ends.

Why this file exists
--------------------
``active_run`` is a module-level dict, and two of its keys -- ``results`` and
``test_results`` -- accumulate one entry per model / per test for the whole run.
Every entry is a result record, and a result record carries the model's entire
response, which for a UI test includes a base64 PNG screenshot. Measured on a
real run, that is tens of megabytes: the newest stored snapshot is 18 model
result sets and re-serialises to 17.1 MB, of which 6.3 MB is 174 base64
screenshots.

During a run that corpus is genuinely wanted -- ``GET /api/status`` feeds it to
the dashboard so the charts animate live. After the run it is wanted by nobody:
the Results tab loads ``/api/results`` instead. But both ``GET /api/status`` and
the SocketIO ``sync_status`` emitted on every connect serialise the whole dict,
so a retained corpus is re-sent on every poll, indefinitely, for data that is
already written to ``saved_as``.

The tests below pin the release. The one that matters most is
``test_status_stays_small_after_a_run_finishes``: it reproduces the actual
symptom (a status payload the size of the corpus) rather than asserting on
internal state, so it fails loudly for the real reason if the corpus is ever
retained again.
"""

import ast
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from web.app import _RUN_SCRATCH_KEYS, active_run, active_run_lock, app

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_PY = REPO_ROOT / "web" / "app.py"
DASHBOARD_JS = REPO_ROOT / "web" / "static" / "js" / "dashboard.js"

# A base64 PNG screenshot of the size grade_code actually returns for a 1024x768
# UI render. One of these is ~700 KB of response body on its own.
FAKE_SCREENSHOT_B64 = "iVBORw0KGgoAAAANSUhEUg" + "A" * 700_000


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        _reset_run_state()
        yield test_client
        _reset_run_state()


def _reset_run_state():
    with active_run_lock:
        active_run.update(
            {
                "status": "idle",
                "type": None,
                "current_model": None,
                "current_test": None,
                "current_category": None,
                "tests_completed": 0,
                "total_tests": 0,
                "models": [],
                "use_proxy": True,
                "results": [],
                "test_results": [],
                "start_time": None,
                "saved_as": None,
            }
        )


def _load_corpus(models=3):
    """Fill active_run with a corpus shaped like a real finished run."""
    with active_run_lock:
        for i in range(models):
            active_run["results"].append(
                {
                    "model": f"model-{i}",
                    "tasks": [
                        {
                            "test_id": f"test-{i}-{j}",
                            "success": True,
                            "response": "x" * 2000,
                            "screenshot": FAKE_SCREENSHOT_B64,
                        }
                        for j in range(4)
                    ],
                }
            )
            active_run["test_results"].append({"test_id": f"test-{i}-0", "result": {}})
    return sum(len(json.dumps(r)) for r in active_run["results"])


@pytest.mark.parametrize("run_type", ["general", "shared_llm", "multistep"])
def test_a_completed_run_releases_its_result_corpus(client, run_type):
    from web.app import get_progress_callback

    callback = get_progress_callback(run_type)
    _load_corpus()
    assert active_run["results"], "precondition: a corpus is loaded"

    with patch("web.app.socketio.emit"):
        callback("benchmark_complete", {"status": "completed", "saved_as": "data/x.json", "results": []})

    assert active_run["results"] == []
    assert active_run["test_results"] == []
    assert active_run["status"] == "completed"
    assert active_run["current_model"] is None
    assert active_run["current_test"] is None
    assert active_run["current_category"] is None


def test_a_cancelled_run_releases_its_result_corpus(client):
    from web.app import get_progress_callback

    callback = get_progress_callback("general")
    _load_corpus()

    with patch("web.app.socketio.emit"):
        callback("benchmark_cancelled", {})

    assert active_run["status"] == "cancelled"
    assert active_run["results"] == []
    assert active_run["test_results"] == []


def test_the_saved_path_outlives_the_release(client):
    """Releasing the corpus must not lose where the run was written."""
    from web.app import get_progress_callback

    callback = get_progress_callback("general")
    _load_corpus()
    with patch("web.app.socketio.emit"), patch("web.app.model_tracker.record_benchmark_result"):
        callback("benchmark_complete", {"status": "completed", "saved_as": "data/llm_benchmarks/x.json", "results": []})

    assert active_run["saved_as"] == "data/llm_benchmarks/x.json"
    assert active_run["results"] == []


@pytest.mark.parametrize("run_type", ["general", "shared_llm", "multistep"])
def test_status_stays_small_after_a_run_finishes(client, run_type):
    """The symptom, not the mechanism.

    While a run is live the corpus is meant to be streamed -- that is how the
    charts animate. Once it is over, GET /api/status must not keep re-sending
    megabytes of base64 screenshots on every poll.
    """
    from web.app import get_progress_callback

    callback = get_progress_callback(run_type)
    corpus_bytes = _load_corpus(models=3)
    assert corpus_bytes > 1_000_000, "precondition: the corpus is genuinely large"

    with patch("web.app.socketio.emit"):
        callback("benchmark_complete", {"status": "completed", "saved_as": "data/x.json", "results": []})

    body = client.get("/api/status").data
    assert len(body) < 10_000, f"/api/status is {len(body)} bytes after a run with a {corpus_bytes}-byte corpus"
    assert b"AAA" not in body, "a base64 screenshot survived into the status payload"
    assert json.loads(body)["status"] == "completed"


def test_the_corpus_is_still_streamed_while_the_run_is_live(client):
    """The counter-test: the release must not break live progress.

    If this fails, the fix has gone too far -- the dashboard animates its charts
    from GET /api/status during a run, so the corpus has to be there while the
    run is running and only released when it stops.
    """
    from web.app import get_progress_callback

    callback = get_progress_callback("general")
    with patch("web.app.socketio.emit"):
        callback("benchmark_start", {"models": ["m"], "use_proxy": True, "total_tests": 2, "timestamp": "t"})
        callback("test_complete", {"model": "m", "category": "c", "test_id": "t", "test_label": "T", "result": {}})

    assert active_run["status"] == "running"
    assert active_run["test_results"], "live progress lost its per-test results"
    assert len(active_run["test_results"]) == 1


def test_live_status_still_carries_the_corpus(client):
    """While running, the status payload is expected to be large. That is fine."""
    from web.app import get_progress_callback

    callback = get_progress_callback("general")
    with patch("web.app.socketio.emit"):
        callback("benchmark_start", {"models": ["m"], "use_proxy": True, "total_tests": 2, "timestamp": "t"})
    _load_corpus(models=2)

    body = client.get("/api/status").data
    assert json.loads(body)["status"] == "running"
    assert len(body) > 1_000, "a running status payload should still carry the corpus"


def test_finish_active_run_keeps_a_previously_saved_path():
    """An omitted saved_as must not blank a path recorded earlier in the run."""
    from web.app import _finish_active_run

    with active_run_lock:
        active_run["saved_as"] = "data/earlier.json"
        active_run["results"] = [{"model": "m"}]
        _finish_active_run("cancelled")
        assert active_run["saved_as"] == "data/earlier.json"
        assert active_run["results"] == []


def test_finish_active_run_overwrites_saved_as_when_given():
    from web.app import _finish_active_run

    with active_run_lock:
        active_run["saved_as"] = "data/earlier.json"
        _finish_active_run("completed", "data/later.json")
        assert active_run["saved_as"] == "data/later.json"


def test_finish_active_run_clears_every_scratch_key():
    from web.app import _finish_active_run

    with active_run_lock:
        for key in _RUN_SCRATCH_KEYS:
            active_run[key] = [{"big": "x"}]
        _finish_active_run("completed")
    for key in _RUN_SCRATCH_KEYS:
        assert active_run[key] == [], f"{key} was not released"


def test_scratch_keys_are_never_aliased():
    """Rebinding instead of clearing would leave the caller's list referenced."""
    from web.app import _finish_active_run

    sentinel = [{"big": "x"}]
    with active_run_lock:
        active_run["results"] = sentinel
        _finish_active_run("completed")
        assert active_run["results"] is not sentinel


def test_the_corpus_keys_are_only_ever_written_when_a_run_starts():
    """A guard against someone re-adding a raw terminal-state assignment.

    Every write to active_run["results"] / ["test_results"] has to be one of:

    * the ``benchmark_start`` branch of the progress callback, or
    * a view that *starts* a run (``start_benchmark`` and its siblings),

    which is the one moment the corpus is legitimately discarded wholesale.
    Everything else must go through ``_finish_active_run``, or the retention
    comes back and no behavioural test would notice until the process was killed
    under memory pressure.
    """
    tree = ast.parse(APP_PY.read_text())
    writes = _scratch_key_writes(tree)
    assert writes, f"no writes to {_RUN_SCRATCH_KEYS} found -- the scan is broken, not the code"

    for lineno, key, func, in_benchmark_start in writes:
        allowed = in_benchmark_start or (func is not None and func.startswith("start_"))
        assert allowed, (
            f"web/app.py:{lineno} assigns active_run['{key}'] inside {func!r}, which neither starts a "
            "run nor finishes one. Route it through _finish_active_run so the corpus is released."
        )


def _scratch_key_writes(tree):
    """Yield (lineno, key, enclosing_function, inside_benchmark_start) for each write."""
    out = []

    def visit(node, func, in_start):
        for child in ast.iter_child_nodes(node):
            now_func = child.name if isinstance(child, ast.FunctionDef) else func
            now_start = in_start
            if isinstance(child, ast.If) and _tests_for(child.test, "benchmark_start"):
                now_start = True
            if (
                isinstance(child, ast.Assign)
                and any(
                    isinstance(t, ast.Subscript)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "active_run"
                    and isinstance(t.slice, ast.Constant)
                    and t.slice.value in _RUN_SCRATCH_KEYS
                    for t in child.targets
                )
                and child.lineno not in out_lines
            ):
                key = next(
                    t.slice.value
                    for t in child.targets
                    if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
                )
                out.append((child.lineno, key, func, in_start))
                out_lines.add(child.lineno)
            visit(child, now_func, now_start)

    out_lines: set[int] = set()
    visit(tree, None, False)
    return out


def _tests_for(test, value):
    return isinstance(test, ast.Compare) and any(
        isinstance(c, ast.Constant) and c.value == value for c in test.comparators
    )


def test_the_dashboard_run_poll_is_gated_on_tab_visibility():
    """A hidden tab must not keep polling.

    The two 2-second pollers already carry `if (!document.hidden)`; the 15s
    /api/status poller did not, so a backgrounded tab polled forever. It is
    pinned here because the gate is invisible to the DOM-load harness.
    """
    src = DASHBOARD_JS.read_text()
    gated = "if (!document.hidden) pollRunStatus()"
    assert gated in src, "the 15s /api/status poll is not gated on document.hidden"
    assert "setInterval(pollRunStatus, 15000)" not in src, "an ungated 15s poller is still registered"


def test_the_three_status_pollers_are_all_visibility_gated():
    src = DASHBOARD_JS.read_text()
    for fn in ("pollProxyStatus", "pollRequestsStatus", "pollRunStatus", "loadErrorLog", "updateCurrentModel"):
        assert f"if (!document.hidden) {fn}()" in src, f"{fn} is not gated on document.hidden"
