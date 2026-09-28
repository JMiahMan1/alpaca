"""Regression tests for the four dashboard bugs found in the Phase 2 audit.

Each of these was a real, user-visible failure that no test covered:

1. ``POST /api/proxy/restart`` was a no-op. It POSTed to ``/admin/restart``,
   which did not exist on the proxy, then fell back to a
   ``subprocess.run(["docker", ...])`` - but ``Dockerfile.web`` installs curl
   only, so that raised FileNotFoundError and was swallowed. "Save & Restart
   Backend" reported success and then timed out in the browser. The frontend
   compounded it by ignoring the response body entirely.
2. ``applyAnalysisRec`` was declared inside the DOMContentLoaded closure but
   invoked by an inline ``onclick``, so "Apply to Profile" raised
   ReferenceError on every click.
3. The API Docs tab filed ``/api/requests`` and ``/api/requests/clear`` under a
   heading reading "Port 11434". They are dashboard-only (:5000); the proxy's
   equivalents are ``/admin/requests*``. A copied curl 404s.
4. ``GET /api/analyze/all`` returned ``results`` on success and ``models`` on
   error, so a client reading ``.results`` got ``undefined`` on failure.
"""

from __future__ import annotations

import datetime
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest

REPO = Path(__file__).resolve().parents[1]
DASHBOARD_JS = REPO / "web" / "static" / "js" / "dashboard.js"
INDEX_HTML = REPO / "web" / "templates" / "index.html"


@pytest.fixture
def client():
    from web.app import app

    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def _fake_response(status_code: int, payload: dict | None = None, text: str = ""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload or {}
    resp.text = text
    return resp


# --- bug 1: /api/proxy/restart -------------------------------------------


def test_restart_reports_failure_when_no_proxy_is_reachable(client):
    """The old code returned "success" here.

    With no proxy endpoint reachable nothing was restarted, so the button
    silently did nothing. It now says so instead of starting a poll loop that
    can only time out.
    """
    with patch("web.app._find_proxy_url", return_value=None):
        res = client.post("/api/proxy/restart")
    assert res.status_code == 502
    data = res.get_json()
    assert data["status"] == "error"
    assert "no alpaca-proxy endpoint is reachable" in data["message"]
    # alpaca-proxy is not bounced: that would take the only route to the
    # model runtime offline while llama-server is still un-restarted.
    proxy_step = next(s for s in data["steps"] if s["step"] == "alpaca-proxy")
    assert proxy_step["ok"] is False
    assert "skipped" in proxy_step["detail"]


def test_restart_uses_the_proxy_admin_restart_endpoint(client):
    """It must call /admin/restart WITHOUT restart_proxy=true.

    The old call passed `?restart_proxy=true`, which the endpoint answers with
    "unsupported" because a process cannot restart its own container.
    """
    with (
        patch("web.app._find_proxy_url", return_value="http://proxy:11434"),
        patch("httpx.Client.post", return_value=_fake_response(200, {"message": "ok"})) as post,
        patch("web.app._restart_container_via_docker", return_value=(True, "")),
    ):
        res = client.post("/api/proxy/restart")
    assert res.status_code == 200
    url, kwargs = post.call_args.args[0], post.call_args.kwargs
    assert url == "http://proxy:11434/admin/restart"
    assert "restart_proxy" not in kwargs.get("params", {})


def test_restart_does_not_fall_through_to_success_on_a_non_200(client):
    """A 404/502 from the proxy used to fall through to the broken fallback."""
    with (
        patch("web.app._find_proxy_url", return_value="http://proxy:11434"),
        patch("httpx.Client.post", return_value=_fake_response(404, text="Not Found")),
    ):
        res = client.post("/api/proxy/restart")
    assert res.status_code == 502
    assert "Not Found" in res.get_json()["message"]


def test_restart_surfaces_a_transport_failure(client):
    with (
        patch("web.app._find_proxy_url", return_value="http://proxy:11434"),
        patch("httpx.Client.post", side_effect=OSError("connection refused")),
    ):
        res = client.post("/api/proxy/restart")
    assert res.status_code == 502
    assert "connection refused" in res.get_json()["message"]


class _CapturedThread:
    """Stands in for threading.Thread so the deferred work runs inline."""

    instances: ClassVar[list[_CapturedThread]] = []

    def __init__(self, target=None, daemon=None, **_kwargs):
        self.target = target
        self.daemon = daemon
        _CapturedThread.instances.append(self)

    def start(self):
        # time.sleep inside the target would stall the test, so the delay is
        # dropped here; the delay itself is asserted via call kwargs below.
        self.target()


@pytest.fixture
def captured_threads(monkeypatch):
    _CapturedThread.instances = []
    monkeypatch.setattr("web.app.threading.Thread", _CapturedThread)
    return _CapturedThread


def test_restart_schedules_the_proxy_restart_after_llama_succeeds(client, captured_threads):
    with (
        patch("web.app._find_proxy_url", return_value="http://proxy:11434"),
        patch("httpx.Client.post", return_value=_fake_response(200, {"message": "ok"})),
        patch("web.app._restart_container_via_docker", return_value=(True, "")) as restart,
    ):
        res = client.post("/api/proxy/restart")

    assert res.status_code == 200
    assert len(captured_threads.instances) == 1
    assert captured_threads.instances[0].daemon is True
    assert restart.call_args.args[0] == "alpaca-proxy"
    # Delayed so the HTTP response is flushed before the proxy serving it goes
    # away.
    assert restart.call_args.kwargs["delay_s"] == 1.0


def test_restart_does_not_schedule_a_proxy_restart_when_llama_fails(client, captured_threads):
    with (
        patch("web.app._find_proxy_url", return_value="http://proxy:11434"),
        patch("httpx.Client.post", return_value=_fake_response(500, text="boom")),
        patch("web.app._restart_container_via_docker", return_value=(True, "")) as restart,
    ):
        res = client.post("/api/proxy/restart")

    assert res.status_code == 502
    assert captured_threads.instances == []
    restart.assert_not_called()


def test_restart_container_helper_uses_the_docker_sdk_not_the_cli(monkeypatch):
    """Dockerfile.web has no `docker` binary, so a subprocess would always fail.

    The web container does have the socket mounted and the `docker` Python SDK
    installed, so the SDK is the only thing that can work here.
    """
    import web.app

    container = MagicMock()
    client_obj = MagicMock()
    client_obj.containers.get.return_value = container
    monkeypatch.setattr(web.app.docker, "from_env", lambda: client_obj)

    ok, detail = web.app._restart_container_via_docker("alpaca-proxy")
    assert ok, detail
    client_obj.containers.get.assert_called_once_with("alpaca-proxy")
    container.restart.assert_called_once_with(timeout=30)


def test_restart_container_helper_reports_failure(monkeypatch):
    import web.app

    def _boom():
        raise RuntimeError("no such container")

    monkeypatch.setattr(web.app.docker, "from_env", _boom)
    ok, detail = web.app._restart_container_via_docker("nope")
    assert not ok
    assert "no such container" in detail


def test_proxy_exposes_admin_restart_and_cannot_restart_itself():
    """The route that /api/proxy/restart calls has to actually exist."""
    import ast

    tree = ast.parse((REPO / "alpaca-proxy.py").read_text(encoding="utf-8"))
    posts: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call) or not isinstance(dec.func, ast.Attribute):
                continue
            if dec.func.attr != "post":
                continue
            path = ast.literal_eval(dec.args[0])
            posts[path] = node.name
    assert "/admin/restart" in posts, sorted(posts)[:5]
    assert posts["/admin/restart"] == "admin_restart"


def test_proxy_admin_restart_refuses_to_restart_itself():
    """`restart_proxy=true` is answered honestly, not silently ignored."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("proxy_admin_restart", REPO / "alpaca-proxy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)

    import asyncio

    result = asyncio.run(module.admin_restart(restart_proxy=True))
    assert result["status"] == "unsupported"
    assert result["llama_server"] == "skipped"
    assert "cannot restart its own container" in result["message"]


# --- bug 2: applyAnalysisRec scope ----------------------------------------


def test_apply_analysis_rec_is_reachable_from_the_rendered_markup():
    """The old button was an inline onclick; inline handlers cannot see closure
    variables, so every click raised `ReferenceError: applyAnalysisRec is not
    defined`. The markup now carries data attributes and a delegated listener.
    """
    source = DASHBOARD_JS.read_text(encoding="utf-8")
    assert "onclick=\"applyAnalysisRec" not in source
    assert "js-apply-analysis" in source
    assert "data-recs=" in source

    # And the function must be at module scope, i.e. after the
    # DOMContentLoaded closure closes.
    closure_end = source.index("\n});\n")
    fn = source.index("async function applyAnalysisRec(")
    assert fn > closure_end, "applyAnalysisRec is still inside the DOMContentLoaded closure"


def test_apply_analysis_rec_does_not_pass_json_through_an_html_attribute():
    """Recommendation values are model/section names from models.ini.

    Interpolating them into an attribute and re-parsing them inside an inline
    handler broke on any value containing a quote, an ampersand or a `<`.
    """
    source = DASHBOARD_JS.read_text(encoding="utf-8")
    button = source[source.index("js-apply-analysis") : source.index("js-apply-analysis") + 700]
    assert "&amp;" in button
    assert "&quot;" in button
    assert "&lt;" in button
    assert "JSON.parse(btn.dataset.recs" in source


@pytest.mark.needs_node
def test_apply_analysis_rec_delegated_handler_parses_the_payload():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for dashboard rendering tests")
    source = DASHBOARD_JS.read_text(encoding="utf-8")
    start = source.index("async function applyAnalysisRec(")
    end = source.index("// ═══════════════════════════ AUDIO STUDIO", start)
    slice_ = source[start:end]

    harness = """
const posted = [];
globalThis.fetch = async (url, opts) => { posted.push([url, JSON.parse(opts.body)]); return { json: async () => ({status:'success'}) }; };
globalThis.showToast = () => {};
"""
    result = subprocess.run(
        [node, "-e", harness + slice_ + "\napplyAnalysisRec('m', {'ctx-size': 8192}).then(() => console.log(JSON.stringify(posted)));"],
        capture_output=True,
        text=True,
        check=True,
    )
    (url, body), = json.loads(result.stdout)
    assert url == "/api/telemetry/recommendations/apply"
    assert body == {"model": "m", "recommendations": {"ctx-size": 8192}}


# --- bug 3: API Docs port labelling ---------------------------------------


def test_api_docs_never_files_a_dashboard_route_under_a_proxy_heading():
    html = INDEX_HTML.read_text(encoding="utf-8")

    # Find each group heading and the menu items that follow it, up to the next
    # heading, then assert every dashboard-only path sits under a 5000 heading.
    headings = [
        (m.start(), m.group(1))
        for m in re.finditer(r'text-transform: uppercase;[^>]*>([^<]+)</div>', html)
    ]
    assert headings
    items = [(m.start(), m.group(1)) for m in re.finditer(r'data-target="([^"]+)"', html)]

    def group_of(pos: int) -> str:
        return next(title for start, title in reversed(headings) if start < pos)

    dashboard_only = {"doc-api-requests", "doc-api-requests-clear"}
    for pos, target in items:
        if target not in dashboard_only:
            continue
        group = group_of(pos)
        assert "5000" in group, f"{target} is filed under {group!r}, so its curl 404s against the proxy"


def test_api_docs_requests_panes_name_the_proxy_equivalent():
    html = INDEX_HTML.read_text(encoding="utf-8")
    pane = html[html.index('id="doc-api-requests"') : html.index('id="doc-api-requests-clear"')]
    assert "Dashboard only (port 5000)" in pane
    assert "/admin/requests" in pane
    assert "localhost:5000/api/requests" in pane

    pane = html[html.index('id="doc-api-requests-clear"') : html.index('id="doc-admin-system"')]
    assert "Dashboard only (port 5000)" in pane
    assert "/admin/requests/clear" in pane


# --- bug 4: /api/analyze/all envelope -------------------------------------


ANALYZE_KEYS = {"strategy", "models_analyzed", "models_skipped", "results"}


def _iso(epoch: float) -> str:
    return datetime.datetime.fromtimestamp(epoch, datetime.UTC).isoformat()


def test_analyze_all_error_path_keeps_the_success_envelope(client, tmp_path, monkeypatch):
    monkeypatch.setenv("TELEMETRY_DIR", str(tmp_path / "absent"))
    res = client.get("/api/analyze/all")
    assert res.status_code == 404
    data = res.get_json()
    # The key used to be "models" here and "results" on success.
    assert set(data) >= ANALYZE_KEYS
    assert data["results"] == []
    assert "error" in data


def test_analyze_all_success_path_uses_the_same_keys(client, tmp_path, monkeypatch):
    # The analyzer age-filters to the last hour (load_telemetry's
    # max_age_seconds=3600), so a fixture with stale stamps reads as
    # insufficient_data and the model lands in models_skipped.
    now = time.time()
    telemetry = tmp_path / "telemetry"
    telemetry.mkdir()
    (telemetry / "m1.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "timestamp": _iso(now - 60 + i),
                    "epoch_time": now - 60 + i,
                    "model_alias": "m1",
                    "system": {"ram_total_gb": 32, "ram_used_gb": 16, "ram_used_pct": 50, "cpu_util_pct": 10},
                    "gpus": [{"vram_total_mb": 8192, "vram_used_mb": 4000, "vram_free_mb": 4192, "vram_used_pct": 49}],
                    "llama_server": {
                        "model_path": "/models/m1.gguf",
                        "n_ctx": 8192,
                        "n_gpu_layers": 99,
                        "flash_attn": True,
                        "slots": {"total": 1, "active": 0, "tokens_cached": 100, "kv_cache_used_pct": 1.2},
                    },
                }
            )
            for i in range(6)
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TELEMETRY_DIR", str(telemetry))
    res = client.get("/api/analyze/all?strategy=safe")
    assert res.status_code == 200
    data = res.get_json()
    assert set(data) >= ANALYZE_KEYS
    assert data["models_analyzed"] == 1
    assert data["results"][0]["model_alias"] == "m1"
    # The success payload must not have grown a "models" alias.
    assert "models" not in data


def test_analyze_all_skips_non_model_telemetry_files(client, tmp_path, monkeypatch):
    telemetry = tmp_path / "telemetry"
    telemetry.mkdir()
    for alias in ("none", "system_idle", "unknown_model"):
        (telemetry / f"{alias}.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("TELEMETRY_DIR", str(telemetry))

    res = client.get("/api/analyze/all")
    assert res.status_code == 200
    data = res.get_json()
    assert data["results"] == []
    assert set(data["models_skipped"]) == {"none", "system_idle", "unknown_model"}
