"""Coverage for the /api/sandbox/* web routes.

Six of these routes had no direct test: /api/sandbox/serve, /stop_serve,
/ui/status, /ui/exec, /ui/restart and /ui/screenshot. They are thin
adapters over ``sandbox_exec``, and the adapters are exactly where a
mis-wired argument or a missing validation shows up: the launcher's
terminal is a live shell, the screenshot is the only evidence a UI
actually rendered, and ``serve`` is what turns a benchmark artifact into
a viewable page.

The same functions are covered at the ``sandbox_exec`` level in
test_sandbox_and_security.py; what is pinned here is the *web* contract:
required fields, error envelopes, and which sandbox_exec argument each
route passes through.
"""

from unittest.mock import patch

import pytest

from web.app import app

CID = "alpaca-serve-deadbeef"


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# --------------------------------------------------------------------------
# /api/sandbox/serve
# --------------------------------------------------------------------------


def test_serve_requires_code(client):
    assert client.post("/api/sandbox/serve", json={}).status_code == 400
    assert client.post("/api/sandbox/serve", json={"code": ""}).status_code == 400
    assert client.post("/api/sandbox/serve", json={"code": None}).status_code == 400


def test_serve_bodyless_post_is_415_not_500(client):
    """The route reads get_json() outside the try, so no body is a 415.

    Pinning it because a 500 here would mean the guard was moved inside
    the try block, which would turn a client mistake into a server error.
    """
    assert client.post("/api/sandbox/serve").status_code == 415


def test_serve_defaults_lang_to_html_and_forwards_both(client):
    with patch("web.app.serve_app") as serve:
        serve.return_value = {"container_id": CID, "host_port": "49152"}
        res = client.post("/api/sandbox/serve", json={"code": "<html></html>"})
    assert res.status_code == 200
    assert serve.call_args.args[0] == "<html></html>"
    assert serve.call_args.args[1] == "html"
    assert res.get_json()["host_port"] == "49152"


def test_serve_passes_explicit_lang_through(client):
    with patch("web.app.serve_app") as serve:
        serve.return_value = {"container_id": CID}
        client.post("/api/sandbox/serve", json={"code": "print(1)", "lang": "python"})
    assert serve.call_args.args[1] == "python"


def test_serve_surfaces_sandbox_error_as_500(client):
    with patch("web.app.serve_app") as serve:
        serve.return_value = {"error": "unsupported language for serving: ruby"}
        res = client.post("/api/sandbox/serve", json={"code": "x", "lang": "ruby"})
    assert res.status_code == 500
    assert res.get_json()["error"] == "unsupported language for serving: ruby"


def test_serve_rejects_get(client):
    assert client.get("/api/sandbox/serve").status_code == 405


# --------------------------------------------------------------------------
# /api/sandbox/serve_ui  (timeout clamping is the interesting part)
# --------------------------------------------------------------------------


def test_serve_ui_requires_code(client):
    assert client.post("/api/sandbox/serve_ui", json={}).status_code == 400


def test_serve_ui_defaults_to_python_named_alpaca_ui(client):
    with patch("web.app.serve_ui") as serve:
        serve.return_value = {"container_id": CID, "host_port": "49153"}
        client.post("/api/sandbox/serve_ui", json={"code": "import pygame"})
    assert serve.call_args.args[1] == "python"
    assert serve.call_args.kwargs["name"] == "alpaca-ui"
    assert serve.call_args.kwargs["exclusive"] is False
    assert serve.call_args.kwargs["timeout"] == 600


@pytest.mark.parametrize(
    "sent,expected",
    [
        (0, 60),  # below the floor: the app needs time to draw one frame
        (-5, 60),
        (1, 60),
        (60, 60),
        (600, 600),
        (28800, 28800),
        (999999, 28800),  # above the ceiling
    ],
)
def test_serve_ui_clamps_timeout_to_60s_8h(client, sent, expected):
    """An arcade session outliving the default must not die mid-game.

    The container's PID1 sleeps ``timeout + 60`` and reaps itself, so an
    unclamped value is what silently kills a two-hour play session.
    """
    with patch("web.app.serve_ui") as serve:
        serve.return_value = {"container_id": CID}
        client.post("/api/sandbox/serve_ui", json={"code": "x", "timeout": sent})
    assert serve.call_args.kwargs["timeout"] == expected


@pytest.mark.parametrize("sent", ["abc", None, [], {}])
def test_serve_ui_non_numeric_timeout_falls_back_to_600(client, sent):
    with patch("web.app.serve_ui") as serve:
        serve.return_value = {"container_id": CID}
        client.post("/api/sandbox/serve_ui", json={"code": "x", "timeout": sent})
    assert serve.call_args.kwargs["timeout"] == 600


def test_serve_ui_forwards_name_and_exclusive(client):
    with patch("web.app.serve_ui") as serve:
        serve.return_value = {"container_id": CID}
        client.post(
            "/api/sandbox/serve_ui",
            json={"code": "x", "name": "alpaca-arcade-abc123", "exclusive": 1},
        )
    # exclusive is coerced with bool() so a truthy int from the arcade
    # (which sends 1) is not treated as a missing field.
    assert serve.call_args.kwargs["name"] == "alpaca-arcade-abc123"
    assert serve.call_args.kwargs["exclusive"] is True


def test_serve_ui_surfaces_sandbox_error(client):
    with patch("web.app.serve_ui") as serve:
        serve.return_value = {"error": "unsupported language for UI serving: html"}
        res = client.post("/api/sandbox/serve_ui", json={"code": "x", "lang": "html"})
    assert res.status_code == 500


# --------------------------------------------------------------------------
# /api/sandbox/stop_serve
# --------------------------------------------------------------------------


def test_stop_serve_requires_container_id(client):
    assert client.post("/api/sandbox/stop_serve", json={}).status_code == 400


def test_stop_serve_passes_cid_through_and_relays_result(client):
    with patch("web.app.stop_serve") as stop:
        stop.return_value = {"stopped": True, "container_id": CID}
        res = client.post("/api/sandbox/stop_serve", json={"container_id": CID})
    assert res.get_json() == {"stopped": True, "container_id": CID}
    assert stop.call_args.args[0] == CID


def test_stop_serve_relays_a_failure_without_raising(client):
    """A container that is already gone is a normal outcome, not a 500.

    The arcade calls stop on every Play even when the user closed the
    tab, so this has to be a 200 the caller can ignore.
    """
    with patch("web.app.stop_serve") as stop:
        stop.return_value = {"error": "no such container", "stopped": False}
        res = client.post("/api/sandbox/stop_serve", json={"container_id": CID})
    assert res.status_code == 200
    assert res.get_json()["stopped"] is False


# --------------------------------------------------------------------------
# /api/sandbox/ui/exec  (arbitrary shell inside the container)
# --------------------------------------------------------------------------


def test_ui_exec_requires_container_id(client):
    assert client.post("/api/sandbox/ui/exec", json={"command": "ls"}).status_code == 400


@pytest.mark.parametrize("command", ["", "   ", None])
def test_ui_exec_requires_a_command(client, command):
    assert client.post("/api/sandbox/ui/exec", json={"container_id": CID, "command": command}).status_code == 400


def test_ui_exec_strips_the_command_before_running_it(client):
    """Whitespace is stripped so a blank-ish terminal line is a 400.

    Otherwise "   " reaches the shell as a successful no-op and the
    launcher shows empty output with no explanation.
    """
    with patch("web.app.ui_exec") as run:
        run.return_value = {"output": "file.py\n"}
        res = client.post("/api/sandbox/ui/exec", json={"container_id": CID, "command": "  ls -la  "})
    assert res.status_code == 200
    assert run.call_args.args[0] == CID
    assert run.call_args.args[1] == "ls -la"


def test_ui_exec_validates_before_it_looks_at_the_command(client):
    """A missing container_id is rejected even when a command is present."""
    with patch("web.app.ui_exec") as run:
        res = client.post("/api/sandbox/ui/exec", json={"command": "ls"})
    assert res.status_code == 400
    assert run.called is False


def test_ui_exec_passes_containers_id_positionally(client):
    with patch("web.app.ui_exec") as run:
        run.return_value = {"output": ""}
        client.post("/api/sandbox/ui/exec", json={"container_id": "  spaced  ", "command": "pwd"})
    # No strip on the id: it is a Docker identifier, not free text.
    assert run.call_args.args[0] == "  spaced  "


def test_ui_exec_relays_output(client):
    with patch("web.app.ui_exec") as run:
        run.return_value = {"output": "total 4\n", "exit_code": 0}
        res = client.post("/api/sandbox/ui/exec", json={"container_id": CID, "command": "ls -la"})
    assert res.get_json()["exit_code"] == 0


def test_ui_exec_relays_an_error_payload_as_200(client):
    with patch("web.app.ui_exec") as run:
        run.return_value = {"error": "container not found", "output": ""}
        res = client.post("/api/sandbox/ui/exec", json={"container_id": CID, "command": "ls"})
    assert res.status_code == 200
    assert res.get_json()["error"] == "container not found"


# --------------------------------------------------------------------------
# /api/sandbox/ui/status, /ui/screenshot, /ui/restart
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "route",
    [
        "/api/sandbox/ui/status",
        "/api/sandbox/ui/screenshot",
        "/api/sandbox/ui/restart",
        "/api/sandbox/stop_serve",
    ],
)
def test_container_routes_require_container_id(client, route):
    assert client.post(route, json={}).status_code == 400
    assert client.post(route, json={"container_id": ""}).status_code == 400
    assert client.post(route, json={"container_id": None}).status_code == 400


def test_ui_status_forwards_and_relays(client):
    with patch("web.app.ui_status") as status:
        status.return_value = {
            "running": True,
            "app_pid": 42,
            "app_exitcode": None,
            "stdout_tail": "frame 1\n",
        }
        res = client.post("/api/sandbox/ui/status", json={"container_id": CID})
    assert status.call_args.args[0] == CID
    body = res.get_json()
    assert body["running"] is True
    assert body["app_pid"] == 42


def test_ui_status_reports_a_dead_app(client):
    with patch("web.app.ui_status") as status:
        status.return_value = {"running": False, "app_exitcode": 1, "stdout_tail": "Traceback..."}
        res = client.post("/api/sandbox/ui/status", json={"container_id": CID})
    assert res.get_json()["app_exitcode"] == 1


def test_ui_screenshot_returns_base64_png(client):
    with patch("web.app.ui_screenshot") as shot:
        shot.return_value = {"image": "iVBORw0KGgo="}
        res = client.post("/api/sandbox/ui/screenshot", json={"container_id": CID})
    assert shot.call_args.args[0] == CID
    assert res.get_json()["image"] == "iVBORw0KGgo="


def test_ui_screenshot_relays_failure(client):
    with patch("web.app.ui_screenshot") as shot:
        shot.return_value = {"error": "screenshot failed", "image": ""}
        res = client.post("/api/sandbox/ui/screenshot", json={"container_id": CID})
    assert res.status_code == 200
    assert res.get_json()["error"] == "screenshot failed"


def test_ui_restart_forwards_and_relays(client):
    with patch("web.app.ui_restart") as restart:
        restart.return_value = {"restarted": True, "app_pid": 43}
        res = client.post("/api/sandbox/ui/restart", json={"container_id": CID})
    assert restart.call_args.args[0] == CID
    assert res.get_json()["app_pid"] == 43


def test_ui_restart_relays_failure(client):
    with patch("web.app.ui_restart") as restart:
        restart.return_value = {"error": "no app to restart"}
        res = client.post("/api/sandbox/ui/restart", json={"container_id": CID})
    assert res.status_code == 200
    assert "error" in res.get_json()


@pytest.mark.parametrize("route", ["/api/sandbox/ui/status", "/api/sandbox/ui/screenshot", "/api/sandbox/ui/restart"])
def test_container_routes_reject_get(client, route):
    """These are POST-only: a GET would read as a legitimate probe."""
    assert client.get(route).status_code == 405


# --------------------------------------------------------------------------
# /api/sandbox/ui_inner_status  (the launcher's noVNC beacon)
# --------------------------------------------------------------------------


def test_ui_inner_status_accepts_a_bodyless_post(client):
    """The beacon can fire before the page has anything to report."""
    assert client.post("/api/sandbox/ui_inner_status").status_code == 200


def test_ui_inner_status_always_acknowledges(client):
    res = client.post("/api/sandbox/ui_inner_status", json={"container_id": CID, "state": "connected"})
    # The route logs and acknowledges; it never surfaces the launcher's
    # own state as an error, because a beacon is not a request.
    assert res.status_code == 200
    assert res.get_json() == {"ok": True}


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"state": "disconnected", "detail": "x" * 5000},
        {"container_id": None, "state": None, "detail": None},
    ],
)
def test_ui_inner_status_tolerates_missing_fields(client, body):
    res = client.post("/api/sandbox/ui_inner_status", json=body)
    assert res.status_code == 200
    assert res.get_json()["ok"] is True


# --------------------------------------------------------------------------
# Cross-cutting: the adapters must not talk to anything but sandbox_exec
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "route,function",
    [
        ("/api/sandbox/serve", "serve_app"),
        ("/api/sandbox/serve_ui", "serve_ui"),
        ("/api/sandbox/stop_serve", "stop_serve"),
        ("/api/sandbox/ui/exec", "ui_exec"),
        ("/api/sandbox/ui/status", "ui_status"),
        ("/api/sandbox/ui/screenshot", "ui_screenshot"),
        ("/api/sandbox/ui/restart", "ui_restart"),
    ],
)
def test_sandbox_route_delegates_to_its_sandbox_exec_function(client, route, function):
    """Each route must reach its own function, and only via that one.

    A copy-paste that sends /ui/restart to ui_status would otherwise be
    invisible: both return a 200 JSON body.
    """
    with patch(f"web.app.{function}") as target, patch("web.app.httpx.Client") as http:
        target.return_value = {"ok": True}
        client.post(route, json={"container_id": CID, "code": "x", "command": "ls"})
    assert target.called is True
    # The sandbox runs out-of-process; no route may reach for a socket.
    assert http.called is False


def test_sandbox_routes_are_all_post_only(client):
    for route in (
        "/api/sandbox/serve",
        "/api/sandbox/serve_ui",
        "/api/sandbox/stop_serve",
        "/api/sandbox/ui/exec",
        "/api/sandbox/ui/status",
        "/api/sandbox/ui/screenshot",
        "/api/sandbox/ui/restart",
        "/api/sandbox/ui_inner_status",
    ):
        assert client.get(route).status_code == 405, route


def test_sandbox_routes_reject_get_with_allow_header_naming_post(client):
    res = client.get("/api/sandbox/ui/exec")
    assert res.status_code == 405
    assert "POST" in res.headers.get("Allow", "")
