import io
import json
import os
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image, ImageDraw

from online_providers import OnlineModelProvider
from sandbox_exec import (
    _find_web_js_error,
    _lint_code,
    _lint_html_js,
    _run_ui,
    _run_web_ui,
    _screenshot_has_content,
    extract_clean_code,
    grade_code,
    run_code_once,
    serve_app,
    serve_ui,
    ui_exec,
    ui_restart,
    ui_screenshot,
    ui_status,
)
from web.model_tracker import ModelTracker


def test_extract_clean_code_with_think_tags():
    raw = (
        "<think>\n"
        "Let's think about this step by step.\n"
        "We need a function that adds two numbers.\n"
        "</think>\n"
        "```python\n"
        "def add(a: int, b: int) -> int:\n"
        "    # Return sum\n"
        "    return a + b\n"
        "```"
    )
    cleaned = extract_clean_code(raw, "python")
    assert "def add(a: int, b: int) -> int:" in cleaned
    assert "# Return sum" in cleaned
    assert "<think>" not in cleaned
    assert "Let's think" not in cleaned
    assert "```" not in cleaned


def test_extract_clean_code_with_conversational_prose():
    raw = (
        "Here is the complete Python solution for your problem:\n\n"
        "def multiply(x, y):\n"
        "    # Legitimate comment\n"
        "    return x * y\n\n"
        "Hope this helps! Let me know if you need any adjustments."
    )
    cleaned = extract_clean_code(raw, "python")
    assert cleaned.startswith("def multiply(x, y):")
    assert "# Legitimate comment" in cleaned
    assert "Here is the complete" not in cleaned
    assert "Hope this helps" not in cleaned


def test_extract_clean_code_javascript_fences():
    raw = (
        "Sure, here is the vanilla JS script:\n"
        "```javascript\n"
        "const canvas = document.getElementById('game');\n"
        "const ctx = canvas.getContext('2d');\n"
        "// Render loop\n"
        "function loop() {\n"
        "    requestAnimationFrame(loop);\n"
        "}\n"
        "```\n"
        "This runs at 60fps."
    )
    cleaned = extract_clean_code(raw, "javascript")
    assert "const canvas = document.getElementById('game');" in cleaned
    assert "// Render loop" in cleaned
    assert "Sure, here is" not in cleaned
    assert "This runs at 60fps" not in cleaned


def test_extract_clean_code_truncated_fence():
    # Generation hit the token cap before the closing ``` arrived.
    raw = (
        "<think>plan</think>\n"
        "The user wants a game. Let me plan it out.\n"
        "```python\n"
        "import pygame\n"
        "pygame.init()\n"
        "screen = pygame.display.set_mode((800, 600))\n"
        "while True:\n"
        "    for event in pygame.event.get():"
    )
    cleaned = extract_clean_code(raw, "python")
    assert "import pygame" in cleaned
    assert "The user wants a game" not in cleaned
    assert "```" not in cleaned


def test_extract_clean_code_prefers_code_over_plan_fence():
    # Some models fence their planning notes before the real implementation.
    raw = (
        "```python\n"
        "# Plan:\n"
        "# 1. Build the game loop\n"
        "# 2. Add scoring\n"
        "```\n"
        "```python\n"
        "import json\n"
        "SCORE_FILE = 'scores.json'\n"
        "def load():\n"
        "    return []\n"
        "```"
    )
    cleaned = extract_clean_code(raw, "python")
    assert "import json" in cleaned
    assert "# Plan:" not in cleaned


def test_extract_clean_code_markdown_prose_not_code():
    # Reasoning prose with markdown headings/bullets must not be mistaken for code.
    raw = (
        "Here's a thinking process:\n"
        "1. **Analyze User Requirements:**\n"
        "   - Libraries: pygame, PyOpenGL\n"
        "## Requirements\n"
        "- for each food eaten grow\n"
        "- while playing keep score\n"
        "import sys\n"
        "def main():\n"
        "    pass\n"
    )
    cleaned = extract_clean_code(raw, "python")
    assert cleaned.startswith("import sys")
    assert "thinking process" not in cleaned
    assert "## Requirements" not in cleaned


def test_sandbox_exec_timeout_handling():
    with patch("docker.DockerClient") as mock_docker:
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_docker.return_value = mock_client
        mock_client.containers.run.return_value = mock_container

        # Simulate exec_run hanging
        def _hang(*args, **kwargs):
            import time

            time.sleep(2)
            return (0, b"")

        mock_container.exec_run.side_effect = _hang

        res = run_code_once("while True: pass", lang="python", timeout=0.01)
        assert res["ran"] is False
        assert res["exit_code"] == 124
        assert "timed out" in res["error"].lower()

        # Check grading result on timeout
        graded = grade_code("while True: pass", lang="python", timeout=0.01)
        assert graded["ran"] is False
        assert graded["score"] == 0
        assert graded["exit_code"] == 124


def _png_bytes(size=(128, 96), color=(0, 0, 0), shapes=None):
    img = Image.new("RGB", size, color)
    if shapes:
        d = ImageDraw.Draw(img)
        for shape in shapes:
            d.rectangle(shape, fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_screenshot_has_content_rejects_blank_frames():
    assert _screenshot_has_content(b"") is False
    assert _screenshot_has_content(_png_bytes(color=(0, 0, 0))) is False
    assert _screenshot_has_content(_png_bytes(color=(255, 255, 255))) is False
    assert _screenshot_has_content(b"not a png") is False


def test_screenshot_has_content_detects_rendered_ui():
    content = _png_bytes(color=(0, 0, 0), shapes=[(10, 10, 60, 50), (20, 20, 40, 30)])
    assert _screenshot_has_content(content) is True


def test_run_ui_blank_screenshot_is_not_a_passing_ui():
    with patch("sandbox_exec._put_file"), patch("sandbox_exec._read_file") as mock_read:
        blank = _png_bytes()
        mock_read.side_effect = lambda c, p: blank if p.endswith("out.png") else b""
        mock_container = MagicMock()
        mock_container.exec_run.return_value = (0, b"")

        result = _run_ui(mock_container, "import pygame", "py", "python3", 30, {})
        assert result["screenshot"] is not None
        assert result["ui_rendered"] is False
        assert result["ran"] is False


def test_run_ui_rendered_screenshot_is_a_passing_ui():
    with patch("sandbox_exec._put_file"), patch("sandbox_exec._read_file") as mock_read:
        content = _png_bytes(shapes=[(5, 5, 60, 60)])
        mock_read.side_effect = lambda c, p: content if p.endswith("out.png") else b""
        mock_container = MagicMock()
        mock_container.exec_run.return_value = (0, b"")

        result = _run_ui(mock_container, "import pygame", "py", "python3", 30, {})
        assert result["screenshot"] is not None
        assert result["ui_rendered"] is True
        assert result["ran"] is True


def test_run_web_ui_rendered_screenshot_is_a_passing_ui():
    with patch("sandbox_exec._put_file") as mock_put, patch("sandbox_exec._read_file") as mock_read:
        content = _png_bytes(shapes=[(5, 5, 60, 60)])
        mock_read.side_effect = lambda c, p: content if p.endswith("out.png") else b""
        mock_container = MagicMock()
        mock_container.exec_run.return_value = (0, b"")

        result = _run_web_ui(mock_container, "<canvas></canvas>", 30, {})
        assert result["screenshot"] is not None
        assert result["ui_rendered"] is True
        assert result["ran"] is True
        # The chromium render script must be written into the container.
        put_paths = [str(c.args[1]) for c in mock_put.call_args_list]
        assert any(p.endswith("render_web.sh") for p in put_paths)


def test_run_web_ui_blank_screenshot_is_not_a_passing_ui():
    with patch("sandbox_exec._put_file"), patch("sandbox_exec._read_file") as mock_read:
        blank = _png_bytes()
        mock_read.side_effect = lambda c, p: blank if p.endswith("out.png") else b""
        mock_container = MagicMock()
        mock_container.exec_run.return_value = (0, b"")

        result = _run_web_ui(mock_container, "<canvas></canvas>", 30, {})
        assert result["screenshot"] is not None
        assert result["ui_rendered"] is False
        assert result["ran"] is False


def test_run_code_once_web_lang_routes_to_chromium_renderer():
    with (
        patch("sandbox_exec._run_web_ui") as mock_web,
        patch("docker.DockerClient") as mock_docker,
    ):
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_docker.return_value = mock_client
        mock_client.containers.run.return_value = mock_container
        mock_web.return_value = {
            "ran": True,
            "exit_code": 0,
            "output": "",
            "error": "",
            "lang": "web",
            "screenshot": "base64",
            "ui_rendered": True,
        }
        result = run_code_once("<canvas></canvas>", lang="web", ui=True)
        mock_web.assert_called_once()
        assert result["ran"] is True


def test_serve_ui_launches_novnc_container():
    with patch("docker.DockerClient") as mock_docker:
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_docker.return_value = mock_client
        mock_client.containers.run.return_value = mock_container

        # Published websockify port (6080 -> host 39781)
        mock_container.ports = {"6080/tcp": [{"HostPort": "39781"}]}

        res = serve_ui("print('hello ui')", lang="python", timeout=5)

        assert res["error"] == ""
        assert res["container_id"] == mock_container.id
        assert res["host_port"] == "39781"
        assert "url" not in res

        # Sandbox container must be created on the bridge network with the
        # websockify port published to all interfaces (browser may be remote).
        kwargs = mock_client.containers.run.call_args.kwargs
        assert kwargs["network_mode"] == "bridge"
        assert kwargs["ports"] == {"6080/tcp": None}
        assert kwargs["user"] == "sandbox"
        assert kwargs["mem_limit"] == "256m"
        assert kwargs["pids_limit"] == 128

        # Wrapper script must chain Xvfb -> x11vnc -> websockify -> app.
        run_ui_tar = None
        for call in mock_container.put_archive.call_args_list:
            tar_data = call.args[1]
            with tarfile.open(fileobj=io.BytesIO(tar_data)) as tf:
                if any(m.name.endswith("run_ui.sh") for m in tf.getmembers()):
                    run_ui_tar = tar_data
                    break
        assert run_ui_tar is not None
        with tarfile.open(fileobj=io.BytesIO(run_ui_tar)) as tf:
            member = next(m for m in tf.getmembers() if m.name.endswith("run_ui.sh"))
            script = tf.extractfile(member).read().decode()
        assert "Xvfb :99" in script
        assert "x11vnc -display :99" in script
        assert "websockify --web /usr/share/novnc 6080 127.0.0.1:5900" in script
        assert "python3 /tmp/code.py" in script

        # App pipeline launched detached.
        exec_calls = [c.args[0] for c in mock_container.exec_run.call_args_list]
        assert any(isinstance(a, list) and any("run_ui.sh" in x for x in a) for a in exec_calls)


def test_serve_ui_rejects_unsupported_language():
    with patch("docker.DockerClient") as mock_docker:
        mock_client = MagicMock()
        mock_docker.return_value = mock_client
        res = serve_ui("print('x')", lang="ruby", timeout=5)
        assert res["error"]
        assert res["host_port"] is None
        mock_client.containers.run.assert_not_called()


def test_serve_app_returns_host_port_no_localhost_url():
    with patch("docker.DockerClient") as mock_docker:
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_docker.return_value = mock_client
        mock_client.containers.run.return_value = mock_container

        # Published app port (8080 -> host 41234)
        mock_container.ports = {"8080/tcp": [{"HostPort": "41234"}]}

        res = serve_app("<h1>hi</h1>", lang="html", timeout=5)

        assert res["error"] == ""
        assert res["container_id"] == mock_container.id
        assert res["host_port"] == "41234"
        # Must NOT hardcode a localhost URL — the browser may be remote.
        assert "url" not in res

        # Port must be published to all interfaces so a remote browser can reach it.
        kwargs = mock_client.containers.run.call_args.kwargs
        assert kwargs["network_mode"] == "bridge"
        assert kwargs["ports"] == {"8080/tcp": None}
        assert kwargs["user"] == "sandbox"

        # Server launched detached.
        exec_calls = [c.args[0] for c in mock_container.exec_run.call_args_list]
        assert any(isinstance(a, list) and "http.server" in a for a in exec_calls)


def test_serve_app_rejects_unsupported_language():
    with patch("docker.DockerClient") as mock_docker:
        mock_client = MagicMock()
        mock_docker.return_value = mock_client
        res = serve_app("code", lang="ruby", timeout=5)
        assert res["error"]
        assert res["host_port"] is None
        mock_client.containers.run.assert_not_called()


def test_ui_exec_runs_command_in_container():
    with patch("sandbox_exec._ui_container") as mock_get:
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_get.return_value = (mock_client, mock_container)
        mock_container.exec_run.return_value = (0, (b"hello world\n", b""))
        res = ui_exec("cid123", "echo hello world", timeout=5)
        assert res["exit_code"] == 0
        assert "hello world" in res["output"]
        kwargs = mock_container.exec_run.call_args.kwargs
        assert kwargs["user"] == "sandbox"
        assert kwargs["workdir"] == "/tmp"
        assert kwargs["environment"] == {"DISPLAY": ":99"}


def test_ui_exec_missing_container():
    with patch("sandbox_exec._ui_container") as mock_get:
        mock_get.return_value = (None, None)
        res = ui_exec("nope", "echo hi")
        assert res["error"]
        assert "not found" in res["error"]


def test_ui_status_reports_app_state():
    with patch("sandbox_exec._ui_container") as mock_get:
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_get.return_value = (mock_client, mock_container)
        mock_container.status = "running"
        mock_container.reload.side_effect = lambda: setattr(mock_container, "status", "running")
        mock_container.exec_run.side_effect = [
            MagicMock(output=b"42\n"),
            MagicMock(output=b"3\n"),
            MagicMock(output=b"Traceback\npygame error\n"),
        ]
        res = ui_status("cid123")
        assert res["running"] is True
        assert res["app_pid"] == "42"
        assert res["app_exitcode"] == 3
        assert "Traceback" in res["stdout_tail"]


def test_ui_status_missing_container():
    with patch("sandbox_exec._ui_container") as mock_get:
        mock_get.return_value = (None, None)
        res = ui_status("nope")
        assert res["running"] is False
        assert res["error"]


def test_ui_screenshot_returns_base64():
    with patch("sandbox_exec._ui_container") as mock_get:
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_get.return_value = (mock_client, mock_container)
        mock_container.exec_run.return_value = (0, b"aGVsbG8=")
        res = ui_screenshot("cid123")
        assert res["image"] == "aGVsbG8="
        assert not res["error"]


def test_ui_screenshot_failure():
    with patch("sandbox_exec._ui_container") as mock_get:
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_get.return_value = (mock_client, mock_container)
        mock_container.exec_run.return_value = (1, b"SCROT_FAIL\n")
        res = ui_screenshot("cid123")
        assert res["image"] is None
        assert res["error"]


def test_ui_restart_relaunches_python_app():
    with patch("sandbox_exec._ui_container") as mock_get:
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_get.return_value = (mock_client, mock_container)
        ls_result = MagicMock()
        ls_result.exit_code = 1  # /tmp/app.js does NOT exist
        run_result = (0, b"STARTED\n")
        mock_container.exec_run.side_effect = [ls_result, run_result]
        res = ui_restart("cid123")
        assert res["restarted"] is True
        cmd = mock_container.exec_run.call_args.args[0]
        assert "/tmp/code.py" in cmd[2]
        assert "kill -9" in cmd[2]


def test_ui_restart_node_app():
    with patch("sandbox_exec._ui_container") as mock_get:
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_get.return_value = (mock_client, mock_container)
        ls_result = MagicMock()
        ls_result.exit_code = 0  # /tmp/app.js found
        run_result = (0, b"STARTED\n")
        mock_container.exec_run.side_effect = [ls_result, run_result]
        res = ui_restart("cid123")
        assert res["restarted"] is True
        cmd = mock_container.exec_run.call_args.args[0]
        assert "/tmp/app.js" in cmd[2]
        assert "node" in cmd[2]


def test_ui_restart_failure():
    with patch("sandbox_exec._ui_container") as mock_get:
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_get.return_value = (mock_client, mock_container)
        ls_result = MagicMock()
        ls_result.exit_code = 0  # app.js exists
        run_result = (1, b"boom\n")
        mock_container.exec_run.side_effect = [ls_result, run_result]
        res = ui_restart("cid123")
        assert res["restarted"] is False
        assert res["error"]


def test_masked_credentials_no_leak():
    provider = OnlineModelProvider()
    provider.alpaca_api_key = "sk-alpaca-secret-key-123456789"
    provider.openrouter_api_key = "sk-or-secret-token-987654321"

    masked = provider.get_masked_credentials()
    assert masked["alpaca"]["configured"] is True
    assert masked["alpaca"]["has_key"] is True
    assert masked["alpaca"]["auth_required"] is True
    assert "••••" in masked["alpaca"]["masked_key"]
    # Ensure plaintext key is NOT present
    assert "key" not in masked["alpaca"]
    assert masked["alpaca"]["masked_key"] != "sk-alpaca-secret-key-123456789"


def test_scan_historical_benchmarks_per_model_and_categories():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        shared_dir = tmp_path / "shared"
        gen_dir = tmp_path / "general"
        shared_models = shared_dir / "models"
        gen_models = gen_dir / "models"

        shared_models.mkdir(parents=True)
        gen_models.mkdir(parents=True)

        # Write per-model general benchmark file with category_* format
        gen_data = {
            "benchmark_version": "3.0.0",
            "generated_at": "2026-08-16T12:00:00",
            "results": [
                {
                    "model": "qwen3.6-35b-test",
                    "category_coding": {
                        "tests": [
                            {"test_id": "c1", "success": True},
                            {"test_id": "c2", "success": True},
                        ]
                    },
                    "category_reasoning": {
                        "tests": [
                            {"test_id": "r1", "success": False},
                            {"test_id": "r2", "success": True},
                        ]
                    },
                }
            ],
        }
        with open(gen_models / "general_qwen3.6-35b-test.json", "w", encoding="utf-8") as f:
            json.dump(gen_data, f)

        tracker = ModelTracker()
        tracker.shared_benchmarks_dir = shared_dir
        tracker.general_benchmarks_dir = gen_dir

        history = tracker.scan_historical_benchmarks()
        assert "qwen3.6-35b-test" in history
        entry = history["qwen3.6-35b-test"]
        assert entry["benchmark_count"] == 1
        assert entry["latest_score"] == 75.0  # 3 passed out of 4 tests = 75.0%
        assert entry["latest_run_type"] == "general"


def test_benchmark_tests_json_all_games_are_ui():
    tests_file = Path("benchmark_tests.json")
    assert tests_file.exists()
    with open(tests_file, encoding="utf-8") as f:
        data = json.load(f)

    for test in data.get("gamedev", []):
        assert test.get("type") == "ui", f"gamedev test {test['id']} must have type: 'ui'"

    for test in data.get("retrogames", []):
        assert test.get("type") == "ui", f"retrogames test {test['id']} must have type: 'ui'"


def test_local_network_auth_exemption():
    import importlib

    alpaca_proxy = importlib.import_module("alpaca-proxy")

    with patch.dict(os.environ, {"ALPACA_API_KEY": "sk-secret-local-test"}):
        from starlette.requests import Request

        def _make_req(ip: str | None = None, xff: str | None = None, auth: str | None = None) -> Request:
            raw_headers = []
            if xff:
                raw_headers.append((b"x-forwarded-for", xff.encode()))
            if auth:
                raw_headers.append((b"authorization", auth.encode()))
            scope = {
                "type": "http",
                "method": "POST",
                "path": "/v1/chat/completions",
                "query_string": b"",
                "headers": raw_headers,
            }
            if ip:
                scope["client"] = (ip, 50000)
            return Request(scope)

        # 1. 192.168.0.0/16 subnets (local LAN) -> Allowed with NO API key
        assert alpaca_proxy.is_request_authorized(_make_req(ip="192.168.0.1"))
        assert alpaca_proxy.is_request_authorized(_make_req(ip="192.168.1.100"))
        assert alpaca_proxy.is_request_authorized(_make_req(ip="192.168.254.254"))

        # 2. Loopback and container/private networks -> Allowed with NO API key
        assert alpaca_proxy.is_request_authorized(_make_req(ip="127.0.0.1"))
        assert alpaca_proxy.is_request_authorized(_make_req(ip="10.0.1.5"))
        assert alpaca_proxy.is_request_authorized(_make_req(ip="172.17.0.2"))  # Docker default bridge
        assert alpaca_proxy.is_request_authorized(_make_req(ip="172.18.0.10"))  # Docker compose network
        assert alpaca_proxy.is_request_authorized(_make_req(ip="172.28.0.5"))  # Custom Docker subnet
        assert alpaca_proxy.is_request_authorized(_make_req(ip="100.64.0.1"))  # Tailscale / CGNAT private network
        assert alpaca_proxy.is_request_authorized(_make_req(ip="fd00::1"))  # IPv6 ULA / Docker IPv6
        assert alpaca_proxy.is_request_authorized(_make_req(ip="::ffff:172.18.0.4"))  # IPv6-mapped IPv4 container
        assert alpaca_proxy.is_request_authorized(_make_req(ip="host.docker.internal"))  # Docker host lookup
        assert alpaca_proxy.is_request_authorized(_make_req(ip="localhost"))

        # 3. Via X-Forwarded-For header
        assert alpaca_proxy.is_request_authorized(_make_req(xff="192.168.1.42, 10.0.0.1"))
        assert alpaca_proxy.is_request_authorized(_make_req(xff="172.18.0.5, 172.17.0.1"))

        # 4. Public external IP WITHOUT key -> Denied
        assert not alpaca_proxy.is_request_authorized(_make_req(ip="93.184.216.34"))
        assert not alpaca_proxy.is_request_authorized(_make_req(xff="8.8.8.8"))

        # 5. Public external IP WITH valid key -> Allowed
        assert alpaca_proxy.is_request_authorized(_make_req(ip="93.184.216.34", auth="Bearer sk-secret-local-test"))


def test_web_login_flow_and_session_auth():
    from web.app import app

    with patch.dict(os.environ, {"ALPACA_API_KEY": "sk-alpaca-web-test-key"}):
        client = app.test_client()

        # 1. External unauthenticated client accessing dashboard -> Redirected to /login
        resp = client.get("/", environ_base={"REMOTE_ADDR": "93.184.216.34"})
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]

        # 2. External client accessing /login page -> 200 OK
        resp_login_page = client.get("/login", environ_base={"REMOTE_ADDR": "93.184.216.34"})
        assert resp_login_page.status_code == 200
        assert b"Alpaca" in resp_login_page.data

        # 3. External client submitting invalid credentials -> 401 Unauthorized
        resp_bad_login = client.post(
            "/login",
            json={"api_key": "wrong-password"},
            environ_base={"REMOTE_ADDR": "93.184.216.34"},
        )
        assert resp_bad_login.status_code == 401
        assert resp_bad_login.get_json()["success"] is False

        # 4. External client submitting valid credentials -> 200 OK & Sets session
        resp_good_login = client.post(
            "/login",
            json={"api_key": "sk-alpaca-web-test-key"},
            environ_base={"REMOTE_ADDR": "93.184.216.34"},
        )
        assert resp_good_login.status_code == 200
        assert resp_good_login.get_json()["success"] is True

        # 5. External client with authenticated session accessing dashboard -> 200 OK
        resp_auth = client.get("/", environ_base={"REMOTE_ADDR": "93.184.216.34"})
        assert resp_auth.status_code == 200

        # 6. External client logging out -> Clears session & Redirects to /login
        resp_logout = client.get("/logout", environ_base={"REMOTE_ADDR": "93.184.216.34"})
        assert resp_logout.status_code == 302
        assert "/login" in resp_logout.headers["Location"]

        # 7. Local network client accessing dashboard directly -> 200 OK (no login required)
        local_client = app.test_client()
        resp_local_192 = local_client.get("/", environ_base={"REMOTE_ADDR": "192.168.1.50"})
        assert resp_local_192.status_code == 200

        resp_local_docker = local_client.get("/", environ_base={"REMOTE_ADDR": "172.18.0.2"})
        assert resp_local_docker.status_code == 200


# ---------------------------------------------------------------------------
# Syntax / lint gates (false-success fix)
# ---------------------------------------------------------------------------
def _mock_container(exec_return=(0, b"")):
    container = MagicMock()
    container.exec_run.return_value = exec_return
    return container


def test_lint_html_js_rejects_truncated_script():
    # Token-budget cutoff mid-<script>: the start screen still renders, but the
    # game JS is cut off. Must be rejected before any screenshot is trusted.
    truncated = "<!DOCTYPE html><html><head></head><body><canvas id='c'></canvas><script>const state = 'gameover"
    ok, err = _lint_html_js(_mock_container(), truncated)
    assert ok is False
    assert "truncated" in err.lower()


def test_lint_html_js_rejects_missing_closing_html():
    truncated = "<html><body><script>const x=1;</script></body>"  # no </html>
    ok, err = _lint_html_js(_mock_container(), truncated)
    assert ok is False
    assert "html" in err.lower()


def test_lint_html_js_rejects_inline_js_syntax_error():
    code = "<!DOCTYPE html><html><body><canvas id='c'></canvas><script>const state = 'gameover</script></body></html>"
    container = _mock_container(exec_return=(1, b"SyntaxError: Invalid or unexpected token"))
    ok, err = _lint_html_js(container, code)
    assert ok is False
    assert "syntax" in err.lower()


def test_lint_html_js_passes_complete_valid_page():
    code = (
        "<!DOCTYPE html><html><body>"
        "<canvas id='c'></canvas>"
        "<script>const x = 1; window.start = () => document.body.appendChild(document.createElement('p'));</script>"
        "</body></html>"
    )
    ok, err = _lint_html_js(_mock_container(), code)
    assert ok is True
    assert err == ""


def test_lint_code_python_syntax_error():
    container = _mock_container(exec_return=(1, b"SyntaxError: invalid syntax"))
    ok, err = _lint_code(container, "def broken(:\n    pass\n", "python")
    assert ok is False
    assert "syntax" in err.lower()


def test_lint_code_python_valid():
    container = _mock_container(exec_return=(0, b""))
    ok, err = _lint_code(container, "print('hello')\n", "python")
    assert ok is True
    assert err == ""


def test_lint_code_unsupported_lang_is_skipped():
    # go/rust/java/sql already fail their build on broken syntax, so the lint
    # gate must not add a second, unreliable check.
    ok, err = _lint_code(_mock_container(), "garbage", "go")
    assert ok is True
    assert err == ""


def test_grade_code_rejects_truncated_html():
    truncated = "<!DOCTYPE html><html><body><canvas id='c'></canvas><script>const state = 'gameover"
    with patch("docker.DockerClient") as mock_docker:
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_docker.return_value = mock_client
        mock_client.containers.run.return_value = mock_container
        mock_container.exec_run.return_value = (0, b"")
        res = grade_code(truncated, lang="web", ui=True)
    assert res["ran"] is False
    assert res["score"] == 0
    assert "truncated" in (res["error"] or "").lower()


def test_grade_code_fails_inline_js_syntax_error():
    code = "<!DOCTYPE html><html><body><canvas id='c'></canvas><script>const state = 'gameover</script></body></html>"
    with patch("docker.DockerClient") as mock_docker:
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_docker.return_value = mock_client
        mock_client.containers.run.return_value = mock_container
        mock_container.exec_run.return_value = (1, b"SyntaxError: Invalid or unexpected token")
        res = grade_code(code, lang="web", ui=True)
    assert res["ran"] is False
    assert res["score"] == 0
    assert "syntax" in (res["error"] or "").lower()


def test_grade_code_fails_python_syntax_error():
    code = "def broken(:\n    pass\n"
    with patch("docker.DockerClient") as mock_docker:
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_docker.return_value = mock_client
        mock_client.containers.run.return_value = mock_container
        mock_container.exec_run.return_value = (1, b"SyntaxError: invalid syntax")
        res = grade_code(code, lang="python")
    assert res["ran"] is False
    assert res["score"] == 0


def test_grade_code_valid_complete_html_still_passes():
    code = (
        "<!DOCTYPE html><html><body>"
        "<canvas id='c'></canvas>"
        "<script>const x = 1; window.start = () => document.body.appendChild(document.createElement('p'));</script>"
        "</body></html>"
    )
    with patch("docker.DockerClient") as mock_docker, patch("sandbox_exec._read_file") as mock_read:
        content = _png_bytes(shapes=[(5, 5, 60, 60)])
        mock_read.side_effect = lambda c, p: content if p.endswith("out.png") else b""
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_docker.return_value = mock_client
        mock_client.containers.run.return_value = mock_container
        mock_container.exec_run.return_value = (0, b"")
        res = grade_code(code, lang="web", ui=True)
    assert res["ran"] is True
    assert res["score"] == 100


def test_find_web_js_error_flags_runtime_failure():
    stderr = (
        "[INFO:CONSOLE(1)] hello\n"
        "ERROR:dbus/bus.cc(123): some dbus noise\n"
        "[ERROR:CONSOLE(5)] Uncaught ReferenceError: THREE is not defined\n"
    )
    assert "THREE is not defined" in _find_web_js_error(stderr)


def test_find_web_js_error_ignores_dbus_noise():
    stderr = "ERROR:dbus/bus.cc(123): some dbus noise\nERROR:gpu: software renderer initialized\n"
    assert _find_web_js_error(stderr) is None


def test_run_web_ui_fails_on_js_console_error():
    with patch("sandbox_exec._put_file"), patch("sandbox_exec._read_file") as mock_read:
        content = _png_bytes(shapes=[(5, 5, 60, 60)])
        mock_read.side_effect = lambda c, p: (
            b"Uncaught ReferenceError: THREE is not defined\n"
            if p.endswith("ui_stdout.txt")
            else (content if p.endswith("out.png") else b"")
        )
        mock_container = MagicMock()
        mock_container.exec_run.return_value = (0, b"")
        result = _run_web_ui(mock_container, "<script>...</script>", 30, {})
    assert result["ran"] is False
    assert result["ui_rendered"] is False
    assert "console" in (result["error"] or "").lower()
