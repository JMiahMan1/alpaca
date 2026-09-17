import io
import json
import os
import tarfile
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image, ImageDraw

from online_providers import OnlineModelProvider
from sandbox_exec import (
    _AUDIO_FFMPEG_SH,
    _AUDIO_PORT,
    _AUDIO_PRESENT_RMS,
    _audio_rms,
    _find_web_js_error,
    _lint_code,
    _lint_html_js,
    _run_ui,
    _run_web_ui,
    _screenshot_has_content,
    ensure_audio_encoder,
    extract_clean_code,
    grade_code,
    run_code_once,
    serve_app,
    serve_ui,
    stop_serve,
    ui_exec,
    ui_restart,
    ui_screenshot,
    ui_status,
)
from web.model_tracker import ModelTracker


@pytest.mark.parametrize("stage", ["get", "remove"])
@pytest.mark.parametrize("missing", [False, True])
def test_stop_serve_missing_container_vs_api_failure(stage, missing):
    from docker.errors import APIError, NotFound

    client = MagicMock()
    operation = client.containers.get if stage == "get" else client.containers.get.return_value.remove
    operation.side_effect = NotFound("missing") if missing else APIError("daemon failure")
    with patch("docker.DockerClient", return_value=client):
        result = stop_serve("arcade-session")
    assert result == ({"stopped": True} if missing else {"stopped": False, "error": "daemon failure"})
    client.containers.get.assert_called_once_with("arcade-session")
    if stage == "remove":
        operation.assert_called_once_with(force=True)
    client.close.assert_called_once_with()


def test_stop_serve_removes_container():
    client = MagicMock()
    with patch("docker.DockerClient", return_value=client):
        assert stop_serve("arcade-session") == {"stopped": True}
    client.containers.get.assert_called_once_with("arcade-session")
    client.containers.get.return_value.remove.assert_called_once_with(force=True)
    client.close.assert_called_once_with()


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


def test_extract_clean_code_rust_use_imports():
    # Unfenced Rust starting with third-party crate imports (not std): the
    # `use` lines are code and must survive extraction, or the build dies
    # with unresolved-import errors (E0425/E0433/E0599).
    raw = (
        "use x11rb::connection::Connection;\n"
        "use x11rb::protocol::xproto::*;\n"
        "use x11rb::COPY_DEPTH_FROM_PARENT;\n"
        "\n"
        "fn main() -> Result<(), Box<dyn std::error::Error>> {\n"
        "    Ok(())\n"
        "}\n"
    )
    cleaned = extract_clean_code(raw, "rust")
    assert cleaned.startswith("use x11rb::connection::Connection;")
    assert "use x11rb::protocol::xproto::*;" in cleaned
    assert "fn main()" in cleaned


def test_extract_clean_code_rust_use_prose_not_code():
    # English prose starting with "Use the ..." must not anchor extraction,
    # but the real `use <crate>::...` import below it must.
    raw = (
        "Here is the complete Rust solution:\n"
        "Use the arrow keys to move the player.\n"
        "use x11rb::protocol::xproto::*;\n"
        "fn main() {\n"
        "}\n"
    )
    cleaned = extract_clean_code(raw, "rust")
    assert cleaned.startswith("use x11rb::protocol::xproto::*;")
    assert "Use the arrow keys" not in cleaned


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


@pytest.mark.parametrize("lang", ["node", "js", "javascript"])
@pytest.mark.parametrize("timeout,budget", [(6, 3000), (30, 8000)])
def test_node_http_ui_launches_chromium(lang, timeout, budget):
    code = "const http = require('http'); http.createServer((req, res) => res.end('Game')).listen(3000, 'localhost');"
    content = _png_bytes(shapes=[(5, 5, 60, 60)])
    output = b"Server listening on localhost:3000\nChromium rendered page\n"
    with (
        patch("docker.DockerClient") as docker,
        patch("sandbox_exec._lint_code", return_value=(True, "")),
        patch("sandbox_exec._put_file") as put,
        patch("sandbox_exec._read_file", side_effect=lambda c, p: content if p.endswith("out.png") else output),
    ):
        container = docker.return_value.containers.run.return_value
        container.exec_run.return_value = (0, b"")
        result = run_code_once(code, lang=lang, timeout=timeout, ui=True)

    files = {call.args[1]: call.args[2] for call in put.call_args_list}
    script = files["/tmp/run_http.sh"].decode()
    assert files["/tmp/code.js"] == code.encode()
    assert "node /tmp/code.js >/tmp/ui_stdout.txt 2>&1 &" in script
    assert "/dev/tcp/localhost/3000" in script
    assert f"timeout {timeout} chromium --headless --no-sandbox --disable-gpu" in script
    assert "--disable-dev-shm-usage" in script
    assert "--enable-logging=stderr" in script
    assert f"--virtual-time-budget={budget} http://localhost:3000/" in script
    assert ">>/tmp/ui_stdout.txt 2>&1" in script
    assert "trap 'kill -9 $SRV_PID 2>/dev/null' EXIT" in script
    assert "exit $?" in script
    assert "/tmp/run_ui.sh" not in files
    assert "Xvfb" not in script
    assert result["ran"] is True
    assert result["ui_rendered"] is True
    assert result["screenshot"]
    assert result["output"] == output.decode()
    assert result["lint_passed"] is True
    assert container.exec_run.call_args.args[0] == ["timeout", "--kill-after=2", str(timeout), "bash", "/tmp/run_http.sh"]
    kwargs = docker.return_value.containers.run.call_args.kwargs
    assert kwargs["network_mode"] == "none"
    assert kwargs["user"] == "sandbox"
    assert kwargs["pids_limit"] == 1024
    container.remove.assert_called_once_with(force=True)


@pytest.mark.parametrize(
    "exit_code,png,output,error",
    [
        (7, None, b"Error: Cannot find module 'express'", "did not become ready"),
        (124, None, b"Server started", "timed out"),
        (1, None, b"Chromium failed", "execution failed"),
        (0, None, b"Server started", "no screenshot"),
        (0, _png_bytes(), b"Server started", "blank screenshot"),
        (1, _png_bytes(shapes=[(5, 5, 60, 60)]), b"Chromium failed", "execution failed"),
        (
            0,
            _png_bytes(shapes=[(5, 5, 60, 60)]),
            b'CONSOLE: "Uncaught ReferenceError: game is not defined"',
            "JS console error",
        ),
    ],
)
def test_node_http_ui_failures_preserve_evidence(exit_code, png, output, error):
    with (
        patch("sandbox_exec._put_file") as put,
        patch("sandbox_exec._read_file", side_effect=lambda c, p: png if p.endswith("out.png") else output),
    ):
        container = MagicMock()
        container.exec_run.return_value = (exit_code, b"")
        result = _run_ui(container, "server.listen(3000)", "js", "node", 30, {"lint_passed": True})

    assert result["ran"] is False
    assert result["lint_passed"] is True
    assert result["exit_code"] == exit_code
    assert result["output"] == output.decode()
    assert error in result["error"]
    assert all(call.args[1] != "/tmp/run_ui.sh" for call in put.call_args_list)


@pytest.mark.parametrize(
    "ext,bin_,code,launch",
    [
        ("py", "python3", "import pygame\npygame.display.set_mode((640, 480))", None),
        ("cpp", "g++", "int main() {}", "/tmp/a.out"),
        ("go", "go", "package main", "/tmp/a.out"),
        ("rs", "rustc", "fn main() {}", "/tmp/a.out"),
    ],
)
def test_native_ui_keeps_desktop_screenshot_flow(ext, bin_, code, launch):
    content = _png_bytes(shapes=[(5, 5, 60, 60)])
    with (
        patch("sandbox_exec._put_file") as put,
        patch("sandbox_exec._read_file", side_effect=lambda c, p: content if p.endswith("out.png") else None),
        patch("sandbox_exec._run_http_ui") as http,
    ):
        container = MagicMock()
        container.exec_run.return_value = (0, b"")
        result = _run_ui(container, code, ext, bin_, 30, {}, launch=launch)

    http.assert_not_called()
    files = {call.args[1]: call.args[2] for call in put.call_args_list}
    script = files["/tmp/run_ui.sh"].decode()
    assert "Xvfb :99" in script
    assert "scrot -o /tmp/out.png" in script
    assert "sleep 8" in script
    assert (launch or f"{bin_} /tmp/code.{ext}") in script
    if ext == "py":
        assert "chromium" not in script
    assert result["ran"] is True
    assert result["ui_rendered"] is True


@pytest.mark.parametrize(
    "ext,bin_,code",
    [
        ("cpp", "g++", "int port = 3000;"),
        ("go", "go", 'http.ListenAndServe(":3000", nil)'),
        ("rs", "rustc", 'TcpListener::bind("127.0.0.1:3000")'),
    ],
)
def test_compiled_ui_with_http_port_gets_chromium(ext, bin_, code):
    content = _png_bytes(shapes=[(5, 5, 60, 60)])
    output = b"server listening on 8080\nchromium\n"
    with (
        patch("sandbox_exec._put_file") as put,
        patch("sandbox_exec._read_file", side_effect=lambda c, p: content if p.endswith("out.png") else output),
    ):
        container = MagicMock()
        container.exec_run.return_value = (0, b"")
        result = _run_ui(container, code, ext, bin_, 30, {}, launch="/tmp/a.out")

    files = {call.args[1]: call.args[2] for call in put.call_args_list}
    script = files["/tmp/run_ui.sh"].decode()
    assert "Xvfb" in script
    assert "scrot" in script
    assert "/dev/tcp/localhost/3000" in script
    browser_branch = script.split('if [ "$READY" = "1" ]; then\n', 1)[1].split("exit $?\nfi\n", 1)
    assert "chromium" in browser_branch[0]
    assert "scrot" not in browser_branch[0]
    assert "scrot" in browser_branch[1]
    assert script.index("touch /tmp/http_ui") < script.index("chromium")
    assert "timeout 30 chromium" in script
    assert "--virtual-time-budget=8000 http://localhost:3000/" in script
    assert "trap 'kill -9 $SRV_PID" in script
    assert container.exec_run.call_args.args[0][0:4] == ["timeout", "--kill-after=2", "30", "bash"]
    assert result["ran"] is True
    assert result["ui_rendered"] is True
    assert result["output"] == output.decode()


@pytest.mark.parametrize("ext,bin_", [("cpp", "g++"), ("go", "go"), ("rs", "rustc")])
def test_compiled_ui_without_http_port_keeps_scrot(ext, bin_):
    with (
        patch("sandbox_exec._put_file") as put,
        patch("sandbox_exec._read_file", side_effect=lambda c, p: None),
    ):
        container = MagicMock()
        container.exec_run.return_value = (0, b"")
        result = _run_ui(container, "int main() {}", ext, bin_, 30, {}, launch="/tmp/a.out")

    script = {call.args[1]: call.args[2] for call in put.call_args_list}["/tmp/run_ui.sh"].decode()
    browser_branch = script.split('if [ "$READY" = "1" ]; then\n', 1)[1].split("exit $?\nfi\n", 1)
    assert "chromium" in browser_branch[0]
    assert "scrot" not in browser_branch[0]
    assert "chromium" not in browser_branch[1]
    assert "scrot -o /tmp/out.png" in browser_branch[1]
    assert "scrot -o /tmp/out.png" in script
    assert "sleep 8" in script
    assert result["ran"] is True
    assert result["ui_rendered"] is False


@pytest.mark.parametrize("ran", [True, False, None])
@pytest.mark.parametrize("lint_passed", [True, False, None])
def test_grade_code_propagates_lint_status_on_every_return(ran, lint_passed):
    run = {"ran": ran, "lint_passed": lint_passed, "output": "evidence", "error": "", "screenshot": "frame"}
    with patch("sandbox_exec.run_code_once", return_value=run):
        result = grade_code("code")
    assert result["lint_passed"] is lint_passed
    assert result["ran"] is ran
    assert result["output"] == "evidence"
    assert result["screenshot"] == "frame"
    assert result["score"] == (100 if ran else (0 if ran is False else None))


@pytest.mark.parametrize(
    "run,ui,error,run_error",
    [
        ({"lint_passed": False, "error": "syntax/lint error: invalid syntax"}, False, "syntax/lint error", False),
        ({"lint_passed": True, "exit_code": 1, "output": "ZeroDivisionError"}, False, "runtime error", True),
        ({"lint_passed": True, "exit_code": 124}, False, "timed out", True),
        ({"lint_passed": True, "error": "execution timed out"}, False, "timed out", True),
        (
            {"lint_passed": True, "exit_code": 0, "ui_rendered": False, "screenshot": "black"},
            True,
            "blank screenshot",
            True,
        ),
        ({"lint_passed": True, "exit_code": 0, "ui_rendered": False}, True, "no screenshot", True),
        ({"lint_passed": True, "error": "JS console error: ReferenceError"}, True, "JS console error", True),
    ],
)
def test_grade_code_distinguishes_lint_runtime_and_render_failures(run, ui, error, run_error):
    with patch("sandbox_exec.run_code_once", return_value={"ran": False, **run}):
        result = grade_code("code", ui=ui)
    assert result["ran"] is False
    assert result["score"] == 0
    assert result["lint_passed"] is run["lint_passed"]
    assert error in result["error"]
    assert result["run_error"] == (result["error"] if run_error else "")
    if run.get("output"):
        assert run["output"] in result["error"]


def _sine_raw(seconds=1.0, rate=22050, freq=440.0, amp=8000.0):
    import math
    import struct

    n = int(seconds * rate)
    return struct.pack(f"<{n}h", *(int(amp * math.sin(2 * math.pi * freq * i / rate)) for i in range(n)))


def test_audio_rms_measures_signal_and_silence():
    import struct

    assert _audio_rms(b"") == 0.0
    assert _audio_rms(b"\x00" * 44100) == 0.0
    assert _audio_rms(_sine_raw()) > _AUDIO_PRESENT_RMS
    # Near-black dither must not count as audio.
    assert _audio_rms(struct.pack("<2205h", *([1] * 2205))) < _AUDIO_PRESENT_RMS


def test_run_ui_reports_audio_present_when_monitor_has_signal():
    with patch("sandbox_exec._put_file") as mock_put, patch("sandbox_exec._read_file") as mock_read:
        content = _png_bytes(shapes=[(5, 5, 60, 60)])
        tone = _sine_raw(seconds=2.0)

        def _fake_read(c, p):
            if p.endswith("out.png"):
                return content
            if p.endswith("audio.raw"):
                return tone
            if p.endswith("audio_state"):
                return b"ok"
            return b""

        mock_read.side_effect = _fake_read
        mock_container = MagicMock()
        mock_container.exec_run.return_value = (0, b"")

        result = _run_ui(mock_container, "import pygame", "py", "python3", 30, {})
        assert result["audio_present"] is True
        assert result["audio_rms"] > _AUDIO_PRESENT_RMS
        assert result["audio_error"] == ""
        # Grade wrapper starts pulse and records the monitor after the build.
        put = {str(c.args[1]): c.args[2] for c in mock_put.call_args_list}
        script = next(v for k, v in put.items() if k.endswith("run_ui.sh")).decode()
        assert "parec -d game_sink.monitor" in script
        assert "kill $PAREC_PID" in script


def test_run_ui_reports_audio_absent_on_silence_or_missing_toolchain():
    # Digital-black monitor: capture worked, game played no sound.
    with patch("sandbox_exec._put_file"), patch("sandbox_exec._read_file") as mock_read:
        content = _png_bytes(shapes=[(5, 5, 60, 60)])
        mock_read.side_effect = (
            lambda c, p: content
            if p.endswith("out.png")
            else (b"\x00" * 44100 if p.endswith("audio.raw") else (b"ok" if p.endswith("audio_state") else b""))
        )
        mock_container = MagicMock()
        mock_container.exec_run.return_value = (0, b"")

        result = _run_ui(mock_container, "import pygame", "py", "python3", 30, {})
        assert result["audio_present"] is False
        assert result["audio_rms"] == 0.0
        assert result["audio_error"] == ""
    # Pre-audio image: no toolchain, honestly reported (not "no audio").
    with patch("sandbox_exec._put_file"), patch("sandbox_exec._read_file") as mock_read:
        content = _png_bytes(shapes=[(5, 5, 60, 60)])
        mock_read.side_effect = (
            lambda c, p: content
            if p.endswith("out.png")
            else (b"no-pulseaudio-toolchain" if p.endswith("audio_state") else b"")
        )
        mock_container = MagicMock()
        mock_container.exec_run.return_value = (0, b"")

        result = _run_ui(mock_container, "import pygame", "py", "python3", 30, {})
        assert result["audio_present"] is False
        assert result["audio_error"] == "no-pulseaudio-toolchain"


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

        # Published websockify port (6080 -> host 39781) + audio MP3 port (8090 -> host 39782)
        mock_container.ports = {"6080/tcp": [{"HostPort": "39781"}], "8090/tcp": [{"HostPort": "39782"}]}

        res = serve_ui("print('hello ui')", lang="python", timeout=5)

        assert res["error"] == ""
        assert res["container_id"] == mock_container.id
        assert res["host_port"] == "39781"
        assert res["audio_host_port"] == "39782"
        assert "url" not in res

        # Sandbox container must be created on the bridge network with the
        # websockify port published to all interfaces (browser may be remote).
        kwargs = mock_client.containers.run.call_args.kwargs
        assert kwargs["network_mode"] == "bridge"
        assert kwargs["ports"] == {"6080/tcp": None, "8090/tcp": None}
        assert kwargs["user"] == "sandbox"
        assert kwargs["mem_limit"] == "256m"
        assert kwargs["pids_limit"] == 128
        # Exact container name: the whole launch (sweep -> create -> setup ->
        # return) holds the UI launch lock, so an overlapping launch can only
        # replace a live, already-returned session — never delete one
        # mid-launch (dead session, noVNC stuck at "connecting"). Non-exclusive
        # still clears the exact-name leftover without a prefix sweep.
        assert kwargs["name"] == "alpaca-ui"
        mock_client.containers.get.assert_called_once_with("alpaca-ui")
        mock_client.containers.get.return_value.remove.assert_called_once_with(force=True)
        # Non-exclusive launches do not sweep by prefix.
        mock_client.containers.list.assert_not_called()


def test_serve_ui_exclusive_sweeps_suffixed_relics():
    """exclusive=True removes <name> and <name>-* strays, keeps others."""
    with patch("docker.DockerClient") as mock_docker:
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_docker.return_value = mock_client
        mock_client.containers.run.return_value = mock_container
        mock_container.ports = {"6080/tcp": [{"HostPort": "39781"}], "8090/tcp": [{"HostPort": "39782"}]}

        def _named(n):
            c = MagicMock()
            c.name = n
            return c

        exact, stray, other = _named("alpaca-ui"), _named("alpaca-ui-deadbeef"), _named("alpaca-proxy")
        mock_client.containers.list.return_value = [exact, stray, other]

        res = serve_ui("print('hi')", lang="python", timeout=5, exclusive=True)

        assert res["error"] == ""
        mock_client.containers.list.assert_called_once_with(all=True, filters={"name": "alpaca-ui"})
        exact.remove.assert_called_once_with(force=True)
        stray.remove.assert_called_once_with(force=True)
        other.remove.assert_not_called()
        # New containers use the exact name (the stray sweep is only legacy
        # cleanup for pre-fix "<name>-<suffix>" containers).
        run_kwargs = mock_client.containers.run.call_args.kwargs
        assert run_kwargs["name"] == "alpaca-ui"

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
        # Virtual audio chain: pulse null sink is set up in the wrapper ...
        assert "pulseaudio --start" in script
        assert "module-null-sink sink_name=game_sink" in script
        # ... but the MP3 encoder is deliberately NOT in the wrapper: it is
        # started lazily per listener (ensure_audio_encoder), because a
        # persistent encoder buffers live audio with no client connected and
        # firehoses minutes of stale backlog at the next listener.
        assert "ffmpeg" not in script
        assert "audio.mp3" not in script

        # App pipeline launched detached.
        exec_calls = [c.args[0] for c in mock_container.exec_run.call_args_list]
        assert any(isinstance(a, list) and any("run_ui.sh" in x for x in a) for a in exec_calls)


def test_audio_ffmpeg_command_is_single_shot():
    """The lazy encoder serves one listener then exits (no restart loop).

    A persistent loop re-opens the Pulse input with no client connected and
    accrues minutes of stale backlog server-side; a fresh single-shot ffmpeg
    always joins the monitor at the live edge.
    """
    assert "-listen 1" in _AUDIO_FFMPEG_SH
    assert "while" not in _AUDIO_FFMPEG_SH
    assert "audio.mp3" in _AUDIO_FFMPEG_SH
    assert f"{_AUDIO_PORT}" in _AUDIO_FFMPEG_SH
    # Replaces the shell so the ffmpeg process itself is the exec-session
    # leader (clean cmdline, reaped by the runtime on exit).
    assert "exec ffmpeg" in _AUDIO_FFMPEG_SH
    assert "-probesize 32 -analyzeduration 1" in _AUDIO_FFMPEG_SH
    assert "-fragment_size 3528" in _AUDIO_FFMPEG_SH
    assert "-reservoir 0" in _AUDIO_FFMPEG_SH
    assert "-flush_packets 1" in _AUDIO_FFMPEG_SH
    assert _AUDIO_FFMPEG_SH.index("-fragment_size") < _AUDIO_FFMPEG_SH.index("-i game_sink.monitor")


def test_ensure_audio_encoder_starts_detached_ffmpeg():
    """ensure_audio_encoder fire-and-forget starts the encoder as sandbox."""
    with patch("sandbox_exec._ui_container") as mock_ui:
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_ui.return_value = (mock_client, mock_container)
        assert ensure_audio_encoder("cid123") is True
        mock_container.exec_run.assert_called_once()
        args, kwargs = mock_container.exec_run.call_args
        assert args[0][:2] == ["/bin/bash", "-c"]
        assert "ffmpeg" in args[0][2] and "audio.mp3" in args[0][2]
        assert kwargs.get("user") == "sandbox"
        assert kwargs.get("detach") is True
        mock_client.close.assert_called_once()


def test_ensure_audio_encoder_missing_container_is_false():
    """Gone container (or no docker) reports False instead of raising."""
    with patch("sandbox_exec._ui_container", return_value=(None, None)):
        assert ensure_audio_encoder("gone") is False


def test_ensure_audio_encoder_exec_error_is_false():
    """A failed exec reports False; the Flask caller still tries to proxy."""
    with patch("sandbox_exec._ui_container") as mock_ui:
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.exec_run.side_effect = RuntimeError("boom")
        mock_ui.return_value = (mock_client, mock_container)
        assert ensure_audio_encoder("cid123") is False
        mock_client.close.assert_called_once()


def test_serve_ui_overlapping_launches_keep_exact_name():
    """Overlapping launches serialize and share the exact container name.

    Regression test for noVNC stuck at "connecting": the whole launch
    (sweep -> create -> setup -> return) holds the UI launch lock, so the
    second launch waits for the first to finish returning its live session
    and can only replace it afterwards — never delete it mid-launch. No
    uuid suffix is needed to keep the returned session alive.
    """
    import threading

    with patch("docker.DockerClient") as mock_docker, patch("time.sleep"):
        mock_client = MagicMock()
        mock_docker.return_value = mock_client
        mock_client.containers.run.side_effect = [MagicMock(), MagicMock()]
        for m in mock_client.containers.run.side_effect:
            m.ports = {"6080/tcp": [{"HostPort": "39781"}], "8090/tcp": [{"HostPort": "39782"}]}
        mock_client.containers.list.return_value = []

        results = []

        def launch(code):
            results.append(serve_ui(code, lang="python", timeout=5, exclusive=True))

        first = threading.Thread(target=launch, args=("print('one')",))
        second = threading.Thread(target=launch, args=("print('two')",))
        first.start()
        second.start()
        first.join()
        second.join()

        assert len(results) == 2
        assert all(r["error"] == "" for r in results)
        names = [c.kwargs["name"] for c in mock_client.containers.run.call_args_list]
        assert names == ["alpaca-ui", "alpaca-ui"]


@pytest.mark.parametrize("exclusive", [False, True])
def test_arcade_launches_overlap_without_replacing_players_or_benchmarks(tmp_path, monkeypatch, exclusive):
    import arcade.app as arcade

    for slug in ("snake", "pong"):
        game_dir = tmp_path / slug
        game_dir.mkdir()
        (game_dir / "game.py").write_text(f"print('{slug}')")
        (game_dir / "meta.json").write_text(json.dumps({"lang": "python"}))
    monkeypatch.setattr(arcade, "GAMES_DIR", tmp_path)
    ready = threading.Barrier(4, timeout=5)
    created = []
    payloads = []
    lock = threading.Lock()
    grade = MagicMock()
    grade.name = "alpaca-grade-running"
    player = MagicMock()
    player.name = "alpaca-arcade-existing"

    def create_container(*args, **kwargs):
        container = MagicMock()
        container.name = kwargs["name"]
        with lock:
            index = len(created)
            container.id = f"session-{index}"
            container.ports = {
                "6080/tcp": [{"HostPort": str(40000 + index * 2)}],
                "8090/tcp": [{"HostPort": str(40001 + index * 2)}],
            }
            created.append(container)
        container.exec_run.side_effect = lambda *a, **kw: ready.wait()
        return container

    def forward_launch(req, timeout):
        payload = json.loads(req.data)
        payloads.append(payload)
        result = serve_ui(**payload)
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(result).encode()
        return response

    def launch(slug):
        with arcade.app.test_client() as client:
            response = client.post(f"/api/games/{slug}/launch")
            assert response.status_code == 200
            return response.get_json()

    with (
        patch("docker.DockerClient") as docker,
        patch("sandbox_exec.time.sleep"),
        patch.object(arcade.urllib.request, "urlopen", side_effect=forward_launch),
    ):
        client = docker.return_value
        client.containers.run.side_effect = create_container
        client.containers.list.return_value = [grade, player]
        client.containers.get.side_effect = KeyError("no previous default session")
        with ThreadPoolExecutor(max_workers=4) as pool:
            benchmark = pool.submit(serve_ui, "print('benchmark')", exclusive=exclusive)
            futures = [pool.submit(launch, slug) for slug in ("snake", "snake", "pong")]
            results = [future.result(timeout=10) for future in futures]
            benchmark_result = benchmark.result(timeout=10)

    assert not benchmark_result["error"]
    assert len({r["container_id"] for r in results} | {benchmark_result["container_id"]}) == 4
    assert len({p["name"] for p in payloads}) == 3
    for payload in payloads:
        assert payload["name"].startswith("alpaca-arcade-")
        assert len(payload["name"].removeprefix("alpaca-arcade-")) == 32
        assert payload["exclusive"] is False
        assert payload["timeout"] == 7200
        assert payload["lang"] == "python"
    for result in results:
        assert result["launcher_url"].endswith(f"/ui/launcher/{result['container_id']}?embed=1")
    assert len({c.ports["6080/tcp"][0]["HostPort"] for c in created}) == 4
    for container in [grade, player, *created]:
        container.remove.assert_not_called()
    if exclusive:
        client.containers.get.assert_not_called()
        client.containers.list.assert_called_once_with(all=True, filters={"name": "alpaca-ui"})
    else:
        client.containers.get.assert_called_once_with("alpaca-ui")
        client.containers.list.assert_not_called()


@pytest.mark.parametrize("name", ["alpaca-ui", "alpaca", "alpaca-arcade"])
def test_exclusive_launch_preserves_arcade_namespace(name):
    with patch("docker.DockerClient") as docker, patch("sandbox_exec.time.sleep"):
        client = docker.return_value
        player = MagicMock()
        player.name = "alpaca-arcade-session"
        exact = MagicMock()
        exact.name = name
        stray = MagicMock()
        stray.name = name + "-legacy"
        client.containers.list.return_value = [player, exact, stray]
        result = serve_ui("print('benchmark')", name=name, exclusive=True)

    assert not result["error"]
    player.remove.assert_not_called()
    exact.remove.assert_called_once_with(force=True)
    if stray.name.startswith("alpaca-arcade-"):
        stray.remove.assert_not_called()
    else:
        stray.remove.assert_called_once_with(force=True)
    assert client.containers.run.call_args.kwargs["name"] == name


@pytest.mark.parametrize("exclusive", [False, True])
def test_arcade_name_collision_does_not_remove_existing_session(exclusive):
    with patch("docker.DockerClient") as docker:
        client = docker.return_value
        client.containers.run.side_effect = RuntimeError("container name already in use")
        result = serve_ui("print('player')", name="alpaca-arcade-session", exclusive=exclusive)

    assert result["container_id"] is None
    assert "already in use" in result["error"]
    client.containers.get.assert_not_called()
    client.containers.list.assert_not_called()
    client.close.assert_called_once()


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


@pytest.mark.parametrize("lang", ["basic", "bas"])
def test_basic_lint_does_not_execute_input(lang, tmp_path):
    from sandbox_exec import _lint_code

    container = MagicMock()
    assert _lint_code(container, 'input "Score: " score$', lang) == (None, "")
    container.exec_run.assert_not_called()


@pytest.mark.parametrize(
    "test_id,lang",
    [
        ("bash_backup_rotate", "bash"),
        ("bash_csv_sums", "bash"),
        ("bash_health_loop", "bash"),
        ("net_http_client", "python"),
        ("net_echo_server", "python"),
        ("bas_grade_calc", "basic"),
    ],
)
def test_cli_fixture_plumbing_and_isolation(test_id, lang, tmp_path):
    import json

    import sandbox_exec

    files = {}
    container = MagicMock()
    container.exec_run.return_value = (0, b"CLI fixture behavior passed")
    with patch("docker.DockerClient") as docker, patch.object(sandbox_exec, "_put_file") as put:
        docker.return_value.containers.run.return_value = container
        put.side_effect = lambda c, path, content: files.update({path: content})
        result = sandbox_exec.grade_code('print "hello"' if lang == "basic" else "print('hello')", lang, test_id=test_id)
    assert result["ran"] is True
    options = docker.return_value.containers.run.call_args.kwargs
    assert options["network_mode"] == "none"
    assert "ports" not in options and "volumes" not in options
    assert options["user"] == "sandbox"
    command = container.exec_run.call_args.args[0]
    assert command[:3] == ["python3", "/tmp/cli_fixture.py", test_id]
    assert json.loads(command[3])[0] == {"basic": "yabasic", "python": "python3", "bash": "bash"}[lang]
    assert "/tmp/stdin.txt" not in files
    assert files["/tmp/cli_fixture.py"] == sandbox_exec._CLI_FIXTURE_RUNNER.encode()
    container.remove.assert_called_once_with(force=True)


def test_unknown_fixture_cannot_supply_shell_and_games_keep_stdin(tmp_path):
    import sandbox_exec

    files = {}
    container = MagicMock()
    container.exec_run.return_value = (0, b"ok")
    with patch("docker.DockerClient") as docker, patch.object(sandbox_exec, "_put_file") as put:
        docker.return_value.containers.run.return_value = container
        put.side_effect = lambda c, path, content: files.update({path: content})
        sandbox_exec.grade_code("print(input())", test_id="; touch /tmp/unwanted")
    assert files["/tmp/stdin.txt"] == sandbox_exec._SAMPLE_STDIN.encode()
    assert "/tmp/cli_fixture.py" not in files
    assert "unwanted" not in str(container.exec_run.call_args)


def test_fixture_rejects_incompatible_repair_language(tmp_path):
    assert grade_code("print('wrong language')", "python", test_id="bash_csv_sums")["ran"] is False


def test_basic_runtime_diagnostics_are_preserved(tmp_path):
    import sandbox_exec

    container = MagicMock()
    container.exec_run.return_value = (1, b"line 3: invalid input at EOF")
    with patch("docker.DockerClient") as docker, patch.object(sandbox_exec, "_put_file"):
        docker.return_value.containers.run.return_value = container
        result = grade_code('input "N: " n', "basic", test_id="bas_fibonacci")
    assert result["lint_passed"] is None
    assert "invalid input at EOF" in result["run_error"]
    assert "syntax/lint" not in result["error"]
    assert "</dev/null" not in str(container.exec_run.call_args)


_CLI_REFERENCE_SCRIPTS = {
    "bash_backup_rotate": r'''src="$1"
dest="$2"
mkdir -p "$dest"
archive="$dest/backup-$(date +%Y%m%d-%H%M%S).tar.gz"
if tar -czf "$archive" -C "$src" .; then
    ls -t "$dest"/backup-*.tar.gz | tail -n +6 | while IFS= read -r path; do rm -- "$path"; done
    printf 'Backup complete: %s\n' "$archive"
else
    printf 'Backup failed\n'
    exit 1
fi
''',
    "bash_csv_sums": r'''if [ ! -f "$1" ]; then printf 'Error: missing file\n'; exit 1; fi
awk -F, '{for (i=1;i<=NF;i++) if ($i ~ /^-?[0-9]+([.][0-9]+)?$/) {sum[i]+=$i; seen[i]=1}} END {for (i in seen) printf "Column %d total: %g\n", i, sum[i]}' "$1"
''',
    "bash_health_loop": r'''URL="$1"
MAX_ATTEMPTS=5
TIMEOUT=5
n=1
while [ "$n" -le "$MAX_ATTEMPTS" ]; do
    code=$(curl -f --max-time "$TIMEOUT" -s -o /dev/null -w '%{http_code}' "$URL")
    status=$?
    printf 'Attempt %s: HTTP %s\n' "$n" "$code"
    if [ "$status" -eq 0 ]; then printf 'Endpoint is healthy\n'; exit 0; fi
    sleep 0.01
    n=$((n+1))
done
printf 'Endpoint is DOWN\n'
exit 1
''',
    "net_http_client": '''import urllib.request
with urllib.request.urlopen(input("URL: ")) as response:
    body = response.read()
    print(response.status, len(body))
    print(body[:300].decode())
''',
    "net_echo_server": '''import socket

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 9000))
        listener.listen()
        try:
            while True:
                connection, address = listener.accept()
                print(address, flush=True)
                with connection:
                    while data := connection.recv(4096):
                        connection.sendall(data)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main()
''',
    "bas_grade_calc": '''score = float(input("Score: "))
while not 0 <= score <= 100:
    print("Invalid score")
    score = float(input("Score: "))
grade = next((g for threshold, g in [(90, "A"), (80, "B"), (70, "C"), (60, "D")] if score >= threshold), "F")
print(f"Score: {score:g} Grade: {grade}")
''',
}


@pytest.mark.parametrize("test_id", _CLI_REFERENCE_SCRIPTS)
@pytest.mark.parametrize("broken", [False, True])
def test_cli_behavior_runner_rejects_noops(tmp_path, test_id, broken):
    import json
    import os
    import socket
    import subprocess
    import sys

    from sandbox_exec import _CLI_FIXTURE_RUNNER

    script = _CLI_REFERENCE_SCRIPTS[test_id]
    runner_code = _CLI_FIXTURE_RUNNER
    if test_id == "net_echo_server":
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        script = script.replace("9000", str(port))
        runner_code = runner_code.replace("9000", str(port))
    interpreter = "bash" if test_id.startswith("bash_") else sys.executable
    if broken:
        script = "exit 0" if interpreter == "bash" else "pass"
    program = tmp_path / "code with spaces"
    program.write_text(script)
    runner = tmp_path / "fixture.py"
    runner.write_text(runner_code)
    result = subprocess.run(
        [sys.executable, str(runner), test_id, json.dumps([interpreter, str(program)])],
        cwd=tmp_path,
        env={**os.environ, "TMPDIR": str(tmp_path)},
        text=True,
        capture_output=True,
        timeout=25,
    )
    assert result.returncode == (1 if broken else 0), result.stdout + result.stderr
    assert ("CLI fixture failed" in result.stderr) is broken
    assert not list(tmp_path.glob("alpaca-fixture-*"))


@pytest.mark.asyncio
@pytest.mark.parametrize("test_id", _CLI_REFERENCE_SCRIPTS)
async def test_suite_repairs_keep_cli_fixture(tmp_path, monkeypatch, test_id):
    from unittest.mock import AsyncMock

    from llm_benchmark_suite import LLMModelBenchmark
    from sandbox_exec import _CLI_FIXTURE_LANGS

    monkeypatch.chdir(tmp_path)
    benchmark = LLMModelBenchmark.__new__(LLMModelBenchmark)
    lang = _CLI_FIXTURE_LANGS[test_id][0]
    response = f"```{lang}\n{_CLI_REFERENCE_SCRIPTS[test_id]}\n```"
    benchmark.test_model_proxy = AsyncMock(return_value={"response": response})
    result = {}
    with patch("llm_benchmark_suite.grade_code", return_value={"ran": False, "score": 0}) as grade:
        await benchmark._attempt_post_generation_repair(
            "test-model", {"id": test_id, "type": "code", "lang": lang}, result, response, "syntax error"
        )
    assert grade.call_args.kwargs["test_id"] == test_id
    assert result["repaired"] is False


@pytest.mark.asyncio
async def test_suite_primary_execution_gets_cli_fixture(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from llm_benchmark_suite import LLMModelBenchmark

    benchmark = LLMModelBenchmark()
    benchmark.RESULTS_DIR = tmp_path
    benchmark.ARTIFACTS_DIR = tmp_path / "artifacts"
    response = "```bash\n" + _CLI_REFERENCE_SCRIPTS["bash_csv_sums"] + "\n```"
    benchmark.test_model_proxy = AsyncMock(return_value={
        "success": True, "tokens_generated": 50, "latency": 1.0, "response": response,
    })
    with patch("llm_benchmark_suite.grade_code", return_value={"ran": True, "score": 100}) as grade:
        await benchmark.run_model_benchmarks(
            models=["test-model"], use_proxy=True, mode="functional", test_ids=["bash_csv_sums"]
        )
    assert grade.call_args.kwargs["test_id"] == "bash_csv_sums"
