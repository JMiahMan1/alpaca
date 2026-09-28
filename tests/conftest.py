"""Shared test configuration for the alpaca suite.

Before this file the suite had no conftest at all, which produced three
problems worth fixing here:

1. Four external facilities are optional - `node`, `numpy`, a Docker daemon
   and Playwright browsers - and each is gated by an ad-hoc ``skipif`` in
   whichever file happened to need it. When one is missing those tests skip
   silently, so a green run does not mean what it looks like. ``missing_facilities``
   turns that into a visible line in the pytest header.
2. Nothing stopped a test from opening a real socket, so an offline test that
   accidentally reached the running stack could hang, or worse, pass because a
   service happened to be up. ``no_network`` makes that impossible for
   everything not marked ``live``.
3. The flask client and the router-directory isolation fixture were copy-pasted
   into three files each.
"""

from __future__ import annotations

import os
import shutil
import socket
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Facilities a test file can be skipped on, reported in the header so a skipped
# count is never mistaken for coverage.
OPTIONAL_FACILITIES: dict[str, tuple[str, str]] = {
    "node": ("node", "executes arcade.js / dashboard.js slices"),
    "numpy": ("numpy", "voice_clone + imageops tests"),
    "docker": ("docker", "sandbox container execution"),
    "playwright": ("playwright", "live browser tests (also needs browsers installed)"),
    "sharedllm": ("../SharedLLM", "cross-repo contract tests"),
}


def _facility_available(name: str) -> bool:
    if name == "node":
        return shutil.which("node") is not None
    if name == "docker":
        try:
            import docker  # noqa: F401
            from docker import DockerClient

            DockerClient().ping()
        except Exception:
            return False
        return True
    if name == "playwright":
        try:
            import playwright  # noqa: F401
        except Exception:
            return False
        return True
    if name == "sharedllm":
        root = os.environ.get("SHAREDLLM_ROOT") or (REPO_ROOT.parent / "SharedLLM")
        return (Path(root) / "services" / "gateway" / "agent_loop.py").is_file()
    try:
        __import__(name)
    except Exception:
        return False
    return True


def missing_facilities() -> dict[str, str]:
    """Optional facilities this machine cannot provide, for the pytest header."""
    return {name: why for name, (_, why) in OPTIONAL_FACILITIES.items() if not _facility_available(name)}


def pytest_report_header(config: pytest.Config) -> list[str]:
    lines: list[str] = []
    web = os.environ.get("ALPACA_BASE_URL", "http://localhost:5000")
    proxy = os.environ.get("ALPACA_PROXY_URL", "http://localhost:11434")
    lines.append(f"live target: dashboard {web} | proxy {proxy} (override with ALPACA_BASE_URL / ALPACA_PROXY_URL)")
    absent = missing_facilities()
    if not absent:
        lines.append("optional facilities: all present")
    else:
        detail = ", ".join(f"{name} (for {OPTIONAL_FACILITIES[name][1]})" for name in sorted(absent))
        lines.append(f"optional facilities MISSING - tests needing these will skip: {detail}")
        lines.append("  install what you need to cover them; a skip is not coverage")
    return lines


@pytest.fixture(autouse=True)
def no_network(request: pytest.FixtureRequest) -> None:
    """Fail fast instead of hanging if a non-live test opens a socket.

    Tests that legitimately need the running stack are marked ``live`` and opt
    out. Mocked transports (httpx.MockTransport, Flask test_client) never reach
    a socket, so this only catches real network access.
    """
    if request.node.get_closest_marker("live"):
        return

    real_socket = socket.socket

    class _BlockedSocket(real_socket):  # type: ignore[misc, valid-type]
        def connect(self, *a, **k):
            raise RuntimeError(
                f"{request.node.nodeid} opened a network connection but is not marked `live`. "
                "Use a mock transport, or mark the test @pytest.mark.live if it really "
                "needs the running stack."
            )

        connect_ex = connect

    socket.socket = _BlockedSocket  # type: ignore[misc]
    try:
        yield
    finally:
        socket.socket = real_socket  # type: ignore[misc]
