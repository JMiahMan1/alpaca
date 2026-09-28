"""Every module the code imports must be present in the image that runs it.

This exists because of a bug that reached `main` and broke the deployed
dashboard: `llm_benchmark_suite` gained a module-scope import of
`manuscript_rubric`, a brand-new top-level file, and neither `Dockerfile.web`
nor the compose bind-mount list mentioned it. The web container therefore died
on boot with `ModuleNotFoundError` - and the entire local suite stayed green,
because locally every file is simply present in the working directory.

That is the whole failure mode: the repository is a directory, the container is
a COPY list, and nothing checks that the second one still covers the first. A
new import is invisible until deploy.

The checks below are structural rather than behavioural on purpose - they read
the same three sources the build reads (each Dockerfile's COPY lines, the
compose volumes, and the imports) - so they fail on the pull that introduces the
problem, not on the deploy that trips over it.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
COMPOSE = (REPO / "docker-compose.yml").read_text(encoding="utf-8")

#: service -> (dockerfile, the name of its section in the compose file)
SERVICES = {
    "alpaca-web": ("Dockerfile.web", "alpaca-web"),
    "alpaca-proxy": ("Dockerfile.proxy", "alpaca-proxy"),
    "audio-server": ("Dockerfile.audio", "audio-server"),
    "alpaca-telemetry": ("Dockerfile.proxy", "alpaca-telemetry"),
    "sd-server": ("Dockerfile.sd-server", "sd-server"),
    "llama-server": ("Dockerfile.llama-server", "llama-server"),
}


def _top_level_modules() -> set[str]:
    """Every top-level module in the repo, symlink and hyphen normalised."""
    out = set()
    for p in REPO.glob("*.py"):
        out.add(p.stem)
        if "-" in p.stem:
            out.add(p.stem.replace("-", "_"))
    return out


def _imports_of(path: Path) -> set[str]:
    """Top-level modules this file imports, ignoring stdlib and third-party.

    Only names that resolve to a file in the repo are returned, so the result is
    exactly the set of local modules this file needs present.
    """
    top = _top_level_modules()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:  # pragma: no cover - a broken module fails elsewhere
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return {m for m in found if m in top and m != path.stem}


def _copied_by(dockerfile: str) -> set[str]:
    """Modules a Dockerfile COPYs into the image."""
    path = REPO / dockerfile
    if not path.exists():  # pragma: no cover
        return set()
    out = set()
    for m in re.finditer(r"^COPY\s+(.+?)\s+/?\S*\s*$", path.read_text(encoding="utf-8"), re.M):
        # Every source of a COPY, not just the first: `COPY audio_server.py
        # tts_text.py voice_clone.py /app/` is how the audio image ships the
        # cloner, and reading only the first source made two real modules look
        # absent.
        for src in m.group(1).split():
            if src.startswith("--") or src in {"AS", "."}:
                continue
            stem = Path(src).stem
            if not stem:
                continue
            out.add(stem)
            if "-" in stem:
                out.add(stem.replace("-", "_"))
    return out


def _mounted_for(service_section: str) -> set[str]:
    """Modules bind-mounted into one compose service's /app directory."""
    # Isolate the service's block: 2-space indented keys under its own name.
    m = re.search(rf"^  {re.escape(service_section)}:\n(.*?)(?=^  \S|\Z)", COMPOSE, re.M | re.S)
    if not m:  # pragma: no cover - a renamed service fails the composition test
        return set()
    block = m.group(1)
    out = set()
    for m2 in re.finditer(r"-\s+\./([A-Za-z0-9_.\-/]+\.py):/app/", block):
        stem = Path(m2.group(1)).stem
        out.add(stem)
        if "-" in stem:
            out.add(stem.replace("-", "_"))
    return out


def _available(service: str) -> set[str]:
    dockerfile, section = SERVICES[service]
    return _copied_by(dockerfile) | _mounted_for(section)


#: Which repo files each service's code actually consists of. The entry point
#: plus the modules it imports, resolved transitively.
ENTRY_POINTS = {
    "alpaca-web": ["web/app.py"],
    "alpaca-proxy": ["alpaca-proxy.py"],
    "audio-server": ["audio_server.py"],
    "alpaca-telemetry": ["telemetry_monitor.py"],
}


def _reachable(entry: str, seen: set[str] | None = None) -> set[str]:
    """entry plus every local module it imports, transitively."""
    seen = seen if seen is not None else set()
    path = REPO / entry
    if not path.exists():  # pragma: no cover
        return seen
    for m in _imports_of(path):
        if m in seen:
            continue
        seen.add(m)
        # The module may be `foo.py` or `foo-bar.py`; follow whichever exists.
        for candidate in (f"{m}.py", f"{m.replace('_', '-')}.py"):
            if (REPO / candidate).exists():
                _reachable(candidate, seen)
                break
    return seen


@pytest.mark.parametrize("service,entry", sorted((s, e) for s, es in ENTRY_POINTS.items() for e in es))
def test_every_module_a_service_imports_reaches_its_image(service, entry):
    """The bug this file exists for.

    `llm_benchmark_suite` imported `manuscript_rubric` and no image carried it,
    so the dashboard crash-looped on boot while the local suite was green.
    """
    available = _available(service)
    missing = sorted(m for m in _reachable(entry) if m not in available)
    assert not missing, (
        f"{entry} transitively needs {missing}, but {SERVICES[service][0]} neither COPYs "
        f"nor is bind-mounted with it. Add it to the Dockerfile's COPY list AND to the "
        f"{SERVICES[service][1]} compose volumes (mounted as well as copied, like "
        f"llm_benchmark_suite.py), or the container dies with ModuleNotFoundError on boot."
    )


def test_the_podcast_mixer_reaches_the_web_image():
    """It is inside web/, so the whole-directory mount covers it - pinned so a
    later change that moves it out of web/ cannot quietly drop it."""
    available = _available("alpaca-web")
    assert "podcast_mixer" in available or "web" in available


@pytest.mark.parametrize("service,dockerfile", sorted({(s, d) for s, (d, _) in SERVICES.items()}))
def test_every_declared_service_dockerfile_exists(service, dockerfile):
    assert (REPO / dockerfile).exists(), f"{service} is declared to build from {dockerfile}, which does not exist"


def test_the_docker_sock_mount_is_still_parameterised():
    """Regression: a hard-coded /var/run/docker.sock silently breaks every
    container on colima, which is a very common macOS runtime."""
    assert re.search(r'\$\{DOCKER_SOCK:-/var/run/docker\.sock\}', COMPOSE), (
        "the docker socket mount lost its ${DOCKER_SOCK:-...} default; on a host whose socket is "
        "elsewhere (colima) every container gets a broken socket"
    )


def test_the_two_required_reasoning_vars_are_documented_for_a_fresh_clone():
    """compose fails to interpolate without them, so a fresh clone cannot start
    the stack unless .env.example says they are required."""
    example = (REPO / ".env.example").read_text(encoding="utf-8")
    for var in ("LLAMA_REASONING_BUDGET", "LLAMA_REASONING_FORMAT"):
        assert re.search(rf"^{var}=", example, re.M), f"{var} is required by docker-compose but absent from .env.example"
        # compose reads it as ${VAR:?message} - a hard failure with an
        # instruction, not a silent default - so match the interpolation form.
        assert re.search(rf"\$\{{{var}:", COMPOSE), f"{var} is not required by docker-compose (expected ${{{var}:?hint}})"
