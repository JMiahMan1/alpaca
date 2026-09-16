"""Reusable one-shot code execution for benchmark grading.

Runs a snippet of Python or Node inside the locked-down ``alpaca-sandbox``
container (no network, non-root, memory/pid limits) and returns whether it ran,
its combined output, and the exit code. This is the grading counterpart to the
interactive terminal in ``web/app.py``: same image, same safety posture, but
captures output once instead of streaming it.

For graphical / UI code (e.g. pygame) the ``ui=True`` path launches the program
under a virtual X framebuffer (Xvfb) and captures a screenshot so the rendered
UI can be inspected ("see the UI").

The functions degrade gracefully: if Docker is unavailable they return
``ran=None`` so callers can fall back to a structural score.
"""

from __future__ import annotations

import base64
import contextlib
import io
import os
import re
import struct
import tarfile
import textwrap
import threading
import time
import uuid
from typing import Any

SANDBOX_IMAGE = "alpaca-sandbox:latest"

# Sample input piped to CLI programs that read stdin. Interactive games
# (guess_game, text_adventure, game_checkers_cli, etc.) call input()/readline
# and would otherwise crash with EOFError because exec_run attaches no stdin.
# This gives them a handful of plausible responses so they can complete a
# typical playthrough; programs that ignore stdin are unaffected.
_SAMPLE_STDIN = (
    "\n".join(
        [
            "5",
            "7",
            "1",
            "10",
            "12",
            "3",
            "2",
            "0",
            "42",
            "50",
            "20",
            "15",
            "Alice",
            "Bob",
            "gold",
            "black",
            "left",
            "right",
            "up",
            "down",
            "a3",
            "b4",
            "c1",
            "d2",
            "1 1",
            "2 2",
            "3 3",
            "4 4",
            "5 5",
            "6 6",
            "7 7",
            "8 8",
            "n",
            "y",
            "yes",
            "no",
            "restart",
            "start",
            "q",
            "quit",
            "exit",
            "1",
            "2",
            "3",
            "n",
            "n",
            "y",
            "2",
            "3",
            "0",
            "12",
            "n",
            "e",
        ]
    )
    * 3
    + "\n"
)

try:  # pragma: no cover - present in both web and sandbox images
    import PIL.Image as _PILImage
except Exception:  # pragma: no cover - fall back to "unknown content"
    _PILImage = None  # type: ignore[assignment]


def _screenshot_has_content(png_bytes: bytes) -> bool:
    """Return True when a captured PNG shows a rendered, non-blank frame.

    A UI program that crashes instantly (or never opens a window) still leaves a
    screenshot behind: the virtual X display shows a blank/black 1024x768 frame,
    so ``scrot`` captures a uniform image. Real content (sprites, text, colored
    geometry) produces measurable pixel variance and more than a couple of
    distinct colors. We downscale the image for speed and measure both signals.

    Returns True when content is detected, False for a blank/black frame, and
    True (indeterminate) when PIL is unavailable so we never reject a real UI.
    """
    if not png_bytes:
        return False
    if _PILImage is None:
        return True
    try:
        with io.BytesIO(png_bytes) as buf:
            img = _PILImage.open(buf).convert("RGB")
            img = img.resize((max(1, img.width // 4), max(1, img.height // 4)))
        pixels = list(img.getdata())
        if not pixels:
            return False
        luma = [0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in pixels]
        mean = sum(luma) / len(luma)
        stddev = (sum((x - mean) ** 2 for x in luma) / len(luma)) ** 0.5
        unique = len({px for px in pixels})
        return stddev >= 2.0 or unique >= 5
    except Exception:  # pragma: no cover - malformed PNG is not a rendered UI
        return False


# Console/error markers that indicate a web page's JavaScript failed at runtime.
# Chromium surfaces these on stderr with --enable-logging=stderr; a page that
# throws during its game loop is a broken game even if the static HTML overlay
# (title screen, buttons) still painted a screenshot.
_WEB_JS_ERROR_MARKERS = (
    "uncaught ",
    "referenceerror",
    "typeerror",
    "syntaxerror",
    "is not defined",
    "failed to load resource",
    "err_file_not_found",
    "three is not defined",
    "[error:console",
)


# Native X11 GUI toolchain. Sandbox containers have no network, so compiled GUI
# languages build against dependencies baked into the image: Xlib/ALSA headers,
# the Go module cache (/usr/local/go-mod-cache, usable as a file proxy), and
# the cargo registry (/usr/local/cargo). Prompts pin the exact import paths
# and crate versions below; anything not in the cache fails the build honestly.
# The sandbox has no audio hardware, so prompts require audio init to be
# optional (game must run video-only when opening the device fails).
_GO_OFFLINE_ENV = (
    "GOPROXY=file:///usr/local/go-mod-cache/cache/download "
    "GOSUMDB=off GOFLAGS=-mod=mod GOMODCACHE=/usr/local/go-mod-cache"
)


# RMS energy threshold (16-bit PCM units) for the grade-path audio check:
# digital black measures 0; any real tone/music/noise measures in the
# hundreds-to-thousands. 30 sits far above dither/quantization noise while
# catching quiet title music. Only ambient audio inside the screenshot window
# counts — a game silent until first keypress measures absent (honest: the
# grader cannot play it).
_AUDIO_PRESENT_RMS = 30.0


def _audio_rms(raw: bytes) -> float:
    """RMS energy of raw s16le mono PCM bytes (0.0 for empty/undecodable)."""
    if not raw or len(raw) < 2:
        return 0.0
    n = len(raw) // 2
    try:
        samples = struct.unpack(f"<{n}h", raw[: n * 2])
    except struct.error:  # pragma: no cover - defensive
        return 0.0
    return (sum(s * s for s in samples) / n) ** 0.5


# Grade-path audio capture (no ffmpeg/streaming here — just record the null
# sink monitor to a file for the RMS check in _run_ui). Starts AFTER the
# build prelude (recording a multi-minute cold cargo build would waste ~13MB
# of disk) and stops right after the screenshot. Best-effort: pre-audio
# images lack pulseaudio/parec, so the toolchain is probed and the outcome
# (ok / reason) lands in /tmp/audio_state for the Python side to report
# honestly instead of scoring silence as "no audio".
_AUDIO_GRADE_SETUP = (
    "PAREC_PID=\n"
    "if command -v pulseaudio >/dev/null 2>&1 && command -v parec >/dev/null 2>&1 "
    "&& command -v pactl >/dev/null 2>&1; then\n"
    "export XDG_RUNTIME_DIR=/tmp/pulse-$(id -u)\n"
    "mkdir -p \"$XDG_RUNTIME_DIR\"\n"
    "printf 'pcm.!default { type pulse }\\nctl.!default { type pulse }\\n' > ~/.asoundrc\n"
    "pulseaudio --start --exit-idle-time=-1 >/dev/null 2>&1\n"
    "for i in 1 2 3 4 5 6 7 8 9 10; do pactl info >/dev/null 2>&1 && break; sleep 0.5; done\n"
    "pactl load-module module-null-sink sink_name=game_sink sink_properties=device.description=GameAudio "
    "rate=44100 >/dev/null 2>&1\n"
    "pactl set-default-sink game_sink >/dev/null 2>&1\n"
    "parec -d game_sink.monitor --format=s16le --rate=22050 --channels=1 /tmp/audio.raw >/dev/null 2>&1 &\n"
    "PAREC_PID=$!\n"
    "echo ok > /tmp/audio_state\n"
    "else\n"
    "echo no-pulseaudio-toolchain > /tmp/audio_state\n"
    "fi\n"
)


# Xvfb needs a moment to create its socket; a bare `sleep 1` is racy (a fast
# cold-cache-or-warm build can start the app before the socket exists, and
# XOpenDisplay then fails silently -> no window, blank screenshot). Poll for
# the socket instead (≤5s).
_XVFB_WAIT = "for i in 1 2 3 4 5 6 7 8 9 10; do [ -S /tmp/.X11-unix/X99 ] && break; sleep 0.5; done\n"


# Virtual audio chain for served UI sessions. Sandbox containers have no sound
# card, so games play into a PulseAudio null sink ("game_sink"). This setup
# snippet only starts the daemon + sink and redirects ALSA clients (Go oto,
# Rust rodio) into Pulse via ~/.asoundrc (needs libasound2-plugins);
# SDL/pygame finds the server via XDG_RUNTIME_DIR.
# The MP3 encoder is NOT started here: it is lazy (see _AUDIO_FFMPEG_SH /
# ensure_audio_encoder). A persistent ffmpeg would open the Pulse input while
# no listener is connected and swallow live audio into a server-side backlog,
# then firehose minutes of stale audio at the next client (~3-4x wire speed),
# so heard audio would drift minutes behind play. A freshly started reader
# always joins at the live edge, so each listener gets a fresh encoder.
# MP3 (not Ogg) so it plays in every desktop/mobile browser incl. Safari.
# The wrapper runs once per container (ui_restart only relaunches the app),
# so no idempotency guard is needed.
_AUDIO_PORT = 8090
_AUDIO_SETUP = (
    "export XDG_RUNTIME_DIR=/tmp/pulse-$(id -u)\n"
    "mkdir -p \"$XDG_RUNTIME_DIR\"\n"
    "printf 'pcm.!default { type pulse }\\nctl.!default { type pulse }\\n' > ~/.asoundrc\n"
    "pulseaudio --start --exit-idle-time=-1 >/dev/null 2>&1\n"
    "for i in 1 2 3 4 5 6 7 8 9 10; do pactl info >/dev/null 2>&1 && break; sleep 0.5; done\n"
    "pactl load-module module-null-sink sink_name=game_sink sink_properties=device.description=GameAudio "
    "rate=44100 >/dev/null 2>&1\n"
    "pactl set-default-sink game_sink >/dev/null 2>&1\n"
)

# Single-shot MP3 encoder for the game_sink monitor (started on demand by
# ensure_audio_encoder, NOT by the serve wrapper). Serves exactly one
# listener on container port 8090 then exits (``-listen 1`` without a restart
# loop): the browser closing/pausing the stream ends the encoder, so no
# backlog can accrue between listeners and idle containers burn no CPU.
# Started via ``/bin/bash -c`` as user ``sandbox`` with ``exec`` so the
# ffmpeg process itself is the exec session leader (clean cmdline containing
# "audio.mp3", reaped by the container runtime on exit).
_AUDIO_FFMPEG_SH = (
    "export XDG_RUNTIME_DIR=/tmp/pulse-$(id -u); "
    "exec ffmpeg -hide_banner -loglevel error -probesize 32 -analyzeduration 1 "
    "-f pulse -sample_rate 44100 -channels 2 -fragment_size 3528 -i game_sink.monitor "
    "-c:a libmp3lame -b:a 96k -ac 2 -ar 44100 -reservoir 0 "
    "-f mp3 -avioflags direct -flush_packets 1 -listen 1 "
    f"http://0.0.0.0:{_AUDIO_PORT}/audio.mp3"
)


def ensure_audio_encoder(container_id: str) -> bool:
    """Start a serving container's MP3 audio encoder (fire-and-forget).

    The encoder binds the container's port 8090, so at most one can run: if a
    previous listener's encoder is still alive this start fails with
    EADDRINUSE and exits harmlessly; if none is running this one binds and
    serves the next connection from the live edge. Either way the caller just
    proceeds to connect (with a short retry covering encoder boot). Returns
    True if the start was issued, False when the container is gone or docker
    is unreachable. Never raises.
    """
    client, container = _ui_container(container_id)
    if container is None:
        return False
    try:
        container.exec_run(
            ["/bin/bash", "-c", _AUDIO_FFMPEG_SH],
            user="sandbox",
            workdir="/tmp",
            detach=True,
        )
        return True
    except Exception:
        return False
    finally:
        with contextlib.suppress(Exception):
            if client is not None:
                client.close()



def _go_build_sh(src: str, out: str) -> str:
    """Shell snippet: build a single-file Go GUI program offline into ``out``.

    Scaffolds a throwaway module (``go mod init`` + ``tidy`` resolves imports
    against the baked module cache). Assumes the cleaned source is at ``src``.
    """
    return (
        f"rm -rf /tmp/gomod && mkdir -p /tmp/gomod && cp {src} /tmp/gomod/main.go"
        f" && cd /tmp/gomod && go mod init game >/dev/null 2>&1"
        f" && {_GO_OFFLINE_ENV} go mod tidy >/tmp/go_build.log 2>&1"
        f" && {_GO_OFFLINE_ENV} go build -o {out} . >>/tmp/go_build.log 2>&1"
    )


_RUST_TOML = (
    '[package]\nname = "game"\nversion = "0.1.0"\nedition = "2021"\n'
    '\n[dependencies]\nx11rb = "0.13"\nrodio = "0.20"\n'
)


def _rust_build_sh(src: str, profile: str = "debug") -> str:
    """Shell snippet: build a single-file Rust GUI program offline.

    Scaffolds a cargo project with the baked x11rb/rodio crates. ``debug`` for
    grading (faster build), ``release`` for serving (smoother play). Returns
    the snippet; the binary lands at /tmp/cargoproj/target/<profile>/game.
    """
    flag = " --release" if profile == "release" else ""
    return (
        f"rm -rf /tmp/cargoproj && mkdir -p /tmp/cargoproj/src && cp {src} /tmp/cargoproj/src/main.rs"
        f" && printf '%s' '{_RUST_TOML}' > /tmp/cargoproj/Cargo.toml"
        f" && cd /tmp/cargoproj && cargo build --offline{flag} >/tmp/cargo_build.log 2>&1"
    )


def _rust_bin(profile: str = "debug") -> str:
    return f"/tmp/cargoproj/target/{profile}/game"


def _guarded_native_build(build_sh: str, log: str, binary: str, run_cmd: str) -> str:
    """Chain an offline native build with its run, surfacing build failures.

    The build itself logs to ``log`` (keeps UI stdout clean), but the tail is
    always echoed so a failed build shows the compiler error in the run output
    instead of dying as a silent blank frame. ``test -x`` keeps a failed
    build from exec'ing a stale binary left by a previous run.
    """
    return f"{build_sh}; tail -5 {log} 2>/dev/null; test -x {binary} && {run_cmd}"


_CPP_GUI_LIBS = "-lX11 -lasound"


def _cpp_build_sh(src: str, out: str, log: str) -> str:
    """Shell snippet: compile a single-file X11 C++ program offline.

    Libraries come AFTER the source file: GNU ld resolves symbols
    left-to-right, so ``-lX11`` before the source silently discards the
    library and the link dies with undefined references (a blank frame
    in the grader, not a loud error).
    """
    return f"g++ -std=c++17 -O2 -o {out} {src} {_CPP_GUI_LIBS} >{log} 2>&1"


def _lint_html_js(container, code: str) -> tuple[bool, str]:
    """Syntax-check a web page and reject truncated/broken output.

    Returns ``(ok, error)``. Token-budget cutoff frequently truncates an HTML
    response mid-``<script>``, leaving the start-screen overlay intact but the
    game code itself syntactically broken. ``node --check`` on each inline
    script catches exactly that, and structural checks reject a page that is
    missing its closing tags entirely.
    """
    low = code.lower()
    if "<script" in low and low.count("<script") != low.count("</script>"):
        return False, "HTML truncated: unclosed <script> tag"
    if "<html" in low and "</html>" not in low:
        return False, "HTML truncated: missing </html>"
    if "<body" in low and "</body>" not in low:
        return False, "HTML truncated: missing </body>"
    inline = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", code, re.S | re.I)
    for i, js in enumerate(inline):
        if not js.strip():
            continue
        _put_file(container, f"/tmp/lint_{i}.js", js.encode("utf-8"))
        try:
            ec, out = container.exec_run(
                ["node", "--check", f"/tmp/lint_{i}.js"], stdout=True, stderr=True, tty=False, demux=False
            )
        except Exception:  # pragma: no cover - lint infra failure shouldn't fail the benchmark
            continue
        if ec != 0:
            txt = out.decode("utf-8", "replace") if isinstance(out, (bytes, bytearray)) else str(out)
            return False, f"JS syntax error in inline script {i}: {txt.strip()[:300]}"
    return True, ""


def _lint_code(container, code: str, lang: str) -> tuple[bool, str]:
    """Run a language-appropriate syntax check inside the container.

    Returns ``(ok, error)``. A syntax error means the model's code is malformed
    or truncated (common when it burns its whole token budget mid-generation),
    so the benchmark must fail rather than grade a partial run. Compiled
    languages (go/rust/java/sql) are skipped: their compiler already fails the
    build when the code is broken.
    """
    if lang in ("html", "htm", "web"):
        return _lint_html_js(container, code)
    if lang in ("python", "py"):
        fname, check = "/tmp/lint.py", ["python3", "-m", "py_compile", "/tmp/lint.py"]
    elif lang in ("node", "js", "javascript"):
        fname, check = "/tmp/lint.js", ["node", "--check", "/tmp/lint.js"]
    elif lang in ("bash", "sh"):
        fname, check = "/tmp/lint.sh", ["bash", "-n", "/tmp/lint.sh"]
    elif lang == "cpp":
        fname, check = "/tmp/lint.cpp", ["bash", "-c", "g++ -std=c++17 -fsyntax-only /tmp/lint.cpp"]
    elif lang in ("basic", "bas"):
        fname, check = "/tmp/lint.bas", ["bash", "-c", "yabasic /tmp/lint.bas </dev/null >/dev/null 2>&1"]
    elif lang in ("pascal", "pas"):
        fname, check = "/tmp/lint.pas", ["bash", "-c", "fpc -S2 -o/tmp/lint_pas /tmp/lint.pas >/dev/null 2>&1"]
    elif lang in ("typescript", "ts"):
        fname, check = (
            "/tmp/lint.ts",
            ["bash", "-c", "tsc --target ES2020 --module commonjs --noEmit /tmp/lint.ts >/dev/null 2>&1"],
        )
    elif lang in ("yaml", "yml"):
        fname, check = (
            "/tmp/lint.yaml",
            ["bash", "-c", "python3 -c \"import yaml,sys; yaml.safe_load(open('/tmp/lint.yaml'))\" >/dev/null 2>&1"],
        )
    elif lang == "terraform":
        fname, check = (
            "/tmp/lint.tf",
            [
                "bash",
                "-c",
                "python3 -c \"import sys; s=open('/tmp/lint.tf').read(); assert s.count('{')==s.count('}'), 'unbalanced braces'; assert 'resource' in s or 'variable' in s or 'provider' in s, 'no terraform blocks'\" >/dev/null 2>&1",
            ],
        )
    elif lang == "rpm":
        fname, check = "/tmp/lint.spec", ["bash", "-c", "rpmspec -P /tmp/lint.spec >/dev/null 2>&1"]
    else:
        return True, ""
    _put_file(container, fname, code.encode("utf-8"))
    try:
        ec, out = container.exec_run(check, stdout=True, stderr=True, tty=False, demux=False)
    except Exception:  # pragma: no cover - lint infra failure shouldn't fail the benchmark
        return True, ""
    if ec == 0:
        return True, ""
    txt = out.decode("utf-8", "replace") if isinstance(out, (bytes, bytearray)) else str(out)
    return False, txt.strip()[:300]


def _find_web_js_error(output: str) -> str | None:
    """Return the first line of chromium stderr that looks like a real JS error.

    Chromium logs page console/JS errors to stderr when run with
    ``--enable-logging=stderr``. Ignore the dbus/GL noise that headless
    chromium always prints; only genuine JS failures (Uncaught, ReferenceError,
    failed resource loads, "three is not defined", etc.) are treated as a
    broken game.
    """
    for line in (output or "").splitlines():
        low = line.lower()
        if "dbus" in low or "gpu" in low or "gl_" in low or "fontconfig" in low:
            continue
        if any(marker in low for marker in _WEB_JS_ERROR_MARKERS):
            return line.strip()[:200]
    return None


_CODE_FIRST_LINE_STARTERS = {
    "python": ("import ", "from ", "def ", "class ", "async def", "if __name__", "#!/", "@"),
    "html": ("<!doctype", "<html", "<head", "<script", "<style", "<body", "<!--"),
    "javascript": ("const ", "let ", "var ", "function ", "import ", "export ", "require(", "//", "'use strict'"),
    "node": ("const ", "let ", "var ", "function ", "import ", "export ", "require(", "//", "'use strict'", "#!/"),
    "typescript": ("const ", "let ", "var ", "function ", "import ", "export ", "require(", "//"),
    "rust": ("use ", "fn ", "mod ", "struct ", "enum ", "#!["),
    "go": ("package ", "import ", "import (", "func ", "//"),
    "cpp": ("#include", "int main", "using namespace", "class ", "//"),
    "c": ("#include", "int main", "void ", "//"),
}
_DEFAULT_CODE_STARTERS = _CODE_FIRST_LINE_STARTERS["python"]


def _block_starts_like_code(block: str, lang: str) -> bool:
    """Heuristic: does a fenced block's first non-empty line look like real code for ``lang``?

    Used to prefer executable code blocks over plan/reasoning snippets that some
    models wrap in fences.
    """
    starters = _CODE_FIRST_LINE_STARTERS.get(lang.lower(), _DEFAULT_CODE_STARTERS)
    for line in block.splitlines():
        stripped = line.strip().lower()
        if not stripped:
            continue
        return stripped.startswith(starters)
    return False


# Rust `use` imports name a crate/module path (`use x11rb::...;`), never an
# English sentence: a bare startswith("use ") would also match prose like
# "use the arrow keys...". Require a path separator, brace, alias, or the
# trailing semicolon so only real imports anchor the start of extracted code.
_RUST_USE_RE = re.compile(r"^use\s+[a-z_][\w:]*(\s*::|;|\s*\{|\s+as\s+)")


def extract_clean_code(text: str, lang: str = "python") -> str:
    """Extract pure, executable code from an LLM response.

    1. Removes <think>/<thinking> reasoning tags.
    2. Extracts from markdown code fences if present (```python, ```js, etc.),
       preferring blocks whose first line looks like real code over fenced
       plan/reasoning snippets.
    3. Handles truncated fences (generation hit the token cap before the
       closing ```) by taking content from the last fence marker to EOF.
    4. Strips conversational prose preambles and postambles if no fences exist,
       using strict syntax indicators so reasoning/markdown prose is never
       mistaken for code.
    5. Preserves legitimate code comments (#, //, /* */, docstrings).
    """
    if not text:
        return ""

    # 1. Strip think blocks
    cleaned = re.sub(r"<think[^>]*>[\s\S]*?</think[^>]*>", "", text, flags=re.IGNORECASE).strip()

    # 2. Check for markdown code fences (complete pairs only)
    fence_patterns = [
        rf"```(?:{lang}|{lang.lower()}|python3|py|javascript|js|node|html|htm|web|cpp|c\+\+|java|sql|bash|sh|basic|bas|pascal|pas|typescript|ts|yaml|yml|terraform|hcl|spec|rpm)?\s*\n([\s\S]*?)```",
        r"```[\w+-]*\s*\n([\s\S]*?)```",
        r"```([\s\S]*?)```",
    ]
    for pat in fence_patterns:
        matches = re.findall(pat, cleaned, flags=re.IGNORECASE)
        if not matches:
            continue
        # Prefer blocks that start like real code for this language (some models
        # fence their planning notes before the actual implementation).
        coded = [m for m in matches if _block_starts_like_code(m, lang)]
        best = max(coded or matches, key=len)
        if best.strip():
            return textwrap.dedent(best).strip()

    # 2b. Truncated final fence: generation hit the token cap before the closing
    # ``` arrived (common on long game outputs at n_predict caps like 8000).
    # Take everything after the last opening fence marker to EOF.
    trunc = re.search(r"```[\w+-]*[ \t]*\r?\n([\s\S]+)$", cleaned)
    if trunc and len(trunc.group(1).strip().splitlines()) >= 3:
        return textwrap.dedent(trunc.group(1)).strip()

    # 3. Clean leading/trailing non-code lines if no markdown code fences.
    # Strict syntax-only indicators: loose ones like "#", "for ", "while ",
    # "const " matched markdown headings and reasoning bullets, injecting
    # thinking prose into extracted code (seen as lint.py SyntaxErrors).
    lines = cleaned.splitlines()
    start_idx = 0
    code_indicators = (
        "import ",
        "from ",
        "def ",
        "class ",
        "async def",
        "if __name__",
        "#!/",
        "#include",
        "use std::",
        "fn main(",
        "pub fn",
        "<!doctype",
        "<html",
        "<script",
        "<style",
        "function ",
        "require(",
        "'use strict'",
        "package ",
        "public class",
        "using namespace",
        "impl ",
        "extern crate",
    )
    for idx, line in enumerate(lines):
        stripped = line.strip().lower()
        if not stripped:
            continue
        if any(stripped.startswith(ind) for ind in code_indicators):
            start_idx = idx
            break
        if _RUST_USE_RE.match(stripped):
            start_idx = idx
            break
        if any(
            stripped.startswith(lead)
            for lead in (
                "here is",
                "here's",
                "sure",
                "below is",
                "this is",
                "the following",
                "certainly",
                "okay",
            )
        ):
            continue

    end_idx = len(lines)
    for idx in range(len(lines) - 1, start_idx - 1, -1):
        stripped = lines[idx].strip().lower()
        if not stripped:
            continue
        if any(
            stripped.startswith(sign)
            for sign in (
                "hope this",
                "let me know",
                "feel free",
                "this code",
                "explanation:",
                "note:",
                "in this code",
            )
        ):
            end_idx = idx
        else:
            break

    extracted = "\n".join(lines[start_idx:end_idx]).strip()
    return textwrap.dedent(extracted).strip() if extracted else cleaned.strip()


def _put_file(container, path: str, data: bytes) -> None:
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w") as tf:
        info = tarfile.TarInfo(name=os.path.basename(path))
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    container.put_archive("/tmp", tar_bytes.getvalue())


def _read_file(container, path: str) -> bytes | None:
    try:
        stream, _ = container.get_archive(path)
        with io.BytesIO() as buf:
            for chunk in stream:
                buf.write(chunk)
            buf.seek(0)
            with tarfile.open(fileobj=buf) as tf:
                member = tf.getmembers()[0]
                ef = tf.extractfile(member)
                return ef.read() if ef is not None else None
    except Exception:
        return None


def run_code_once(code: str, lang: str = "python", timeout: int = 30, ui: bool = False) -> dict[str, Any]:
    """Execute ``code`` once and capture the result.

    Returns a dict with keys: ``ran`` (bool|None), ``exit_code`` (int|None),
    ``output`` (str), ``error`` (str), ``lang`` (str), and ``screenshot``
    (base64 str, only for UI/web runs that captured a frame).

    ``lang="web"`` (or ``"html"``/``"htm"``) renders the response with
    headless Chromium and screenshots it, so HTML5/Canvas/WebGL/three.js
    games are graded on a real render rather than a blank frame.
    """
    result: dict[str, Any] = {
        "ran": None,
        "exit_code": None,
        "output": "",
        "error": "",
        "lang": lang,
        "screenshot": None,
        "lint_passed": True,
    }
    try:
        import docker
    except Exception as e:  # pragma: no cover - environment dependent
        result["error"] = f"docker unavailable: {e}"
        return result

    cleaned_code = extract_clean_code(code, lang)
    if not cleaned_code:
        result["ran"] = False
        result["exit_code"] = 1
        result["error"] = "no executable code found in response"
        return result

    if lang in ("python", "py"):
        ext, bin_ = "py", "python3"
        cmd = [bin_, f"/tmp/code.{ext}"]
    elif lang in ("node", "js", "javascript"):
        ext, bin_ = "js", "node"
        cmd = [bin_, f"/tmp/code.{ext}"]
    elif lang in ("html", "htm", "web"):
        ext, bin_ = "html", "chromium"
        cmd = ["bash", "/tmp/render_web.sh"]
    elif lang == "cpp":
        ext, bin_ = "cpp", "g++"
        cmd = ["bash", "-c", "g++ -std=c++17 -O2 -o /tmp/a.out /tmp/code.cpp -lX11 -lasound && /tmp/a.out"]
    elif lang == "go":
        ext, bin_ = "go", "go"
        cmd = ["bash", "-c", "cd /tmp && GO111MODULE=off go build -o /tmp/a.out code.go && /tmp/a.out"]
    elif lang == "rust":
        ext, bin_ = "rs", "rustc"
        cmd = ["bash", "-c", "rustc -O -o /tmp/a.out /tmp/code.rs && /tmp/a.out"]
    elif lang == "java":
        ext, bin_ = "java", "javac"
        match = re.search(r"public\s+class\s+([A-Za-z0-9_$]+)", code)
        class_name = match.group(1) if match else "Main"
        cmd = [
            "bash",
            "-c",
            f"cd /tmp && cp code.java {class_name}.java && javac {class_name}.java && java {class_name}",
        ]
    elif lang == "sql":
        ext, bin_ = "sql", "sqlite3"
        cmd = ["bash", "-c", "sqlite3 :memory: < /tmp/code.sql"]
    elif lang in ("bash", "sh"):
        ext, bin_ = "sh", "bash"
        cmd = ["bash", "/tmp/code.sh"]
    elif lang in ("basic", "bas"):
        ext, bin_ = "bas", "yabasic"
        cmd = ["bash", "-c", "yabasic /tmp/code.bas </dev/null 2>&1"]
    elif lang in ("pascal", "pas"):
        ext, bin_ = "pas", "fpc"
        cmd = [
            "bash",
            "-c",
            "cd /tmp && fpc -S2 -o/tmp/pascal_out code.pas >/tmp/fpc_build.log 2>&1 && /tmp/pascal_out",
        ]
    elif lang in ("typescript", "ts"):
        ext, bin_ = "ts", "tsc"
        cmd = [
            "bash",
            "-c",
            "cd /tmp && tsc --target ES2020 --module commonjs code.ts >/tmp/tsc_build.log 2>&1 && node /tmp/code.js",
        ]
    elif lang in ("yaml", "yml"):
        ext, bin_ = "yaml", "python3"
        cmd = ["bash", "-c", "python3 -c \"import yaml,sys; yaml.safe_load(open('/tmp/code.yaml')); print('YAML OK')\""]
    elif lang == "terraform":
        ext, bin_ = "tf", "python3"
        cmd = [
            "bash",
            "-c",
            "python3 -c \"import sys; s=open('/tmp/code.tf').read(); print('terraform config OK, bytes=', len(s))\"",
        ]
    elif lang == "rpm":
        ext, bin_ = "spec", "rpmspec"
        cmd = ["bash", "-c", "rpmspec -P /tmp/code.spec"]
    else:
        result["error"] = f"unsupported language for execution: {lang}"
        return result

    client = None
    container = None
    # Toolchain RSS: bare rustc/go builds fit in 256m, but a cargo offline
    # build (x11rb+rodio) and a Go module build need headroom. One-shot
    # containers on a 30GB host: 2g/1g caps are still tightly bounded.
    grade_mem = "2g" if lang == "rust" else ("1g" if lang == "go" else "256m")
    try:
        client = docker.DockerClient(base_url="unix:///var/run/docker.sock")
        container = client.containers.run(
            SANDBOX_IMAGE,
            command=["sleep", "300"],
            detach=True,
            tty=False,
            stdin_open=True,
            network_mode="none",
            mem_limit=grade_mem,
            pids_limit=1024 if ui else 128,
            user="sandbox",
            working_dir="/tmp",
            name=f"alpaca-grade-{uuid.uuid4().hex[:8]}",
            remove=False,
        )

        # Syntax gate: reject truncated/malformed code BEFORE running it, so a
        # token-budget cutoff (unclosed </script>, missing </html>, invalid JS)
        # fails the benchmark instead of grading a broken partial render.
        lint_ok, lint_err = _lint_code(container, cleaned_code, lang)
        result["lint_passed"] = bool(lint_ok)
        if not lint_ok:
            result["ran"] = False
            result["exit_code"] = 1
            result["error"] = f"syntax/lint error: {lint_err}"
            return result

        if ui:
            if lang in ("html", "htm", "web"):
                return _run_web_ui(container, cleaned_code, timeout, result)
            # Native compiled GUI: build AND run under Xvfb. The bare
            # toolchain command never opens a window (blank frame, false
            # fail), and single-file go/rust builds cannot resolve GUI
            # crates — scaffold the offline module/cargo project first.
            # The build runs in the FOREGROUND inside _run_ui (a cold cargo
            # build takes minutes; backgrounding it would screenshot an
            # empty display while the compiler is still running).
            launch = None
            build = None
            if lang == "cpp":
                build = (
                    _cpp_build_sh("/tmp/code.cpp", "/tmp/a.out", "/tmp/cpp_build.log"),
                    "/tmp/cpp_build.log",
                    "/tmp/a.out",
                )
                launch = "/tmp/a.out"
            elif lang == "go":
                build = (
                    _go_build_sh("/tmp/code.go", "/tmp/a.out"),
                    "/tmp/go_build.log",
                    "/tmp/a.out",
                )
                launch = "/tmp/a.out"
            elif lang == "rust":
                build = (
                    _rust_build_sh("/tmp/code.rs"),
                    "/tmp/cargo_build.log",
                    _rust_bin(),
                )
                launch = _rust_bin()
            return _run_ui(container, cleaned_code, ext, bin_, timeout, result, launch=launch, build=build)

        if lang in ("html", "htm", "web"):
            return _run_web_ui(container, cleaned_code, timeout, result)

        _put_file(container, f"/tmp/code.{ext}", cleaned_code.encode("utf-8"))

        # Pipe sample input to CLI programs that read stdin (interactive
        # games call input()/readline; without this they crash with EOFError
        # because exec_run attaches no stdin).
        stdin_redirect = lang not in ("sql",)
        if stdin_redirect:
            _put_file(container, "/tmp/stdin.txt", _SAMPLE_STDIN.encode("utf-8"))
            if cmd[0] == "bash" and cmd[1] == "-c":
                cmd = ["bash", "-c", f"{cmd[2]} < /tmp/stdin.txt"]
            else:
                cmd = ["bash", "-c", f"{bin_} /tmp/code.{ext} < /tmp/stdin.txt"]

        holder: dict = {}

        def _exec():
            try:
                ec, out = container.exec_run(
                    cmd,
                    stdout=True,
                    stderr=True,
                    tty=False,
                    demux=False,
                )
                holder["exit_code"] = ec
                holder["output"] = out.decode("utf-8", "replace") if isinstance(out, (bytes, bytearray)) else str(out)
            except Exception as e:  # pragma: no cover - runtime dependent
                holder["error"] = str(e)

        t = threading.Thread(target=_exec, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            result["ran"] = False
            result["exit_code"] = 124
            result["error"] = "execution timed out"
            return result

        result["exit_code"] = holder.get("exit_code")
        result["output"] = holder.get("output", "")
        result["error"] = holder.get("error", "")
        result["ran"] = result["exit_code"] == 0
        return result
    except Exception as e:  # pragma: no cover - runtime dependent
        result["error"] = str(e)
        return result
    finally:
        with contextlib.suppress(Exception):
            if container is not None:
                container.remove(force=True)
        with contextlib.suppress(Exception):
            if client is not None:
                client.close()


# Markers (any language) of a program that serves its UI over HTTP instead of
# opening a desktop window. Such programs paint nothing under Xvfb, so the
# grader must load the served page in Chromium rather than screenshot X.
_HTTP_SERVER_MARKERS = (
    ".listen(",
    "listen(",
    "http.server",
    "net/http",
    "createserver",
    "tcplistener",
    "tcp listener",
    "serve_forever",
    "app.run(",
    "httpserver",
)


def _looks_like_http_server(code: str) -> bool:
    """True when the program serves its UI over HTTP (any language)."""
    low = (code or "").lower()
    return any(m in low for m in _HTTP_SERVER_MARKERS)


def _http_server_port(code: str) -> int:
    """Port the program serves on (default 8080)."""
    for line in (code or "").splitlines():
        low = line.lower()
        if any(k in low for k in ("listen", "port", "serve", "bind")):
            for m in re.finditer(r"(?<!\d)(\d{4,5})(?!\d)", line):
                port = int(m.group(1))
                if 1024 <= port <= 65535:
                    return port
    return 8080


def _ui_launch_shell(code: str, lang: str, ext: str, bin_: str) -> str:
    """Shell command that starts a UI program (desktop window or HTTP server).

    Unlike the bare ``{bin_} /tmp/code.{ext}`` invocation, compiled languages
    are built first and the resulting binary is executed — running only the
    toolchain (``go`` with no subcommand, a bare ``rustc``/``g++`` compile)
    never starts the program and grades a blank frame.
    """
    if lang == "go":
        return "cd /tmp && GO111MODULE=off go build -o /tmp/a.out code.go && /tmp/a.out"
    if lang == "rust":
        return "rustc -O -o /tmp/a.out /tmp/code.rs && /tmp/a.out"
    if lang == "cpp":
        return "g++ -std=c++17 -O2 -o /tmp/a.out /tmp/code.cpp && /tmp/a.out"
    if lang == "java":
        match = re.search(r"public\s+class\s+([A-Za-z0-9_$]+)", code)
        class_name = match.group(1) if match else "Main"
        return f"cd /tmp && cp code.java {class_name}.java && javac {class_name}.java && java {class_name}"
    return f"{bin_} /tmp/code.{ext}"


def _run_ui(
    container,
    code: str,
    ext: str,
    bin_: str,
    timeout: int,
    result: dict[str, Any],
    launch: str | None = None,
    build: tuple[str, str, str] | None = None,
) -> dict[str, Any]:
    """Launch ``code`` under Xvfb and capture a screenshot of the rendered UI.

    ``build`` is an optional ``(build_sh, build_log, binary)`` triple for
    compiled languages: the build runs in the foreground (a cold cargo build
    takes minutes — screenshotting before it finishes yields a false blank),
    its log tail lands in the run output, and ``test -x`` keeps a failed
    build from exec'ing a stale binary.
    """
    capture_delay = max(2, min(timeout - 3, 8))
    # Compiled languages must be built AND run; the bare toolchain command
    # never starts the program (no window opens, blank frame, false fail).
    start_cmd = launch or f"{bin_} /tmp/code.{ext}"
    if build is not None:
        build_sh, build_log, binary = build
        prelude = (
            ": > /tmp/ui_stdout.txt\n"
            f"{build_sh}\n"
            f"tail -5 {build_log} >>/tmp/ui_stdout.txt 2>/dev/null\n"
        )
        run_line = f"test -x {binary} && {start_cmd} >>/tmp/ui_stdout.txt 2>&1 &\n"
    else:
        prelude = ""
        run_line = f"{start_cmd} >/tmp/ui_stdout.txt 2>&1 &\n"
    wrapper = (
        "#!/bin/bash\n"
        "Xvfb :99 -screen 0 1024x768x24 >/dev/null 2>&1 &\n"
        "XVFB_PID=$!\n"
        f"{_XVFB_WAIT}"
        "export DISPLAY=:99\n"
        f"{prelude}"
        f"{_AUDIO_GRADE_SETUP}"
        f"{run_line}"
        "PY_PID=$!\n"
        f"sleep {capture_delay}\n"
        "scrot -o /tmp/out.png 2>/dev/null\n"
        "kill $PAREC_PID 2>/dev/null\n"
        "kill -9 $PY_PID 2>/dev/null\n"
        "kill -9 $XVFB_PID 2>/dev/null\n"
    )
    _put_file(container, "/tmp/code." + ext, code.encode("utf-8"))
    _put_file(container, "/tmp/run_ui.sh", wrapper.encode("utf-8"))
    holder: dict = {}

    def _exec():
        try:
            ec, out = container.exec_run(
                ["bash", "/tmp/run_ui.sh"],
                stdout=True,
                stderr=True,
                tty=False,
                demux=False,
            )
            holder["exit_code"] = ec
            holder["output"] = out.decode("utf-8", "replace") if isinstance(out, (bytes, bytearray)) else str(out)
        except Exception as e:  # pragma: no cover - runtime dependent
            holder["error"] = str(e)

    t = threading.Thread(target=_exec, daemon=True)
    t.start()
    t.join(timeout + 5)
    result["output"] = _read_file(container, "/tmp/ui_stdout.txt") or b""
    result["output"] = (
        result["output"].decode("utf-8", "replace") if isinstance(result["output"], bytes) else str(result["output"])
    )
    png = _read_file(container, "/tmp/out.png")
    if png:
        result["screenshot"] = base64.b64encode(png).decode("ascii")
    result["exit_code"] = holder.get("exit_code")
    result["error"] = holder.get("error", "")
    # A GUI app that launches and renders is a working UI. A blank/black frame
    # (instant crash, never-opened window) must NOT count as rendered: the
    # screenshot exists, but it contains no content to grade.
    if png is not None:
        try:
            rendered = _screenshot_has_content(png)
        except Exception:  # pragma: no cover - guard against image decode surprises
            rendered = True
        result["ui_rendered"] = rendered
        # The wrapper script backgrounds the app, so its exit code is always 0
        # regardless of whether the game itself crashed. Rendered content is
        # the only trustworthy success signal when a screenshot exists.
        result["ran"] = rendered
    else:
        result["ui_rendered"] = False
        result["ran"] = result["exit_code"] == 0
    # Measured audio check (replaces the honor-system AUDIO prompt clause for
    # native apps): the wrapper records the null-sink monitor while the app
    # runs; RMS energy above _AUDIO_PRESENT_RMS means the game audibly played.
    audio_state = ((_read_file(container, "/tmp/audio_state") or b"").decode("utf-8", "replace")).strip()
    if audio_state == "ok":
        result["audio_rms"] = _audio_rms(_read_file(container, "/tmp/audio.raw") or b"")
        result["audio_present"] = result["audio_rms"] >= _AUDIO_PRESENT_RMS
        result["audio_error"] = ""
    else:
        result["audio_rms"] = 0.0
        result["audio_present"] = False
        result["audio_error"] = audio_state or "audio capture unavailable"
    return result


def _run_web_ui(container, code: str, timeout: int, result: dict[str, Any]) -> dict[str, Any]:
    """Render an HTML/JS game with headless Chromium and screenshot it.

    Unlike the Xvfb path (which needs a windowing toolkit and can only show a
    black frame for web content), Chromium executes the page's own Canvas/WebGL
    code and paints the real output, so HTML5/three.js games get a genuine
    render that ``_screenshot_has_content`` can grade.
    """
    capture_delay = max(2, min(timeout - 3, 8))
    wrapper = (
        "#!/bin/bash\n"
        "cd /tmp\n"
        # Make the bundled three.js available to code.html when referenced as
        # <script src="three.min.js">. Copy from the image; ignore if absent.
        "if [ -f /usr/local/share/three.min.js ]; then cp /usr/local/share/three.min.js /tmp/three.min.js; fi\n"
        "timeout 25 chromium --headless --no-sandbox --disable-gpu "
        "--disable-dev-shm-usage --hide-scrollbars --force-device-scale-factor=1 "
        "--enable-logging=stderr --window-size=1024,768 --screenshot=/tmp/out.png "
        f"--virtual-time-budget={capture_delay * 1000} file:///tmp/code.html "
        ">/tmp/ui_stdout.txt 2>&1\n"
    )
    _put_file(container, "/tmp/code.html", code.encode("utf-8"))
    _put_file(container, "/tmp/render_web.sh", wrapper.encode("utf-8"))
    holder: dict = {}

    def _exec():
        try:
            ec, out = container.exec_run(
                ["bash", "/tmp/render_web.sh"],
                stdout=True,
                stderr=True,
                tty=False,
                demux=False,
            )
            holder["exit_code"] = ec
            holder["output"] = out.decode("utf-8", "replace") if isinstance(out, (bytes, bytearray)) else str(out)
        except Exception as e:  # pragma: no cover - runtime dependent
            holder["error"] = str(e)

    t = threading.Thread(target=_exec, daemon=True)
    t.start()
    t.join(timeout + 5)
    result["output"] = _read_file(container, "/tmp/ui_stdout.txt") or b""
    result["output"] = (
        result["output"].decode("utf-8", "replace") if isinstance(result["output"], bytes) else str(result["output"])
    )
    png = _read_file(container, "/tmp/out.png")
    if png:
        result["screenshot"] = base64.b64encode(png).decode("ascii")
    result["exit_code"] = holder.get("exit_code")
    result["error"] = holder.get("error", "")
    if png is not None:
        try:
            rendered = _screenshot_has_content(png)
        except Exception:  # pragma: no cover - guard against image decode surprises
            rendered = True
        result["ui_rendered"] = rendered
        result["ran"] = rendered
        # A page can paint a static overlay (title screen, buttons) while its
        # game-loop JavaScript crashed, which would otherwise pass the
        # screenshot check. Fail it when the console shows a real JS error.
        console_error = _find_web_js_error(result["output"])
        if console_error:
            result["ran"] = False
            result["ui_rendered"] = False
            result["error"] = f"JS console error: {console_error}"
    else:
        result["ui_rendered"] = False
        result["ran"] = result["exit_code"] == 0
    return result


def _run_http_ui(
    container,
    code: str,
    ext: str,
    launch: str,
    port: int,
    timeout: int,
    result: dict[str, Any],
) -> bool:
    """Start an HTTP-server program and screenshot its served page.

    Server programs (node/go/rust/cpp/python HTTP games) open no X window,
    so the Xvfb path always grades them a blank frame. Instead the server is
    started in the background, its port is polled, and headless Chromium
    screenshots the served page — the same genuine render ``_run_web_ui``
    gives static pages. Returns True when the port opened (grading done);
    False leaves ``result`` untouched so the caller falls back to Xvfb
    (covers non-servers whose text merely tripped the server heuristic).
    """
    capture_delay = max(2, min(timeout - 3, 8))
    wrapper = (
        "#!/bin/bash\n"
        "cd /tmp\n"
        # Make the bundled three.js available when the served page references
        # <script src="three.min.js"> (same best-effort as the static path).
        "if [ -f /usr/local/share/three.min.js ]; then cp /usr/local/share/three.min.js /tmp/three.min.js; fi\n"
        f"{launch} >/tmp/ui_stdout.txt 2>&1 &\n"
        "SRV_PID=$!\n"
        "READY=0\n"
        "for i in $(seq 1 20); do\n"
        f"  if (echo > /dev/tcp/127.0.0.1/{port}) >/dev/null 2>&1; then READY=1; break; fi\n"
        "  sleep 0.5\n"
        "done\n"
        'if [ "$READY" != "1" ]; then kill -9 $SRV_PID 2>/dev/null; exit 7; fi\n'
        "timeout 25 chromium --headless --no-sandbox --disable-gpu "
        "--disable-dev-shm-usage --hide-scrollbars --force-device-scale-factor=1 "
        "--enable-logging=stderr --window-size=1024,768 --screenshot=/tmp/out.png "
        f"--virtual-time-budget={capture_delay * 1000} http://127.0.0.1:{port}/ "
        ">>/tmp/ui_stdout.txt 2>&1\n"
        "kill -9 $SRV_PID 2>/dev/null\n"
        "exit 0\n"
    )
    _put_file(container, f"/tmp/code.{ext}", code.encode("utf-8"))
    _put_file(container, "/tmp/run_http.sh", wrapper.encode("utf-8"))
    holder: dict = {}

    def _exec():
        try:
            ec, out = container.exec_run(
                ["bash", "/tmp/run_http.sh"],
                stdout=True,
                stderr=True,
                tty=False,
                demux=False,
            )
            holder["exit_code"] = ec
            holder["output"] = out.decode("utf-8", "replace") if isinstance(out, (bytes, bytearray)) else str(out)
        except Exception as e:  # pragma: no cover - runtime dependent
            holder["error"] = str(e)

    t = threading.Thread(target=_exec, daemon=True)
    t.start()
    t.join(timeout + 5)
    if holder.get("exit_code") == 7:
        # Port never opened: not actually a server (or it crashed on boot).
        # Leave result for the Xvfb fallback when it might still paint.
        with contextlib.suppress(Exception):
            err = _read_file(container, "/tmp/ui_stdout.txt") or b""
            holder["boot_output"] = err.decode("utf-8", "replace") if isinstance(err, bytes) else str(err)
        result["_http_fallback_hint"] = holder.get("boot_output", "")
        return False
    result["output"] = _read_file(container, "/tmp/ui_stdout.txt") or b""
    result["output"] = (
        result["output"].decode("utf-8", "replace") if isinstance(result["output"], bytes) else str(result["output"])
    )
    png = _read_file(container, "/tmp/out.png")
    if png:
        result["screenshot"] = base64.b64encode(png).decode("ascii")
    result["exit_code"] = holder.get("exit_code")
    result["error"] = holder.get("error", "")
    if png is not None:
        try:
            rendered = _screenshot_has_content(png)
        except Exception:  # pragma: no cover - guard against image decode surprises
            rendered = True
        result["ui_rendered"] = rendered
        result["ran"] = rendered
        # Same static-overlay guard as the static path: fail a painted page
        # whose game-loop JavaScript crashed.
        console_error = _find_web_js_error(result["output"])
        if console_error:
            result["ran"] = False
            result["ui_rendered"] = False
            result["error"] = f"JS console error: {console_error}"
    else:
        result["ui_rendered"] = False
        result["ran"] = result["exit_code"] == 0
    return True


def grade_code(
    code: str,
    lang: str = "python",
    expected_output: str | None = None,
    timeout: int = 30,
    ui: bool = False,
) -> dict:
    """Run ``code`` and translate the outcome into a 0-100 score.

    Scoring: 0 if it fails to run, 60 for a clean run, plus up to 40 more when
    the output matches ``expected_output`` (or when no expectation is supplied,
    the clean run is accepted as correct). For ``ui=True`` runs, a screenshot
    showing actual rendered content (non-blank frame) counts as a working UI
    (score 100); a blank/black frame — an app that crashed before drawing —
    fails the run.
    """
    if ui and lang in ("rust", "go", "cpp"):
        # Cold offline builds (cargo/go modules) need minutes, not seconds.
        timeout = max(timeout, 300)
    run = run_code_once(code, lang, timeout, ui=ui)
    ran = run.get("ran")
    out = run.get("output", "")
    if ran is None:
        # Sandbox unavailable: caller decides fallback.
        return {
            "ran": None,
            "score": None,
            "output": out,
            "error": run.get("error", ""),
            "exit_code": None,
            "screenshot": run.get("screenshot"),
        }
    if not ran:
        # A hard crash or a timeout (e.g. a non-terminating program) is an honest
        # failure: the code did not run to completion.
        return {
            "ran": False,
            "score": 0,
            "output": out,
            "error": run.get("error", ""),
            "exit_code": run.get("exit_code"),
            "screenshot": run.get("screenshot"),
        }
    score = 100
    if ui:
        score = 100 if run.get("screenshot") else 60
    elif expected_output and expected_output.strip():
        score = 60 + (40 if expected_output.strip() in out else 0)
    return {
        "ran": True,
        "score": score,
        "output": out,
        "error": "",
        "exit_code": run.get("exit_code"),
        "screenshot": run.get("screenshot"),
    }


def serve_app(code: str, lang: str = "html", port: int = 8080, timeout: int = 600) -> dict[str, Any]:
    """Run ``code`` as a long-lived web server and publish it on a host port.

    Returns a dict with ``container_id`` and the assigned ``host_port``. The
    caller builds the browser URL from ``host_port`` and their own host name —
    the server never assumes ``localhost`` (the dashboard may be accessed from a
    remote machine on the LAN or over the internet). The port is published on
    all interfaces so it is reachable remotely. Unlike the one-shot grader, this
    container stays up (bridge network, published port) so the rendered
    web/Node/Python app can actually be viewed.

    Only use this for code the user explicitly asked to view; it intentionally
    enables networking (the grading sandbox does not).
    """
    result: dict[str, Any] = {"container_id": None, "host_port": None, "error": ""}
    try:
        import docker
    except Exception as e:  # pragma: no cover - environment dependent
        result["error"] = f"docker unavailable: {e}"
        return result

    if lang in ("html", "htm", "web"):
        fname, cmd = "index.html", ["python3", "-m", "http.server", str(port), "--directory", "/tmp"]
    elif lang in ("node", "js", "javascript"):
        fname, cmd = "app.js", ["node", "/tmp/app.js"]
    elif lang in ("python", "py"):
        fname, cmd = "app.py", ["python3", "/tmp/app.py"]
    else:
        result["error"] = f"unsupported language for serving: {lang}"
        return result

    client = None
    container = None
    try:
        client = docker.DockerClient(base_url="unix:///var/run/docker.sock")
        container = client.containers.run(
            SANDBOX_IMAGE,
            command=["sleep", str(timeout + 60)],
            detach=True,
            tty=False,
            stdin_open=False,
            network_mode="bridge",
            ports={f"{port}/tcp": None},
            mem_limit="256m",
            pids_limit=128,
            user="sandbox",
            working_dir="/tmp",
            name=f"alpaca-serve-{uuid.uuid4().hex[:8]}",
            remove=False,
        )
        cleaned_code = extract_clean_code(code, lang)
        _put_file(container, f"/tmp/{fname}", cleaned_code.encode("utf-8"))
        # Launch the server in the background inside the running container.
        container.exec_run(cmd, detach=True)
        # Give the server a moment to bind, then read the published port.
        time.sleep(2)
        container.reload()
        port_info = (container.ports or {}).get(f"{port}/tcp")
        host_port = port_info[0]["HostPort"] if port_info else None
        result["container_id"] = container.id
        result["host_port"] = host_port
    except Exception as e:  # pragma: no cover - runtime dependent
        result["error"] = str(e)
    finally:
        with contextlib.suppress(Exception):
            if client is not None:
                client.close()
    return result


def stop_serve(container_id: str) -> dict[str, Any]:
    """Stop and remove a serving container started by ``serve_app``."""
    try:
        import docker

        client = docker.DockerClient(base_url="unix:///var/run/docker.sock")
        try:
            c = client.containers.get(container_id)
            c.remove(force=True)
        finally:
            client.close()
        return {"stopped": True}
    except Exception as e:  # pragma: no cover - runtime dependent
        return {"stopped": False, "error": str(e)}


# Serializes the whole serve_ui launch (sweep -> create -> setup -> return)
# so overlapping launches can never hand out a container another launch
# deleted (dead session in the browser, noVNC stuck at "connecting").
_UI_LAUNCH_LOCK = threading.Lock()


def serve_ui(
    code: str,
    lang: str = "python",
    timeout: int = 600,
    name: str = "alpaca-ui",
    exclusive: bool = False,
) -> dict[str, Any]:
    """Run ``code`` as a graphical (X11) app and stream it to the browser.

    Launches the code inside a sandbox container under a virtual X display
    (Xvfb), then serves the live display through x11vnc + websockify/noVNC so it
    can be viewed and interacted with in an HTML iframe. Returns ``container_id``,
    the published ``host_port`` (websockify), and a ``url`` (noVNC client page)
    suitable for embedding in an iframe.

    The container stays up (bridge network, published port) until stopped via
    ``stop_serve``. Only use this for code the user explicitly asked to view; it
    intentionally enables networking (the grading sandbox does not).

    ``name`` is the exact container name (default ``alpaca-ui``). Each launch
    fully serializes with other launches under a lock — sweep, create, app
    setup, and return — so an overlapping launch can never remove a session
    another in-flight launch already returned (that left the browser pointing
    at a dead container with noVNC stuck at "connecting"). ``exclusive``
    additionally sweeps same-prefix sessions (including pre-fix suffixed
    strays) so only one game session exists at a time.
    """
    result: dict[str, Any] = {"container_id": None, "host_port": None, "audio_host_port": None, "error": ""}
    try:
        import docker
    except Exception as e:  # pragma: no cover - environment dependent
        result["error"] = f"docker unavailable: {e}"
        return result

    if lang in ("python", "py"):
        fname = "code.py"
        http_game = False
        app_cmd, serve_mem = "python3 /tmp/code.py", "256m"
    elif lang in ("node", "js", "javascript"):
        fname = "app.js"
        http_game = True
        app_cmd, serve_mem = "", "256m"
    elif lang in ("go"):
        # Native X11 GUI: module-aware offline build, run under Xvfb.
        fname = "code.go"
        http_game = False
        app_cmd = _guarded_native_build(
            _go_build_sh("/tmp/code.go", "/tmp/app"), "/tmp/go_build.log", "/tmp/app", "/tmp/app"
        )
        serve_mem = "1g"
    elif lang in ("rust"):
        # Native X11 GUI: cargo project over baked crates, release binary.
        fname = "code.rs"
        http_game = False
        app_cmd = _guarded_native_build(
            _rust_build_sh("/tmp/code.rs", "release"),
            "/tmp/cargo_build.log",
            _rust_bin("release"),
            _rust_bin("release"),
        )
        serve_mem = "2g"
    elif lang in ("cpp", "c++", "cxx"):
        # Native X11 GUI: link X11/ALSA, run under Xvfb.
        fname = "code.cpp"
        http_game = False
        app_cmd = _guarded_native_build(
            _cpp_build_sh("/tmp/code.cpp", "/tmp/app", "/tmp/cpp_build.log"), "/tmp/cpp_build.log", "/tmp/app", "/tmp/app"
        )
        serve_mem = "256m"
    else:
        result["error"] = f"unsupported language for UI serving: {lang}"
        return result

    if http_game:
        # HTTP-rendered game: start HTTP server, then Chromium in Xvfb
        # renders the page visually so it can be streamed via noVNC.
        http_server = (
            "node /tmp/app.js"
            if lang in ("node", "js", "javascript")
            else "/tmp/app"
        )
        wrapper = (
            "#!/bin/bash\n"
            "Xvfb :99 -screen 0 1024x768x24 >/dev/null 2>&1 &\n"
            "XVFB_PID=$!\n"
            f"{_XVFB_WAIT}"
            "export DISPLAY=:99\n"
            f"{_AUDIO_SETUP}"
            "x11vnc -display :99 -rfbport 5900 -nopw -forever -shared >/dev/null 2>&1 &\n"
            "VNC_PID=$!\n"
            "websockify --web /usr/share/novnc 6080 127.0.0.1:5900 >/dev/null 2>&1 &\n"
            "WS_PID=$!\n"
            "sleep 1\n"
            f"{http_server} >/tmp/ui_stdout.txt 2>&1 &\n"
            "HTTP_PID=$!\n"
            "sleep 2\n"
            "chromium --no-sandbox --disable-gpu --window-size=1024x768 "
            "--start-fullscreen --no-first-run http://localhost:8080 "
            ">/dev/null 2>&1 &\n"
            "APP_PID=$!\n"
            "echo $APP_PID > /tmp/app.pid\n"
            "wait $APP_PID\n"
            "echo $? > /tmp/app.exitcode\n"
            "sleep infinity\n"
        )
    else:
        # Native X11 app (e.g. pygame, Xlib C++, xgb Go, x11rb Rust): compile
        # first (no-op for interpreted langs), then run directly under Xvfb.
        wrapper = (
            "#!/bin/bash\n"
            "Xvfb :99 -screen 0 1024x768x24 >/dev/null 2>&1 &\n"
            "XVFB_PID=$!\n"
            f"{_XVFB_WAIT}"
            "export DISPLAY=:99\n"
            f"{_AUDIO_SETUP}"
            "x11vnc -display :99 -rfbport 5900 -nopw -forever -shared >/dev/null 2>&1 &\n"
            "VNC_PID=$!\n"
            "websockify --web /usr/share/novnc 6080 127.0.0.1:5900 >/dev/null 2>&1 &\n"
            "WS_PID=$!\n"
            "sleep 1\n"
            f"{app_cmd} >/tmp/ui_stdout.txt 2>&1 &\n"
            "APP_PID=$!\n"
            "echo $APP_PID > /tmp/app.pid\n"
            "wait $APP_PID\n"
            "echo $? > /tmp/app.exitcode\n"
            "sleep infinity\n"
        )

    client = None
    container = None
    try:
        client = docker.DockerClient(base_url="unix:///var/run/docker.sock")
        # The whole launch holds the lock: an overlapping launch waits until
        # this session is fully set up and returned, so it can only replace a
        # live, already-returned session — never delete one mid-launch.
        with _UI_LAUNCH_LOCK:
            # A new launch replaces any leftover, so sessions never pile up.
            # The prefix sweep also clears pre-fix "<name>-<suffix>" strays.
            if exclusive:
                with contextlib.suppress(Exception):
                    for c in client.containers.list(all=True, filters={"name": name}):
                        if c.name == name or c.name.startswith(name + "-"):
                            with contextlib.suppress(Exception):
                                c.remove(force=True)
            else:
                with contextlib.suppress(Exception):
                    client.containers.get(name).remove(force=True)
            container = client.containers.run(
                SANDBOX_IMAGE,
                command=["sleep", str(timeout + 60)],
                detach=True,
                tty=False,
                stdin_open=False,
                network_mode="bridge",
                ports={"6080/tcp": None, f"{_AUDIO_PORT}/tcp": None},
                mem_limit=serve_mem,
                pids_limit=128,
                user="sandbox",
                working_dir="/tmp",
                name=name,
                remove=False,
            )
            cleaned_code = extract_clean_code(code, lang)
            _put_file(container, f"/tmp/{fname}", cleaned_code.encode("utf-8"))
            _put_file(container, "/tmp/run_ui.sh", wrapper.encode("utf-8"))
            # Launch the Xvfb + VNC + app pipeline in the background.
            container.exec_run(["/bin/bash", "/tmp/run_ui.sh"], detach=True)
            # Give the services a moment to bind, then read the published port.
            time.sleep(3)
            container.reload()
            port_info = (container.ports or {}).get("6080/tcp")
            host_port = port_info[0]["HostPort"] if port_info else None
            audio_info = (container.ports or {}).get(f"{_AUDIO_PORT}/tcp")
            audio_host_port = audio_info[0]["HostPort"] if audio_info else None
            result["container_id"] = container.id
            result["host_port"] = host_port
            result["audio_host_port"] = audio_host_port
    except Exception as e:  # pragma: no cover - runtime dependent
        result["error"] = str(e)
    finally:
        with contextlib.suppress(Exception):
            if client is not None:
                client.close()
    return result


def _ui_container(container_id: str):
    """Connect to the docker socket and fetch a running UI container (or None)."""
    try:
        import docker

        client = docker.DockerClient(base_url="unix:///var/run/docker.sock")
        try:
            return client, client.containers.get(container_id)
        except Exception:
            client.close()
            return None, None
    except Exception:  # pragma: no cover - environment dependent
        return None, None


def ui_exec(container_id: str, command: str, timeout: int = 15) -> dict[str, Any]:
    """Run an arbitrary shell command inside the UI container for troubleshooting."""
    result: dict[str, Any] = {"output": "", "exit_code": None, "error": ""}
    client, container = _ui_container(container_id)
    if container is None:
        result["error"] = f"container {container_id} not found"
        return result
    try:
        code, out = container.exec_run(
            ["/bin/bash", "-c", command],
            user="sandbox",
            workdir="/tmp",
            environment={"DISPLAY": ":99"},
            demux=True,
        )
        stdout = (
            (out[0] or b"").decode("utf-8", errors="replace")
            if isinstance(out, tuple)
            else (out or b"").decode("utf-8", errors="replace")
        )
        stderr = (out[1] or b"").decode("utf-8", errors="replace") if isinstance(out, tuple) else ""
        result["output"] = stdout + (("\n[stderr]\n" + stderr) if stderr else "")
        result["exit_code"] = code
    except Exception as e:  # pragma: no cover - runtime dependent
        result["error"] = str(e)
    finally:
        with contextlib.suppress(Exception):
            if client is not None:
                client.close()
    return result


def ui_status(container_id: str) -> dict[str, Any]:
    """Report the UI container + app runtime state (pid, exit code, stdout tail)."""
    result: dict[str, Any] = {
        "running": False,
        "app_pid": None,
        "app_exitcode": None,
        "stdout_tail": "",
        "host_port": None,
        "error": "",
    }
    client, container = _ui_container(container_id)
    if container is None:
        result["error"] = f"container {container_id} not found"
        return result
    try:
        container.reload()
        result["running"] = container.status == "running"
        port_info = (container.ports or {}).get("6080/tcp")
        result["host_port"] = port_info[0]["HostPort"] if port_info else None
        app_pid = (
            container.exec_run(["cat", "/tmp/app.pid"], user="sandbox").output.decode("utf-8", errors="replace").strip()
        )
        result["app_pid"] = app_pid if app_pid.isdigit() else None
        exitcode = (
            container.exec_run(["cat", "/tmp/app.exitcode"], user="sandbox")
            .output.decode("utf-8", errors="replace")
            .strip()
        )
        result["app_exitcode"] = int(exitcode) if exitcode.lstrip("-").isdigit() else None
        tail = container.exec_run(
            ["bash", "-c", "tail -c 4000 /tmp/ui_stdout.txt 2>/dev/null || echo '(no stdout yet)'"],
            user="sandbox",
        ).output.decode("utf-8", errors="replace")
        result["stdout_tail"] = tail
    except Exception as e:  # pragma: no cover - runtime dependent
        result["error"] = str(e)
    finally:
        with contextlib.suppress(Exception):
            if client is not None:
                client.close()
    return result


def ui_screenshot(container_id: str) -> dict[str, Any]:
    """Capture the current Xvfb :99 framebuffer and return it as a PNG (base64)."""
    result: dict[str, Any] = {"image": None, "error": ""}
    client, container = _ui_container(container_id)
    if container is None:
        result["error"] = f"container {container_id} not found"
        return result
    try:
        _, out = container.exec_run(
            [
                "/bin/bash",
                "-c",
                "DISPLAY=:99 scrot -o /tmp/ui_shot.png 2>/dev/null && base64 -w0 /tmp/ui_shot.png || echo SCROT_FAIL",
            ],
            user="sandbox",
        )
        data = (out or b"").decode("utf-8", errors="replace").strip()
        if data == "SCROT_FAIL" or not data:
            result["error"] = "screenshot capture failed"
        else:
            result["image"] = data
    except Exception as e:  # pragma: no cover - runtime dependent
        result["error"] = str(e)
    finally:
        with contextlib.suppress(Exception):
            if client is not None:
                client.close()
    return result


def ui_restart(container_id: str) -> dict[str, Any]:
    """Kill the running app and relaunch ``/tmp/code.py`` (or app.js) on the same X display."""
    result: dict[str, Any] = {"restarted": False, "error": ""}
    client, container = _ui_container(container_id)
    if container is None:
        result["error"] = f"container {container_id} not found"
        return result
    try:
        lang_check = container.exec_run(["ls", "/tmp/app.js"], user="sandbox").exit_code
        fname, bin_ = ("app.js", "node") if lang_check == 0 else ("code.py", "python3")
        cmd = (
            "if [ -f /tmp/app.pid ]; then kill -9 $(cat /tmp/app.pid) 2>/dev/null; fi; "
            "rm -f /tmp/app.exitcode; "
            f"{bin_} /tmp/{fname} >/tmp/ui_stdout.txt 2>&1 & "
            "echo $! > /tmp/app.pid; echo STARTED"
        )
        code, out = container.exec_run(
            ["/bin/bash", "-c", cmd],
            user="sandbox",
            environment={"DISPLAY": ":99"},
        )
        result["restarted"] = code == 0 and b"STARTED" in (out or b"")
        if not result["restarted"]:
            result["error"] = (out or b"").decode("utf-8", errors="replace")
    except Exception as e:  # pragma: no cover - runtime dependent
        result["error"] = str(e)
    finally:
        with contextlib.suppress(Exception):
            if client is not None:
                client.close()
    return result
