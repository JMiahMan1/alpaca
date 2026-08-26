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
        cmd = ["bash", "-c", "g++ -std=c++17 -O2 -o /tmp/a.out /tmp/code.cpp && /tmp/a.out"]
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
    try:
        client = docker.DockerClient(base_url="unix:///var/run/docker.sock")
        container = client.containers.run(
            SANDBOX_IMAGE,
            command=["sleep", "300"],
            detach=True,
            tty=False,
            stdin_open=True,
            network_mode="none",
            mem_limit="256m",
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
            return _run_ui(container, cleaned_code, ext, bin_, timeout, result)

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


def _run_ui(container, code: str, ext: str, bin_: str, timeout: int, result: dict[str, Any]) -> dict[str, Any]:
    """Launch ``code`` under Xvfb and capture a screenshot of the rendered UI."""
    capture_delay = max(2, min(timeout - 3, 8))
    wrapper = (
        "#!/bin/bash\n"
        "Xvfb :99 -screen 0 1024x768x24 >/dev/null 2>&1 &\n"
        "XVFB_PID=$!\n"
        "sleep 1\n"
        "export DISPLAY=:99\n"
        f"{bin_} /tmp/code.{ext} >/tmp/ui_stdout.txt 2>&1 &\n"
        "PY_PID=$!\n"
        f"sleep {capture_delay}\n"
        "scrot -o /tmp/out.png 2>/dev/null\n"
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


def serve_ui(
    code: str,
    lang: str = "python",
    timeout: int = 600,
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
    """
    result: dict[str, Any] = {"container_id": None, "host_port": None, "error": ""}
    try:
        import docker
    except Exception as e:  # pragma: no cover - environment dependent
        result["error"] = f"docker unavailable: {e}"
        return result

    if lang in ("python", "py"):
        fname, bin_ = "code.py", "python3"
    elif lang in ("node", "js", "javascript"):
        fname, bin_ = "app.js", "node"
    else:
        result["error"] = f"unsupported language for UI serving: {lang}"
        return result

    wrapper = (
        "#!/bin/bash\n"
        "Xvfb :99 -screen 0 1024x768x24 >/dev/null 2>&1 &\n"
        "XVFB_PID=$!\n"
        "sleep 1\n"
        "export DISPLAY=:99\n"
        "x11vnc -display :99 -rfbport 5900 -nopw -forever -shared >/dev/null 2>&1 &\n"
        "VNC_PID=$!\n"
        "websockify --web /usr/share/novnc 6080 127.0.0.1:5900 >/dev/null 2>&1 &\n"
        "WS_PID=$!\n"
        "sleep 1\n"
        f"{bin_} /tmp/{fname} >/tmp/ui_stdout.txt 2>&1 &\n"
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
        container = client.containers.run(
            SANDBOX_IMAGE,
            command=["sleep", str(timeout + 60)],
            detach=True,
            tty=False,
            stdin_open=False,
            network_mode="bridge",
            ports={"6080/tcp": None},
            mem_limit="256m",
            pids_limit=128,
            user="sandbox",
            working_dir="/tmp",
            name=f"alpaca-ui-{uuid.uuid4().hex[:8]}",
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
        result["container_id"] = container.id
        result["host_port"] = host_port
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
