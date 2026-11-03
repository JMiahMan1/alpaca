"""Mobile readiness for benchmark game pages.

Benchmark models write standalone HTML, and many draw at a fixed 800x600.
The arcade can still show those on a phone if the page is allowed to scale:
give it a viewport, keep the body from scrolling or rubber-banding, and cap
canvases and fixed-width wrappers to the screen. Existing responsive games
already set their own width and are left alone.

The score is deliberately simple and explainable. It is used by the OODA
refresh to prefer games that were designed for a small screen, not as a
hard gate: every playable game is still served through the fit CSS.
"""

from __future__ import annotations

import re

VIEWPORT_RE = re.compile(r"<meta[^>]+name=[\"']viewport[\"']", re.IGNORECASE)
HEAD_RE = re.compile(r"</head\s*>", re.IGNORECASE)
HTML_RE = re.compile(r"<!doctype|<html", re.IGNORECASE)

# Only constrain intrinsic sizes. A responsive game that already says
# width:100% keeps working; a fixed 800x600 canvas is scaled down by the
# max-width/max-height pair while preserving its aspect ratio.
FIT_CSS = """
<style id="arcade-mobile-fit">
  html, body {
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    max-width: 100vw;
    max-height: 100vh;
    overflow: hidden;
    overscroll-behavior: none;
    background: #000;
    touch-action: none;
  }
  body { display: flex; align-items: center; justify-content: center; }
  canvas {
    max-width: 100vw !important;
    max-height: 100vh !important;
    touch-action: none;
  }
  #arcade, #game, #game-container, .game-container, .canvas-wrap, .game-wrap {
    max-width: 100vw !important;
    max-height: 100vh !important;
  }
</style>
"""

# A page that already lays itself out (fluid width, its own scrolling) must NOT
# get the block above: pinning html/body to 100% height with overflow:hidden and
# touch-action:none centres the page, clips anything past one screen and stops it
# scrolling. A card game with a board and a sidebar would be unreachable on a
# phone. Those pages only need their canvases capped.
SOFT_CSS = """
<style id="arcade-mobile-fit">
  canvas {
    max-width: 100vw !important;
    max-height: 100vh !important;
  }
  #arcade, #game, #game-container, .game-container, .canvas-wrap, .game-wrap {
    max-width: 100vw !important;
    max-height: 100vh !important;
  }
</style>
"""

CANVAS_RE = re.compile(r"<canvas", re.IGNORECASE)


def needs_full_fit(html: str) -> bool:
    """Whether a page needs the whole-screen fit, or just its canvas capped.

    A page that draws to a canvas is scaled by the fit. A page that already lays
    itself out - fluid width, its own media queries - is left to scroll: forcing
    the fit on it hides the lower half of the board behind a pinned body.
    """
    if CANVAS_RE.search(html or ""):
        return True
    return "fluid-width" not in (assess_html(html).get("reasons") or [])


def assess_html(html: str) -> dict:
    """Return an explainable 0-100 mobile readiness score for a game page."""
    text = html or ""
    score = 0
    reasons: list[str] = []

    if VIEWPORT_RE.search(text):
        score += 30
        reasons.append("viewport")
    if re.search(r"width\s*:\s*(?:100%|100vw|min\()", text, re.IGNORECASE):
        score += 25
        reasons.append("fluid-width")
    if re.search(r"@media\s*\(", text, re.IGNORECASE):
        score += 15
        reasons.append("media-query")
    if re.search(r"(?:touchstart|pointerdown|touch-action|ontouchstart)", text, re.IGNORECASE):
        score += 15
        reasons.append("touch")
    if re.search(r"max-width\s*:\s*100(?:%|vw)", text, re.IGNORECASE):
        score += 10
        reasons.append("max-width")
    if re.search(r"<canvas[^>]+width=[\"']?(?:[7-9]\d\d|1\d{3,})", text, re.IGNORECASE):
        score -= 15
        reasons.append("fixed-canvas")
    score = max(0, min(100, score))
    return {"score": score, "ready": score >= 40, "reasons": reasons}


def fit_css_for(html: str) -> str:
    """The fit CSS a page needs.

    A page that draws at a fixed size (a canvas, or no fluid width of its own)
    gets the full screen-fitting block. A page that already lays itself out gets
    only the canvas caps - see SOFT_CSS.
    """
    return FIT_CSS if needs_full_fit(html) else SOFT_CSS


def prepare_playable_html(html: str) -> str:
    """Make a standalone game page fit a phone without rewriting its game."""
    if not html:
        return html
    output = html
    if not VIEWPORT_RE.search(output):
        viewport = '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
        if HEAD_RE.search(output):
            output = HEAD_RE.sub(viewport + "\n</head>", output, count=1)
        elif HTML_RE.search(output):
            output = re.sub(r"(<html[^>]*>)", r"\1<head>" + viewport + "</head>", output, count=1, flags=re.IGNORECASE)
        else:
            output = viewport + output
    fit = fit_css_for(output)
    if "</head>" in output.lower():
        output = re.sub(r"</head\s*>", fit + "\n</head>", output, count=1, flags=re.IGNORECASE)
    else:
        output = fit + output
    return output


# The Family Games mobile contract. A game is "mobile compatible" - and may
# join or be published in the Family Games collection - only when every key
# is true. The ``mobilegames`` benchmark grades answers with the same report,
# so a passing benchmark game is exactly a game the family shelf accepts.
CONTRACT_LABELS = {
    "viewport": "a width=device-width viewport",
    "fills_screen": "sizing from the screen and on resize",
    "no_fixed_canvas": "no fixed 600px+ canvas",
    "hidpi": "devicePixelRatio scaling",
    "touch_input": "touch/pointer controls",
    "no_page_gestures": "touch-action and preventDefault",
    "safe_area": "safe-area insets",
    "pauses_when_hidden": "pause on visibilitychange",
    "animation_loop": "a requestAnimationFrame loop",
    "personal_best": "a localStorage personal best",
    "score_bridge": "the arcade score bridge",
    "no_name_prompt": "no name/initials prompt",
}


def contract_report(html: str) -> dict[str, bool]:
    """Which parts of the mobile contract a game page meets (one key each)."""
    text = (html or "").lower()
    return {
        "viewport": bool(re.search(r"<meta[^>]+name=[\"']viewport[\"'][^>]*width=device-width", text)),
        "fills_screen": any(x in text for x in ("innerwidth", "clientwidth", "getboundingclientrect"))
        and bool(re.search(r"addeventlistener\(\s*[\"']resize|onresize|resizeobserver", text)),
        "no_fixed_canvas": not re.search(r"<canvas[^>]*\bwidth\s*=\s*[\"']?(?:[6-9]\d\d|1\d{3})", text),
        "hidpi": "devicepixelratio" in text,
        "touch_input": any(x in text for x in ("pointerdown", "touchstart")),
        "no_page_gestures": "touch-action" in text and "preventdefault" in text,
        "safe_area": "safe-area-inset" in text,
        "pauses_when_hidden": "visibilitychange" in text,
        "animation_loop": "requestanimationframe" in text,
        "personal_best": "localstorage" in text,
        "score_bridge": "postmessage" in text
        and "arcade-game" in text
        and bool(re.search(r"type\s*:\s*[\"']score[\"']", text))
        and "run_id" in text,
        "no_name_prompt": "prompt(" not in text,
    }


def contract_failures(html: str) -> list[str]:
    """Human-readable names of the contract items a page misses (empty = compatible)."""
    return [CONTRACT_LABELS[k] for k, ok in contract_report(html).items() if not ok]


def is_mobile_compatible(html: str) -> bool:
    return not contract_failures(html)
