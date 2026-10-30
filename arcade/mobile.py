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
