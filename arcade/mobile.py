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
    if "</head>" in output.lower():
        output = re.sub(r"</head\s*>", FIT_CSS + "\n</head>", output, count=1, flags=re.IGNORECASE)
    else:
        output = FIT_CSS + output
    return output
