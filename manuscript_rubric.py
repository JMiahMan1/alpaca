"""Rubric for creative_illuminated_manuscript_john1.

A late-medieval illuminated manuscript is not "John 1 with a border". Three of
its defining features are machine-checkable, and all three are things a model
reliably forgets:

  1. the text is *complete* - all 18 verses, unaltered;
  2. the decoration is *drawn*, not described - real vector work, because the
     repository ships zero font files and every stable-diffusion preset already
     bans "garbled text, distorted letters, bad typography", so a generated
     capital would be both a font dependency and a diffusion failure mode;
  3. the page is a *page* - a bounded leaf with a frame, which is what makes the
     page-flip affordance meaningful rather than decorative.

Kept out of llm_benchmark_suite.py so the text, the rules and their tests can
evolve together. The suite imports :func:`manuscript_rubric` from here.
"""

from __future__ import annotations

import html as html_module
import re

#: King James Version, John chapter 1, all 18 verses, unabridged. Kept as data
#: rather than fetched so a network hiccup cannot change what the test grades,
#: and so the test and the grader can never disagree about the source text.
#:
#: Verse 14 keeps its parentheses. The audio pipeline rewrites them to commas
#: before narration (``tts_text._paren_to_commas``), which is the desired
#: behaviour for speech and is exactly why they can stay here.
JOHN_1_KJV: tuple[str, ...] = (
    "In the beginning was the Word, and the Word was with God, and the Word was God.",
    "The same was in the beginning with God.",
    "All things were made by him; and without him not any thing was made that was made.",
    "In him was life; and the life was the light of men.",
    "And the light shineth in darkness; and the darkness hath not overcome it.",
    "There was a man sent from God, whose name was John.",
    "The same came to witness unto the light, that all men through him might believe.",
    "He was not that light, but was sent to witness unto the light.",
    "The light was the true light, which lighteth every man that cometh into the world.",
    "He was in the world, and the world was made by him, and the world knew him not.",
    "He came unto his own, and his own received him not.",
    "But as many as received him, to them gave he power to become the sons of God, even to them that believe on his name:",
    "Which were born, not of blood, nor of the will of the flesh, nor of the will of man, but of God.",
    "And the Word was made flesh, and dwelt among us, (and we beheld his glory, the glory as of the onlybegotten of the Father,) full of grace and truth.",
    "John bare witness of him, and cried, saying, Behold, the Lamb of God, which taketh away the sin of the world.",
    "And of his fulness have we all received, and grace for grace.",
    "For the law was given by Moses; but grace and truth came by Jesus Christ.",
    "No man hath seen God at any time; the only begotten Son, which is in the bosom of the Father, he hath declared him.",
)

#: Elements that can actually put a mark on the page. A <g> does not count - an
#: empty group is not decoration.
_SVG_SHAPE = re.compile(
    r"<\s*(path|circle|ellipse|rect|line|polyline|polygon|text)\b[^>]*?(?:/>|>)",
    re.IGNORECASE,
)
_SVG_ROOT = re.compile(r"<\s*svg\b[^>]*>", re.IGNORECASE)
_SVG_CLOSE = re.compile(r"<\s*/\s*svg\s*>", re.IGNORECASE)

#: Markers of a turnable leaf. Deliberately generous: there are several honest
#: ways to build a page flip (3D transform, stacked leaves, a scroll-snap
#: carousel), and the point is to catch a model that shipped a single static
#: image, not to mandate one technique.
_FLIP_MARKERS = (
    r"perspective",
    r"rotate[XY]?\s*\(",
    r"rotate[XY]?deg",
    r"transform-style\s*:\s*preserve-3d",
    r"backface-visibility",
    r"\bflip",
    r"\bturn\b",
    r"scroll-snap",
    r"nextPage",
    r"prevPage",
    r"pageTurn",
)

#: A page container that is more than the <body>: it needs its own box, because
#: "a bounded leaf with a frame" is what the flip affordance animates. The frame
#: has to be declared on the *container's own* rule - a border anywhere in the
#: stylesheet (on <body>, say) is a page background, not a leaf.
_PAGE_FRAME = re.compile(
    r"(?:border(?:-image|-width)?\s*:|outline\s*:|box-shadow\s*:|background(?:-image)?\s*:)",
    re.IGNORECASE,
)
_PAGE_CONTAINER_TAGS = ("section", "article", "div", "main", "figure", "aside")
_PAGE_NAME = re.compile(
    r"\b(page|leaf|folio|spread|codex|manuscript|book|parchment|vellum)\w*", re.IGNORECASE
)

#: Verse numbers are rendered as superscripts, in the margin, or omitted
#: entirely depending on the design, so they are stripped on both sides.
_LEADING_NUMBER = re.compile(r"^\s*\d{1,2}\s*")

_QUOTES = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "′": "'",
    "″": '"',
}
_DASHES = {
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "―": "-",
    "−": "-",
    "­": "",
}


def normalise_prose(text: str) -> str:
    """Fold a string to the form two spellings of the same English can share.

    Smart quotes and dashes are folded, entities are decoded, tags are removed
    and every run of whitespace collapses to one space, so a verse split across
    a drop cap, a <sup> number and three wrapped lines still matches.
    """
    out = html_module.unescape(text)
    out = re.sub(r"<(script|style)\b.*?</\1>", " ", out, flags=re.IGNORECASE | re.DOTALL)
    out = re.sub(r"<[^>]+>", " ", out)
    for src, dst in _QUOTES.items():
        out = out.replace(src, dst)
    for src, dst in _DASHES.items():
        out = out.replace(src, dst)
    out = out.replace(" ", " ")
    return re.sub(r"\s+", " ", out).strip().lower()


def verses_present(document_html: str) -> list[int]:
    """Return the 1-based verse numbers found in the rendered text of a page."""
    flat = normalise_prose(document_html)
    found: list[int] = []
    for index, verse in enumerate(JOHN_1_KJV, start=1):
        needle = normalise_prose(_LEADING_NUMBER.sub("", verse))
        if needle and needle in flat:
            found.append(index)
    return found


def _svg_blocks(document_html: str) -> list[str]:
    """Every complete <svg>...</svg> block, self-closing or not.

    Unterminated blocks are dropped so a response truncated by the token budget
    cannot half-satisfy the decoration check.
    """
    blocks: list[str] = []
    for match in _SVG_ROOT.finditer(document_html):
        close = _SVG_CLOSE.search(document_html, match.end())
        if close:
            blocks.append(document_html[match.start() : close.end()])
    return blocks


def _has_drawn_svg(document_html: str) -> bool:
    """True when at least one <svg> has a usable viewBox and real drawn content.

    A zero viewBox renders nothing, and a viewBox that is not four numbers is
    not a viewBox at all - both would pass a bare `<svg>` presence check.
    """
    for block in _svg_blocks(document_html):
        root = _SVG_ROOT.search(block)
        view_box = None
        if root:
            found = re.search(r'\bviewBox\s*=\s*["\']([^"\']*)["\']', root.group(0), re.IGNORECASE)
            if found:
                parts = re.split(r"[\s,]+", found.group(1).strip())
                if len(parts) == 4:
                    try:
                        view_box = [float(p) for p in parts]
                    except ValueError:
                        view_box = None
        if not view_box or view_box[2] <= 0 or view_box[3] <= 0:
            continue
        body = block[root.end() :] if root else block
        if _SVG_SHAPE.search(body):
            return True
    return False


def _strip_comments(document_html: str) -> str:
    """Remove CSS and HTML comments.

    Without this, a model that *describes* the affordance it was asked for -
    "/* rotateY(180deg) on click */" - satisfies a keyword search. A comment is
    a claim about the page, not the page.
    """
    without_css = re.sub(r"/\*.*?\*/", " ", document_html, flags=re.DOTALL)
    return re.sub(r"<!--.*?-->", " ", without_css, flags=re.DOTALL)


def _has_flip_affordance(document_html: str) -> bool:
    source = _strip_comments(document_html)
    if not any(re.search(pattern, source, re.IGNORECASE) for pattern in _FLIP_MARKERS):
        return False
    # A marker in prose is not an affordance. Require machinery behind it. A
    # CSS-only flip is a legitimate implementation (:hover / :checked / a class
    # toggle), so a transition or keyframe counts; so does a handler for the
    # JS-driven case.
    mechanism = re.search(
        r"(?:transform\s*:|perspective\s*:|@keyframes|transition\s*:|addEventListener"
        r"|onclick\s*=|\.onclick|querySelector)",
        source,
        re.IGNORECASE,
    )
    return bool(mechanism)


def _css_rule_for(class_name: str, document_html: str) -> str:
    """The body of the CSS rules that declare properties *on* ``class_name``.

    Only rules whose **subject** names the class count. A real manuscript
    stylesheet writes ``.page { border: … }`` next to ``.page .ornament { border:
    … }``, and a border on the ornament does not frame the page - it decorates
    the child. Matching the class anywhere in the selector credits the child's
    frame to its parent, which is exactly the pass this check must not give.

    Selectors are otherwise read loosely (``.page``, ``.page:hover``,
    ``.leaf > .page``, ``.page, .folio``), because the goal is to find the
    declarations that land on the element, not to validate the selector.
    """
    escaped = re.escape(class_name)
    rule = re.compile(r"([^{}]*)\{", re.MULTILINE)
    bodies: list[str] = []
    for match in rule.finditer(document_html):
        selectors = match.group(1)
        # The last compound of each comma-separated selector is the subject.
        subject_matches = False
        for selector in selectors.split(","):
            compounds = [c for c in re.split(r"[\s>+~]+", selector.strip()) if c]
            if not compounds:
                continue
            subject = compounds[-1]
            if re.search(rf"(?<![\w-]){escaped}(?![\w-])", subject, re.IGNORECASE):
                subject_matches = True
                break
        if not subject_matches:
            continue
        depth = 0
        start = match.end()
        for index in range(start, len(document_html)):
            char = document_html[index]
            if char == "{":
                depth += 1
            elif char == "}":
                if depth == 0:
                    bodies.append(document_html[start:index])
                    break
                depth -= 1
    return " ".join(bodies)


def _has_page_frame(document_html: str) -> bool:
    """True when a distinct page element declares its own frame.

    Without this the flip marker is just a word in a stylesheet: there is no
    leaf to animate, which is the difference between a manuscript and a
    screenshot of one. The frame must be on the container's own rule, so a
    background on <body> does not pass for a framed leaf.
    """
    source = _strip_comments(document_html)
    for tag in re.finditer(rf"<\s*({'|'.join(_PAGE_CONTAINER_TAGS)})\b[^>]*>", source, re.IGNORECASE):
        attributes = tag.group(0)
        class_match = re.search(r'\bclass\s*=\s*["\']([^"\']*)["\']', attributes, re.IGNORECASE)
        candidates = class_match.group(1).split() if class_match else []
        candidates += re.findall(r'\bid\s*=\s*["\']([^"\']*)["\']', attributes, re.IGNORECASE)
        for name in candidates:
            if not name or not _PAGE_NAME.search(name):
                continue
            if _PAGE_FRAME.search(_css_rule_for(name, source)):
                return True
        # An inline style is the same claim in a different place.
        inline = re.search(r'\bstyle\s*=\s*["\']([^"\']*)["\']', attributes, re.IGNORECASE)
        if (
            inline
            and _PAGE_FRAME.search(inline.group(1))
            and any(_PAGE_NAME.search(name or "") for name in candidates)
        ):
            return True
    return False


def is_complete_document(document_html: str) -> bool:
    """True when the answer is a whole document, not a fragment.

    ``arcade_publish`` applies the same rule: a fragment is silently downgraded
    to a frozen ``game.py`` instead of being served as a page, so an answer that
    does not satisfy it can never be played either.
    """
    return bool(
        re.search(r"<!doctype\s+html", document_html, re.IGNORECASE)
        and re.search(r"<\s*html\b", document_html, re.IGNORECASE)
        and re.search(r"<\s*/\s*html\s*>", document_html, re.IGNORECASE)
        and re.search(r"<\s*body\b", document_html, re.IGNORECASE)
    )


def manuscript_rubric(document_html: str) -> dict:
    """Score a candidate manuscript out of the four properties that matter.

    Returns a dict with a boolean per criterion, the verse numbers found, and
    ``score`` as the percentage of criteria met. The score is reported alongside
    the benchmark score so a near-miss reads as a near-miss rather than as zero.
    """
    html_source = document_html or ""
    present = verses_present(html_source)
    criteria = {
        "complete_document": is_complete_document(html_source),
        "all_verses_present": len(present) == len(JOHN_1_KJV),
        "decorative_svg_drawn": _has_drawn_svg(html_source),
        "framed_page": _has_page_frame(html_source),
        "page_flip_affordance": _has_flip_affordance(html_source),
    }
    met = sum(1 for passed in criteria.values() if passed)
    return {
        **criteria,
        "verses_found": present,
        "verses_missing": [n for n in range(1, len(JOHN_1_KJV) + 1) if n not in present],
        "verses_total": len(JOHN_1_KJV),
        "criteria_met": met,
        "criteria_total": len(criteria),
        "score": round(100 * met / len(criteria)),
        "passed": all(criteria.values()),
    }


def grade_illuminated_manuscript(document_html: str) -> tuple[bool, str]:
    """Gate a candidate manuscript, returning ``(passed, reason_when_failed)``.

    The reason names the specific missing property and, for the text, the verse
    numbers, so a failed 16k-token run can be diagnosed without re-reading the
    model output.
    """
    rubric = manuscript_rubric(document_html)
    if rubric["passed"]:
        return True, ""
    reasons: list[str] = []
    if not rubric["complete_document"]:
        reasons.append("not a complete <!doctype html> document with <body> and </html>")
    if not rubric["all_verses_present"]:
        missing = rubric["verses_missing"]
        shown = ", ".join(str(n) for n in missing[:6]) + ("..." if len(missing) > 6 else "")
        reasons.append(
            f"missing John 1 verse(s) {shown} ({(rubric['verses_found'] and len(rubric['verses_found'])) or 0}"
            f"/{rubric['verses_total']} verses present)"
        )
    if not rubric["decorative_svg_drawn"]:
        reasons.append("no inline <svg> with a positive viewBox and drawn shapes")
    if not rubric["framed_page"]:
        reasons.append("no framed page container (a class naming the page/leaf plus a border or background)")
    if not rubric["page_flip_affordance"]:
        reasons.append("no page-flip affordance (perspective/rotate transform plus a handler)")
    return False, "; ".join(reasons)
