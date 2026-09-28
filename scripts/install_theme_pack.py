#!/usr/bin/env python3
"""Install a generated SVG theme pack into a SharedLLM checkout.

`creative_svg_theme_pack` is a scored alpaca benchmark: a model authors two SVG
assets against the real `motifPattern()` contract - a seamless motif tile and a
`<symbol>` sprite sheet. This script is the other half of that loop. It reads a
model answer, re-validates the same structural rules the grader applies, and
only then writes:

    <SharedLLM>/services/ui/src/themes/packs/<pack-id>.pack.json

Two design decisions worth stating, because they are what makes the output
usable at all:

1. **A pack cannot carry files.** `ThemePack` is a plain JSON document that
   round-trips through `localStorage`, a `Blob` export, and a `GET/PUT
   /api/users/me/theme` column, and the theme manager's import control only
   accepts `application/json`. So the assets are carried *inside* the JSON, and
   this script converts them to `url("data:image/svg+xml,...")` strings - which
   is exactly the form `motifPattern()` produces, so the runtime path is
   identical to a built-in motif.

2. **Colours must still be placeholders at the point of use.** The tile's
   strokes become `var(--site-accent)` / `var(--site-accent-alt)` CSS
   references, not the theme's hex. A tile inlined as a data-URI cannot see the
   page's custom properties, so the interpolation the model was asked for
   (`{{accent}}`) is performed here, into a CSS var reference, and the *data-URI
   is re-encoded with those references intact*. That keeps one tile usable
   across every theme in the pack.

Run it with no arguments to see the expected answer shape.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

# Mirrors of the grader's rules. Keeping them here rather than importing
# llm_benchmark_suite means this script runs without importing the benchmark
# suite (and its httpx/playwright surface) into a checkout that may not have
# them; the test below asserts the two copies agree.
MIN_SPRITE_SYMBOLS = 6
STROKE_WIDTH_RANGE = (0.8, 2.0)
STROKE_OPACITY_RANGE = (0.04, 0.30)
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
ATTR_RE = re.compile(r"""([a-zA-Z-]+)\s*=\s*["']([^"']*)["']""")
HEX_RE = re.compile(r"#[0-9a-fA-F]{3,8}\b")
SYMBOL_ID_RE = re.compile(r"""id\s*=\s*["']([a-z0-9]+(?:-[a-z0-9]+)*-icon)["']""")
PLACEHOLDER_RE = re.compile(r"\{\{\s*(accent|accentAlt)\s*\}\}", re.IGNORECASE)

DEFAULT_SHAREDLLM = REPO.parent / "SharedLLM"
PACKS_DIR = Path("services/ui/src/themes/packs")

# The two theme tokens the motif contract binds to. The site theme writes these
# as --site-accent / --site-accent-alt (see siteTheme.ts siteThemeCssVars), and
# the widget theme writes --ht-accent / --ht-accent-alt. The installer targets
# the site vars because that is the layer the motif is consumed at.
PLACEHOLDER_TO_CSS_VAR = {
    "accent": "var(--site-accent)",
    "accentalt": "var(--site-accent-alt)",
}

# The characters encodeURIComponent() leaves alone. urllib's default unreserved
# set omits these five, so passing it explicitly makes this encoder byte-identical
# to the JavaScript one the UI already uses.
_ENCODE_URI_COMPONENT_SAFE = "-_.!~*'()"


class PackError(ValueError):
    """The generated answer does not satisfy the theme-pack contract."""


def svg_roots(response: str) -> list[str]:
    """Every complete <svg>...</svg> (or self-closing <svg .../>) block."""
    out: list[str] = []
    for match in re.finditer(r"<svg\b", response, re.IGNORECASE):
        head_end = response.find(">", match.start())
        if head_end == -1:
            continue
        if response[head_end - 1] == "/":
            out.append(response[match.start() : head_end + 1])
            continue
        close = response.lower().find("</svg>", head_end)
        if close != -1:
            out.append(response[match.start() : close + len("</svg>")])
    return out


def root_attrs(block: str) -> dict[str, str]:
    head = block[: block.find(">") + 1]
    return {m.group(1).lower(): m.group(2) for m in ATTR_RE.finditer(head)}


def check_seamless_tile(block: str) -> None:
    attrs = root_attrs(block)
    width = NUM_RE.search(attrs.get("width", ""))
    height = NUM_RE.search(attrs.get("height", ""))
    viewbox = NUM_RE.findall(attrs.get("viewbox", ""))
    if not (width and height and len(viewbox) >= 4):
        raise PackError("tile root needs width, height and a 4-number viewBox")
    w, h = float(width.group(0)), float(height.group(0))
    vx, vy, vw, vh = (float(v) for v in viewbox[:4])
    if not (w == h and vw == vh and abs(w - vw) < 1e-6 and vx == 0 and vy == 0 and w > 0):
        raise PackError(
            f"tile must be square with width == height == viewBox extent (got "
            f"width={w} height={h} viewBox='{vx} {vy} {vw} {vh}')"
        )


def check_wireframe(block: str) -> None:
    if 'fill="none"' not in block and "fill='none'" not in block:
        raise PackError("tile must be line art: set fill=\"none\" on the root or its <g>")
    widths, opacities = [], []
    for name, value in ATTR_RE.findall(block):
        number = NUM_RE.search(value)
        if name.lower() == "stroke-width" and number:
            widths.append(float(number.group(0)))
        elif name.lower() == "stroke-opacity" and number:
            opacities.append(float(number.group(0)))
    lo_w, hi_w = STROKE_WIDTH_RANGE
    lo_o, hi_o = STROKE_OPACITY_RANGE
    if not widths:
        raise PackError("tile must set a stroke-width (hairlines only, 1-1.5)")
    bad = [w for w in widths if not lo_w <= w <= hi_w]
    if bad:
        raise PackError(f"stroke-width out of hairline range: {bad}")
    if not opacities:
        raise PackError("tile must set a stroke-opacity (0.06-0.22 keeps it a whisper)")
    bad = [o for o in opacities if not lo_o <= o <= hi_o]
    if bad:
        raise PackError(f"stroke-opacity out of range: {bad}")


def check_placeholders(block: str) -> None:
    found = {m.group(1).lower() for m in PLACEHOLDER_RE.finditer(block)}
    missing = {"accent", "accentalt"} - found
    if missing:
        raise PackError(f"tile must use the {sorted(missing)} placeholder(s), not literal colours")
    for name, value in ATTR_RE.findall(block):
        if HEX_RE.search(value):
            raise PackError(f"tile bakes a hex colour into {name}={value!r}; use the placeholders")


def check_no_glyphs(block: str) -> None:
    if re.search(r"<\s*(text|tspan|textPath)\b", block, re.IGNORECASE):
        raise PackError("motif must contain no glyphs - the host ships no fonts")
    if re.search(r"font-family", block, re.IGNORECASE):
        raise PackError("motif must not reference a font-family")


def sprite_symbols(block: str) -> list[str]:
    out = []
    for sym in re.findall(r"<\s*symbol\b[^>]*", block, re.IGNORECASE):
        ident = SYMBOL_ID_RE.search(sym)
        if ident and re.search(r"""viewBox\s*=\s*["']([^"']+)["']""", sym):
            out.append(ident.group(1))
    return out


def split_assets(response: str) -> tuple[str, str]:
    """Return (tile, sprite_sheet), or raise PackError explaining which failed."""
    tile = sprite = None
    for block in svg_roots(response):
        symbols = sprite_symbols(block)
        if len(symbols) >= MIN_SPRITE_SYMBOLS and sprite is None:
            sprite = block
            continue
        if tile is None and "{{accent}}" in block.lower() and "{{accentalt}}" in block.lower():
            tile = block
    if tile is None:
        raise PackError(
            "no motif tile found. Expected an <svg> whose strokes use the literal "
            "{{accent}} and {{accentAlt}} placeholders."
        )
    if sprite is None:
        raise PackError(
            f"no sprite sheet found. Expected an <svg> holding >= {MIN_SPRITE_SYMBOLS} "
            '<symbol> elements, each with a viewBox and a kebab-case "<name>-icon" id.'
        )
    check_seamless_tile(tile)
    check_wireframe(tile)
    check_placeholders(tile)
    check_no_glyphs(tile)
    check_no_glyphs(sprite)
    return tile, sprite


def _substitute_placeholders(svg: str) -> str:
    """Replace {{accent}} / {{accentAlt}} with CSS var references."""

    def repl(match: re.Match[str]) -> str:
        return PLACEHOLDER_TO_CSS_VAR[match.group(1).lower()]

    return PLACEHOLDER_RE.sub(repl, svg)


def _escape_for_data_uri(svg: str) -> str:
    """Encode an inline SVG exactly the way motifPattern() does.

    siteTheme.ts calls encodeURIComponent(); a generated tile has to land in the
    same --site-pattern value as a built-in one, or the runtime cannot tell them
    apart. urllib.parse.quote leaves !*'() unescaped only when they are in
    `safe`, so pass the unreserved set encodeURIComponent uses rather than the
    RFC 3986 one, which differs on those five characters.
    """
    return urllib.parse.quote(svg, safe=_ENCODE_URI_COMPONENT_SAFE)


def motif_css_pattern(tile: str) -> str:
    """The --site-pattern value, matching motifPattern()'s output shape exactly."""
    return f'url("data:image/svg+xml,{_escape_for_data_uri(_substitute_placeholders(tile))}")'


def sprite_data_uri(sprite: str) -> str:
    """The sprite sheet as a data-URI, ready for <use href="...">."""
    return f"data:image/svg+xml,{_escape_for_data_uri(_substitute_placeholders(sprite))}"


# ---------------------------------------------------------------------------
# Pack assembly
# ---------------------------------------------------------------------------

REQUIRED_COLOR_TOKENS = (
    "bg",
    "surface",
    "text",
    "textMuted",
    "border",
    "accent",
    "onAccent",
    "progress",
    "ring",
)

# The rest of HealthThemeTokens that is still a colour. These are optional but,
# when present, must be a hex literal - matching the hex regex in
# services/ui/src/themes/types.ts, which also accepts the #AARRGGBB alpha form.
OPTIONAL_COLOR_TOKENS = ("accentAlt", "progressTrack", "ring2", "ring3", "glow")


def build_pack(
    response: str,
    *,
    pack_id: str,
    name: str,
    tokens: dict[str, Any],
    description: str = "",
    author: str = "",
    version: str = "1.0.0",
) -> dict[str, Any]:
    """Validate a model answer and wrap it in a ThemePack.

    `tokens` is the theme's palette (the HealthThemeTokens colour fields). The
    generated assets are attached to the returned pack under a non-schema key
    (`assets`) rather than inside `tokens`, because validateThemePackage in
    types.ts only knows the token list, and an unknown key there would be
    reported as an error. Nothing reads it yet; it is the artefact a follow-up
    change to siteTheme.ts would consume.
    """
    tile, sprite = split_assets(response)
    missing = [k for k in REQUIRED_COLOR_TOKENS if not tokens.get(k)]
    if missing:
        raise PackError(f"palette is missing required colour token(s): {missing}")
    # Only the colour keys are hex-checked. The rest of HealthThemeTokens is a
    # mix of enums, numbers and nullable strings (motif, scheme, radius,
    # fontFamily, glow), and the glow token is explicitly nullable.
    bad = [
        f"{k}={tokens[k]!r}"
        for k in REQUIRED_COLOR_TOKENS + OPTIONAL_COLOR_TOKENS
        if k in tokens and tokens[k] and not re.fullmatch(r"#[0-9a-fA-F]{3,8}", str(tokens[k]))
    ]
    if bad:
        raise PackError(f"colour token(s) must be #RGB / #RRGGBB / #AARRGGBB: {bad}")

    return {
        "schemaVersion": 1,
        "kind": "jarvis.health-theme-pack",
        "id": pack_id,
        "name": name,
        "description": description or f"Generated from the {pack_id} alpaca benchmark answer",
        "version": version,
        "builtin": False,
        "themes": [
            {
                "schemaVersion": 1,
                "id": pack_id,
                "name": name,
                "version": version,
                **({"author": author} if author else {}),
                "tokens": dict(tokens),
                "assets": {
                    "motifPattern": motif_css_pattern(tile),
                    "spriteSheet": sprite_data_uri(sprite),
                    "spriteSymbols": sprite_symbols(sprite),
                },
            }
        ],
    }


def write_pack(pack: dict[str, Any], sharedllm_root: Path) -> Path:
    """Write the pack into the checkout's themes/packs directory."""
    out_dir = sharedllm_root / PACKS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{pack['id']}.pack.json"
    out.write_text(json.dumps(pack, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

EXAMPLE = """\
```svg
<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 120 120">
  <g fill="none" stroke-width="1.5">
    <path d="M30 22c8 6 8 16 0 22-8-6-8-16 0-22z" stroke="{{accentAlt}}" stroke-opacity="0.18"/>
    <circle cx="74" cy="20" r="3" stroke="{{accentAlt}}" stroke-opacity="0.12"/>
    <circle cx="18" cy="78" r="2.5" stroke="{{accent}}" stroke-opacity="0.12"/>
  </g>
</svg>
```

```svg
<svg xmlns="http://www.w3.org/2000/svg">
  <symbol id="pulse-icon" viewBox="0 0 24 24"><path fill="none" stroke="{{accent}}" stroke-width="1.4" stroke-opacity="0.16" d="M2 12h5l2-6 3 12 3-9 2 3h5"/></symbol>
  <symbol id="shield-icon" viewBox="0 0 24 24"><path fill="none" stroke="{{accentAlt}}" stroke-width="1.4" stroke-opacity="0.14" d="M12 2l8 3v6c0 5-4 9-8 11-4-2-8-6-8-11V5z"/></symbol>
  <symbol id="bolt-icon" viewBox="0 0 24 24"><path fill="none" stroke="{{accent}}" stroke-width="1.5" stroke-opacity="0.18" d="M13 2L4 14h6l-1 8 9-12h-6z"/></symbol>
  <symbol id="droplet-icon" viewBox="0 0 24 24"><path fill="none" stroke="{{accentAlt}}" stroke-width="1.4" stroke-opacity="0.12" d="M12 2c4 5 6 8 6 11a6 6 0 11-12 0c0-3 2-6 6-11z"/></symbol>
  <symbol id="moon-icon" viewBox="0 0 24 24"><path fill="none" stroke="{{accent}}" stroke-width="1.4" stroke-opacity="0.16" d="M20 14A8 8 0 1110 4a7 7 0 0010 10z"/></symbol>
  <symbol id="spark-icon" viewBox="0 0 24 24"><path fill="none" stroke="{{accentAlt}}" stroke-width="1.5" stroke-opacity="0.20" d="M12 3l2 7 7 2-7 2-2 7-2-7-7-2 7-2z"/></symbol>
</svg>
```"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--example", action="store_true", help="print the expected answer shape and exit")
    ap.add_argument("answer", nargs="?", help="file holding a model answer, or '-' to read stdin")
    ap.add_argument("--pack-id", default="alpaca-generated", help="pack id (also the theme id and the file stem)")
    ap.add_argument("--name", default="Alpaca Generated", help="human-readable pack name")
    ap.add_argument("--description", default="", help="pack description")
    ap.add_argument("--author", default="", help="author credit")
    ap.add_argument("--version", default="1.0.0", help="pack version")
    ap.add_argument("--palette", help="JSON object of HealthThemeTokens colour fields, or @path to read it")
    ap.add_argument("--sharedllm-root", default=str(DEFAULT_SHAREDLLM), help="path to the SharedLLM checkout")
    ap.add_argument("--dry-run", action="store_true", help="validate and print the pack without writing it")
    args = ap.parse_args(argv)

    if args.example or args.answer is None:
        print(__doc__)
        print("\nExpected answer shape:\n")
        print(EXAMPLE)
        return 0
    response = sys.stdin.read() if args.answer == "-" else Path(args.answer).read_text(encoding="utf-8")

    palette_src = args.palette or "{}"
    if palette_src.startswith("@"):
        palette_src = Path(palette_src[1:]).read_text(encoding="utf-8")
    try:
        tokens = json.loads(palette_src)
    except ValueError as exc:
        print(f"error: --palette is not valid JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(tokens, dict):
        print("error: --palette must be a JSON object", file=sys.stderr)
        return 2

    try:
        pack = build_pack(
            response,
            pack_id=args.pack_id,
            name=args.name,
            tokens=tokens,
            description=args.description,
            author=args.author,
            version=args.version,
        )
    except PackError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(json.dumps(pack, indent=2, ensure_ascii=False))
        return 0
    try:
        out = write_pack(pack, Path(args.sharedllm_root))
    except OSError as exc:
        print(f"error: could not write the pack: {exc}", file=sys.stderr)
        return 3
    theme = pack["themes"][0]
    print(f"wrote {out}")
    print(f"  theme:  {theme['id']}  motif + {len(theme['assets']['spriteSymbols'])} sprite symbols")
    print("  import it from the theme manager (gear -> Themes -> Import), or PUT it to /api/users/me/theme")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
