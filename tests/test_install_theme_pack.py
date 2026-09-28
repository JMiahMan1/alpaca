"""Tests for scripts/install_theme_pack.py - the consumer half of the
creative_svg_theme_pack benchmark.

The benchmark grades a model-authored SVG tile + sprite sheet. This script takes
that answer and makes it a real `jarvis.health-theme-pack` JSON document. The
tests here pin the two things that make the output usable rather than merely
well-formed:

* the assets are data-URIs, because a ThemePack is a JSON document that has to
  survive localStorage, a Blob export and a `GET/PUT /api/users/me/theme`
  column, and the theme manager's import control only accepts
  `application/json`; and
* the `{{accent}}` placeholder the model was asked to use becomes a CSS custom
  property reference, not a baked hex, so one tile still works on every theme in
  the pack - and the encoding is byte-identical to the `encodeURIComponent` call
  the real `motifPattern()` uses, so a generated tile and a built-in one are
  indistinguishable to the runtime.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import urllib.parse
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("install_theme_pack_under_test", REPO / "scripts/install_theme_pack.py")
assert _spec and _spec.loader
itp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(itp)

GOOD_TILE = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 120 120">\n'
    '  <g fill="none" stroke-width="1.5">\n'
    '    <path d="M30 22c8 6 8 16 0 22-8-6-8-16 0-22z" stroke="{{accentAlt}}" stroke-opacity="0.18"/>\n'
    '    <circle cx="74" cy="20" r="3" stroke="{{accentAlt}}" stroke-opacity="0.12"/>\n'
    '    <circle cx="18" cy="78" r="2.5" stroke="{{accent}}" stroke-opacity="0.12"/>\n'
    "  </g>\n"
    "</svg>"
)
GOOD_SPRITE = (
    '<svg xmlns="http://www.w3.org/2000/svg">'
    '<symbol id="pulse-icon" viewBox="0 0 24 24"><path fill="none" stroke="{{accent}}" stroke-width="1.4" '
    'stroke-opacity="0.16" d="M2 12h5l2-6 3 12 3-9 2 3h5"/></symbol>'
    '<symbol id="shield-icon" viewBox="0 0 24 24"><path fill="none" stroke="{{accentAlt}}" stroke-width="1.4" '
    'stroke-opacity="0.14" d="M12 2l8 3v6c0 5-4 9-8 11-4-2-8-6-8-11V5z"/></symbol>'
    '<symbol id="bolt-icon" viewBox="0 0 24 24"><path fill="none" stroke="{{accent}}" stroke-width="1.5" '
    'stroke-opacity="0.18" d="M13 2L4 14h6l-1 8 9-12h-6z"/></symbol>'
    '<symbol id="droplet-icon" viewBox="0 0 24 24"><path fill="none" stroke="{{accentAlt}}" stroke-width="1.4" '
    'stroke-opacity="0.12" d="M12 2c4 5 6 8 6 11a6 6 0 11-12 0c0-3 2-6 6-11z"/></symbol>'
    '<symbol id="moon-icon" viewBox="0 0 24 24"><path fill="none" stroke="{{accent}}" stroke-width="1.4" '
    'stroke-opacity="0.16" d="M20 14A8 8 0 1110 4a7 7 0 0010 10z"/></symbol>'
    '<symbol id="spark-icon" viewBox="0 0 24 24"><path fill="none" stroke="{{accentAlt}}" stroke-width="1.5" '
    'stroke-opacity="0.20" d="M12 3l2 7 7 2-7 2-2 7-2-7-7-2 7-2z"/></symbol>'
    "</svg>"
)
GOOD_ANSWER = f"```svg\n{GOOD_TILE}\n```\n\n```svg\n{GOOD_SPRITE}\n```"

PALETTE = {
    "bg": "#0B1020",
    "surface": "#151A2E",
    "text": "#E2E8F0",
    "textMuted": "#94A3B8",
    "border": "#334755",
    "accent": "#863BFF",
    "onAccent": "#F8FAFC",
    "progress": "#863BFF",
    "ring": "#863BFF",
}


def build(response: str = GOOD_ANSWER, **overrides):
    kwargs = {"pack_id": "test-pack", "name": "Test Pack", "tokens": dict(PALETTE)}
    kwargs.update(overrides)
    return itp.build_pack(response, **kwargs)


def decode_motif(pattern: str) -> str:
    """Undo the data-URI wrapping so a test can read the SVG source back."""
    prefix = 'url("data:image/svg+xml,'
    assert pattern.startswith(prefix) and pattern.endswith('")')
    return urllib.parse.unquote(pattern[len(prefix) : -2])


# ==========================================================================
# Validation - the installer must be as strict as the grader
# ==========================================================================


def test_a_good_answer_builds():
    pack = build()
    assert pack["kind"] == "jarvis.health-theme-pack"
    assert pack["schemaVersion"] == 1
    assert pack["builtin"] is False
    assert len(pack["themes"]) == 1
    assert pack["themes"][0]["id"] == "test-pack"


def test_a_seaming_tile_is_rejected():
    with pytest.raises(itp.PackError, match="square"):
        build(GOOD_TILE.replace('height="120"', 'height="240"') + GOOD_SPRITE)


def test_a_hex_baked_tile_is_rejected():
    with pytest.raises(itp.PackError, match="hex colour"):
        # Both placeholders are still present, so the tile is recognised; the
        # hex is on a third shape, which is the case a placeholder-only check
        # would miss.
        sneaky = GOOD_TILE.replace("</g>", '<path d="M0 0" stroke="#863BFF" stroke-opacity="0.1"/></g>')
        build(sneaky + GOOD_SPRITE)


def test_dropping_a_placeholder_is_reported_as_a_missing_tile():
    with pytest.raises(itp.PackError, match="no motif tile"):
        build(GOOD_TILE.replace("{{accentAlt}}", "{{accent}}") + GOOD_SPRITE)


def test_a_glyph_in_the_tile_is_rejected():
    with pytest.raises(itp.PackError, match="no glyphs"):
        build(GOOD_TILE.replace("</g>", '<text x="1" y="2">hi</text></g>') + GOOD_SPRITE)


def test_a_glyph_in_the_sprite_sheet_is_rejected():
    with pytest.raises(itp.PackError, match="no glyphs"):
        build(GOOD_TILE + GOOD_SPRITE.replace("</svg>", "<text>hi</text></svg>"))


def test_five_symbols_is_one_short():
    trimmed = GOOD_SPRITE.replace(
        '<symbol id="spark-icon" viewBox="0 0 24 24"><path fill="none" stroke="{{accentAlt}}" stroke-width="1.5" '
        'stroke-opacity="0.20" d="M12 3l2 7 7 2-7 2-2 7-2-7-7-2 7-2z"/></symbol>',
        "",
    )
    with pytest.raises(itp.PackError, match="sprite sheet"):
        build(GOOD_TILE + trimmed)


def test_a_bold_stroke_is_rejected():
    with pytest.raises(itp.PackError, match="stroke-width"):
        build(GOOD_TILE.replace('stroke-width="1.5"', 'stroke-width="6"') + GOOD_SPRITE)


def test_a_missing_sprite_sheet_is_reported_clearly():
    with pytest.raises(itp.PackError, match="no sprite sheet"):
        build(GOOD_TILE)


def test_a_missing_tile_is_reported_clearly():
    with pytest.raises(itp.PackError, match="no motif tile"):
        build(GOOD_SPRITE)


# ==========================================================================
# Palette validation
# ==========================================================================


def test_a_missing_required_token_is_reported_by_name():
    tokens = dict(PALETTE)
    del tokens["ring"]
    with pytest.raises(itp.PackError, match=r"missing required colour token\(s\): \['ring'\]"):
        build(tokens=tokens)


@pytest.mark.parametrize("key", ["bg", "accent", "progress"])
def test_a_malformed_required_colour_is_rejected(key):
    with pytest.raises(itp.PackError, match="#RGB"):
        build(tokens={**PALETTE, key: "rebeccapurple"})


def test_optional_colour_tokens_are_checked_when_present():
    with pytest.raises(itp.PackError, match="#RGB"):
        build(tokens={**PALETTE, "ring2": "not-a-colour"})


def test_a_null_optional_colour_is_allowed():
    """glow is explicitly nullable in HealthThemeTokens."""
    pack = build(tokens={**PALETTE, "glow": None, "ring3": ""})
    assert pack["themes"][0]["tokens"]["glow"] is None


def test_non_colour_tokens_are_not_hex_checked():
    """motif/scheme/radius/fontFamily are enums, numbers and strings, not colours."""
    tokens = {**PALETTE, "motif": "petal", "scheme": "dark", "radius": 14, "fontFamily": "Outfit", "showCornerCut": False}
    assert build(tokens=tokens)["themes"][0]["tokens"]["radius"] == 14


def test_the_eight_digit_alpha_colour_form_is_accepted():
    """types.ts allows #AARRGGBB (its own border token is #33475569)."""
    assert build(tokens={**PALETTE, "border": "#33475569"})["themes"][0]["tokens"]["border"] == "#33475569"


# ==========================================================================
# The asset payloads - this is the part that has to work at runtime
# ==========================================================================


def test_the_motif_lands_as_a_data_uri_exactly_like_motifpattern():
    pattern = build()["themes"][0]["assets"]["motifPattern"]
    assert pattern.startswith('url("data:image/svg+xml,')
    assert pattern.endswith('")')
    assert "<svg" not in pattern, "the SVG must be percent-encoded, not inlined raw"


def test_the_encoding_is_byte_identical_to_javascript_encodeuricomponent():
    """motifPattern() uses encodeURIComponent(). If this encoder differed even on
    one character, a generated tile and a built-in one would not be the same
    kind of value."""
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="48"><g fill="none" stroke-width="1"><path d="M0 0h8" stroke="{{accent}}" stroke-opacity="0.06"/></g></svg>'
    try:
        js = subprocess.run(
            ["node", "-e", f"process.stdout.write(encodeURIComponent({json.dumps(svg)}))"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("Node.js required to compare against encodeURIComponent")
    assert itp._escape_for_data_uri(svg) == js


def test_placeholders_become_css_custom_properties_not_baked_hex():
    inner = decode_motif(build()["themes"][0]["assets"]["motifPattern"])
    assert "var(--site-accent)" in inner
    assert "var(--site-accent-alt)" in inner
    assert "{{" not in inner
    assert "#" not in inner, "a tile that can only work on one theme is not a theme tile"


def test_the_sprite_sheet_also_uses_css_variables():
    sheet = urllib.parse.unquote(build()["themes"][0]["assets"]["spriteSheet"].split(",", 1)[1])
    assert "var(--site-accent)" in sheet and "{{" not in sheet


def test_the_sprite_symbol_ids_are_reported_for_later_use():
    assert build()["themes"][0]["assets"]["spriteSymbols"] == [
        "pulse-icon",
        "shield-icon",
        "bolt-icon",
        "droplet-icon",
        "moon-icon",
        "spark-icon",
    ]


def test_the_assets_live_beside_tokens_not_inside_them():
    """validateThemePackage only knows the HealthThemeTokens list, so an unknown
    key inside `tokens` would be reported as an error on import."""
    theme = build()["themes"][0]
    assert "assets" in theme
    assert "assets" not in theme["tokens"]
    assert not any(k not in itp.REQUIRED_COLOR_TOKENS and k in theme["tokens"] for k in ("motifPattern", "spriteSheet"))


def test_the_assets_survive_a_json_round_trip():
    """The pack is persisted, exported as a Blob, and PUT to the identity
    service, so every character has to be JSON-safe and re-parseable."""
    pack = build()
    again = json.loads(json.dumps(pack, ensure_ascii=False))
    assert again["themes"][0]["assets"]["motifPattern"] == pack["themes"][0]["assets"]["motifPattern"]


# ==========================================================================
# Writing
# ==========================================================================


def test_write_pack_lands_in_the_themes_packs_directory(tmp_path):
    out = itp.write_pack(build(), tmp_path)
    assert out == tmp_path / "services/ui/src/themes/packs/test-pack.pack.json"
    assert json.loads(out.read_text())["id"] == "test-pack"
    assert out.read_text().endswith("\n"), "a hand-editable JSON file should end with a newline"


def test_write_pack_creates_the_directory(tmp_path):
    assert not (tmp_path / itp.PACKS_DIR).exists()
    itp.write_pack(build(), tmp_path)
    assert (tmp_path / itp.PACKS_DIR).is_dir()


def test_write_pack_overwrites_a_previous_run(tmp_path):
    itp.write_pack(build(), tmp_path)
    second = itp.write_pack(build(pack_id="test-pack", name="Renamed"), tmp_path)
    assert json.loads(second.read_text())["name"] == "Renamed"
    assert len(list((tmp_path / itp.PACKS_DIR).glob("*.pack.json"))) == 1


def test_the_author_is_omitted_unless_given():
    assert "author" not in build()["themes"][0]
    assert build(author="alpaca")["themes"][0]["author"] == "alpaca"


def test_a_description_defaults_to_naming_the_benchmark():
    # description lives on the pack, not the theme: ThemePackage.description is
    # optional and validateThemePack does not require it.
    assert "alpaca" in build()["description"]
    assert build(description="handmade")["description"] == "handmade"


# ==========================================================================
# CLI
# ==========================================================================


def run_cli(*args, stdin: str = "") -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO / "scripts/install_theme_pack.py"), *args],
        capture_output=True,
        text=True,
        input=stdin,
        timeout=60,
        check=False,
    )


def test_the_script_is_executable_and_has_a_shebang():
    text = (REPO / "scripts/install_theme_pack.py").read_text()
    assert text.startswith("#!/usr/bin/env python3")


def test_no_arguments_prints_the_expected_answer_shape():
    res = run_cli()
    assert res.returncode == 0
    assert "Expected answer shape" in res.stdout
    assert "creative_svg_theme_pack" in res.stdout


def test_the_documented_example_actually_validates():
    """The example in --help has to be a working input, or it is a trap."""
    res = run_cli("--example")
    body = res.stdout[res.stdout.index("```svg") : res.stdout.rindex("```") + 3]
    assert run_cli("-", "--dry-run", "--palette", json.dumps(PALETTE), stdin=body).returncode == 0


def test_a_bad_answer_exits_non_zero_with_a_reason(tmp_path):
    answer = tmp_path / "bad.md"
    answer.write_text("I cannot produce that.")
    res = run_cli(str(answer), "--palette", json.dumps(PALETTE))
    assert res.returncode == 1
    assert "error:" in res.stderr
    assert "no motif tile" in res.stderr


def test_a_bad_palette_exits_two(tmp_path):
    answer = tmp_path / "good.md"
    answer.write_text(GOOD_ANSWER)
    assert run_cli(str(answer), "--palette", "{not json").returncode == 2
    assert run_cli(str(answer), "--palette", "[1,2]").returncode == 2


def test_a_palette_can_be_read_from_a_file(tmp_path):
    answer = tmp_path / "good.md"
    answer.write_text(GOOD_ANSWER)
    palette = tmp_path / "palette.json"
    palette.write_text(json.dumps(PALETTE))
    assert run_cli(str(answer), "--palette", f"@{palette}", "--dry-run").returncode == 0


def test_dry_run_prints_the_pack_and_writes_nothing(tmp_path):
    answer = tmp_path / "good.md"
    answer.write_text(GOOD_ANSWER)
    res = run_cli(str(answer), "--palette", json.dumps(PALETTE), "--dry-run", "--sharedllm-root", str(tmp_path / "nope"))
    assert res.returncode == 0
    assert json.loads(res.stdout)["id"] == "alpaca-generated"
    assert not (tmp_path / "nope").exists()


def test_a_real_write_reports_the_path_and_how_to_load_it(tmp_path):
    answer = tmp_path / "good.md"
    answer.write_text(GOOD_ANSWER)
    res = run_cli(
        str(answer),
        "--palette",
        json.dumps(PALETTE),
        "--pack-id",
        "cli-pack",
        "--sharedllm-root",
        str(tmp_path),
    )
    assert res.returncode == 0
    assert "cli-pack.pack.json" in res.stdout
    assert "Import" in res.stdout
    assert (tmp_path / "services/ui/src/themes/packs/cli-pack.pack.json").is_file()


def test_reading_from_stdin_works(tmp_path):
    res = run_cli("-", "--palette", json.dumps(PALETTE), "--dry-run", stdin=GOOD_ANSWER)
    assert res.returncode == 0
    assert json.loads(res.stdout)["kind"] == "jarvis.health-theme-pack"
