"""The Jubilee TV page on a weak TV browser (LG webOS).

The LG's built-in browser is an old Chromium on a slow CPU: the 3D card flip,
filters, blurred and animated shadows and the camera glide made every
animation crawl there while an Android TV drew them smoothly. The page turns
on a low-power mode (``html.lowfx``) for webOS only. These tests pin that the
detection exists and leaves Android TV alone, that the low-power styles really
flatten 3D and drop filters, and that the page stays ES5 (an old webOS
Chromium may not parse ES6 at all, which would leave the TV blank).
"""
from __future__ import annotations

import re
from pathlib import Path

PAGE = Path(__file__).resolve().parent.parent / "arcade" / "templates" / "jubilee_tv.html"

LG_UA = ("Mozilla/5.0 (Web0S; Linux/SmartTV) AppleWebKit/537.36 (KHTML, like Gecko) "
         "Chrome/68.0.3440.106 Safari/537.36 WebAppManager")
NETCAST_UA = "Mozilla/5.0 (DirectFB; Linux armv7l) AppleWebKit/534.26+ (KHTML, like Gecko) Version/5.0 Safari/534.26+ LG Browser/5.00.00(+mouse+3D+SCREEN+TUNER; LGE; 42LM6700-ZA; 04.25.05; 0x00000001;); LG NetCast.TV-2012 0"
ANDROID_TV_UA = ("Mozilla/5.0 (Linux; Android 12; BRAVIA 4K VH2 Build/STT1.211025.001.Z4; wv) AppleWebKit/537.36 "
                 "(KHTML, like Gecko) Version/4.0 Chrome/120.0.6099.230 Mobile Safari/537.36 Android TV")
CAST_UA = ("Mozilla/5.0 (X11; Linux armv7l) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 "
           "Safari/537.36 CrKey/1.56.500000 DeviceType/AndroidTV")
DESKTOP_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/130.0.0.0 Safari/537.36")


def _html() -> str:
    return PAGE.read_text(encoding="utf-8")


def _styles() -> str:
    css = "\n".join(re.findall(r"<style>(.*?)</style>", _html(), re.S))
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


def _scripts() -> list[str]:
    return re.findall(r"<script>(.*?)</script>", _html(), re.S)


def _lowfx_regex() -> re.Pattern[str]:
    head = _scripts()[0]
    m = re.search(r"weak = /([^/]+)/i\.test\(navigator\.userAgent", head)
    assert m, "the head script must detect a weak TV browser from its user agent"
    return re.compile(m.group(1), re.I)


def test_low_power_mode_is_decided_before_first_paint_with_an_override():
    head = _scripts()[0]
    assert _html().index(head) < _html().index("<style>"), "decided in the head, before the styles apply"
    assert "/[?&]lowfx=([01])/" in head, "?lowfx=1 forces it, ?lowfx=0 turns it off"
    assert 'className += " lowfx"' in head


def test_low_power_mode_is_for_webos_only():
    weak = _lowfx_regex()
    assert weak.search(LG_UA) and weak.search(NETCAST_UA)
    for ua in (ANDROID_TV_UA, CAST_UA, DESKTOP_UA):
        assert not weak.search(ua), f"low-power mode must leave this browser alone: {ua}"


def _lowfx_rules() -> dict[str, str]:
    """Every ``html.lowfx`` rule: selector -> declarations."""
    rules: dict[str, str] = {}
    for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", _styles()):
        for one in sel.split(","):
            one = one.strip()
            if one.startswith("html.lowfx"):
                rules[one] = rules.get(one, "") + body
    return rules


def _decl(rules: dict[str, str], selector: str) -> str:
    assert selector in rules, f"no low-power rule for {selector}"
    return re.sub(r"\s+", " ", rules[selector])


def test_low_power_styles_flatten_3d_and_drop_filters():
    rules = _lowfx_rules()
    # The drawn card: no perspective, no preserve-3d, no turned face.
    assert "perspective: none" in _decl(rules, "html.lowfx .flyCard")
    assert "transform-style: flat" in _decl(rules, "html.lowfx .flyInner")
    assert "transform: none" in _decl(rules, "html.lowfx .flyCard .face.front")
    assert "display: none" in _decl(rules, "html.lowfx .flyCard .face.back")
    # No filters anywhere they are used.
    for sel in ("html.lowfx .ring .tkB", "html.lowfx .plr.out", "html.lowfx .crown"):
        assert "filter: none" in _decl(rules, sel)
    # No always-running decoration: the glowing sign (an animated box-shadow
    # repainting the board) and the turning sunburst.
    sign = _decl(rules, "html.lowfx .cSign")
    assert "animation: none" in sign and "box-shadow: none" in sign
    assert "display: none" in _decl(rules, "html.lowfx .cRays")
    # Blurred shadows gone, tumbling dice not spinning.
    for sel in ("html.lowfx .tvDie", "html.lowfx .tvDice", "html.lowfx .plr.turn"):
        assert "box-shadow: none" in _decl(rules, sel)
    assert "animation: none" in _decl(rules, "html.lowfx .tvDicePair.tumbling .tvDie")
    # And nothing in low-power mode brings 3D, filters or blur back.
    for sel, body in rules.items():
        assert not re.search(r"rotate[XYZ3]|translate3d|perspective:\s*\d|preserve-3d|blur\(|drop-shadow", body), sel


def test_the_page_has_no_filters_or_3d_outside_what_low_power_mode_overrides():
    """Every expensive declaration in the page has a low-power override, so a
    new one cannot slip in and slow the LG down unnoticed."""
    rules = _lowfx_rules()
    covered = {s.replace("html.lowfx ", "", 1) for s in rules}
    for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", _styles()):
        if "html.lowfx" in sel or "@keyframes" in sel:
            continue
        if re.search(r"(?<!-)filter:\s*(?!none)|perspective:|preserve-3d|backdrop-filter", body):
            for one in sel.split(","):
                one = one.strip()
                if one.startswith("html.lite") or one.startswith("@media"):
                    continue
                assert one in covered, f"{one} uses a filter or 3D with no html.lowfx override"


def test_the_card_flight_skips_the_3d_turn_in_low_power_mode():
    js = _scripts()[1]
    assert "var LOWFX = " in js
    assert "(RM || LOWFX) ? 1 : 1.7" in js, "no camera glide in low-power mode"
    fly = js[js.index("function flyCard"):js.index("window.__jbStage")]
    assert "LOWFX ? ''" in fly, "no back face to turn over"
    assert fly.count('if (!LOWFX) inner.style.transform = "rotateY(180deg)"') == 2


def test_the_frame_loop_sleeps_when_nothing_moves():
    js = _scripts()[1]
    tick = js[js.index("function tick(now)"):js.index("/* ---------- the middle")]
    assert "requestAnimationFrame(tick); else rafOn = false" in tick
    assert "clientWidth" not in tick, "no layout read inside the frame loop"
    cam = js[js.index("function camera(now)"):js.index("function tick(now)")]
    assert "if (!board.W)" in cam, "the board's size is read once, not every frame"


def test_the_qr_is_not_refetched_on_every_update():
    js = _scripts()[1]
    assert "board.right.appendChild(board.joinEl)" in js


def test_no_css_inset_which_old_webos_ignores():
    assert not re.search(r"(?<![-\w])inset\s*:", _styles()), "Chrome < 87 (webOS) ignores `inset`"


def _strip_js(src: str) -> str:
    """The code without comments, strings or regex literals."""
    out, i, n = [], 0, len(src)
    prev = ""   # last significant character, to tell a regex from a division
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            i = src.find("\n", i)
            i = n if i < 0 else i
            continue
        if src.startswith("/*", i):
            i = src.index("*/", i + 2) + 2
            continue
        if c in "'\"":
            j = i + 1
            while src[j] != c:
                j += 2 if src[j] == "\\" else 1
            out.append('""')
            i, prev = j + 1, '"'
            continue
        if c == "/" and (prev in "(,=:[!&|?{};+" or prev == "" or (out and "".join(out).rstrip().endswith("return"))):
            j, in_class = i + 1, False
            while True:
                ch = src[j]
                if ch == "\\":
                    j += 2
                    continue
                if ch == "[":
                    in_class = True
                elif ch == "]":
                    in_class = False
                elif ch == "/" and not in_class:
                    break
                j += 1
            j += 1
            while j < n and src[j].isalpha():
                j += 1
            out.append("/r/")
            i, prev = j, "/"
            continue
        out.append(c)
        if not c.isspace():
            prev = c
        i += 1
    return "".join(out)


def test_the_page_is_es5():
    for src in _scripts():
        code = _strip_js(src)
        for pat, what in ((r"=>", "an arrow function"), (r"\blet\b", "let"), (r"\bconst\b", "const"),
                          (r"`", "a template literal"), (r"\bclass\s+[A-Za-z]", "a class"),
                          (r"\.\.\.", "spread"), (r"\basync\b|\bawait\b", "async/await"),
                          (r"\bfor\s*\([^)]*\bof\b", "for...of")):
            assert not re.search(pat, code), f"{what} in the TV page; old webOS Chromium may not parse it"
