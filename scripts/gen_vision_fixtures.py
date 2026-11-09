#!/usr/bin/env python3
"""Generate the `vision` benchmark category: drawn images with known answers.

Every image is drawn here, deterministically, so its answer is a fact of the
drawing rather than a judgement: an OCR test's text is the string we drew, a
counting test's count is the number of shapes we placed. The images go into
benchmark_tests.json as `data_base64` attachments (data/ is gitignored, and the
test file is already bind-mounted into the dashboard), and the runner sends
them through the proxy's Ollama-style `images` field.

What the eight tests cover, so a vision model is measured on the jobs
SharedLLM gives its OCR model (screenshots, documents, labels):
  OCR of a document (receipt total), of a label (serial number), of a table;
  reading a UI (which switch is on, a settings screen's battery level);
  a chart (tallest bar against gridlines); counting with a distractor
  (red circles among red squares); spatial layout (a cell of a grid).

Answers are numeric wherever possible: a letter answer would join the
multiple-choice key balance (scripts/rebalance_choice_keys.py), and a word
answer can pass by accident when a model lists every option.

Usage:
  python scripts/gen_vision_fixtures.py            # report, write nothing
  python scripts/gen_vision_fixtures.py --apply    # (re)write the vision category
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TESTS_JSON = REPO / "benchmark_tests.json"

W, H = 640, 420
BG = (250, 250, 247)
INK = (28, 30, 36)
MUTED = (120, 124, 132)


def _font(size: int):
    from PIL import ImageFont

    return ImageFont.load_default(size=size)


def _canvas():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (W, H), BG)
    return img, ImageDraw.Draw(img)


def _png_b64(img) -> str:
    buf = io.BytesIO()
    # A small palette keeps these flat drawings to a few KB each.
    img.convert("P", palette=1, colors=64).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def switch_panel():
    img, d = _canvas()
    d.text((24, 18), "Living room - light switches", font=_font(26), fill=INK)
    # No ON/OFF text: the state is read from the rocker's position and colour,
    # as on a real panel, not from a printed word.
    on = 3
    for i in range(4):
        x = 40 + i * 150
        d.rounded_rectangle((x, 90, x + 110, 330), radius=14, outline=MUTED, width=3, fill=(236, 236, 232))
        lit = i + 1 == on
        knob_y = 120 if lit else 230
        d.rounded_rectangle(
            (x + 22, knob_y, x + 88, knob_y + 80), radius=10, fill=(52, 168, 83) if lit else (150, 150, 150)
        )
        d.text((x + 42, 345), str(i + 1), font=_font(34), fill=INK)
    return img, str(on)


def receipt():
    img, d = _canvas()
    d.rectangle((150, 10, 490, 410), fill=(255, 255, 255), outline=MUTED)
    f, fb = _font(22), _font(26)
    d.text((228, 24), "CORNER MARKET", font=fb, fill=INK)
    d.text((240, 56), "Receipt #40817", font=_font(18), fill=MUTED)
    items = [("Coffee beans", "14.50"), ("Oat milk", "3.75"), ("Sourdough", "6.25"), ("Apples (6)", "4.80")]
    y = 100
    for name, price in items:
        d.text((170, y), name, font=f, fill=INK)
        d.text((400, y), price, font=f, fill=INK)
        y += 38
    d.text((170, y + 4), "Tax", font=f, fill=INK)
    d.text((410, y + 4), "2.31", font=f, fill=INK)
    d.line((165, y + 46, 475, y + 46), fill=INK, width=2)
    d.text((170, y + 58), "TOTAL", font=fb, fill=INK)
    d.text((390, y + 58), "31.61", font=fb, fill=INK)
    assert f"{14.50 + 3.75 + 6.25 + 4.80 + 2.31:.2f}" == "31.61"
    return img, "31.61"


def device_label():
    img, d = _canvas()
    d.rounded_rectangle((60, 70, 580, 350), radius=18, fill=(232, 236, 241), outline=(70, 80, 95), width=4)
    f = _font(28)
    rows = [
        ("MODEL", "HX-220 Air Purifier"),
        ("SERIAL", "7Q4K-92LM"),
        ("MFG DATE", "2026-03"),
        ("INPUT", "120V 60Hz 45W"),
    ]
    for i, (k, v) in enumerate(rows):
        d.text((90, 100 + i * 58), k, font=_font(22), fill=MUTED)
        d.text((260, 96 + i * 58), v, font=f, fill=INK)
    return img, "7Q4K-92LM"


def table():
    img, d = _canvas()
    cols = [("Name", 40), ("Age", 250), ("City", 380)]
    rows = [("Marcus", "41", "Denver"), ("Priya", "29", "Austin"), ("Tomas", "57", "Boise"),
            ("Ines", "33", "Tulsa"), ("Grace", "62", "Omaha"), ("Wen", "24", "Reno")]  # fmt: skip
    f = _font(24)
    d.rectangle((24, 24, 616, 70), fill=(220, 226, 236))
    for name, x in cols:
        d.text((x, 34), name, font=f, fill=INK)
    for r, row in enumerate(rows):
        y = 82 + r * 52
        if r % 2:
            d.rectangle((24, y - 8, 616, y + 40), fill=(242, 242, 238))
        for (_, x), cell in zip(cols, row, strict=True):
            d.text((x, y), cell, font=f, fill=INK)
    return img, "29"


def bar_chart():
    img, d = _canvas()
    left, bottom, top = 80, 370, 50
    scale = (bottom - top) / 80  # y axis 0..80
    for v in range(0, 81, 10):
        y = bottom - v * scale
        d.line((left, y, 610, y), fill=(220, 220, 215), width=1)
        d.text((36, y - 10), str(v), font=_font(18), fill=MUTED)
    months = [("Jan", 30), ("Feb", 50), ("Mar", 40), ("Apr", 70), ("May", 60), ("Jun", 20)]
    for i, (m, v) in enumerate(months):
        x = left + 20 + i * 88
        d.rectangle((x, bottom - v * scale, x + 56, bottom), fill=(66, 110, 180))
        d.text((x + 10, bottom + 8), m, font=_font(18), fill=INK)
    d.text((left, 12), "Units sold per month", font=_font(22), fill=INK)
    return img, "70"


def count_shapes():
    img, d = _canvas()
    rng = random.Random(4242)
    shapes = ["red_circle"] * 7 + ["red_square"] * 3 + ["blue_square"] * 4
    rng.shuffle(shapes)
    cells = [(c, r) for r in range(3) for c in range(5)]
    rng.shuffle(cells)
    for kind, (c, r) in zip(shapes, cells, strict=False):
        x, y = 50 + c * 115 + rng.randint(-8, 8), 50 + r * 120 + rng.randint(-8, 8)
        colour = (214, 48, 49) if kind.startswith("red") else (40, 90, 200)
        if kind.endswith("circle"):
            d.ellipse((x, y, x + 70, y + 70), fill=colour)
        else:
            d.rectangle((x, y, x + 66, y + 66), fill=colour)
    return img, "7"


def settings_screen():
    img, d = _canvas()
    d.rectangle((0, 0, W, 56), fill=(34, 40, 52))
    d.text((24, 14), "Settings", font=_font(26), fill=(255, 255, 255))
    d.text((520, 16), "9:41", font=_font(22), fill=(255, 255, 255))
    rows = [("Wi-Fi", True), ("Bluetooth", False), ("Airplane mode", False), ("Battery saver", True)]
    for i, (name, on) in enumerate(rows):
        y = 80 + i * 66
        d.text((32, y + 10), name, font=_font(24), fill=INK)
        d.rounded_rectangle((520, y + 8, 590, y + 42), radius=17, fill=(52, 168, 83) if on else (190, 190, 190))
        kx = 556 if on else 524
        d.ellipse((kx, y + 11, kx + 30, y + 39), fill=(255, 255, 255))
    d.text((32, 360), "Battery", font=_font(24), fill=INK)
    d.text((470, 360), "63%", font=_font(26), fill=INK)
    return img, "63"


def grid_cell():
    img, d = _canvas()
    numbers = [4, 9, 2, 6, 1, 7, 8, 3, 5]
    size, ox, oy = 120, 140, 30
    for i, n in enumerate(numbers):
        r, c = divmod(i, 3)
        x, y = ox + c * size, oy + r * size
        d.rectangle((x, y, x + size, y + size), outline=INK, width=3, fill=(255, 255, 255))
        d.text((x + 46, y + 34), str(n), font=_font(44), fill=INK)
    return img, "8"


TESTS = [
    ("vision_switch_panel", "Vision: which wall switch is on", switch_panel,
     "The image shows a row of numbered light switches. Which switch is ON? Answer with only the switch number."),
    ("vision_receipt_total", "Vision: read a receipt total", receipt,
     "What is the TOTAL on this receipt? Answer with only the number, without a currency symbol."),
    ("vision_device_serial", "Vision: read a device label's serial", device_label,
     "What is the serial number printed on this device label? Answer with only the serial number."),
    ("vision_table_lookup", "Vision: look up a table cell", table,
     "According to the table in the image, how old is Priya? Answer with only the number."),
    ("vision_bar_chart", "Vision: read a bar chart against gridlines", bar_chart,
     "In this bar chart, what is the value of the tallest bar? Answer with only the number."),
    ("vision_count_red_circles", "Vision: count with a distractor", count_shapes,
     "How many red circles are in the image? Do not count squares. Answer with only the number."),
    ("vision_settings_battery", "Vision: read a settings screenshot", settings_screen,
     "This is a phone settings screen. What battery percentage does it show? Answer with only the number."),
    ("vision_grid_cell", "Vision: locate a grid cell", grid_cell,
     "The image shows a 3x3 grid of numbers. What number is in the bottom-left cell? Answer with only the number."),
]  # fmt: skip


def build() -> list[dict]:
    out = []
    for test_id, label, draw, prompt in TESTS:
        img, answer = draw()
        out.append(
            {
                "id": test_id,
                "kind": "image",
                "label": label,
                "prompt": prompt,
                "num_predict": 200,
                "type": "functional",
                "reasoning_estimate": 1024,
                "expected": answer,
                "attachments": [
                    {"name": f"{test_id}.png", "mime": "image/png", "kind": "image", "data_base64": _png_b64(img)}
                ],
            }
        )
    return out


def render_category(tests: list[dict]) -> str:
    body = json.dumps({"vision": tests}, indent=2, ensure_ascii=False)
    # The inner '"vision": [ ... ]' block, indented for the top-level object.
    return body[2:-2]


def apply(text: str, tests: list[dict]) -> str:
    block = render_category(tests)
    existing = re.search(r'\n  "vision": \[\n.*?\n  \]', text, re.S)
    if existing:
        return text[: existing.start()] + "\n" + block + text[existing.end() :]
    end = text.rstrip().rfind("}")
    head = text[:end].rstrip()
    return head + ",\n" + block + "\n}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="write benchmark_tests.json")
    args = parser.parse_args()
    tests = build()
    for t in tests:
        kb = len(t["attachments"][0]["data_base64"]) * 3 / 4 / 1024
        print(f"{t['id']:28} expected={t['expected']:12} {kb:5.1f} KB")
    if not args.apply:
        print("\n(dry run - pass --apply to write)")
        return
    text = TESTS_JSON.read_text(encoding="utf-8")
    new = apply(text, tests)
    json.loads(new)
    TESTS_JSON.write_text(new, encoding="utf-8")
    print(f"\nwrote {TESTS_JSON}")


if __name__ == "__main__":
    main()
