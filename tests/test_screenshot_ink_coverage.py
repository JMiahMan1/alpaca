"""Tests for the ink-coverage floor in sandbox_exec._screenshot_has_content.

Variance and colour count cannot tell a designed page from a page with one line
of text on it, because anti-aliasing alone produces hundreds of distinct greys.
These tests pin the new third signal, and - more importantly - pin that it does
not reject the sparsest *real* benchmark frame we have on disk.
"""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("sandbox_exec_under_test", REPO / "sandbox_exec.py")
assert _spec and _spec.loader
sandbox_exec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sandbox_exec)

SCREENSHOTS = REPO / "docs" / "screenshots"


def png_bytes(img: Image.Image) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, "PNG")
    return buffer.getvalue()


def text_page(lines: list[str], size: tuple[int, int] = (1024, 768)) -> bytes:
    """A near-blank parchment page carrying only the given lines of text."""
    img = Image.new("RGB", size, (250, 246, 238))
    draw = ImageDraw.Draw(img)
    for index, line in enumerate(lines):
        draw.text((120, 60 + index * 40), line, fill=(40, 30, 20))
    return png_bytes(img)


# --------------------------------------------------------------------------
# The floor is derived from real frames, so pin the frames it sits between
# --------------------------------------------------------------------------


@pytest.mark.skipif(not (SCREENSHOTS / "game_pong.png").exists(), reason="benchmark screenshot not present")
def test_the_sparsest_real_screenshot_still_passes():
    """docs/screenshots/game_pong.png is a real, accepted benchmark frame and it
    is the sparsest one in the repository. If the floor ever rises above its ink
    coverage this is the test that says so, by name."""
    coverage = sandbox_exec._frame_ink_coverage((SCREENSHOTS / "game_pong.png").read_bytes())
    assert coverage is not None
    assert coverage > sandbox_exec._MIN_INK_COVERAGE
    assert sandbox_exec._screenshot_has_content((SCREENSHOTS / "game_pong.png").read_bytes()) is True


@pytest.mark.skipif(not (SCREENSHOTS / "retro_space_invaders.png").exists(), reason="benchmark screenshot not present")
def test_a_dense_real_screenshot_passes():
    raw = (SCREENSHOTS / "retro_space_invaders.png").read_bytes()
    assert sandbox_exec._frame_ink_coverage(raw) > sandbox_exec._MIN_INK_COVERAGE
    assert sandbox_exec._screenshot_has_content(raw) is True


@pytest.mark.skipif(not (SCREENSHOTS / "game_falling_sand.png").exists(), reason="benchmark screenshot not present")
def test_a_uniform_frame_is_still_rejected():
    raw = (SCREENSHOTS / "game_falling_sand.png").read_bytes()
    assert sandbox_exec._frame_ink_coverage(raw) == pytest.approx(0.0, abs=1e-9)
    assert sandbox_exec._screenshot_has_content(raw) is False


# --------------------------------------------------------------------------
# The gap this closes
# --------------------------------------------------------------------------


def test_a_page_with_one_line_of_text_is_no_longer_rendered():
    raw = text_page(["In the beginning was the Word, and the Word was with God."])
    # It clears both old signals...
    assert sandbox_exec._screenshot_has_content(raw) is False
    # ...and the reason is coverage, not variance.
    coverage = sandbox_exec._frame_ink_coverage(raw)
    assert coverage is not None
    assert coverage < sandbox_exec._MIN_INK_COVERAGE


def test_a_filled_page_of_text_is_rendered():
    verses = [f"{i}. In the beginning was the Word, verse {i} of the chapter." for i in range(1, 19)]
    raw = text_page(verses)
    assert sandbox_exec._screenshot_has_content(raw) is True


def test_one_bordered_page_is_rendered():
    img = Image.new("RGB", (1024, 768), (250, 246, 238))
    ImageDraw.Draw(img).rectangle([20, 20, 1003, 747], outline=(150, 40, 40), width=6)
    assert sandbox_exec._screenshot_has_content(png_bytes(img)) is True


# --------------------------------------------------------------------------
# Ink coverage itself
# --------------------------------------------------------------------------


def test_ink_coverage_is_none_for_unreadable_input():
    assert sandbox_exec._frame_ink_coverage(b"") is None
    assert sandbox_exec._frame_ink_coverage(b"not a png at all") is None


def test_ink_coverage_is_zero_for_a_uniform_frame():
    assert sandbox_exec._frame_ink_coverage(png_bytes(Image.new("RGB", (200, 200), (10, 20, 30)))) == 0.0


def test_ink_coverage_of_a_half_covered_frame_is_about_a_half():
    img = Image.new("RGB", (400, 400), (0, 0, 0))
    ImageDraw.Draw(img).rectangle([0, 0, 200, 400], fill=(255, 255, 255))
    coverage = sandbox_exec._frame_ink_coverage(png_bytes(img))
    assert coverage == pytest.approx(0.5, abs=0.02)


def test_coverage_is_measured_after_the_quarter_scale_downsample():
    """A one-pixel checkerboard is 50% ink at full resolution, but the 4x
    downscale averages every 4x4 block to its mean, so at the resolution the
    measurement happens at the frame is a single flat colour and reads as
    blank. That is the correct verdict - indistinguishable from an empty frame
    is empty - and it is why the floor had to be calibrated against real
    1024x768 screenshots rather than against synthetic texture."""
    img = Image.new("RGB", (256, 256))
    img.putdata([(0, 0, 0) if (i % 2 == 0) else (255, 255, 255) for i in range(256 * 256)])
    coverage = sandbox_exec._frame_ink_coverage(png_bytes(img))
    assert coverage == 0.0


def test_paper_texture_below_the_delta_is_not_ink():
    """A real vellum ground has noise. It must not read as decoration, or every
    textured page would sail past the floor."""
    img = Image.new("RGB", (400, 400), (250, 246, 238))
    pixels = img.load()
    for y in range(400):
        for x in range(400):
            jitter = (x * 7 + y * 3) % 9 - 4  # within +/-4, well under the delta
            pixels[x, y] = (250 + jitter, 246 + jitter, 238 + jitter)
    assert sandbox_exec._frame_ink_coverage(png_bytes(img)) == 0.0


def test_the_floor_is_between_the_one_line_page_and_the_sparsest_real_frame():
    """The floor is a judgement call; pin the judgement so raising it is a
    deliberate act."""
    one_line = sandbox_exec._frame_ink_coverage(text_page(["In the beginning was the Word."]))
    filled = sandbox_exec._frame_ink_coverage(
        text_page([f"{i}. In the beginning was the Word, and the Word was with God." for i in range(18)])
    )
    assert one_line < sandbox_exec._MIN_INK_COVERAGE < filled


# --------------------------------------------------------------------------
# Unchanged behaviour
# --------------------------------------------------------------------------


def test_empty_input_is_not_content():
    assert sandbox_exec._screenshot_has_content(b"") is False


def test_a_malformed_png_is_not_content():
    assert sandbox_exec._screenshot_has_content(b"\x89PNG\r\n\x1a\n garbage") is False


def test_pure_black_and_pure_white_are_not_content():
    assert sandbox_exec._screenshot_has_content(png_bytes(Image.new("RGB", (64, 64), (0, 0, 0)))) is False
    assert sandbox_exec._screenshot_has_content(png_bytes(Image.new("RGB", (64, 64), (255, 255, 255)))) is False


def test_a_tiny_frame_still_renders_when_it_has_ink():
    assert sandbox_exec._screenshot_has_content(png_bytes(Image.new("RGB", (4, 4), (0, 0, 0)))) is False
    img = Image.new("RGB", (8, 8), (0, 0, 0))
    img.putdata([(255, 255, 255) if i % 2 else (0, 0, 0) for i in range(64)])
    assert sandbox_exec._screenshot_has_content(png_bytes(img)) is True


def test_missing_pillow_still_reports_content_so_no_real_ui_is_rejected(monkeypatch):
    monkeypatch.setattr(sandbox_exec, "_PILImage", None)
    assert sandbox_exec._screenshot_has_content(b"anything") is True
    assert sandbox_exec._frame_ink_coverage(b"anything") is None


def test_the_grader_gate_uses_has_content_for_ui_results():
    """ui_rendered is the success signal for a windowed app, so the floor has to
    reach the result dict and not just a private helper."""
    source = (REPO / "sandbox_exec.py").read_text()
    assert "rendered = _screenshot_has_content(png)" in source
    assert 'result["ran"] = rendered' in source
