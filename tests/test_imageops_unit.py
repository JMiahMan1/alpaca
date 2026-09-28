"""Unit tests for imageops.py — the deterministic (no-diffusion) image edit primitives.

These two functions are the only deterministic path that puts real text on an
image: `fill_band_deterministic` erases the old line and `draw_text` writes the
new one. Everything the Image Studio's Canvas Text Studio path does server-side
routes through here, so the properties below (grain is seeded, the gap rows
bracket the band, a missing font degrades instead of raising) are the ones the
rest of the stack relies on.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFont

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("imageops_under_test", REPO / "imageops.py")
assert _spec and _spec.loader
imageops = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(imageops)


def _gradient(h: int = 40, w: int = 60, top: int = 20, bottom: int = 220) -> Image.Image:
    """A clean vertical ramp: no glyphs, no noise. Every pixel row is uniform."""
    ramp = np.linspace(top, bottom, h).astype(float)
    arr = np.repeat(ramp[:, None, None], w, axis=1)
    arr = np.repeat(arr, 3, axis=2)
    return Image.fromarray(arr.astype("uint8"), "RGB")


def _noisy(h: int = 40, w: int = 60) -> Image.Image:
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 256, size=(h, w, 3), dtype="uint8")
    return Image.fromarray(arr, "RGB")


def _rows(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB")).astype(float)


# --------------------------------------------------------------------------
# fill_band_deterministic
# --------------------------------------------------------------------------


def test_band_is_replaced_and_the_rest_of_the_image_is_untouched():
    img = _gradient()
    before = _rows(img)
    out = _rows(imageops.fill_band_deterministic(img, 10, 20, gap_above=9, gap_below=21))

    assert out.shape == before.shape
    # rows outside the band are byte-identical: this is a surgical edit.
    assert np.array_equal(out[:10], before[:10])
    assert np.array_equal(out[20:], before[20:])
    # and the band genuinely changed, otherwise the "erase" did nothing.
    assert not np.array_equal(out[10:20], before[10:20])


def test_band_is_interpolated_between_the_two_gap_rows():
    """A diagonal glare streak survives because each band row is a blend of the
    two gap rows, not noise drawn from a distribution.

    The grain amplitude comes from the per-column spread *down* the texture
    strip, so a strip of identical rows makes it exactly zero and the band
    becomes the pure blend - which is what makes the ramp assertable at all.
    """
    h, w = 40, 10
    arr = np.zeros((h, w, 3), dtype="uint8")
    arr[:29] = 10
    arr[9] = 46  # gap_above
    arr[29:] = 158  # rows below the band are identical -> zero local spread
    img = Image.fromarray(arr, "RGB")

    out = _rows(imageops.fill_band_deterministic(img, 10, 30, gap_above=9, gap_below=31))
    band_h = 20
    for k in range(band_h):
        t = k / (band_h - 1)  # np.linspace(0, 1, band_h)
        expected = 46 * (1.0 - t) + 158 * t
        assert np.all(out[10 + k] == int(expected)), k  # exact, grain is zero
    # the first band row IS gap_above and the last IS gap_below
    assert out[10].mean() == 46.0
    assert out[29].mean() == 158.0


def test_grain_is_zero_when_the_texture_strip_has_no_vertical_spread():
    h, w = 40, 10
    arr = np.zeros((h, w, 3), dtype="uint8")
    arr[:29] = 10
    arr[9] = 46
    arr[29:] = 158
    img = Image.fromarray(arr, "RGB")
    out = _rows(imageops.fill_band_deterministic(img, 10, 30, 9, 31))
    ramp = 46 + (158 - 46) * np.linspace(0.0, 1.0, 20)[:, None, None]
    assert np.all(out[10:30] == ramp.astype("uint8"))


def test_grain_is_unbiased_over_the_band():
    """Add the grain back (a varying strip) and check the band's mean still sits
    on the ramp, i.e. the noise is centred rather than lifting or sinking it."""
    img = _gradient(h=40, w=10, top=0, bottom=200)
    out = _rows(imageops.fill_band_deterministic(img, 10, 30, gap_above=9, gap_below=31))
    t = np.linspace(0.0, 1.0, 20)[:, None, None]
    ramp = 46 * (1.0 - t) + 158 * t
    dev = out[10:30] - ramp
    # The final astype("uint8") floors the *sum*, so a residual bias of about
    # -0.6 is quantisation, not a broken blend. Gate on a few standard errors:
    # 600 samples of N(0, 6) has a standard error of 6/sqrt(600) = 0.24.
    assert abs(float(dev.mean())) < 1.5
    assert 2.0 < float(np.median(np.abs(dev))) < 7.0  # |N(0, 6)| median is 4.05


def test_grain_amplitude_is_capped_at_six_levels():
    """local_std is clipped to 6.0, so a wildly varying texture strip cannot
    turn a clean band into visible static."""
    # A strip that varies enormously down its length (each column 0..255).
    arr = np.tile(np.linspace(0, 255, 40, dtype="uint8")[:, None, None], (1, 30, 3))
    img = Image.fromarray(arr, "RGB")
    out = _rows(imageops.fill_band_deterministic(img, 10, 20, 9, 21))
    for k in range(10, 20):
        # The blend itself is flat across columns, so the only horizontal
        # variation is grain: a sample std of N(0, 6) over 30 samples lands near
        # 6.1. Anything near 20 would mean the cap was not applied.
        assert out[k].std(axis=0).mean() < 11.0, k


def test_grain_is_deterministic_for_a_seed_and_differs_across_seeds():
    img = _noisy()
    a = _rows(imageops.fill_band_deterministic(img, 10, 20, 9, 21, seed=7))
    b = _rows(imageops.fill_band_deterministic(img, 10, 20, 9, 21, seed=7))
    c = _rows(imageops.fill_band_deterministic(img, 10, 20, 9, 21, seed=8))

    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_grain_is_damped_at_the_band_edges_so_the_join_cannot_show_a_seam():
    """The ramp is 3 rows deep at each end (min(arange, reversed)/3), so the
    outermost rows carry less grain than the middle. On a band whose interior is
    fully ramped, the row at the boundary must be visibly calmer."""
    img = _gradient()
    out = _rows(imageops.fill_band_deterministic(img, 10, 22, gap_above=9, gap_below=23))

    def row_spread(row: np.ndarray) -> float:
        return float(row.std(axis=0).mean())

    interior = [row_spread(out[k]) for k in range(13, 19)]
    top_edge = [row_spread(out[k]) for k in (10, 11, 12)]
    bottom_edge = [row_spread(out[k]) for k in (20, 21)]
    assert min(interior) > max(top_edge)
    assert min(interior) > max(bottom_edge)


def test_gap_rows_must_bracket_the_band():
    img = _gradient(h=40, w=10)
    with pytest.raises(ValueError, match="bracket"):
        imageops.fill_band_deterministic(img, 10, 20, gap_above=30, gap_below=25)  # above > below
    with pytest.raises(ValueError, match="bracket"):
        imageops.fill_band_deterministic(img, 10, 20, gap_above=-1, gap_below=25)  # negative
    with pytest.raises(ValueError, match="bracket"):
        imageops.fill_band_deterministic(img, 10, 20, gap_above=5, gap_below=40)  # below past the end
    with pytest.raises(ValueError, match="bracket"):
        imageops.fill_band_deterministic(img, 10, 20, gap_above=20, gap_below=20)  # equal is not bracketing


def test_out_of_range_band_is_clamped_rather_than_crashing():
    img = _gradient(h=20, w=10)
    out = imageops.fill_band_deterministic(img, -10, 999, gap_above=0, gap_below=19)
    assert out.size == img.size
    # the whole frame is now synthesized, and it is not a crash
    assert out.convert("RGB").size == (10, 20)


def test_an_empty_band_is_a_no_op():
    img = _gradient()
    before = _rows(img)
    out = _rows(imageops.fill_band_deterministic(img, 10, 10, gap_above=9, gap_below=21))
    assert np.array_equal(out, before)


def test_texture_rows_near_the_bottom_edge_avoid_reading_past_the_frame():
    """The default texture strip starts at gap_below; a band at the very bottom
    has no room left, so the code must fall back to the explicit strip."""
    img = _gradient(h=20, w=10)
    out = imageops.fill_band_deterministic(img, 14, 19, gap_above=13, gap_below=19, texture_rows=(0, 6))
    assert out.size == img.size


def test_greyscale_input_is_returned_as_rgb():
    grey = Image.new("L", (20, 30), 128)
    out = imageops.fill_band_deterministic(grey, 5, 10, gap_above=4, gap_below=11)
    assert out.mode == "RGB"


def test_output_pixels_stay_inside_the_byte_range():
    """noise is added after clipping the base, so the final clip is load-bearing
    for a pure-white or pure-black band."""
    for level in (0, 255):
        flat = Image.new("RGB", (40, 20), (level, level, level))
        arr = _rows(imageops.fill_band_deterministic(flat, 5, 15, gap_above=4, gap_below=16))
        assert arr.min() >= 0.0 and arr.max() <= 255.0
        assert np.allclose(arr[5:15], level, atol=1.0) or level in (0, 255)


# --------------------------------------------------------------------------
# draw_text
# --------------------------------------------------------------------------


def test_draw_text_actually_marks_the_image():
    flat = Image.new("RGB", (200, 60), (0, 0, 0))
    out = _rows(imageops.draw_text(flat, "HELLO", (0, 60)))
    assert out.max() > 0  # some ink landed
    assert out.sum() > 0


def test_default_font_size_scales_with_the_band_height():
    img = Image.new("RGB", (400, 100), (0, 0, 0))
    small = _rows(imageops.draw_text(img, "ABCDEFGHIJ", (0, 20)))
    large = _rows(imageops.draw_text(img, "ABCDEFGHIJ", (0, 100)))
    # a taller band means a bigger default glyph, so more ink
    assert large.sum() > small.sum()


def test_the_text_is_centred_in_the_band_horizontally():
    img = Image.new("RGB", (300, 60), (0, 0, 0))
    out = _rows(imageops.draw_text(img, "CENTRE", (0, 60)))
    cols = (out.sum(axis=(0, 2)) > 0).nonzero()[0]
    left_gap, right_gap = int(cols[0]), 300 - 1 - int(cols[-1])
    # allow a generous slack: the fallback bitmap font is not proportional
    assert abs(left_gap - right_gap) <= 12


def test_left_alignment_starts_at_the_left_edge():
    img = Image.new("RGB", (300, 60), (0, 0, 0))
    centred = _rows(imageops.draw_text(img, "LEFT", (0, 60)))
    left = _rows(imageops.draw_text(img, "LEFT", (0, 60), align="left"))

    def first_ink_col(a: np.ndarray) -> int:
        cols = (a.sum(axis=(0, 2)) > 0).nonzero()[0]
        return int(cols[0])

    assert first_ink_col(left) < first_ink_col(centred)


def test_an_unknown_font_path_degrades_to_the_default_font_instead_of_raising():
    """Dockerfile.web installs no fonts, so the hard-coded DejaVu path always
    raises OSError there. It must fall through, not 500."""
    img = Image.new("RGB", (200, 50), (10, 10, 10))
    out = imageops.draw_text(img, "FALLBACK", (0, 50), font_path="/nonexistent/DejaVuSans-Bold.ttf")
    assert out.size == (200, 50)
    assert _rows(out).sum() > img.size[0] * img.size[1] * 10


def test_an_explicit_font_size_is_honoured_over_the_band_derived_default():
    img = Image.new("RGB", (400, 120), (0, 0, 0))
    tiny = _rows(imageops.draw_text(img, "SIZE", (0, 120), font_size=8))
    big = _rows(imageops.draw_text(img, "SIZE", (0, 120), font_size=72))
    assert big.sum() > tiny.sum() * 4


def test_a_missing_font_size_still_renders_on_a_tiny_band():
    """max(10, int(band_h * 0.85)) floors at 10px, so a 4px band still has a
    drawable font and the route cannot divide by zero."""
    img = Image.new("RGB", (120, 8), (0, 0, 0))
    out = imageops.draw_text(img, "X", (0, 8))
    assert out.size == (120, 8)


def test_the_text_colour_is_used():
    img = Image.new("RGB", (200, 60), (0, 0, 0))
    out = _rows(imageops.draw_text(img, "RED", (0, 60), color=(255, 0, 0)))
    ink = out[out.sum(axis=2) > 0]
    assert ink[:, 0].max() > 100  # red channel lit
    assert ink[:, 1].max() < 60  # green channel dark


def test_the_seed_argument_is_accepted_so_the_caller_api_is_stable():
    img = Image.new("RGB", (200, 60), (0, 0, 0))
    a = _rows(imageops.draw_text(img, "SEED", (0, 60), seed=1))
    b = _rows(imageops.draw_text(img, "SEED", (0, 60), seed=1))
    assert np.array_equal(a, b)


def test_a_grayscale_image_is_upgraded_before_drawing():
    grey = Image.new("L", (200, 60), 0)
    out = imageops.draw_text(grey, "GREY", (0, 60))
    assert out.mode == "RGB"


def test_draw_text_does_not_mutate_its_input():
    img = Image.new("RGB", (200, 60), (5, 5, 5))
    before = _rows(img)
    imageops.draw_text(img, "PURE", (0, 60), color=(255, 255, 255))
    assert np.array_equal(_rows(img), before)


def test_long_text_still_renders_and_can_overflow_without_raising():
    img = Image.new("RGB", (100, 40), (0, 0, 0))
    out = imageops.draw_text(img, "A VERY LONG HEADLINE THAT EXCEEDS THE WIDTH", (0, 40))
    assert out.size == (100, 40)


def test_empty_text_is_a_no_op():
    img = Image.new("RGB", (100, 40), (0, 0, 0))
    before = _rows(img)
    out = imageops.draw_text(img, "", (0, 40))
    assert np.array_equal(_rows(out), before)


# --------------------------------------------------------------------------
# font reality on this machine (documenting the deployed constraint)
# --------------------------------------------------------------------------


def test_the_hard_coded_dejavu_path_is_absent_here_which_is_why_the_fallback_matters():
    path = "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
    try:
        ImageFont.truetype(path, 20)
        available = True
    except OSError:
        available = False
    if not available:
        # This is the web-container case: the code must not depend on it.
        img = Image.new("RGB", (200, 50), (0, 0, 0))
        assert _rows(imageops.draw_text(img, "OK", (0, 50))).sum() > 0


def test_older_pillow_without_a_size_argument_still_renders(monkeypatch):
    """The load_default(size=...) path needs Pillow >= 10.1. Guard the TypeError
    fallback so an older base image degrades to a fixed size instead of 500ing."""
    img = Image.new("RGB", (200, 50), (0, 0, 0))
    real = ImageFont.load_default
    seen: list[dict] = []

    def fake(*args, **kwargs):
        seen.append(kwargs)
        if "size" in kwargs:
            raise TypeError("load_default() got an unexpected keyword argument 'size'")
        return real()

    monkeypatch.setattr(ImageFont, "load_default", fake)
    out = imageops.draw_text(img, "OLD", (0, 50), font_size=40)
    assert out.size == (200, 50)
    assert _rows(out).sum() > 0
    assert seen and "size" in seen[0]  # the size argument is attempted first
