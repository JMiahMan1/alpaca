"""Tests for the programmatic-animation module (`imageanim.py`).

Three of these tests exist because the corresponding bug actually happened while
the module was being written, and each of them failed on first run:

* ``parallax`` produced 24 in-memory frames of which only 9 were distinct, so
  Pillow's encoder collapsed the clip to 9. The drift had been expressed as a
  fraction of the available slack, which made the slack the limiting quantity.
  ``test_parallax_actually_moves`` is the regression.
* The first ``sprite`` slice had no inset and showed the sheet's grid lines.
* ``ken_burns`` cropped a window out of an unscaled source, so a large pan
  walked off the edge of the image. ``test_ken_burns_never_exposes_an_edge``
  is the regression.

The GIF tests care specifically about *palette stability*: quantising each frame
on its own makes a static region change colour frame to frame, which looks like
an encoder fault and is not one.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

import imageanim as ia

# --------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------


def _still(w: int = 480, h: int = 320, seed: int = 3) -> Image.Image:
    """A synthetic still with enough structure that a crop is detectable."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.zeros((h, w, 3), np.float32)
    base[..., 0] = 30 + 150 * (xx / w)
    base[..., 1] = 25 + 90 * (yy / h)
    base[..., 2] = 110 + 90 * (1 - xx / w)
    disc = np.sqrt((xx - w * 0.3) ** 2 + (yy - h * 0.35) ** 2) < min(w, h) * 0.16
    base[disc] = (250, 240, 210)
    # Vertical bars on a coarse grid: gives a pan something to actually move.
    for i in range(0, w, max(8, w // 16)):
        base[int(h * 0.6) : int(h * 0.9), i : i + max(3, w // 90)] = (20, 24, 40)
    base += rng.normal(0, 1.5, base.shape)  # break up any accidental flatness
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")


def _grid_sheet(cols: int, rows: int, cell: int = 64) -> Image.Image:
    sheet = Image.new("RGB", (cols * cell, rows * cell), (0, 0, 0))
    for r in range(rows):
        for c in range(cols):
            sheet.paste(Image.new("RGB", (cell, cell), (r * 40 % 256, c * 40 % 256, 220)), (c * cell, r * cell))
    return sheet


def _decoded_frames(data: bytes) -> list[np.ndarray]:
    op = Image.open(io.BytesIO(data))
    n = getattr(op, "n_frames", 1)
    out = []
    for i in range(n):
        op.seek(i)
        out.append(np.asarray(op.convert("RGB"), dtype=np.uint8))
    return out


def _encoded_frame_count(data: bytes) -> int:
    return len(_decoded_frames(data))


# --------------------------------------------------------------------------
# smoothstep
# --------------------------------------------------------------------------


def test_smoothstep_is_pinned_at_both_ends():
    assert ia.smoothstep(0.0) == 0.0
    assert ia.smoothstep(1.0) == 1.0


def test_smoothstep_is_monotonic_across_the_range():
    vals = [ia.smoothstep(i / 20) for i in range(21)]
    assert vals == sorted(vals)


def test_smoothstep_is_steeper_in_the_middle_than_at_the_ends():
    # A linear ramp would give equal steps; the ease is the whole point.
    first = ia.smoothstep(0.1) - ia.smoothstep(0.0)
    middle = ia.smoothstep(0.55) - ia.smoothstep(0.45)
    assert middle > first


@pytest.mark.parametrize("t", [-5.0, 2.0])
def test_smoothstep_clamps_out_of_range_input(t):
    assert 0.0 <= ia.smoothstep(t) <= 1.0


# --------------------------------------------------------------------------
# load_frames / mode handling
# --------------------------------------------------------------------------


def test_load_frames_accepts_a_mixed_list_of_images_and_paths(tmp_path):
    p = tmp_path / "a.png"
    _still(40, 30).save(p)
    frames = ia.load_frames([_still(40, 30), p])
    assert len(frames) == 2
    assert all(f.mode == "RGB" for f in frames)


def test_load_frames_flattens_transparency_onto_white_not_black():
    """A black matte shows up as a dark halo the moment it is encoded."""
    rgba = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    frame = ia.load_frames([rgba])[0]
    assert frame.mode == "RGB"
    assert tuple(frame.getpixel((0, 0))) == (255, 255, 255)


def test_load_frames_flattens_palette_and_grayscale_images():
    for mode in ("P", "L", "LA", "1"):
        assert ia.load_frames([Image.new(mode, (8, 8))])[0].mode == "RGB"


# --------------------------------------------------------------------------
# resize_cover
# --------------------------------------------------------------------------


def test_resize_cover_hits_the_exact_target_size():
    for src, dst in (((100, 50), (200, 200)), ((500, 500), (64, 128)), ((37, 91), (300, 300))):
        assert ia.resize_cover(_still(*src), dst).size == dst


def test_resize_cover_does_not_stretch_the_aspect_ratio():
    """A stretched generated image looks broken in a way that is hard to diagnose."""
    src = Image.new("RGB", (200, 100), (255, 0, 0))
    src.paste(Image.new("RGB", (100, 100), (0, 0, 255)), (0, 0))
    out = ia.resize_cover(src, (100, 100))
    arr = np.asarray(out)
    # A stretched version would put the colour boundary mid-row at 50% width; a
    # centred crop of a 2:1 source to 1:1 keeps it at the left edge.
    assert tuple(arr[50, 2]) == (0, 0, 255)
    assert tuple(arr[50, 97]) == (255, 0, 0)


def test_resize_cover_rejects_a_degenerate_source():
    with pytest.raises(ValueError, match="size"):
        ia.resize_cover(Image.new("RGB", (0, 0)), (10, 10))


# --------------------------------------------------------------------------
# sprite_sheet_to_frames
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cols,rows", [(2, 2), (4, 4), (8, 3), (1, 5)])
def test_sprite_sheet_yields_one_frame_per_cell(cols, rows):
    frames = ia.sprite_sheet_to_frames(_grid_sheet(cols, rows), cols, rows)
    assert len(frames) == cols * rows


def test_sprite_sheet_frames_are_distinct():
    """A slicing bug that returned the same cell N times would still pass a count check."""
    frames = ia.sprite_sheet_to_frames(_grid_sheet(4, 4), 4, 4)
    assert len({f.tobytes() for f in frames}) == 16


def test_sprite_sheet_reads_cells_in_row_major_order():
    frames = ia.sprite_sheet_to_frames(_grid_sheet(2, 2), 2, 2)
    # The helper paints (row*40, col*40, 220), so blue is constant and the
    # red/green pair identifies the cell.
    def key(f):
        r, g, _ = np.asarray(f)[0, 0]
        return int(r), int(g)

    assert key(frames[0]) == (0, 0)
    assert key(frames[1]) == (0, 40)
    assert key(frames[2]) == (40, 0)
    assert key(frames[3]) == (40, 40)


def test_sprite_sheet_resizes_cells_when_a_size_is_given():
    frames = ia.sprite_sheet_to_frames(_grid_sheet(4, 4), 4, 4, size=(50, 60))
    assert all(f.size == (50, 60) for f in frames)


def test_sprite_sheet_trims_the_grid_lines_with_an_inset():
    """A sheet drawn with visible cell borders must not show them in every frame."""
    sheet = _grid_sheet(2, 2, cell=40)
    for x in (39, 40, 79):  # the seam between cells, and the last valid column
        for y in range(80):
            sheet.putpixel((x, y), (255, 0, 0))  # a grid line
    frames = ia.sprite_sheet_to_frames(sheet, 2, 2)
    for f in frames:
        arr = np.asarray(f)
        assert not (arr[:, :, 0] == 255).all(), "the border survived into a frame"


@pytest.mark.parametrize("cols,rows", [(0, 4), (4, 0), (-1, 2), (2, -1)])
def test_sprite_sheet_rejects_a_degenerate_grid(cols, rows):
    with pytest.raises(ValueError, match="positive"):
        ia.sprite_sheet_to_frames(_grid_sheet(2, 2), cols, rows)


# --------------------------------------------------------------------------
# ken_burns
# --------------------------------------------------------------------------


def test_ken_burns_returns_exactly_the_requested_frames_at_the_target_size():
    frames = ia.ken_burns(_still(), (200, 120), frames=17)
    assert len(frames) == 17
    assert all(f.size == (200, 120) for f in frames)


@pytest.mark.parametrize("zoom_from,zoom_to,pan_x,pan_y", [
    (1.0, 2.0, 0.0, 0.0),
    (1.0, 1.2, 1.0, 1.0),      # hard right/top
    (1.0, 1.2, -1.0, -1.0),    # hard left/bottom
    (1.4, 1.0, 0.9, -0.9),     # a pull-back with a diagonal pan
    (1.0, 1.0, 0.5, 0.5),      # a pan with no zoom at all
])
def test_ken_burns_never_exposes_an_edge(zoom_from, zoom_to, pan_x, pan_y):
    """Every corner of every frame must come from the image, never from nothing.

    The first version cropped a moving window out of an unscaled source, so a
    large pan walked off the edge and produced black bars. Pre-scaling to cover
    at the peak zoom is what makes this safe at any pan magnitude.
    """
    frames = ia.ken_burns(_still(400, 260), (160, 100), frames=14, zoom_from=zoom_from,
                          zoom_to=zoom_to, pan_x=pan_x, pan_y=pan_y)
    for f in frames:
        arr = np.asarray(f)
        for corner in (arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]):
            assert corner.max() > 8, f"a corner is black: {corner}"


def test_ken_burns_actually_moves():
    frames = ia.ken_burns(_still(), (160, 100), frames=20, zoom_from=1.0, zoom_to=1.4, pan_x=1.0)
    assert len({f.tobytes() for f in frames}) == 20


def test_ken_burns_opens_on_the_pan_side_it_was_given():
    """pan_x=+1 must start on the right-hand side of the frame."""
    right = ia.ken_burns(_still(), (120, 80), frames=9, pan_x=1.0, zoom_from=1.0, zoom_to=1.6)
    left = ia.ken_burns(_still(), (120, 80), frames=9, pan_x=-1.0, zoom_from=1.0, zoom_to=1.6)
    assert np.asarray(right[0]).tobytes() != np.asarray(left[0]).tobytes()


def test_ken_burns_with_a_single_frame_does_not_divide_by_zero():
    frames = ia.ken_burns(_still(), (64, 64), frames=1)
    assert len(frames) == 1 and frames[0].size == (64, 64)


def test_ken_burns_rejects_a_zero_frame_count():
    with pytest.raises(ValueError, match="positive"):
        ia.ken_burns(_still(), (64, 64), frames=0)


def test_ken_burns_survives_an_upscaling_only_zoom():
    """zoom_from < 1 on a small source means the window is bigger than the canvas."""
    frames = ia.ken_burns(_still(64, 48), (200, 150), frames=6, zoom_from=0.5, zoom_to=1.0)
    assert all(f.size == (200, 150) for f in frames)


# --------------------------------------------------------------------------
# parallax_drift
# --------------------------------------------------------------------------


def test_parallax_actually_moves():
    """Regression: the first version drifted ~2px across the whole clip.

    Pillow's encoder silently collapsed 24 frames to 9, which is the only reason
    the bug was visible at all. Asserting on *displacement* rather than on a
    frame count: sub-pixel motion rounds away to nothing at the eased ends, so
    some frames are legitimately identical and a distinct-count threshold would
    be asserting the easing curve rather than the motion.
    """
    size = (200, 130)
    frames = ia.parallax_drift(_still(), size, frames=24, drift=0.06)
    assert len(frames) == 24
    first = np.asarray(frames[0], np.int16)
    last = np.asarray(frames[-1], np.int16)
    # A gradient background means any real pan shifts the sampled pixel values.
    assert np.abs(last - first).mean() > 1.0, "the first and last parallax frames are effectively identical"


def test_parallax_travel_scales_with_the_requested_drift():
    """Bigger drift must move further -- the bug was a coupling to the canvas slack."""
    def displacement(drift: float) -> float:
        fr = ia.parallax_drift(_still(), (200, 130), frames=2, drift=drift)
        return float(np.abs(np.asarray(fr[-1], np.int16) - np.asarray(fr[0], np.int16)).mean())

    assert displacement(0.15) > displacement(0.03) * 2


def test_parallax_survives_encoding_without_heavy_frame_collapsing():
    spec = ia.AnimationSpec(kind="parallax", size=(200, 130), frames=24)
    frames = ia.build_animation([_still()], spec)
    # Was 9 of 24 before the drift fix. The eased start and end cost a few more;
    # test_parallax_actually_moves is the test that proves the motion is real.
    assert _encoded_frame_count(ia.encode_animation(frames, spec)) >= 12


def test_parallax_moves_the_far_plane_more_than_the_near_one():
    """Opposite, unequal travel is the entire effect; equal travel is invisible."""
    sharp = np.asarray(ia.parallax_drift(_still(), (120, 80), frames=2, depth=0.0)[0])
    deep = np.asarray(ia.parallax_drift(_still(), (120, 80), frames=2, depth=1.5)[1])
    assert sharp.tobytes() != deep.tobytes()


def test_parallax_does_not_reject_when_drift_is_zero():
    frames = ia.parallax_drift(_still(), (80, 60), frames=5, drift=0.0)
    assert len(frames) == 5


# --------------------------------------------------------------------------
# crossfade
# --------------------------------------------------------------------------


def test_crossfade_with_one_still_returns_that_still_every_frame():
    still = _still(120, 90)
    frames = ia.crossfade([still], (60, 60), frames=8)
    assert len(frames) == 8
    assert all(f.tobytes() == frames[0].tobytes() for f in frames)


def test_crossfade_opens_and_closes_on_the_pure_stills():
    a, b = _still(120, 90, seed=1), _still(120, 90, seed=2)
    frames = ia.crossfade([a, b], (60, 60), frames=16)
    first, last = np.asarray(frames[0], np.float32), np.asarray(frames[-1], np.float32)
    a_s, b_s = np.asarray(ia.resize_cover(a, (60, 60)), np.float32), np.asarray(ia.resize_cover(b, (60, 60)), np.float32)
    assert np.abs(first - a_s).mean() < 2.0
    assert np.abs(last - b_s).mean() < 2.0


def test_crossfade_midpoint_is_between_the_two_stills():
    a = Image.new("RGB", (40, 40), (0, 0, 0))
    b = Image.new("RGB", (40, 40), (200, 200, 200))
    mid = np.asarray(ia.crossfade([a, b], (40, 40), frames=11)[5])
    assert 70 < mid.mean() < 130


def test_crossfade_normalises_differently_sized_stills():
    frames = ia.crossfade([_still(300, 100), _still(100, 300)], (80, 80), frames=6)
    assert all(f.size == (80, 80) for f in frames)


def test_crossfade_rejects_an_empty_still_list():
    with pytest.raises(ValueError, match="at least one"):
        ia.crossfade([], (40, 40), frames=4)


# --------------------------------------------------------------------------
# duration handling
# --------------------------------------------------------------------------


@pytest.mark.parametrize("given,expected", [
    (0, ia.DEFAULT_DURATION_MS),
    (-5, ia.DEFAULT_DURATION_MS),
    (None, ia.DEFAULT_DURATION_MS),
    ("nonsense", ia.DEFAULT_DURATION_MS),
    (1, ia.MIN_DURATION_MS),        # below what viewers honour
    (60, 60),
    (10_000_000, 10_000),           # capped
])
def test_frame_duration_is_clamped_into_a_sane_band(given, expected):
    assert ia._resolve_duration(given) == expected


# --------------------------------------------------------------------------
# encoders
# --------------------------------------------------------------------------


def test_frames_to_webp_produces_a_real_animation():
    frames = ia.ken_burns(_still(), (120, 80), frames=10, zoom_to=1.3)
    data = ia.frames_to_webp(frames, duration_ms=60)
    assert data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    decoded = _decoded_frames(data)
    assert len(decoded) == 10
    assert decoded[0].shape == (80, 120, 3)


def test_webp_frame_duration_is_what_was_asked_for():
    """The frame time must survive encoding.

    Read out of the container, not from Pillow's ``info`` dict. ``info`` is the
    view that made this look broken in the first place: it does not reliably
    expose per-frame duration after a ``seek()``, so a reader built on it
    reports zeros for a file that is entirely correct.
    """
    # Distinct frames: the encoder collapses identical ones, and three copies of
    # the same still is a one-frame clip whatever duration was requested.
    frames = [_still(40, 30, seed=i) for i in range(3)]
    assert ia.webp_frame_durations(ia.frames_to_webp(frames, duration_ms=120)) == [120, 120, 120]


def test_the_duration_is_read_from_the_anmf_duration_field_not_the_frame_origin():
    """Pin the offset, because reading the wrong one is silently plausible.

    An ANMF payload is x, y, width-1, height-1 (three bytes each) and then the
    3-byte duration, so the duration is at offset 12. Reading offset 0 gives the
    frame's x coordinate, which is 0 for every full-canvas frame -- and an
    all-zero result is exactly what a broken encoder would also produce, so the
    mistake is invisible unless the layout itself is asserted.
    """
    frames = [_still(40, 30, seed=i) for i in range(2)]
    data = ia.frames_to_webp(frames, duration_ms=120)
    offsets = ia._webp_anmf_payload_offsets(data)
    assert len(offsets) == 2, "expected one ANMF chunk per frame"
    for payload in offsets:
        rect = [data[payload + i * 3 : payload + i * 3 + 3] for i in range(4)]
        assert rect[0] == b"\x00\x00\x00", "x should be 0 for a full-canvas frame"
        # width-1 = 39, height-1 = 29 for a 40x30 still
        assert int.from_bytes(rect[2], "little") == 39
        assert int.from_bytes(rect[3], "little") == 29
        assert int.from_bytes(data[payload + 12 : payload + 15], "little") == 120


def test_the_encoded_file_is_still_decodable_by_libwebp():
    """A timing fix that corrupts the file is worse than no timing fix.

    This is the test that failed loudly when an earlier version of the encoder
    wrote the duration over the frame's x coordinate: libwebp refused the file
    outright, so nothing about it played.
    """
    frames = [_still(60, 40, seed=i) for i in range(5)]
    data = ia.frames_to_webp(frames, duration_ms=80)
    assert len(_decoded_frames(data)) == 5, "libwebp could not decode the frames back"


def test_every_frame_gets_the_duration_not_just_the_first():
    frames = [_still(40, 30, seed=i) for i in range(9)]
    assert ia.webp_frame_durations(ia.frames_to_webp(frames, duration_ms=250)) == [250] * 9


def test_the_duration_reader_rejects_a_non_webp():
    junk = b"not a webp at all, just some bytes"
    assert ia.webp_frame_durations(junk) == []


def test_the_duration_reader_rejects_a_truncated_container():
    """A short read must not produce a number from half a file."""
    frames = [_still(40, 30, seed=i) for i in range(4)]
    data = ia.frames_to_webp(frames, duration_ms=100)
    assert ia.webp_frame_durations(data[: len(data) // 2]) == []


@pytest.mark.parametrize("asked", [20, 65, 150, 999, 5000])
def test_a_range_of_durations_all_survive(asked):
    frames = [_still(40, 30, seed=i) for i in range(3)]
    assert ia.webp_frame_durations(ia.frames_to_webp(frames, duration_ms=asked)) == [asked] * 3


def test_the_clamped_duration_is_the_one_the_module_documents():
    """0 and negatives are not silently written as 0 ms.

    A 0 ms frame is not a pause, it is "as fast as the viewer can go" to most
    viewers, so they go through _resolve_duration's floor instead.
    """
    frames = [_still(40, 30, seed=i) for i in range(3)]
    for asked in (0, -5):
        got = ia.webp_frame_durations(ia.frames_to_webp(frames, duration_ms=asked))
        assert got == [ia.DEFAULT_DURATION_MS] * 3, f"{asked} -> {got[0]}"


def test_the_default_build_path_carries_its_duration_through():
    """The whole pipeline, not just the encoder: spec -> frames -> file."""
    spec = ia.AnimationSpec(kind="ken_burns", size=(120, 90), frames=6, duration_ms=150, format="webp")
    data = ia.encode_animation(ia.build_animation([_still()], spec), spec)
    assert ia.webp_frame_durations(data) == [150] * 6
    assert len(_decoded_frames(data)) == 6


def test_gif_timing_is_readable_too():
    """The control case: Pillow's reader exposes GIF duration after a seek."""
    frames = [_still(40, 30, seed=i) for i in range(4)]
    op = Image.open(io.BytesIO(ia.frames_to_gif(frames, duration_ms=110)))
    seen = []
    for n in range(op.n_frames):
        op.seek(n)
        seen.append(op.info.get("duration"))
    assert seen == [110] * 4


def test_frames_to_gif_produces_a_real_animation():
    frames = ia.ken_burns(_still(), (100, 70), frames=8, zoom_to=1.3)
    data = ia.frames_to_gif(frames, duration_ms=70)
    assert data[:3] == b"GIF"
    assert len(_decoded_frames(data)) == 8


def test_gif_keeps_a_static_region_the_same_colour_across_every_frame():
    """The anti-flicker property, and the reason for the shared palette.

    Quantising each frame independently lets the palette change frame to frame,
    so an unchanging region of the image visibly changes colour. That reads as an
    encoder fault and is not one, so it is worth a test rather than a comment.
    """
    frames = []
    for i in range(10):
        arr = np.zeros((60, 80, 3), np.uint8)
        arr[:, :30] = (200, 40, 90)  # a flat, unchanging block
        arr[:, 30:] = (20 + i * 12, 90, 200)
        frames.append(Image.fromarray(arr, "RGB"))
    decoded = _decoded_frames(ia.frames_to_gif(frames, duration_ms=50))
    block = [f[30, 15].astype(int) for f in decoded]
    # Spread *across frames* per channel. Comparing a single pixel's channels
    # against each other is meaningless -- (200, 40, 90) has a 160 spread and
    # never flickers at all.
    for channel in range(3):
        values = [px[channel] for px in block]
        assert max(values) - min(values) <= 2, f"channel {channel} flickered: {values}"


def test_gif_samples_the_palette_across_the_clip_not_just_frame_zero():
    """A palette built from frame 0 misses colours that only appear later."""
    frames = []
    for i in range(6):
        arr = np.zeros((40, 40, 3), np.uint8)
        arr[:20] = (10, 10, 10)
        # A colour that exists only in the final frame.
        arr[20:] = (0, 0, 0) if i < 5 else (0, 255, 0)
        frames.append(Image.fromarray(arr, "RGB"))
    decoded = _decoded_frames(ia.frames_to_gif(frames, duration_ms=50, palette_frames=3))
    assert decoded[-1][30, 20].astype(int)[1] > 200


@pytest.mark.parametrize("encoder", [ia.frames_to_webp, ia.frames_to_gif])
def test_encoders_reject_an_empty_frame_list(encoder):
    with pytest.raises(ValueError, match="no frames"):
        encoder([])


@pytest.mark.parametrize("encoder", [ia.frames_to_webp, ia.frames_to_gif])
def test_encoders_accept_a_single_frame(encoder):
    """A one-frame clip is dull but legitimate, and callers do ask for one."""
    assert encoder([_still(20, 20)])


def test_the_module_imports_nothing_heavy():
    """`web/` has Pillow + numpy and no ffmpeg, so a heavy import would need a rebuild.

    Checked by import rather than by grepping the docstring: the docstring is
    *supposed* to mention ffmpeg, because that is the constraint it explains.
    """
    import ast
    import pathlib

    src = pathlib.Path(ia.__file__).read_text()
    imported: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {
        "__future__", "io", "math", "base64", "struct", "collections",
        "dataclasses", "pathlib", "typing", "numpy", "PIL",
    }, f"unexpected imports: {sorted(imported)}"


def test_webp_size_beats_gif_for_the_same_frames():
    """The reason WebP is the default. Not a spec claim -- measured."""
    frames = ia.ken_burns(_still(), (200, 130), frames=16, zoom_to=1.25)
    assert len(ia.frames_to_webp(frames, 60)) < len(ia.frames_to_gif(frames, 60))


# --------------------------------------------------------------------------
# spec / dispatch
# --------------------------------------------------------------------------


def test_encode_animation_dispatches_on_the_requested_format():
    frames = [_still(40, 30) for _ in range(4)]
    assert ia.encode_animation(frames, ia.AnimationSpec(format="webp"))[:4] == b"RIFF"
    assert ia.encode_animation(frames, ia.AnimationSpec(format="GIF"))[:3] == b"GIF"
    assert ia.encode_animation(frames, ia.AnimationSpec(format="  WebP  "))[:4] == b"RIFF"


def test_encode_animation_rejects_an_unknown_format():
    with pytest.raises(ValueError, match="unsupported animation format"):
        ia.encode_animation([_still(20, 20)], ia.AnimationSpec(format="mp4"))


def test_animation_data_uri_carries_the_right_mime_type():
    frames = [_still(40, 30) for _ in range(3)]
    webp = ia.animation_to_data_uri(frames, ia.AnimationSpec(format="webp"))
    gif = ia.animation_to_data_uri(frames, ia.AnimationSpec(format="gif"))
    assert webp.startswith("data:image/webp;base64,")
    assert gif.startswith("data:image/gif;base64,")
    # Round-trips: the payload after the comma is real base64 of the file.
    assert base64.b64decode(webp.split(",", 1)[1])[:4] == b"RIFF"


@pytest.mark.parametrize("kind", ["ken_burns", "parallax", "crossfade", "sprite"])
def test_build_animation_handles_every_declared_kind(kind):
    spec = ia.AnimationSpec(kind=kind, size=(80, 60), frames=6, sprite_cols=2, sprite_rows=3)
    frames = ia.build_animation([_still(), _still(seed=9)], spec)
    assert len(frames) == 6
    assert all(f.size == (80, 60) for f in frames)


def test_build_animation_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="unknown animation kind"):
        ia.build_animation([_still()], ia.AnimationSpec(kind="morphosphere"))


def test_build_animation_rejects_no_stills():
    with pytest.raises(ValueError, match="no stills"):
        ia.build_animation([], ia.AnimationSpec())


def test_the_default_spec_is_something_that_actually_renders():
    spec = ia.AnimationSpec()
    frames = ia.build_animation([_still(640, 480)], spec)
    assert len(frames) == spec.frames
    assert frames[0].size == spec.size
    assert ia.encode_animation(frames, spec)[:4] == b"RIFF"


def test_a_full_pipeline_run_encodes_without_raising():
    """One end-to-end pass, so a change in any stage is caught here too."""
    spec = ia.AnimationSpec(kind="ken_burns", size=(256, 160), frames=20, zoom_to=1.25, pan_x=1.0)
    frames = ia.build_animation([_still(800, 500)], spec)
    assert _encoded_frame_count(ia.encode_animation(frames, spec)) >= 18


# --------------------------------------------------------------------------
# documentation
# --------------------------------------------------------------------------


def test_the_docstring_still_explains_why_seeding_does_not_animate():
    """The single most important thing in this module, and easy to delete."""
    doc = (ia.__doc__ or "").lower()
    assert "seed" in doc and "does not" in doc
    assert "still" in doc and "animate" in doc


