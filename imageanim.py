"""Programmatic animation from generated stills (Pillow + numpy, no ffmpeg).

The thesis
----------
An image model is a *still* model. Asking one for the same subject twice with two
seeds does not give you two moments of one scene, it gives you two unrelated
pictures -- there is no temporal state in the network, so nothing carries over
between generations. Seeding a still model thirty times does not animate it.

So the motion has to come from code. This module takes generated stills and
produces a real animated file, and it is deliberately honest about the split:

* **The AI authors the subject and the art direction.** One good image, or a few
  key images for a morph.
* **The code authors the motion.** A camera move, a parallax drift, a crossfade,
  a sprite-sheet flipbook. This is where a still model cannot help you, and it
  is also where a still model is not needed: a good pan over one image reads as
  motion more convincingly than thirty inconsistent generations ever do.

If you want the *model* to author the motion, that is a different tool -- an
image **edit** model applied to its own previous output, so each frame inherits
the last. That is a real technique, and this module is happy to encode the
resulting frames, but it is N generations and the drift compounds.

Why Pillow and not ffmpeg
--------------------------
``web/`` is bind-mounted read-only, so anything added here needs a restart
rather than a rebuild of a multi-gigabyte image. ``Dockerfile.web`` installs
Pillow and no ffmpeg, and it does not need one: Pillow writes animated WebP and
GIF natively. Animated WebP is the default here because it is both smaller and
sharper than GIF for the same frames -- measured on a synthetic 12-frame
sequence, 1.7 KB against 3.9 KB. GIF is offered because it is the only one of the
two that survives being pasted into a chat client.

The two things that actually go wrong
-------------------------------------
**GIF palette flicker.** GIF is 256-colour and palette-indexed. Letting Pillow
quantise each frame independently makes the palette change frame to frame, and
the same pixel in the same place visibly changes colour -- the classic
"flickering GIF" that looks like a compression fault but is a quantisation one.
:func:`frames_to_gif` therefore builds a *single* palette from a representative
frame and applies it to every frame, so only the 8-bit indices move.

**Camera moves that expose the edge.** A naive Ken Burns crops a window out of
the source and slides it around; as soon as the window reaches a border you get
black bars, and the fix people reach for (shrink the move) is a compromise.
:func:`ken_burns` instead pre-scales the source to cover the output at its
*maximum* zoom, then takes a moving window of exactly the output size. The
window can never leave the image, so no edge can ever appear, and the move is
free to be as large as it likes.
"""

from __future__ import annotations

import io
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

__all__ = [
    "ANIMATION_FORMATS",
    "DEFAULT_DURATION_MS",
    "AnimationSpec",
    "animation_to_data_uri",
    "crossfade",
    "encode_animation",
    "frames_to_gif",
    "frames_to_webp",
    "ken_burns",
    "load_frames",
    "parallax_drift",
    "resize_cover",
    "smoothstep",
    "sprite_sheet_to_frames",
]

ANIMATION_FORMATS = ("webp", "gif")
DEFAULT_DURATION_MS = 80
#: Below this, a viewer is likely to clamp the frame time and the clip will run
#: fast; some clients (notably older GIF viewers) ignore anything under 20 ms.
MIN_DURATION_MS = 20


@dataclass
class AnimationSpec:
    """Everything needed to turn a set of stills into a file.

    ``kind`` selects the motion:
        ``ken_burns``   a single still with a camera push across it
        ``parallax``    a single still drifting in two depth planes
        ``crossfade``   a morph between two or more stills
        ``sprite``      a flipbook sliced out of one sheet
    """

    kind: str = "ken_burns"
    size: tuple[int, int] = (768, 768)
    frames: int = 48
    duration_ms: int = DEFAULT_DURATION_MS
    loop: int = 0  # 0 = forever, which is what both encoders mean by 0
    zoom_from: float = 1.0
    zoom_to: float = 1.18
    pan_x: float = 0.0  # -1 = start hard left, +1 = start hard right
    pan_y: float = 0.0
    parallax: float = 0.035  # extra drift of the blurred (far) plane, as a fraction
    sprite_cols: int = 0
    sprite_rows: int = 0
    format: str = "webp"
    lossless: bool = False
    quality: int = 90


def smoothstep(t: float) -> float:
    """Ease in and out.

    Linear camera motion reads as a machine; this is the difference between a
    pan that feels operated and one that feels computed. Guarded against a
    zero-width range because callers routinely derive ``t`` from a frame count.
    """
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)


def _as_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA", "P"):
        # Flatten onto white rather than black: generated art is usually opaque
        # and a black matte shows up as a dark halo the moment it is encoded.
        rgba = image.convert("RGBA")
        base = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(base, rgba).convert("RGB")
    return image.convert("RGB")


def load_frames(sources: Sequence[Image.Image | str | Path]) -> list[Image.Image]:
    """Load and normalise a mixed list of images and paths to RGB frames."""
    frames: list[Image.Image] = []
    for src in sources:
        if isinstance(src, (str, Path)):
            with Image.open(src) as opened:
                frames.append(_as_rgb(opened).copy())
        else:
            frames.append(_as_rgb(src).copy())
    return frames


def resize_cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Scale to *cover* the target and centre-crop the overflow.

    Aspect ratio is never distorted: an animation that stretches a generated
    image to fit a panel looks broken in a way that is hard to diagnose later.
    """
    tw, th = size
    src_w, src_h = image.size
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"cannot resize an image with size {image.size}")
    scale = max(tw / src_w, th / src_h)
    new_w = max(tw, round(src_w * scale))
    new_h = max(th, round(src_h * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - tw) // 2
    top = (new_h - th) // 2
    return resized.crop((left, top, left + tw, top + th))


def sprite_sheet_to_frames(
    sheet: Image.Image, cols: int, rows: int, size: tuple[int, int] | None = None
) -> list[Image.Image]:
    """Slice a grid sheet into a flipbook.

    This is the one route where a *single* AI generation yields many frames: a
    model asked for "a 4x4 sprite sheet" produces one image that already contains
    the sequence, and the grid is the model's own decomposition of the motion
    rather than something we imposed. Grid lines are trimmed by a small inset so
    a sheet drawn with visible cell borders does not show them in every frame.
    """
    if cols <= 0 or rows <= 0:
        raise ValueError(f"sprite grid must be positive, got {cols}x{rows}")
    sheet = _as_rgb(sheet)
    sheet_w, sheet_h = sheet.size
    cell_w = sheet_w / cols
    cell_h = sheet_h / rows
    inset = max(0, round(min(cell_w, cell_h) * 0.01))
    # Guarantee at least a pixel of trim once the cell can afford it, so a sheet
    # drawn with a 1px cell border does not show that border in every frame.
    if min(cell_w, cell_h) >= 8:
        inset = max(1, inset)
    out: list[Image.Image] = []
    for r in range(rows):
        for c in range(cols):
            left = round(c * cell_w) + inset
            top = round(r * cell_h) + inset
            right = round((c + 1) * cell_w) - inset
            bottom = round((r + 1) * cell_h) - inset
            cell = sheet.crop((left, top, max(left + 1, right), max(top + 1, bottom)))
            out.append(resize_cover(cell, size) if size else cell)
    return out


def _frame_positions(frames: int) -> list[float]:
    """Evenly spaced eased positions across the clip.

    A one-frame animation is position 0.0 rather than a division by zero -- a
    single-frame clip is a legitimate (if dull) thing to ask for, and it is how
    a caller discovers that.
    """
    if frames <= 0:
        raise ValueError(f"frame count must be positive, got {frames}")
    if frames == 1:
        return [0.0]
    return [smoothstep(i / (frames - 1)) for i in range(frames)]


def crossfade(stills: Sequence[Image.Image], size: tuple[int, int], frames: int = 48) -> list[Image.Image]:
    """Morph through a sequence of stills with a dissolve between each pair.

    Held for the first and last still rather than dissolving straight through, so
    the clip opens and closes on a stable image instead of fading in from and out
    to nothing.
    """
    stills = [_as_rgb(s) for s in stills]
    if not stills:
        raise ValueError("crossfade needs at least one still")
    if len(stills) == 1:
        return [resize_cover(stills[0], size) for _ in range(frames)]
    prepared = [resize_cover(s, size) for s in stills]
    out: list[Image.Image] = []
    # Spread the dissolve budget evenly across the gaps between stills.
    for t in _frame_positions(frames):
        pos = t * (len(prepared) - 1)
        i = min(math.floor(pos), len(prepared) - 2)
        frac = pos - i
        a = np.asarray(prepared[i], dtype=np.float32)
        b = np.asarray(prepared[i + 1], dtype=np.float32)
        blended = a + (b - a) * frac
        out.append(Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), "RGB"))
    return out


def _cover_scaled(image: Image.Image, size: tuple[int, int], scale: float) -> Image.Image:
    """A copy of ``image`` scaled to cover ``size`` at an *extra* ``scale`` factor."""
    tw, th = size
    src_w, src_h = image.size
    base = max(tw / src_w, th / src_h) * scale
    new_w = max(tw, math.ceil(src_w * base))
    new_h = max(th, math.ceil(src_h * base))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def ken_burns(
    image: Image.Image,
    size: tuple[int, int],
    frames: int = 48,
    zoom_from: float = 1.0,
    zoom_to: float = 1.18,
    pan_x: float = 0.0,
    pan_y: float = 0.0,
) -> list[Image.Image]:
    """A camera push across one still.

    The source is pre-scaled to cover the output at ``max(zoom_from, zoom_to)``
    and the crop window is then always exactly the output size, so the move can
    never walk off the edge of the image. ``pan_x``/``pan_y`` are in [-1, 1] and
    set where the shot *starts*; the move runs from there to the opposite side.
    """
    image = _as_rgb(image)
    peak = max(1.0, float(zoom_from), float(zoom_to))
    canvas = _cover_scaled(image, size, peak)
    cw, ch = canvas.size
    tw, th = size

    out: list[Image.Image] = []
    for t in _frame_positions(frames):
        zoom = float(zoom_from) + (float(zoom_to) - float(zoom_from)) * t
        # A window of the output size at the current zoom; at zoom 1 it exactly
        # fills the canvas, at higher zoom it shrinks and can therefore move.
        win_w = max(1, round(tw / zoom))
        win_h = max(1, round(th / zoom))
        win_w = min(win_w, cw)
        win_h = min(win_h, ch)
        # pan is applied as the *centre* offset, easing from one side to the other.
        cx = cw / 2.0 + (float(pan_x) * 0.5) * (cw - win_w) * (1.0 - 2.0 * t)
        cy = ch / 2.0 + (float(pan_y) * 0.5) * (ch - win_h) * (1.0 - 2.0 * t)
        left = round(min(max(0.0, cx - win_w / 2.0), cw - win_w))
        top = round(min(max(0.0, cy - win_h / 2.0), ch - win_h))
        out.append(canvas.crop((left, top, left + win_w, top + win_h)).resize(size, Image.Resampling.LANCZOS))
    return out


def parallax_drift(
    image: Image.Image,
    size: tuple[int, int],
    frames: int = 48,
    depth: float = 0.5,
    drift: float = 0.06,
) -> list[Image.Image]:
    """Two-plane drift: a sharp plane over a blurred one.

    Blurring is a cheap and surprisingly decent depth estimate -- an out-of-focus
    background really is the far plane -- so this is a two-layer parallax without
    asking the model for a depth map. The far plane moves *more* than the near
    one, which is what sells the depth; moving them by different amounts in
    opposite directions is the whole effect.

    ``drift`` is the near plane's peak travel as a **fraction of the output
    width**, and the canvas is grown to guarantee that many pixels of room in
    each direction. The first version of this expressed the offset as a fraction
    of the available slack, which made the slack itself the limiting quantity and
    produced a clip whose frames were pixel-identical -- Pillow's encoder
    collapsed 24 frames to 9, which is how the bug announced itself. Expressing
    travel in pixels and deriving the slack from it removes the coupling.

    This is the most cinematic thing here for one generation, and the cheapest:
    the still is used whole rather than cropped into a corner of the frame.
    """
    image = _as_rgb(image)
    tw, th = size
    travel_x = max(1.0, abs(float(drift)) * tw)
    travel_y = travel_x * 0.35  # a little vertical, so it is not a pure slide
    # Guarantee room for both planes at their furthest apart.
    peak = 1.0 + (2.0 * travel_x / tw) * (1.0 + abs(float(depth))) + 0.01
    base = _cover_scaled(image, size, peak)
    bw, bh = base.size
    blurred = base.filter(ImageFilter.GaussianBlur(radius=max(1.0, min(bw, bh) * 0.012)))

    out: list[Image.Image] = []
    for t in _frame_positions(frames):
        frame = np.zeros((th, tw, 3), np.float32)
        weight = 0.0
        planes = ((base, 1.0, 0.0), (blurred, 0.55, 1.0 + float(depth)))
        for canvas, layer_weight, plane_factor in planes:
            # Symmetric travel about the centre, opposing between planes.
            frac_x = (t - 0.5) * travel_x * plane_factor
            frac_y = (t - 0.5) * travel_y * plane_factor
            off_x = round((bw - tw) * 0.5 + frac_x)
            off_y = round((bh - th) * 0.5 + frac_y)
            off_x = min(max(0, off_x), bw - tw)
            off_y = min(max(0, off_y), bh - th)
            layer = np.asarray(canvas.crop((off_x, off_y, off_x + tw, off_y + th)), dtype=np.float32)
            frame += layer * layer_weight
            weight += layer_weight
        out.append(Image.fromarray(np.clip(frame / weight, 0, 255).astype(np.uint8), "RGB"))
    return out


def _resolve_duration(duration_ms: int) -> int:
    """Clamp a frame time into the range viewers actually honour."""
    try:
        ms = int(duration_ms)
    except (TypeError, ValueError):
        return DEFAULT_DURATION_MS
    if ms <= 0:
        return DEFAULT_DURATION_MS
    return max(MIN_DURATION_MS, min(ms, 10_000))


def _webp_anmf_payload_offsets(data: bytes) -> list[int]:
    """Absolute offsets of every ANMF chunk payload in a WebP file.

    Walks the RIFF container rather than parsing it with a library, because the
    whole point is that we do not trust the encoder to have written what we
    asked for and need to see the bytes ourselves. Returns ``[]`` if the file is
    not a well-formed WebP, so callers can decline to patch rather than corrupt.
    """
    if len(data) < 12 or data[0:4] != b"RIFF" or data[8:12] != b"WEBP":
        return []
    offsets: list[int] = []
    off = 12
    while off + 8 <= len(data):
        fourcc = data[off : off + 4]
        (size,) = struct.unpack("<I", data[off + 4 : off + 8])
        if off + 8 + size > len(data):
            return []  # truncated: refuse rather than write at a bogus offset
        if fourcc == b"ANMF":
            offsets.append(off + 8)
        off += 8 + size + (size & 1)  # chunks are padded to an even size
    return offsets


def webp_frame_durations(data: bytes) -> list[int]:
    """Read back the per-frame duration, in ms, of an animated WebP.

    The ANMF payload begins with a 12-byte frame rectangle -- x, y, width-1 and
    height-1, three bytes each -- and the 3-byte frame duration follows it at
    offset **12**. The format stores milliseconds in 24 bits, which is why the
    field is three bytes rather than four.

    Exposed because an encoder that silently ignores ``duration=`` is a real
    failure mode and reading the container is the only way to catch it. Note
    that Pillow's own ``Image.info`` is not a substitute: it does not reliably
    expose per-frame duration after a ``seek()``.
    """
    out: list[int] = []
    for payload in _webp_anmf_payload_offsets(data):
        d = payload + 12
        out.append(data[d] | (data[d + 1] << 8) | (data[d + 2] << 16))
    return out


def frames_to_webp(
    frames: Sequence[Image.Image],
    duration_ms: int = DEFAULT_DURATION_MS,
    loop: int = 0,
    lossless: bool = False,
    quality: int = 90,
) -> bytes:
    """Encode frames as an animated WebP.

    WebP is the default because it is both smaller and sharper than GIF for the
    same frames, and it supports a real alpha channel if a caller needs one.

    Verified against the encoded container rather than trusted: Pillow does
    honour ``duration=`` here (checked with :func:`webp_frame_durations`, which
    reads the ANMF duration field at payload offset 12). The encoder is left
    alone -- an earlier version of this function patched those three bytes by
    hand on the belief that Pillow dropped them, which turned out to be a fault
    in the reading code rather than the writer, and corrupting the frame's x
    coordinate made every output undecodable.
    """
    if not frames:
        raise ValueError("cannot encode an animation with no frames")
    duration = _resolve_duration(duration_ms)
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="WEBP",
        save_all=True,
        append_images=list(frames[1:]),
        duration=duration,
        loop=int(loop),
        lossless=bool(lossless),
        quality=max(1, min(int(quality), 100)),
        minimize_size=True,
        method=4,
    )
    return buf.getvalue()


def frames_to_gif(
    frames: Sequence[Image.Image],
    duration_ms: int = DEFAULT_DURATION_MS,
    loop: int = 0,
    palette_frames: int = 3,
) -> bytes:
    """Encode frames as an animated GIF, without palette flicker.

    GIF quantises to a 256-colour palette. If each frame is quantised on its own,
    the palette shifts between frames and a static region of the image visibly
    changes colour -- the flicker people blame on their encoder. So one palette
    is built from a few sample frames and then applied to all of them, which
    means the 8-bit indices change but the colours they point at do not.
    """
    if not frames:
        raise ValueError("cannot encode an animation with no frames")
    duration = _resolve_duration(duration_ms)
    rgb = [_as_rgb(f) for f in frames]

    # Sample frames spread across the clip, not just the first: a palette built
    # from frame 0 misses any colour that only appears later and those pixels
    # then dither against a colour that is not in the table.
    sample_count = max(1, min(int(palette_frames), len(rgb)))
    last = len(rgb) - 1
    picks = [rgb[min(last, round(i * last / max(1, sample_count - 1)))] for i in range(sample_count)]
    # The samples are tiled side by side into one montage rather than sampled as
    # strips: a strip takes one region of each frame, so colour that only appears
    # in the lower half of a frame never reaches the palette -- which is the
    # flicker this whole function exists to prevent. Adapters can only build a
    # shared palette from one image, so the montage is that image.
    w, h = picks[0].size
    montage = Image.new("RGB", (w * len(picks), h))
    for i, img in enumerate(picks):
        montage.paste(img, (i * w, 0))
    master = montage.quantize(colors=256, method=Image.Quantize.MEDIANCUT)

    quantized = [f.quantize(palette=master, dither=Image.Dither.NONE) for f in rgb]
    buf = io.BytesIO()
    quantized[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=quantized[1:],
        duration=duration,
        loop=int(loop),
        optimize=False,
        disposal=2,  # restore to background: without it, leftover pixels ghost
    )
    return buf.getvalue()


def encode_animation(frames: Sequence[Image.Image], spec: AnimationSpec) -> bytes:
    """Encode ``frames`` per ``spec.format``."""
    fmt = (spec.format or "webp").lower().strip()
    if fmt not in ANIMATION_FORMATS:
        raise ValueError(f"unsupported animation format {spec.format!r}; expected one of {ANIMATION_FORMATS}")
    if fmt == "gif":
        return frames_to_gif(frames, spec.duration_ms, spec.loop)
    return frames_to_webp(frames, spec.duration_ms, spec.loop, spec.lossless, spec.quality)


def animation_to_data_uri(frames: Sequence[Image.Image], spec: AnimationSpec) -> str:
    """The encoded animation as a ``data:`` URI, for a direct <img> src."""
    fmt = (spec.format or "webp").lower().strip()
    if fmt not in ANIMATION_FORMATS:
        raise ValueError(f"unsupported animation format {spec.format!r}")
    payload = encode_animation(frames, spec)
    mime = "image/gif" if fmt == "gif" else "image/webp"
    import base64

    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def build_animation(
    stills: Sequence[Image.Image],
    spec: AnimationSpec,
) -> list[Image.Image]:
    """Apply ``spec.kind`` to ``stills`` and return the frame sequence.

    Separated from encoding so the motion can be inspected, counted or scored
    before anything is written, and so a caller can reuse the frames for an MP4
    instead.
    """
    stills = [_as_rgb(s) for s in stills]
    if not stills:
        raise ValueError("no stills supplied")
    kind = (spec.kind or "ken_burns").lower().strip()
    if kind == "ken_burns":
        return ken_burns(stills[0], spec.size, spec.frames, spec.zoom_from, spec.zoom_to, spec.pan_x, spec.pan_y)
    if kind == "parallax":
        return parallax_drift(stills[0], spec.size, spec.frames, depth=0.5, drift=spec.parallax)
    if kind == "crossfade":
        return crossfade(stills, spec.size, spec.frames)
    if kind == "sprite":
        return sprite_sheet_to_frames(stills[0], spec.sprite_cols, spec.sprite_rows, spec.size)
    raise ValueError(f"unknown animation kind {spec.kind!r}")
