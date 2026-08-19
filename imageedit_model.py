"""VL-first text editing via qwen-image-edit through the alpaca proxy.

Crop the text strip, upscale height-only (keeps width under the VLM
resolution cap so small glyphs become legible to the model), POST the strip
to the proxy /v1/images/edits, composite the edited strip back. This is the
only editing path: no LaMa, no PIL pixel manipulation.

Proxy params (steps/cfg/strength) ride in via <sd_cpp_extra_args>, which
alpaca-proxy.py:3287-3313 embeds into the sd-server native request.
"""

from __future__ import annotations

import base64
import json
import sys
import urllib.error
import urllib.request
from io import BytesIO

from PIL import Image


def _upscale_strip(crop: Image.Image, min_glyph_px: int = 30) -> tuple[Image.Image, float]:
    text_h = crop.height - 12  # approx glyph height within strip (margins ~6px)
    k = max(1.0, min(8.0, min_glyph_px / max(4, text_h)))
    out = crop.resize((int(crop.width * k), int(crop.height * k)), Image.Resampling.LANCZOS)
    return out, k


def text_edit_model(
    image: Image.Image,
    band: tuple[int, int],
    text: str | None,
    proxy_url: str = "http://localhost:11434",
    model: str = "qwen-image-edit-rapid-aio:q4_k",
    min_glyph_px: int = 30,
    steps: int = 20,
    cfg: float = 2.5,
    seed: int = 42,
    timeout: int = 900,
    log: bool = True,
) -> tuple[Image.Image, dict]:
    """Edit a horizontal text line via qwen-image-edit (proxy /v1/images/edits)."""
    y0, y1 = band
    margin = 6
    crop = image.crop((0, max(0, y0 - margin), image.width, min(image.height, y1 + margin)))
    strip, k = _upscale_strip(crop, min_glyph_px)
    if log:
        print(f"[model] strip {crop.size} -> {strip.size} (k={k:.2f})", file=sys.stderr)

    buf = BytesIO()
    strip.save(buf, format="PNG")
    if text:
        prompt = (
            f"Replace the text shown on this sign panel with exactly: {text}. "
            "The new text must be clean, sharp, perfectly readable, same font style, "
            "size and position as the current text. Keep everything else unchanged."
            f"<sd_cpp_extra_args>{json.dumps({'sample_params': {'sample_steps': steps, 'guidance': {'txt_cfg': cfg}}, 'seed': seed})}</sd_cpp_extra_args>"
        )
    else:
        prompt = (
            "Erase the text shown on this sign panel completely. Fill the area with a "
            "clean blank surface matching the surrounding panel texture and lighting. "
            "No text, no letters, no numbers, nothing written there."
            f"<sd_cpp_extra_args>{json.dumps({'sample_params': {'sample_steps': steps, 'guidance': {'txt_cfg': cfg}}, 'seed': seed})}</sd_cpp_extra_args>"
        )

    boundary = "----alpacaTextCompose"
    body = b"".join(
        [
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="strip.png"\r\n'
            f"Content-Type: image/png\r\n\r\n".encode(),
            buf.getvalue(),
            b"\r\n",
            f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\n{model}\r\n'.encode(),
            f'--{boundary}\r\nContent-Disposition: form-data; name="prompt"\r\n\r\n{prompt}\r\n'.encode(),
            f'--{boundary}\r\nContent-Disposition: form-data; name="size"\r\n\r\n{strip.width}x{strip.height}\r\n'.encode(),
            f'--{boundary}\r\nContent-Disposition: form-data; name="response_format"\r\n\r\nb64_json\r\n'.encode(),
            f"--{boundary}--\r\n".encode(),
        ]
    )
    req = urllib.request.Request(
        f"{proxy_url}/v1/images/edits",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"proxy returned HTTP {e.code}: {e.read().decode()[:300]}") from e

    b64 = data["data"][0]["b64_json"]
    edited = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    edited = edited.resize(crop.size, Image.Resampling.LANCZOS)

    out = image.convert("RGB")
    out.paste(edited, (0, max(0, y0 - margin)))
    return out, {"k": k, "strip": strip.size, "model": model}
