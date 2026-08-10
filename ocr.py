#!/usr/bin/env python3
"""CLI for alpaca's vision OCR backend with automatic VL-model fallback.

Primary path:  POST /api/vision/ocr on the alpaca-web backend (port 5000).
Fallback path: direct call to the proxy /v1/chat/completions with a
discovered VL model when the web backend is unreachable.

Model resolution:
  1. If the requested model is present in the router, use it.
  2. Otherwise discover any VL-capable model (qwen2.5-vl, llava, minicpm-v,
     gemma3-vision...) and use the best available.
  3. If no VL model exists, fail with a clear error.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import sys
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

DEFAULT_MODEL = "qwen2.5-vl--7b"
VL_PATTERNS = ("qwen2.5-vl", "qwen3-vl", "vl--", "llava", "minicpm", "gemma3", "-vl")
PREFERRED = ("qwen2.5-vl--7b", "qwen2.5-vl--3b", "qwen2.5-vl:7b", "qwen2.5-vl:3b")

OCR_PROMPT = (
    "You are an expert Document AI and OCR vision assistant.\n"
    "Analyze the uploaded image or document and perform text extraction and layout parsing.\n\n"
    "Respond ONLY with a valid JSON object with the following structure:\n"
    "{\n"
    '  "full_text": "Complete extracted text from top to bottom...",\n'
    '  "headline": "Main title or headline text found in the image",\n'
    '  "subtext": "Subtitle, body text, or event details",\n'
    '  "badge": "Badge, price tag, or call-to-action text (e.g. 50% OFF, GET TICKETS)"\n'
    "}"
)


def _http_json(url: str, payload: dict | None = None, timeout: int = 300) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def discover_vl_models(proxy_url: str) -> list[str]:
    """List VL-capable models known to the router."""
    try:
        models = _http_json(f"{proxy_url}/v1/models", timeout=30).get("data", [])
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return []
    ids = [m.get("id", "") for m in models]
    return [i for i in ids if any(p in i.lower() for p in VL_PATTERNS)]


def resolve_model(requested: str, proxy_url: str) -> str:
    """Use the requested model, or fall back to a discovered VL model."""
    if requested:
        known = discover_vl_models(proxy_url)
        if requested in known:
            return requested
        if known:
            for pref in PREFERRED:
                if pref in known:
                    return pref
            return known[0]
        sys.exit(
            f"ERROR: model '{requested}' not found and no VL model is available. "
            f"Pull one (e.g. qwen2.5-vl:7b) via: python alpaca-puller.py pull qwen2.5-vl:7b"
        )
    return requested


def image_to_b64(path: Path) -> str:
    """Convert image (or first PDF page) to base64 JPEG like the web backend."""
    from PIL import Image

    if path.suffix.lower() == ".pdf":
        import fitz  # PyMuPDF

        doc = fitz.open(path)
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    else:
        img = Image.open(path).convert("RGB")
    img.thumbnail((1024, 1024))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def ocr_via_web(web_url: str, path: Path, model: str) -> dict:
    """Primary: alpaca-web /api/vision/ocr backend (multipart upload)."""
    import mimetypes
    import uuid

    boundary = uuid.uuid4().hex
    content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    raw = path.read_bytes()
    body = b"".join(
        [
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            raw,
            b"\r\n",
            f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\n{model}\r\n'.encode(),
            f"--{boundary}--\r\n".encode(),
        ]
    )
    req = urllib.request.Request(
        f"{web_url}/api/vision/ocr", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req, timeout=320) as resp:
        return json.loads(resp.read().decode())


def ocr_via_proxy(proxy_url: str, path: Path, model: str) -> dict:
    """Fallback: direct proxy /v1/chat/completions with the same OCR prompt."""
    proxy_model = model.replace("--", ":") if ("--" in model and ":" not in model) else model
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": OCR_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_b64(path)}"}},
            ],
        }
    ]
    try:
        resp = _http_json(
            f"{proxy_url}/v1/chat/completions",
            {"model": proxy_model, "messages": messages, "max_tokens": 1000, "temperature": 0.1},
            timeout=320,
        )
    except urllib.error.HTTPError as e:
        sys.exit(f"ERROR: proxy returned HTTP {e.code} for model '{proxy_model}': {e.reason}")
    except (urllib.error.URLError, OSError) as e:
        sys.exit(f"ERROR: proxy unreachable at {proxy_url}: {e}")
    raw_text = resp["choices"][0]["message"]["content"]
    parsed = {"full_text": raw_text, "headline": "", "subtext": "", "badge": ""}
    clean = raw_text.strip()
    if "```json" in clean:
        clean = clean.split("```json")[1].split("```")[0].strip()
    elif "```" in clean:
        clean = clean.split("```")[1].split("```")[0].strip()
    with contextlib.suppress(json.JSONDecodeError):
        parsed = json.loads(clean)
    return {"status": "success", "ocr_result": parsed, "raw_response": raw_text}


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR via alpaca vision backend with VL-model fallback")
    parser.add_argument("path", type=Path, help="Image file or PDF to OCR")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Preferred VL model (default: {DEFAULT_MODEL})")
    parser.add_argument("--proxy-url", default="http://localhost:11434")
    parser.add_argument("--web-url", default="http://localhost:5000")
    parser.add_argument("--direct", action="store_true", help="Skip the web backend, call the proxy directly")
    args = parser.parse_args()

    if not args.path.exists():
        sys.exit(f"ERROR: file not found: {args.path}")

    model = resolve_model(args.model, args.proxy_url)
    print(f"[ocr] model: {model}", file=sys.stderr)

    result: dict | None = None
    if not args.direct:
        try:
            result = ocr_via_web(args.web_url, args.path, model)
        except (urllib.error.URLError, OSError) as e:
            print(f"[ocr] web backend unreachable ({e}); falling back to proxy", file=sys.stderr)
    if result is None:
        result = ocr_via_proxy(args.proxy_url, args.path, model)

    parsed = result.get("ocr_result", {})
    print(json.dumps(parsed, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
