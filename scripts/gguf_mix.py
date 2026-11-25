#!/usr/bin/env python3
"""Assemble a GGUF from two quantizations of the same model, tensor by tensor.

    python -I scripts/gguf_mix.py OUT.gguf BASE.gguf SRC.gguf REGEX [SRC2.gguf REGEX2 ...]

Every tensor comes from BASE unless its name matches a REGEX, in which case its type and data come
from that SRC (a later pair wins). Metadata is BASE's, copied byte for byte. Nothing is
requantized: tensor bytes are copied as they are, so the result is exactly as good as the parts.

Why: an 8 GB card holds a looped MoE's core layers only at a small quantization (the experts that
run several times per token, on the GPU), while the outer layers' experts live in RAM, where a
larger quantization costs memory, not VRAM. Mixing IQ3 core experts with Q4 outer experts keeps
the GPU part small without lowering the quality of what runs on the CPU.
"""
from __future__ import annotations

import os
import re
import struct
import sys
from typing import BinaryIO

GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32

# GGUF metadata value types (gguf spec)
_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_STRING, _ARRAY = 8, 9


def _read(f: BinaryIO, fmt: str):
    size = struct.calcsize("<" + fmt)
    data = f.read(size)
    if len(data) != size:
        raise ValueError("unexpected end of file")
    return struct.unpack("<" + fmt, data)[0]


def _read_str(f: BinaryIO) -> str:
    n = _read(f, "Q")
    return f.read(n).decode("utf-8")


def _skip_value(f: BinaryIO, vtype: int):
    """Read (and return) one metadata value; only general.alignment's value is used."""
    if vtype in _SCALARS:
        return _read(f, _SCALARS[vtype])
    if vtype == _STRING:
        return _read_str(f)
    if vtype == _ARRAY:
        etype, n = _read(f, "I"), _read(f, "Q")
        for _ in range(n):
            _skip_value(f, etype)
        return None
    raise ValueError(f"unknown GGUF value type {vtype}")


def parse(path: str) -> dict:
    """Header facts and every tensor's name, dims, type, absolute offset and stored size."""
    with open(path, "rb") as f:
        if f.read(4) != GGUF_MAGIC:
            raise ValueError(f"{path} is not a GGUF file")
        version = _read(f, "I")
        n_tensors, n_kv = _read(f, "Q"), _read(f, "Q")
        kv_start = f.tell()
        alignment = DEFAULT_ALIGNMENT
        for _ in range(n_kv):
            key = _read_str(f)
            value = _skip_value(f, _read(f, "I"))
            if key == "general.alignment":
                alignment = int(value)
        kv_end = f.tell()
        tensors = []
        for _ in range(n_tensors):
            name = _read_str(f)
            dims = [_read(f, "Q") for _ in range(_read(f, "I"))]
            ttype, offset = _read(f, "I"), _read(f, "Q")
            tensors.append({"name": name, "dims": dims, "type": ttype, "off": offset, "path": path})
        data_start = -(-f.tell() // alignment) * alignment
        f.seek(kv_start)
        kv_bytes = f.read(kv_end - kv_start)
    data_end = os.path.getsize(path) - data_start
    ordered = sorted(tensors, key=lambda t: t["off"])
    for cur, nxt in zip(ordered, [*ordered[1:], {"off": data_end}], strict=True):
        cur["size"] = nxt["off"] - cur["off"]
    for t in tensors:
        t["abs"] = data_start + t["off"]
    return {"version": version, "n_kv": n_kv, "kv": kv_bytes, "tensors": tensors, "alignment": alignment}


def mix(out: str, base: str, pairs: list[tuple[str, str]]) -> dict[str, int]:
    """Write OUT from BASE with the tensors named by each (src, regex) pair taken from src."""
    head = parse(base)
    tensors = head["tensors"]
    counts: dict[str, int] = {}
    for src_path, pattern in pairs:
        rx = re.compile(pattern)
        src = {t["name"]: t for t in parse(src_path)["tensors"]}
        n = 0
        for i, t in enumerate(tensors):
            if rx.search(t["name"]):
                s = src.get(t["name"])
                if s is None:
                    raise ValueError(f"{t['name']} is not in {src_path}")
                if s["dims"] != t["dims"]:
                    raise ValueError(f"{t['name']}: dims {s['dims']} in {src_path} != {t['dims']} in {base}")
                tensors[i] = s
                n += 1
        counts[f"{os.path.basename(src_path)} ~ {pattern}"] = n
    align = head["alignment"]
    pad = lambda n: (-n) % align  # noqa: E731
    infos, offset = b"", 0
    for t in tensors:
        name = t["name"].encode("utf-8")
        infos += struct.pack("<Q", len(name)) + name + struct.pack("<I", len(t["dims"]))
        infos += b"".join(struct.pack("<Q", d) for d in t["dims"]) + struct.pack("<IQ", t["type"], offset)
        t["new_off"] = offset
        offset += t["size"] + pad(t["size"])
    header = GGUF_MAGIC + struct.pack("<IQQ", head["version"], len(tensors), head["n_kv"]) + head["kv"] + infos
    tmp = out + ".partial"
    sources: dict[str, BinaryIO] = {}
    try:
        with open(tmp, "wb") as o:
            o.write(header)
            o.write(b"\0" * pad(len(header)))
            data0 = o.tell()
            for t in tensors:
                if o.tell() - data0 != t["new_off"]:
                    raise AssertionError(f"offset drift at {t['name']}")
                f = sources.setdefault(t["path"], open(t["path"], "rb"))  # noqa: SIM115
                f.seek(t["abs"])
                left = t["size"]
                while left:
                    chunk = f.read(min(left, 64 << 20))
                    if not chunk:
                        raise ValueError(f"{t['path']} ended inside {t['name']}")
                    o.write(chunk)
                    left -= len(chunk)
                o.write(b"\0" * pad(t["size"]))
        os.replace(tmp, out)
    finally:
        for f in sources.values():
            f.close()
        if os.path.exists(tmp):
            os.unlink(tmp)
    return counts


def main(argv: list[str]) -> int:
    if len(argv) < 4 or len(argv[2:]) % 2:
        print(__doc__)
        return 2
    out, base, rest = argv[0], argv[1], argv[2:]
    counts = mix(out, base, list(zip(rest[0::2], rest[1::2], strict=True)))
    for what, n in counts.items():
        print(f"{n} tensors from {what}")
    print(f"wrote {out} ({os.path.getsize(out) / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
