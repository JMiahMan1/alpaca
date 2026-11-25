"""scripts/gguf_mix.py: tensors move between quantizations byte for byte, metadata stays the base's."""
from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gguf_mix", ROOT / "scripts" / "gguf_mix.py")
gguf_mix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gguf_mix)


def _s(text: str) -> bytes:
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path: Path, tensors: list[tuple[str, list[int], int, bytes]], kv_note: str = "base") -> None:
    """A minimal valid GGUF v3: two metadata keys (one an array) and the given tensors."""
    kv = _s("general.name") + struct.pack("<I", 8) + _s(kv_note)
    kv += _s("test.list") + struct.pack("<I", 9) + struct.pack("<IQ", 4, 3) + struct.pack("<III", 1, 2, 3)
    infos, off, blobs = b"", 0, []
    for name, dims, ttype, data in tensors:
        infos += _s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        infos += struct.pack("<IQ", ttype, off)
        blobs.append(data + b"\0" * ((-len(data)) % 32))
        off += len(blobs[-1])
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 2) + kv + infos
    path.write_bytes(header + b"\0" * ((-len(header)) % 32) + b"".join(blobs))


def tensor_bytes(path: Path) -> dict[str, tuple[int, bytes]]:
    info = gguf_mix.parse(str(path))
    raw = path.read_bytes()
    return {t["name"]: (t["type"], raw[t["abs"]:t["abs"] + t["size"]].rstrip(b"\0")) for t in info["tensors"]}


def test_matching_tensors_come_from_the_source_and_the_rest_from_the_base(tmp_path):
    base, src, out = tmp_path / "iq3.gguf", tmp_path / "q4.gguf", tmp_path / "mix.gguf"
    names = ["blk.0.ffn_down_exps.weight", "blk.7.ffn_down_exps.weight", "blk.19.ffn_up_exps.weight", "token_embd.weight"]
    write_gguf(base, [(n, [4, 2], 20, b"IQ3-" + n.encode()) for n in names])
    write_gguf(src, [(n, [4, 2], 12, b"Q4----" + n.encode()) for n in names], kv_note="src")
    counts = gguf_mix.mix(str(out), str(base), [(str(src), r"^blk\.([0-5]|1[89])\.ffn_.*_exps\.")])
    assert list(counts.values()) == [2]
    got = tensor_bytes(out)
    assert got["blk.0.ffn_down_exps.weight"] == (12, b"Q4----blk.0.ffn_down_exps.weight")
    assert got["blk.19.ffn_up_exps.weight"] == (12, b"Q4----blk.19.ffn_up_exps.weight")
    assert got["blk.7.ffn_down_exps.weight"] == (20, b"IQ3-blk.7.ffn_down_exps.weight")
    assert got["token_embd.weight"] == (20, b"IQ3-token_embd.weight")
    assert b"base" in out.read_bytes()[:200] and b"src" not in out.read_bytes()[:200]  # the base's metadata
    assert not (tmp_path / "mix.gguf.partial").exists()


def test_mismatched_shapes_are_refused_and_leave_nothing_behind(tmp_path):
    base, src, out = tmp_path / "a.gguf", tmp_path / "b.gguf", tmp_path / "out.gguf"
    write_gguf(base, [("blk.0.ffn_down_exps.weight", [4, 2], 20, b"x")])
    write_gguf(src, [("blk.0.ffn_down_exps.weight", [8, 2], 12, b"y")])
    with pytest.raises(ValueError, match="dims"):
        gguf_mix.mix(str(out), str(base), [(str(src), r"ffn")])
    assert not out.exists() and not (tmp_path / "out.gguf.partial").exists()


def test_a_tensor_missing_from_the_source_is_refused(tmp_path):
    base, src, out = tmp_path / "a.gguf", tmp_path / "b.gguf", tmp_path / "out.gguf"
    write_gguf(base, [("blk.0.ffn_down_exps.weight", [4, 2], 20, b"x")])
    write_gguf(src, [("other.weight", [4, 2], 12, b"y")])
    with pytest.raises(ValueError, match="not in"):
        gguf_mix.mix(str(out), str(base), [(str(src), r"ffn")])


def test_not_a_gguf_is_refused(tmp_path):
    bad = tmp_path / "x.gguf"
    bad.write_bytes(b"nope")
    with pytest.raises(ValueError, match="not a GGUF"):
        gguf_mix.parse(str(bad))
