#!/usr/bin/env python3
"""
Detect whether the active GGUF model is MoE or dense, then emit
llama-server flags that are safe for the detected architecture.

MoE indicators (any match → MoE):
  - metadata key `llm.expert_count` > 0
  - metadata key `llm.expert_used_count` > 0
  - tensor name contains `experts.` or `ffn_gate` + `ffn_up` pattern

Dense models get a stripped flag set (no MoE / speculative flags).
"""
import json
import os
import struct
import sys
from pathlib import Path

ROUTER_MODELS_DIR = os.getenv("ROUTER_MODELS_DIR", "/router-models")


def _read_gguf_metadata(path: str) -> dict:
    """Parse only the metadata header from a GGUF file (fast, no full load)."""
    meta: dict = {}
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            return meta
        version = struct.unpack("<I", f.read(4))[0]
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        kv_count = struct.unpack("<Q", f.read(8))[0]

        for _ in range(kv_count):
            key_len = struct.unpack("<I", f.read(4))[0]
            key = f.read(key_len).decode("utf-8", errors="replace")
            val_type = struct.unpack("<I", f.read(4))[0]

            # GGUF value types: 0=uint8,1=int8,2=uint16,3=int16,4=uint32,
            # 5=int32,6=float32,7=bool,8=str,9=array,10=uint64,11=int64,12=float64
            if val_type == 8:  # string
                str_len = struct.unpack("<Q", f.read(8))[0]
                val = f.read(str_len).decode("utf-8", errors="replace")
                meta[key] = val
            elif val_type in (0, 1, 4, 5, 10, 11):  # integer types
                fmt = {0: "<B", 1: "<b", 4: "<I", 5: "<i", 10: "<Q", 11: "<q"}
                val = struct.unpack(fmt[val_type], f.read(struct.calcsize(fmt[val_type])))[0]
                meta[key] = val
            elif val_type == 7:  # bool
                meta[key] = struct.unpack("<?", f.read(1))[0]
            elif val_type == 6:  # float32
                meta[key] = struct.unpack("<f", f.read(4))[0]
            elif val_type == 12:  # float64
                meta[key] = struct.unpack("<d", f.read(8))[0]
            elif val_type == 9:  # array
                arr_type = struct.unpack("<I", f.read(4))[0]
                arr_len = struct.unpack("<Q", f.read(8))[0]
                # Skip array contents — we only care about scalar metadata
                if arr_type == 8:  # array of strings
                    for _ in range(arr_len):
                        sl = struct.unpack("<Q", f.read(8))[0]
                        f.read(sl)
                else:
                    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
                    skip = arr_len * sizes.get(arr_type, 0)
                    f.read(skip)
            else:
                # Unknown type — skip conservatively
                break

    return meta


def _model_size_params(meta: dict) -> int:
    """Return total parameter count from GGUF metadata, or 0 if unknown."""
    pc = meta.get("general.parameter_count", 0)
    if isinstance(pc, (int, float)):
        return int(pc)
    # Fallback: try size_label like "8B", "70B"
    label = meta.get("general.size_label", "")
    if isinstance(label, str):
        import re
        m = re.match(r"(\d+(?:\.\d+)?)\s*[Bb]", label)
        if m:
            return int(float(m.group(1)) * 1_000_000_000)
    return 0


def _is_moe(meta: dict) -> bool:
    expert_count = meta.get("llm.expert_count", 0)
    expert_used = meta.get("llm.expert_used_count", 0)
    if isinstance(expert_count, (int, float)) and int(expert_count) > 0:
        return True
    if isinstance(expert_used, (int, float)) and int(expert_used) > 0:
        return True
    arch = meta.get("general.architecture", "")
    if "moe" in arch.lower():
        return True
    return False


def _find_active_model() -> str | None:
    """Find the first .gguf symlink/file in the router models directory."""
    router_dir = Path(ROUTER_MODELS_DIR)
    if not router_dir.exists():
        return None
    for entry in sorted(router_dir.iterdir()):
        if entry.suffix == ".gguf":
            resolved = entry.resolve()
            if resolved.exists():
                return str(resolved)
    return None


def get_llama_server_flags() -> list[str]:
    """Return the full llama-server flag list appropriate for the active model."""
    model_path = _find_active_model()
    is_moe = False
    param_count = 0

    if model_path:
        try:
            meta = _read_gguf_metadata(model_path)
            is_moe = _is_moe(meta)
            param_count = _model_size_params(meta)
        except Exception as e:
            print(f"[llama-flags] Warning: could not read model metadata: {e}", file=sys.stderr)
            print("[llama-flags] Falling back to dense (safe) config", file=sys.stderr)

    small_model = param_count > 0 and param_count < 9_000_000_000

    # Base flags — safe for both dense and MoE
    flags = [
        "--host", "0.0.0.0",
        "--port", "8080",
        "--models-dir", "/router-models",
        "-ngl", "99",
        "--no-mmap",
        "--mlock",
        "-t", "8",
    ]

    if small_model:
        # Small models on 8GB VRAM: reduce context + use f16 cache to avoid OOM
        flags.extend([
            "-c", "8192",
            "--cache-type-k", "f16",
            "--cache-type-v", "f16",
        ])
    else:
        # Large models: full context + q8_0 cache (assumes sufficient VRAM)
        flags.extend([
            "-c", "65536",
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
        ])

    if is_moe:
        flags.extend([
            "--n-cpu-moe", "38",
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", "3",
        ])
        print("[llama-flags] Detected MoE model — enabling MoE + speculative flags", file=sys.stderr)
    else:
        print("[llama-flags] Detected dense model — using safe config (no MoE/speculative flags)", file=sys.stderr)

    if small_model:
        print(f"[llama-flags] Model < 9B ({param_count/1e9:.1f}B) — using 8K context + f16 cache for 8GB VRAM", file=sys.stderr)

    if model_path:
        print(f"[llama-flags] Model: {os.path.basename(model_path)} (MoE={is_moe}, params={param_count/1e9:.1f}B)", file=sys.stderr)
    else:
        print("[llama-flags] No model found — using safe dense config", file=sys.stderr)

    return flags


if __name__ == "__main__":
    flags = get_llama_server_flags()
    # Print flags as a single line for docker-compose command substitution
    print(" ".join(flags))
