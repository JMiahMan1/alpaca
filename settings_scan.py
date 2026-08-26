#!/usr/bin/env python3
"""Settings scan: sweep llama.cpp runtime settings per model and record
throughput (tok/s), time-to-first-token (ms), VRAM used (MB), and whether the
run produced non-empty content. Supports both MoE and Dense text models
(auto-detected via GGUF metadata) and always starts with q8_0 KV cache.
Writes data/llm_benchmarks/settings_scan.json.

Each combination requires a llama-server restart (router mode re-reads
models.ini on start), so the sweep is slow by design - run it as a background
job and poll the output file. For quick insight without restarts, use
existing telemetry and benchmark logs in data/telemetry/*.jsonl and
data/llm_benchmarks/.

Usage:
    python settings_scan.py [--models m1,m2] [--combos N]
    python settings_scan.py --models ornith-1-5-9b-q4-k-m--latest,qwen3-6-35b-a3b-ud-iq4-nl-mtp--latest
"""

import argparse
import configparser
import json
import os
import subprocess
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent


def _model_temperature(model: str) -> float:
    for ini in [
        Path(os.getenv("MODELS_INI_PATH", "")).resolve() if os.getenv("MODELS_INI_PATH", "").strip() else None,
        REPO / ".alpaca-router" / "models.ini",
        Path(".alpaca-router/models.ini"),
    ]:
        if ini is None or not ini.exists():
            continue
        cp = configparser.ConfigParser()
        try:
            cp.read(ini)
            if model in cp and "temperature" in cp[model]:
                return float(cp[model]["temperature"])
            if "*" in cp and "temperature" in cp["*"]:
                return float(cp["*"]["temperature"])
        except ValueError:
            raise
        except Exception:
            continue
    raise ValueError(
        f"temperature not set for model '{model}' (and no [*] default) in {REPO / '.alpaca-router/models.ini'} - set via Settings > UI"
    )


INI = REPO / ".alpaca-router" / "models.ini"
OUT = REPO / "data" / "llm_benchmarks" / "settings_scan.json"
COMPOSE = ["sudo", "docker", "compose"]
LLAMA_URL = "http://localhost:8080"
PROXY_URL = "http://localhost:11434"

# Fixed probe prompt: realistic benchmark workload. Uses 4000 tokens like
# the actual gamedev benchmark category so generation quality is meaningful.
# 600 tokens is insufficient for a full game and creates truncated output.
PROBE = (
    "Write a complete, runnable Python program using pygame that implements "
    "a playable Pong game. Open an 800x600 window, draw a paddle moved with "
    "UP/DOWN arrow keys, a ball that bounces off walls and the paddle, a "
    "numeric score with pygame.font, a game loop handling QUIT and "
    "pygame.display.flip(). Use only primitive drawing. Save top 5 scores to "
    "high_scores.json, reset to 0 on new game. Respond with ONLY the raw "
    "runnable source code in a single fenced code block, no preamble."
)
MAX_TOKENS = 4000


def read_ini():
    import configparser

    c = configparser.ConfigParser(delimiters=("=",))
    c.read(str(INI))
    return c


def write_ini(c):
    with open(INI, "w") as f:
        c.write(f)
    os.chmod(INI, 0o666)


def is_moe_model(model_path: str) -> bool:
    """Detect MoE vs Dense by reading GGUF metadata, same logic as llama-server-flags.py.
    No hardcoded model names; reads the GGUF file directly. Handles host's broken
    symlink by resolving to /usr/share/ollama blob via sudo if needed.
    """
    try:
        import struct

        # Build candidate paths without hardcoding model names
        candidates = []
        p0 = Path(model_path)
        candidates.append(p0)
        host_link = REPO / ".alpaca-router" / Path(model_path).name
        candidates.append(host_link)
        # If host_link is a broken symlink, also try its blob target
        if host_link.is_symlink():
            try:
                tgt = host_link.readlink()
                blob = Path("/usr/share/ollama/.ollama/models/blobs") / Path(tgt).name
                candidates.append(blob)
            except Exception:
                pass

        for cand in candidates:
            try:
                # Try direct open; if run with sudo, blob at /usr/share/ollama will be readable.
                # No internal sudo call here - run the script with sudo if needed.
                with open(cand, "rb") as f:
                    if f.read(4) != b"GGUF":
                        continue
                    f.read(4)
                    f.read(8)
                    kv_count = struct.unpack("<Q", f.read(8))[0]
                    for _ in range(min(kv_count, 300)):
                        try:
                            key_len = struct.unpack("<Q", f.read(8))[0]
                            k = f.read(key_len).decode(errors="replace")
                        except Exception:
                            break
                        try:
                            vt = struct.unpack("<I", f.read(4))[0]
                        except Exception:
                            break
                        if k.endswith(".expert_count") or k.endswith(".expert_used_count"):
                            return True
                        if "moe" in k.lower() and "architecture" in k.lower():
                            return True
                        try:
                            if vt == 8:
                                sl = struct.unpack("<Q", f.read(8))[0]
                                f.read(sl)
                            elif vt in (0, 1, 4, 5, 10, 11):
                                fmt2 = {0: "<B", 1: "<b", 4: "<I", 5: "<i", 10: "<Q", 11: "<q"}[vt]
                                f.read(struct.calcsize(fmt2))
                            elif vt == 7:
                                f.read(1)
                            elif vt == 6:
                                f.read(4)
                            elif vt == 12:
                                f.read(8)
                            elif vt == 9:
                                at = struct.unpack("<I", f.read(4))[0]
                                al = struct.unpack("<Q", f.read(8))[0]
                                if at == 8:
                                    for _ in range(al):
                                        sl = struct.unpack("<Q", f.read(8))[0]
                                        f.read(sl)
                                else:
                                    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
                                    f.read(al * sizes.get(at, 0))
                            else:
                                break
                        except Exception:
                            break
                # If we got here without returning True, this candidate is dense
                continue
            except Exception:
                continue
        return False
    except Exception:
        return False


def _model_supports_mtp(model_name: str) -> bool:
    """Timeless MTP detection: name contains mtp OR existing models.ini already
    has spec-type draft-mtp for that section. No hardcoded model list."""
    if "mtp" in model_name.lower():
        return True
    try:
        import configparser

        c = configparser.ConfigParser(delimiters=("=",))
        c.read(str(INI))
        if c.has_section(model_name) and c[model_name].get("spec-type", "").strip() == "draft-mtp":
            return True
    except Exception:
        pass
    return False


def build_combos_for_model(is_moe: bool, model_name: str = ""):
    """Return list of (settings dict, label) for given model type. Always q8_0 first.
    For MoE, n-gpu-layers is fixed to 99 (kat-coder proved 99 22.5 tok/s >> 26 12-13 tok/s;
    99 offloads all layers to GPU, 26 wastes GPU). Dense still tests both.
    sweeps n-cpu-moe 34/31/28/30/auto and MTP draft-mtp 2,3,5 if supported.
    """
    base = []
    # Context must be 68K+ per user: MoE needs large KV for long benchmarks/RAG,
    # 98304 is proven for ornith/qwen3-mtp on 8GB with q8_0+flash-on, 65536 is min.
    ctx_choices = [98304, 65536] if is_moe else [8192, 4096]
    # For MoE, also test 8192 as fallback if 68K OOMs, but prioritize 68K+.
    # User says 8192 unrealistic, so we prioritize 98304/65536.
    for ctx in ctx_choices:
        for kv in ["q8_0", "q4_0"]:
            for ngl in [99, 26]:
                for fa in ["on", "off"]:
                    if kv in ("q8_0", "q4_0") and fa == "off":
                        continue
                    label = f"ctx={ctx} kv={kv} ngl={ngl} fa={fa}"
                    base.append(
                        (
                            {
                                "ctx-size": str(ctx),
                                "cache-type-k": kv,
                                "cache-type-v": kv,
                                "n-gpu-layers": str(ngl),
                                "flash-attn": fa,
                            },
                            label,
                        )
                    )
    # MoE: n-gpu-layers 99 is strictly better (22.5 vs 12.5 tok/s) - skip 26 to save 50% time.
    if is_moe:
        base = [b for b in base if b[0]["n-gpu-layers"] == "99"]
    if not is_moe:
        # Dense: also test MTP if supported
        if _model_supports_mtp(model_name):
            expanded = []
            for settings, label in base:
                for spec, draft in [
                    ("none", "0"),
                    ("draft-mtp", "1"),
                    ("draft-mtp", "2"),
                    ("draft-mtp", "3"),
                    ("draft-mtp", "5"),
                ]:
                    ns = dict(settings)
                    ns["spec-type"] = spec
                    ns["spec-draft-n-max"] = draft
                    nlabel = label + f" spec={spec} nmax={draft}"
                    expanded.append((ns, nlabel))
            return expanded
        return base
    # MoE: expand each base combo with n-cpu-moe variants.
    # Order starts at 36 per user (try 36 first to guarantee success on 8GB with 99 layers,
    # then work down 34/31/30/28 for speed); 36 frees more VRAM than 34. For MTP,
    # higher moe frees VRAM for draft (user: adjust moe up when MTP enabled).
    expanded = []
    for settings, label in base:
        for moe in ["36", "34", "31", "30", "28", ""]:
            ns = dict(settings)
            ns["n-cpu-moe"] = moe
            nlabel = label + (f" moe={moe}" if moe else " moe=auto")
            expanded.append((ns, nlabel))
    # MTP sweep for MoE models that support it (e.g. qwen3-mtp, ornith mtp)
    if _model_supports_mtp(model_name):
        mtp_expanded = []
        for settings, label in expanded:
            for spec, draft in [
                ("none", "0"),
                ("draft-mtp", "1"),
                ("draft-mtp", "2"),
                ("draft-mtp", "3"),
                ("draft-mtp", "5"),
            ]:
                ns = dict(settings)
                ns["spec-type"] = spec
                ns["spec-draft-n-max"] = draft
                nlabel = label + f" spec={spec} nmax={draft}"
                mtp_expanded.append((ns, nlabel))
        return mtp_expanded
    return expanded


def restart_llama():
    subprocess.run([*COMPOSE, "stop", "llama-server"], capture_output=True)
    subprocess.run([*COMPOSE, "up", "-d", "--no-deps", "llama-server"], capture_output=True)
    for _ in range(60):
        try:
            r = httpx.get(f"{LLAMA_URL}/health", timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def vram_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return int(out.stdout.strip().split()[0])
    except Exception:
        return 0


def run_probe(model, think=True, budget=None):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROBE}],
        "stream": False,
        "think": think,
        "options": {"num_predict": MAX_TOKENS, "temperature": _model_temperature(model)},
    }
    if budget is not None:
        payload["reasoning_budget"] = budget
    t0 = time.time()
    try:
        r = httpx.post(f"{PROXY_URL}/api/chat", json=payload, timeout=300.0)
        if r.status_code != 200:
            return {"success": False, "error": f"HTTP {r.status_code}: {r.text[:200]}"}
        d = r.json()
        msg = d.get("message", {}) or {}
        content = msg.get("content") or d.get("response") or ""
        thinking = msg.get("thinking") or ""
        return {
            "success": bool(content.strip()),
            "content_len": len(content),
            "thinking_len": len(thinking),
            "tokens": d.get("eval_count", 0),
            "ttft_ms": round((d.get("prompt_eval_duration", 0) or 0) / 1e6),
            "elapsed_s": round(time.time() - t0, 2),
            "error": None if content.strip() else "EMPTY CONTENT",
        }
    except Exception as e:
        return {"success": False, "error": str(e)[:200], "elapsed_s": round(time.time() - t0, 2)}


def load_section(c, section, overrides):
    """Apply the sweep overrides to a model section while preserving every
    other existing key (model path, spec-type, cache-reuse, etc.)."""
    if not c.has_section(section):
        c.add_section(section)
    for k, v in overrides.items():
        if v is None or v == "":
            c[section].pop(k, None)
        else:
            c[section][k] = str(v)


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark each model and find best runtime settings for resource utilization and speed. "
        "Auto-detects MoE vs Dense and always starts with q8_0."
    )
    ap.add_argument(
        "--models",
        default="",
        help="Comma-separated model sections from models.ini, or empty/all to scan every model we have",
    )
    ap.add_argument("--combos", type=int, default=0, help="limit combos per model (0=all)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--skip-restart", action="store_true")
    args = ap.parse_args()

    c_all = read_ini()
    if not args.models or args.models.strip().lower() == "all":
        models = [s for s in c_all.sections() if s != "*"]
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text())
        except Exception:
            results = {}

    c = c_all
    for model in models:
        if not c.has_section(model):
            print(f"[scan] section {model} not in models.ini, skipping")
            continue
        # Detect MoE vs Dense to choose appropriate sweep
        model_path = c[model].get("model", "")
        is_moe = is_moe_model(model_path)
        combos = build_combos_for_model(is_moe, model)
        if args.combos:
            combos = combos[: args.combos]
        print(f"\n[scan] === model {model} ({'MoE' if is_moe else 'Dense'} {len(combos)} combos) ===")
        # Hill order for freeing VRAM: higher n-cpu-moe => more experts on CPU => less VRAM.
        # Start at 36 per user ( ornith needs 36 to even start at 98304/q8_0/99), then 40/48 if still OOM.
        MOE_HIER = ["", "28", "30", "31", "34", "36", "40", "48"]

        def _next_moe(cur: str, _hier=MOE_HIER) -> str | None:
            try:
                idx = _hier.index(cur or "")
            except ValueError:
                return None
            return _hier[idx + 1] if idx + 1 < len(_hier) else None

        for settings, label in combos:
            key = f"{model}::{label}"
            if key in results and results[key].get("done"):
                print(f"[scan] skip {key}")
                continue
            print(f"[scan] combo {label}")
            # Timeless adaptive: if MTP was requesting but VRAM fails, bump n-cpu-moe to free memory,
            # then hill-climb back down when lower mtp (2 vs 5) frees overhead - the next combo in the sweep
            # already tests the lower moe again, so we only need to escalate on failure.
            cur_settings = dict(settings)
            cur_label = label
            cur_key = key
            attempt = 0
            while True:
                load_section(c, model, cur_settings)
                write_ini(c)
                if not args.skip_restart and not restart_llama():
                    results[cur_key] = {
                        "done": True,
                        "success": False,
                        "error": "llama-server did not become ready",
                        "settings": cur_settings,
                    }
                    out_path.write_text(json.dumps(results, indent=2))
                    break
                time.sleep(3)
                v0 = vram_mb()
                res = run_probe(model)
                v1 = vram_mb()
                res.update(
                    {
                        "done": True,
                        "settings": dict(cur_settings),
                        "label": cur_label,
                        "model": model,
                        "vram_before_mb": v0,
                        "vram_after_mb": v1,
                        "vram_peak_mb": max(v0, v1),
                    }
                )
                results[cur_key] = res
                out_path.write_text(json.dumps(results, indent=2))
                print(f"    -> {res}")
                # Detect memory pressure: timed out / OOM / HTTP 503 queue_timeout or VRAM > 85% (7500 MB)
                # Timeless: for MoE, bump n-cpu-moe up (e.g. 34->36->40->48) until it starts,
                # then later combos hill-climb back down for speed. Works for both MTP and non-MTP
                # (ornith 98304 non-MTP was failing repeatedly at 34 without bump).
                err = (res.get("error") or "").lower()
                mem_fail = (not res.get("success")) and (
                    "timed out" in err
                    or "memory" in err
                    or "oom" in err
                    or "queue_timeout" in err
                    or "no llama-server slot" in err
                    or res.get("vram_peak_mb", 0) > 7500
                )
                if is_moe and mem_fail and attempt < 3:
                    nxt = _next_moe(cur_settings.get("n-cpu-moe", "") or "")
                    if nxt and nxt != cur_settings.get("n-cpu-moe", ""):
                        attempt += 1
                        # bump moe to free VRAM and retry same mtp draft
                        new_settings = dict(cur_settings)
                        new_settings["n-cpu-moe"] = nxt
                        # keep label consistent for tracking but mark adaptive
                        new_label = (
                            cur_label.replace(f"moe={cur_settings.get('n-cpu-moe', '')}", f"moe={nxt}")
                            if "moe=" in cur_label
                            else cur_label + f" moe={nxt}(adapt)"
                        )
                        new_key = f"{model}::{new_label}"
                        if new_key in results and results[new_key].get("done"):
                            break
                        print(
                            f"    [adapt] VRAM pressure ({res.get('vram_peak_mb')} MB, {err[:60]}) -> bumping n-cpu-moe {cur_settings.get('n-cpu-moe')}->{nxt} and retrying"
                        )
                        cur_settings = new_settings
                        cur_label = new_label
                        cur_key = new_key
                        continue
                break

    # summary and write best config to each model profile
    print("\n[scan] Summary (resource utilization and speed):")
    for m in models:
        entries = [v for k, v in results.items() if k.startswith(m + "::") and v.get("success")]
        if not entries:
            print(f"  {m}: no successful runs")
            continue
        for e in entries:
            e["tps"] = round(e.get("tokens", 0) / e.get("elapsed_s", 1), 1) if e.get("elapsed_s") else 0
        best_speed = max(entries, key=lambda x: x.get("tps", 0))
        best_vram = min(entries, key=lambda x: x.get("vram_peak_mb", 999999))
        print(f"  {m}:")
        print(
            f"    best speed: {best_speed['label']} -> {best_speed['tps']} tok/s, peak {best_speed['vram_peak_mb']} MB VRAM"
        )
        print(
            f"    best resource: {best_vram['label']} -> {best_vram['tps']} tok/s, peak {best_vram['vram_peak_mb']} MB VRAM"
        )
        # write best speed config to the model profile (resource-aware: require <90% VRAM if possible)
        # prefer best speed among those under 85% VRAM, fallback to absolute best speed
        vram_total = 8188  # RTX 4060
        candidates = [e for e in entries if e.get("vram_peak_mb", 0) < vram_total * 0.90]
        chosen = max(candidates, key=lambda x: x.get("tps", 0)) if candidates else best_speed
        print(f"    writing chosen to profile: {chosen['label']}")
        load_section(c, m, chosen["settings"])
    write_ini(c)
    print("\n[scan] done ->", out_path)
    print("[scan] best configs written to", INI)


if __name__ == "__main__":
    main()
