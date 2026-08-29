#!/usr/bin/env python3
"""Settings scan: find the best llama.cpp runtime settings per model by
resource-guided adaptation (works for both MoE and Dense models).

Method (per user contract):
  1. Read system data (VRAM total/used, system RAM) and the model GGUF
     metadata (layer count, KV-head geometry, file size) to compute a smart
     baseline config (kv quant + n-gpu-layers) for the target context.
  2. Restart llama-server and send a trivial hello query. The combo advances
     ONLY if the model actually produced a response - speed is a ranking
     signal, never a pass/fail gate. No response => escalate (MoE: bump
     n-cpu-moe; then both: lower n-gpu-layers) and retry, bounded.
  3. Run the long-form probe; only a complete, closed-fence response counts.
  4. Hill-climb around the working baseline (more GPU layers / kv upgrade /
     MoE knob down) while the full probe keeps producing valid responses and
     VRAM stays under budget; keep the fastest valid config.
  5. Write the winner to models.ini AND mirror it into the model's
     .profile.json (proxy reindex protection). No valid response at any phase
     => model marked FAILED, profile left untouched.

Results (settings, tps, VRAM, and the full response text of every probe) are
appended incrementally to data/llm_benchmarks/settings_scan.json.

Usage:
    sudo python settings_scan.py [--models m1,m2] [--ctx 65536]
"""

import argparse
import configparser
import json
import os
import re
import struct
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
DENSE_MAX_TOKENS = 3584  # long-form probe: observed need 2914 tok at temp=1.0; 3072 truncated 1 of 2 runs
PROBE_TIMEOUT_S = 2400.0  # 2048 tok at a ~1 tok/s dense floor needs up to ~40 min

# Hello pre-flight: the response-establishment gate. The model rambles on a
# trivial prompt (observed: prompt echo + inline reasoning), so give it enough
# tokens to land on something visible.
HELLO_PROMPT = "Reply with exactly: hi"
HELLO_NUM_PREDICT = 96
HELLO_TIMEOUT_S = 180.0
HELLO_MIN_TPS = 1.0  # memory-pressure signal for the MoE escalation path only

CTX_TARGET_DEFAULT = 65536  # user: context must be at least 64K
VRAM_OVERHEAD_MB = 600  # cuda buffers/activation headroom held back from estimates
VRAM_TARGET_FRAC = 0.85  # guarded margin after the 64K hard reboot
GUARD_STALL_READ_S = 120.0  # abort probe when no token flows for 2 min (pre-crash stall signature)
GUARD_VRAM_HEADROOM_MB = 1500
GUARD_RAM_HEADROOM_MB = 4000
GUARD_MAX_TEMP_C = 82
# Pre-cool gate: after a restart the 64K load/KV-alloc burst leaves Tctl high;
# probing into that residual heat caused every tok=0 thermal abort at 64K.
GUARD_PRECOOL_C = 76.0
GUARD_PRECOOL_TIMEOUT_S = 240.0
GUARD_UNPIN_CTX = 32768  # at/above this ctx disable mlock/no-mmap so RAM pressure pages, not panics
LAST_GUARD_ERROR = ""
# In-run thermal watchdog (hard-reboot protection): sample temps DURING generation,
# abort the probe and the whole scan when the box runs too hot. AMD Tctl trips ~95C.
THERMAL_CPU_ABORT_C = 93.0  # user-raised: 90 fired during the 64K load/prefill spike before generation started
# Active cooling: at THERMAL_THROTTLE_C stop reading the stream; the server blocks on
# its socket buffer and generation stalls, letting the CPU cool, then resume reads.
THERMAL_THROTTLE_C = 85.0
THERMAL_RESUME_C = 78.0
THERMAL_CHECK_INTERVAL_S = 5.0
THERMAL_LOG_INTERVAL_S = 30.0
THERMAL_STOP = False


def read_temps() -> dict:
    """GPU temp via nvidia-smi, CPU package temp via lm-sensors Tctl (AMD)."""
    return {"gpu": gpu_temp_c(), "cpu": _cpu_tctl_c()}


def _cpu_tctl_c() -> float | None:
    try:
        out = subprocess.run(["sensors"], capture_output=True, text=True, timeout=5, check=False)
        m = re.search(r"Tctl:\s+\+?([\d.]+)", out.stdout)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def thermal_exceeded(temps: dict) -> tuple[bool, str]:
    """True when any temperature crosses its abort threshold."""
    gpu, cpu = temps.get("gpu"), temps.get("cpu")
    if gpu is not None and gpu >= GUARD_MAX_TEMP_C:
        return True, f"gpu={gpu}C >= {GUARD_MAX_TEMP_C}C"
    if cpu is not None and cpu >= THERMAL_CPU_ABORT_C:
        return True, f"cpu Tctl={cpu}C >= {THERMAL_CPU_ABORT_C}C"
    return False, ""


def gpu_temp_c() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def probe_guard(shape: dict, settings: dict, ctx: int, vram_total: int) -> tuple[bool, str]:
    """Safety pre-flight: VRAM/RAM headroom and GPU temperature before a probe."""
    temp = gpu_temp_c()
    if temp is not None and temp >= GUARD_MAX_TEMP_C:
        return False, f"gpu temp {temp}C >= {GUARD_MAX_TEMP_C}C - cooling down"
    if not shape:
        return True, ""
    n_layers = int(shape.get("n_layers") or 0)
    if not n_layers:
        return True, ""
    try:
        ngl = int(settings.get("n-gpu-layers") or 0)
    except ValueError:
        ngl = 0
    kv_bpt = kv_bytes_per_token(shape, settings.get("cache-type-k") or "f16")
    kv_total_mb = (kv_bpt * ctx / 1e6) if kv_bpt else 0.0
    kv_in_ram = str(settings.get("no-kv-offload", "")).lower() in ("1", "true", "yes", "on")
    weights_mb = float(shape.get("file_size_mb") or 0)
    gpu_est = weights_mb * (ngl / n_layers) + (0.0 if kv_in_ram else kv_total_mb) + GUARD_VRAM_HEADROOM_MB
    free_vram = vram_total - (vram_mb() or 0) + llama_server_vram_mb()
    if free_vram < gpu_est:
        return False, f"vram guard: est {gpu_est:.0f} MB > free {free_vram} MB (ngl={ngl})"
    cpu_layers = max(n_layers - ngl, 0)
    kv_cpu_mb = kv_total_mb if kv_in_ram else kv_total_mb * (cpu_layers / n_layers)
    avail_ram = ram_available_mb()
    if kv_cpu_mb + GUARD_RAM_HEADROOM_MB > avail_ram:
        return False, f"ram guard: cpu-kv est {kv_cpu_mb:.0f} MB + {GUARD_RAM_HEADROOM_MB} > avail {avail_ram} MB"
    return True, ""


KV_BYTES_PER_ELEM = {"f16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625}  # incl. block scale overhead
MOE_HIER = ["", "28", "30", "31", "34", "36", "40", "48"]  # n-cpu-moe escalation (higher = less VRAM)


def hello_verdict(hello: dict) -> tuple[bool, str]:
    """A combo may proceed only if the model actually produced a response.

    The response artifact is the benchmark; tokens/speed just rank candidates.
    """
    if not hello.get("success") or not (hello.get("content") or "").strip():
        return False, hello.get("error") or "no response content"
    return True, ""


def effective_num_predict(is_moe: bool) -> int:
    """Full-probe token budget: dense models cannot finish MAX_TOKENS in the timeout window."""
    return MAX_TOKENS if is_moe else min(MAX_TOKENS, DENSE_MAX_TOKENS)


def read_ini():
    c = configparser.ConfigParser(delimiters=("=",))
    c.read(str(INI))
    return c


def write_ini(c):
    with open(INI, "w") as f:
        c.write(f)
    os.chmod(INI, 0o666)


def _gguf_candidates(model_path: str) -> list[Path]:
    candidates = [Path(model_path)]
    host_link = REPO / ".alpaca-router" / Path(model_path).name
    candidates.append(host_link)
    if host_link.is_symlink():
        try:
            tgt = host_link.readlink()
            candidates.append(Path("/usr/share/ollama/.ollama/models/blobs") / Path(tgt).name)
        except Exception:
            pass
    return candidates


def read_gguf_shape(model_path: str) -> dict:
    """Read model geometry from GGUF metadata for resource estimation.

    Returns {} when the file cannot be read (script must run with sudo for
    /usr/share/ollama blobs). Keys: n_layers, n_head_kv, head_dim,
    expert_count, file_size_mb, arch.
    """
    for cand in _gguf_candidates(model_path):
        try:
            with open(cand, "rb") as f:
                if f.read(4) != b"GGUF":
                    continue
                f.read(4)
                f.read(8)
                kv_count = struct.unpack("<Q", f.read(8))[0]
                meta: dict = {}
                arch = ""
                for _ in range(min(kv_count, 400)):
                    key_len = struct.unpack("<Q", f.read(8))[0]
                    k = f.read(key_len).decode(errors="replace")
                    vt = struct.unpack("<I", f.read(4))[0]
                    val = None
                    if vt == 8:
                        sl = struct.unpack("<Q", f.read(8))[0]
                        val = f.read(sl).decode(errors="replace")
                    elif vt in (0, 1, 2, 3, 4, 5, 10, 11):
                        fmt2 = {
                            0: "<B",
                            1: "<b",
                            2: "<H",
                            3: "<h",
                            4: "<I",
                            5: "<i",
                            10: "<Q",
                            11: "<q",
                        }[vt]
                        val = struct.unpack(fmt2, f.read(struct.calcsize(fmt2)))[0]
                    elif vt == 6:
                        val = struct.unpack("<f", f.read(4))[0]
                    elif vt == 7:
                        val = bool(f.read(1)[0])
                    elif vt == 12:
                        val = struct.unpack("<d", f.read(8))[0]
                    elif vt == 9:
                        at = struct.unpack("<I", f.read(4))[0]
                        al = struct.unpack("<Q", f.read(8))[0]
                        if at == 8:
                            total = 0
                            for _ in range(al):
                                sl = struct.unpack("<Q", f.read(8))[0]
                                f.read(sl)
                                total += 8 + sl
                            val = total
                        else:
                            sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
                            total = al * sizes.get(at, 0)
                            f.read(total)
                            val = total
                    else:
                        break
                    if k == "general.architecture":
                        arch = val or ""
                    if val is not None:
                        meta[k] = val
            if not arch:
                continue
            return {
                "arch": arch,
                "n_layers": int(meta.get(f"{arch}.block_count", 0) or 0),
                "n_head_kv": int(meta.get(f"{arch}.attention.head_count_kv", 0) or 0),
                "head_dim": int(meta.get(f"{arch}.attention.key_length", 0) or 0),
                "expert_count": int(meta.get(f"{arch}.expert_count", 0) or 0),
                "file_size_mb": round(cand.stat().st_size / 1e6, 1),
            }
        except Exception:
            continue
    return {}


def is_moe_model(model_path: str) -> bool:
    """Detect MoE vs Dense by reading GGUF metadata (expert_count key)."""
    shape = read_gguf_shape(model_path)
    if shape:
        return shape["expert_count"] > 0
    # Fallback: scan header keys directly (no geometry needed)
    for cand in _gguf_candidates(model_path):
        try:
            with open(cand, "rb") as f:
                if f.read(4) != b"GGUF":
                    continue
                f.read(4)
                f.read(8)
                kv_count = struct.unpack("<Q", f.read(8))[0]
                for _ in range(min(kv_count, 300)):
                    key_len = struct.unpack("<Q", f.read(8))[0]
                    k = f.read(key_len).decode(errors="replace")
                    vt = struct.unpack("<I", f.read(4))[0]
                    if k.endswith(".expert_count") or k.endswith(".expert_used_count"):
                        return True
                    if "moe" in k.lower() and "architecture" in k.lower():
                        return True
                    if not _gguf_skip_value(f, vt):
                        break
        except Exception:
            continue
    return False


def _gguf_skip_value(f, vt: int) -> bool:
    """Advance the GGUF file past one metadata value. False on unknown type."""
    try:
        if vt == 8:
            sl = struct.unpack("<Q", f.read(8))[0]
            f.read(sl)
        elif vt in (0, 1, 4, 5, 10, 11):
            fmt2 = {0: "<B", 1: "<b", 4: "<I", 5: "<i", 10: "<Q", 11: "<q"}[vt]
            f.read(struct.calcsize(fmt2))
        elif vt == 7:
            f.read(4)
        elif vt == 6 or vt == 12:
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
            return False
        return True
    except Exception:
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


def vram_total_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return int(out.stdout.strip().split()[0])
    except Exception:
        return 0


def llama_server_vram_mb() -> int:
    """VRAM currently held by llama-server processes.

    apply_and_restart restarts the server (releasing this memory) right after
    the guard runs, so the guard must add it back or every estimate false-positives
    while a model is resident.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        total = 0
        for line in out.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3 and "llama-server" in parts[1]:
                total += int(parts[2] or 0)
        return total
    except Exception:
        return 0


def ram_available_mb():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except Exception:
        pass
    return 0


def kv_bytes_per_token(shape: dict, kv_quant: str) -> float:
    """Estimated KV cache bytes per token: 2 (K+V) * layers * kv_heads * head_dim * bytes."""
    n_layers = shape.get("n_layers") or 48
    n_head_kv = shape.get("n_head_kv") or 8
    head_dim = shape.get("head_dim") or 128
    bpc = KV_BYTES_PER_ELEM.get(kv_quant, 2.0)
    return 2 * n_layers * n_head_kv * head_dim * bpc


def compute_smart_baseline(
    shape: dict,
    ctx: int,
    kv_choices: list[str],
    vram_total: int,
    overhead_mb: int = VRAM_OVERHEAD_MB,
    target_frac: float = VRAM_TARGET_FRAC,
    is_moe: bool = False,
) -> tuple[dict, int]:
    """Pick kv quant + n-gpu-layers maximizing GPU-offloaded layers within budget.

    Returns (settings, est_vram_mb). Ties prefer the earlier (higher quality)
    kv quant. Falls back to a conservative config when geometry is unknown.
    The KV-in-RAM lever (--no-kv-offload) was retired: KV-over-PCIe decode
    collapses long-form generation (see comment below).
    """
    n_layers = shape.get("n_layers") or 0
    file_mb = shape.get("file_size_mb") or 0.0
    if not n_layers or not file_mb:
        return {
            "ctx-size": str(ctx),
            "cache-type-k": "q4_0",
            "cache-type-v": "q4_0",
            "n-gpu-layers": "26",
            "flash-attn": "on",
        }, 0
    weights_per_layer_mb = file_mb / n_layers
    best: tuple[int, str, int] | None = None  # (ngl, kv, est_vram)
    for kv in kv_choices:  # ordered by quality: q8_0 first
        kv_total_mb = kv_bytes_per_token(shape, kv) * ctx / 1e6
        budget = vram_total * target_frac - kv_total_mb - overhead_mb
        ngl = max(0, min(n_layers, int(budget / weights_per_layer_mb)))
        est = int(ngl * weights_per_layer_mb + kv_total_mb + overhead_mb)
        if best is None or ngl > best[0]:
            best = (ngl, kv, est)
    # KV-in-RAM lever (--no-kv-offload) RETIRED: live evidence shows KV-over-PCIe
    # collapses long-form generation (2048-token probe yielded 86 usable chars
    # at ngl=16 vs 3449 chars with KV in VRAM). More GPU layers is the wrong
    # trade when every decode token must cross PCIe.
    ngl, kv, est = best  # type: ignore[misc]
    return {
        "ctx-size": str(ctx),
        "cache-type-k": kv,
        "cache-type-v": kv,
        "n-gpu-layers": str(ngl),
        "flash-attn": "on",
    }, est


def label_for(s: dict) -> str:
    label = f"ctx={s.get('ctx-size')} kv={s.get('cache-type-k')} ngl={s.get('n-gpu-layers')} fa={s.get('flash-attn')}"
    if s.get("n-cpu-moe"):
        label += f" moe={s['n-cpu-moe']}"
    return label


def escalate_step(settings: dict, shape: dict, is_moe: bool) -> dict | None:
    """Next more-conservative config (frees VRAM) or None when exhausted.

    MoE first bumps n-cpu-moe up the hierarchy; both arches then step
    n-gpu-layers down. Never returns an ngl below 8.
    """
    s = dict(settings)
    if is_moe:
        cur = s.get("n-cpu-moe", "") or ""
        if cur in MOE_HIER:
            idx = MOE_HIER.index(cur)
            if idx + 1 < len(MOE_HIER):
                s["n-cpu-moe"] = MOE_HIER[idx + 1]
                return s
    n_layers = shape.get("n_layers") or 48
    step = max(4, n_layers // 16)
    ngl = int(s.get("n-gpu-layers", "0") or 0) - step
    if ngl < 8:
        return None
    s["n-gpu-layers"] = str(ngl)
    return s


def _est_vram_mb(s: dict, shape: dict, ctx: int) -> int:
    n_layers = shape.get("n_layers") or 48
    file_mb = shape.get("file_size_mb") or 0.0
    wpl = file_mb / n_layers if n_layers else 0.0
    kv_mb = kv_bytes_per_token(shape, s.get("cache-type-k", "q8_0")) * ctx / 1e6
    return int(int(s.get("n-gpu-layers", "0") or 0) * wpl + kv_mb + VRAM_OVERHEAD_MB)


def hill_neighbors(
    settings: dict,
    shape: dict,
    is_moe: bool,
    ctx: int,
    vram_total: int,
    target_frac: float = VRAM_TARGET_FRAC,
) -> list[dict]:
    """Ordered upgrade variants of a working baseline (faster if they fit)."""
    out: list[dict] = []
    budget = vram_total * target_frac
    n_layers = shape.get("n_layers") or 48
    step = max(2, n_layers // 24)

    def fits(s: dict) -> bool:
        return _est_vram_mb(s, shape, ctx) <= budget

    up = dict(settings)
    ngl = int(up.get("n-gpu-layers", "0") or 0)
    if ngl < n_layers:
        up["n-gpu-layers"] = str(min(n_layers, ngl + step))
        if fits(up):
            out.append(up)
    if settings.get("cache-type-k") == "q4_0":
        kvu = dict(settings)
        kvu["cache-type-k"] = "q8_0"
        kvu["cache-type-v"] = "q8_0"
        if fits(kvu):
            out.append(kvu)
    if is_moe:
        cur = settings.get("n-cpu-moe", "") or ""
        if cur in MOE_HIER and MOE_HIER.index(cur) > 0:
            down = dict(settings)
            down["n-cpu-moe"] = MOE_HIER[MOE_HIER.index(cur) - 1]
            if fits(down):
                out.append(down)
    return [n for n in out if label_for(n) != label_for(settings)]


def write_profile_mirror(model: str, c) -> Path:
    """Mirror a models.ini section into .alpaca-router/{model}.profile.json.

    The proxy refreshes profiles from models.ini on reindex (alpaca-proxy.py
    ~5911) but keeps the file as source of truth between reindexes - keep it
    in sync so the winning settings survive.
    """
    prof_path = REPO / ".alpaca-router" / f"{model}.profile.json"
    prof: dict = {}
    if prof_path.exists():
        try:
            prof = json.loads(prof_path.read_text())
        except Exception:
            prof = {}
    for k, v in c[model].items():
        prof[k] = v
    prof_path.write_text(json.dumps(prof, indent=2) + "\n")
    return prof_path


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


def run_probe(model, think=False, budget=None, quick=False, num_predict=None):
    # think defaults OFF: settings scanning measures raw generation throughput;
    # a thinking phase can eat the whole token budget at dense-spill speeds.
    # Streams NDJSON: keeps per-chunk read gaps small so intermediate proxies'
    # read timeouts (600-1200s) can never kill a long generation, and records
    # first-token time to expose model stalls (alloc/VRAM thrash) vs slow gen.
    prompt = HELLO_PROMPT if quick else PROBE
    tokens = num_predict or (HELLO_NUM_PREDICT if quick else MAX_TOKENS)
    timeout = HELLO_TIMEOUT_S if quick else PROBE_TIMEOUT_S
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "think": think,
        "options": {"num_predict": tokens, "temperature": _model_temperature(model)},
    }
    if budget is not None:
        payload["reasoning_budget"] = budget
    t0 = time.time()
    try:
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        eval_count = 0
        eval_duration = 0.0
        ttft_ms = None
        peak_gpu = peak_cpu = None
        last_check = last_log = 0.0
        with httpx.stream(
            "POST",
            f"{PROXY_URL}/api/chat",
            json=payload,
            timeout=httpx.Timeout(timeout, connect=10.0, read=GUARD_STALL_READ_S, write=60.0),
        ) as r:
            if r.status_code != 200:
                body = r.read().decode(errors="replace")[:200]
                return {"success": False, "error": f"HTTP {r.status_code}: {body}", "tps": 0, "content": ""}
            for line in r.iter_lines():
                # In-run thermal watchdog: sample temps while tokens flow
                now = time.time()
                if now - last_check >= THERMAL_CHECK_INTERVAL_S:
                    last_check = now
                    temps = read_temps()
                    g, cp = temps.get("gpu"), temps.get("cpu")
                    if g is not None:
                        peak_gpu = g if peak_gpu is None else max(peak_gpu, g)
                    if cp is not None:
                        peak_cpu = cp if peak_cpu is None else max(peak_cpu, cp)
                    hot, hreason = thermal_exceeded(temps)
                    if hot:
                        global THERMAL_STOP
                        THERMAL_STOP = True
                        print(f"[thermal] ABORT probe: {hreason} (peak gpu={peak_gpu}C cpu={peak_cpu}C)")
                        return {
                            "success": False,
                            "thermal_abort": True,
                            "error": f"thermal guard: {hreason}",
                            "tps": 0,
                            "content": "".join(content_parts),
                            "peak_gpu_c": peak_gpu,
                            "peak_cpu_c": peak_cpu,
                            "elapsed_s": round(now - t0, 2),
                        }
                    if cp is not None and cp >= THERMAL_THROTTLE_C:
                        # Active cooling: stop reading; server backpressure stalls generation.
                        throttled = True
                        while throttled:
                            time.sleep(3)
                            temps = read_temps()
                            cp = temps.get("cpu")
                            g = temps.get("gpu")
                            if g is not None:
                                peak_gpu = g if peak_gpu is None else max(peak_gpu, g)
                            if cp is not None:
                                peak_cpu = cp if peak_cpu is None else max(peak_cpu, cp)
                            hot, hreason = thermal_exceeded(temps)
                            if hot:
                                THERMAL_STOP = True
                                print(f"[thermal] ABORT probe during throttle: {hreason}")
                                return {
                                    "success": False,
                                    "thermal_abort": True,
                                    "error": f"thermal guard: {hreason}",
                                    "tps": 0,
                                    "content": "".join(content_parts),
                                    "peak_gpu_c": peak_gpu,
                                    "peak_cpu_c": peak_cpu,
                                    "elapsed_s": round(time.time() - t0, 2),
                                }
                            if cp is None or cp <= THERMAL_RESUME_C:
                                throttled = False
                                print(
                                    f"[thermal] cooled to {cp}C after {round(time.time() - now)}s pause - resuming (t={time.time() - t0:.0f}s tok={eval_count})"
                                )
                    if now - last_log >= THERMAL_LOG_INTERVAL_S:
                        last_log = now
                        print(f"[thermal] t={now - t0:.0f}s gpu={g}C cpu={cp}C tok={eval_count}")
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError:
                    continue
                if isinstance(chunk.get("error"), (str, dict)):
                    return {
                        "success": False,
                        "error": str(chunk["error"])[:200],
                        "tps": 0,
                        "content": "".join(content_parts),
                    }
                msg = chunk.get("message") or {}
                piece = msg.get("content") or chunk.get("response") or ""
                if piece:
                    if ttft_ms is None:
                        ttft_ms = round((time.time() - t0) * 1000)
                    content_parts.append(piece)
                tpiece = msg.get("thinking") or ""
                if tpiece:
                    thinking_parts.append(tpiece)
                if chunk.get("done"):
                    eval_count = chunk.get("eval_count", 0) or 0
                    eval_duration = (chunk.get("eval_duration", 0) or 0) / 1e9
        content = "".join(content_parts)
        thinking = "".join(thinking_parts)
        elapsed = time.time() - t0
        tps = (
            round(eval_count / eval_duration, 1)
            if eval_duration
            else (round(eval_count / elapsed, 1) if elapsed else 0)
        )
        return {
            "success": bool(content.strip()),
            "content": content,
            "content_len": len(content),
            "thinking_len": len(thinking),
            "tokens": eval_count,
            "ttft_ms": ttft_ms,
            "elapsed_s": round(elapsed, 2),
            "tps": tps,
            "error": None if content.strip() else "EMPTY CONTENT",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)[:200],
            "elapsed_s": round(time.time() - t0, 2),
            "tps": 0,
            "content": "".join(content_parts),
        }


def long_response_valid(res: dict) -> tuple[bool, str]:
    """Long-form probe validity: a complete fenced code response, not truncated.

    The PROBE prompt demands ONLY a single fenced code block; a combo whose
    settings cannot produce a complete, closed response fails the benchmark.
    """
    if not res.get("success"):
        return False, res.get("error") or "probe failed"
    content = res.get("content") or ""
    if len(content) < 400:
        return False, f"response too short ({len(content)} chars)"
    if content.count("```") < 2:
        return False, "no closed code fence (truncated long response)"
    return True, ""


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
    # Guardrail: at large context, never pin RAM (mlock+no-mmap + spilled dense
    # weights contributed to the hard reboot); let mmap page gracefully.
    try:
        if int(c[section].get("ctx-size", "") or 0) >= GUARD_UNPIN_CTX:
            c[section]["mlock"] = "false"
            c[section]["no-mmap"] = "false"
    except ValueError:
        pass


def main():
    ap = argparse.ArgumentParser(
        description="Resource-guided settings search: establish a working hello response, "
        "verify a valid long response, then hill-climb to the fastest valid config. "
        "Works for both MoE and Dense models."
    )
    ap.add_argument(
        "--models",
        default="",
        help="Comma-separated model sections from models.ini, or empty/all to scan every model",
    )
    ap.add_argument("--ctx", type=int, default=CTX_TARGET_DEFAULT, help="target context size (default 65536)")
    ap.add_argument("--max-attempts", type=int, default=6, help="max response-establishment escalations per model")
    ap.add_argument("--max-hill", type=int, default=5, help="max hill-climb probes per model")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--skip-restart", action="store_true")
    args = ap.parse_args()

    c = read_ini()
    if not args.models or args.models.strip().lower() == "all":
        models = [s for s in c.sections() if s != "*"]
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

    vram_total = vram_total_mb() or 8188
    print(f"[scan] VRAM total: {vram_total} MB, RAM available: {ram_available_mb()} MB")

    def save():
        out_path.write_text(json.dumps(results, indent=2))

    for model in models:
        if THERMAL_STOP:
            print(f"[scan] thermal stop - skipping remaining model {model}")
            continue
        if not c.has_section(model):
            print(f"[scan] section {model} not in models.ini, skipping")
            continue
        model_path = c[model].get("model", "")
        shape = read_gguf_shape(model_path)
        is_moe = is_moe_model(model_path)
        print(
            f"\n[scan] === {model} ({'MoE' if is_moe else 'Dense'}, shape={'ok' if shape else 'unknown'}, "
            f"{shape.get('file_size_mb', '?')} MB, {shape.get('n_layers', '?')} layers) ==="
        )

        cur_model = model

        def record(key: str, data: dict, _model: str = cur_model):
            data["done"] = True
            data["model"] = _model
            results[key] = data
            save()

        last_applied: dict | None = None

        def apply_and_restart(settings: dict, _model: str = cur_model, _shape: dict = shape) -> bool:
            nonlocal last_applied
            if settings == last_applied:
                print("[scan] settings unchanged - skipping restart")
                return True
            global LAST_GUARD_ERROR
            ok, greason = probe_guard(_shape, settings, args.ctx, vram_total)
            if not ok:
                LAST_GUARD_ERROR = greason
                print(f"    [guard] BLOCKED {label_for(settings)}: {greason}")
                time.sleep(GUARD_STALL_READ_S / 4)  # cool down before next attempt
                return False
            LAST_GUARD_ERROR = ""
            load_section(c, _model, settings)
            write_ini(c)
            if args.skip_restart:
                return True
            if not restart_llama():
                return False
            last_applied = dict(settings)
            time.sleep(20)  # settle: let mmap/page-cache warm before probing
            # Pre-cool gate: wait out the residual load/KV-alloc heat before the
            # first probe - probing into it produced every tok=0 thermal abort.
            cool_t0 = time.time()
            while True:
                temps = read_temps()
                cp = temps.get("cpu")
                if cp is None or cp <= GUARD_PRECOOL_C:
                    break
                if time.time() - cool_t0 > GUARD_PRECOOL_TIMEOUT_S:
                    print(f"[guard] pre-cool timeout at {cp}C - proceeding")
                    break
                if int(time.time() - cool_t0) % 30 == 0:
                    print(f"[guard] pre-cool: waiting for Tctl {cp}C <= {GUARD_PRECOOL_C}C")
                time.sleep(5)
            return True

        # ---- Phase A: establish a working hello response (smart guessing) ----
        baseline, est = compute_smart_baseline(shape, args.ctx, ["q8_0", "q4_0"], vram_total, is_moe=is_moe)
        if est:
            print(f"[scan] smart baseline: {label_for(baseline)} (est {est} MB VRAM)")
        else:
            print(f"[scan] GGUF shape unknown - conservative start: {label_for(baseline)}")

        working: dict | None = None
        cur = dict(baseline)
        attempt_retried = False
        resumed_settings: dict | None = None
        if f"{model}::response" in results and results[f"{model}::response"].get("success"):
            resumed_settings = dict(results[f"{model}::response"]["settings"])

        def _ngl_of(s: dict) -> int:
            try:
                return int(s.get("n-gpu-layers") or 0)
            except (TypeError, ValueError):
                return 0

        # A stale resume (e.g. ngl=6 left by pre-lever escalation thrash) must
        # never block a strictly better fresh baseline - compare GPU layers.
        if resumed_settings is not None and _ngl_of(resumed_settings) >= _ngl_of(baseline):
            working = resumed_settings
            print(f"[scan] resume: response already established at {label_for(working)}")
        else:
            if resumed_settings is not None:
                print(
                    f"[scan] stale resume (ngl={_ngl_of(resumed_settings)}) < baseline "
                    f"(ngl={_ngl_of(baseline)}) - re-probing baseline"
                )
            for attempt in range(1, args.max_attempts + 1):
                if THERMAL_STOP:
                    print("[scan] thermal stop - abandoning response escalation")
                    break
                print(f"[scan] response attempt {attempt}/{args.max_attempts}: {label_for(cur)}")
                if not apply_and_restart(cur):
                    record(
                        f"{model}::response::{label_for(cur)}",
                        {
                            "success": False,
                            "error": "llama-server did not become ready",
                            "settings": dict(cur),
                        },
                    )
                    break
                v0 = vram_mb()
                hello = run_probe(model, quick=True)
                v1 = vram_mb()
                ok, reason = hello_verdict(hello)
                record(
                    f"{model}::response::{label_for(cur)}",
                    {
                        "success": ok,
                        "error": None if ok else f"no response: {reason}",
                        "settings": dict(cur),
                        "label": label_for(cur),
                        "hello_content": hello.get("content", ""),
                        "tokens": hello.get("tokens", 0),
                        "tps": hello.get("tps", 0),
                        "elapsed_s": hello.get("elapsed_s"),
                        "vram_before_mb": v0,
                        "vram_after_mb": v1,
                        "vram_peak_mb": max(v0, v1),
                        "attempt": attempt,
                    },
                )
                if ok:
                    working = dict(cur)
                    record(
                        f"{model}::response",
                        {"success": True, "settings": dict(working), "label": label_for(working)},
                    )
                    print(f"[scan] RESPONSE OK ({hello.get('tps')} tok/s): {hello.get('content', '')[:80]!r}")
                    break
                # Empty content is often non-deterministic (ramble vs blank) -
                # retry the same settings once before burning a restart+load cycle.
                if reason.startswith("empty") and not attempt_retried:
                    attempt_retried = True
                    print(f"[scan] no response ({reason[:60]}) - retrying same settings once")
                    continue
                print(f"[scan] no response ({reason[:60]}) - escalating")
                nxt = escalate_step(cur, shape, is_moe)
                if nxt is None:
                    break
                cur = nxt
                attempt_retried = False
            if working is None:
                # Escalation exhausted - try a previously-proven config (labeled
                # response:: entries are durable evidence across runs).
                for rkey, rres in sorted(results.items()):
                    if not rkey.startswith(f"{model}::response::"):
                        continue
                    if not rres.get("success") or not rres.get("settings"):
                        continue
                    rset = dict(rres["settings"])
                    print(f"[scan] escalation exhausted - retrying previously-proven {label_for(rset)}")
                    if apply_and_restart(rset):
                        hello = run_probe(model, quick=True)
                        if hello_verdict(hello)[0]:
                            working = rset
                            record(
                                f"{model}::response",
                                {"success": True, "settings": dict(working), "label": label_for(working)},
                            )
                            print(f"[scan] RESPONSE OK (proven config): {label_for(working)}")
                            break
            if working is None:
                record(
                    f"{model}::failed",
                    {
                        "success": False,
                        "error": "no hello response at any attempted config - profile NOT changed",
                        "settings": dict(cur),
                        "ram_available_mb": ram_available_mb(),
                    },
                )
                print(f"[scan] FAILED {model}: no response - profile NOT changed")
                continue

        # ---- Phase B: valid long response at the working config ----
        num_predict = effective_num_predict(is_moe)
        base_label = label_for(working)
        long_key = f"{model}::long::{base_label}"
        winner_settings = dict(working)
        winner_res: dict | None = None
        if long_key in results and results[long_key].get("done") and results[long_key].get("valid"):
            winner_res = results[long_key]
            print(f"[scan] resume: long probe already valid at {base_label}")
        else:
            print(f"[scan] long probe ({num_predict} tok) at {base_label}")
            if not apply_and_restart(working):
                record(
                    long_key,
                    {"success": False, "valid": False, "error": "llama-server not ready", "settings": dict(working)},
                )
            else:
                v0 = vram_mb()
                res = run_probe(model, num_predict=num_predict)
                v1 = vram_mb()
                valid, vreason = long_response_valid(res)
                res.update(
                    {
                        "valid": valid,
                        "validity_error": None if valid else vreason,
                        "label": base_label,
                        "settings": dict(working),
                        "vram_before_mb": v0,
                        "vram_after_mb": v1,
                        "vram_peak_mb": max(v0, v1),
                    }
                )
                record(long_key, res)
                if valid:
                    winner_res = res
                    print(
                        f"[scan] LONG OK: {res.get('tps')} tok/s, {res.get('content_len')} chars, peak {max(v0, v1)} MB"
                    )
                else:
                    if res.get("thermal_abort"):
                        # Thermal-aware retry: reduce CPU threads (package power)
                        # rather than GPU layers - generation heat scales with
                        # thread count, and the box trips near 95C. llama-server
                        # defaults to PHYSICAL cores (logical // 2 on SMT).
                        default_threads = max(4, (os.cpu_count() or 8) // 2)
                        cur_threads = int(working.get("threads") or default_threads)
                        if cur_threads - 2 >= 4:
                            low = dict(working)
                            low["threads"] = str(cur_threads - 2)
                            low_label = label_for(low) + f" thr={low['threads']}"
                            low_key = f"{model}::long::{low_label}"
                            print(f"[scan] long THERMAL ABORT - retry with threads={low['threads']}")
                            if apply_and_restart(low):
                                v0 = vram_mb()
                                res2 = run_probe(model, num_predict=num_predict)
                                v1 = vram_mb()
                                valid2, vreason2 = long_response_valid(res2)
                                res2.update(
                                    {
                                        "valid": valid2,
                                        "validity_error": None if valid2 else vreason2,
                                        "label": low_label,
                                        "settings": dict(low),
                                        "vram_before_mb": v0,
                                        "vram_after_mb": v1,
                                        "vram_peak_mb": max(v0, v1),
                                    }
                                )
                                record(low_key, res2)
                                if valid2:
                                    winner_res = res2
                                    print(
                                        f"[scan] LONG OK (thr={low['threads']}): {res2.get('tps')} tok/s, "
                                        f"{res2.get('content_len')} chars, peak {max(v0, v1)} MB"
                                    )
                                else:
                                    print(f"[scan] long retry at thr={low['threads']} invalid ({vreason2[:70]})")
                            else:
                                print(f"[scan] retry with threads={low['threads']} blocked by guard")
                        else:
                            print("[scan] long probe THERMAL ABORTED - threads floor reached (machine safety)")
                    else:
                        print(f"[scan] long probe invalid at baseline ({vreason[:70]}) - one escalation retry")
                        esc = escalate_step(working, shape, is_moe)
                        if esc is not None:
                            esc_label = label_for(esc)
                            esc_key = f"{model}::long::{esc_label}"
                            if apply_and_restart(esc):
                                hello = run_probe(model, quick=True)
                                if hello_verdict(hello)[0]:
                                    v0 = vram_mb()
                                    res2 = run_probe(model, num_predict=num_predict)
                                    v1 = vram_mb()
                                    valid2, vreason2 = long_response_valid(res2)
                                    res2.update(
                                        {
                                            "valid": valid2,
                                            "validity_error": None if valid2 else vreason2,
                                            "label": esc_label,
                                            "settings": dict(esc),
                                            "vram_before_mb": v0,
                                            "vram_after_mb": v1,
                                            "vram_peak_mb": max(v0, v1),
                                        }
                                    )
                                    record(esc_key, res2)
                                    if valid2:
                                        winner_settings, winner_res = dict(esc), res2
                            if winner_res is None:
                                print("[scan] escalation retry also failed")

        if winner_res is None:
            record(
                f"{model}::failed",
                {
                    "success": False,
                    "error": "no valid long response at any attempted config - profile NOT changed",
                    "settings": dict(working),
                },
            )
            print(f"[scan] FAILED {model}: long response never valid - profile NOT changed")
            continue

        # ---- Phase C: hill climb around the working baseline ----
        tried = {winner_settings and label_for(winner_settings)}
        hill_done = 0
        neighbors = hill_neighbors(winner_settings, shape, is_moe, args.ctx, vram_total)
        for nset in neighbors:
            if hill_done >= args.max_hill or THERMAL_STOP:
                break
            nlabel = label_for(nset)
            if nlabel in tried:
                continue
            tried.add(nlabel)
            nkey = f"{model}::hill::{nlabel}"
            print(f"[scan] hill: {nlabel}")
            if not apply_and_restart(nset):
                record(nkey, {"success": False, "error": "llama-server not ready", "settings": dict(nset)})
                continue
            hello = run_probe(model, quick=True)
            if not hello_verdict(hello)[0]:
                record(
                    nkey,
                    {
                        "success": False,
                        "error": "no response at neighbor config",
                        "settings": dict(nset),
                        "hello_content": hello.get("content", ""),
                    },
                )
                continue
            v0 = vram_mb()
            res = run_probe(model, num_predict=num_predict)
            v1 = vram_mb()
            valid, vreason = long_response_valid(res)
            res.update(
                {
                    "valid": valid,
                    "validity_error": None if valid else vreason,
                    "label": nlabel,
                    "settings": dict(nset),
                    "vram_before_mb": v0,
                    "vram_after_mb": v1,
                    "vram_peak_mb": max(v0, v1),
                }
            )
            record(nkey, res)
            hill_done += 1
            cur_best_tps = (winner_res or {}).get("tps", 0)
            if valid and res.get("tps", 0) > cur_best_tps:
                winner_settings, winner_res = dict(nset), res
                print(f"[scan] hill IMPROVED: {res.get('tps')} tok/s (was {cur_best_tps})")

        # ---- Phase D: write the winner to the model profile ----
        final_label = label_for(winner_settings)
        load_section(c, model, winner_settings)
        # Bench-verified marker: the proxy's VRAM budgeter checks this key and
        # leaves benchmarked settings alone (runtime budgeting is skipped).
        c[model]["bench-verified"] = "1"
        c[model]["bench-verified-at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        write_ini(c)
        prof_path = write_profile_mirror(model, c)
        print(
            f"[scan] WINNER {model}: {final_label} -> {winner_res.get('tps')} tok/s, "
            f"peak {winner_res.get('vram_peak_mb')} MB VRAM (models.ini + {prof_path.name} updated)"
        )

    print("\n[scan] done ->", out_path)
    print("[scan] winning configs written to", INI, "+ .profile.json mirrors")


if __name__ == "__main__":
    main()
