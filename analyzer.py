#!/usr/bin/env python3
"""
analyzer.py

Configuration Recommendation Engine.
Analyzes historical telemetry data and suggests specific llama.cpp
flag/ini adjustments to prevent system RAM and VRAM OOM crashes.
"""

import argparse
import configparser
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Constants
TELEMETRY_DIR = Path(os.getenv("TELEMETRY_DIR", "data/telemetry"))
ROUTER_MODELS_DIR = Path(os.getenv("ROUTER_MODELS_DIR", "/router-models"))
BENCHMARK_DIR = Path("data/llm_benchmarks")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("analyzer")


def load_telemetry(
    model_alias: str, limit: int = 500, max_age_seconds: int = 3600
) -> List[Dict[str, Any]]:
    """Load the latest telemetry data points for a specific model.

    Filters to the most recent *max_age_seconds* window so that stale data
    from prior sessions does not skew creep-detection slopes.
    """
    log_file = TELEMETRY_DIR / f"{model_alias}.jsonl"
    if not log_file.exists():
        logger.warning(f"Telemetry log file not found: {log_file}")
        return []

    points = []
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    points.append(json.loads(line))
        # Keep only recent points within the age window, then cap to limit
        cutoff = time.time() - max_age_seconds
        points = [p for p in points if p.get("epoch_time", 0) >= cutoff]
        return points[-limit:]
    except Exception as e:
        logger.error(f"Error reading telemetry log: {e}")
        return []


def read_current_config(model_alias: str) -> Dict[str, str]:
    """Read the current model settings from models.ini or profile.json."""
    config_dict = {}

    # Try models.ini first
    ini_path = ROUTER_MODELS_DIR / "models.ini"
    # Fallback to current directory models.ini for local testing
    if not ini_path.exists():
        ini_path = Path("data/models.ini")
    if not ini_path.exists():
        ini_path = Path("models.ini")

    if ini_path.exists():
        try:
            config = configparser.ConfigParser()
            config.read(ini_path)

            # Read defaults first
            if config.has_section("*"):
                for k, v in config["*"].items():
                    config_dict[k] = v

            # Overlay model-specific settings
            matched_section = None
            if config.has_section(model_alias):
                matched_section = model_alias
            else:

                def clean_str(s):
                    return s.replace("/", "").replace("_", "").replace("-", "").lower()

                clean_alias = clean_str(model_alias)
                for sec in config.sections():
                    clean_sec = clean_str(sec)
                    if (
                        clean_alias in clean_sec
                        or clean_sec in clean_alias
                        or clean_alias.replace("latest", "") in clean_sec
                    ):
                        matched_section = sec
                        break

            if matched_section:
                for k, v in config[matched_section].items():
                    config_dict[k] = v
                logger.info(
                    f"Loaded config from models.ini [{matched_section}] for alias [{model_alias}]"
                )
            else:
                logger.warning(f"No matching section in models.ini found for [{model_alias}]")
        except Exception as e:
            logger.warning(f"Failed to read models.ini: {e}")

    # If empty, try reading alias.profile.json
    profile_path = ROUTER_MODELS_DIR / f"{model_alias}.profile.json"
    if not profile_path.exists():
        profile_path = Path(f"{model_alias}.profile.json")

    if profile_path.exists():
        try:
            with open(profile_path, "r") as f:
                profile_data = json.load(f)
                config_dict.update({str(k): str(v) for k, v in profile_data.items()})
                logger.info(f"Loaded config overlays from profile {profile_path.name}")
        except Exception as e:
            logger.warning(f"Failed to read profile.json: {e}")

    return config_dict


def load_latest_benchmark(model_alias: str) -> Optional[Dict[str, Any]]:
    """Load latest performance benchmark data for baseline comparison."""
    if not BENCHMARK_DIR.exists():
        return None

    newest_file = None
    newest_mtime = 0.0

    # Search for latest performance benchmark results
    for path in BENCHMARK_DIR.glob("*.json"):
        if path.stat().st_mtime > newest_mtime:
            newest_mtime = path.stat().st_mtime
            newest_file = path

    if not newest_file:
        return None

    try:
        with open(newest_file, "r") as f:
            data = json.load(f)
            # Find the entry for our model
            for res in data.get("results", []):
                if (
                    res.get("model") == model_alias
                    or Path(res.get("model", "")).stem == model_alias
                ):
                    return res
    except Exception as e:
        logger.warning(f"Failed to load benchmark baseline: {e}")

    return None


def analyze_telemetry(
    model_alias: str,
    current_config: Optional[Dict[str, str]] = None,
    performance_first: bool = True,
) -> Dict[str, Any]:
    """Analyze telemetry data and return tuning recommendations."""
    points = load_telemetry(model_alias)
    if not points:
        return {
            "status": "insufficient_data",
            "model_alias": model_alias,
            "detected_issues": ["No telemetry data found for this model."],
            "recommendations": {},
            "explanation": "Please ensure the telemetry daemon is running and has gathered data for this model.",
        }

    # Fetch current config if not supplied
    if not current_config:
        current_config = read_current_config(model_alias)

    # Statistical aggregations
    ram_pcts = [p["system"]["ram_used_pct"] for p in points]
    cpu_pcts = [p["system"]["cpu_util_pct"] for p in points]

    # GPU parsing (focusing on primary GPU 0)
    gpu_pcts = []
    vram_pcts = []
    vram_used_mbs = []
    vram_totals = []

    for p in points:
        gpus = p.get("gpus", [])
        if gpus:
            primary_gpu = gpus[0]
            gpu_pcts.append(primary_gpu.get("gpu_util_pct", 0.0))
            vram_pcts.append(primary_gpu.get("vram_used_pct", 0.0))
            vram_used_mbs.append(primary_gpu.get("vram_used_mb", 0))
            vram_totals.append(primary_gpu.get("vram_total_mb", 0))
        else:
            gpu_pcts.append(0.0)
            vram_pcts.append(0.0)
            vram_used_mbs.append(0)
            vram_totals.append(0)

    tokens_cached_vals = [
        p["llama_server"]["slots"]["tokens_cached"]
        for p in points
        if "llama_server" in p and "slots" in p["llama_server"]
    ]
    if not tokens_cached_vals:
        tokens_cached_vals = [0]

    max_ram = max(ram_pcts)
    mean_ram = sum(ram_pcts) / len(ram_pcts)
    final_ram = ram_pcts[-1]

    max_vram = max(vram_pcts)
    final_vram = vram_pcts[-1]
    max_vram_used = max(vram_used_mbs)
    vram_total = max(vram_totals) if vram_totals else 0
    vram_headroom_mb = vram_total - max_vram_used

    max_cpu = max(cpu_pcts)
    mean_cpu = sum(cpu_pcts) / len(cpu_pcts)

    max_gpu = max(gpu_pcts)
    mean_gpu = sum(gpu_pcts) / len(gpu_pcts)

    max_tokens = max(tokens_cached_vals)

    # Dynamic Threshold Mapping
    if performance_first:
        # Relaxed thresholds: we are okay with occasional OOMs to maximize speed
        ram_critical = 97.5
        ram_warning = 94.0
        vram_critical = 98.0
        vram_warning = 95.0
    else:
        # Safe thresholds
        ram_critical = 92.0
        ram_warning = 85.0
        vram_critical = 95.0
        vram_warning = 85.0

    # Detect Creep: RAM growth trend (linear fit slope of the second half of data)
    creep_detected = False
    ram_slope = 0.0
    if len(ram_pcts) >= 10:
        half_idx = len(ram_pcts) // 2
        x = list(range(len(ram_pcts) - half_idx))
        y = ram_pcts[half_idx:]
        n = len(x)
        if n > 1:
            mean_x = sum(x) / n
            mean_y = sum(y) / n
            num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
            den = sum((x[i] - mean_x) ** 2 for i in range(n))
            ram_slope = num / den if den != 0 else 0.0
            # If slope is positive and RAM is near warnings, creep is occurring
            if ram_slope > 0.05 and final_ram > ram_warning:
                creep_detected = True

    # Identify issues
    issues = []
    status = "ok"

    if final_ram > ram_critical or max_ram > (ram_critical + 1.0):
        issues.append(
            f"CRITICAL: System RAM is nearly exhausted (Peak: {max_ram}%, Current: {final_ram}%). High risk of system OOM crash."
        )
        status = "critical"
    elif final_ram > ram_warning or (not performance_first and creep_detected):
        issues.append(f"WARNING: System RAM is highly utilized (Current: {final_ram}%).")
        status = "warning"

    if max_vram > vram_critical:
        issues.append(
            f"CRITICAL: VRAM is fully exhausted (Peak VRAM usage: {max_vram}%). High risk of CUDA OOM."
        )
        status = "critical"
    elif max_vram > vram_warning:
        issues.append(f"WARNING: VRAM is highly utilized (Peak VRAM usage: {max_vram}%).")
        if status != "critical":
            status = "warning"

    # Analyze current config to base recommendations — all INI values are strings
    # so we guard each cast to avoid ValueError on malformed or float-formatted values.
    def _safe_int(val: str, fallback: int) -> int:
        try:
            return int(float(val))  # handles "99.0" as well as "99"
        except (ValueError, TypeError):
            return fallback

    curr_ngl = _safe_int(current_config.get("n-gpu-layers", "-1"), -1)
    curr_ctx = _safe_int(current_config.get("ctx-size", "4096"), 4096)
    curr_cache_k = current_config.get("cache-type-k", "f16").lower()
    curr_cache_v = current_config.get("cache-type-v", "f16").lower()
    curr_batch = _safe_int(current_config.get("batch-size", "1024"), 1024)
    recommendations = {}
    actions = []

    # Load failed configs to prevent recommending failed setups
    failed_configs_file = Path("data/failed_configs.json")
    failed_configs = []
    if failed_configs_file.exists():
        try:
            with open(failed_configs_file, "r") as f:
                failed_configs = json.load(f)
        except Exception:
            pass

    def is_blacklisted(cache_k, cache_v, ngl=None, ctx=None):
        ngl_val = str(ngl) if ngl is not None else str(curr_ngl)
        ctx_val = str(ctx) if ctx is not None else str(curr_ctx)
        for fc in failed_configs:
            fc_model = fc.get("model", "")
            if model_alias in fc_model or fc_model in model_alias:
                if (
                    fc.get("cache-type-k") == cache_k
                    and fc.get("cache-type-v") == cache_v
                    and fc.get("n-gpu-layers") == ngl_val
                    and fc.get("ctx-size") == ctx_val
                ):
                    return True
        return False

    current_is_failed = is_blacklisted(curr_cache_k, curr_cache_v, curr_ngl, curr_ctx)

    if current_is_failed:
        status = "critical"
        issues.append(
            f"CRITICAL: The current configuration of {model_alias} is known to have failed to load recently."
        )

    # Recommendation Logic
    if current_is_failed:
        # Immediately suggest a downgrade path since the current settings failed
        if curr_cache_k in ("f16", "f32"):
            recommendations["cache-type-k"] = "q8_0"
            recommendations["cache-type-v"] = "q8_0"
            actions.append(
                "Downgrade KV Cache quantization (f16 -> q8_0) to resolve model load failure."
            )
        elif curr_cache_k == "q8_0":
            recommendations["cache-type-k"] = "q5_0"
            recommendations["cache-type-v"] = "q5_0"
            actions.append(
                "Downgrade KV Cache quantization (q8_0 -> q5_0) to resolve model load failure."
            )
        elif curr_cache_k in ("q5_0", "q5_1"):
            recommendations["cache-type-k"] = "q4_0"
            recommendations["cache-type-v"] = "q4_0"
            actions.append(
                "Downgrade KV Cache quantization (q5_0 -> q4_0) to resolve model load failure."
            )
        elif curr_ngl > 10:
            suggested_ngl = max(0, curr_ngl - 10)
            recommendations["n-gpu-layers"] = str(suggested_ngl)
            actions.append(
                f"Reduce GPU layers (n-gpu-layers: {curr_ngl} -> {suggested_ngl}) to free up VRAM."
            )
        elif curr_ctx > 8192:
            recommendations["ctx-size"] = "8192"
            actions.append("Reduce context window to 8192 to resolve load failure.")

    elif status in ("warning", "critical"):
        # 1. System RAM Exhaustion, but VRAM has Headroom
        # We shift layers to VRAM to relieve system RAM
        vram_limit = 95.0 if performance_first else 80.0
        vram_headroom_target = 600 if performance_first else 1500

        if (final_ram > ram_warning) and (
            max_vram < vram_limit or vram_headroom_mb > vram_headroom_target
        ):
            if curr_ngl != 99 and curr_ngl >= 0:
                step = 15 if performance_first else 10
                suggested_ngl = min(curr_ngl + step, 99)
                recommendations["n-gpu-layers"] = str(suggested_ngl)
                actions.append(
                    f"Increase GPU offloaded layers (n-gpu-layers: {curr_ngl} -> {suggested_ngl}) to offload system RAM to VRAM."
                )
            elif curr_ngl < 0:
                recommendations["n-gpu-layers"] = "99"
                actions.append(
                    "Force full GPU layer offloading (n-gpu-layers: 99) to minimize system memory footprint."
                )

        # 2. System RAM Exhaustion, and VRAM is ALSO exhausted (No Headroom)
        # We must reduce overall memory footprint.
        elif (final_ram > ram_warning) and (
            max_vram >= vram_limit or vram_headroom_mb <= vram_headroom_target
        ):
            # Performance strategy: try 8-bit cache (q8_0) first to protect quality
            if performance_first and (
                curr_cache_k in ("f16", "f32") or curr_cache_v in ("f16", "f32")
            ):
                if not is_blacklisted("q8_0", "q8_0"):
                    recommendations["cache-type-k"] = "q8_0"
                    recommendations["cache-type-v"] = "q8_0"
                    actions.append(
                        "Enable high-quality 8-bit KV Cache quantization (cache-type-k/v: f16 -> q8_0) to save 50% of cache memory with zero output degradation."
                    )
                elif not is_blacklisted("q5_0", "q5_0"):
                    recommendations["cache-type-k"] = "q5_0"
                    recommendations["cache-type-v"] = "q5_0"
                    actions.append(
                        "Enable 5-bit KV Cache quantization (cache-type-k/v: f16 -> q5_0) to save KV cache memory."
                    )
                else:
                    recommendations["cache-type-k"] = "q4_0"
                    recommendations["cache-type-v"] = "q4_0"
                    actions.append(
                        "Enable 4-bit KV Cache quantization (cache-type-k/v: f16 -> q4_0) to save 75% of cache memory."
                    )
            # Safe strategy or fallback: use 4-bit cache (q4_0)
            elif curr_cache_k in ("f16", "f32", "q8_0", "q5_0") or curr_cache_v in (
                "f16",
                "f32",
                "q8_0",
                "q5_0",
            ):
                recommendations["cache-type-k"] = "q4_0"
                recommendations["cache-type-v"] = "q4_0"
                actions.append(
                    "Enable 4-bit KV Cache quantization (cache-type-k/v: -> q4_0) to save 75% of cache memory."
                )

            # Reduce context window if already quantized
            elif curr_ctx > 8192:
                suggested_ctx = 8192
                recommendations["ctx-size"] = str(suggested_ctx)
                actions.append(
                    f"Reduce context window (ctx-size: {curr_ctx} -> {suggested_ctx}) to decrease memory usage at high slot loads."
                )
            elif curr_ctx > 4096:
                suggested_ctx = 4096
                recommendations["ctx-size"] = str(suggested_ctx)
                actions.append(
                    f"Reduce context window (ctx-size: {curr_ctx} -> {suggested_ctx}) to decrease memory usage at high slot loads."
                )

            # Reduce batch size and micro-batch size
            if curr_batch > 512:
                recommendations["batch-size"] = "512"
                recommendations["ubatch-size"] = "512"
                actions.append(
                    f"Decrease batch size parameters (batch-size: {curr_batch} -> 512) to lower peak compute allocations."
                )

        # 3. VRAM OOM Warning, but System RAM has headroom
        # Suggest shifting layers back to CPU
        elif (max_vram > vram_critical) and (final_ram < ram_warning):
            if curr_ngl > 0:
                suggested_ngl = max(0, curr_ngl - 10)
                recommendations["n-gpu-layers"] = str(suggested_ngl)
                actions.append(
                    f"Reduce GPU offloaded layers (n-gpu-layers: {curr_ngl} -> {suggested_ngl}) to prevent CUDA OOM."
                )

        # 4. VRAM Underutilization: GPU Layer Offload Opportunity
        # If VRAM is underutilized and the model is running with less than full GPU offload,
        # recommend increasing n-gpu-layers to shift weights onto GPU for faster inference.
        # This is the highest-impact optimization when VRAM < 55% and layers < 99.
        if max_vram < 55.0 and vram_headroom_mb > 1500 and 0 <= curr_ngl < 99:
            # Estimate how many more layers we can fit with 80% of the headroom
            # Use a simple heuristic: each extra layer ≈ headroom / (99 - curr_ngl) MiB
            extra_layers_estimate = int((vram_headroom_mb * 0.80) / max(1, vram_headroom_mb / max(1, 99 - curr_ngl)))
            suggested_ngl = min(curr_ngl + max(10, extra_layers_estimate), 99)
            recommendations["n-gpu-layers"] = str(suggested_ngl)
            actions.append(
                f"Increase GPU layer offload (n-gpu-layers: {curr_ngl} → {suggested_ngl}): "
                f"VRAM is only {round(max_vram, 1)}% utilized with {vram_headroom_mb}MB free. "
                f"Offloading more layers to GPU should significantly improve inference speed (TPS)."
            )

        # 5. VRAM Underutilization / Quality Optimization (Plenty of VRAM headroom)
        # If VRAM usage is low and KV cache is quantized, suggest upgrading progressively to f16/q8_0 to improve quality
        if max_vram < 75.0 and vram_headroom_mb > 2000:
            # Map progressive upgrade steps: q4_0 -> q5_0 -> q8_0 -> f16
            upgrade_map = {
                "q4_0": "q5_0",
                "q4_1": "q5_0",
                "q5_0": "q8_0",
                "q5_1": "q8_0",
                "q8_0": "f16",
            }
            target_cache_k = upgrade_map.get(curr_cache_k, "f16")
            target_cache_v = upgrade_map.get(curr_cache_v, "f16")

            if not performance_first:
                target_cache_k = "f16"
                target_cache_v = "f16"

            # Resolve target against blacklist
            while target_cache_k != curr_cache_k or target_cache_v != curr_cache_v:
                if not is_blacklisted(target_cache_k, target_cache_v):
                    break
                if target_cache_k == "f16":
                    target_cache_k, target_cache_v = "q8_0", "q8_0"
                elif target_cache_k == "q8_0":
                    target_cache_k, target_cache_v = "q5_0", "q5_0"
                elif target_cache_k == "q5_0":
                    target_cache_k, target_cache_v = "q4_0", "q4_0"
                else:
                    target_cache_k, target_cache_v = curr_cache_k, curr_cache_v

            if target_cache_k != curr_cache_k or target_cache_v != curr_cache_v:
                recommendations["cache-type-k"] = target_cache_k
                recommendations["cache-type-v"] = target_cache_v
                actions.append(
                    f"Upgrade KV Cache quantization ({curr_cache_k} → {target_cache_k}) to improve text generation coherence and quality, utilizing the available {vram_headroom_mb}MB VRAM headroom."
                )

        # 6. Context Window Expansion Opportunity
        # If VRAM headroom is large and current ctx-size is modest, suggest expanding for longer conversations.
        if max_vram < 60.0 and vram_headroom_mb > 3000 and curr_ctx <= 16384:
            # Suggest doubling context, capped at 32768
            suggested_ctx = min(curr_ctx * 2, 32768)
            if suggested_ctx > curr_ctx:
                # Don't add if already recommended something that conflicts
                if "ctx-size" not in recommendations:
                    recommendations["ctx-size"] = str(suggested_ctx)
                    actions.append(
                        f"Expand context window (ctx-size: {curr_ctx} → {suggested_ctx}): "
                        f"VRAM headroom ({vram_headroom_mb}MB free) is sufficient to support a larger context, "
                        f"enabling longer document processing and multi-turn conversations."
                    )

    # Try loading baseline benchmarks for comparison
    benchmark = load_latest_benchmark(model_alias)
    baseline_stats = {}
    if benchmark:
        baseline_stats = {
            "baseline_ttft_ms": benchmark.get("avg_ttft_ms"),
            "baseline_tps": benchmark.get("avg_tokens_per_sec"),
        }

    # Format explanation
    if actions:
        explanation = " ".join(actions)
    else:
        explanation = "No tuning adjustments required. System resources are operating within safe utilization thresholds."

    return {
        "status": status,
        "tuning_strategy": "performance_first" if performance_first else "safe_first",
        "model_alias": model_alias,
        "metrics_summary": {
            "system_ram": {
                "max_pct": max_ram,
                "mean_pct": round(mean_ram, 1),
                "final_pct": final_ram,
                "creep_slope": round(ram_slope, 3),
            },
            "vram": {
                "total_mb": vram_total,
                "max_used_mb": max_vram_used,
                "headroom_mb": vram_headroom_mb,
                "max_pct": max_vram,
                "final_pct": final_vram,
            },
            "cpu_util_pct": {"max": max_cpu, "mean": round(mean_cpu, 1)},
            "gpu_util_pct": {"max": max_gpu, "mean": round(mean_gpu, 1)},
            "context_slots": {"max_tokens_cached": max_tokens},
        },
        "baseline_comparison": baseline_stats,
        "detected_issues": issues if issues else ["No resource utilization issues detected."],
        "recommendations": recommendations,
        "explanation": explanation,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze model telemetry logs and generate tuning recommendations."
    )
    parser.add_argument("model", help="The model alias to analyze (e.g. qwen3.5-9b)")
    parser.add_argument(
        "--strategy",
        choices=["performance", "safe"],
        default="performance",
        help="Tuning strategy. 'performance' maximizes speed (relaxing OOM thresholds), 'safe' avoids OOMs.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the recommendations directly to models.ini / profile overlays",
    )
    args = parser.parse_args()

    model_alias = args.model
    perf_first = args.strategy == "performance"
    analysis = analyze_telemetry(model_alias, performance_first=perf_first)

    print(json.dumps(analysis, indent=2))

    if args.apply and analysis["recommendations"]:
        print(f"\nApplying tuning recommendations for {model_alias}...")

        # We can write settings to the model section in models.ini or profile.json
        # Here we write to .profile.json which overrides the defaults or updates them.
        profile_path = ROUTER_MODELS_DIR / f"{model_alias}.profile.json"

        # If we are in local development testing, write to current dir
        if not ROUTER_MODELS_DIR.exists():
            profile_path = Path(f"{model_alias}.profile.json")

        try:
            profile_data = {}
            if profile_path.exists():
                with open(profile_path, "r") as f:
                    profile_data = json.load(f)

            # Apply recommendations
            for k, v in analysis["recommendations"].items():
                profile_data[k] = v
                print(f"  Setting {k} = {v}")

            with open(profile_path, "w") as f:
                json.dump(profile_data, f, indent=2)

            print(f"Tuning properties saved successfully to {profile_path}.")

            # If update_models_ini is available, call it to regenerate models.ini
            try:
                # Add workspace path to sys.path
                sys.path.append(os.getcwd())
                from alpaca_puller import update_models_ini

                update_models_ini()
                print("Regenerated models.ini successfully.")
            except Exception as e:
                print(f"Note: Could not run update_models_ini parser directly: {e}")

        except Exception as e:
            print(f"Error applying recommendations: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
