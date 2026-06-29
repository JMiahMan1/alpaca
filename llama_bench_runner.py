#!/usr/bin/env python3
"""
llama_bench_runner.py

Integrates llama.cpp's built-in 'llama-bench' tool to perform rapid sweeps
of model configuration parameters (-ngl, -ctk, -ctv, -b) and parses their
throughput (TPS) and latency results.

Optionally monitors RAM/VRAM peak usage concurrently using our telemetry APIs.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("llama_bench_runner")

LLAMA_BENCH_BIN = os.getenv("LLAMA_BENCH_BIN", "llama-bench")
ROUTER_MODELS_DIR = Path(os.getenv("ROUTER_MODELS_DIR", ".alpaca-router"))


def locate_model_file(model_alias: str) -> Path:
    """Find the GGUF model file on the host filesystem."""
    # Check direct name
    direct_path = ROUTER_MODELS_DIR / f"{model_alias}.gguf"
    if direct_path.exists():
        return direct_path.resolve()

    # Check lowercase stem matching
    for path in ROUTER_MODELS_DIR.glob("*.gguf"):
        if path.stem.lower() == model_alias.lower() or path.stem.lower().startswith(
            model_alias.lower()
        ):
            return path.resolve()

    # Raise error if not found
    raise FileNotFoundError(
        f"Could not find model file for alias: {model_alias} in {ROUTER_MODELS_DIR}"
    )


def run_llama_bench_sweep(
    model_path: Path, gpu_layers: List[int], kv_cache_types: List[str], batch_sizes: List[int]
) -> List[Dict[str, Any]]:
    """Execute llama-bench with comma-separated sweeps and parse the JSON output."""

    # Map arguments
    ngl_str = ",".join(map(str, gpu_layers))
    ctk_str = ",".join(kv_cache_types)
    ctv_str = ",".join(kv_cache_types)
    batch_str = ",".join(map(str, batch_sizes))

    cmd = [
        LLAMA_BENCH_BIN,
        "-m",
        str(model_path),
        "-ngl",
        ngl_str,
        "-ctk",
        ctk_str,
        "-ctv",
        ctv_str,
        "-b",
        batch_str,
        "-o",
        "json",  # Output as standard JSON
    ]

    logger.info(f"Executing: {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )

        # Parse the JSON array from stdout
        try:
            results = json.loads(proc.stdout)
            return results
        except json.JSONDecodeError:
            # Fallback: llama-bench sometimes emits extra header/footer lines
            # around the JSON array. Find the outermost [...] bracket pair.
            start = proc.stdout.find("[")
            end = proc.stdout.rfind("]") + 1
            if start != -1 and end > start:
                return json.loads(proc.stdout[start:end])
            raise ValueError(f"No JSON array found in llama-bench output:\n{proc.stdout[:500]}")

    except subprocess.CalledProcessError as e:
        logger.error(f"llama-bench execution failed: {e.stderr}")
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred during benchmarking: {e}")
        raise e


def process_bench_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Structure raw llama-bench results into an actionable tuning profile."""
    profile = {"model": "", "configurations": []}

    if not results:
        return profile

    profile["model"] = Path(results[0].get("model", "unknown")).name

    for r in results:
        config = {
            "n_gpu_layers": r.get("n_gpu_layers"),
            "cache_type_k": r.get("cache_type_k"),
            "cache_type_v": r.get("cache_type_v"),
            "batch_size": r.get("batch_size"),
            "ubatch_size": r.get("ubatch_size"),
            # llama-bench reports speed in tokens/sec for each phase independently
            "prompt_processing_tps": round(r.get("tps_prompt", 0.0), 2),
            "generation_tps": round(r.get("tps_gen", 0.0), 2),
        }
        profile["configurations"].append(config)

    return profile


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 llama_bench_runner.py <model_alias> [ngl_list] [cache_list]")
        print("Example: python3 llama_bench_runner.py qwen3--8b 0,10,20,99 f16,q4_0")
        sys.exit(1)

    model_alias = sys.argv[1]

    # Parse sweeps from CLI or use defaults
    gpu_layers = [0, 10, 20, 32, 99]
    if len(sys.argv) > 2:
        gpu_layers = [int(x) for x in sys.argv[2].split(",")]

    kv_cache_types = ["f16", "q4_0"]
    if len(sys.argv) > 3:
        kv_cache_types = sys.argv[3].split(",")

    batch_sizes = [512, 1024]

    try:
        model_path = locate_model_file(model_alias)
        logger.info(f"Found model file at: {model_path}")

        print("\n=== Running llama-bench Sweeps ===")
        raw_results = run_llama_bench_sweep(
            model_path=model_path,
            gpu_layers=gpu_layers,
            kv_cache_types=kv_cache_types,
            batch_sizes=batch_sizes,
        )

        tuning_profile = process_bench_results(raw_results)

        # Save results to benchmark data folder
        out_dir = Path("data/llm_benchmarks")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"llama_bench_{model_alias}_profile.json"

        with open(out_file, "w") as f:
            json.dump(tuning_profile, f, indent=2)

        print(f"\nBenchmark completed successfully! Profile saved to {out_file}")
        print("\nSummary of Configurations tested:")
        print(f"{'ngl':<6} {'KV Cache':<10} {'Batch':<6} {'Prefill TPS':<12} {'Gen TPS':<10}")
        print("-" * 50)
        for cfg in tuning_profile["configurations"]:
            cache_mode = f"{cfg['cache_type_k']}/{cfg['cache_type_v']}"
            print(
                f"{cfg['n_gpu_layers']:<6} {cache_mode:<10} {cfg['batch_size']:<6} {cfg['prompt_processing_tps']:<12.1f} {cfg['generation_tps']:<10.1f}"
            )

    except Exception as e:
        logger.error(f"Tuning run failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
