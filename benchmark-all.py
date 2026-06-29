#!/usr/bin/env python3
"""
benchmark-all.py

Orchestrator to run benchmarking sweeps sequentially for all available models
in the alpaca-proxy router setup. It generates optimal profiles for each.

Context sizes are chosen per model type:
  - MoE models (qwen3.6-35b, MTP):   skip ctx sweep — already tuned at 98304
  - Dense large (>=9B, non-MoE):     8192, 16384, 32768, 65536
  - Small (<9B):                      8192, 16384, 32768
"""

import json
import subprocess
import sys

# Models that are already tuned and should not have their ctx swept
CTX_LOCKED_ALIASES = {
    "qwen3.6-35b-a3b--q4_k_m",
    "Qwen3.6-35B-A3B-MTP-GGUF--Qwen3.6-35B-A3B-UD-Q4_K_M--latest",
}

# Dense large models — can safely test larger ctx sizes
DENSE_LARGE_ALIASES = {
    "gemma-4-12b-fable5--latest",
}

# Default ctx sweep for small / unknown models
CTX_SMALL = "8192,16384,32768"
CTX_LARGE = "8192,16384,32768,65536"
CTX_LOCKED = "32768"  # single neutral run just to generate quality profile


def discover_models():
    print("[benchmark-all] Discovering local models from alpaca-proxy container...")
    py_code = """
import importlib.util, json
spec = importlib.util.spec_from_file_location("alpaca_puller", "/app/alpaca-puller.py")
alpaca_puller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alpaca_puller)
models = []
for model_name, _, _ in alpaca_puller.iter_local_models():
    alias = alpaca_puller.router_filename_for_model_name(model_name)
    if alias.endswith(".gguf"):
        alias = alias[:-5]
    models.append((alias, model_name))
print(json.dumps(models))
"""
    cmd = ["docker", "exec", "alpaca-proxy", "python3", "-c", py_code]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        models = json.loads(res.stdout.strip())
        print(f"[benchmark-all] Discovered {len(models)} model(s): {models}")
        return models
    except Exception as e:
        print(f"Error during model discovery: {e}", file=sys.stderr)
        if "res" in locals():
            print(f"Stdout: {res.stdout}", file=sys.stderr)
            print(f"Stderr: {res.stderr}", file=sys.stderr)
        return []


def main():
    print("=== Starting Dynamic Benchmark Sweep for All Available Models ===")
    models = discover_models()
    if not models:
        print("No models discovered. Exiting.")
        sys.exit(1)

    for alias, public in models:
        print("\n\n==========================================")
        print(f"BENCHMARKING: {alias} ({public})")
        print("==========================================")

        # Pick ctx-sizes based on model type
        if alias in CTX_LOCKED_ALIASES:
            ctxs = CTX_LOCKED
            print(f"  [ctx] MoE/locked model — using neutral ctx {ctxs} (quality test only)")
        elif alias in DENSE_LARGE_ALIASES:
            ctxs = CTX_LARGE
            print(f"  [ctx] Dense large model — sweeping {ctxs}")
        else:
            ctxs = CTX_SMALL
            print(f"  [ctx] Small/unknown model — sweeping {ctxs}")

        cmd = [
            "docker",
            "exec",
            "alpaca-proxy",
            "python3",
            "-u",
            "benchmark-configs.py",
            "--model",
            alias,
            "--public-name",
            public,
            "--ctx-sizes",
            ctxs,
        ]

        # Stream output in real-time
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(output.strip())

        rc = process.poll()
        if rc == 0:
            print(f"Successfully finished benchmark for {alias}.")
        else:
            print(f"Benchmark failed for {alias} with exit code {rc}.")


if __name__ == "__main__":
    main()
