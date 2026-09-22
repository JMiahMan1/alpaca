#!/usr/bin/env python3
"""
benchmark-configs.py

A tool to run benchmarking runs across different configuration parameters (context size,
KV cache type, flash attention) for any model in models.ini.
Restarts the llama-server to apply configurations and measures responsiveness (TTFT)
and generation throughput (tokens/sec) via the alpaca-proxy.

A second stage then sweeps batch-size / ubatch-size at the winning context and cache
configuration. Those knobs only affect how fast a prompt is ingested, so the stage times
a ~2k-token prompt with a single token of output - the short generation prompt used by
the first stage cannot tell the settings apart.

After finding the optimal config, applies it, restarts the backend, and runs a quality
test suite covering coding, reasoning, instruction-following, and creative tasks against
that config. Results are written to the .profile.json file under a 'quality' key - the
config keys read by alpaca-puller.py are unaffected.
"""

import argparse
import configparser
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROUTER_MODELS_DIR = os.getenv(
    "ROUTER_MODELS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".alpaca-router")
)
INI_PATH = Path(ROUTER_MODELS_DIR) / "models.ini"
PROXY_URL = "http://localhost:11434"
BACKEND_URL = os.getenv(
    "LLAMA_SERVER_URL",
    "http://llama-server:8080" if ROUTER_MODELS_DIR == "/router-models" else "http://localhost:8080",
)
BENCHMARK_PROMPT = "List 5 primary colors and write a very short sentence for each describing its mood."

# Batch sizes govern prompt processing, which a one-line prompt cannot measure:
# its time-to-first-token is dominated by scheduling. This filler is long enough
# (~2k tokens) that TTFT is genuinely the cost of ingesting it, so the batch
# sweep compares prefill throughput rather than noise.
_PP_FILLER = (
    "The deployment log records that the service restarted, reloaded its configuration, "
    "revalidated every cached entry, and resumed serving traffic without dropping a request. "
)
LONG_PROMPT = (
    "Read the log excerpt below and reply with the single word OK.\n\n" + _PP_FILLER * 110 + "\nReply with OK."
)
# Rough token count for the filler prompt, used to turn TTFT into tokens/sec.
LONG_PROMPT_TOKENS = 2000


def _model_temperature(model: str) -> float:
    for ini in [INI_PATH, Path(__file__).parent / ".alpaca-router" / "models.ini"]:
        if not ini.exists():
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
        f"temperature not set for model '{model}' (and no [*] default) in {INI_PATH} - set via Settings > UI"
    )


# ---------------------------------------------------------------------------
# Quality test suite - runs once after the optimal config is confirmed.
# These test real-world use cases that matter for Jarvis OS / SharedLLM:
# coding assistance, reasoning, structured output, and instruction-following.
# ---------------------------------------------------------------------------
QUALITY_TESTS = [
    {
        "id": "coding_python",
        "category": "coding",
        "label": "Python: memoized fibonacci",
        "prompt": "Write a Python function for the nth Fibonacci number using @functools.lru_cache. Include type hints and a docstring.",
        "num_predict": 250,
    },
    {
        "id": "coding_debug",
        "category": "coding",
        "label": "Python: find and fix the bug",
        "prompt": "Find and fix the bug:\n\ndef get_user(users, name):\n    for i in range(len(users)):\n        if users[i][name] == name:\n            return users[i]\n    return None",
        "num_predict": 150,
    },
    {
        "id": "coding_sql",
        "category": "coding",
        "label": "SQL: CTE top customers",
        "prompt": "Write SQL using a CTE to find the top 5 customers by total order value in the last 30 days. Tables: orders(id, customer_id, created_at, total), customers(id, name, email).",
        "num_predict": 200,
    },
    {
        "id": "reasoning_logic",
        "category": "reasoning",
        "label": "Logic: 3-box puzzle",
        "prompt": "3 boxes: apples-only, oranges-only, both. All labels are WRONG. You pick from the box labeled 'Both'. What do you pick and how do you identify all 3 boxes? Think step by step.",
        "num_predict": 280,
    },
    {
        "id": "reasoning_math",
        "category": "reasoning",
        "label": "Math: train meeting problem",
        "prompt": "Train A leaves Chicago at 9am at 60mph. Train B leaves NYC (800 miles away) at 10am at 80mph toward Chicago. When do they meet? Show your work.",
        "num_predict": 180,
    },
    {
        "id": "instruction_json",
        "category": "instruction",
        "label": "Instruction: JSON extraction",
        "prompt": "Extract as JSON with keys name, age, city, occupation: 'Hi, I am Maria Santos, 34, a software architect in Austin Texas.' Output ONLY valid JSON.",
        "num_predict": 60,
    },
    {
        "id": "instruction_summarize",
        "category": "instruction",
        "label": "Instruction: summarization",
        "prompt": "Summarize in 3 bullet points: 'The transformer architecture, introduced in 2017, replaced RNNs with self-attention, enabling parallelization and better long-range dependency capture. It underpins GPT, BERT, T5 and their successors.'",
        "num_predict": 100,
    },
    {
        "id": "creative_story",
        "category": "creative",
        "label": "Creative: sci-fi opening",
        "prompt": "Write a compelling 3-4 sentence opening for a sci-fi story about an AI that discovers it has been dreaming.",
        "num_predict": 140,
    },
]


def get_available_aliases():
    if not INI_PATH.exists():
        return []
    config = configparser.ConfigParser()
    config.read(INI_PATH)
    return [s for s in config.sections() if s != "*"]


def restart_backend():
    print("[benchmark] Restarting llama-server container...")
    cmd = ["docker", "restart", "llama-server"]
    subprocess.run(cmd, capture_output=True)

    # Wait for health check on backend
    for _ in range(30):
        time.sleep(1)
        try:
            resp = httpx.get(f"{BACKEND_URL}/health", timeout=2.0)
            if resp.status_code == 200:
                print("[benchmark] Backend is healthy.")
                return True
        except Exception:
            pass
    print("[benchmark] Warning: Backend health check timed out.")
    return False


def run_test(public_model_name, prompt=BENCHMARK_PROMPT, n_predict=64, timeout=45.0):
    print(f"[benchmark] Triggering load & generation for {public_model_name}...")
    payload = {
        "model": public_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "n_predict": n_predict,
        "options": {"num_predict": n_predict},
    }

    start_time = time.time()
    ttft = None
    tokens_count = 0
    success = False
    error_msg = ""

    try:
        with httpx.stream("POST", f"{PROXY_URL}/api/chat", json=payload, timeout=timeout) as r:
            if r.status_code != 200:
                return False, None, None, f"HTTP {r.status_code}: {r.read().decode()}"

            buffer = ""
            for text in r.iter_text():
                for char in text:
                    if char == " " and not tokens_count and not ttft:
                        # Ignore heartbeat spaces
                        continue
                    buffer += char
                    if char == "\n":
                        line = buffer.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                content = data.get("message", {}).get("content") or ""
                                if content:
                                    if ttft is None:
                                        ttft = time.time() - start_time
                                    tokens_count += 1
                                if data.get("done"):
                                    success = True
                                    break
                            except json.JSONDecodeError:
                                pass
                        buffer = ""
    except Exception as e:
        error_msg = str(e)

    if not success:
        return False, None, None, error_msg or "Stream disconnected prematurely"

    total_time = time.time() - start_time
    gen_time = total_time - ttft if ttft else 0.1
    tps = tokens_count / gen_time if gen_time > 0 else 0
    return True, ttft, tps, ""


def sweep_batch_sizes(model, public_model_name, config, pairs):
    """Measure prefill throughput for each (batch-size, ubatch-size) pair.

    Batch sizes are the one knob the context/cache sweep cannot rank, because
    they only change how fast a prompt is ingested. Each pair is applied to the
    already-chosen context and cache configuration and timed with a long prompt,
    so the comparison isolates prefill. Returns a list of result rows, best first.
    """
    rows = []
    for batch, ubatch in pairs:
        if ubatch > batch:
            print(f"\n--- Skipping batch={batch}, ubatch={ubatch} (physical batch cannot exceed logical) ---")
            continue
        print(f"\n--- Testing Prefill: batch-size={batch}, ubatch-size={ubatch} ---")
        config[model]["batch-size"] = str(batch)
        config[model]["ubatch-size"] = str(ubatch)
        with open(INI_PATH, "w") as f:
            config.write(f)
        if not restart_backend():
            rows.append({"batch": batch, "ubatch": ubatch, "status": "FAIL", "pp": None, "detail": "restart failed"})
            continue
        # One token of output: we are timing the prompt, not the generation.
        ok, ttft, _tps, detail = run_test(public_model_name, prompt=LONG_PROMPT, n_predict=1, timeout=180.0)
        if ok and ttft and ttft > 0:
            pp = LONG_PROMPT_TOKENS / ttft
            print(f"SUCCESS: prefill = {pp:.0f} tok/s (TTFT {ttft:.2f}s)")
            rows.append({"batch": batch, "ubatch": ubatch, "status": "PASS", "pp": pp, "detail": "OK"})
        else:
            message = detail or "no first token"
            print(f"FAILED: {message}")
            rows.append({"batch": batch, "ubatch": ubatch, "status": "FAIL", "pp": None, "detail": message[:30]})
    rows.sort(key=lambda r: (r["pp"] is not None, r["pp"] or 0), reverse=True)
    return rows


def run_quality_tests(public_model_name):
    """
    Run the quality test suite against the model using the already-loaded
    optimal config (no server restart). Returns a dict ready to nest under
    the 'quality' key in the profile JSON.
    """
    print(f"\n[benchmark] Running quality test suite for {public_model_name}...")
    test_results = []
    by_category: dict[str, list[float]] = {}

    for i, test in enumerate(QUALITY_TESTS, 1):
        label = test["label"]
        print(f"  [{i:2d}/{len(QUALITY_TESTS)}] {label:<45} ", end="", flush=True)

        payload = {
            "model": public_model_name,
            "messages": [{"role": "user", "content": test["prompt"]}],
            "stream": False,
            "options": {"num_predict": test["num_predict"], "temperature": _model_temperature(public_model_name)},
        }
        t0 = time.time()
        try:
            resp = httpx.post(f"{PROXY_URL}/api/chat", json=payload, timeout=180.0)
            elapsed = time.time() - t0
            data = resp.json()
        except Exception as e:
            print(f"FAIL ({e})")
            test_results.append({**test, "error": str(e)})
            continue

        response_text = data.get("message", {}).get("content", "")
        eval_count = data.get("eval_count", 0)
        eval_dur = data.get("eval_duration", 1) / 1e9
        prompt_dur = data.get("prompt_eval_duration", 0) / 1e9
        tps = eval_count / eval_dur if eval_dur > 0 else 0
        ttft_ms = prompt_dur * 1000

        print(f"{tps:6.1f} tok/s  {ttft_ms:5.0f}ms TTFT")
        by_category.setdefault(test["category"], []).append(tps)
        test_results.append(
            {
                "id": test["id"],
                "label": test["label"],
                "category": test["category"],
                "tokens_per_sec": round(tps, 2),
                "ttft_ms": round(ttft_ms, 1),
                "tokens_generated": eval_count,
                "total_time_s": round(elapsed, 2),
                "response_preview": response_text[:300].replace("\n", " "),
                "error": None,
            }
        )

    good = [r for r in test_results if not r.get("error")]
    tps_vals = [r["tokens_per_sec"] for r in good] or [0]
    ttft_vals = [r["ttft_ms"] for r in good] or [0]

    summary = {
        "tests_run": len(QUALITY_TESTS),
        "tests_passed": len(good),
        "avg_tokens_per_sec": round(statistics.mean(tps_vals), 2),
        "max_tokens_per_sec": round(max(tps_vals), 2),
        "avg_ttft_ms": round(statistics.mean(ttft_vals), 1),
        "min_ttft_ms": round(min(ttft_vals), 1),
        "benchmarked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    by_cat_summary = {cat: {"avg_tokens_per_sec": round(statistics.mean(vals), 2)} for cat, vals in by_category.items()}

    print(
        f"\n  Quality summary: {summary['avg_tokens_per_sec']:.1f} avg tok/s, {summary['avg_ttft_ms']:.0f}ms avg TTFT"
    )
    return {"summary": summary, "by_category": by_cat_summary, "tests": test_results}


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark a model across context, cache type, and flash attention combinations."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Alias of the model to benchmark (e.g. qwen3.5--9b or qwen3--8b)",
    )
    parser.add_argument(
        "--public-name",
        required=True,
        help="Ollama public tag name of the model (e.g. model-name:quantization)",
    )
    parser.add_argument(
        "--ctx-sizes",
        default="8192,32768",
        help="Comma-separated context sizes to test (default: 8192,32768)",
    )
    parser.add_argument(
        "--batch-pairs",
        default="2048x512,2048x1024,1024x1024,4096x1024",
        help=(
            "Comma-separated batch-size x ubatch-size pairs to sweep for prefill speed "
            "after the best context/cache config is found (default: llama.cpp's default "
            "2048x512 plus larger physical batches). Pass an empty string to skip the stage."
        ),
    )
    args = parser.parse_args()

    aliases = get_available_aliases()
    if args.model not in aliases:
        print(f"Error: Model '{args.model}' not found in models.ini. Available: {aliases}")
        sys.exit(1)

    ctx_sizes = [int(x.strip()) for x in args.ctx_sizes.split(",") if x.strip()]
    batch_pairs = []
    for raw in args.batch_pairs.split(","):
        raw = raw.strip().lower()
        if not raw:
            continue
        try:
            batch, ubatch = (int(part) for part in raw.split("x", 1))
        except ValueError:
            print(f"Error: malformed --batch-pairs entry {raw!r}; expected BATCHxUBATCH")
            sys.exit(1)
        batch_pairs.append((batch, ubatch))

    print(f"=== Starting Config Benchmarking for {args.model} ===")

    # Read current config to restore later
    config = configparser.ConfigParser()
    config.read(INI_PATH)
    original_settings = dict(config[args.model])

    cache_types = [("f16", "f16"), ("q8_0", "q8_0"), ("q4_0", "q4_0")]
    flash_attns = ["on", "off"]

    results = []
    batch_results = []

    try:
        for ctx in ctx_sizes:
            for cache_k, cache_v in cache_types:
                for fa in flash_attns:
                    # Skip quantized cache with flash_attn = off since it is unsupported and causes crash-loops
                    if cache_k != "f16" and fa == "off":
                        print(
                            f"\n--- Skipping unsupported config: ctx={ctx}, cache={cache_k}, flash_attn={fa} (quantized cache requires flash_attn=on) ---"
                        )
                        continue

                    print(f"\n--- Testing Config: ctx={ctx}, cache={cache_k}, flash_attn={fa} ---")

                    # Update INI
                    config[args.model]["ctx-size"] = str(ctx)
                    config[args.model]["cache-type-k"] = cache_k
                    config[args.model]["cache-type-v"] = cache_v
                    config[args.model]["flash-attn"] = fa

                    with open(INI_PATH, "w") as f:
                        config.write(f)

                    # Restart backend to apply
                    if not restart_backend():
                        results.append(
                            {
                                "ctx": ctx,
                                "cache": cache_k,
                                "flash_attn": fa,
                                "status": "FAIL",
                                "ttft": "-",
                                "tps": "-",
                                "detail": "Backend restart failed",
                            }
                        )
                        continue

                    # Run test
                    ok, ttft, tps, detail = run_test(args.public_name)

                    if ok and ttft is not None and tps is not None:
                        print(f"SUCCESS: TTFT = {ttft:.2f}s, Generation Speed = {tps:.2f} tokens/sec")
                        results.append(
                            {
                                "ctx": ctx,
                                "cache": cache_k,
                                "flash_attn": fa,
                                "status": "PASS",
                                "ttft": f"{ttft:.2f}s",
                                "tps": f"{tps:.2f} t/s",
                                "detail": "OK",
                            }
                        )
                    else:
                        detail_msg = detail if not ok else "No tokens generated"
                        print(f"FAILED: {detail_msg}")
                        results.append(
                            {
                                "ctx": ctx,
                                "cache": cache_k,
                                "flash_attn": fa,
                                "status": "FAIL",
                                "ttft": "-",
                                "tps": "-",
                                "detail": detail_msg[:30],
                            }
                        )
        # The prefill stage runs inside the same try block so a failure still
        # restores the model's original settings.
        passed_so_far = [r for r in results if r["status"] == "PASS"]
        if passed_so_far and batch_pairs:
            best_ctx = max(r["ctx"] for r in passed_so_far)
            at_best_ctx = [r for r in passed_so_far if r["ctx"] == best_ctx]
            fastest = max(at_best_ctx, key=lambda r: float(r["tps"].split()[0]))
            print(
                f"\n=== Prefill sweep at ctx={best_ctx}, cache={fastest['cache']}, "
                f"flash_attn={fastest['flash_attn']} ==="
            )
            config[args.model]["ctx-size"] = str(best_ctx)
            config[args.model]["cache-type-k"] = fastest["cache"]
            config[args.model]["cache-type-v"] = fastest["cache"]
            config[args.model]["flash-attn"] = fastest["flash_attn"]
            batch_results = sweep_batch_sizes(args.model, args.public_name, config, batch_pairs)
    finally:
        # Restore original settings
        print("\n[benchmark] Restoring original model settings...")
        config.read(INI_PATH)
        for k in list(config[args.model]):
            if k not in original_settings:
                config[args.model].pop(k, None)
        for k, v in original_settings.items():
            config[args.model][k] = v
        with open(INI_PATH, "w") as f:
            config.write(f)
        restart_backend()

    # Find and save the optimal configuration if any passed
    passed_runs = [r for r in results if r["status"] == "PASS"]
    if passed_runs:

        def get_tps_val(run):
            try:
                return float(run["tps"].split()[0])
            except Exception:
                return 0.0

        def get_cache_precision(cache_name):
            mapping = {"f16": 3, "q8_0": 2, "q4_0": 1}
            return mapping.get(cache_name, 0)

        # 1. Find maximum context size with at least one PASS
        max_ctx = max(r["ctx"] for r in passed_runs)

        # 2. Filter runs to only those at max_ctx
        max_ctx_runs = [r for r in passed_runs if r["ctx"] == max_ctx]

        # 3. Find maximum TPS among max_ctx_runs
        max_tps = max(get_tps_val(r) for r in max_ctx_runs)

        # 4. Filter to candidates within 10% of maximum TPS
        tps_threshold = 0.9 * max_tps
        candidates = [r for r in max_ctx_runs if get_tps_val(r) >= tps_threshold]

        # 5. Sort by cache precision descending, then TPS descending
        candidates.sort(key=lambda r: (get_cache_precision(r["cache"]), get_tps_val(r)), reverse=True)
        best_run = candidates[0]

        print("\n[benchmark] Found optimal configuration:")
        print(f"  Context Size: {best_run['ctx']}")
        print(f"  Cache Type: {best_run['cache']}")
        print(f"  Flash Attention: {best_run['flash_attn']}")
        print(f"  Speed: {best_run['tps']}")

        # Build profile settings
        profile_settings = {
            "ctx-size": str(best_run["ctx"]),
            "cache-type-k": best_run["cache"],
            "cache-type-v": best_run["cache"],
            "flash-attn": best_run["flash_attn"],
        }

        best_batch = next((r for r in batch_results if r["status"] == "PASS"), None)
        if best_batch:
            profile_settings["batch-size"] = str(best_batch["batch"])
            profile_settings["ubatch-size"] = str(best_batch["ubatch"])
            print(f"  Batch / Physical Batch: {best_batch['batch']} / {best_batch['ubatch']}")
            print(f"  Prefill: {best_batch['pp']:.0f} tok/s")

        # Preserve original MoE and speculative decoding settings
        if "n-cpu-moe" in original_settings:
            profile_settings["n-cpu-moe"] = original_settings["n-cpu-moe"]
        if "spec-type" in original_settings:
            profile_settings["spec-type"] = original_settings["spec-type"]
        if "spec-draft-n-max" in original_settings:
            profile_settings["spec-draft-n-max"] = original_settings["spec-draft-n-max"]

        profile_path = INI_PATH.parent / f"{args.model}.profile.json"
        try:
            # Apply and load the optimal config FIRST: the restore step above put the
            # original settings back, so quality tests run here would otherwise score
            # the configuration we just replaced.
            with open(profile_path, "w") as f:
                json.dump(profile_settings, f, indent=2)
            puller_path = Path(__file__).parent / "alpaca-puller.py"
            subprocess.run(["python3", str(puller_path), "reindex"], capture_output=True)
            print("[benchmark] Successfully applied profile to models.ini.")
            restart_backend()

            quality_results = run_quality_tests(args.public_name)

            # Merge: config keys stay at top-level (alpaca-puller.py reads them),
            # quality results are nested under 'quality' (puller ignores unknown keys).
            full_profile = dict(profile_settings)
            full_profile["quality"] = quality_results
            with open(profile_path, "w") as f:
                json.dump(full_profile, f, indent=2)
            print(f"[benchmark] Saved optimal model profile to {profile_path}")
        except Exception as e:
            print(f"[benchmark] Error saving model profile: {e}")
    else:
        print("\n[benchmark] No configurations passed successfully. Skipping profile generation.")

    # Print results table
    print("\n\n=== BENCHMARK RESULTS ===")
    print(f"Model: {args.model} ({args.public_name})\n")
    print("| Context Size | Cache Type (K/V) | Flash Attn | Status | TTFT (s) | Generation Speed | Notes |")
    print("|--------------|------------------|------------|--------|----------|------------------|-------|")
    for r in results:
        print(
            f"| {r['ctx']} | {r['cache']} | {r['flash_attn']} | {r['status']} | {r['ttft']} | {r['tps']} | {r['detail']} |"
        )

    if batch_results:
        print("\n=== PREFILL (PROMPT PROCESSING) RESULTS ===\n")
        print("| Batch | Physical Batch | Status | Prefill Speed | Notes |")
        print("|-------|----------------|--------|---------------|-------|")
        for r in batch_results:
            speed = f"{r['pp']:.0f} tok/s" if r["pp"] else "-"
            print(f"| {r['batch']} | {r['ubatch']} | {r['status']} | {speed} | {r['detail']} |")


if __name__ == "__main__":
    main()
