#!/usr/bin/env python3
"""
alpaca-benchmark - Multi-model benchmark suite for alpaca-proxy / llama-server
Usage:
  python3 benchmark_all_models.py                  # benchmark all models (full suite)
  python3 benchmark_all_models.py gemma-4-12b-fable5:latest qwen3.6-35b-a3b:q4_k_m
  python3 benchmark_all_models.py --list           # list available models
  python3 benchmark_all_models.py --out /my/dir    # custom output dir
  python3 benchmark_all_models.py --suite assistant --attempts 3   # assistant latency suite
  python3 benchmark_all_models.py --suite assistant --think        # allow thinking
  python3 benchmark_all_models.py --suite assistant --direct       # via llama-server OAI endpoint,
                                                                   # thinking hard-disabled
                                                                   # (fixes empty answers from
                                                                   # reasoning-tuned models that the
                                                                   # proxy always forces into think mode)

Profiles are saved as:  <output_dir>/<model-name>.profile.json
A combined report is:   <output_dir>/benchmark_report.json

Timing notes:
  Requests stream NDJSON from the server. ttft_wall_ms is measured to the first
  streamed content token (true prefill+first-decode wall clock, what a voice
  assistant experiences). ttft_ms remains the server-side prompt-eval figure.

--direct mode:
  Hot-swaps the model via the proxy (one tiny warmup call), waits for
  llama-server /health, then benchmarks against {host}:8080/v1/chat/completions
  with chat_template_kwargs {"enable_thinking": false}. This is the only path
  that truly disables thinking, because the proxy always forces thinking=True
  upstream and only filters client-side. Over SSE the server-side prompt-eval
  metric is unavailable (ttft_ms is null); tokens_per_sec is a decode-only
  wall-clock estimate (completion tokens / (total - TTFT)).
"""

import argparse
import datetime
import json
import os
import statistics
import sys
import time
import urllib.parse
import urllib.request

BASE = os.getenv("OLLAMA_URL", "http://localhost:11434")
DIRECT_BASE = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080")
OUTPUT_DIR = os.getenv("BENCHMARK_OUTPUT", os.path.join(os.path.dirname(__file__), ".alpaca-router"))

# Assistant-focused suite: the four questions a voice assistant must answer
# instantly and sensibly, with a short personal-assistant system prompt.
# num_predict is capped so rambling can't inflate wall time; TTFT is the metric.
ASSISTANT_TESTS = [
    {
        "id": "asst_personal",
        "category": "assistant",
        "label": "Assistant - Personal knowledge",
        "prompt": "Who is Zachary Summers?",
        "num_predict": 120,
    },
    {
        "id": "asst_iot",
        "category": "assistant",
        "label": "Assistant - IoT state check",
        "prompt": "Is the Piano Lamp on?",
        "num_predict": 120,
    },
    {
        "id": "asst_weather",
        "category": "assistant",
        "label": "Assistant - Weather",
        "prompt": "What will the weather be like today?",
        "num_predict": 120,
    },
    {
        "id": "asst_datetime",
        "category": "assistant",
        "label": "Assistant - Date and time",
        "prompt": "What is the date and time?",
        "num_predict": 80,
    },
]

ASSISTANT_SYSTEM = "You are a concise voice assistant. Answer in one or two short sentences."

TESTS = [
    {
        "id": "ttft_short",
        "category": "performance",
        "label": "TTFT - Short prompt",
        "prompt": "What is 2+2?",
        "num_predict": 20,
    },
    {
        "id": "ttft_medium",
        "category": "performance",
        "label": "TTFT - Medium prompt",
        "prompt": "Explain what a REST API is and give a simple example.",
        "num_predict": 150,
    },
    {
        "id": "coding_python",
        "category": "coding",
        "label": "Coding - Python (memoized fibonacci)",
        "prompt": "Write a Python function for the nth Fibonacci number using @functools.lru_cache with type hints, a docstring, and a usage example.",
        "num_predict": 300,
    },
    {
        "id": "coding_sql",
        "category": "coding",
        "label": "Coding - SQL (CTE query)",
        "prompt": "Write a SQL query using CTEs to find the top 5 customers by total order value in the last 30 days. Tables: orders(id, customer_id, created_at, total), customers(id, name, email).",
        "num_predict": 200,
    },
    {
        "id": "coding_debug",
        "category": "coding",
        "label": "Coding - Debug (find the bug)",
        "prompt": "Find and fix the bug in this Python code:\n\ndef get_user(users, name):\n    for i in range(len(users)):\n        if users[i][name] == name:\n            return users[i]\n    return None",
        "num_predict": 150,
    },
    {
        "id": "reasoning_logic",
        "category": "reasoning",
        "label": "Reasoning - Logic puzzle (3 boxes)",
        "prompt": "There are 3 boxes: one has only apples, one has only oranges, one has both. ALL labels are wrong. You pick one fruit from the box labeled 'Both'. What do you pick and how do you correctly identify all boxes? Think step by step.",
        "num_predict": 300,
    },
    {
        "id": "reasoning_math",
        "category": "reasoning",
        "label": "Reasoning - Math word problem",
        "prompt": "Train A leaves Chicago at 9am at 60mph. Train B leaves NYC (800 miles away) at 10am at 80mph toward Chicago. At what time do they meet? Show your work.",
        "num_predict": 200,
    },
    {
        "id": "instruction_json",
        "category": "instruction",
        "label": "Instruction - JSON extraction",
        "prompt": "Extract as JSON with keys name, age, city, occupation: 'Hi, my name is Maria Santos. I am 34 years old, a software architect based in Austin, Texas.' Output ONLY valid JSON, nothing else.",
        "num_predict": 80,
    },
    {
        "id": "instruction_rewrite",
        "category": "instruction",
        "label": "Instruction - Formal rewrite",
        "prompt": "Rewrite formally for a corporate email: 'Hey, just wanted to check if you got the files I sent? Need them back asap, its kinda urgent lol'",
        "num_predict": 120,
    },
    {
        "id": "instruction_system",
        "category": "instruction",
        "label": "Instruction - System prompt adherence",
        "prompt": "Tell me what the weather is like today.",
        "system": "You are a pirate. Always respond in pirate speak. Never break character.",
        "num_predict": 100,
    },
    {
        "id": "creative_story",
        "category": "creative",
        "label": "Creative - Sci-fi story opening",
        "prompt": "Write a compelling 3-4 sentence opening paragraph for a science fiction story about an AI that discovers it has been dreaming.",
        "num_predict": 150,
    },
    {
        "id": "summarization",
        "category": "instruction",
        "label": "Instruction - Summarization",
        "prompt": "Summarize in 2-3 bullet points: 'The transformer architecture, introduced in the 2017 paper Attention Is All You Need, replaced recurrent neural networks with self-attention mechanisms. This allowed for much greater parallelization during training and enabled models to capture long-range dependencies more effectively. The architecture has since become the foundation for large language models including GPT, BERT, T5, and their successors.'",
        "num_predict": 100,
    },
]


def list_models():
    req = urllib.request.Request(f"{BASE}/api/tags")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        print(f"Error listing models: {e}")
        return []


def call_direct_openai(router_model, prompt, num_predict, system=None, temperature=0.3):
    """Stream /v1/chat/completions SSE from llama-server, thinking hard-off.

    router_model is the full router id (e.g. 'foo--latest'); enable_thinking
    goes through chat_template_kwargs which llama-server applies at template
    render time — the only mechanism the proxy cannot override.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps(
        {
            "model": router_model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": num_predict,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream_options": {"include_usage": True},
        }
    ).encode()

    req = urllib.request.Request(
        f"{DIRECT_BASE}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft_wall = None
    text_parts = []
    usage = {}
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                chunk = json.loads(data_str)
                err = chunk.get("error")
                if err:
                    return None, str(err)
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                if content:
                    if ttft_wall is None:
                        ttft_wall = time.perf_counter() - t0
                    text_parts.append(content)
                if chunk.get("usage"):
                    usage = chunk["usage"]
        elapsed = time.perf_counter() - t0
    except Exception as e:
        return None, str(e)

    completion_tokens = usage.get("completion_tokens", 0)
    decode_s = max(elapsed - (ttft_wall or 0), 1e-6) if completion_tokens else 0
    return {
        "response": "".join(text_parts),
        "tokens_generated": completion_tokens,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "total_time_s": elapsed,
        "ttft_ms": None,  # server-side prompt-eval not exposed over SSE
        "ttft_wall_ms": (ttft_wall or elapsed) * 1000,
        # decode-only wall-clock estimate: tokens seen after first token
        "tokens_per_sec": (completion_tokens - 1) / decode_s if completion_tokens > 1 and decode_s > 0 else 0,
        "eval_duration_s": decode_s,
    }, None


def resolve_router_model(model):
    """Map a proxy model name to the full router id llama-server expects."""
    # Already a router id (contains the '--' separator)?
    if "--" in model:
        return model
    family, _, quant = model.partition(":")
    req = urllib.request.Request(f"{DIRECT_BASE}/v1/models")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        ids = [m.get("id", "") for m in data.get("data", [])]
    except Exception:
        return f"{family}--{quant}"
    for rid in ids:
        if rid.startswith(f"{family}--"):
            return rid
    return f"{family}--{quant}"


def wait_healthy(timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{DIRECT_BASE}/health", timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def swap_model_via_proxy(model):
    """Load the requested model on llama-server by routing one tiny call through the proxy."""
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
            "think": False,
            "options": {"num_predict": 5},
        }
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            resp.read()
        return True
    except Exception as e:
        print(f"swap failed: {e}")
        return False


def call_model(model, prompt, num_predict, system=None, think=False, temperature=0.3):
    """Stream /api/chat NDJSON. Returns metrics incl. wall-clock first-token latency."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": think,
            "options": {
                "num_predict": num_predict,
                "temperature": temperature,
            },
        }
    ).encode()

    req = urllib.request.Request(
        f"{BASE}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft_wall = None
    text_parts = []
    final = {}
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    return None, chunk["error"]
                content = chunk.get("message", {}).get("content", "")
                if content:
                    if ttft_wall is None:
                        ttft_wall = time.perf_counter() - t0
                    text_parts.append(content)
                if chunk.get("done"):
                    final = chunk
        elapsed = time.perf_counter() - t0
    except Exception as e:
        return None, str(e)

    eval_count = final.get("eval_count", 0)
    eval_dur = final.get("eval_duration", 1) / 1e9
    prompt_dur = final.get("prompt_eval_duration", 0) / 1e9
    tps = eval_count / eval_dur if eval_dur > 0 else 0

    return {
        "response": "".join(text_parts),
        "tokens_generated": eval_count,
        "prompt_tokens": final.get("prompt_eval_count", 0),
        "total_time_s": elapsed,
        "ttft_ms": prompt_dur * 1000,
        "ttft_wall_ms": (ttft_wall or elapsed) * 1000,
        "tokens_per_sec": tps,
        "eval_duration_s": eval_dur,
    }, None


def benchmark_model(model, suite="full", attempts=1, think=False, temperature=0.3, direct=False):
    tests = ASSISTANT_TESTS if suite == "assistant" else TESTS
    system = ASSISTANT_SYSTEM if suite == "assistant" else None

    def call_once(prompt, num_predict):
        if direct:
            return call_direct_openai(resolve_router_model(model), prompt, num_predict, system, temperature)
        return call_model(model, prompt, num_predict, system, think=think, temperature=temperature)

    print(f"\n{'-' * 62}")
    print(f"  MODEL: {model}")
    mode = "direct llama-server (thinking off)" if direct else f"proxy (think={think})"
    print(f"  Mode:  {mode}  Suite: {suite}  Attempts: {attempts}  Temp: {temperature}")
    print(f"  Time:  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'-' * 62}")

    if direct:
        print("  Hot-swapping via proxy... ", end="", flush=True)
        if not swap_model_via_proxy(model):
            print("FAILED - cannot swap model")
            return None
        print("OK")
        print("  Waiting for llama-server health... ", end="", flush=True)
        if not wait_healthy():
            print("TIMEOUT")
            return None
        print("OK")

    # Warmup ping (also forces the model load so attempt 1 is a warm run)
    print("  Warming up... ", end="", flush=True)
    _, err = call_once("Hi", 5)
    if err:
        print(f"FAILED - cannot reach model: {err}")
        return None
    print("OK")

    results = []
    for i, test in enumerate(tests, 1):
        label = test["label"]
        runs = []
        for attempt in range(attempts):
            suffix = f" (try {attempt + 1}/{attempts})" if attempts > 1 else ""
            print(f"  [{i:2d}/{len(tests)}] {label:<45}{suffix:<12} ", end="", flush=True)
            m, err = call_once(test["prompt"], test["num_predict"])
            if err:
                print(f"FAIL ({err[:60]})")
                continue
            print(
                f"{m['tokens_per_sec']:6.1f} tok/s  TTFT wall {m['ttft_wall_ms']:6.0f}ms  total {m['total_time_s']:5.2f}s"
            )
            runs.append(m)
        if not runs:
            results.append({**test, "error": "all attempts failed"})
            continue

        def agg(key, runs=runs):
            vals = [r[key] for r in runs if r.get(key) is not None]
            return (
                {"avg": round(statistics.mean(vals), 2), "min": round(min(vals), 2), "max": round(max(vals), 2)}
                if vals
                else None
            )

        best = min(runs, key=lambda r: r["ttft_wall_ms"])
        results.append(
            {
                **test,
                "attempts": len(runs),
                "ttft_wall_ms": agg("ttft_wall_ms"),
                "ttft_ms": agg("ttft_ms"),
                "tokens_per_sec": agg("tokens_per_sec"),
                "total_time_s": agg("total_time_s"),
                "tokens_generated": agg("tokens_generated"),
                "response": best["response"],
                "error": None,
            }
        )

    return results


def _val(x):
    """Aggregated metrics are {avg,min,max} dicts; fall back to raw numbers."""
    if isinstance(x, dict):
        return x.get("avg", 0)
    return x or 0


def build_profile(model, results, suite="full"):
    good = [r for r in results if not r.get("error")]
    if not good:
        return {"model": model, "error": "all tests failed"}

    tps = [_val(r["tokens_per_sec"]) for r in good]
    ttft = [_val(r["ttft_wall_ms"]) for r in good]
    by_cat = {}
    for r in good:
        by_cat.setdefault(r["category"], []).append(_val(r["tokens_per_sec"]))

    profile = {
        "model": model,
        "suite": suite,
        "benchmarked_at": datetime.datetime.now().isoformat(),
        "host": BASE,
        "summary": {
            "tests_run": len(results),
            "tests_passed": len(good),
            "tests_failed": len(results) - len(good),
            "avg_tokens_per_sec": round(statistics.mean(tps), 2),
            "max_tokens_per_sec": round(max(tps), 2),
            "min_tokens_per_sec": round(min(tps), 2),
            "avg_ttft_wall_ms": round(statistics.mean(ttft), 1),
            "min_ttft_wall_ms": round(min(ttft), 1),
            "max_ttft_wall_ms": round(max(ttft), 1),
        },
        "by_category": {cat: {"avg_tokens_per_sec": round(statistics.mean(vals), 2)} for cat, vals in by_cat.items()},
        "tests": [
            {
                "id": r["id"],
                "label": r["label"],
                "category": r["category"],
                "attempts": r.get("attempts", 1),
                "ttft_wall_ms": r.get("ttft_wall_ms"),
                "ttft_ms": r.get("ttft_ms"),
                "tokens_per_sec": r.get("tokens_per_sec"),
                "tokens_generated": r.get("tokens_generated"),
                "total_time_s": r.get("total_time_s"),
                "response": (r.get("response") or r.get("response_preview", ""))[:600],
                "error": r.get("error"),
            }
            for r in results
        ],
    }
    return profile


def save_profile(profile, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    safe_name = profile["model"].replace(":", "--").replace("/", "_")
    path = os.path.join(output_dir, f"{safe_name}.profile.json")
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)
    return path


def print_comparison(all_profiles):
    valid = [p for p in all_profiles if "summary" in p]
    if len(valid) < 2:
        return
    print(f"\n{'=' * 78}")
    print("  COMPARISON TABLE")
    print(f"{'=' * 78}")
    print(f"  {'Model':<35} {'Avg tok/s':>10} {'Peak':>8} {'TTFT wall ms':>13}")
    print(f"  {'-' * 35} {'-' * 10} {'-' * 8} {'-' * 13}")
    ranked = sorted(valid, key=lambda p: p["summary"]["avg_ttft_wall_ms"])
    for p in ranked:
        s = p["summary"]
        print(
            f"  {p['model']:<35} {s['avg_tokens_per_sec']:>10.1f} {s['max_tokens_per_sec']:>8.1f} {s['avg_ttft_wall_ms']:>13.0f}"
        )
    print(f"{'=' * 78}\n")


def main():
    global BASE
    parser = argparse.ArgumentParser(description="Multi-model alpaca benchmark")
    parser.add_argument("models", nargs="*", help="Model names to benchmark (default: all)")
    parser.add_argument("--list", action="store_true", help="List available models and exit")
    parser.add_argument("--out", default=OUTPUT_DIR, help="Output directory for profiles")
    parser.add_argument("--base", default=BASE, help="Ollama base URL")
    parser.add_argument("--suite", choices=["full", "assistant"], default="full", help="Test suite to run")
    parser.add_argument("--attempts", type=int, default=1, help="Attempts per test (best/min/max aggregated)")
    parser.add_argument("--think", action="store_true", help="Allow reasoning/thinking mode")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Benchmark llama-server directly (OAI endpoint, thinking hard-disabled). "
        "Hot-swap still goes through the proxy.",
    )
    args = parser.parse_args()

    BASE = args.base

    if args.list:
        models = list_models()
        print("Available models:")
        for m in models:
            print(f"  {m}")
        return

    models = args.models if args.models else list_models()
    if not models:
        print("No models found. Is alpaca-proxy running?")
        sys.exit(1)

    print(f"\n{'=' * 78}")
    print("  ALPACA BENCHMARK SUITE")
    print(f"  Host:    {BASE}")
    print(f"  Models:  {len(models)}")
    print(f"  Suite:   {args.suite} ({len(ASSISTANT_TESTS if args.suite == 'assistant' else TESTS)} tests)")
    print(f"  Output:  {args.out}")
    if args.direct:
        print(f"  Direct:  {DIRECT_BASE} (thinking hard-off)")
    print(f"{'=' * 78}")

    all_profiles = []
    for model in models:
        results = benchmark_model(model, args.suite, args.attempts, args.think, args.temperature, args.direct)
        if results is None:
            all_profiles.append({"model": model, "error": "unreachable"})
            continue
        profile = build_profile(model, results, args.suite)
        path = save_profile(profile, args.out)
        all_profiles.append(profile)
        s = profile.get("summary", {})
        print(
            f"\n  ✓ {model}: {s.get('avg_tokens_per_sec', 0):.1f} avg tok/s, {s.get('avg_ttft_wall_ms', 0):.0f}ms TTFT wall → {path}"
        )

    # Save combined report
    report = {
        "generated_at": datetime.datetime.now().isoformat(),
        "host": BASE,
        "models_tested": len(models),
        "profiles": all_profiles,
    }
    report_path = os.path.join(args.out, "benchmark_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print_comparison(all_profiles)
    print(f"  Combined report → {report_path}")


if __name__ == "__main__":
    main()
