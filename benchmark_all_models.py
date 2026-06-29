#!/usr/bin/env python3
"""
alpaca-benchmark — Multi-model benchmark suite for alpaca-proxy / llama-server
Usage:
  python3 benchmark_all_models.py                  # benchmark all models
  python3 benchmark_all_models.py gemma-4-12b-fable5:latest qwen3:8b
  python3 benchmark_all_models.py --list           # list available models
  python3 benchmark_all_models.py --out /my/dir    # custom output dir

Profiles are saved as:  <output_dir>/<model-name>.profile.json
A combined report is:   <output_dir>/benchmark_report.json
"""

import argparse
import datetime
import json
import os
import statistics
import sys
import time
import urllib.request

BASE = os.getenv("OLLAMA_URL", "http://localhost:11434")
OUTPUT_DIR = os.getenv(
    "BENCHMARK_OUTPUT", os.path.join(os.path.dirname(__file__), ".alpaca-router")
)

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


def call_model(model, prompt, num_predict, system=None):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": num_predict,
                "temperature": 0.3,
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
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            raw = resp.read()
            elapsed = time.perf_counter() - t0
            data = json.loads(raw)
    except Exception as e:
        return None, str(e)

    text = data.get("message", {}).get("content", "")
    eval_count = data.get("eval_count", 0)
    eval_dur = data.get("eval_duration", 1) / 1e9
    prompt_dur = data.get("prompt_eval_duration", 0) / 1e9
    tps = eval_count / eval_dur if eval_dur > 0 else 0

    return {
        "response": text,
        "tokens_generated": eval_count,
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "total_time_s": elapsed,
        "ttft_ms": prompt_dur * 1000,
        "tokens_per_sec": tps,
        "eval_duration_s": eval_dur,
    }, None


def benchmark_model(model):
    print(f"\n{'─' * 62}")
    print(f"  MODEL: {model}")
    print(f"  Time:  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'─' * 62}")

    # Warmup ping
    print("  Warming up... ", end="", flush=True)
    _, err = call_model(model, "Hi", 5)
    if err:
        print(f"FAILED — cannot reach model: {err}")
        return None
    print("OK")

    results = []
    for i, test in enumerate(TESTS, 1):
        label = test["label"]
        print(f"  [{i:2d}/{len(TESTS)}] {label:<45} ", end="", flush=True)
        m, err = call_model(model, test["prompt"], test["num_predict"], test.get("system"))
        if err:
            print(f"FAIL ({err[:60]})")
            results.append({**test, "error": err})
            continue
        print(f"{m['tokens_per_sec']:6.1f} tok/s  {m['ttft_ms']:5.0f}ms TTFT")
        results.append({**test, **m, "error": None})

    return results


def build_profile(model, results):
    good = [r for r in results if not r.get("error")]
    if not good:
        return {"model": model, "error": "all tests failed"}

    tps = [r["tokens_per_sec"] for r in good]
    ttft = [r["ttft_ms"] for r in good]
    by_cat = {}
    for r in good:
        by_cat.setdefault(r["category"], []).append(r["tokens_per_sec"])

    return {
        "model": model,
        "benchmarked_at": datetime.datetime.now().isoformat(),
        "host": BASE,
        "summary": {
            "tests_run": len(results),
            "tests_passed": len(good),
            "tests_failed": len(results) - len(good),
            "avg_tokens_per_sec": round(statistics.mean(tps), 2),
            "max_tokens_per_sec": round(max(tps), 2),
            "min_tokens_per_sec": round(min(tps), 2),
            "avg_ttft_ms": round(statistics.mean(ttft), 1),
            "min_ttft_ms": round(min(ttft), 1),
            "max_ttft_ms": round(max(ttft), 1),
        },
        "by_category": {
            cat: {"avg_tokens_per_sec": round(statistics.mean(vals), 2)}
            for cat, vals in by_cat.items()
        },
        "tests": [
            {
                "id": r["id"],
                "label": r["label"],
                "category": r["category"],
                "tokens_per_sec": round(r.get("tokens_per_sec", 0), 2),
                "ttft_ms": round(r.get("ttft_ms", 0), 1),
                "tokens_generated": r.get("tokens_generated", 0),
                "total_time_s": round(r.get("total_time_s", 0), 2),
                "response_preview": r.get("response", "")[:300].replace("\n", " "),
                "error": r.get("error"),
            }
            for r in results
        ],
    }


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
    print(f"\n{'=' * 62}")
    print("  COMPARISON TABLE")
    print(f"{'=' * 62}")
    print(f"  {'Model':<35} {'Avg tok/s':>10} {'Peak':>8} {'TTFT ms':>9}")
    print(f"  {'─' * 35} {'─' * 10} {'─' * 8} {'─' * 9}")
    ranked = sorted(valid, key=lambda p: p["summary"]["avg_tokens_per_sec"], reverse=True)
    for p in ranked:
        s = p["summary"]
        print(
            f"  {p['model']:<35} {s['avg_tokens_per_sec']:>10.1f} {s['max_tokens_per_sec']:>8.1f} {s['avg_ttft_ms']:>9.0f}"
        )
    print(f"{'=' * 62}\n")


def main():
    global BASE
    parser = argparse.ArgumentParser(description="Multi-model alpaca benchmark")
    parser.add_argument("models", nargs="*", help="Model names to benchmark (default: all)")
    parser.add_argument("--list", action="store_true", help="List available models and exit")
    parser.add_argument("--out", default=OUTPUT_DIR, help="Output directory for profiles")
    parser.add_argument("--base", default=BASE, help="Ollama base URL")
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

    print(f"\n{'=' * 62}")
    print("  ALPACA BENCHMARK SUITE")
    print(f"  Host:    {BASE}")
    print(f"  Models:  {len(models)}")
    print(f"  Tests:   {len(TESTS)} per model")
    print(f"  Output:  {args.out}")
    print(f"{'=' * 62}")

    all_profiles = []
    for model in models:
        results = benchmark_model(model)
        if results is None:
            all_profiles.append({"model": model, "error": "unreachable"})
            continue
        profile = build_profile(model, results)
        path = save_profile(profile, args.out)
        all_profiles.append(profile)
        s = profile.get("summary", {})
        print(
            f"\n  ✓ {model}: {s.get('avg_tokens_per_sec', 0):.1f} avg tok/s, {s.get('avg_ttft_ms', 0):.0f}ms TTFT → {path}"
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
