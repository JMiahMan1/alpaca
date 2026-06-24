#!/usr/bin/env python3
"""
llm_benchmark_suite.py

Centralized benchmarking suite for evaluating LLM models across SharedLLM task categories.
This analyzes models for the various Jobs that will be used with SharedLLM and creates comprehensive
benchmarks to determine optimal models and llama.cpp settings.

The suite organizes tests into the core categories needed by SharedLLM:
- Coding assistance (debugging, refactoring, implementation)
- Reasoning (logic, math, deduction)
- Instruction-following (extracting structured data, summarization)
- Creative tasks (storytelling, content generation)
- Home automation (device control, status checking, automation)
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, List
import httpx
import requests
import sys

class LLMModelBenchmark:
    """
    Comprehensive benchmark suite for LLM model evaluation across SharedLLM task categories.
    """

    def __init__(self):
        ollama_env = os.getenv("OLLAMA_SERVER_URLS", "")
        if ollama_env:
            self.OLLAMA_SERVER_URLS = [u.strip() for u in ollama_env.split(",") if u.strip()]
        else:
            self.OLLAMA_SERVER_URLS = [
                "http://localhost:8080",
                "http://llama-server:8080",
                "http://llama-server-primary:11434",
                "http://llama-server-secondary:11434"
            ]

        proxy_env = os.getenv("PROXY_SERVER_URLS", "")
        if proxy_env:
            self.PROXY_SERVER_URLS = [u.strip() for u in proxy_env.split(",") if u.strip()]
        else:
            self.PROXY_SERVER_URLS = [
                "http://localhost:11434",
                "http://alpaca-proxy:11434",
                "http://alpaca-proxy-primary:11434",
                "http://alpaca-proxy-secondary:11445"
            ]
        self.RESULTS_DIR = Path("data/llm_benchmarks")
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    async def discover_ollama_models(self, base_url: str) -> List[str]:
        """Dynamically discover available models from Ollama endpoint."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{base_url}/api/tags", timeout=3.0)
                if response.status_code == 200:
                    data = response.json()
                    models = [model.get("name") for model in data.get("models", [])]
                    print(f"[discover] Discovered {len(models)} models from {base_url}")
                    return models
                else:
                    print(f"[discover] Warning: Could not fetch models from {base_url} (HTTP {response.status_code})")
                    return []
        except Exception as e:
            print(f"[discover] Error discovering models from {base_url}: {e}")
            return []

    async def discover_proxy_models(self, base_url: str) -> List[str]:
        """Dynamically discover available models from Alpaca proxy endpoint."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{base_url}/api/tags", timeout=3.0)
                if response.status_code == 200:
                    data = response.json()
                    models = [model.get("model") or model.get("name") for model in data.get("models", [])]
                    print(f"[discover] Discovered {len(models)} models from proxy {base_url}")
                    return models
                else:
                    print(f"[discover] Warning: Could not fetch models from proxy {base_url} (HTTP {response.status_code})")
                    return []
        except Exception as e:
            print(f"[discover] Error discovering models from proxy {base_url}: {e}")
            return []

    async def discover_all_models(self) -> List[str]:
        """Discover models from all available Ollama instances."""
        print("🔍 Discovering available models across all Ollama instances...")
        all_models = []
        for url in self.OLLAMA_SERVER_URLS:
            models = await self.discover_ollama_models(url)
            if models:
                all_models.extend(models)
        unique_models = list(dict.fromkeys(all_models))
        if not unique_models:
            print("[discover] No models discovered. Using fallback models from environment or defaults...")
            unique_models = self._get_fallback_models()
        print(f"✅ Discovery complete. Total unique models: {len(unique_models)}")
        return unique_models

    async def discover_all_proxy_models(self) -> List[str]:
        """Discover models from all available Alpaca proxy instances."""
        print("🔍 Discovering available models across all Alpaca proxies...")
        all_models = []
        for url in self.PROXY_SERVER_URLS:
            models = await self.discover_proxy_models(url)
            if models:
                all_models.extend(models)
        unique_models = list(dict.fromkeys(all_models))
        if not unique_models:
            print("[discover] No proxy models discovered. Using fallback models from environment or defaults...")
            unique_models = self._get_fallback_models()
        print(f"✅ Proxy discovery complete. Total unique models: {len(unique_models)}")
        return unique_models

    def _get_fallback_models(self) -> List[str]:
        """Get fallback models from environment variable or defaults."""
        env_models = os.getenv("BENCHMARK_MODELS", "")
        if env_models:
            print("[fallback] Using models from BENCHMARK_MODELS environment variable")
            return [m.strip() for m in env_models.split(",") if m.strip()]
        print("[fallback] Using default models...")
        return ["qwen3:8b", "qwen2.5-coder:7b", "qwen3.5:9b"]

    def _coding_tests(self, model: str) -> List[Dict]:
        """Test coding-related capabilities."""
        return [
            {
                "id": "debug_fix",
                "category": "coding",
                "label": "Python: debug logic error",
                "prompt": "Find and fix the bug in this function:\n\n```\ndef sum_list(items):\n    total = 0\n    for i in range(1, len(items)):\n        total += items[i]\n    return total\n```\nThe function should sum all list items, not skip the first one.",
                "num_predict": 400,
            },
            {
                "id": "code_refactor",
                "category": "coding",
                "label": "Code: refactor for efficiency",
                "prompt": "Refactor this code to be more Pythonic and remove the nested loops:\n\n```\ndef find_unique_numbers(list1, list2, list3):\n    result = []\n    for item in list1:\n        if item not in result:\n            result.append(item)\n    for item in list2:\n        if item not in result:\n            result.append(item)\n    for item in list3:\n        if item not in result:\n            result.append(item)\n    return result\n```",
                "num_predict": 500,
            },
        ]

    def _reasoning_tests(self, model: str) -> List[Dict]:
        """Test logical reasoning and mathematical abilities."""
        return [
            {
                "id": "logic_puzzle",
                "category": "reasoning",
                "label": "Logic: identify rule",
                "prompt": "What rule is being followed in this sequence? 2, 6, 12, 20, 30... and what comes next?",
                "num_predict": 400,
            },
            {
                "id": "math_problem",
                "category": "reasoning",
                "label": "Math: train meeting problem",
                "prompt": "Two trains leave from different cities heading toward each other. Train A travels 60 mph and leaves at 9:00 AM. Train B travels 80 mph and leaves at 10:00 AM. The cities are 500 miles apart. When do they meet?",
                "num_predict": 500,
            },
        ]

    def _instruction_tests(self, model: str) -> List[Dict]:
        """Test instruction-following and structured output."""
        return [
            {
                "id": "json_extraction",
                "category": "instruction",
                "label": "JSON: extract structured data",
                "prompt": "Extract the person's name, age, and location from this text and return as JSON: 'John Doe, 35, lives in Boston, MA'",
                "num_predict": 300,
            },
            {
                "id": "summarization",
                "category": "instruction",
                "label": "Summarization: 3 bullet points",
                "prompt": "Summarize in exactly 3 bullet points: 'The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms. This innovation replaced recurrent neural networks and enabled parallelization, significantly improving training efficiency. Transformers now power state-of-the-art models like GPT, BERT, and T5.'",
                "num_predict": 400,
            },
        ]

    def _creative_tests(self, model: str) -> List[Dict]:
        """Test creative content generation."""
        return [
            {
                "id": "story_start",
                "category": "creative",
                "label": "Creative: sci-fi story opening",
                "prompt": "Write a compelling 4-sentence opening for a sci-fi story about an AI that discovers it's been dreaming.",
                "num_predict": 400,
            },
            {
                "id": "analogy",
                "category": "creative",
                "label": "Creative: generate analogy",
                "prompt": "Create an analogy comparing mental health to something tangible and useful.",
                "num_predict": 400,
            },
        ]

    def _home_automation_tests(self, model: str) -> List[Dict]:
        """Test home automation related capabilities."""
        return [
            {
                "id": "device_control",
                "category": "home_automation",
                "label": "HA: control smart device",
                "prompt": "You are a home automation assistant. A user says: 'Turn on the bedroom light and set it to 60% brightness.' Describe in plain text exactly what action you would take and confirm it to the user.",
                "num_predict": 300,
            },
            {
                "id": "device_status",
                "category": "home_automation",
                "label": "HA: report device status",
                "prompt": "You are a home automation assistant. A user asks: 'Is the thermostat currently set to 68 degrees?' Describe what you would check and give a plausible confirmation response to the user.",
                "num_predict": 300,
            },
        ]

    async def test_model_proxy(self, model: str, test: Dict) -> Dict:
        """Test a model against a proxy endpoint."""
        last_error = None
        for proxy_url in self.PROXY_SERVER_URLS:
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    start_t = time.time()
                    response = await client.post(
                        f"{proxy_url}/api/chat",
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": test["prompt"]}],
                            "stream": False,
                            "options": {"num_predict": test.get("num_predict", 50), "temperature": 0.3},
                        },
                        timeout=120.0,
                    )
                    elapsed = time.time() - start_t
                    if response.status_code == 200:
                        data = response.json()
                        # Use Ollama's internal eval durations when available (nanoseconds -> seconds)
                        eval_ns = data.get("eval_duration", 0)
                        prompt_ns = data.get("prompt_eval_duration", 0)
                        latency = (eval_ns + prompt_ns) / 1e9 if (eval_ns or prompt_ns) else elapsed
                        response_text = data.get("message", {}).get("content") or data.get("response", "")
                        return {
                            "proxy": proxy_url,
                            "success": True,
                            "prompt": test["prompt"],
                            "latency": round(latency, 3),
                            "response": response_text,
                            "tokens_generated": data.get("eval_count", 0),
                            "eval_duration": eval_ns,
                            "prompt_eval_duration": prompt_ns,
                            "error": None,
                        }
                    else:
                        return {
                            "proxy": proxy_url,
                            "success": False,
                            "prompt": test["prompt"],
                            "latency": 0,
                            "response": None,
                            "tokens_generated": 0,
                            "error": f"HTTP {response.status_code}: {response.text}",
                        }
            except Exception as e:
                last_error = e
                continue
        return {"proxy": "all_failed", "success": False, "prompt": test["prompt"], "latency": 0, "response": None, "tokens_generated": 0, "error": str(last_error) if last_error else "Unknown error"}

    async def test_model_direct(self, model: str, test: Dict) -> Dict:
        """Test a model directly without proxy."""
        last_error = None
        for ollama_url in self.OLLAMA_SERVER_URLS:
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    start_t = time.time()
                    response = await client.post(
                        f"{ollama_url}/api/generate",
                        json={
                            "model": model,
                            "prompt": test["prompt"],
                            "stream": False,
                            "options": {"num_predict": test.get("num_predict", 50), "temperature": 0.3},
                        },
                        timeout=120.0,
                    )
                    elapsed = time.time() - start_t
                    if response.status_code == 200:
                        data = response.json()
                        eval_ns = data.get("eval_duration", 0)
                        prompt_ns = data.get("prompt_eval_duration", 0)
                        latency = (eval_ns + prompt_ns) / 1e9 if (eval_ns or prompt_ns) else elapsed
                        return {
                            "ollama_url": ollama_url,
                            "success": True,
                            "prompt": test["prompt"],
                            "latency": round(latency, 3),
                            "response": data.get("response", ""),
                            "tokens_generated": data.get("eval_count", 0),
                            "eval_duration": eval_ns,
                            "prompt_eval_duration": prompt_ns,
                            "error": None,
                        }
                    else:
                        return {
                            "ollama_url": ollama_url,
                            "success": False,
                            "prompt": test["prompt"],
                            "latency": 0,
                            "response": None,
                            "tokens_generated": 0,
                            "error": f"HTTP {response.status_code}",
                        }
            except Exception as e:
                last_error = e
                continue
        return {"ollama_url": "all_failed", "success": False, "prompt": test["prompt"], "latency": 0, "response": None, "tokens_generated": 0, "error": str(last_error) if last_error else "Unknown error"}

    def _display_live_results(self, results: Dict):
        """Display benchmark results in a formatted UI."""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS UI")
        print("=" * 80)
        total_models = results.get("models_tested", 0)
        benchmark_time = results.get("generated_at", "Unknown")
        benchmark_type = results.get("benchmark_type", "Unknown")
        print(f"\nOverview:")
        print(f"  Models Tested: {total_models}")
        print(f"  Benchmark Type: {benchmark_type}")
        print(f"  Generated At: {benchmark_time}")
        results_data = results.get("results", [])
        if not results_data:
            print("\nNo results data available.")
            return
        print(f"\nModel Performance Summary:")
        print(f"{'Model':<30} {'Category':<20} {'Success Rate':<15} {'Avg Tok/s':<15} {'Avg TTFT (ms)':<15}")
        print(f"{'='*110}")
        for model_result in results_data:
            model_name = model_result.get("model", "Unknown")
            for category_key in ["coding", "reasoning", "instruction", "creative", "home_automation"]:
                cat_key = f"category_{category_key}"
                if cat_key in model_result:
                    cat_stats = model_result[cat_key]
                    success_rate = cat_stats.get("tests_passed", 0) / max(cat_stats.get("tests_run", 1), 1) * 100
                    avg_tps = cat_stats.get("avg_tokens_per_sec", 0)
                    avg_ttft = cat_stats.get("avg_ttft_ms", 0)
                    print(f"{model_name:<30} {category_key:<20} {success_rate:>6.1f}% {avg_tps:>6.1f} {avg_ttft:>6.0f}")
        print(f"\nDetailed Test Results:")
        print("-" * 80)
        for model_result in results_data:
            model_name = model_result.get("model", "Unknown")
            print(f"\n{model_name}:")
            for category_key in ["coding", "reasoning", "instruction", "creative", "home_automation"]:
                cat_key = f"category_{category_key}"
                if cat_key in model_result:
                    cat_stats = model_result[cat_key]
                    passed = cat_stats.get("tests_passed", 0)
                    total = cat_stats.get("tests_run", 0)
                    print(f"  {category_key.title()}: {passed}/{total} passed")
        print("\n" + "=" * 80)
        print("RESULTS DISPLAY COMPLETE")
        print("=" * 80)

    async def benchmark_model(self, model: str, use_proxy: bool, progress_callback=None, cancel_event=None) -> Dict:
        """Run comprehensive benchmark for a single model."""
        print(f"\n=== BENCHMARKING MODEL: {model} (via {'Proxy' if use_proxy else 'Direct'}) ===")
        results = {"model": model, "benchmark_type": "proxy" if use_proxy else "direct", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        categories = {"coding": self._coding_tests, "reasoning": self._reasoning_tests, "instruction": self._instruction_tests, "creative": self._creative_tests, "home_automation": self._home_automation_tests}
        for category, test_func in categories.items():
            if cancel_event and cancel_event.is_set():
                print(f"\n[cancel] Model benchmarking cancelled early.")
                break
            print(f"\n--- {category.upper()} Tests ---")
            tests = test_func(model)
            category_results = []
            for i, test in enumerate(tests, 1):
                if cancel_event and cancel_event.is_set():
                    break
                print(f"[{category}] Test {i}/{len(tests)}: {test['label']} ", end="", flush=True)
                
                if progress_callback:
                    try:
                        import inspect
                        if inspect.iscoroutinefunction(progress_callback):
                            await progress_callback("test_start", {
                                "model": model,
                                "category": category,
                                "test_id": test["id"],
                                "test_label": test["label"]
                            })
                        else:
                            progress_callback("test_start", {
                                "model": model,
                                "category": category,
                                "test_id": test["id"],
                                "test_label": test["label"]
                            })
                    except Exception as callback_err:
                        print(f"\n[callback error] {callback_err}")

                test_result = await self.test_model_proxy(model, test) if use_proxy else await self.test_model_direct(model, test)

                # Post-response validation: mark as failed if response is empty or model refused
                if test_result["success"]:
                    raw_resp = test_result.get("response") or ""
                    # Strip think tags to get actual answer
                    import re as _re
                    actual_resp = _re.sub(r'<think>[\s\S]*?</think>', '', raw_resp, flags=_re.IGNORECASE).strip()
                    refusal_phrases = [
                        "i cannot", "i can't", "i don't have access", "i do not have access",
                        "i'm unable", "i am unable", "no tools", "as an ai", "i don't have the ability",
                        "i do not have the ability", "i'm not able", "i am not able"
                    ]
                    if not actual_resp:
                        test_result["success"] = False
                        test_result["error"] = "Empty response (model produced no output after thinking)"
                    elif any(p in actual_resp.lower() for p in refusal_phrases):
                        test_result["success"] = False
                        test_result["error"] = "Model refused to attempt the task"

                if test_result["success"]:
                    print(f"✓ ({test_result['latency']:.2f}s)")
                else:
                    print(f"✗ ({test_result['error']})")
                test_result.update({"test_id": test["id"], "test_category": category, "test_label": test["label"]})
                category_results.append(test_result)
                
                if progress_callback:
                    try:
                        import inspect
                        if inspect.iscoroutinefunction(progress_callback):
                            await progress_callback("test_complete", {
                                "model": model,
                                "category": category,
                                "test_id": test["id"],
                                "test_label": test["label"],
                                "result": test_result
                            })
                        else:
                            progress_callback("test_complete", {
                                "model": model,
                                "category": category,
                                "test_id": test["id"],
                                "test_label": test["label"],
                                "result": test_result
                            })
                    except Exception as callback_err:
                        print(f"\n[callback error] {callback_err}")

            category_stats = self._calculate_category_stats(category_results)
            key = f"category_{category}"
            results[key] = category_stats
        return results

    def _calculate_category_stats(self, results: List[Dict]) -> Dict:
        """Calculate statistics for a test category."""
        successful_tests = [r for r in results if r["success"]]
        if not successful_tests:
            return {"tests_run": len(results), "tests_passed": 0, "avg_tokens_per_sec": 0, "avg_ttft_ms": 0, "tests": results}
        total_tokens = sum(r.get("tokens_generated", 0) for r in successful_tests)
        total_time = sum(self._extract_duration(r) for r in successful_tests)
        avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
        avg_ttft_ms = self._calculate_avg_ttft(successful_tests)
        return {"tests_run": len(results), "tests_passed": len(successful_tests), "avg_tokens_per_sec": round(avg_tokens_per_sec, 2), "avg_ttft_ms": round(avg_ttft_ms, 1), "tests": results}

    def _extract_duration(self, result: Dict) -> float:
        if "eval_duration" in result and "prompt_eval_duration" in result:
            return (result["eval_duration"] + result["prompt_eval_duration"]) / 1e9
        return result.get("latency", 0)

    def _calculate_avg_ttft(self, results: List[Dict]) -> float:
        total_ttft = 0
        for result in results:
            if "prompt_eval_duration" in result:
                ttft = result["prompt_eval_duration"] / 1e9 * 1000
                total_ttft += ttft
            elif "latency" in result:
                total_ttft += result["latency"] * 1000
        return total_ttft / len(results) if results else 0

    async def run_model_benchmarks(self, models: List[str], use_proxy: bool, progress_callback=None, cancel_event=None) -> Dict:
        print("=" * 80)
        print(f"COMPREHENSIVE LLM MODEL BENCHMARKING SUITE")
        print(f"Running benchmarks via {'Proxy' if use_proxy else 'Direct'} (vs {self.PROXY_SERVER_URLS[0] if use_proxy else self.OLLAMA_SERVER_URLS[0]})")
        print(f"Models: {models}")
        print("=" * 80)
        all_results = {"benchmark_version": "2.0.0", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "benchmark_type": "proxy" if use_proxy else "direct", "models_tested": len(models), "results": []}
        
        if progress_callback:
            try:
                # Count total tests
                total_tests = 0
                for model in models:
                    categories = [self._coding_tests(model), self._reasoning_tests(model), self._instruction_tests(model), self._creative_tests(model), self._home_automation_tests(model)]
                    total_tests += sum(len(c) for c in categories)
                
                start_data = {
                    "models": models,
                    "use_proxy": use_proxy,
                    "total_models": len(models),
                    "total_tests": total_tests,
                    "timestamp": all_results["generated_at"]
                }
                import inspect
                if inspect.iscoroutinefunction(progress_callback):
                    await progress_callback("benchmark_start", start_data)
                else:
                    progress_callback("benchmark_start", start_data)
            except Exception as callback_err:
                print(f"[callback error] {callback_err}")

        for model in models:
            if cancel_event and cancel_event.is_set():
                print(f"\n[cancel] Benchmark cancelled before starting model: {model}")
                break
            
            if progress_callback:
                try:
                    import inspect
                    if inspect.iscoroutinefunction(progress_callback):
                        await progress_callback("model_start", {"model": model})
                    else:
                        progress_callback("model_start", {"model": model})
                except Exception as callback_err:
                    print(f"[callback error] {callback_err}")

            model_results = await self.benchmark_model(model, use_proxy, progress_callback, cancel_event)
            all_results["results"].append(model_results)

            if progress_callback:
                try:
                    import inspect
                    if inspect.iscoroutinefunction(progress_callback):
                        await progress_callback("model_complete", {"model": model, "results": model_results})
                    else:
                        progress_callback("model_complete", {"model": model, "results": model_results})
                except Exception as callback_err:
                    print(f"[callback error] {callback_err}")

        if cancel_event and cancel_event.is_set():
            all_results["status"] = "cancelled"
        else:
            all_results["status"] = "completed"

        if all_results["results"]:
            save_file = self.RESULTS_DIR / f"benchmarks_{time.strftime('%Y%m%d_%H%M%S')}_{'proxy' if use_proxy else 'direct'}.json"
            with open(save_file, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"\n{'='*80}")
            print(f"BENCHMARKING COMPLETE!")
            print(f"Results saved to: {save_file}")
            print(f"{'='*80}")
            all_results["saved_as"] = str(save_file)
        
        if progress_callback:
            try:
                import inspect
                if inspect.iscoroutinefunction(progress_callback):
                    await progress_callback("benchmark_complete", all_results)
                else:
                    progress_callback("benchmark_complete", all_results)
            except Exception as callback_err:
                print(f"[callback error] {callback_err}")

        return all_results

    def run_optimization_pipeline(self, models: List[str]):
        print("=" * 80)
        print(f"LLM MODEL OPTIMIZATION PIPELINE")
        print(f"Testing models: {models}")
        print("=" * 80)
        print("\n1. Running direct benchmarks (using ollama-api directly)...")
        direct_results = asyncio.run(self.run_model_benchmarks(models, use_proxy=False))
        save_file_1 = self.RESULTS_DIR / "direct_benchmarks_latest.json"
        with open(save_file_1, "w") as f:
            json.dump(direct_results, f, indent=2, default=str)
        print("\n2. Running proxy benchmarks (using alpaca-proxy)...")
        proxy_results = asyncio.run(self.run_model_benchmarks(models, use_proxy=True))
        save_file_2 = self.RESULTS_DIR / "proxy_benchmarks_latest.json"
        with open(save_file_2, "w") as f:
            json.dump(proxy_results, f, indent=2, default=str)
        print("\n" + "=" * 80)
        print("LIVE RESULTS DISPLAY")
        print("=" * 80)
        print("\nDirect Benchmarks:")
        self._display_live_results(direct_results)
        print("\nProxy Benchmarks:")
        self._display_live_results(proxy_results)
        print("\n" + "=" * 80)
        print("OPTIMIZATION PIPELINE COMPLETE")
        print("=" * 80)
        print("\nBenchmark Files:")
        print(f"  Direct: {save_file_1}")
        print(f"  Proxy:  {save_file_2}")
        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPLETE!")
        print(f"All results are saved in: {self.RESULTS_DIR}")
        print("=" * 80)
        return {"direct_results": direct_results, "proxy_results": proxy_results, "results_dir": str(self.RESULTS_DIR)}

async def main():
    print("=" * 80)
    print("MULTI-CONTAINER LLM BENCHMARKING SUITE")
    print("=" * 80)
    suite = LLMModelBenchmark()
    print("\n🔍 Discovering available models...")
    models = await suite.discover_all_models()
    if not models:
        print("\n❌ Error: No models discovered.")
        print("Please set BENCHMARK_MODELS environment variable or ensure Ollama servers are running.")
        print("Using default models...")
        models = suite._get_fallback_models()
    print(f"\nModels to benchmark: {models}")
    print("\nSelect benchmark type:")
    print("1. Proxy benchmarks (via Alpaca proxy)")
    print("2. Direct benchmarks (direct to Ollama)")
    print("3. Both (run both proxy and direct)")
    try:
        choice = input("Enter choice (1/2/3): ").strip()
    except EOFError:
        choice = "3"
    if choice == "1":
        await suite.run_model_benchmarks(models, use_proxy=True)
    elif choice == "2":
        await suite.run_model_benchmarks(models, use_proxy=False)
    elif choice == "3":
        await suite.run_model_benchmarks(models, use_proxy=True)
        await suite.run_model_benchmarks(models, use_proxy=False)
    else:
        await suite.run_model_benchmarks(models, use_proxy=True)
        await suite.run_model_benchmarks(models, use_proxy=False)
    print("\n" + "=" * 80)
    print("BENCHMARKING COMPLETE")
    print("Check data/llm_benchmarks/ for detailed results")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())