#!/usr/bin/env python3
"""
llm_benchmark_suite.py

Centralized benchmarking suite for evaluating LLM models across SharedLLM task categories.
Divided into two distinct execution phases:
1. Functional Benchmarks: Evaluates accuracy and output quality (pass/fail correctness).
2. Performance Benchmarks: Evaluates hardware footprint (RAM/VRAM), TTFT, and TPS under load.
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("benchmark_suite")


class LLMModelBenchmark:
    """
    Refactored benchmark suite separating functional capabilities from resource footprints.
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
                "http://llama-server-secondary:11434",
            ]

        proxy_env = os.getenv("PROXY_SERVER_URLS", "")
        if proxy_env:
            self.PROXY_SERVER_URLS = [u.strip() for u in proxy_env.split(",") if u.strip()]
        else:
            self.PROXY_SERVER_URLS = [
                "http://localhost:11434",
                "http://alpaca-proxy:11434",
                "http://alpaca-proxy-primary:11434",
                "http://alpaca-proxy-secondary:11445",
            ]
        self.RESULTS_DIR = Path("data/llm_benchmarks")
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    async def discover_ollama_models(self, base_url: str) -> List[str]:
        """Dynamically discover available models from Ollama endpoint."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = [model.get("name") for model in data.get("models", [])]
                    print(f"[discover] Discovered {len(models)} models from {base_url}")
                    return models
                return []
        except Exception as e:
            print(f"[discover] Error discovering models from {base_url}: {e}")
            return []

    async def discover_proxy_models(self, base_url: str) -> List[str]:
        """Dynamically discover available models from Alpaca proxy endpoint."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = [
                        model.get("model") or model.get("name") for model in data.get("models", [])
                    ]
                    print(f"[discover] Discovered {len(models)} models from proxy {base_url}")
                    return models
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
            unique_models = self._get_fallback_models()
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
            unique_models = self._get_fallback_models()
        return unique_models

    def _get_fallback_models(self) -> List[str]:
        env_models = os.getenv("BENCHMARK_MODELS", "")
        if env_models:
            return [m.strip() for m in env_models.split(",") if m.strip()]
        return ["qwen3:8b", "qwen2.5-coder:7b", "qwen3.5:9b"]

    def _coding_tests(self, model: str) -> List[Dict]:
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
            {
                "id": "guess_game",
                "category": "coding",
                "label": "Game: Number Guessing Game",
                "prompt": "Write a fully functional, interactive Python script for a CLI Number Guessing Game. The game should randomly select a secret number between 1 and 100, give the player 7 attempts, provide higher/lower feedback, and display game stats (attempts used, win/loss) at the end.",
                "num_predict": 600,
            },
            {
                "id": "text_adventure",
                "category": "coding",
                "label": "Game: Text Adventure Game",
                "prompt": "Write a short, interactive text-based adventure game in Python. The player should start in a room with at least two doors (e.g. gold door, monster door). The script must use input() to take player choices, branch the story path based on choices, and lead to at least one winning outcome and one losing outcome.",
                "num_predict": 600,
            },
        ]

    def _reasoning_tests(self, model: str) -> List[Dict]:
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
                "prompt": "Create an analogy comparing friendship to something tangible and useful.",
                "num_predict": 400,
            },
        ]

    def _home_automation_tests(self, model: str) -> List[Dict]:
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

    def strip_thinking(self, text: str) -> str:
        """Remove <think>/<thinking> blocks from response — thinking models may still emit them."""
        return re.sub(r"<(think|thinking)>[\s\S]*?</\1>", "", text, flags=re.IGNORECASE).strip()

    def _verify_functional_response(self, test_id: str, response: str) -> bool:
        """Evaluate functional response correctness based on the target requirements."""
        if not response or len(response.strip()) == 0:
            return False

        # Clean the response from potential think tags
        cleaned = (
            re.sub(r"<(think|thinking)>[\s\S]*?</\1>", "", response, flags=re.IGNORECASE).strip().lower()
        )

        refusals = [
            "cannot",
            "can't",
            "unable",
            "not able",
            "do not have access",
            "don't have access",
        ]
        if any(ref in cleaned for ref in refusals) and len(cleaned) < 150:
            return False

        if test_id == "debug_fix":
            return any(x in cleaned for x in ["range(len", "range(0", "for item in", "sum("])

        elif test_id == "code_refactor":
            return "set(" in cleaned

        elif test_id == "guess_game":
            return "random" in cleaned and "input(" in cleaned and any(x in cleaned for x in ["randint", "randrange", "secret", "number"])

        elif test_id == "text_adventure":
            return "input(" in cleaned and any(x in cleaned for x in ["door", "choice", "room", "path"]) and any(x in cleaned for x in ["win", "lose", "gold", "monster"])

        elif test_id == "logic_puzzle":
            return "42" in cleaned

        elif test_id == "math_problem":
            return any(x in cleaned for x in ["1:08", "1:09", "1 pm", "meeting", "4 hours"])

        elif test_id == "json_extraction":
            has_name = "john" in cleaned
            has_age = "35" in cleaned
            has_loc = "boston" in cleaned
            has_brackets = "{" in cleaned and "}" in cleaned
            return has_brackets and has_name and has_age and has_loc

        elif test_id == "summarization":
            bullets = re.findall(r"(?:^\s*[-*•+\d]\s+.*$)|(?:^\d+\..*$)", response, re.MULTILINE)
            return len(bullets) >= 2

        elif test_id == "device_control":
            return "bedroom" in cleaned and ("60%" in cleaned or "60" in cleaned)

        elif test_id == "device_status":
            return "thermostat" in cleaned and ("68" in cleaned or "sixty-eight" in cleaned)

        return True

    class ResourceSampler:
        """Context manager to sample peak CPU, RAM, and VRAM utilization during a query."""

        def __init__(self, container_name: str = "llama-server"):
            self.container_name = container_name
            self.peak_ram_pct = 0.0
            self.peak_vram_mb = 0
            self.vram_total_mb = 0
            self.active = False

        async def _sample_loop(self):
            while self.active:
                ram = psutil.virtual_memory().percent
                if ram > self.peak_ram_pct:
                    self.peak_ram_pct = ram

                try:
                    proc = await asyncio.create_subprocess_exec(
                        "docker",
                        "exec",
                        self.container_name,
                        "nvidia-smi",
                        "--query-gpu=memory.total,memory.used",
                        "--format=csv,noheader,nounits",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, _ = await proc.communicate()
                    if proc.returncode == 0:
                        lines = stdout.decode().strip().split("\n")
                        if lines:
                            total, used = [int(x.strip()) for x in lines[0].split(",")]
                            self.vram_total_mb = total
                            if used > self.peak_vram_mb:
                                self.peak_vram_mb = used
                except Exception:
                    try:
                        proc = await asyncio.create_subprocess_exec(
                            "nvidia-smi",
                            "--query-gpu=memory.total,memory.used",
                            "--format=csv,noheader,nounits",
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        stdout, _ = await proc.communicate()
                        if proc.returncode == 0:
                            lines = stdout.decode().strip().split("\n")
                            if lines:
                                total, used = [int(x.strip()) for x in lines[0].split(",")]
                                self.vram_total_mb = total
                                if used > self.peak_vram_mb:
                                    self.peak_vram_mb = used
                    except Exception:
                        pass
                await asyncio.sleep(0.2)

        def start(self):
            self.active = True
            self.loop_task = asyncio.create_task(self._sample_loop())

        async def stop(self):
            self.active = False
            if hasattr(self, "loop_task"):
                await self.loop_task

    async def test_model_proxy(
        self, model: str, test: Dict, sampler: Optional[ResourceSampler] = None
    ) -> Dict:
        """Test a model against a proxy endpoint."""
        last_error = None
        for proxy_url in self.PROXY_SERVER_URLS:
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    start_t = time.time()
                    if sampler:
                        sampler.start()
                    response = await client.post(
                        f"{proxy_url}/api/chat",
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": test["prompt"]}],
                            "stream": False,
                            # Disable thinking mode so MoE / thinking models don't
                            # spend all tokens on <thinking> and return empty content.
                            "think": False,
                            "options": {
                                "num_predict": test.get("num_predict", 50),
                                "temperature": 0.3,
                            },
                        },
                        headers={"X-Request-Source": "shared-llm/benchmark"},
                        timeout=120.0,
                    )
                    elapsed = time.time() - start_t
                    if sampler:
                        await sampler.stop()

                    if response.status_code == 200:
                        data = response.json()
                        eval_ns = data.get("eval_duration", 0)
                        prompt_ns = data.get("prompt_eval_duration", 0)
                        latency = (eval_ns + prompt_ns) / 1e9 if (eval_ns or prompt_ns) else elapsed
                        response_text = self.strip_thinking(data.get("message", {}).get("content") or data.get(
                            "response", ""
                        ))
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
                if sampler:
                    await sampler.stop()
                last_error = e
                continue
        return {
            "proxy": "all_failed",
            "success": False,
            "prompt": test["prompt"],
            "latency": 0,
            "response": None,
            "tokens_generated": 0,
            "error": str(last_error) if last_error else "Unknown error",
        }

    async def test_model_direct(
        self, model: str, test: Dict, sampler: Optional[ResourceSampler] = None
    ) -> Dict:
        """Test a model directly without proxy."""
        last_error = None
        for ollama_url in self.OLLAMA_SERVER_URLS:
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    start_t = time.time()
                    if sampler:
                        sampler.start()
                    response = await client.post(
                        f"{ollama_url}/api/generate",
                        json={
                            "model": model,
                            "prompt": test["prompt"],
                            "stream": False,
                            # Disable thinking mode so MoE / thinking models don't
                            # spend all tokens on <thinking> and return empty content.
                            "think": False,
                            "options": {
                                "num_predict": test.get("num_predict", 50),
                                "temperature": 0.3,
                            },
                        },
                        timeout=120.0,
                    )
                    elapsed = time.time() - start_t
                    if sampler:
                        await sampler.stop()

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
                            "response": self.strip_thinking(data.get("response", "")),
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
                if sampler:
                    await sampler.stop()
                last_error = e
                continue
        return {
            "ollama_url": "all_failed",
            "success": False,
            "prompt": test["prompt"],
            "latency": 0,
            "response": None,
            "tokens_generated": 0,
            "error": str(last_error) if last_error else "Unknown error",
        }

    def _display_live_results(self, results: Dict):
        """Display benchmark results in a formatted UI."""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS UI")
        print("=" * 80)
        total_models = results.get("models_tested", 0)
        benchmark_time = results.get("generated_at", "Unknown")
        mode = results.get("benchmark_mode", "all")
        print("\nOverview:")
        print(f"  Models Tested: {total_models}")
        print(f"  Execution Mode: {mode}")
        print(f"  Generated At: {benchmark_time}")

        results_data = results.get("results", [])
        if not results_data:
            print("\nNo results data available.")
            return

        if mode in ("functional", "all"):
            print("\n--- Functional Correctness Results ---")
            print(f"{'Model':<30} {'Category':<20} {'Accuracy':<15}")
            print("-" * 70)
            for model_result in results_data:
                model_name = model_result.get("model", "Unknown")
                for category_key in [
                    "coding",
                    "reasoning",
                    "instruction",
                    "creative",
                    "home_automation",
                ]:
                    cat_key = f"category_{category_key}"
                    if cat_key in model_result:
                        cat_stats = model_result[cat_key]
                        success_rate = (
                            cat_stats.get("tests_passed", 0)
                            / max(cat_stats.get("tests_run", 1), 1)
                            * 100
                        )
                        print(f"{model_name:<30} {category_key:<20} {success_rate:>6.1f}%")

        if mode in ("performance", "all"):
            print("\n--- Hardware & Performance Results ---")
            print(
                f"{'Model':<30} {'TPS':<12} {'TTFT (ms)':<12} {'Peak RAM %':<14} {'Peak VRAM (MB)':<14}"
            )
            print("-" * 85)
            for model_result in results_data:
                model_name = model_result.get("model", "Unknown")
                perf = model_result.get("performance_metrics", {})
                if perf:
                    print(
                        f"{model_name:<30} {perf.get('avg_tps', 0.0):>6.1f} {perf.get('avg_ttft_ms', 0.0):>10.1f} {perf.get('peak_ram_pct', 0.0):>12.1f}% {perf.get('peak_vram_mb', 0):>12}"
                    )

        print("\n" + "=" * 80)

    async def benchmark_model_functional(
        self, model: str, use_proxy: bool, progress_callback=None, cancel_event=None
    ) -> Dict:
        """Run only functional (accuracy) tests on a model."""
        print(f"\n--- Running Functional Correctness Suite for: {model} ---")
        results: Dict[str, Any] = {"model": model, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        categories = {
            "coding": self._coding_tests,
            "reasoning": self._reasoning_tests,
            "instruction": self._instruction_tests,
            "creative": self._creative_tests,
            "home_automation": self._home_automation_tests,
        }

        # Calculate total tests for progress
        total_tests = sum(len(test_func(model)) for test_func in categories.values())
        completed_tests = 0

        for category, test_func in categories.items():
            if cancel_event and cancel_event.is_set():
                break
            tests = test_func(model)
            category_results = []
            for i, test in enumerate(tests, 1):
                if cancel_event and cancel_event.is_set():
                    break
                print(f"[{category}] Running verification {i}/{len(tests)}... ", end="", flush=True)

                # Emit test_start progress
                if progress_callback:
                    try:
                        import inspect
                        if inspect.iscoroutinefunction(progress_callback):
                            await progress_callback(
                                "test_start",
                                {
                                    "model": model,
                                    "category": category,
                                    "test_id": test["id"],
                                    "test_label": test["label"],
                                },
                            )
                        else:
                            progress_callback(
                                "test_start",
                                {
                                    "model": model,
                                    "category": category,
                                    "test_id": test["id"],
                                    "test_label": test["label"],
                                },
                            )
                    except Exception as e:
                        print(f"Callback error: {e}")

                test_result = (
                    await self.test_model_proxy(model, test)
                    if use_proxy
                    else await self.test_model_direct(model, test)
                )

                if test_result["success"]:
                    actual_correct = self._verify_functional_response(
                        test["id"], test_result.get("response", "")
                    )
                    if not actual_correct:
                        test_result["success"] = False
                        test_result["error"] = "Failed correctness verification check"

                if test_result["success"]:
                    print("✓")
                else:
                    print(f"✗ ({test_result.get('error')})")

                test_result.update(
                    {"test_id": test["id"], "test_category": category, "test_label": test["label"]}
                )
                category_results.append(test_result)
                completed_tests += 1

                # Emit test_complete progress
                if progress_callback:
                    try:
                        import inspect
                        if inspect.iscoroutinefunction(progress_callback):
                            await progress_callback(
                                "test_complete",
                                {
                                    "model": model,
                                    "category": category,
                                    "test_id": test["id"],
                                    "test_label": test["label"],
                                    "result": test_result,
                                    "progress": {
                                        "completed": completed_tests,
                                        "total": total_tests,
                                        "percentage": round((completed_tests / total_tests) * 100),
                                    },
                                },
                            )
                        else:
                            progress_callback(
                                "test_complete",
                                {
                                    "model": model,
                                    "category": category,
                                    "test_id": test["id"],
                                    "test_label": test["label"],
                                    "result": test_result,
                                    "progress": {
                                        "completed": completed_tests,
                                        "total": total_tests,
                                        "percentage": round((completed_tests / total_tests) * 100),
                                    },
                                },
                            )
                    except Exception as e:
                        print(f"Callback error: {e}")

            results[f"category_{category}"] = self._calculate_category_stats(category_results)
        return results

    async def benchmark_model_performance(
        self, model: str, use_proxy: bool, progress_callback=None, cancel_event=None
    ) -> Dict:
        """Run performance suite measuring speed and peak footprint metrics."""
        print(f"\n--- Running Performance Suite for: {model} ---")

        load_tests = [
            {
                "id": "perf_medium",
                "prompt": "Write a detailed 10-paragraph essay explaining quantum mechanics and its impact on modern computing structures.",
                "num_predict": 800,
            },
            {
                "id": "perf_long",
                "prompt": "Generate a complete Python script implementing a web crawler that scans a local directory recursively, extracting links, saving metadata, and building a structured JSON index file.",
                "num_predict": 1000,
            },
        ]

        tps_list = []
        ttft_list = []
        sampler = self.ResourceSampler()
        total_tests = len(load_tests)
        completed_tests = 0

        print("Measuring footprint under active inference load...")
        for i, test in enumerate(load_tests, 1):
            if cancel_event and cancel_event.is_set():
                break

            # Emit test_start progress
            if progress_callback:
                try:
                    import inspect
                    if inspect.iscoroutinefunction(progress_callback):
                        await progress_callback(
                            "test_start",
                            {
                                "model": model,
                                "category": "performance",
                                "test_id": test["id"],
                                "test_label": test["id"],
                            },
                        )
                    else:
                        progress_callback(
                            "test_start",
                            {
                                "model": model,
                                "category": "performance",
                                "test_id": test["id"],
                                "test_label": test["id"],
                            },
                        )
                except Exception as e:
                    print(f"Callback error: {e}")

            print(f"  Executing Performance Load {i}/{len(load_tests)}... ", end="", flush=True)

            res = (
                await self.test_model_proxy(model, test, sampler)
                if use_proxy
                else await self.test_model_direct(model, test, sampler)
            )

            if res["success"]:
                tps = (
                    res.get("tokens_generated", 0) / self._extract_duration(res)
                    if self._extract_duration(res) > 0
                    else 0
                )
                tps_list.append(tps)

                if "prompt_eval_duration" in res:
                    ttft = res["prompt_eval_duration"] / 1e9 * 1000
                else:
                    ttft = res.get("latency", 0) * 1000
                ttft_list.append(ttft)
                print(f"✓ ({tps:.1f} tok/s, {ttft:.0f}ms TTFT)")
            else:
                print(f"✗ ({res.get('error')})")

            completed_tests += 1

            # Emit test_complete progress
            if progress_callback:
                try:
                    import inspect
                    if inspect.iscoroutinefunction(progress_callback):
                        await progress_callback(
                            "test_complete",
                            {
                                "model": model,
                                "category": "performance",
                                "test_id": test["id"],
                                "test_label": test["id"],
                                "result": res,
                                "progress": {
                                    "completed": completed_tests,
                                    "total": total_tests,
                                    "percentage": round((completed_tests / total_tests) * 100),
                                },
                            },
                        )
                    else:
                        progress_callback(
                            "test_complete",
                            {
                                "model": model,
                                "category": "performance",
                                "test_id": test["id"],
                                "test_label": test["id"],
                                "result": res,
                                "progress": {
                                    "completed": completed_tests,
                                    "total": total_tests,
                                    "percentage": round((completed_tests / total_tests) * 100),
                                },
                            },
                        )
                except Exception as e:
                    print(f"Callback error: {e}")

        avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0.0
        avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0.0

        return {
            "model": model,
            "performance_metrics": {
                "avg_tps": round(avg_tps, 2),
                "avg_ttft_ms": round(avg_ttft, 1),
                "peak_ram_pct": round(sampler.peak_ram_pct, 1),
                "peak_vram_mb": sampler.peak_vram_mb,
                "vram_total_mb": sampler.vram_total_mb,
            },
        }

    def _calculate_category_stats(self, results: List[Dict]) -> Dict:
        successful_tests = [r for r in results if r["success"]]
        if not successful_tests:
            return {
                "tests_run": len(results),
                "tests_passed": 0,
                "avg_tokens_per_sec": 0,
                "avg_ttft_ms": 0,
                "tests": results,
            }
        total_tokens = sum(r.get("tokens_generated", 0) for r in successful_tests)
        total_time = sum(self._extract_duration(r) for r in successful_tests)
        avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
        avg_ttft_ms = self._calculate_avg_ttft(successful_tests)
        return {
            "tests_run": len(results),
            "tests_passed": len(successful_tests),
            "avg_tokens_per_sec": round(avg_tokens_per_sec, 2),
            "avg_ttft_ms": round(avg_ttft_ms, 1),
            "tests": results,
        }

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

    async def run_model_benchmarks(
        self,
        models: List[str],
        use_proxy: bool,
        progress_callback=None,
        cancel_event=None,
        mode: str = "all",
    ) -> Dict:
        """Run split model benchmarks based on mode: 'functional', 'performance', or 'all'."""
        print("=" * 80)
        print("COMPREHENSIVE LLM MODEL BENCHMARKING SUITE")
        print(f"Running via: {'Proxy' if use_proxy else 'Direct'} | Mode: {mode.upper()}")
        print(f"Models: {models}")
        print("=" * 80)

        all_results: Dict[str, Any] = {
            "benchmark_version": "3.0.0",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "benchmark_type": "proxy" if use_proxy else "direct",
            "benchmark_mode": mode,
            "models_tested": len(models),
            "results": [],
        }

        # Emit benchmark_start event
        if progress_callback:
            try:
                import inspect
                start_data = {
                    "models": models,
                    "use_proxy": use_proxy,
                    "total_tests": len(models) * (10 if mode in ("functional", "all") else 0) + len(models) * (2 if mode in ("performance", "all") else 0),
                    "timestamp": all_results["generated_at"],
                }
                if inspect.iscoroutinefunction(progress_callback):
                    await progress_callback("benchmark_start", start_data)
                else:
                    progress_callback("benchmark_start", start_data)
            except Exception as e:
                print(f"Callback error: {e}")

        for model in models:
            if cancel_event and cancel_event.is_set():
                break

            model_data = {"model": model}

            # Emit model_start event
            if progress_callback:
                try:
                    import inspect
                    if inspect.iscoroutinefunction(progress_callback):
                        await progress_callback("model_start", {"model": model})
                    else:
                        progress_callback("model_start", {"model": model})
                except Exception as e:
                    print(f"Callback error: {e}")

            if mode in ("functional", "all"):
                func_data = await self.benchmark_model_functional(
                    model, use_proxy, progress_callback, cancel_event
                )
                model_data.update(func_data)

            if mode in ("performance", "all"):
                perf_data = await self.benchmark_model_performance(
                    model, use_proxy, progress_callback, cancel_event
                )
                model_data.update(perf_data)

            all_results["results"].append(model_data)

            # Emit model_complete event
            if progress_callback:
                try:
                    import inspect
                    if inspect.iscoroutinefunction(progress_callback):
                        await progress_callback("model_complete", {"model": model, "results": model_data})
                    else:
                        progress_callback("model_complete", {"model": model, "results": model_data})
                except Exception as e:
                    print(f"Callback error: {e}")

        # Emit benchmark_complete event
        if progress_callback:
            try:
                import inspect
                complete_data = {"status": "completed", "saved_as": all_results.get("saved_as")}
                if inspect.iscoroutinefunction(progress_callback):
                    await progress_callback("benchmark_complete", complete_data)
                else:
                    progress_callback("benchmark_complete", complete_data)
            except Exception as e:
                print(f"Callback error: {e}")

        if all_results["results"]:
            save_file = (
                self.RESULTS_DIR
                / f"benchmarks_{time.strftime('%Y%m%d_%H%M%S')}_{mode}_{'proxy' if use_proxy else 'direct'}.json"
            )
            with open(save_file, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

            latest_file = self.RESULTS_DIR / f"{mode}_benchmarks_latest.json"
            with open(latest_file, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

            print(f"\n{'=' * 80}")
            print("BENCHMARKING COMPLETE!")
            print(f"Results saved to: {save_file}")
            print(f"{'=' * 80}")
            all_results["saved_as"] = str(save_file)

        return all_results

    def run_optimization_pipeline(self, models: List[str]):
        print("=" * 80)
        print("LLM MODEL OPTIMIZATION PIPELINE")
        print(f"Testing models: {models}")
        print("=" * 80)

        direct_results = asyncio.run(self.run_model_benchmarks(models, use_proxy=False, mode="all"))
        proxy_results = asyncio.run(self.run_model_benchmarks(models, use_proxy=True, mode="all"))

        print("\n" + "=" * 80)
        print("LIVE RESULTS DISPLAY")
        print("=" * 80)
        self._display_live_results(direct_results)
        self._display_live_results(proxy_results)
        return {"direct_results": direct_results, "proxy_results": proxy_results}


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Container LLM Benchmarking Suite")
    parser.add_argument("models", nargs="?", help="Comma-separated list of models to benchmark")
    parser.add_argument(
        "--mode",
        choices=["functional", "performance", "all"],
        default="all",
        help="Phase mode: functional verification, performance testing, or both.",
    )
    parser.add_argument(
        "--type", choices=["proxy", "direct", "both"], default="both", help="Endpoint target type."
    )

    args = parser.parse_args()

    suite = LLMModelBenchmark()

    models = []
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        print("\n🔍 Discovering available models...")
        models = await suite.discover_all_models()

    if not models:
        models = suite._get_fallback_models()

    print(f"\nModels to benchmark: {models}")
    print(f"Mode: {args.mode.upper()}")

    if args.type == "proxy":
        await suite.run_model_benchmarks(models, use_proxy=True, mode=args.mode)
    elif args.type == "direct":
        await suite.run_model_benchmarks(models, use_proxy=False, mode=args.mode)
    else:
        await suite.run_model_benchmarks(models, use_proxy=True, mode=args.mode)
        await suite.run_model_benchmarks(models, use_proxy=False, mode=args.mode)

    print("\n" + "=" * 80)
    print("BENCHMARKING COMPLETE")
    print("Check data/llm_benchmarks/ for detailed results")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) == 1 and sys.stdin.isatty():
        print("1. Run Functional Correctness Suite")
        print("2. Run Performance Footprint Suite")
        print("3. Run Both Suites")
        choice = input("Enter choice (1/2/3): ").strip()
        mode_map = {"1": "functional", "2": "performance", "3": "all"}
        mode = mode_map.get(choice, "all")

        asyncio.run(
            LLMModelBenchmark().run_model_benchmarks(
                models=LLMModelBenchmark()._get_fallback_models(), use_proxy=True, mode=mode
            )
        )
    else:
        asyncio.run(main())
