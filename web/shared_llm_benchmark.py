#!/usr/bin/env python3
"""
SharedLLM validation and benchmarking suite.
Evaluates LLM models based on Jarvis / SharedLLM task patterns:
1. FastPath classification (Simple HA command intent)
2. Tool Use (Microservice parameter matching)
3. Code Gen (Raven autonomous coding lock task with AST structural validation)
"""

import ast
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, List

import httpx


class SharedLLMModelBenchmark:
    """
    Validation benchmark harness based on SharedLLM tasks and AST code parsing.
    """

    def __init__(self):
        # Read endpoints from environment or use proxy default
        ollama_env = os.getenv("OLLAMA_SERVER_URLS", "")
        if ollama_env:
            self.OLLAMA_SERVER_URLS = [u.strip() for u in ollama_env.split(",") if u.strip()]
        else:
            self.OLLAMA_SERVER_URLS = ["http://localhost:8080", "http://llama-server:8080"]

        proxy_env = os.getenv("PROXY_SERVER_URLS", "")
        if proxy_env:
            self.PROXY_SERVER_URLS = [u.strip() for u in proxy_env.split(",") if u.strip()]
        else:
            self.PROXY_SERVER_URLS = ["http://localhost:11434", "http://alpaca-proxy:11434"]

        self.RESULTS_DIR = Path("data/shared_llm_benchmarks")
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def strip_thinking(self, text: str) -> str:
        """Remove <think>...</think> blocks from response — thinking models may still emit them."""
        import re

        # Remove <think>...</think> including multiline
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return cleaned.strip()

    def validate_code(self, code: str) -> Dict:
        """Parse generated code via AST to verify structural correctness."""
        try:
            # Strip thinking blocks before parsing
            code = self.strip_thinking(code)
            # Clean markdown code blocks
            clean_code = code
            if "```python" in code:
                clean_code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                clean_code = code.split("```")[1].split("```")[0].strip()

            tree = ast.parse(clean_code)

            has_class = False
            has_acquire = False
            has_release = False

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == "MultiTenantLock":
                    has_class = True
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            if item.name == "acquire":
                                has_acquire = True
                            if item.name == "release":
                                has_release = True

            is_complete = has_class and has_acquire and has_release
            return {
                "valid_syntax": True,
                "has_class": has_class,
                "has_acquire": has_acquire,
                "has_release": has_release,
                "is_complete": is_complete,
                "error": None,
            }
        except Exception as e:
            return {
                "valid_syntax": False,
                "has_class": False,
                "has_acquire": False,
                "has_release": False,
                "is_complete": False,
                "error": str(e),
            }

    async def query_model(
        self, model: str, use_proxy: bool, prompt: str, max_tokens: int = 250
    ) -> Dict:
        """Execute request against proxy or direct llama-server.

        Thinking mode is explicitly disabled (think=false) so that thinking models
        (e.g. Qwen3.6 MoE) don't consume the entire token budget on internal reasoning,
        which would produce an empty content field in the response.
        """
        urls = self.PROXY_SERVER_URLS if use_proxy else self.OLLAMA_SERVER_URLS
        last_error = None

        for base_url in urls:
            try:
                start_t = time.time()
                async with httpx.AsyncClient(timeout=180.0) as client:
                    if use_proxy:
                        resp = await client.post(
                            f"{base_url}/api/chat",
                            json={
                                "model": model,
                                "messages": [{"role": "user", "content": prompt}],
                                "stream": False,
                                # Disable thinking mode so MoE / thinking models don't
                                # spend all tokens on <think> and return empty content.
                                "think": False,
                                "options": {"num_predict": max_tokens, "temperature": 0.2},
                            },
                            headers={"X-Request-Source": "shared-llm/benchmark"},
                            timeout=180.0,
                        )
                        latency = time.time() - start_t
                        if resp.status_code == 200:
                            data = resp.json()
                            content = self.strip_thinking(
                                data.get("message", {}).get("content", "") or ""
                            )
                            eval_cnt = data.get("eval_count", 0)
                            return {
                                "success": True,
                                "latency": latency,
                                "response": content,
                                "tokens_generated": eval_cnt,
                                "error": None,
                            }
                    else:
                        resp = await client.post(
                            f"{base_url}/api/generate",
                            json={
                                "model": model,
                                "prompt": prompt,
                                "stream": False,
                                "think": False,
                                "options": {"num_predict": max_tokens, "temperature": 0.2},
                            },
                            timeout=180.0,
                        )
                        latency = time.time() - start_t
                        if resp.status_code == 200:
                            data = resp.json()
                            content = self.strip_thinking(data.get("response", "") or "")
                            eval_cnt = data.get("eval_count", 0)
                            return {
                                "success": True,
                                "latency": latency,
                                "response": content,
                                "tokens_generated": eval_cnt,
                                "error": None,
                            }

                    last_error = f"HTTP {resp.status_code}: {resp.text}"
            except Exception as e:
                last_error = str(e)
                continue

        return {
            "success": False,
            "latency": 0.0,
            "response": None,
            "tokens_generated": 0,
            "error": last_error or "Endpoint unavailable",
        }

    async def run_shared_llm_benchmarks(
        self,
        models: List[str],
        use_proxy: bool,
        progress_callback: Callable = None,
        cancel_event=None,
    ) -> Dict:
        """Run tasks for FastPath, Tool Use, and Code Gen validation."""
        all_results = {
            "benchmark_version": "SharedLLM-v1",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "benchmark_type": "proxy" if use_proxy else "direct",
            "models_tested": len(models),
            "results": [],
        }

        # /no_think prefix: Qwen3-family models respect this at the tokenizer level,
        # instructing them to skip the internal reasoning phase and output directly.
        # Other models safely ignore it. Token budgets are set high enough that
        # thinking models can finish reasoning AND produce content.
        TASKS = [
            {
                "id": "fast_path",
                "category": "FastPath Intent",
                "label": "HA Intent: turn on lights",
                "prompt": "/no_think You are a smart home intent classifier. Classify this request: 'turn on the office lights'. Options: [light_on, light_off, thermostat_temp, check_status, conversational]. Respond with EXACTLY one of the options, nothing else.",
                "max_tokens": 100,
            },
            {
                "id": "tool_use",
                "category": "Librarian Tools",
                "label": "Tool Selection: list files",
                "prompt": '/no_think Determine which tool and parameter to call for: \'list my files on nextcloud\'. Tools available: [ha_light_control, nextcloud_list_files(directory=\'/\'), nextcloud_download_file(file_path)]. Output only valid JSON like: {"tool": "name", "args": {}}',
                "max_tokens": 150,
            },
            {
                "id": "code_gen",
                "category": "Raven Code Gen",
                "label": "Raven: Redis scoping lock",
                "prompt": "/no_think Write a Python class 'MultiTenantLock' that uses Redis (redis-py) to implement a distributed lock. The lock MUST be scoped to a 'user_id' and a 'resource_id'. It MUST have 'acquire' and 'release' methods. Ensure it handles timeouts and uses a TTL to prevent deadlocks. Provide ONLY the Python code, no explanation, no markdown preamble outside of backticks.",
                "max_tokens": 600,
            },
        ]

        total_tests = len(models) * len(TASKS)
        if progress_callback:
            try:
                import inspect

                start_data = {
                    "models": models,
                    "use_proxy": use_proxy,
                    "total_models": len(models),
                    "total_tests": total_tests,
                    "timestamp": all_results["generated_at"],
                }
                if inspect.iscoroutinefunction(progress_callback):
                    await progress_callback("benchmark_start", start_data)
                else:
                    progress_callback("benchmark_start", start_data)
            except Exception as e:
                print(f"Callback error: {e}")

        completed_count = 0
        for model in models:
            if cancel_event and cancel_event.is_set():
                break

            if progress_callback:
                try:
                    import inspect

                    if inspect.iscoroutinefunction(progress_callback):
                        await progress_callback("model_start", {"model": model})
                    else:
                        progress_callback("model_start", {"model": model})
                except Exception as e:
                    print(f"Callback error: {e}")

            model_record = {
                "model": model,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "tasks": [],
            }

            for task in TASKS:
                if cancel_event and cancel_event.is_set():
                    break

                if progress_callback:
                    try:
                        import inspect

                        if inspect.iscoroutinefunction(progress_callback):
                            await progress_callback(
                                "test_start",
                                {
                                    "model": model,
                                    "category": task["category"],
                                    "test_id": task["id"],
                                    "test_label": task["label"],
                                },
                            )
                        else:
                            progress_callback(
                                "test_start",
                                {
                                    "model": model,
                                    "category": task["category"],
                                    "test_id": task["id"],
                                    "test_label": task["label"],
                                },
                            )
                    except Exception as e:
                        print(f"Callback error: {e}")

                # Query endpoint
                res = await self.query_model(model, use_proxy, task["prompt"], task["max_tokens"])

                # Custom evaluations for SharedLLM tiers
                validation_results = {}
                if res["success"]:
                    response_text = self.strip_thinking(res["response"] or "").strip()
                    if task["id"] == "fast_path":
                        passed = "light_on" in response_text.lower()
                        validation_results = {
                            "correct_intent": passed,
                            "expected": "light_on",
                            "actual": response_text,
                        }
                        res["success"] = passed
                    elif task["id"] == "tool_use":
                        try:
                            # Clean json markup
                            json_str = response_text
                            if "```json" in response_text:
                                json_str = response_text.split("```json")[1].split("```")[0].strip()
                            elif "```" in response_text:
                                json_str = response_text.split("```")[1].split("```")[0].strip()

                            parsed = json.loads(json_str)
                            passed = "nextcloud" in parsed.get("tool", "").lower()
                            validation_results = {
                                "valid_json": True,
                                "tool_match": passed,
                                "parsed": parsed,
                            }
                            res["success"] = passed
                        except Exception as je:
                            validation_results = {
                                "valid_json": False,
                                "error": str(je),
                                "actual": response_text,
                            }
                            res["success"] = False
                    elif task["id"] == "code_gen":
                        validation_results = self.validate_code(response_text)
                        res["success"] = validation_results["is_complete"]

                test_result = {
                    "test_id": task["id"],
                    "test_category": task["category"],
                    "test_label": task["label"],
                    "success": res["success"],
                    "latency": res["latency"],
                    "tokens_generated": res["tokens_generated"],
                    "prompt": task["prompt"],
                    "response": res["response"],
                    "error": res["error"],
                    "validation": validation_results,
                }

                model_record["tasks"].append(test_result)
                completed_count += 1

                if progress_callback:
                    try:
                        import inspect

                        if inspect.iscoroutinefunction(progress_callback):
                            await progress_callback(
                                "test_complete",
                                {
                                    "model": model,
                                    "category": task["category"],
                                    "test_id": task["id"],
                                    "test_label": task["label"],
                                    "result": test_result,
                                    "progress": {
                                        "completed": completed_count,
                                        "total": total_tests,
                                        "percentage": round((completed_count / total_tests) * 100),
                                    },
                                },
                            )
                        else:
                            progress_callback(
                                "test_complete",
                                {
                                    "model": model,
                                    "category": task["category"],
                                    "test_id": task["id"],
                                    "test_label": task["label"],
                                    "result": test_result,
                                    "progress": {
                                        "completed": completed_count,
                                        "total": total_tests,
                                        "percentage": round((completed_count / total_tests) * 100),
                                    },
                                },
                            )
                    except Exception as e:
                        print(f"Callback error: {e}")

            all_results["results"].append(model_record)
            if progress_callback:
                try:
                    import inspect

                    if inspect.iscoroutinefunction(progress_callback):
                        await progress_callback(
                            "model_complete", {"model": model, "results": model_record}
                        )
                    else:
                        progress_callback(
                            "model_complete", {"model": model, "results": model_record}
                        )
                except Exception as e:
                    print(f"Callback error: {e}")

        if cancel_event and cancel_event.is_set():
            all_results["status"] = "cancelled"
        else:
            all_results["status"] = "completed"

        if all_results["results"]:
            save_file = (
                self.RESULTS_DIR
                / f"shared_llm_benchmarks_{time.strftime('%Y%m%d_%H%M%S')}_{'proxy' if use_proxy else 'direct'}.json"
            )
            with open(save_file, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            all_results["saved_as"] = str(save_file)

        if progress_callback:
            try:
                import inspect

                if inspect.iscoroutinefunction(progress_callback):
                    await progress_callback("benchmark_complete", all_results)
                else:
                    progress_callback("benchmark_complete", all_results)
            except Exception as e:
                print(f"Callback error: {e}")

        return all_results
