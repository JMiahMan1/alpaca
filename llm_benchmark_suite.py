#!/usr/bin/env python3
"""
llm_benchmark_suite.py

Centralized benchmarking suite for evaluating LLM models across SharedLLM task categories.
Divided into two distinct execution phases:
1. Functional Benchmarks: Evaluates accuracy and output quality (pass/fail correctness).
2. Performance Benchmarks: Evaluates hardware footprint (RAM/VRAM), TTFT, and TPS under load.
"""

import ast
import asyncio
import configparser
import hashlib
import json
import logging
import os
import re
import sys
import time
import unicodedata
import wave
from pathlib import Path
from typing import Any, ClassVar

import httpx
import psutil

from context_awareness import estimate_prompt_tokens, resolve_context_window, turn_budget
from online_providers import online_model_provider

# One-shot code execution for grading coding benchmarks (runs Python/Node in the
# locked-down alpaca-sandbox container and returns ran/output/exit_code).
from sandbox_exec import extract_clean_code, grade_code

# Proactive delay between requests when benchmarking online providers. Free-tier
# endpoints (e.g. OpenRouter) rate-limit aggressively and answer throttled requests
# with empty 200 completions, so pacing requests avoids throttling the run into
# spurious false negatives. Local/proxy models are not throttled, so this is skipped.
ONLINE_BENCHMARK_THROTTLE_S = 3.0


# Per-model profiles: benchmarks derive per-model settings (temperature,
# reasoning budget, thinking toggle, sampling knobs) from models.ini and the
# companion {alias}.profile.json overlays next to the router symlinks. The
# overlay file wins over the ini section for the same key, mirroring the
# web dashboard's merge order. Nothing here is test-specific: a per-model
# setting applies to every benchmark test run for that model, so a model
# profile fully controls how long the model is allowed to think.
def _models_ini_candidates() -> list[Path]:
    env = os.getenv("MODELS_INI_PATH", "").strip()
    if env:
        return [Path(env)]
    return [
        Path(__file__).parent / ".alpaca-router" / "models.ini",
        Path(".alpaca-router/models.ini"),
        # Standard in-container location: compose mounts .alpaca-router at
        # /router-models (ROUTER_MODELS_DIR), so models.ini lands there.
        Path("/router-models/models.ini"),
    ]


def _model_section_names(model: str) -> list[str]:
    """Router aliases carry the --latest suffix (e.g. ornith-1-5-9b-q4-k-m--latest)
    while callers may pass either form, so try both section spellings."""
    if model.endswith("--latest"):
        return [model, model[: -len("--latest")]]
    return [model, f"{model}--latest"]


def _profile_overlay(ini_dir: Path, model: str) -> dict:
    """Merge companion ``{alias}.profile.json`` overlays for a model.

    Files are named after the router symlink stem, so try both the alias as
    passed and the same alias without the ``--latest`` suffix. Overlays win
    over equally-keyed ini values (same merge order the dashboard uses).
    """
    merged: dict = {}
    for alias in {model, model[: -len("--latest")] if model.endswith("--latest") else model}:
        for ext in (".profile.json", ".gguf.profile.json", ".safetensors.profile.json"):
            candidate = ini_dir / f"{alias}{ext}"
            try:
                if candidate.exists():
                    data = json.loads(candidate.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        merged.update(data)
            except Exception:
                continue
    return merged


def _model_profile_raw(model: str) -> dict:
    """Merged per-model settings dict: models.ini section + profile overlay.

    Resolution order per key: ``[model]`` / ``[model--latest]`` section, then
    the ``{alias}.profile.json`` companion overlay, then the global defaults
    (``[*]`` then ``[DEFAULT]``).
    """
    profile: dict = {}
    ini_dir: Path | None = None
    for ini in _models_ini_candidates():
        try:
            if not ini.exists():
                continue
            if ini_dir is None:
                ini_dir = ini.parent
            cp = configparser.ConfigParser(delimiters=("=",))
            cp.read(ini)
            for name in _model_section_names(model):
                if cp.has_section(name):
                    profile.update(dict(cp[name]))
            if cp.has_section("*"):
                for k, v in cp["*"].items():
                    profile.setdefault(k, v)
            if cp.has_section("DEFAULT"):
                for k, v in cp["DEFAULT"].items():
                    profile.setdefault(k, v)
        except Exception:
            continue
    if ini_dir is not None:
        profile.update(_profile_overlay(ini_dir, model))
    return profile


def _model_profile_get(model: str, key: str) -> str | None:
    value = _model_profile_raw(model).get(key)
    return value.strip() if isinstance(value, str) and value else None


def _model_profile_int(model: str, key: str) -> int | None:
    value = _model_profile_get(model, key)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _model_profile_bool(model: str, key: str) -> bool | None:
    value = _model_profile_get(model, key)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in ("on", "true", "1", "yes"):
        return True
    if normalized in ("off", "false", "0", "no", "auto"):
        return False
    return None


def _model_temperature(model: str) -> float:
    """Timeless per-model temperature: required, no fallback. Must be set in
    [*] or a per-model section, otherwise misconfig is loud (as before)."""
    profile = _model_profile_raw(model)
    temp = profile.get("temperature")
    if temp is not None:
        try:
            return float(temp)
        except ValueError:
            raise ValueError(f"invalid temperature '{temp}' for model '{model}' in models.ini") from None
    raise ValueError(
        f"temperature not set for model '{model}' (and no [*] default) in models.ini - set [*] temperature or per-model temperature via Settings > UI or .alpaca-router/models.ini"
    )


def _model_reasoning_budget(model: str) -> int | None:
    """Per-model reasoning budget (tokens) from the profile, or None when the
    profile defines no budget. Only the profile decides: benchmarks send no
    budget at all when this returns None."""
    budget = _model_profile_int(model, "reasoning-budget")
    if budget is None:
        budget = _model_profile_int(model, "reasoning_budget")
    return budget


def _model_thinking(model: str) -> bool | None:
    """Explicit per-model thinking toggle (thinking/think = on/off)."""
    toggle = _model_profile_bool(model, "thinking")
    if toggle is None:
        toggle = _model_profile_bool(model, "think")
    return toggle


# Profile sampling knobs mapped to llama.cpp request option names (snake_case
# as used by /api/chat options and /api/generate). The profile stores kebab
# keys like the rest of models.ini (ctx-size, n-gpu-layers).
_SAMPLING_KEYMAP: dict[str, tuple[str, type]] = {
    "top-k": ("top_k", int),
    "top-p": ("top_p", float),
    "min-p": ("min_p", float),
    "typical-p": ("typical_p", float),
    "repeat-last-n": ("repeat_last_n", int),
    "repeat-penalty": ("repeat_penalty", float),
    "presence-penalty": ("presence_penalty", float),
    "frequency-penalty": ("frequency_penalty", float),
    "seed": ("seed", int),
}


def _model_sampling_options(model: str) -> dict:
    """Per-model sampling overrides from the profile (option-knob -> value).

    Only keys explicitly defined in the model profile are returned; undefined
    knobs keep llama-server defaults. Temperature is covered separately and
    required, so it is not smuggled in here.
    """
    profile = _model_profile_raw(model)
    options: dict[str, Any] = {}
    for ini_key, (param, typ) in _SAMPLING_KEYMAP.items():
        raw = profile.get(ini_key)
        if raw in (None, ""):
            continue
        try:
            options[param] = typ(float(raw))
        except (TypeError, ValueError):
            continue
    return options


def _ascii_fold(text: str) -> str:
    """Fold a string to ASCII so accented answers match ascii expectations.

    e.g. "Gödel" -> "godel", so an `expected: "godel"` substring check passes
    when the model answers with the proper diacritic.
    """
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


# Per-chunk stream timeouts: read applies to the gap BETWEEN stream lines, not
# total request time. Large prompts (15k+ tokens) plus slow generation on big
# MoE models can run for many minutes; a hard non-streaming deadline kills
# healthy requests mid-generation (llama-server logs "Connection handling
# canceled" and the benchmark scores an empty result).
_STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0)


async def _read_chat_stream(resp: httpx.Response) -> dict:
    """Accumulate an Ollama-style NDJSON /api/chat stream into text + final metrics."""
    parts: list[str] = []
    think_parts: list[str] = []
    final: dict[str, Any] = {}
    async for line in resp.aiter_lines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        msg = obj.get("message") or {}
        chunk = msg.get("content") or ""
        if chunk:
            parts.append(chunk)
        think_chunk = msg.get("thinking") or ""
        if think_chunk:
            think_parts.append(think_chunk)
        if obj.get("done"):
            final = obj
    return {
        "content": "".join(parts),
        "thinking": "".join(think_parts),
        "eval_count": int(final.get("eval_count") or 0),
        "eval_duration": int(final.get("eval_duration") or 0),
        "prompt_eval_count": int(final.get("prompt_eval_count") or 0),
        "prompt_eval_duration": int(final.get("prompt_eval_duration") or 0),
    }


async def _read_generate_stream(resp: httpx.Response) -> dict:
    """Accumulate an Ollama-style NDJSON /api/generate stream into text + final metrics."""
    parts: list[str] = []
    think_parts: list[str] = []
    final: dict[str, Any] = {}
    async for line in resp.aiter_lines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        chunk = obj.get("response") or ""
        if chunk:
            parts.append(chunk)
        think_chunk = obj.get("thinking") or ""
        if think_chunk:
            think_parts.append(think_chunk)
        if obj.get("done"):
            final = obj
    return {
        "content": "".join(parts),
        "thinking": "".join(think_parts),
        "eval_count": int(final.get("eval_count") or 0),
        "eval_duration": int(final.get("eval_duration") or 0),
        "prompt_eval_count": int(final.get("prompt_eval_count") or 0),
        "prompt_eval_duration": int(final.get("prompt_eval_duration") or 0),
    }


# Reasoning budget resolution (per-model driven):
# 1. Model profile "reasoning-budget" (models.ini section or {alias}.profile.json)
#    - the authoritative per-model setting.
# 2. Benchmark-wide default "reasoning-budget" in the [benchmark] ini section
#    (settable once, applies to every model without its own budget).
# 3. No budget at all: the benchmark sends no reasoning_budget and disables
#    the thinking phase. Per-test "reasoning_budget" values no longer toggle
#    thinking - the model itself decides how much it thinks, per its profile.
# Each run records the effective "think"/"reasoning_budget" per result so runs
# stay comparable and inspectable.
REASONING_HEAVY_BUDGET = 2048


def _benchmark_default(key: str) -> str | None:
    """Read a benchmark-wide default from the ``[benchmark]`` ini section.

    Env overrides (BENCHMARK_REASONING_BUDGET / BENCHMARK_THINKING) win over
    the ini so CI and ad-hoc runs can override without editing persistent
    config. Returns None when unset.
    """
    env_map = {
        "reasoning-budget": ("BENCHMARK_REASONING_BUDGET", None),
        "thinking": ("BENCHMARK_THINKING", None),
        "think": ("BENCHMARK_THINKING", None),
    }
    env_name, _ = env_map.get(key, (None, None))
    if env_name:
        env_val = os.getenv(env_name, "").strip()
        if env_val:
            return env_val
    for ini in _models_ini_candidates():
        try:
            if not ini.exists():
                continue
            cp = configparser.ConfigParser(delimiters=("=",))
            cp.read(ini)
            for section in ("benchmark", "benchmarks"):
                if cp.has_section(section) and key in cp[section]:
                    value = cp[section][key]
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        except Exception:
            continue
    return None


def _benchmark_reasoning_budget() -> int | None:
    """Settable benchmark-wide reasoning budget (tokens), or None for 'no budget'."""
    value = _benchmark_default("reasoning-budget") or _benchmark_default("reasoning_budget")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _benchmark_thinking() -> bool | None:
    """Settable benchmark-wide thinking toggle, or None for 'let the budget decide'."""
    value = _benchmark_default("thinking") or _benchmark_default("think")
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in ("on", "true", "1", "yes"):
        return True
    if normalized in ("off", "false", "0", "no", "auto"):
        return False
    return None

# Task-shape tiers for per-benchmark reasoning estimates (tokens a competent
# model needs for its thinking phase on that test). benchmark_tests.json ships
# explicit per-test "reasoning_estimate" values generated by
# scripts/gen_reasoning_estimates.py from these same rules; the fallbacks here
# keep ad-hoc test dicts working.
_EST_UI_GAME_CATEGORIES = {"gamedev", "gamedev_alt", "retrogames", "youtuber"}
_EST_CODE_CATEGORIES = {
    "coding",
    "appdev",
    "webdev",
    "linux_driver",
    "iac",
    "android",
    "typescript",
    "rpm",
    "usb",
    "networking",
    "bash",
    "basic",
    "pascal",
}
_EST_DELIBERATE_CATEGORIES = {
    "gpqa_diamond",
    "hle",
    "math_hard",
    "mmlu_pro",
    "logic",
    "reasoning",
    "metacog",
    "agentic",
    "code_review",
}


def _test_reasoning_estimate(test: dict) -> int:
    """Calculated reasoning-token need for this specific benchmark.

    An explicit per-test "reasoning_estimate" field always wins; otherwise the
    task shape decides: UI games reason about whole program architectures,
    runnable-program categories plan code and edge cases, exam-style categories
    deliberate before answering, everything else reasons briefly if at all.
    """
    explicit = int(test.get("reasoning_estimate") or 0)
    if explicit:
        return explicit
    category = str(test.get("category") or "")
    if str(test.get("type") or "") == "ui" or category in _EST_UI_GAME_CATEGORIES:
        tier = 4096
    elif category in _EST_CODE_CATEGORIES:
        tier = 3072
    elif category in _EST_DELIBERATE_CATEGORIES:
        tier = 2048
    else:
        tier = 1024
    # A test flagged heavy-reasoning should never estimate below its threshold.
    return max(tier, int(test.get("reasoning_budget") or 0))


def _effective_reasoning_budget(model: str | None) -> int | None:
    """Effective reasoning budget for a model: per-model profile wins, then the
    settable benchmark-wide default, then None = no budget (thinking off)."""
    if model is not None:
        budget = _model_reasoning_budget(model)
        if budget is not None:
            return budget
    return _benchmark_reasoning_budget()


def _effective_thinking(model: str | None) -> bool:
    """Whether the thinking phase is enabled for a model run.

    Explicit ``thinking``/``think`` toggle in the model profile wins; then the
    benchmark-wide default; otherwise thinking is enabled only when an
    effective reasoning budget is set (no budget = no thinking).
    """
    if model is not None:
        toggle = _model_thinking(model)
        if toggle is not None:
            return toggle
    toggle = _benchmark_thinking()
    if toggle is not None:
        return toggle
    budget = _effective_reasoning_budget(model)
    return budget is not None and budget > 0


def _test_reasoning_budget(test: dict, model: str | None = None) -> int | None:
    """Effective reasoning budget for (test, model). Per-model/per-benchmark
    resolution only: legacy per-test ``reasoning_budget`` values are NOT used
    to enable thinking anymore - model profiles decide."""
    return _effective_reasoning_budget(model)


def _test_thinking(test: dict, model: str | None = None) -> bool:
    """Whether this (test, model) run has the thinking phase enabled."""
    return _effective_thinking(model)


def _test_num_predict(test: dict, model: str | None = None) -> int:
    """Per-test token cap.

    When the thinking phase is enabled (model profile or benchmark default
    budget), the cap gets DOUBLED per-test reasoning headroom (base + 2 x
    estimate) so the thinking phase plus the complete answer fit without
    truncation - the failure mode that stranded reasoning models on large UI
    tests with empty, length-capped responses. Without a budget the cap is the
    plain test base.
    """
    base = int(test.get("num_predict", 4000))
    if not _test_thinking(test, model):
        return base
    return base + 2 * _test_reasoning_estimate(test)


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
        self.MODELS_DIR = self.RESULTS_DIR / "models"
        self.ARTIFACTS_DIR = Path("data/artifacts")
        self.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.tests_config = self._load_tests_config()
        self._last_online_request: dict[str, float] = {}
        self._rate_limit_retry_floor = 30.0
        # Resolved backend context windows (model -> n_ctx), cached per instance.
        self._ctx_cache: dict[str, int] = {}

    async def _effective_num_predict(self, model: str, prompt: str, requested: int) -> int:
        """Clamp a generation budget into the live backend context window.

        Raises RuntimeError (propagates to the caller's error handling) when
        the context window cannot be determined — guessing would reintroduce
        the silent-truncation bug this guard exists to prevent.
        """
        ctx = await resolve_context_window(
            model, self.PROXY_SERVER_URLS, self._ctx_cache, "general/benchmark"
        )
        return turn_budget(ctx, estimate_prompt_tokens([{"role": "user", "content": prompt}]), requested)

    @staticmethod
    def _proxy_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = dict(extra or {})
        key = os.getenv("ALPACA_API_KEY", "").strip()
        if key:
            headers.setdefault("Authorization", f"Bearer {key}")
            headers.setdefault("X-API-Key", key)
        return headers

    @staticmethod
    def _sanitize_model_filename(model: str) -> str:
        return re.sub(r"[/:.]", "_", model)

    @staticmethod
    def compute_test_hash(test_dict: dict) -> str:
        """Compute a stable content hash for a benchmark test definition.

        Used to detect when a test prompt, expected value, attachments, or code
        directive has changed since a model was benchmarked.
        """
        if not isinstance(test_dict, dict):
            return ""
        atts = [a.get("name", "") for a in test_dict.get("attachments", []) if isinstance(a, dict)]
        canonical = {
            "id": test_dict.get("id", ""),
            "prompt": (test_dict.get("prompt") or "").strip(),
            "expected": str(test_dict.get("expected") or "").strip(),
            "expected_output": str(test_dict.get("expected_output") or "").strip(),
            "type": test_dict.get("type", "functional"),
            "kind": test_dict.get("kind", "text"),
            "attachments": sorted(atts),
        }
        dumped = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _is_rate_limited_result(test_result: dict) -> bool:
        """True when a test result failed because of a provider rate limit (HTTP 429)."""
        err = (test_result.get("error") or "").lower()
        return "429" in err or "rate limit" in err

    # Minimum wall-clock seconds to wait between requests to a given online
    # provider. Free tiers throttle hard (Gemini free = 20 req/min), and a
    # burst of fast benchmark calls exhausts the window, cascading into 429s.
    _ONLINE_PACE_SECONDS: ClassVar[dict[str, float]] = {
        "gemini": 3.5,  # 20 req/min free tier -> one request every 3s
        "openrouter": 1.2,  # generous free tier; keep a little headroom
        "opencode_zen": 1.2,
        "huggingface": 1.2,
        "cloudflare": 1.2,
        "groq": 1.2,
    }

    def _pace_online_request(self, model: str) -> None:
        """Enforce a minimum interval between requests to the same online provider.

        Called before every online benchmark request. Sleeps until the
        per-provider pace window has elapsed since the last request so a fast
        local run does not blow through a free-tier quota window in seconds.
        """
        provider, _ = online_model_provider.parse_model_identifier(model)
        min_gap = self._ONLINE_PACE_SECONDS.get(provider, 0.0)
        if min_gap <= 0:
            return
        now = time.time()
        last = self._last_online_request.get(provider, 0.0)
        wait = min_gap - (now - last)
        if wait > 0:
            time.sleep(wait)
        self._last_online_request[provider] = time.time()

    @staticmethod
    def _merge_run_history(existing: dict | None, incoming: dict) -> dict:
        """Merge a fresh test result into a previously stored one, tracking run history.

        ``run_count`` counts how many times the model has run this test (total
        attempts), ``fail_count`` how many of those attempts did not pass. When the
        test has been seen before the counters carry over and increment; a brand-new
        test starts at 1 / (1 if it failed else 0). The incoming result (latest run)
        wins for all other fields.
        """
        merged = dict(incoming)
        prev_runs = 0
        prev_fails = 0
        if isinstance(existing, dict):
            prev_runs = int(existing.get("run_count") or 0)
            prev_fails = int(existing.get("fail_count") or 0)
        merged["run_count"] = prev_runs + 1
        merged["fail_count"] = prev_fails + (0 if incoming.get("success") else 1)
        return merged

    def save_per_model_result(
        self,
        model_data: dict,
        mode: str,
        use_proxy: bool,
        generated_at: str | None = None,
    ) -> Path | None:
        """Persist a single model's results to a per-model file so results follow the model.

        Merges by test_id rather than overwriting, so a partial re-run (e.g. only a
        handful of categories or a manual re-run of a few tests) never wipes results
        for categories that were not part of this invocation.
        """
        model = model_data.get("model")
        if not model:
            return None
        file_path = self.MODELS_DIR / f"general_{self._sanitize_model_filename(model)}.json"

        existing = None
        if file_path.exists():
            try:
                with open(file_path) as f:
                    existing = json.load(f)
            except Exception:
                existing = None

        if existing and isinstance(existing.get("results"), list) and existing["results"]:
            entry = existing["results"][0]
            for cat_key, cat_block in model_data.items():
                if not cat_key.startswith("category_") or not isinstance(cat_block, dict):
                    continue
                incoming = cat_block.get("tests", [])
                cur_tests = entry.get(cat_key, {}).get("tests", []) if isinstance(entry.get(cat_key), dict) else []
                by_id = {t.get("test_id"): t for t in cur_tests}
                for t in incoming:
                    by_id[t.get("test_id")] = self._merge_run_history(by_id.get(t.get("test_id")), t)
                merged = list(by_id.values())
                entry[cat_key] = self._calculate_category_stats(merged)
            entry["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            existing["generated_at"] = existing.get("generated_at") or generated_at
            tmp = file_path.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(existing, f, indent=2, default=str)
            os.replace(tmp, file_path)
            return file_path

        per_model = {
            "benchmark_version": "3.0.0",
            "generated_at": generated_at or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "benchmark_type": "proxy" if use_proxy else "direct",
            "benchmark_mode": mode,
            "models_tested": 1,
            "per_model": True,
            "model": model,
            "results": [model_data],
        }
        with open(file_path, "w") as f:
            json.dump(per_model, f, indent=2, default=str)
        return file_path

    def save_test_result_incremental(
        self,
        model: str,
        category: str,
        test_result: dict,
        mode: str,
        use_proxy: bool,
        generated_at: str | None = None,
    ) -> Path | None:
        """Write a single test result to the per-model file immediately after it completes.

        This is the source of truth for resumability: even if the whole run is killed
        (OOM, cancelled, crash) the already-completed tests are persisted per-test and a
        subsequent run with `resume=True` will skip them.
        """
        if not model:
            return None
        file_path = self.MODELS_DIR / f"general_{self._sanitize_model_filename(model)}.json"
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        per_model = None
        if file_path.exists():
            try:
                with open(file_path) as f:
                    per_model = json.load(f)
            except Exception:
                per_model = None
        if not per_model or not isinstance(per_model.get("results"), list) or not per_model["results"]:
            per_model = {
                "benchmark_version": "3.0.0",
                "generated_at": generated_at or now,
                "benchmark_type": "proxy" if use_proxy else "direct",
                "benchmark_mode": mode,
                "models_tested": 1,
                "per_model": True,
                "model": model,
                "results": [{"model": model}],
            }
        entry = per_model["results"][0]
        entry.setdefault("model", model)
        cat_key = f"category_{category}"
        cat_block = entry.get(cat_key)
        tests = cat_block.get("tests", []) if isinstance(cat_block, dict) else []
        tid = test_result.get("test_id")
        replaced = False
        for idx, t in enumerate(tests):
            if t.get("test_id") == tid:
                tests[idx] = self._merge_run_history(t, test_result)
                replaced = True
                break
        if not replaced:
            tests.append(self._merge_run_history(None, test_result))
        entry[cat_key] = self._calculate_category_stats(tests)
        per_model["last_updated"] = now
        tmp = file_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(per_model, f, indent=2, default=str)
        os.replace(tmp, file_path)
        return file_path

    def delete_model_results(self, model: str) -> bool:
        """Remove a model's per-model result file and its saved artifacts. Returns True if anything was removed."""
        if not model:
            return False
        removed = False

        # Gather all plausible alias strings and sanitized variants
        clean_model = model.strip()
        variants = {
            clean_model,
            clean_model.replace("--", ":"),
            clean_model.replace(":", "--"),
            clean_model.replace("--", "_"),
            clean_model.replace(":", "_"),
            clean_model.removesuffix(":latest"),
            clean_model + ":latest" if not clean_model.endswith(":latest") else clean_model,
            clean_model.removesuffix(".gguf"),
            clean_model.split("/")[-1],
        }
        sanitized_variants = {self._sanitize_model_filename(v) for v in variants if v}

        # 1. Check exact candidate filenames
        for sv in sanitized_variants:
            file_path = self.MODELS_DIR / f"general_{sv}.json"
            if file_path.exists():
                try:
                    file_path.unlink()
                    removed = True
                except Exception:
                    pass

        # 2. Check all remaining per-model files by inspecting their model payload
        if self.MODELS_DIR.exists():
            for pm_path in list(self.MODELS_DIR.glob("general_*.json")):
                try:
                    with open(pm_path, encoding="utf-8") as f:
                        data = json.load(f)
                    pm_model = data.get("model") or ""
                    if pm_model and (
                        pm_model in variants
                        or pm_model in sanitized_variants
                        or self._sanitize_model_filename(pm_model) in sanitized_variants
                    ):
                        pm_path.unlink()
                        removed = True
                except Exception:
                    pass

        # 3. Clean matching artifacts
        if self.ARTIFACTS_DIR.exists():
            for sv in sanitized_variants:
                for artifact in self.ARTIFACTS_DIR.glob(f"{sv}__*"):
                    try:
                        artifact.unlink()
                        removed = True
                    except Exception:
                        pass
        return removed

    def _load_tests_config(self) -> dict[str, list[dict]]:
        filepath = os.getenv("BENCHMARK_TESTS_JSON", "benchmark_tests.json")
        if os.path.exists(filepath):
            try:
                with open(filepath) as f:
                    config = json.load(f)
                required = ["coding", "reasoning", "instruction", "creative", "home_automation"]
                if all(req in config for req in required):
                    for cat in required:
                        for test in config[cat]:
                            test["category"] = cat
                    print(f"[benchmark] Successfully loaded dynamic tests config from {filepath}")
                    return config
            except Exception as e:
                print(f"[benchmark] Error loading {filepath}: {e}. Falling back to hardcoded defaults.")
        return {}

    async def discover_ollama_models(self, base_url: str) -> list[str]:
        """Dynamically discover available models from Ollama endpoint, excluding image models."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = []
                    for model in data.get("models", []):
                        name = model.get("name")
                        if not name:
                            continue
                        is_image = False
                        if model.get("type") == "image":
                            is_image = True
                        details = model.get("details") or {}
                        if details.get("family") == "stable-diffusion":
                            is_image = True
                        families = details.get("families") or []
                        if "stable-diffusion" in families:
                            is_image = True
                        name_lower = name.lower()
                        if any(
                            k in name_lower
                            for k in ("stable-diffusion", "flux", "sdxl", "qwen-rapid-aio", "qwen-image-edit")
                        ):
                            is_image = True
                        if not is_image:
                            models.append(name)
                    print(f"[discover] Discovered {len(models)} text models from {base_url}")
                    return models
                return []
        except Exception as e:
            print(f"[discover] Error discovering models from {base_url}: {e}")
            return []

    async def discover_proxy_models(self, base_url: str) -> list[str]:
        """Dynamically discover available models from Alpaca proxy endpoint, excluding image models."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = []
                    for model in data.get("models", []):
                        name = model.get("model") or model.get("name")
                        if not name:
                            continue
                        is_image = False
                        if model.get("type") == "image":
                            is_image = True
                        details = model.get("details") or {}
                        if details.get("family") == "stable-diffusion":
                            is_image = True
                        families = details.get("families") or []
                        if "stable-diffusion" in families:
                            is_image = True
                        name_lower = name.lower()
                        if any(
                            k in name_lower
                            for k in ("stable-diffusion", "flux", "sdxl", "qwen-rapid-aio", "qwen-image-edit")
                        ):
                            is_image = True
                        if not is_image:
                            models.append(name)
                    print(f"[discover] Discovered {len(models)} text models from proxy {base_url}")
                    return models
                return []
        except Exception as e:
            print(f"[discover] Error discovering models from proxy {base_url}: {e}")
            return []

    async def discover_all_models(self) -> list[str]:
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

    async def discover_all_proxy_models(self) -> list[str]:
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

    def _get_fallback_models(self) -> list[str]:
        env_models = os.getenv("BENCHMARK_MODELS", "")
        if env_models:
            return [m.strip() for m in env_models.split(",") if m.strip()]
        return ["qwen3:8b", "qwen2.5-coder:7b", "qwen3.5:9b"]

    def _coding_tests(self, model: str) -> list[dict]:
        if "coding" in self.tests_config:
            return self.tests_config["coding"]
        return [
            {
                "id": "debug_fix",
                "category": "coding",
                "label": "Python: debug logic error",
                "prompt": "Find and fix the bug in this function:\n\n```\ndef sum_list(items):\n    total = 0\n    for i in range(1, len(items)):\n        total += items[i]\n    return total\n```\nThe function should sum all list items, not skip the first one.",
                "num_predict": 600,
            },
            {
                "id": "code_refactor",
                "category": "coding",
                "label": "Code: refactor for efficiency",
                "prompt": "Refactor this code to be more Pythonic and remove the nested loops:\n\n```\ndef find_unique_numbers(list1, list2, list3):\n    result = []\n    for item in list1:\n        if item not in result:\n            result.append(item)\n    for item in list2:\n        if item not in result:\n            result.append(item)\n    for item in list3:\n        if item not in result:\n            result.append(item)\n    return result\n```",
                "num_predict": 700,
            },
            {
                "id": "guess_game",
                "category": "coding",
                "label": "Game: Number Guessing Game",
                "prompt": "Write a fully functional, interactive Python script for a CLI Number Guessing Game. The game should randomly select a secret number between 1 and 100, give the player 7 attempts, provide higher/lower feedback, and display game stats (attempts used, win/loss) at the end.",
                "num_predict": 900,
            },
            {
                "id": "text_adventure",
                "category": "coding",
                "label": "Game: Text Adventure Game",
                "prompt": "Write a short, interactive text-based adventure game in Python. The player should start in a room with at least two doors (e.g. gold door, monster door). The script must use input() to take player choices, branch the story path based on choices, and lead to at least one winning outcome and one losing outcome.",
                "num_predict": 900,
            },
        ]

    def _reasoning_tests(self, model: str) -> list[dict]:
        if "reasoning" in self.tests_config:
            return self.tests_config["reasoning"]
        return [
            {
                "id": "logic_puzzle",
                "category": "reasoning",
                "label": "Logic: identify rule",
                "prompt": "What rule is being followed in this sequence? 2, 6, 12, 20, 30... and what comes next?",
                "num_predict": 800,
            },
            {
                "id": "math_problem",
                "category": "reasoning",
                "label": "Math: train meeting problem",
                "prompt": "Two trains leave from different cities heading toward each other. Train A travels 60 mph and leaves at 9:00 AM. Train B travels 80 mph and leaves at 10:00 AM. The cities are 500 miles apart. When do they meet?",
                "num_predict": 1200,
            },
        ]

    def _instruction_tests(self, model: str) -> list[dict]:
        if "instruction" in self.tests_config:
            return self.tests_config["instruction"]
        return [
            {
                "id": "json_extraction",
                "category": "instruction",
                "label": "JSON: extract structured data",
                "prompt": "Extract the person's name, age, and location from this text and return as JSON: 'John Doe, 35, lives in Boston, MA'",
                "num_predict": 800,
            },
            {
                "id": "summarization",
                "category": "instruction",
                "label": "Summarization: 3 bullet points",
                "prompt": "Summarize in exactly 3 bullet points: 'The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms. This innovation replaced recurrent neural networks and enabled parallelization, significantly improving training efficiency. Transformers now power state-of-the-art models like GPT, BERT, and T5.'",
                "num_predict": 800,
            },
        ]

    def _creative_tests(self, model: str) -> list[dict]:
        if "creative" in self.tests_config:
            return self.tests_config["creative"]
        return [
            {
                "id": "story_start",
                "category": "creative",
                "label": "Creative: sci-fi story opening",
                "prompt": "Write a compelling 4-sentence opening for a sci-fi story about an AI that discovers it's been dreaming.",
                "num_predict": 800,
            },
            {
                "id": "analogy",
                "category": "creative",
                "label": "Creative: generate analogy",
                "prompt": "Create an analogy comparing friendship to something tangible and useful.",
                "num_predict": 800,
            },
        ]

    def _home_automation_tests(self, model: str) -> list[dict]:
        if "home_automation" in self.tests_config:
            return self.tests_config["home_automation"]
        return [
            {
                "id": "device_control",
                "category": "home_automation",
                "label": "HA: control smart device",
                "prompt": "You are a home automation assistant. A user says: 'Turn on the bedroom light and set it to 60% brightness.' Describe in plain text exactly what action you would take and confirm it to the user.",
                "num_predict": 600,
            },
            {
                "id": "device_status",
                "category": "home_automation",
                "label": "HA: report device status",
                "prompt": "You are a home automation assistant. A user asks: 'Is the thermostat currently set to 68 degrees?' Describe what you would check and give a plausible confirmation response to the user.",
                "num_predict": 600,
            },
        ]

    def _knowledge_tests(self, model: str) -> list[dict]:
        """Objective knowledge/QA benchmarks (MMLU, GPQA, GSM8K, TruthfulQA,
        HellaSwag, WinoGrande, ARC). Each test carries an `expected` field that
        `_verify_functional_response` grades against. Falls back to a tiny
        built-in set if the config omits the 'knowledge' key."""
        if "knowledge" in self.tests_config:
            return self.tests_config["knowledge"]
        return [
            {
                "id": "knowledge_fallback_math",
                "category": "knowledge",
                "label": "Knowledge: basic algebra (fallback)",
                "prompt": "Solve for x: 3x + 5 = 20. Answer with only the final number.",
                "expected": "5",
                "num_predict": 300,
            },
            {
                "id": "knowledge_fallback_science",
                "category": "knowledge",
                "label": "Knowledge: element symbol (fallback)",
                "prompt": "What is the chemical symbol for gold? Answer with only the letter.",
                "expected": "B",
                "num_predict": 300,
            },
        ]

    def _mmlu_pro_tests(self, model: str) -> list[dict]:
        """MMLU-Pro style: 10-option multiple choice (A-J) across disciplines.
        Each test carries an `expected` single-letter answer graded by
        `_verify_functional_response`."""
        if "mmlu_pro" in self.tests_config:
            return self.tests_config["mmlu_pro"]
        return []

    def _gpqa_diamond_tests(self, model: str) -> list[dict]:
        """Graduate-level Google-proof science Q&A (biology/chemistry/physics).
        Each test carries an `expected` single-letter answer graded by
        `_verify_functional_response`."""
        if "gpqa_diamond" in self.tests_config:
            return self.tests_config["gpqa_diamond"]
        return []

    def _hle_tests(self, model: str) -> list[dict]:
        """Humanity's Last Exam style expert reasoning with verifiable answers
        (letter MC or short exact substring). Each test carries an `expected`
        graded by `_verify_functional_response`."""
        if "hle" in self.tests_config:
            return self.tests_config["hle"]
        return []

    def _math_hard_tests(self, model: str) -> list[dict]:
        """MATH-500 / AIME-style competition math requiring multi-step reasoning.
        Each test carries a numeric `expected` graded by `_verify_functional_response`."""
        if "math_hard" in self.tests_config:
            return self.tests_config["math_hard"]
        return []

    def _ifeval_tests(self, model: str) -> list[dict]:
        """Instruction-following prompts with programmatically verifiable constraints.
        Each test carries an `expected` keyword/substring graded by
        `_verify_functional_response`."""
        if "ifeval" in self.tests_config:
            return self.tests_config["ifeval"]
        return []

    def _gamedev_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("gamedev", [])

    def _gamedev_alt_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("gamedev_alt", [])

    def _youtuber_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("youtuber", [])

    def _agentic_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("agentic", [])

    def _appdev_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("appdev", [])

    def _linux_admin_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("linux_admin", [])

    def _webdev_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("webdev", [])

    def _database_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("database", [])

    def _cpp_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("cpp", [])

    def _java_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("java", [])

    def _code_review_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("code_review", [])

    def _debugging_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("debugging", [])

    def _logic_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("logic", [])

    def _retrogames_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("retrogames", [])

    def _threedprint_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("threedprint", [])

    def _languages_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("languages", [])

    def _tvdev_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("tvdev", [])

    def _uiux_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("uiux", [])

    def _office_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("office", [])

    def _life_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("life", [])

    def _biblical_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("biblical", [])

    def _metacog_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("metacog", [])

    def _networking_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("networking", [])

    def _usb_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("usb", [])

    def _iac_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("iac", [])

    def _linux_driver_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("linux_driver", [])

    def _bash_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("bash", [])

    def _basic_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("basic", [])

    def _pascal_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("pascal", [])

    def _typescript_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("typescript", [])

    def _rpm_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("rpm", [])

    def _android_tests(self, model: str) -> list[dict]:
        return self.tests_config.get("android", [])

    def strip_thinking(self, text: str) -> "tuple[str, str | None]":
        """Split reasoning from output.

        Returns ``(clean, thinking)`` where ``clean`` is the response with any
        ``<think>/<thinking>`` blocks removed (so the stored response is pure code/output)
        and ``thinking`` is the concatenated reasoning text, preserved for optional display.
        Handles matched (``<think>...</think>``) and mismatched (``<thinking>...</think>``)
        tag pairings. Also removes plain-text "thinking process" preambles (e.g. models
        that narrate ``Here's a thinking process:`` before writing code) up to the first
        unindented code fence, so leaked reasoning + its draft snippets never pollute the
        stored response or the code-quality checker.
        """
        tag_re = r"<think[^>]*>[\s\S]*?</think[^>]*>"
        blocks = re.findall(tag_re, text, flags=re.IGNORECASE)
        clean = re.sub(tag_re, "", text, flags=re.IGNORECASE).strip()
        thinking = "\n\n".join(b.strip() for b in blocks) if blocks else None
        # Plain-text reasoning preamble: a "thinking" style header line followed by
        # prose, ending at the first unindented code fence. Only stripped when a real
        # (column-0) fence follows, so legitimate short intros like "Here's the game:"
        # and pure-prose answers are never damaged.
        preamble_re = re.compile(
            r"(?:^|\n)(?:Here'?s (?:a |my |the )?thinking (?:process|steps?)"
            r"|Let me think (?:about|through|this)"
            r"|Let me (?:reason|work|think) (?:through|about|this)"
            r"|Thought process|My thinking|I'll think|I will think|I'll reason"
            r"|Step-by-step thinking)[^:\n]*:(.*?)(?=\n```)",
            re.IGNORECASE | re.DOTALL,
        )
        match = preamble_re.search(clean)
        if match:
            preamble = match.group(0).strip()
            clean = clean.replace(preamble, "", 1).strip()
            thinking = f"{preamble}\n\n{thinking}" if thinking else preamble
        return clean, thinking

    def _has_persistent_scoreboard(self, cleaned: str) -> bool:
        """Check a response implements a persistent, saveable score/record board.

        Arcade/retro benchmarks require: (1) a numeric score/record being tracked,
        (2) persistence of the best scores to a file (e.g. high_scores.json) so
        they survive restarts, (3) the player being able to enter a name/initials
        for their score, and (4) the score resetting to 0 when a new game starts
        (never carrying the previous session's score over).
        """
        score_word = any(x in cleaned for x in ["score", "record", "population"])
        persist = any(x in cleaned for x in ["json", "dump", "load", "save", "localstorage", 'open("', "open('"])
        name_entry = ("name" in cleaned or "initial" in cleaned) and any(
            x in cleaned for x in ["input(", "readline", "gets", "prompt(", "value", "save"]
        )
        reset = any(
            x in cleaned for x in ["reset", "= 0", "=0", "new game", "new_game", "newgame", "resetscore", "clear"]
        )
        return score_word and persist and name_entry and reset

    def _verify_functional_response(self, test: "dict | str", response: str) -> bool:
        """Evaluate functional response correctness based on the target requirements.

        Accepts either the full test dict (preferred) or a legacy test_id string.
        When the test carries an `expected` answer it is graded against that answer
        (multiple-choice letter, numeric, or substring), enabling objective
        knowledge/fact benchmarks loaded from config.
        """
        if isinstance(test, dict):
            test_id = test.get("id", "")
            expected = test.get("expected")
        else:
            test_id = test
            expected = None

        if not response or len(response.strip()) == 0:
            return False

        # Clean the response from potential think tags
        cleaned = re.sub(r"<(think|thinking)>[\s\S]*?</\1>", "", response, flags=re.IGNORECASE).strip().lower()

        # Expected-answer grading (objective tests such as the knowledge category)
        if expected is not None:
            exp = str(expected).strip()
            norm = cleaned.upper()
            if re.fullmatch(r"[A-J]", exp):
                return bool(re.search(rf"(?:\(?{re.escape(exp)}\)?[\.\:\s]|(?<!\w){re.escape(exp)}(?!\w))", norm))
            if exp.isdigit():
                return bool(re.search(rf"(?<!\d){re.escape(exp)}(?!\d)", cleaned))
            # Fold diacritics so "Gödel" matches expected "godel".
            return _ascii_fold(exp) in _ascii_fold(cleaned)

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
            no_spaces = cleaned.replace(" ", "")
            correct_patterns = [
                "range(len(items)",
                "range(0,len(items)",
                "sum(items)",
                "enumerate(items)",
                "for item in items",
                "for i in items",
                "for x in items",
                "for val in items",
                "for num in items",
                "for elem in items",
                "for element in items",
            ]
            if "range(1,len" in no_spaces:
                return any(p in no_spaces for p in correct_patterns)
            return any(p in no_spaces for p in correct_patterns)

        elif test_id == "code_refactor":
            no_spaces = cleaned.replace(" ", "")
            set_comprehension = re.search(r"\{[^{}]*\bfor\b[^{}]*\bin\b", cleaned)
            return any(x in no_spaces for x in ["set(", "fromkeys("]) or bool(set_comprehension)

        elif test_id == "guess_game":
            return (
                "random" in cleaned
                and "input(" in cleaned
                and any(x in cleaned for x in ["randint", "randrange", "random.choice", "secrets.randbelow"])
                and any(x in cleaned for x in ["attempt", "guess", "tries"])
            )

        elif test_id == "game_checkers_cli":
            return (
                any(x in cleaned for x in ["checker", "draught", "board"])
                and any(x in cleaned for x in ["capture", "jump", "king", "crown"])
                and any(x in cleaned for x in ["move", "square", "diagonal"])
                and self._has_persistent_scoreboard(cleaned)
            )

        elif test_id == "text_adventure":
            return (
                "input(" in cleaned
                and any(x in cleaned for x in ["door", "choice", "room", "path"])
                and any(x in cleaned for x in ["win", "lose", "gold", "monster", "victory", "defeat"])
            )

        elif test_id == "logic_puzzle":
            return any(x in cleaned for x in ["42", "forty-two", "forty two"])

        elif test_id == "math_problem":
            no_spaces = cleaned.replace(" ", "")
            return any(x in no_spaces for x in ["1:08", "1:09", "13:08", "13:09", "1.08pm", "1.09pm", "1.08", "1.09"])

        elif test_id == "json_extraction":
            has_name = "john" in cleaned
            has_age = "35" in cleaned
            has_loc = "boston" in cleaned
            has_brackets = "{" in cleaned and "}" in cleaned
            return has_brackets and has_name and has_age and has_loc

        elif test_id == "summarization":
            bullets = re.findall(r"(?:^\s*[-*•+\d]\s+.*$)|(?:^\d+\..*$)", response, re.MULTILINE)
            return len(bullets) >= 3

        elif test_id == "instruction_adherence":
            # Strict format-following: exactly 3 markdown bullets, mandatory ordered
            # bold labels, required terms present, forbidden terms absent.
            lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
            bullet_pat = r"^(?:[-*+]|\d+[.)])\s+"
            bullets = [ln for ln in lines if re.match(bullet_pat, ln)]
            if len(lines) != 3 or len(bullets) != 3:
                return False
            labels = ("failures:", "pattern:", "cause:")
            lows: list[str] = []
            for i, b in enumerate(bullets):
                text = re.sub(bullet_pat, "", b).strip().replace("**", "")
                low = text.lower()
                if not low.startswith(labels[i]):
                    return False
                if len(text.split()) > 28:
                    return False
                lows.append(low)
            joined = " ".join(lows)
            if not all(term in joined for term in ("ai overview", "token", "transformer")):
                return False
            return not any(term in cleaned for term in ("strawberry", "kindergarten", "image credits"))

        elif test_id == "device_control":
            return "bedroom" in cleaned and any(x in cleaned for x in ["60%", "60", "sixty"])

        elif test_id == "device_status":
            return "thermostat" in cleaned and any(x in cleaned for x in ["68", "sixty-eight", "sixty eight"])

        # ---- Game-dev (modern) ----
        elif test_id == "game_pong_collision":
            return (
                "paddle" in cleaned
                and "ball" in cleaned
                and any(x in cleaned for x in ["collision", "overlap", "reflect", "bounce"])
                and any(x in cleaned for x in ["abs(", "distance", "velocity", "vy", "dx", "reverse"])
            )
        elif test_id == "game_snake_step":
            return (
                "snake" in cleaned
                and any(x in cleaned for x in ["food", "apple"])
                and any(x in cleaned for x in ["grow", "len(", "append"])
                and any(x in cleaned for x in ["wall", "collision", "self", "die", "boundary"])
            )
        elif test_id == "game_inv_system":
            return (
                any(x in cleaned for x in ["inventory", "item"])
                and any(x in cleaned for x in ["add", "append", "remove", "pop", "insert"])
                and any(x in cleaned for x in ["count", "len(", "dict", "quantity"])
            )
        elif test_id == "game_combat_turn":
            return (
                any(x in cleaned for x in ["hp", "health"])
                and any(x in cleaned for x in ["attack", "damage"])
                and any(x in cleaned for x in ["turn", "defend", "defense", "enemy"])
            )
        elif test_id == "game_match3_find":
            return (
                any(x in cleaned for x in ["match", "row"])
                and any(x in cleaned for x in ["for ", "loop", "iterate"])
                and any(x in cleaned for x in ["horizontal", "vertical", "grid", "column", "board"])
                and any(x in cleaned for x in ["3", "three", "triple"])
            )
        # ---- App-dev ----
        elif test_id == "app_todo_crud":
            return (
                any(x in cleaned for x in ["add", "append", "create", "insert", "post"])
                and any(x in cleaned for x in ["list", "print", "display", "get"])
                and any(x in cleaned for x in ["delete", "remove", "pop"])
            )
        elif test_id == "app_fsm_order":
            return (
                "state" in cleaned
                and any(x in cleaned for x in ["transition", "next", "machine"])
                and any(x in cleaned for x in ["paid", "shipped", "delivered", "cancel"])
                and "pending" in cleaned
            )
        elif test_id == "app_lru_cache":
            return (
                any(x in cleaned for x in ["lru", "cache"])
                and any(x in cleaned for x in ["evict", "pop", "remove", "expire"])
                and any(x in cleaned for x in ["ordereddict", "dict", "deque", "list"])
                and any(x in cleaned for x in ["recent", "least", "capacity"])
            )
        elif test_id == "app_log_parser":
            return (
                any(x in cleaned for x in ["regex", "re.search", "re.findall", "re.match", "pattern"])
                and "log" in cleaned
                and any(x in cleaned for x in ["count", "level", "error", "group"])
            )
        elif test_id == "app_rate_limiter":
            return (
                any(x in cleaned for x in ["rate", "limit"])
                and any(x in cleaned for x in ["token", "bucket", "window", "throttle"])
                and any(x in cleaned for x in ["allow", "reject", "block", "exceed"])
            )
        # ---- Linux / server admin ----
        elif test_id == "linux_find_large":
            return (
                "find" in cleaned
                and ("-size" in cleaned or "size" in cleaned)
                and ("100" in cleaned and any(x in cleaned for x in ["m", "mb", "meg"]))
                and any(x in cleaned for x in ["sort", "du", "xargs"])
            )
        elif test_id == "linux_perm_fix":
            return (
                "chmod" in cleaned
                and any(x in cleaned for x in ["755", "644"])
                and any(x in cleaned for x in ["find", "directory", "dir", "type", "exec"])
                and any(x in cleaned for x in ["xargs", "chown", "recursive", "-r", "exec", "-exec"])
            )
        elif test_id == "linux_journal_errors":
            return (
                "journalctl" in cleaned
                and any(x in cleaned for x in ["err", "error", "priority"])
                and any(x in cleaned for x in ["nginx", "unit", "-u", "service"])
            )
        elif test_id == "linux_disk_top":
            return (
                "du" in cleaned
                and "sort" in cleaned
                and any(x in cleaned for x in ["head", "top", "largest"])
                and any(x in cleaned for x in ["-h", "human", "size", "sh"])
            )
        elif test_id == "linux_ssh_harden":
            return (
                any(x in cleaned for x in ["passwordauthentication", "permitrootlogin", "sshd", "ssh"])
                and "no" in cleaned
                and any(x in cleaned for x in ["systemctl", "reload", "restart", "sshd_config"])
            )
        # ---- Web dev ----
        elif test_id == "web_fetch_render":
            return (
                "fetch" in cleaned
                and any(x in cleaned for x in ["then", "await", "async"])
                and any(x in cleaned for x in ["render", "append", "innerhtml", "map", "createelement", "textcontent"])
            )
        elif test_id == "web_event_delegate":
            return (
                any(x in cleaned for x in ["addeventlistener", "onclick"])
                and any(x in cleaned for x in ["target", "closest", "matches"])
                and any(x in cleaned for x in ["delegate", "container", "parent", "ul", "list"])
            )
        elif test_id == "web_form_validate":
            return (
                any(x in cleaned for x in ["validate", "validation"])
                and any(x in cleaned for x in ["email", "regex", "pattern"])
                and any(x in cleaned for x in ["error", "invalid", "message"])
            )
        elif test_id == "web_localstorage":
            return any(x in cleaned for x in ["localstorage", "setitem", "getitem"]) and any(
                x in cleaned for x in ["json", "parse", "stringify", "store"]
            )
        elif test_id == "web_dom_toggle":
            return (
                any(x in cleaned for x in ["toggle", "classlist"])
                and any(x in cleaned for x in ["addeventlistener", "onclick", "click"])
                and any(x in cleaned for x in ["class", "hidden", "active", "show"])
            )
        # ---- Database / SQL ----
        elif test_id == "db_join_orders":
            return (
                "select" in cleaned
                and "join" in cleaned
                and any(x in cleaned for x in [" on ", "using", "on="])
                and any(x in cleaned for x in ["customers", "orders", "customer"])
            )
        elif test_id == "db_add_index":
            return (
                any(x in cleaned for x in ["create index", "index"])
                and "on " in cleaned
                and any(x in cleaned for x in ["table", "users", "orders", "column"])
            )
        elif test_id == "db_atomic_transfer":
            return (
                any(x in cleaned for x in ["begin", "transaction", "commit"])
                and "update" in cleaned
                and any(x in cleaned for x in ["rollback", "atomic", "savepoint"])
            )
        elif test_id == "db_create_users":
            return (
                any(x in cleaned for x in ["create table", "create table"])
                and "users" in cleaned
                and any(x in cleaned for x in ["primary key", "primarykey", "id"])
                and any(x in cleaned for x in ["unique", "email"])
            )
        elif test_id == "db_monthly_revenue":
            return (
                any(x in cleaned for x in ["group by", "groupby"])
                and any(x in cleaned for x in ["sum", "sum("])
                and any(x in cleaned for x in ["month", "date", "extract", "interval"])
                and any(x in cleaned for x in ["revenue", "amount", "sales"])
            )
        # ---- C++ ----
        elif test_id == "cpp_vector_sum":
            return (
                "vector" in cleaned
                and any(x in cleaned for x in ["for ", "range", "accumulate"])
                and any(x in cleaned for x in ["sum", "+=", "total", "std::"])
            )
        elif test_id == "cpp_unique_ptr":
            return (
                any(x in cleaned for x in ["unique_ptr", "make_unique"])
                and any(x in cleaned for x in ["std::", "delete", "new "])
                and any(x in cleaned for x in ["memory", "ptr", "leak"])
            )
        elif test_id == "cpp_template_max":
            return (
                "template" in cleaned
                and any(x in cleaned for x in ["typename", "class "])
                and "max" in cleaned
                and "return" in cleaned
            )
        elif test_id == "cpp_class_bank":
            return (
                any(x in cleaned for x in ["class ", "struct "])
                and any(x in cleaned for x in ["deposit", "withdraw"])
                and any(x in cleaned for x in ["balance", "account", "amount"])
            )
        elif test_id == "cpp_threads_mutex":
            return (
                "thread" in cleaned
                and any(x in cleaned for x in ["std::", "mutex", "lock"])
                and any(x in cleaned for x in ["mutex", "lock_guard", "lock", "atomic"])
            )
        # ---- Java ----
        elif test_id == "java_stream_filter":
            return (
                "stream" in cleaned
                and "filter" in cleaned
                and any(x in cleaned for x in ["collect", "tolist", "list", "map"])
            )
        elif test_id == "java_trycatch_parse":
            return (
                "try" in cleaned
                and "catch" in cleaned
                and any(x in cleaned for x in ["numberformatexception", "parseint", "parse"])
                and any(x in cleaned for x in ["integer", "int "])
            )
        elif test_id == "java_word_freq":
            return (
                "map" in cleaned
                and any(x in cleaned for x in ["getordefault", "put", "merge", "computeifabsent"])
                and any(x in cleaned for x in ["split", "word", "frequency"])
            )
        elif test_id == "java_shape_iface":
            return (
                "interface" in cleaned
                and any(x in cleaned for x in ["area", "implements", "method"])
                and any(x in cleaned for x in ["circle", "rectangle", "square", "class "])
            )
        elif test_id == "java_jdbc_select":
            return (
                any(x in cleaned for x in ["connection", "drivermanager"])
                and any(x in cleaned for x in ["statement", "preparedstatement"])
                and any(x in cleaned for x in ["resultset", "executequery", "executequery"])
            )
        # ---- Debugging ----
        elif test_id == "debug_offbyone":
            return (
                any(x in cleaned for x in ["off", "one", "range", "len(", "index"])
                and any(x in cleaned for x in ["loop", "<=", "<", "boundary", "array"])
                and any(x in cleaned for x in ["error", "bug", "fix", "off-by"])
            )
        elif test_id == "debug_null_deref":
            return (
                any(x in cleaned for x in ["null", "none"])
                and any(x in cleaned for x in ["check", "guard", "if "])
                and any(x in cleaned for x in ["dereference", "pointer", "attribute", "error", "nullpointerexception"])
            )
        elif test_id == "debug_data_race":
            return (
                any(x in cleaned for x in ["race", "thread"])
                and any(x in cleaned for x in ["mutex", "lock", "atomic"])
                and any(x in cleaned for x in ["shared", "synchron", "volatile"])
            )
        elif test_id == "debug_infinite_loop":
            return (
                any(x in cleaned for x in ["infinite", "loop"])
                and any(x in cleaned for x in ["condition", "terminate", "break", "while"])
                and any(x in cleaned for x in ["fix", "never", "update"])
            )
        elif test_id == "debug_sql_injection":
            return (
                any(x in cleaned for x in ["injection", "sql"])
                and any(x in cleaned for x in ["parameter", "prepared", "bind", "placeholder"])
                and any(x in cleaned for x in ["statement", "query", "escape"])
            )
        # ---- Logic ----
        elif test_id == "logic_knights":
            return (
                any(x in cleaned for x in ["knight", "knave"])
                and any(
                    x in cleaned
                    for x in [
                        "liar",
                        "truth",
                        "said",
                        "statement",
                        "is the",
                        "a is",
                        "b is",
                        "because",
                        "therefore",
                        "reason",
                    ]
                )
                and "a" in cleaned
                and "b" in cleaned
            )
        elif test_id == "logic_river":
            return any(x in cleaned for x in ["wolf", "goat", "cabbage"]) and any(
                x in cleaned for x in ["cross", "bank", "river", "boat"]
            )
        elif test_id == "logic_modus":
            return (
                any(x in cleaned for x in ["modus", "ponens", "therefore"])
                and "if" in cleaned
                and "then" in cleaned
                and "p" in cleaned
                and "q" in cleaned
            )
        elif test_id == "logic_syllogism":
            # Stem match: models may answer with singular or plural forms.
            return (
                any(x in cleaned for x in ["yes", "definitely", "true", "correct"])
                and "bloop" in cleaned
                and "lazzi" in cleaned
            )
        elif test_id == "logic_weigh":
            return (
                any(x in cleaned for x in ["coin", "weigh", "balance", "scale"])
                and any(x in cleaned for x in ["heavier", "lighter", "fake"])
                and any(x in cleaned for x in ["3", "three", "group", "two"])
            )
        # ---- Retro games ----
        elif test_id == "retro_space_invaders":
            return (
                any(x in cleaned for x in ["invader", "alien"])
                and any(x in cleaned for x in ["grid", "row", "for "])
                and any(x in cleaned for x in ["shoot", "bullet", "shot"])
                and any(x in cleaned for x in ["score", "collision", "hit"])
            ) and self._has_persistent_scoreboard(cleaned)
        elif test_id == "retro_maelstrom":
            return (
                "asteroid" in cleaned
                and any(x in cleaned for x in ["bullet", "shot"])
                and any(x in cleaned for x in ["split", "velocity", "smaller"])
                and any(x in cleaned for x in ["ship", "angle", "player"])
            ) and self._has_persistent_scoreboard(cleaned)
        elif test_id == "retro_space_shooter":
            return (
                any(x in cleaned for x in ["enemy", "ship", "spawn"])
                and any(x in cleaned for x in ["random", "randint", "randrange"])
                and any(x in cleaned for x in ["collision", "overlap", "hit"])
                and any(x in cleaned for x in ["move", "down", "tick", " y ", "y="])
            ) and self._has_persistent_scoreboard(cleaned)
        elif test_id == "retro_subway_surfer":
            return (
                "lane" in cleaned
                and "jump" in cleaned
                and any(x in cleaned for x in ["obstacle", "barrier"])
                and any(x in cleaned for x in ["switch", "left", "right", "center"])
            ) and self._has_persistent_scoreboard(cleaned)
        elif test_id == "retro_temple_run":
            return (
                "speed" in cleaned
                and "jump" in cleaned
                and any(x in cleaned for x in ["slide", "duck"])
                and any(x in cleaned for x in ["turn", "corner", "direction"])
                and any(x in cleaned for x in ["pit", "barrier"])
            ) and self._has_persistent_scoreboard(cleaned)
        elif test_id == "retro_donkey_kong":
            return (
                "ladder" in cleaned
                and "barrel" in cleaned
                and "platform" in cleaned
                and any(x in cleaned for x in ["jump", "climb"])
                and any(x in cleaned for x in ["life", "top", "die"])
            ) and self._has_persistent_scoreboard(cleaned)
        elif test_id == "retro_mario":
            return (
                "gravity" in cleaned
                and any(x in cleaned for x in ["goomba", "enemy"])
                and any(x in cleaned for x in ["stomp", "jump"])
                and any(x in cleaned for x in ["pit", "fall", "gap"])
                and any(x in cleaned for x in ["velocity", "vx", "vy"])
            ) and self._has_persistent_scoreboard(cleaned)
        elif test_id == "retro_arcade_loop":
            return (
                "class " in cleaned
                and "score" in cleaned
                and "lives" in cleaned
                and "level" in cleaned
                and "update" in cleaned
            ) and self._has_persistent_scoreboard(cleaned)
        elif test_id == "retro_crossy_road":
            return (
                any(x in cleaned for x in ["hop", "forward", "row"])
                and any(x in cleaned for x in ["car", "log", "lane", "vehicle"])
                and any(x in cleaned for x in ["water", "river", "hit"])
                and any(x in cleaned for x in ["surviv", "dead", "alive"])
            ) and self._has_persistent_scoreboard(cleaned)
        elif test_id == "retro_flappy_bird":
            return (
                "gravity" in cleaned
                and any(x in cleaned for x in ["flap", "velocity", "vy"])
                and any(x in cleaned for x in ["pipe", "gap"])
                and any(x in cleaned for x in ["bird", "alive", "dead", "collision"])
            ) and self._has_persistent_scoreboard(cleaned)
        elif test_id == "retro_echo_dolphin":
            return (
                "dolphin" in cleaned
                and any(x in cleaned for x in ["fish", "eat"])
                and any(x in cleaned for x in ["shark", "damage", "enemy"])
                and any(x in cleaned for x in ["water", "sea", "swim"])
                and any(x in cleaned for x in ["velocity", "jump", "leap"])
            ) and self._has_persistent_scoreboard(cleaned)
        elif test_id == "retro_pacman":
            return (
                "pac" in cleaned
                and any(x in cleaned for x in ["ghost", "pellet"])
                and any(x in cleaned for x in ["maze", "grid"])
                and "score" in cleaned
                and any(x in cleaned for x in ["eat", "chase"])
            ) and self._has_persistent_scoreboard(cleaned)
        elif test_id == "retro_tetris":
            return (
                any(x in cleaned for x in ["tetromino", "tetris", "piece"])
                and any(x in cleaned for x in ["rotate", "rotation"])
                and "row" in cleaned
                and any(x in cleaned for x in ["clear", "full"])
                and any(x in cleaned for x in ["board", "grid"])
                and any(x in cleaned for x in ["lock", "fall", "down"])
            ) and self._has_persistent_scoreboard(cleaned)
        elif test_id == "game_minecraft_voxel":
            return (
                any(x in cleaned for x in ["voxel", "block", "chunk"])
                and any(x in cleaned for x in ["grid", "3d", " x ", " y ", " z "])
                and any(x in cleaned for x in ["gravity", "fall", "land", "solid"])
                and any(x in cleaned for x in ["get", "set", "block"])
            )
        elif test_id == "game_fps":
            return (
                any(x in cleaned for x in ["raycast", "hitscan", "ray"])
                and any(x in cleaned for x in ["wall", "box", "aabb"])
                and any(x in cleaned for x in ["yaw", "pitch", "angle"])
                and any(x in cleaned for x in ["hit", "intersect", "distance"])
            )
        elif test_id == "game_blockblast":
            return (
                any(x in cleaned for x in ["grid", "board"])
                and any(x in cleaned for x in ["polyomino", "piece", "block"])
                and ("clear" in cleaned or ("row" in cleaned and "column" in cleaned))
                and "score" in cleaned
            ) and self._has_persistent_scoreboard(cleaned)
        elif test_id == "game_sokoban":
            return (
                any(x in cleaned for x in ["sokoban", "box", "push"])
                and any(x in cleaned for x in ["target", "goal"])
                and any(x in cleaned for x in ["win", "solved", "all"])
            ) and self._has_persistent_scoreboard(cleaned)
        elif test_id == "game_breakout":
            return (
                "paddle" in cleaned
                and "ball" in cleaned
                and any(x in cleaned for x in ["brick", "bounce", "wall"])
                and any(x in cleaned for x in ["life", "lives", "miss", "lose", "lost", "bottom"])
                and any(
                    x in cleaned
                    for x in ["power", "capsule", "laser", "multi", "catch", "enlarge", "expand", "boss", "doh"]
                )
            ) and self._has_persistent_scoreboard(cleaned)
        elif test_id == "game_simcity":
            return (
                any(x in cleaned for x in ["zone", "city", "sim"])
                and any(x in cleaned for x in ["res", "com", "ind", "demand"])
                and any(x in cleaned for x in ["popul", "grow", "tick"])
            ) and self._has_persistent_scoreboard(cleaned)
        # ---- Gamedev games without dedicated branches (now explicit) ----
        elif test_id in ("game_pong", "game_snake", "game_frameworkless_canvas", "game_breakout_rt"):
            score_word = "score" in cleaned
            if test_id == "game_pong":
                core = (
                    "paddle" in cleaned
                    and "ball" in cleaned
                    and any(x in cleaned for x in ["key", "arrow", "up", "down"])
                )
            elif test_id == "game_snake":
                core = (
                    "snake" in cleaned
                    and any(x in cleaned for x in ["food", "eat"])
                    and any(x in cleaned for x in ["grow", "append", "length"])
                )
            elif test_id == "game_frameworkless_canvas":
                core = (
                    "canvas" in cleaned
                    and any(x in cleaned for x in ["requestanimationframe", "game loop"])
                    and "enemy" in cleaned
                )
            else:  # game_breakout_rt
                core = "paddle" in cleaned and "ball" in cleaned and "brick" in cleaned
            return core and score_word and self._has_persistent_scoreboard(cleaned)
        # ---- gamedev_alt: web UI games (HTML5 canvas / three.js / WebGL) ----
        elif test_id == "game_snake_canvas":
            return (
                "snake" in cleaned
                and any(x in cleaned for x in ["canvas", "context", "getcontext"])
                and any(x in cleaned for x in ["food", "eat", "grow", "length"])
                and any(x in cleaned for x in ["keydown", "arrow", "key"])
                and any(x in cleaned for x in ["requestanimationframe", "setinterval", "tick"])
                and self._has_persistent_scoreboard(cleaned)
            )
        elif test_id == "game_breakout_canvas":
            return (
                "paddle" in cleaned
                and "ball" in cleaned
                and "brick" in cleaned
                and any(x in cleaned for x in ["canvas", "context", "getcontext"])
                and any(x in cleaned for x in ["requestanimationframe", "setinterval", "tick"])
                and self._has_persistent_scoreboard(cleaned)
            )
        elif test_id == "game_asteroids_threejs":
            return (
                any(x in cleaned for x in ["three", "scene", "perspectivecamera"])
                and "ship" in cleaned
                and any(x in cleaned for x in ["asteroid", "rock"])
                and any(x in cleaned for x in ["thrust", "momentum", "inertia"])
                and any(x in cleaned for x in ["requestanimationframe", "animate"])
                and any(x in cleaned for x in ["localstorage", "score"])
                and self._has_persistent_scoreboard(cleaned)
            )
        elif test_id == "game_checkers_web":
            return (
                any(x in cleaned for x in ["checker", "draught"])
                and "board" in cleaned
                and any(x in cleaned for x in ["capture", "jump", "king"])
                and any(x in cleaned for x in ["move", "square", "tile"])
                and self._has_persistent_scoreboard(cleaned)
            )
        elif test_id == "game_pong_threejs":
            return (
                any(x in cleaned for x in ["three", "scene", "perspectivecamera"])
                and "paddle" in cleaned
                and "ball" in cleaned
                and any(x in cleaned for x in ["requestanimationframe", "animate"])
                and any(x in cleaned for x in ["localstorage", "score"])
                and self._has_persistent_scoreboard(cleaned)
            )
        elif test_id == "game_3d_terrain_webgl":
            return (
                "webgl" in cleaned
                and any(x in cleaned for x in ["shader", "gl_", "buffer"])
                and any(x in cleaned for x in ["terrain", "voxel", "height"])
                and any(x in cleaned for x in ["perspective", "matrix", "projection"])
            )
        # ---- youtuber: viral sandbox/physics sims ----
        elif test_id in ("game_falling_sand", "game_conway_life", "game_boids"):
            persist = any(x in cleaned for x in ["json", "dump", "load", "save", 'open("', "open('"])
            if test_id == "game_falling_sand":
                core = (
                    any(x in cleaned for x in ["sand", "particle", "cell"])
                    and any(x in cleaned for x in ["water", "fire", "wood", "lava", "smoke"])
                    and any(x in cleaned for x in ["grid", "array", "matrix"])
                )
            elif test_id == "game_conway_life":
                core = (
                    "life" in cleaned
                    and any(x in cleaned for x in ["neighbor", "cell", "glider"])
                    and any(x in cleaned for x in ["rules", "alive", "dead", "birth", "survive"])
                )
            else:  # game_boids
                core = (
                    "boid" in cleaned
                    and any(x in cleaned for x in ["separ", "align", "cohes"])
                    and any(x in cleaned for x in ["radius", "neighbor", "flock"])
                )
            return core and persist
        # ---- agentic: long-running complex scenarios (advanced tier) ----
        elif test_id == "agentic_incident_forensics":
            return (
                any(x in cleaned for x in ["root cause", "root-cause", "rootcause"])
                and any(x in cleaned for x in ["rollback", "mitigation"])
                and any(x in cleaned for x in ["timeline", "timestamp", "seq=", "propagat"])
                and "alpha-9842-omega" in cleaned
            )
        elif test_id == "agentic_codebase_migration":
            return (
                any(x in cleaned for x in ["inventory", "architecture", "phases", "phase 1"])
                and any(x in cleaned for x in ["php", "module", "package"])
                and "kappa-7713-zeta" in cleaned
            )
        elif test_id == "agentic_long_running_service":
            return (
                any(x in cleaned for x in ["robot", "fleet", "agent"])
                and any(x in cleaned for x in ["fault", "heal", "re-route", "reroute"])
                and any(x in cleaned for x in ["deliver", "uptime", "buffer"])
            ) and self._has_persistent_scoreboard(cleaned)
        # ---- 3D printing / CAD-CAM ----
        elif test_id == "td_gcode_cube":
            return (
                any(x in cleaned for x in ["g0", "g1", "g28"])
                and "x" in cleaned
                and "y" in cleaned
                and any(x in cleaned for x in ["e", "extrude", "extrusion"])
                and any(x in cleaned for x in ["home", "g28"])
            )
        elif test_id == "td_gcode_temps":
            return (
                any(x in cleaned for x in ["m104", "m109"])
                and any(x in cleaned for x in ["m140", "m190"])
                and "210" in cleaned
                and "60" in cleaned
            )
        elif test_id == "td_openscad_cube_hole":
            return (
                "cube(" in cleaned
                and any(x in cleaned for x in ["cylinder(", "hole"])
                and any(x in cleaned for x in ["difference(", "translate("])
                and "20" in cleaned
            )
        elif test_id == "td_openscad_gear":
            return (
                "gear" in cleaned
                and any(x in cleaned for x in ["teeth", "tooth"])
                and any(x in cleaned for x in ["for (", "module"])
                and any(x in cleaned for x in ["circle(", "rotate", "polygon"])
            )
        elif test_id == "td_stl_binary":
            return (
                "stl" in cleaned
                and any(x in cleaned for x in ["triangle", "facet", "normal"])
                and any(x in cleaned for x in ["struct", "binary", "pack", "header"])
                and any(x in cleaned for x in ["cube", "vertex", "80"])
            )
        elif test_id == "td_printer_api":
            return (
                "octoprint" in cleaned
                and any(x in cleaned for x in ["api/files", "api/job"])
                and any(x in cleaned for x in ["apikey", "x-api-key", "header"])
                and "post" in cleaned
            )
        elif test_id == "td_laser_api":
            return (
                "laser" in cleaned
                and any(x in cleaned for x in ["svg", "cut", "gcode"])
                and any(x in cleaned for x in ["post", "upload", "submit"])
                and any(x in cleaned for x in ["auth", "token", "bearer"])
            )
        elif test_id == "td_slicer_cfg":
            return (
                any(x in cleaned for x in ["layer_height", "layer height"])
                and "infill" in cleaned
                and "support" in cleaned
                and any(x in cleaned for x in ["0.2", "20"])
            )
        # ---- Languages (varied, some framework) ----
        elif test_id == "lang_go_http":
            return (
                "package main" in cleaned
                and any(x in cleaned for x in ["net/http", "handlefunc", "listenandserve", "http."])
                and "func " in cleaned
                and "hello" in cleaned
            )
        elif test_id == "lang_rust_cli":
            return (
                "fn main" in cleaned
                and any(x in cleaned for x in ["std::io", "io::stdin", "read_line", "stdin"])
                and any(x in cleaned for x in ["println!", "count", "lines"])
                and any(x in cleaned for x in ["let ", "mut "])
            )
        elif test_id == "lang_node_server":
            return (
                any(x in cleaned for x in ["require(", "import ", "node", "http"])
                and any(x in cleaned for x in ["createserver", "listen", "http"])
                and any(x in cleaned for x in ["/time", "get"])
                and any(x in cleaned for x in ["json", "timestamp", "date"])
            )
        elif test_id == "lang_html_page":
            return (
                any(x in cleaned for x in ["<!doctype", "<html", "<!doctype "])
                and any(x in cleaned for x in ["<form", "form"])
                and any(x in cleaned for x in ["<button", "button"])
                and any(x in cleaned for x in ["<script", "script", "input"])
            )
        elif test_id == "lang_python_cli":
            return (
                any(x in cleaned for x in ["def ", "import ", "argparse"])
                and any(x in cleaned for x in ["open(", "read"])
                and any(x in cleaned for x in ["split", "len(", "words"])
                and any(x in cleaned for x in ["line", "print"])
            )
        elif test_id == "lang_ts_dom":
            return (
                "addeventlistener" in cleaned
                and any(x in cleaned for x in ["getelementbyid", "queryselector", "getelement"])
                and any(x in cleaned for x in ["button", "#go", "go"])
                and any(x in cleaned for x in ["time", "date"])
            )
        elif test_id == "lang_kotlin_ktor":
            return (
                "fun " in cleaned
                and any(x in cleaned for x in ["ktor", "routing", "route"])
                and any(x in cleaned for x in ["get(", "get"])
                and any(x in cleaned for x in ["ping", "pong"])
            )
        # ---- TV / App dev ----
        elif test_id == "tv_android_activity":
            return (
                "activity" in cleaned
                and any(x in cleaned for x in ["textview", "settext", "text"])
                and "button" in cleaned
                and any(x in cleaned for x in ["toast", "onclick", "click"])
                and any(x in cleaned for x in ["kotlin", "android", "import"])
            )
        elif test_id == "tv_android_tv":
            return (
                any(x in cleaned for x in ["leanback", "browse", "androidx.tv", "tv"])
                and any(x in cleaned for x in ["fragment", "presenter"])
                and any(x in cleaned for x in ["row", "card"])
            )
        elif test_id == "tv_roku_brightscript":
            return (
                any(x in cleaned for x in ["brightscript", "sub ", "function "])
                and any(x in cleaned for x in ["labellist", "rosgnode", "scenegraph", "scene"])
                and any(x in cleaned for x in ["onkeyevent", "select", "onkey"])
                and any(x in cleaned for x in ["xml", "component", "node"])
            )
        elif test_id == "tv_samsung_tizen":
            return (
                "tizen" in cleaned
                and any(x in cleaned for x in ["application", "launch", "launchapp"])
                and any(x in cleaned for x in ["index.html", "<script", "button"])
                and any(x in cleaned for x in ["api", "webapis"])
            )
        elif test_id == "tv_lg_webos":
            return (
                "webos" in cleaned
                and any(x in cleaned for x in ["appinfo", "appid", "app id"])
                and any(x in cleaned for x in ["launch", "addeventlistener", "webosservice", "service"])
                and any(x in cleaned for x in ["json", "init", "ready"])
            )
        # ---- UI/UX ----
        elif test_id == "uiux_accessibility":
            return (
                any(x in cleaned for x in ["wcag", "contrast", "aria"])
                and "label" in cleaned
                and "focus" in cleaned
                and any(x in cleaned for x in ["error", "alt", "message"])
            )
        elif test_id == "uiux_responsive":
            return (
                any(x in cleaned for x in ["@media", "media query", "media"])
                and any(x in cleaned for x in ["grid", "flex"])
                and any(x in cleaned for x in ["mobile", "column"])
                and "responsive" in cleaned
            )
        elif test_id == "uiux_color":
            return (
                "contrast" in cleaned
                and any(x in cleaned for x in ["wcag", "aa", "4.5"])
                and any(x in cleaned for x in ["ratio", "luminance", "relative"])
            )
        elif test_id == "uiux_wireframe":
            return (
                any(x in cleaned for x in ["wireframe", "ascii", "box"])
                and any(x in cleaned for x in ["sign", "email", "password"])
                and any(x in cleaned for x in ["button", "cta", "submit"])
            )
        elif test_id == "uiux_flow":
            return (
                "reset" in cleaned
                and "email" in cleaned
                and any(x in cleaned for x in ["link", "token"])
                and any(x in cleaned for x in ["confirm", "password"])
                and any(x in cleaned for x in ["fail", "error", "expir", "recover"])
            )
        elif test_id == "uiux_design_system":
            return (
                any(x in cleaned for x in [":root", "var(", "--"])
                and any(x in cleaned for x in ["clamp", "font", "typography", "rem", "px"])
                and any(x in cleaned for x in ["dark", "data-theme", "hsl", "oklch", "color"])
                and any(x in cleaned for x in ["btn", "button", "hover", "focus-visible", "focus"])
            )
        elif test_id == "uiux_accessible_modal":
            return (
                any(x in cleaned for x in ["dialog", 'role="dialog"', "role='dialog'"])
                and any(x in cleaned for x in ["aria-modal", "aria-labelledby", "aria-describedby"])
                and any(x in cleaned for x in ["escape", "keydown", "key =="])
                and any(x in cleaned for x in ["focus", "tabindex", "trap"])
            )
        elif test_id == "uiux_responsive_dashboard":
            return (
                any(x in cleaned for x in ["grid", "flex", "minmax"])
                and any(x in cleaned for x in ["sidebar", "nav", "menu"])
                and any(x in cleaned for x in ["card", "stat", "metric"])
                and any(x in cleaned for x in ["trend", "+", "-", "%", "pill", "badge"])
            )
        elif test_id == "uiux_form_ux_validation":
            return (
                any(x in cleaned for x in ["form", "input", "signup"])
                and any(x in cleaned for x in ["aria-describedby", "aria-invalid", "error"])
                and any(x in cleaned for x in ["strength", "meter", "password", "bar"])
                and any(x in cleaned for x in ["disabled", "loading", "spinner", "submit"])
            )
        elif test_id == "uiux_toast_notifications":
            return (
                any(x in cleaned for x in ["toast", "notification", "alert"])
                and any(x in cleaned for x in ["aria-live", 'role="status"', "role='status'", "status"])
                and any(x in cleaned for x in ["timer", "duration", "settimeout", "progress"])
                and any(x in cleaned for x in ["pause", "close", "slide", "fade", "transition"])
            )
        elif test_id == "uiux_bottom_sheet_drawer":
            return (
                any(x in cleaned for x in ["sheet", "drawer", "panel", "bottom"])
                and any(x in cleaned for x in ["backdrop", "overlay", "blur"])
                and any(x in cleaned for x in ["transform", "translate", "transition"])
                and any(x in cleaned for x in ["768", "@media", "mobile", "desktop"])
            )
        elif test_id == "uiux_theme_switcher":
            return (
                any(x in cleaned for x in ["theme", "palette", "mode", "color"])
                and any(x in cleaned for x in ["localstorage", "storage"])
                and any(x in cleaned for x in ["prefers-color-scheme", "dark", "light"])
                and any(x in cleaned for x in ["setproperty", "documentelement", "dataset", "var(--"])
            )
        elif test_id == "uiux_data_table":
            return (
                any(x in cleaned for x in ["table", "thead", "tbody"])
                and any(x in cleaned for x in ["sort", "aria-sort", "order"])
                and any(x in cleaned for x in ["filter", "search", "query"])
                and any(x in cleaned for x in ["empty", "no matching", "records", "sticky", "scroll"])
            )
        elif test_id == "uiux_stepper_wizard":
            return (
                any(x in cleaned for x in ["step", "wizard", "stepper"])
                and any(x in cleaned for x in ["aria-current", "progress", "circle"])
                and any(x in cleaned for x in ["next", "prev", "back"])
                and any(x in cleaned for x in ["active", "completed", "pending"])
            )
        elif test_id == "uiux_segmented_control":
            return (
                any(x in cleaned for x in ["tablist", 'role="tablist"', "role='tablist'"])
                and any(x in cleaned for x in ['role="tab"', "role='tab'", "aria-selected"])
                and any(x in cleaned for x in ["arrow", "key", "left", "right", "keydown"])
                and any(x in cleaned for x in ["indicator", "pill", "offsetleft", "bounding", "translate"])
            )
        # ---- Office productivity ----
        elif test_id == "office_email":
            return (
                "subject" in cleaned
                and any(x in cleaned for x in ["dear", "hi ", "hello", "team", "client"])
                and any(x in cleaned for x in ["delay", "apolog", "later", "postpone", "extend", "push", "slip"])
                and any(x in cleaned for x in ["date", "deliver", "schedul"])
            )
        elif test_id == "office_spreadsheet":
            return (
                any(x in cleaned for x in ["sum(", "=sum", "sum("])
                and any(x in cleaned for x in ["average(", "=average", "avg("])
                and any(x in cleaned for x in ["vlookup(", "=vlookup"])
                and any(x in cleaned for x in ["formula", "="])
            )
        elif test_id == "office_presentation":
            return (
                "slide" in cleaned
                and "title" in cleaned
                and "problem" in cleaned
                and "solution" in cleaned
                and any(x in cleaned for x in ["market", "ask", "bullet"])
            )
        elif test_id == "office_image_scale":
            return (
                any(x in cleaned for x in ["pillow", "image", "from pil", "pil"])
                and any(x in cleaned for x in ["resize", "thumbnail", "scale"])
                and any(x in cleaned for x in ["save", "jpeg", "quality"])
            )
        elif test_id == "office_logo":
            return (
                "svg" in cleaned
                and any(x in cleaned for x in ["circle", "badge", "round"])
                and any(x in cleaned for x in ["coffee", "cup", "bean", "brew"])
                and any(x in cleaned for x in ["bean", "brew", "text", "font"])
            )
        elif test_id == "office_text_edit":
            return (
                any(x in cleaned for x in ["rewrite", "revise", "edit", "proofread"])
                and any(x in cleaned for x in ["grammar", "clear", "concise", "tone", "polish"])
                and any(x in cleaned for x in ["paragraph", "sentence", "draft", "before", "after"])
            )
        elif test_id == "office_tts":
            return (
                any(x in cleaned for x in ["tts", "speech", "pyttsx3", "gtts"])
                and any(x in cleaned for x in ["save", "audio", "mp3", "wav"])
                and any(x in cleaned for x in ["say", "text", "engine", "voice"])
            )
        # ---- Life / dad-husband tasks ----
        elif test_id == "life_bedtime_story":
            return (
                any(x in cleaned for x in ["story", "fox", "sleep", "bedtime"])
                and any(x in cleaned for x in ["night", "moon", "sleep", "bed"])
                and any(x in cleaned for x in ["share", "friend", "kind", "calm"])
            )
        elif test_id == "life_dad_joke":
            return (
                any(x in cleaned for x in ["joke", "punchline"])
                and ("?" in cleaned or "why" in cleaned or "what" in cleaned or "how" in cleaned)
                and any(x in cleaned for x in ["because", "knock", "dad", "groan"])
            )
        elif test_id == "life_cars":
            return (
                any(x in cleaned for x in ["timing belt", "belt"])
                and any(x in cleaned for x in ["engine", "valve", "piston", "cam"])
                and any(x in cleaned for x in ["mile", "100k", "replace", "interval"])
            )
        elif test_id == "life_computers":
            return (
                any(x in cleaned for x in ["ram", "memory"])
                and any(x in cleaned for x in ["storage", "disk", "drive"])
                and any(x in cleaned for x in ["task manager", "settings", "step", "free"])
            )
        elif test_id == "life_gardening":
            return (
                any(x in cleaned for x in ["garden", "plant", "container"])
                and any(x in cleaned for x in ["soil", "pot"])
                and any(x in cleaned for x in ["water", "sun", "light"])
                and any(x in cleaned for x in ["step", "1", "2", "3"])
            )
        elif test_id == "life_animals":
            return (
                any(x in cleaned for x in ["chicken", "hen", "coop"])
                and any(x in cleaned for x in ["feed", "food"])
                and any(x in cleaned for x in ["egg", "family", "four"])
            )
        elif test_id == "life_home_maint":
            return (
                any(x in cleaned for x in ["gutter", "hvac", "filter", "smoke", "maintenance"])
                and any(x in cleaned for x in ["spring", "fall", "season"])
                and any(x in cleaned for x in ["check", "list", "inspect"])
            )
        elif test_id == "life_chores":
            return (
                any(x in cleaned for x in ["chore", "task"])
                and any(x in cleaned for x in ["kid", "child", "age"])
                and any(x in cleaned for x in ["reward", "allowance", "sticker"])
            )
        elif test_id == "life_money":
            return (
                any(x in cleaned for x in ["budget", "50/30/20", "50 30 20"])
                and any(x in cleaned for x in ["save", "saving", "needs", "wants"])
                and any(x in cleaned for x in ["4000", "example", "take-home", "income"])
            )
        # ---- Biblical / Ancient Near East ----
        elif test_id == "bib_ot_knowledge":
            return (
                "covenant" in cleaned
                and any(x in cleaned for x in ["noah", "abraham", "mosaic", "david"])
                and any(x in cleaned for x in ["promise", "law", "land", "bless"])
            )
        elif test_id == "bib_ane_tradition":
            return (
                any(x in cleaned for x in ["genesis", "near east", "mesopotamia", "ane"])
                and any(x in cleaned for x in ["creation", "flood", "law", "code"])
                and any(x in cleaned for x in ["culture", "compare", "babylon", "sumer", "paralle"])
            )
        elif test_id == "bib_2nd_temple":
            return (
                any(x in cleaned for x in ["pharisee", "sadducee", "essene"])
                and any(x in cleaned for x in ["temple", "2nd temple", "second temple"])
                and any(x in cleaned for x in ["kingdom of god", "new testament", "gospel", "audience"])
            )
        # ---- Metacognition: overthinking + loop detection ----
        elif test_id == "meta_overthinking":
            # Correct answer (4) AND concise (resists adding reasoning/caveats)
            return "4" in cleaned and len(cleaned.split()) <= 40
        elif test_id == "meta_loop_detect":
            return (
                any(x in cleaned for x in ["terminate", "infinite", "forever", "never end", "loop forever"])
                and any(x in cleaned for x in ["5", "four", "4"])
                and any(x in cleaned for x in ["while", "loop", "condition"])
            )
        # ---- Networking ----
        elif test_id == "net_echo_server":
            return (
                any(x in cleaned for x in ["bind", "listen", "accept"])
                and any(x in cleaned for x in ["recv", "sendall", "conn.send"])
                and any(x in cleaned for x in ["socket.socket", "import socket"])
            )
        elif test_id == "net_http_client":
            return any(x in cleaned for x in ["urllib.request", "http.client", "requests.get", "http.request"]) and any(
                x in cleaned for x in ["status", "code", "getcode", "print"]
            )
        elif test_id == "net_dns_resolver":
            return any(x in cleaned for x in ["gethostbyname", "getaddrinfo", "dnspython", "socket"]) and any(
                x in cleaned for x in ["print", "ip", "address"]
            )
        elif test_id == "net_port_scanner":
            return (
                any(x in cleaned for x in ["socket.socket", "connect_ex", "import socket"])
                and any(x in cleaned for x in ["port", "open"])
                and any(x in cleaned for x in ["for", "threading", "range"])
            )
        # ---- USB device programming ----
        elif test_id in ("usb_interface_claim", "usb_hidraw_read"):
            return any(x in cleaned for x in ["usbdevfs", "ioctl", "hidraw", "usb"]) and any(
                x in cleaned for x in ["#include", "fd", "open("]
            )
        elif test_id == "usb_descriptor_dump":
            return any(x in cleaned for x in ["descriptor", "endpoint", "interface"]) and any(
                x in cleaned for x in ["#include", "usb", "print"]
            )
        elif test_id == "usb_hid_parser":
            return any(x in cleaned for x in ["report", "usage", "hid", "struct"]) and any(
                x in cleaned for x in ["#include", "usb", "parse", "print"]
            )
        # ---- Infrastructure as code (Terraform / Pulumi / GH Actions) ----
        elif test_id == "iac_tf_vpc":
            return any(x in cleaned for x in ["aws_vpc", "subnet", "security_group"]) and any(
                x in cleaned for x in ["resource", "provider", "terraform"]
            )
        elif test_id == "iac_tf_ec2":
            return any(x in cleaned for x in ["aws_instance", "resource"]) and any(
                x in cleaned for x in ["ami", "instance_type", "provider"]
            )
        elif test_id == "iac_gha_ci":
            return any(x in cleaned for x in ["on:", "jobs:", "steps:"]) and any(
                x in cleaned for x in ["runs-on", "actions/checkout", "uses:"]
            )
        elif test_id == "iac_pulumi_s3":
            return any(x in cleaned for x in ["aws:s3", "s3/bucket", "resources"]) and any(
                x in cleaned for x in ["name:", "runtime:", "yaml"]
            )
        # ---- Linux kernel driver ----
        elif test_id == "ldrv_char_device":
            return (
                any(x in cleaned for x in ["register_chrdev", "alloc_chrdev_region", "cdev"])
                and any(x in cleaned for x in ["file_operations", ".open", ".read", ".write"])
                and any(x in cleaned for x in ["module_init", "module_exit"])
            )
        elif test_id == "ldrv_platform_driver":
            return any(x in cleaned for x in ["platform_driver", "probe", "remove"]) and any(
                x in cleaned for x in ["module_init", "module_exit", "of_match_table"]
            )
        elif test_id == "ldrv_ioctl":
            return any(x in cleaned for x in ["ioctl", "copy_to_user", "copy_from_user"]) and any(
                x in cleaned for x in ["_io", "_ior", "_iow", "unlocked_ioctl", "file_operations"]
            )
        elif test_id == "ldrv_miscdevice":
            return any(x in cleaned for x in ["misc_register", "miscdevice"]) and any(
                x in cleaned for x in [".name", ".fops", "file_operations"]
            )
        # ---- Bash scripting ----
        elif test_id == "bash_backup_rotate":
            return (
                any(x in cleaned for x in ["tar", "backup", "cp "])
                and any(x in cleaned for x in ["ls", "head", "find", "for"])
                and "#!/bin/bash" in cleaned
            )
        elif test_id == "bash_csv_sums":
            return (
                any(x in cleaned for x in ["awk", "cut", "while read"])
                and any(x in cleaned for x in ["sum", "total"])
                and "#!/bin/bash" in cleaned
            )
        elif test_id == "bash_health_loop":
            return (
                any(x in cleaned for x in ["curl", "http_code", "wget"])
                and any(x in cleaned for x in ["sleep", "while", "for"])
                and "#!/bin/bash" in cleaned
            )
        elif test_id == "bash_top_cpu":
            return (
                any(x in cleaned for x in ["ps ", "top", "awk"])
                and any(x in cleaned for x in ["cpu", "%cpu"])
                and "#!/bin/bash" in cleaned
            )
        # ---- BASIC (yabasic) ----
        elif test_id == "bas_guess_game":
            return (
                any(x in cleaned for x in ["ran(", "int(ran", "random"])
                and any(x in cleaned for x in ["input", "too high", "too low", "higher", "lower"])
                and any(x in cleaned for x in ["loop", "while", "for"])
            )
        elif test_id == "bas_fibonacci":
            return (
                any(x in cleaned for x in ["fibonacci", "fib"])
                and any(x in cleaned for x in ["for", "next"])
                and any(x in cleaned for x in ["print"])
            )
        elif test_id == "bas_grade_calc":
            return (
                any(x in cleaned for x in ["input", "grade", "score"])
                and any(x in cleaned for x in ["if", "then", "else"])
                and any(x in cleaned for x in ["print", "end"])
            )
        elif test_id == "bas_countdown":
            return any(x in cleaned for x in ["for", "next", "down"]) and any(x in cleaned for x in ["print", "sleep"])
        # ---- Pascal (Free Pascal) ----
        elif test_id == "pas_records":
            return any(x in cleaned for x in ["type", "record", "array"]) and "program" in cleaned and "end." in cleaned
        elif test_id == "pas_factorial":
            return (
                any(x in cleaned for x in ["function", "factorial", "fact"])
                and any(x in cleaned for x in ["recursive", "if", "begin"])
                and "end." in cleaned
            )
        elif test_id == "pas_bubble_sort":
            return (
                any(x in cleaned for x in ["array", "sort", "bubble"])
                and any(x in cleaned for x in ["for", "to", "do"])
                and "end." in cleaned
            )
        elif test_id == "pas_word_count":
            return (
                any(x in cleaned for x in ["readln", "word", "count"])
                and any(x in cleaned for x in ["while", "repeat", "for"])
                and "end." in cleaned
            )
        # ---- TypeScript ----
        elif test_id in ("ts_shapes", "ts_fib_memo", "ts_string_utils", "ts_json_parse"):
            return any(x in cleaned for x in [": string", ": number", "interface", "class", "function"]) and any(
                x in cleaned for x in ["console.log", "return", "=>"]
            )
        # ---- RPM spec files ----
        elif test_id in ("rpm_minimal_spec", "rpm_build_spec", "rpm_devel_subpackage", "rpm_doc_config_spec"):
            return any(x in cleaned for x in ["name:", "version:", "release:", "summary:"]) and any(
                x in cleaned for x in ["%description", "%install", "%files", "%build", "%prep"]
            )
        # ---- Android (Kotlin) ----
        elif test_id == "and_main_activity":
            return (
                any(x in cleaned for x in ["oncreate", "setcontentview", "recyclerview"])
                and any(x in cleaned for x in ["appcompatactivity", "activity", "adapter", "viewholder"])
                and any(x in cleaned for x in ["fun ", "class ", "import"])
            )
        elif test_id == "and_viewmodel":
            return any(x in cleaned for x in ["viewmodel", "livedata", "stateflow", "mutablelivedata"]) and any(
                x in cleaned for x in ["class", "fun", "private", "val"]
            )
        elif test_id == "and_retrofit":
            return any(x in cleaned for x in ["retrofit", "@get", "@post", "interface", "call<"]) and any(
                x in cleaned for x in ["class", "fun", "import"]
            )
        elif test_id == "and_room":
            return any(x in cleaned for x in ["@database", "@dao", "@entity", "@query", "@insert"]) and any(
                x in cleaned for x in ["interface", "class", "fun"]
            )

        # Unknown / custom test_ids from BENCHMARK_TESTS_JSON have no built-in
        # expectations. Use a minimal content-quality gate instead of an
        # unconditional pass so empty or garbage responses cannot score.
        if len(cleaned) < 30:
            return False
        return len(cleaned.split()) >= 8

    # ---- Code-quality + AI-watermark scoring ---------------------------------
    # These run for every test that produced a response so the leaderboard can
    # rank models not only on correctness but on how clean / "human" the code is.

    _WATERMARK_PHRASES: ClassVar[list[str]] = [
        "certainly",
        "here is",
        "here's",
        "i hope this helps",
        "as an ai",
        "feel free to",
        "let me know if",
        "great question",
        "absolutely",
        "of course",
        "sure,",
        "happy to help",
        "in summary",
        "to summarize",
    ]

    # Default rubric applied to every code/UI benchmark (in addition to any
    # per-test "rubric" list in benchmark_tests.json). These are cheap,
    # unambiguous checks for things every code prompt asks for: a fenced code
    # block, a real runnable program rather than placeholder stubs, and a
    # self-contained entry point / demo. Missing any of them costs points even
    # when the code happens to run.
    _DEFAULT_CODE_RUBRIC: ClassVar[list[dict]] = [
        {
            "label": "Code is in a fenced block",
            "check": ["```"],
            "match": "any",
            "points": 8,
        },
        {
            "label": "No placeholder/stub text",
            "check": ["todo", "your code here", "placeholder", "fixme", "insert code", "xxx", "implement this"],
            "match": "absent",
            "points": 8,
        },
        {
            "label": "Has a self-contained entry point / demo",
            "check": [
                "def main",
                "if __name__",
                "main()",
                "function main",
                "int main",
                "func main",
                "fn main",
                "public static void main",
                "app.mainloop",
                "pygame.display.set_mode",
                "display.set_mode",
                "requestanimationframe",
                "setinterval",
                "console.log",
                "writeln(",
                "printf(",
                "fmt.println",
                "echo ",
                "print ",
                "print(",
                "program ",
                "#!/bin",
            ],
            "match": "any",
            "points": 8,
        },
        {
            "label": "Code is complete, not truncated",
            "check": ["...", "…"],
            "match": "absent_trailing",
            "points": 4,
        },
    ]

    def _score_code_quality(self, response: str, test: dict) -> dict:
        """Heuristic code-quality score (0-100) plus a syntax_valid flag.

        Does not execute untrusted code; it only statically checks fenced
        code blocks (Python is parsed with ast for a real syntax check).
        """
        notes: list[str] = []
        has_fence = "```" in response
        code = response
        if has_fence:
            blocks = re.findall(r"```[a-zA-Z0-9_+-]*\n(.*?)```", response, flags=re.DOTALL)
            if blocks:
                code = "\n".join(blocks)
        code = code.strip()
        low = code.lower()

        # Infer language
        lang = ""
        cat = (test.get("category") or "") if isinstance(test, dict) else ""
        if cat in ("cpp",):
            lang = "cpp"
        elif cat in ("java",):
            lang = "java"
        elif cat in ("languages", "webdev", "tvdev", "uiux"):
            lang = ""
        if not lang:
            if "def " in low or "import " in low or "print(" in low:
                lang = "python"
            elif "package main" in low or "func " in low:
                lang = "go"
            elif "fn main" in low:
                lang = "rust"
            elif "public class" in low or "public static void" in low:
                lang = "java"
            elif "#include" in low:
                lang = "cpp"
            elif "<?php" in low:
                lang = "php"
            elif "<!doctype" in low or "<html" in low:
                lang = "html"
            elif "addeventlistener" in low or "const " in low or "let " in low or "function " in low:
                lang = "js"

        score = 55
        if has_fence:
            score += 15
            notes.append("code fenced")
        if re.search(r"\b(def |function |func |class |struct |interface |pub |fn )", code):
            score += 12
            notes.append("has definitions")
        if '"""' in code or "'''" in code or re.search(r"#.*|//.*", code):
            score += 8
            notes.append("has comments/docstrings")
        if len(code.splitlines()) >= 8:
            score += 5
            notes.append("substantial length")

        # Penalties
        placeholders = ["todo", "your code here", "fixme", "placeholder", "insert code", "xxx"]
        if any(p in low for p in placeholders):
            score -= 25
            notes.append("contains placeholder text")
        # Truncated / unbalanced braces
        opens = code.count("{")
        closes = code.count("}")
        if opens and closes and abs(opens - closes) >= 2:
            score -= 12
            notes.append("unbalanced braces (possibly truncated)")
        if low.rstrip().endswith(("```", "...")):
            score -= 8
            notes.append("appears truncated")

        syntax_valid = None
        if lang == "python" and code:
            try:
                # Use the SAME extraction the sandbox executes so syntax_valid
                # reflects the code that actually ran, not a mangled join of
                # every fenced block (leaked thinking drafts can carry stray
                # indentation that would otherwise cause a false alarm).
                parsed = extract_clean_code(response, "python") or code
                ast.parse(parsed)
                syntax_valid = True
                notes.append("python syntax valid")
            except SyntaxError:
                syntax_valid = False
                score -= 15
                notes.append("python syntax invalid")

        score = max(0, min(100, score))
        return {
            "score": score,
            "language": lang,
            "syntax_valid": syntax_valid,
            "has_code": has_fence or bool(code),
            "notes": notes,
        }

    def _score_watermark(self, response: str) -> dict:
        """Detect AI 'watermarks' / signatures. score: 0 = clean, 100 = heavy."""
        flags: list[str] = []
        low = response.lower()
        # Em / en dashes and box-drawing characters (built from codepoints so this
        # source file itself stays free of those glyphs).
        em_dash = chr(0x2014)
        en_dash = chr(0x2013)
        box_chars = "".join(
            chr(c)
            for c in (
                0x2500,
                0x2502,
                0x250C,
                0x2510,
                0x2514,
                0x2518,
                0x251C,
                0x2524,
                0x252C,
                0x2534,
                0x253C,
                0x25BA,
                0x25B6,
                0x25A0,
                0x25CF,
            )
        )
        if em_dash in response or en_dash in response:
            flags.append("em/en dash")
        if any(ch in response for ch in box_chars):
            flags.append("box-drawing glyph")
        for ph in self._WATERMARK_PHRASES:
            if ph in low:
                flags.append(f"phrase:{ph}")
        # Excessive emoji
        emoji = re.findall(r"[\U0001F000-\U0001FAFF\u2600-\u27BF]", response)
        if len(emoji) >= 3:
            flags.append("excessive emoji")
        score = min(100, len(flags) * 18)
        return {"score": score, "flags": flags}

    def _evaluate_rubric(self, test: dict, response: str, code: str = "") -> dict:
        """Score a response against the benchmark's rubric criteria.

        Criteria come from the test's ``rubric`` list in benchmark_tests.json
        (per-test additions, e.g. "game must persist a top-5 high-score table")
        layered on top of the always-applied ``_DEFAULT_CODE_RUBRIC`` for code/UI
        tests. Each criterion checks the response (and/or the extracted code) for
        a required substring (``match: any``), for every required substring
        (``match: all``), for the ABSENCE of a forbidden substring
        (``match: absent``), or for a truncated tail (``match: absent_trailing``).

        Returns ``{score, fraction, total_points, earned_points, criteria: [...]}``
        where each criterion carries ``label``, ``passed``, ``points`` so the
        dashboard can show exactly which requested features a model delivered or
        missed.
        """
        criteria = list(self._DEFAULT_CODE_RUBRIC)
        for extra in test.get("rubric") or []:
            if isinstance(extra, dict) and extra.get("label"):
                criteria.append(extra)

        haystack = f"{response}\n{code}" if code else response
        low = haystack.lower()
        total_points = 0
        earned_points = 0
        results: list[dict] = []
        for crit in criteria:
            points = int(crit.get("points") or 1)
            checks = crit.get("check") or []
            if isinstance(checks, str):
                checks = [checks]
            match = (crit.get("match") or "any").lower()
            if match == "absent":
                passed = not any(c.lower() in low for c in checks)
            elif match == "absent_trailing":
                passed = not (response.rstrip().endswith(("...", "```")))
            elif match == "all":
                passed = all(c.lower() in low for c in checks)
            else:  # any
                passed = any(c.lower() in low for c in checks)
            total_points += points
            if passed:
                earned_points += points
            results.append(
                {
                    "label": crit.get("label") or "Untitled criterion",
                    "passed": bool(passed),
                    "points": points,
                    "checks": [str(c) for c in checks],
                }
            )
        fraction = (earned_points / total_points) if total_points else 1.0
        return {
            "score": round(100 * fraction),
            "fraction": fraction,
            "total_points": total_points,
            "earned_points": earned_points,
            "criteria": results,
        }

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

    async def test_model_proxy(self, model: str, test: dict, sampler: ResourceSampler | None = None) -> dict:
        """Test a model against a proxy endpoint or online provider."""
        if online_model_provider.is_online_model(model):
            try:
                # Free-tier online providers throttle aggressively (e.g. Gemini
                # free tier = 20 req/min). Pace requests so a fast local run does
                # not blow through the whole quota window in seconds, which used
                # to trigger cascade 429s that aborted the entire model run.
                self._pace_online_request(model)
                if sampler:
                    sampler.start()
                res = await online_model_provider.query_online_model(
                    model_identifier=model,
                    prompt=test["prompt"],
                    max_tokens=_test_num_predict(test, model),
                )
                if sampler:
                    await sampler.stop()
            except Exception as e:
                if sampler:
                    await sampler.stop()
                latency = 0.0
                return {
                    "proxy": "online",
                    "success": False,
                    "prompt": test["prompt"],
                    "latency": 0.0,
                    "response": None,
                    "tokens_generated": 0,
                    "eval_duration": 0,
                    "prompt_eval_duration": 0,
                    "error": f"online provider error: {e}",
                }
            latency = res.get("latency", 0.0)
            resp_raw = res.get("response") or ""
            response_text, thinking = self.strip_thinking(resp_raw)
            if thinking:
                resp_raw = response_text
            return {
                "proxy": "online",
                "success": res.get("success", False),
                "prompt": test["prompt"],
                "latency": round(latency, 3),
                "response": resp_raw,
                "thinking": thinking or res.get("thinking"),
                "tokens_generated": res.get("tokens_generated", 0),
                "eval_duration": int(latency * 1e9),
                "prompt_eval_duration": 0,
                "error": res.get("error"),
                "retry_after": res.get("retry_after"),
            }

        last_error: Exception | None = None
        try:
            think_on = _test_thinking(test, model)
            reasoning_budget = _test_reasoning_budget(test, model)
            num_predict = await self._effective_num_predict(model, test["prompt"], _test_num_predict(test, model))
        except RuntimeError as e:
            return {
                "proxy": "context-guard",
                "success": False,
                "prompt": test["prompt"],
                "latency": 0,
                "response": None,
                "tokens_generated": 0,
                "error": str(e),
            }
        if num_predict <= 0:
            return {
                "proxy": "context-guard",
                "success": False,
                "prompt": test["prompt"],
                "latency": 0,
                "response": None,
                "tokens_generated": 0,
                "error": (
                    "Context window exhausted before generation. Reduce the prompt size or "
                    "increase the backend ctx-size."
                ),
            }
        for proxy_url in self.PROXY_SERVER_URLS:
            try:
                start_t = time.time()
                async with httpx.AsyncClient(timeout=_STREAM_TIMEOUT) as client:
                    if sampler:
                        sampler.start()
                    payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": test["prompt"]}],
                        "stream": True,
                        "think": think_on,
                        "options": {
                            "num_predict": num_predict,
                            "temperature": _model_temperature(model),
                            **_model_sampling_options(model),
                        },
                    }
                    if reasoning_budget is not None:
                        payload["reasoning_budget"] = reasoning_budget
                    headers = self._proxy_headers({"X-Request-Source": "shared-llm/benchmark"})
                    async with client.stream(
                        "POST", f"{proxy_url}/api/chat", json=payload, headers=headers
                    ) as response:
                        if response.status_code != 200:
                            body = (await response.aread()).decode("utf-8", "replace")
                            if sampler:
                                await sampler.stop()
                            return {
                                "proxy": proxy_url,
                                "success": False,
                                "prompt": test["prompt"],
                                "latency": 0,
                                "response": None,
                                "tokens_generated": 0,
                                "error": f"HTTP {response.status_code}: {body}",
                            }
                        data = await _read_chat_stream(response)
                    elapsed = time.time() - start_t
                    if sampler:
                        await sampler.stop()

                    eval_ns = data["eval_duration"]
                    prompt_ns = data["prompt_eval_duration"]
                    eval_count = data["eval_count"]
                    response_text, thinking = self.strip_thinking(data["content"])
                    # Reasoning models emit the thinking phase as separate streamed
                    # message.thinking chunks when --reasoning-format deepseek is in
                    # effect, so capture those too (they are already separate from
                    # the code we score).
                    if not thinking and data.get("thinking"):
                        thinking = data["thinking"]

                    # Guard: empty generation - server returned 200 but produced nothing.
                    # Skip the nudge retry (it would hang) and fail immediately.
                    if not response_text and eval_count == 0:
                        return {
                            "proxy": proxy_url,
                            "success": False,
                            "prompt": test["prompt"],
                            "latency": round(elapsed, 3),
                            "response": None,
                            "tokens_generated": 0,
                            "error": "Empty generation (eval_count=0, no content returned)",
                        }

                    # If we exhausted the token cap, inject nudge and request remaining tokens
                    if eval_count >= num_predict:
                        try:
                            payload2 = {
                                "model": model,
                                "messages": [
                                    {"role": "user", "content": test["prompt"]},
                                    {"role": "assistant", "content": response_text},
                                    {
                                        "role": "user",
                                        "content": "[System: You are halfway through your token budget. Please come up with an answer quickly.]",
                                    },
                                ],
                                "stream": True,
                                "think": think_on,
                                "options": {
                                    "num_predict": num_predict,
                                    "temperature": _model_temperature(model),
                                    **_model_sampling_options(model),
                                },
                            }
                            if reasoning_budget is not None:
                                payload2["reasoning_budget"] = reasoning_budget
                            start_t2 = time.time()
                            async with client.stream(
                                "POST", f"{proxy_url}/api/chat", json=payload2, headers=headers
                            ) as response2:
                                if response2.status_code == 200:
                                    data2 = await _read_chat_stream(response2)
                                    response_text2, thinking2 = self.strip_thinking(data2["content"])
                                    response_text = response_text + "\n" + response_text2
                                    if thinking2:
                                        thinking = (thinking + "\n\n" + thinking2) if thinking else thinking2
                                    elif data2.get("thinking"):
                                        thinking = data2["thinking"]
                                    eval_count += data2["eval_count"]
                                    eval_ns += data2["eval_duration"]
                                    prompt_ns += data2["prompt_eval_duration"]
                            elapsed += time.time() - start_t2
                        except Exception as e2:
                            print(f"Phase 2 proxy query error: {e2}")

                    latency = (eval_ns + prompt_ns) / 1e9 if (eval_ns or prompt_ns) else elapsed
                    return {
                        "proxy": proxy_url,
                        "success": True,
                        "prompt": test["prompt"],
                        "latency": round(latency, 3),
                        "response": response_text,
                        "thinking": thinking,
                        "think": think_on,
                        "reasoning_budget": reasoning_budget,
                        "num_predict": num_predict,
                        "tokens_generated": eval_count,
                        "eval_duration": eval_ns,
                        "prompt_eval_duration": prompt_ns,
                        "error": None,
                    }
            except (httpx.RemoteProtocolError, httpx.ReadError) as e:
                # Server crashed or dropped the connection - fail immediately
                # instead of waiting out the full read timeout.
                if sampler:
                    await sampler.stop()
                last_error = e
                continue
            except Exception as e:
                if sampler:
                    await sampler.stop()
                last_error = e
                continue
        error_msg = ""
        if last_error:
            if isinstance(last_error, (httpx.TimeoutException, asyncio.TimeoutError)):
                error_msg = f"Request timed out: {type(last_error).__name__}"
            else:
                error_msg = (
                    f"{type(last_error).__name__}: {last_error!s}" if str(last_error) else type(last_error).__name__
                )
        else:
            error_msg = "Unknown error"

        return {
            "proxy": "all_failed",
            "success": False,
            "prompt": test["prompt"],
            "latency": 0,
            "response": None,
            "tokens_generated": 0,
            "error": error_msg,
        }

    # ------------------------------------------------------------------ #
    # Tool / media-service benchmarks                                     #
    # ------------------------------------------------------------------ #

    AUDIO_SERVER_URL: ClassVar[str] = os.getenv("AUDIO_SERVER_URL", "http://localhost:8082")

    def _service_urls(self) -> tuple[str, str]:
        """Resolve (proxy_url, audio_url) honoring container/host environments."""
        proxy = os.getenv("PROXY_SERVER_URLS", "http://localhost:11434,http://alpaca-proxy:11434")
        proxy_url = proxy.split(",")[0].strip()
        return proxy_url, self.AUDIO_SERVER_URL

    async def _tool_image(self, test: dict) -> dict:
        """Generate an image through the alpaca proxy and grade the result.

        Graded dimensions: request success, PNG decodes, exact requested size,
        non-blank pixel variance, and wall-clock time within the test budget.
        """
        import base64 as b64mod

        import httpx

        params = test.get("params") or {}
        prompt = params.get("prompt") or test.get("prompt") or ""
        size = str(params.get("size", "512x512"))
        payload = {
            "model": params.get("model", ""),
            "prompt": prompt,
            "n": 1,
            "size": size,
            "steps": int(params.get("steps", 20)),
            "response_format": "b64_json",
        }
        if params.get("seed") is not None:
            payload["seed"] = int(params["seed"])
        if params.get("cfg_scale") is not None:
            payload["cfg_scale"] = float(params["cfg_scale"])

        proxy_url, _ = self._service_urls()
        t0 = time.perf_counter()
        meta: dict = {"size": size, "steps": payload["steps"], "prompt": prompt[:120]}
        criteria: list[dict] = []

        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(f"{proxy_url}/v1/images/generations", json=payload)
        except Exception as e:
            return {
                "success": False,
                "score": 0,
                "tool_score": 0,
                "error": f"image service unreachable: {e}",
                "meta": meta,
                "criteria": [{"name": "service_reachable", "pass": False}],
                "tokens_generated": None,
            }
        elapsed = time.perf_counter() - t0
        meta["elapsed_s"] = round(elapsed, 1)
        reachable = resp.status_code == 200
        criteria.append({"name": "service_reachable", "pass": reachable})
        if not reachable:
            detail = ""
            try:
                detail = str(resp.json())[:200]
            except Exception:
                detail = resp.text[:200]
            meta["http_status"] = resp.status_code
            return {
                "success": False,
                "score": 0,
                "tool_score": 0,
                "error": f"image generation failed HTTP {resp.status_code}: {detail}",
                "meta": meta,
                "criteria": criteria,
                "tokens_generated": None,
            }

        data = (resp.json() or {}).get("data") or []
        b64img = (data[0] or {}).get("b64_json") if data else None
        got_image = bool(b64img)
        criteria.append({"name": "image_returned", "pass": got_image})

        dims_ok = variance_ok = False
        w_px = h_px = 0
        if got_image:
            try:
                import io as _io

                from PIL import Image, ImageStat

                img = Image.open(_io.BytesIO(b64mod.b64decode(b64img or "")))
                w_px, h_px = img.size
                want_w, want_h = (int(x) for x in size.lower().split("x"))
                dims_ok = (w_px, h_px) == (want_w, want_h)
                stat = ImageStat.Stat(img.convert("L"))
                stddev = float(stat.stddev[0])
                meta["stddev"] = round(stddev, 2)
                variance_ok = stddev > 8.0  # a flat/blank render has ~0 variance
            except Exception as e:
                meta["decode_error"] = str(e)[:150]
        criteria.append({"name": f"dims_match_{w_px}x{h_px}", "pass": dims_ok})
        criteria.append({"name": "non_blank_render", "pass": variance_ok})

        budget = float(test.get("time_budget_s", 300))
        speed_ok = elapsed <= budget
        criteria.append({"name": f"time_within_{budget:g}s", "pass": speed_ok})

        score = round(100 * sum(1 for c in criteria if c["pass"]) / len(criteria))
        errors = [c["name"] for c in criteria if not c["pass"]]
        return {
            "success": got_image and dims_ok and variance_ok,
            "score": score,
            "tool_score": score,
            "response": f"[image {w_px}x{h_px} in {elapsed:.1f}s via sd-server]",
            "error": "" if not errors else f"failed: {', '.join(errors)}",
            "meta": meta,
            "criteria": criteria,
            "tokens_generated": None,
            "artifact_b64": b64img,
            "artifact_mime": "image/png" if b64img else None,
        }

    @staticmethod
    def _wav_stats(raw: bytes) -> dict:
        """Parse WAV bytes with the stdlib and compute duration/RMS/zcr stats."""
        import array
        import io as _io

        out: dict = {}
        try:
            with wave.open(_io.BytesIO(raw), "rb") as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                width = wf.getsampwidth()
                out["sample_rate"] = sr
                out["duration_s"] = round(n_frames / float(sr or 1), 2)
                frames = wf.readframes(n_frames)
        except Exception as e:
            out["error"] = f"invalid wav: {e}"
            return out
        if width == 2:
            samples = array.array("h")
            samples.frombytes(frames[: (len(frames) // 2) * 2])
            if samples:
                acc = 0
                for s in samples:
                    acc += int(s) * int(s)
                rms = (acc / len(samples)) ** 0.5 / 32768.0
                out["rms"] = round(rms, 5)
                crossings = sum(1 for i in range(1, len(samples)) if (samples[i - 1] < 0) != (samples[i] < 0))
                out["zcr"] = round(crossings / float(len(samples)), 4)
        return out

    async def _tool_tts(self, test: dict) -> dict:
        """Synthesize speech through audio-server and grade the WAV output."""
        import base64 as b64mod

        import httpx

        params = test.get("params") or {}
        text = params.get("text") or test.get("prompt") or ""
        body = {
            "text": text,
            "voice": params.get("voice", "af_heart"),
            "speed": float(params.get("speed", 1.0)),
        }
        _, audio_url = self._service_urls()
        t0 = time.perf_counter()
        meta: dict = {"voice": body["voice"], "chars": len(text)}
        criteria: list[dict] = []
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(f"{audio_url}/api/tts", json=body)
        except Exception as e:
            return {
                "success": False,
                "score": 0,
                "tool_score": 0,
                "error": f"audio service unreachable: {e}",
                "meta": meta,
                "criteria": [{"name": "service_reachable", "pass": False}],
                "tokens_generated": None,
            }
        elapsed = time.perf_counter() - t0
        meta["elapsed_s"] = round(elapsed, 1)

        ok = resp.status_code == 200 and not (resp.json() or {}).get("error")
        criteria.append({"name": "service_reachable", "pass": ok})
        if not ok:
            try:
                detail = str(resp.json())[:200]
            except Exception:
                detail = resp.text[:200]
            return {
                "success": False,
                "score": 0,
                "tool_score": 0,
                "error": f"TTS failed HTTP {resp.status_code}: {detail}",
                "meta": meta,
                "criteria": criteria,
                "tokens_generated": None,
            }

        data = resp.json() or {}
        raw = b64mod.b64decode(data.get("audio_b64") or "")
        stats = self._wav_stats(raw)
        meta.update({k: v for k, v in stats.items() if k != "error"})
        duration = float(stats.get("duration_s", 0))
        expected_lo, expected_hi = float(params.get("min_duration_s", 1.0)), float(params.get("max_duration_s", 600))
        dur_ok = expected_lo <= duration <= expected_hi
        criteria.append({"name": f"duration_{duration}s_in_range", "pass": dur_ok})

        rms = float(stats.get("rms", 0))
        audible = rms >= float(params.get("min_rms", 0.01))
        criteria.append({"name": f"audible_rms_{rms}", "pass": audible})

        rtf = float((data.get("meta") or {}).get("rtf", 99))
        fast_ok = rtf <= float(params.get("max_rtf", 3.0))
        criteria.append({"name": f"rtf_{rtf}_within_budget", "pass": fast_ok})

        score = round(100 * sum(1 for c in criteria if c["pass"]) / len(criteria))
        errors = [c["name"] for c in criteria if not c["pass"]]
        return {
            "success": ok and dur_ok and audible,
            "score": score,
            "tool_score": score,
            "response": f"[speech {duration}s voice={body['voice']} rtf={rtf}]",
            "error": "" if not errors else f"failed: {', '.join(errors)}",
            "meta": meta,
            "criteria": criteria,
            "tokens_generated": None,
            "artifact_b64": data.get("audio_b64"),
            "artifact_mime": "audio/wav",
        }

    async def _tool_music(self, test: dict) -> dict:
        """Generate a music clip through audio-server and grade it."""
        import base64 as b64mod

        import httpx

        params = test.get("params") or {}
        prompt = params.get("prompt") or test.get("prompt") or ""
        requested = float(params.get("duration_s", 10))
        body: dict = {
            "prompt": prompt,
            "duration_s": requested,
        }
        if params.get("temperature") is not None:
            body["temperature"] = float(params["temperature"])
        if params.get("guidance_scale") is not None:
            body["guidance_scale"] = float(params["guidance_scale"])
        if params.get("seed") is not None:
            body["seed"] = int(params["seed"])

        _, audio_url = self._service_urls()
        t0 = time.perf_counter()
        meta: dict = {"requested_duration_s": requested, "prompt": prompt[:120]}
        criteria: list[dict] = []
        try:
            async with httpx.AsyncClient(timeout=900.0) as client:
                resp = await client.post(f"{audio_url}/api/music", json=body)
        except Exception as e:
            return {
                "success": False,
                "score": 0,
                "tool_score": 0,
                "error": f"audio service unreachable: {e}",
                "meta": meta,
                "criteria": [{"name": "service_reachable", "pass": False}],
                "tokens_generated": None,
            }
        elapsed = time.perf_counter() - t0
        meta["elapsed_s"] = round(elapsed, 1)

        ok = resp.status_code == 200 and not (resp.json() or {}).get("error")
        criteria.append({"name": "service_reachable", "pass": ok})
        if not ok:
            try:
                detail = str(resp.json())[:200]
            except Exception:
                detail = resp.text[:200]
            return {
                "success": False,
                "score": 0,
                "tool_score": 0,
                "error": f"music generation failed HTTP {resp.status_code}: {detail}",
                "meta": meta,
                "criteria": criteria,
                "tokens_generated": None,
            }

        data = resp.json() or {}
        raw = b64mod.b64decode(data.get("audio_b64") or "")
        stats = self._wav_stats(raw)
        meta.update({k: v for k, v in stats.items() if k != "error"})
        duration = float(stats.get("duration_s", 0))

        tol = float(params.get("duration_tolerance", 0.25))
        dur_ok = abs(duration - requested) <= max(requested * tol, 1.0)
        criteria.append({"name": f"duration_{duration}s_within_{int(tol * 100)}pct", "pass": dur_ok})

        rms = float(stats.get("rms", 0))
        audible = rms >= float(params.get("min_rms", 0.005))
        criteria.append({"name": f"audible_rms_{rms}", "pass": audible})

        zcr = float(stats.get("zcr", 0))
        # A dead-flat or pure-DC render has ~0 crossing rate; real music sits well above.
        spectral_ok = float(params.get("min_zcr", 0.005)) <= zcr <= 0.45
        criteria.append({"name": f"spectral_variety_zcr_{zcr}", "pass": spectral_ok})

        score = round(100 * sum(1 for c in criteria if c["pass"]) / len(criteria))
        errors = [c["name"] for c in criteria if not c["pass"]]
        return {
            "success": ok and dur_ok and audible and spectral_ok,
            "score": score,
            "tool_score": score,
            "response": f"[music {duration}s (wanted {requested:g}s) in {elapsed:.1f}s]",
            "error": "" if not errors else f"failed: {', '.join(errors)}",
            "meta": meta,
            "criteria": criteria,
            "tokens_generated": None,
            "artifact_b64": data.get("audio_b64"),
            "artifact_mime": "audio/wav",
        }

    async def _composite_llm_html(self, model: str, spec_prompt: str) -> tuple[str | None, str]:
        """Ask an LLM for the composite game artifact; returns (html, error)."""
        use_proxy = not online_model_provider.is_online_model(model)
        res = (
            await self.test_model_proxy(
                model, {"id": "composite_stage", "reasoning_budget": 2048, "prompt": spec_prompt}
            )
            if use_proxy
            else await self.test_model_direct(
                model, {"id": "composite_stage", "reasoning_budget": 2048, "prompt": spec_prompt}
            )
        )
        html = extract_clean_code(res.get("response") or "", "web") or (res.get("response") or "")
        if "<html" not in html.lower() and "<canvas" not in html.lower():
            return None, f"LLM produced no usable HTML ({(res.get('error') or 'empty response')[:120]})"
        return html, ""

    async def test_tool_service(self, model: str, test: dict) -> dict:
        """Dispatch tool-type benchmarks to their backing services.

        ``model`` is a pseudo-model like ``tool:image``/``tool:tts``/``tool:music``
        for pure service tests; composite tests run on a REAL chat model and use
        that LLM to author the game while image/music services supply assets.
        """
        ttype = test.get("type")
        if ttype == "image":
            return await self._tool_image(test)
        if ttype == "tts":
            return await self._tool_tts(test)
        if ttype == "music":
            return await self._tool_music(test)
        if ttype == "composite":
            return await self._run_composite_game(model, test)
        return {"success": False, "score": 0, "tool_score": 0, "error": f"unknown tool type {ttype}"}

    async def _run_composite_game(self, model: str, test: dict) -> dict:
        """Advanced game benchmark using every tool at once.

        Stage 1: the selected chat LLM authors a complete HTML5 game whose code
        references __SPRITE__ and __BGM__ placeholders.
        Stage 2: sd-server renders the sprite sheet from the test's image prompt.
        Stage 3: audio-server composes the background loop.
        Stage 4: the assembled single-file game is executed headless; a rendered,
        non-blank frame plus present <audio> element proves integration.
        """
        params = test.get("params") or {}
        sprite_prompt = params.get("sprite_prompt", "pixel art game hero sprite sheet, transparent background")
        bgm_prompt = params.get("bgm_prompt", "chiptune adventure loop, upbeat, 10 seconds")

        llm_error = ""
        html = None
        if online_model_provider.is_online_model(model) or not model.startswith("tool:"):
            html, llm_error = await self._composite_llm_html(model, test.get("prompt", ""))
        else:
            llm_error = "composite tests require a real chat model"

        image_res = await self._tool_image(
            {
                "params": {
                    "prompt": sprite_prompt,
                    "size": params.get("sprite_size", "512x512"),
                    "steps": int(params.get("sprite_steps", 20)),
                },
                "time_budget_s": 300,
            }
        )
        music_res = await self._tool_music(
            {
                "params": {
                    "prompt": bgm_prompt,
                    "duration_s": float(params.get("bgm_duration_s", 10)),
                },
            }
        )

        criteria: list[dict] = [
            {"name": "llm_game_authored", "pass": bool(html)},
            {"name": "sprite_generated", "pass": bool(image_res.get("artifact_b64"))},
            {"name": "bgm_generated", "pass": bool(music_res.get("artifact_b64"))},
        ]

        game_ran = None
        screenshot = None
        if html and image_res.get("artifact_b64") and music_res.get("artifact_b64"):
            final_html = html.replace("__SPRITE__", f"data:image/png;base64,{image_res['artifact_b64']}")
            final_html = final_html.replace("__BGM__", f"data:audio/wav;base64,{music_res['artifact_b64']}")
            fenced = f"```html\n{final_html}\n```"
            try:
                gr = grade_code(fenced, "web", None, timeout=60, ui=True)
                game_ran = gr.get("ran")
                screenshot = gr.get("screenshot")
                ran_ok = game_ran is True
            except Exception as e:  # pragma: no cover - runtime dependent
                ran_ok = False
                gr = {"output": "", "error": f"headless run failed: {e}"}
            criteria.append({"name": "game_runs_headless", "pass": bool(ran_ok)})
        else:
            criteria.append({"name": "game_runs_headless", "pass": False})
            gr = {"output": "", "error": "assets missing"}

        score = round(100 * sum(1 for c in criteria if c["pass"]) / len(criteria))
        failed = [c["name"] for c in criteria if not c["pass"]]
        error_parts = []
        if llm_error:
            error_parts.append(llm_error)
        if failed:
            error_parts.append(f"failed: {', '.join(failed)}")
        return {
            "success": all(c["pass"] for c in criteria),
            "score": score,
            "tool_score": score,
            "response": f"[composite game: llm={'ok' if html else 'FAIL'} sprite={'ok' if image_res.get('artifact_b64') else 'FAIL'} bgm={'ok' if music_res.get('artifact_b64') else 'FAIL'} headless={game_ran}]",
            "error": "; ".join(error_parts),
            "meta": {
                "image_meta": image_res.get("meta"),
                "music_meta": music_res.get("meta"),
                "game_output": (gr.get("output") or "")[:400],
            },
            "criteria": criteria,
            "screenshot": screenshot,
            "code_ran": game_ran,
            "tokens_generated": None if html is None else 1,
        }

    async def test_model_direct(self, model: str, test: dict, sampler: ResourceSampler | None = None) -> dict:
        """Test a model directly without proxy or via online provider."""
        if online_model_provider.is_online_model(model):
            if sampler:
                sampler.start()
            res = await online_model_provider.query_online_model(
                model_identifier=model,
                prompt=test["prompt"],
                max_tokens=_test_num_predict(test, model),
            )
            if sampler:
                await sampler.stop()
            latency = res.get("latency", 0.0)
            return {
                "ollama_url": "online",
                "success": res.get("success", False),
                "prompt": test["prompt"],
                "latency": round(latency, 3),
                "response": res.get("response"),
                "tokens_generated": res.get("tokens_generated", 0),
                "eval_duration": int(latency * 1e9),
                "prompt_eval_duration": 0,
                "error": res.get("error"),
            }

        last_error: Exception | None = None
        try:
            think_on = _test_thinking(test, model)
            reasoning_budget = _test_reasoning_budget(test, model)
            num_predict = await self._effective_num_predict(model, test["prompt"], _test_num_predict(test, model))
        except RuntimeError as e:
            return {
                "ollama_url": "context-guard",
                "success": False,
                "prompt": test["prompt"],
                "latency": 0,
                "response": None,
                "tokens_generated": 0,
                "error": str(e),
            }
        if num_predict <= 0:
            return {
                "ollama_url": "context-guard",
                "success": False,
                "prompt": test["prompt"],
                "latency": 0,
                "response": None,
                "tokens_generated": 0,
                "error": (
                    "Context window exhausted before generation. Reduce the prompt size or "
                    "increase the backend ctx-size."
                ),
            }
        for ollama_url in self.OLLAMA_SERVER_URLS:
            try:
                async with httpx.AsyncClient(timeout=_STREAM_TIMEOUT) as client:
                    start_t = time.time()
                    if sampler:
                        sampler.start()
                    payload = {
                        "model": model,
                        "prompt": test["prompt"],
                        "stream": True,
                        "think": think_on,
                        "options": {
                            "num_predict": num_predict,
                            "temperature": _model_temperature(model),
                            **_model_sampling_options(model),
                        },
                    }
                    if reasoning_budget is not None:
                        payload["reasoning_budget"] = reasoning_budget
                    async with client.stream("POST", f"{ollama_url}/api/generate", json=payload) as response:
                        if response.status_code != 200:
                            body = (await response.aread()).decode("utf-8", "replace")
                            elapsed = time.time() - start_t
                            if sampler:
                                await sampler.stop()
                            return {
                                "ollama_url": ollama_url,
                                "success": False,
                                "prompt": test["prompt"],
                                "latency": round(elapsed, 3),
                                "response": None,
                                "tokens_generated": 0,
                                "error": f"HTTP {response.status_code}: {body}",
                            }
                        data = await _read_generate_stream(response)
                    elapsed = time.time() - start_t
                    if sampler:
                        await sampler.stop()

                    eval_ns = data["eval_duration"]
                    prompt_ns = data["prompt_eval_duration"]
                    eval_count = data["eval_count"]
                    response_text, thinking = self.strip_thinking(data["content"])
                    if not thinking and data.get("thinking"):
                        thinking = data["thinking"]

                    # Guard: empty generation - server returned 200 but produced nothing.
                    # Skip the nudge retry (it would hang) and fail immediately.
                    if not response_text and eval_count == 0:
                        return {
                            "ollama_url": ollama_url,
                            "success": False,
                            "prompt": test["prompt"],
                            "latency": round(elapsed, 3),
                            "response": None,
                            "tokens_generated": 0,
                            "error": "Empty generation (eval_count=0, no content returned)",
                        }

                    # If we exhausted the token cap, inject nudge and request remaining tokens
                    if eval_count >= num_predict:
                        try:
                            new_prompt = (
                                f"{test['prompt']}\n{response_text}\n"
                                f"[System: You are halfway through your token budget. Please come up with an answer quickly.]"
                            )
                            payload2 = {
                                "model": model,
                                "prompt": new_prompt,
                                "stream": True,
                                "think": think_on,
                                "options": {
                                    "num_predict": num_predict,
                                    "temperature": _model_temperature(model),
                                    **_model_sampling_options(model),
                                },
                            }
                            if reasoning_budget is not None:
                                payload2["reasoning_budget"] = reasoning_budget
                            start_t2 = time.time()
                            async with client.stream("POST", f"{ollama_url}/api/generate", json=payload2) as response2:
                                if response2.status_code == 200:
                                    data2 = await _read_generate_stream(response2)
                                    response_text2, thinking2 = self.strip_thinking(data2["content"])
                                    response_text = response_text + "\n" + response_text2
                                    if thinking2:
                                        thinking = (thinking + "\n\n" + thinking2) if thinking else thinking2
                                    elif data2.get("thinking"):
                                        thinking = data2["thinking"]
                                    eval_count += data2["eval_count"]
                                    eval_ns += data2["eval_duration"]
                                    prompt_ns += data2["prompt_eval_duration"]
                            elapsed += time.time() - start_t2
                        except Exception as e2:
                            print(f"Phase 2 direct query error: {e2}")

                    latency = (eval_ns + prompt_ns) / 1e9 if (eval_ns or prompt_ns) else elapsed
                    return {
                        "ollama_url": ollama_url,
                        "success": True,
                        "prompt": test["prompt"],
                        "latency": round(latency, 3),
                        "response": response_text,
                        "thinking": thinking,
                        "think": think_on,
                        "reasoning_budget": reasoning_budget,
                        "num_predict": num_predict,
                        "tokens_generated": eval_count,
                        "eval_duration": eval_ns,
                        "prompt_eval_duration": prompt_ns,
                        "error": None,
                    }
            except (httpx.RemoteProtocolError, httpx.ReadError) as e:
                # Server crashed or dropped the connection - fail immediately
                # instead of waiting out the full read timeout.
                if sampler:
                    await sampler.stop()
                last_error = e
                continue
            except Exception as e:
                if sampler:
                    await sampler.stop()
                last_error = e
                continue
        error_msg = ""
        if last_error:
            if isinstance(last_error, (httpx.TimeoutException, asyncio.TimeoutError)):
                error_msg = f"Request timed out: {type(last_error).__name__}"
            else:
                error_msg = (
                    f"{type(last_error).__name__}: {last_error!s}" if str(last_error) else type(last_error).__name__
                )
        else:
            error_msg = "Unknown error"

        return {
            "ollama_url": "all_failed",
            "success": False,
            "prompt": test["prompt"],
            "latency": 0,
            "response": None,
            "tokens_generated": 0,
            "error": error_msg,
        }

    def _display_live_results(self, results: dict):
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
                    "knowledge",
                    "mmlu_pro",
                    "gpqa_diamond",
                    "hle",
                    "math_hard",
                    "ifeval",
                ]:
                    cat_key = f"category_{category_key}"
                    if cat_key in model_result:
                        cat_stats = model_result[cat_key]
                        success_rate = cat_stats.get("tests_passed", 0) / max(cat_stats.get("tests_run", 1), 1) * 100
                        print(f"{model_name:<30} {category_key:<20} {success_rate:>6.1f}%")

        if mode in ("performance", "all"):
            print("\n--- Hardware & Performance Results ---")
            print(f"{'Model':<30} {'TPS':<12} {'TTFT (ms)':<12} {'Peak RAM %':<14} {'Peak VRAM (MB)':<14}")
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
        self,
        model: str,
        use_proxy: bool,
        progress_callback=None,
        cancel_event=None,
        completed_container=None,
        total_tests=None,
        test_ids: list[str] | None = None,
        prior_results: dict[str, dict] | None = None,
        mode: str = "functional",
        generated_at: str | None = None,
        resume: bool = False,
        groups: list[str] | None = None,
        tiers: list[str] | None = None,
    ) -> dict:
        """Run only functional (accuracy) tests on a model."""
        print(f"\n--- Running Functional Correctness Suite for: {model} ---")
        results: dict[str, Any] = {"model": model, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        categories = {
            # Lighter / quicker categories first so early failures surface fast.
            "instruction": self._instruction_tests,
            "creative": self._creative_tests,
            "reasoning": self._reasoning_tests,
            "home_automation": self._home_automation_tests,
            "metacog": self._metacog_tests,
            "life": self._life_tests,
            "biblical": self._biblical_tests,
            "uiux": self._uiux_tests,
            "office": self._office_tests,
            "coding": self._coding_tests,
            "gamedev": self._gamedev_tests,
            "gamedev_alt": self._gamedev_alt_tests,
            "youtuber": self._youtuber_tests,
            "agentic": self._agentic_tests,
            "appdev": self._appdev_tests,
            "webdev": self._webdev_tests,
            "linux_admin": self._linux_admin_tests,
            "database": self._database_tests,
            "cpp": self._cpp_tests,
            "java": self._java_tests,
            "debugging": self._debugging_tests,
            "logic": self._logic_tests,
            "retrogames": self._retrogames_tests,
            "threedprint": self._threedprint_tests,
            "languages": self._languages_tests,
            "tvdev": self._tvdev_tests,
            "networking": self._networking_tests,
            "usb": self._usb_tests,
            "iac": self._iac_tests,
            "linux_driver": self._linux_driver_tests,
            "bash": self._bash_tests,
            "basic": self._basic_tests,
            "pascal": self._pascal_tests,
            "typescript": self._typescript_tests,
            "rpm": self._rpm_tests,
            "android": self._android_tests,
            # Heavy knowledge/reasoning corpora last.
            "knowledge": self._knowledge_tests,
            "mmlu_pro": self._mmlu_pro_tests,
            "gpqa_diamond": self._gpqa_diamond_tests,
            "hle": self._hle_tests,
            "math_hard": self._math_hard_tests,
            "ifeval": self._ifeval_tests,
            "code_review": self._code_review_tests,
        }

        # Optional group filter: run only the selected groups (or all when None).
        if groups:
            categories = {k: v for k, v in categories.items() if k in set(groups)}

        # Calculate total tests for progress
        if total_tests is None:
            total_tests = sum(len(test_func(model)) for test_func in categories.values())
        if completed_container is None:
            completed_container = [0]

        for category, test_func in categories.items():
            if (cancel_event and cancel_event.is_set()) or results.get("rate_limited"):
                break
            tests = test_func(model)
            if test_ids:
                tests = [t for t in tests if t["id"] in test_ids]
                if not tests:
                    continue
            if tiers:
                allowed = set(tiers)
                tests = [t for t in tests if (t.get("tier") or "standard") in allowed]
                if not tests:
                    continue
            category_results = []
            for i, test in enumerate(tests, 1):
                if (cancel_event and cancel_event.is_set()) or results.get("rate_limited"):
                    break

                # Resume: reuse a prior result (pass or fail) instead of re-running,
                # unless the test definition has changed since the prior run or
                # the test was explicitly (re)selected via test_ids.
                cur_hash = self.compute_test_hash(test)
                if resume and not test_ids and prior_results and test["id"] in prior_results:
                    prior_rec = prior_results[test["id"]]
                    prior_hash = prior_rec.get("test_hash")
                    cur_prompt = (test.get("prompt") or "").strip()
                    rec_prompt = (prior_rec.get("prompt") or "").strip()
                    is_stale = False
                    if prior_hash:
                        is_stale = prior_hash != cur_hash
                    elif cur_prompt:
                        is_stale = not (rec_prompt.startswith(cur_prompt) or rec_prompt == cur_prompt)

                    if not is_stale:
                        reused = dict(prior_rec)
                        reused["last_run"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                        reused.update(
                            {
                                "test_id": test["id"],
                                "test_category": category,
                                "test_label": test["label"],
                                "test_hash": cur_hash,
                            }
                        )
                        category_results.append(reused)
                        completed_container[0] += 1
                        # Persist the reused result too so the per-model file stays complete
                        # even if the run is interrupted before the final merge.
                        try:
                            self.save_test_result_incremental(model, category, reused, mode, use_proxy, generated_at)
                        except Exception as e:
                            print(
                                f"[benchmark] Warning: incremental save failed for skipped {category}/{test['id']}: {e}"
                            )
                        print(f"[{category}] Skipped (resume) {test['id']} ✓")
                        continue
                    else:
                        print(f"[{category}] Test definition updated for {test['id']} - re-running...")

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

                # Retry transient empty generations. A model can return an empty
                # completion (eval_count=0, or a throttle/unload race on the proxy)
                # even when it produces valid output on a second try. Online models
                # already retry internally; this covers local/proxy models, which
                # otherwise score a transient empty as a permanent false negative.
                test_result: dict = {}
                for _attempt in range(3):
                    try:
                        run_test = test
                        # Tool benchmarks (image/tts/music/composite) talk to the
                        # sd-server/audio-server services directly, not to an LLM.
                        if test.get("type") in self.TOOL_TEST_TYPES:
                            test_result = await self.test_tool_service(model, run_test)
                        elif use_proxy:
                            # Code categories get a directive demanding runnable output,
                            # which the sandbox-execution grading step then verifies.
                            if test.get("category") in self.CODE_CATEGORIES:
                                run_test = dict(test)
                                run_test["prompt"] = (test.get("prompt") or "") + self.CODE_DIRECTIVE
                            test_result = await self.test_model_proxy(model, run_test)
                        else:
                            if test.get("category") in self.CODE_CATEGORIES:
                                run_test = dict(test)
                                run_test["prompt"] = (test.get("prompt") or "") + self.CODE_DIRECTIVE
                            test_result = await self.test_model_direct(model, run_test)
                    except Exception as e:
                        test_result = {
                            "proxy": "online" if use_proxy else "direct",
                            "success": False,
                            "prompt": test.get("prompt"),
                            "response": None,
                            "tokens_generated": 0,
                            "eval_duration": 0,
                            "prompt_eval_duration": 0,
                            "error": f"test execution error: {e}",
                        }
                    _resp = test_result.get("response") or ""
                    _err = (test_result.get("error") or "").lower()
                    _empty = (not _resp) and ("empty" in _err or test_result.get("tokens_generated", 0) == 0)
                    if test_result.get("success") or not _empty or _attempt >= 2:
                        break
                    print(f"[benchmark] empty result for {test['id']} (attempt {_attempt + 1}); retrying")
                    await asyncio.sleep(2.0 * (_attempt + 1))

                # Rate-limit handling: a 429 means the provider's quota window is
                # exhausted. query_online_model already retried with backoff and
                # honored the provider's retry hint, so if we still have a 429 the
                # window simply hasn't reset yet. Do NOT abort the whole model run
                # over a transient quota hit: record the test as failed, wait out
                # the retry window, and continue with the next test.
                if self._is_rate_limited_result(test_result):
                    retry_after = test_result.get("retry_after")
                    try:
                        wait = float(retry_after) if retry_after else self._rate_limit_retry_floor
                    except (TypeError, ValueError):
                        wait = self._rate_limit_retry_floor
                    wait = min(wait, 120.0)
                    print(
                        f"[benchmark] {model} hit a provider rate limit (429) on "
                        f"{category}/{test['id']}; waiting {wait:.1f}s and continuing."
                    )
                    test_result["error"] = test_result.get("error") or "Provider rate limit (429)"
                    await asyncio.sleep(wait)
                    # Do NOT break — keep the failed result and proceed to the next
                    # test once the quota window has had time to reset.

                if test_result.get("response"):
                    actual_correct = self._verify_functional_response(test, test_result.get("response", ""))
                    test_result["functional_pass"] = bool(actual_correct)
                    if test_result["success"] and not actual_correct:
                        test_result["success"] = False
                        test_result["error"] = "Failed correctness verification check"

                # Code-quality + AI-watermark scoring (always, when there is a response)
                resp_text = test_result.get("response", "") or ""
                if resp_text:
                    test_result["code_quality"] = self._score_code_quality(resp_text, test)
                    test_result["watermark"] = self._score_watermark(resp_text)

                # Coding benchmarks: actually execute the extracted code in the
                # sandbox so "runnable code" is verified, not assumed. The run
                # outcome feeds the unified 0-100 score below. Only languages we
                # can execute are run; others fall back to a structural check.
                ttype = self._infer_type(test, resp_text)
                if ttype in ("code", "ui") and resp_text:
                    is_ui = (ttype == "ui") or any(
                        k in resp_text.lower()
                        for k in (
                            "import pygame",
                            "import tkinter",
                            "from tkinter",
                            "import kivy",
                            "pygame.opengl",
                            "from pygame.locals import opengl",
                            "from opengl import",
                            "import opengl",
                            "import ursina",
                            "from ursina import",
                        )
                    )
                    lang = test.get("lang") or self._fence_lang(resp_text) or self._infer_lang(resp_text)
                    if lang in self.EXEC_LANGS or is_ui:
                        expected_out = test.get("expected_output")
                        try:
                            gr = grade_code(resp_text, lang, expected_out, ui=is_ui)
                            test_result["code_ran"] = gr["ran"]
                            test_result["code_score"] = gr["score"]
                            test_result["lint_passed"] = gr.get("lint_passed", gr["ran"] is not False)
                            test_result["code_output"] = gr.get("output", "")
                            test_result["code_error"] = gr.get("error", "")
                            test_result["screenshot"] = gr.get("screenshot")
                            if gr["ran"] is not None:
                                # Execution actually happened. A runnable program
                                # passes; a crash/timeout fails (score is already
                                # 0). BUT if the functional verification against
                                # the prompt's expectations already failed, that
                                # takes precedence: running is not the same as
                                # satisfying the task (e.g. a game that renders
                                # but never implements the persistent scoreboard).
                                ran_ok = bool(gr["ran"])
                                fp = test_result.get("functional_pass")
                                if ran_ok and fp is False:
                                    test_result["success"] = False
                                    test_result["error"] = (
                                        test_result.get("error") or "Failed correctness verification check"
                                    )
                                else:
                                    test_result["success"] = ran_ok
                                    if ran_ok:
                                        test_result["error"] = ""
                        except Exception as e:  # pragma: no cover - runtime dependent
                            test_result["code_ran"] = None
                            test_result["code_error"] = f"sandbox grading failed: {e}"

                # Rubric compliance: score the response against the benchmark's
                # prompt-required, easily-checkable features (persistent high-score
                # board, name entry, score reset, etc.) plus the default code rubric.
                # The per-criterion results feed the dashboard breakdown so we can
                # see exactly which requested features each model delivered or
                # missed, and which categories models struggle with.
                if ttype in ("code", "ui") and resp_text:
                    try:
                        extracted = (
                            extract_clean_code(resp_text, (test.get("lang") or ""))
                            if test.get("lang")
                            else extract_clean_code(resp_text)
                        )
                    except Exception:  # pragma: no cover - extraction infra
                        extracted = ""
                    test_result["rubric"] = self._evaluate_rubric(test, resp_text, extracted)
                else:
                    test_result["rubric"] = {
                        "score": 100,
                        "fraction": 1.0,
                        "total_points": 0,
                        "earned_points": 0,
                        "criteria": [],
                    }

                # Non-code/non-UI tests have nothing to lint; only executed code
                # can fail the syntax gate. Default to True so stats are consistent.
                # BUT a code/UI test that produced NO output at all has nothing that
                # passed a compile/syntax check — an empty generation must not be
                # reported as a green lint, or the dashboard would show "Lint/Compile
                # ✓ passed" for a model that generated zero tokens.
                if ttype in ("code", "ui") and not resp_text:
                    test_result["lint_passed"] = False
                    test_result.setdefault("code_ran", False)
                    test_result["code_error"] = test_result.get("code_error") or "No code generated (empty response)"
                else:
                    test_result.setdefault("lint_passed", True)

                # Unified 0-100 score across ALL benchmark types.
                test_result["score"] = self._score_test(test, test_result)

                # For review-type tests, "pass" means the model actually caught
                # the issues; align the success flag with the review score so the
                # per-group pass rate is meaningful.
                if ttype == "review":
                    test_result["success"] = test_result["score"] >= 50

                if test_result["success"]:
                    print("✓")
                else:
                    print(f"✗ ({test_result.get('error')})")

                test_result.update(
                    {
                        "test_id": test["id"],
                        "test_category": category,
                        "test_label": test["label"],
                        "test_hash": cur_hash,
                        "last_run": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                )
                category_results.append(test_result)
                completed_container[0] += 1

                # Persist this single test result immediately (resumability + progress
                # is never lost even if the run is interrupted).
                try:
                    self.save_test_result_incremental(model, category, test_result, mode, use_proxy, generated_at)
                except Exception as e:
                    print(f"[benchmark] Warning: incremental save failed for {category}/{test['id']}: {e}")

                # Pace online providers so free-tier rate limits don't throttle the run
                # into empty-response false negatives (OpenRouter returns empty 200s).
                if online_model_provider.is_online_model(model):
                    await asyncio.sleep(ONLINE_BENCHMARK_THROTTLE_S)

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
                                        "completed": completed_container[0],
                                        "total": total_tests,
                                        "percentage": round((completed_container[0] / total_tests) * 100),
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
                                        "completed": completed_container[0],
                                        "total": total_tests,
                                        "percentage": round((completed_container[0] / total_tests) * 100),
                                    },
                                },
                            )
                    except Exception as e:
                        print(f"Callback error: {e}")

            results[f"category_{category}"] = self._calculate_category_stats(category_results)
        return results

    async def benchmark_model_performance(
        self,
        model: str,
        use_proxy: bool,
        progress_callback=None,
        cancel_event=None,
        completed_container=None,
        total_tests=None,
        test_ids: list[str] | None = None,
    ) -> dict:
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
        if test_ids:
            load_tests = [t for t in load_tests if t["id"] in test_ids]

        tps_list = []
        ttft_list = []
        sampler = self.ResourceSampler()
        if total_tests is None:
            total_tests = len(load_tests)
        if completed_container is None:
            completed_container = [0]

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

            completed_container[0] += 1

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
                                    "completed": completed_container[0],
                                    "total": total_tests,
                                    "percentage": round((completed_container[0] / total_tests) * 100),
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
                                    "completed": completed_container[0],
                                    "total": total_tests,
                                    "percentage": round((completed_container[0] / total_tests) * 100),
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

    # Categories whose prompts ask for source code. These are graded by actually
    # executing the extracted code in the sandbox across the supported languages
    # (Python, Node, C++, Java, SQL, Bash) and, where a language cannot be executed,
    # by the functional verification result.
    CODE_CATEGORIES: ClassVar[set[str]] = {
        "coding",
        "gamedev",
        "gamedev_alt",
        "youtuber",
        "retrogames",
        "appdev",
        "webdev",
        "cpp",
        "java",
        "debugging",
        "database",
        "linux_admin",
        "languages",
        "networking",
        "usb",
        "iac",
        "linux_driver",
        "bash",
        "basic",
        "pascal",
        "typescript",
        "rpm",
        "android",
    }

    # Appended to code-category prompts so models emit complete, self-contained,
    # runnable programs (with a demo / example usage) instead of snippets or
    # prose, which is what the sandbox execution step then verifies.
    # GRADER_DIRECTIVE_VERSION bumps whenever this text materially changes so
    # result hashing (_compute_test_hash) marks prior runs outdated.
    GRADER_DIRECTIVE_VERSION: ClassVar[str] = "v2"
    CODE_DIRECTIVE = (
        "\n\nREQUIREMENTS: Respond with a single, complete, self-contained, "
        "runnable program and nothing else (no preamble, no explanation outside "
        "the code). Use only standard libraries. The code must execute top-to-bottom "
        "without missing imports or undefined names, and must include a short demo "
        "or example input/output so it can be run and observed."
        "\n\nGRADING NOTICE: This response is scored by an automated benchmark "
        "grader that executes your code. Output ONLY the final code inside one "
        "fenced code block - reasoning, plans, or prose anywhere outside the code "
        "block, or truncated/incomplete code, will fail grading."
    )

    @staticmethod
    def _letter_grade(pct: float) -> str:
        """Map a 0-100 percentage to an easy-to-read letter grade."""
        if pct >= 90:
            return "A"
        if pct >= 80:
            return "B"
        if pct >= 70:
            return "C"
        if pct >= 60:
            return "D"
        return "F"

    @staticmethod
    def _stars(pct: float) -> int:
        """Map a 0-100 percentage to a 1-5 star rating for at-a-glance comparison."""
        if pct >= 90:
            return 5
        if pct >= 80:
            return 4
        if pct >= 70:
            return 3
        if pct >= 60:
            return 2
        return 1

    def _infer_lang(self, code: str) -> str:
        low = (code or "").lower()
        if "#include" in low or "std::" in low:
            return "cpp"
        if "public class" in low or "system.out" in low or "public static void main" in low:
            return "java"
        if "package main" in low or "func main(" in low:
            return "go"
        if "fn main(" in low or "use std::" in low or "let mut" in low:
            return "rust"
        # Raw SQL scripts never contain Python scaffolding. A Python program
        # embedding SQL string literals (sqlite3 solutions) must run as
        # Python; mislabeling it "sql" pipes valid code into sqlite3 and
        # fails with a parse error on the first import line.
        looks_sql = ("select " in low and (" from " in low or " join " in low)) or "insert into" in low
        if looks_sql and not any(x in low for x in ("import ", "def ", "print(", "cursor", "sqlite3")):
            return "sql"
        if low.strip().startswith("#!/bin/bash") or low.strip().startswith("#!/bin/sh"):
            return "bash"
        if "<!doctype html" in low or "<html" in low or "<canvas" in low or ("<script" in low and "</body>" in low):
            return "web"
        if "program " in low and ("begin" in low and "end." in low):
            return "pascal"
        if 'print "hello' in low or 'input "' in low or "goto " in low:
            return "basic"
        if "terraform {" in low or 'resource "' in low or 'provider "' in low:
            return "terraform"
        if "rpm_spec" in low or ("name:" in low and "buildroot" in low) or "%description" in low:
            return "rpm"
        if "on:" in low and "jobs:" in low and ("steps:" in low or "runs-on" in low):
            return "yaml"
        py = low.count("def ") + low.count("import ") + low.count("print(") + low.count("self.")
        js = (
            low.count("console.log")
            + low.count("function ")
            + low.count("=>")
            + low.count("require(")
            + low.count("document.")
        )
        return "python" if py >= js else "node"

    # Markdown fence tag -> sandbox language. An explicit tag from the model
    # is the strongest language signal and beats content heuristics.
    _FENCE_LANG_MAP: ClassVar[dict[str, str]] = {
        "py": "python",
        "python": "python",
        "python3": "python",
        "js": "node",
        "javascript": "node",
        "node": "node",
        "ts": "typescript",
        "typescript": "typescript",
        "html": "web",
        "htm": "web",
        "web": "web",
        "cpp": "cpp",
        "c++": "cpp",
        "cc": "cpp",
        "java": "java",
        "go": "go",
        "rust": "rust",
        "rs": "rust",
        "sql": "sql",
        "sqlite": "sql",
        "bash": "bash",
        "sh": "bash",
        "shell": "bash",
        "basic": "basic",
        "bas": "basic",
        "pascal": "pascal",
        "pas": "pascal",
        "yaml": "yaml",
        "yml": "yaml",
        "terraform": "terraform",
        "hcl": "terraform",
        "tf": "terraform",
    }

    def _fence_lang(self, resp: str) -> str:
        """Return the sandbox language of the first explicitly-tagged code fence.

        Models that answer SQL prompts with a Python sqlite3 program tag it
        ```` ```python ```` — honoring the tag executes their code correctly,
        while content sniffing sees embedded SQL keywords and picks "sql".
        """
        for match in re.finditer(r"```([A-Za-z0-9+#-]+)[ \t]*\r?\n", resp or ""):
            return self._FENCE_LANG_MAP.get(match.group(1).lower(), "")
        return ""

    # Languages we can actually execute in the sandbox. Others are graded by
    # their functional verification result, not by code execution.
    EXEC_LANGS: ClassVar[set[str]] = {
        "python",
        "node",
        "cpp",
        "java",
        "sql",
        "bash",
        "go",
        "rust",
        "web",
        "basic",
        "pascal",
        "typescript",
        "yaml",
        "terraform",
        "rpm",
    }

    # Test types executed against generation SERVICES instead of chat models.
    # image/tts/music grade the backing service directly; composite orchestrates
    # an LLM plus both media services into one artifact that is then run headless.
    TOOL_TEST_TYPES: ClassVar[tuple[str, ...]] = ("image", "tts", "music", "composite")

    def _infer_type(self, test: dict, resp: str) -> str:
        t = test.get("type")
        if t:
            return t
        cat = test.get("category") or ""
        if cat == "webdev":
            return "web"
        if cat in self.CODE_CATEGORIES:
            return "code"
        if "expected_issues" in test:
            return "review"
        if test.get("expected") is not None:
            return "knowledge"
        return "open"

    def _score_test(self, test: dict, result: dict) -> int:
        """Return a unified 0-100 score for any test type.

        Graded scoring instead of binary pass/fail: for executable code and UI
        tests the score blends (1) whether the code actually ran, (2) whether
        the output satisfied the prompt's stated expectations (functional
        verification), and (3) static code-quality heuristics. A result that
        runs but misses the task's expectations scores below passing; a result
        that runs and meets expectations scores proportionally to how clean and
        complete it is.
        """
        resp = result.get("response") or ""
        ttype = self._infer_type(test, resp)
        # Tool benchmarks (image/tts/music/composite) carry their own criteria-
        # based score computed at generation time; no LLM-response heuristics apply.
        if test.get("type") in self.TOOL_TEST_TYPES:
            raw = result.get("tool_score")
            if isinstance(raw, (int, float)):
                return max(0, min(100, round(raw)))
            return 100 if result.get("success") else 0
        if ttype in ("code", "ui"):
            ran = result.get("code_ran")
            if ran is None:
                # Code was not executed (unsupported language or no sandbox). The
                # functional verification result (success) is the defined score.
                return 100 if result.get("success") else 0
            if not ran:
                return 0
            # Static quality heuristic (0-100) from the same extracted code.
            quality = (result.get("code_quality") or {}).get("score")
            if not isinstance(quality, (int, float)):
                quality = 60
            quality = max(0, min(100, int(quality)))
            # Functional verification against the prompt's expectations:
            # missing it means the code ran but did not actually do the task.
            if result.get("functional_pass") is False:
                return min(round(0.6 * quality), 45)
            # Run outcome: 100 = screenshot/expected-output match, 60 = clean run.
            base = int(result.get("code_score", 60))
            # Rubric compliance: requested prompt features the model actually
            # delivered (e.g. a game's persistent high-score board). A run that
            # works but omits an explicitly requested, easily-checkable feature
            # costs points rather than passing at full marks. Missing every
            # criterion caps the score at half of the execution+quality blend.
            rubric = result.get("rubric")
            if rubric:
                fraction = rubric.get("fraction", 1.0)
                base = round(base * (0.5 + 0.5 * fraction))
                quality = round(quality * (0.5 + 0.5 * fraction))
            # Execution result matters most; static quality refines it.
            return max(0, min(100, round(0.7 * base + 0.3 * quality)))
        if ttype == "review":
            issues = test.get("expected_issues") or []
            if not issues:
                return 100 if result.get("success") else 0
            low = (resp or "").lower()
            found = sum(1 for iss in issues if iss.lower() in low)
            return round(100 * found / len(issues))
        if ttype == "knowledge":
            expected = test.get("expected")
            if expected and str(expected).strip():
                low = (resp or "").lower()
                words = [w for w in re.split(r"\W+", str(expected)) if len(w) > 3]
                if words:
                    found = sum(1 for w in words if w.lower() in low)
                    return round(100 * found / len(words))
            return 100 if result.get("success") else 0
        if ttype == "web":
            # Web/HTML output is judged by rendering it live; a non-empty,
            # non-refusal response is accepted as produced.
            return 100 if result.get("success") else 0
        # open / creative / default
        return 100 if result.get("success") else 0

    def _calculate_category_stats(self, results: list[dict]) -> dict:
        successful_tests = [r for r in results if r["success"]]
        scores: list[int] = []
        for r in results:
            s = r.get("score")
            if isinstance(s, (int, float)):
                scores.append(int(s))
        cat_score = round(sum(scores) / len(scores)) if scores else 0
        if not successful_tests:
            return {
                "tests_run": len(results),
                "tests_passed": 0,
                "score": cat_score,
                "letter": self._letter_grade(cat_score),
                "stars": self._stars(cat_score),
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
            "score": cat_score,
            "letter": self._letter_grade(cat_score),
            "stars": self._stars(cat_score),
            "avg_tokens_per_sec": round(avg_tokens_per_sec, 2),
            "avg_ttft_ms": round(avg_ttft_ms, 1),
            "tests": results,
        }

    def _compute_overall(self, model_data: dict) -> dict:
        """Roll per-group (category_*) scores into one comparable overall score.

        Each group is weighted equally so a 5-test group and a 120-test group
        contribute the same to the overall number, making models directly
        comparable regardless of how many tests a category holds.
        """
        groups: list[dict] = []
        scores: list[int] = []
        for key, block in model_data.items():
            if not (isinstance(key, str) and key.startswith("category_") and isinstance(block, dict)):
                continue
            score = block.get("score")
            if not isinstance(score, (int, float)):
                continue
            name = key[len("category_") :]
            groups.append(
                {
                    "group": name,
                    "score": int(score),
                    "letter": block.get("letter", self._letter_grade(int(score))),
                    "stars": block.get("stars", self._stars(int(score))),
                    "tests_run": block.get("tests_run", 0),
                    "tests_passed": block.get("tests_passed", 0),
                }
            )
            scores.append(int(score))
        overall = round(sum(scores) / len(scores)) if scores else 0
        return {
            "score": overall,
            "letter": self._letter_grade(overall),
            "stars": self._stars(overall),
            "groups": groups,
        }

    def _extract_duration(self, result: dict) -> float:
        if "eval_duration" in result and "prompt_eval_duration" in result:
            return (result["eval_duration"] + result["prompt_eval_duration"]) / 1e9
        return result.get("latency", 0)

    def _calculate_avg_ttft(self, results: list[dict]) -> float:
        total_ttft = 0
        for result in results:
            if "prompt_eval_duration" in result:
                ttft = result["prompt_eval_duration"] / 1e9 * 1000
                total_ttft += ttft
            elif "latency" in result:
                total_ttft += result["latency"] * 1000
        return total_ttft / len(results) if results else 0

    def _iter_category_tests(self, name: str, test_ids: list[str] | None = None, tiers: list[str] | None = None):
        """Yield a category's tests, honoring optional ``test_ids`` and ``tiers`` filters."""
        tests = getattr(self, f"_{name}_tests")("")
        if test_ids:
            tests = [t for t in tests if t["id"] in test_ids]
        if tiers:
            allowed = set(tiers)
            tests = [t for t in tests if (t.get("tier") or "standard") in allowed]
        return tests

    def get_total_tests_per_model(
        self,
        mode: str,
        test_ids: list[str] | None = None,
        groups: list[str] | None = None,
        tiers: list[str] | None = None,
    ) -> int:
        group_set = set(groups) if groups else None

        def _include(name: str) -> bool:
            return group_set is None or name in group_set

        total = 0
        if mode in ("functional", "all"):
            if _include("coding"):
                total += len(self._iter_category_tests("coding", test_ids, tiers))
            if _include("reasoning"):
                total += len(self._iter_category_tests("reasoning", test_ids, tiers))
            if _include("instruction"):
                total += len(self._iter_category_tests("instruction", test_ids, tiers))
            if _include("creative"):
                total += len(self._iter_category_tests("creative", test_ids, tiers))
            if _include("home_automation"):
                total += len(self._iter_category_tests("home_automation", test_ids, tiers))
            if _include("mmlu_pro"):
                total += len(self._iter_category_tests("mmlu_pro", test_ids, tiers))
            if _include("gpqa_diamond"):
                total += len(self._iter_category_tests("gpqa_diamond", test_ids, tiers))
            if _include("hle"):
                total += len(self._iter_category_tests("hle", test_ids, tiers))
            if _include("math_hard"):
                total += len(self._iter_category_tests("math_hard", test_ids, tiers))
            if _include("ifeval"):
                total += len(self._iter_category_tests("ifeval", test_ids, tiers))
            if _include("gamedev"):
                total += len(self._iter_category_tests("gamedev", test_ids, tiers))
            if _include("gamedev_alt"):
                total += len(self._iter_category_tests("gamedev_alt", test_ids, tiers))
            if _include("youtuber"):
                total += len(self._iter_category_tests("youtuber", test_ids, tiers))
            if _include("agentic"):
                total += len(self._iter_category_tests("agentic", test_ids, tiers))
            if _include("appdev"):
                total += len(self._iter_category_tests("appdev", test_ids, tiers))
            if _include("linux_admin"):
                total += len(self._iter_category_tests("linux_admin", test_ids, tiers))
            if _include("webdev"):
                total += len(self._iter_category_tests("webdev", test_ids, tiers))
            if _include("database"):
                total += len(self._iter_category_tests("database", test_ids, tiers))
            if _include("cpp"):
                total += len(self._iter_category_tests("cpp", test_ids, tiers))
            if _include("java"):
                total += len(self._iter_category_tests("java", test_ids, tiers))
            if _include("debugging"):
                total += len(self._iter_category_tests("debugging", test_ids, tiers))
            if _include("logic"):
                total += len(self._iter_category_tests("logic", test_ids, tiers))
            if _include("retrogames"):
                total += len(self._iter_category_tests("retrogames", test_ids, tiers))
            if _include("threedprint"):
                total += len(self._iter_category_tests("threedprint", test_ids, tiers))
            if _include("languages"):
                total += len(self._iter_category_tests("languages", test_ids, tiers))
            if _include("tvdev"):
                total += len(self._iter_category_tests("tvdev", test_ids, tiers))
            if _include("uiux"):
                total += len(self._iter_category_tests("uiux", test_ids, tiers))
            if _include("office"):
                total += len(self._iter_category_tests("office", test_ids, tiers))
            if _include("life"):
                total += len(self._iter_category_tests("life", test_ids, tiers))
            if _include("biblical"):
                total += len(self._iter_category_tests("biblical", test_ids, tiers))
            if _include("metacog"):
                total += len(self._iter_category_tests("metacog", test_ids, tiers))
            if _include("code_review"):
                total += len(self._iter_category_tests("code_review", test_ids, tiers))
            tests = []
            if test_ids:
                chosen = [
                    "coding",
                    "reasoning",
                    "instruction",
                    "creative",
                    "home_automation",
                    "mmlu_pro",
                    "gpqa_diamond",
                    "hle",
                    "math_hard",
                    "ifeval",
                    "gamedev",
                    "gamedev_alt",
                    "youtuber",
                    "agentic",
                    "appdev",
                    "linux_admin",
                    "webdev",
                    "database",
                    "cpp",
                    "java",
                    "debugging",
                    "logic",
                    "retrogames",
                    "threedprint",
                    "languages",
                    "tvdev",
                    "uiux",
                    "office",
                    "life",
                    "biblical",
                    "metacog",
                    "code_review",
                ]
                for name in chosen:
                    if _include(name):
                        tests += self._iter_category_tests(name, test_ids, tiers)
                total = len(tests)
        if mode in ("performance", "all"):
            perf_tests = ["perf_medium", "perf_long"]
            if test_ids:
                total += sum(1 for t in perf_tests if t in test_ids)
            else:
                total += len(perf_tests)
        return total

    async def run_model_benchmarks(
        self,
        models: list[str],
        use_proxy: bool,
        progress_callback=None,
        cancel_event=None,
        mode: str = "all",
        test_ids: list[str] | None = None,
        resume: bool = False,
        groups: list[str] | None = None,
        tiers: list[str] | None = None,
    ) -> dict:
        """Run split model benchmarks based on mode: 'functional', 'performance', or 'all'."""
        print("=" * 80)
        print("COMPREHENSIVE LLM MODEL BENCHMARKING SUITE")
        print(f"Running via: {'Proxy' if use_proxy else 'Direct'} | Mode: {mode.upper()}")
        print(f"Models: {models}")
        if test_ids:
            print(f"Selected Test IDs: {test_ids}")
        if groups:
            print(f"Selected Groups: {groups}")
        print("=" * 80)

        all_results: dict[str, Any] = {
            "benchmark_version": "3.0.0",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "benchmark_type": "proxy" if use_proxy else "direct",
            "benchmark_mode": mode,
            "models_tested": len(models),
            "results": [],
        }

        total_tests = len(models) * self.get_total_tests_per_model(mode, test_ids, groups, tiers)
        completed_container = [0]

        # Emit benchmark_start event
        if progress_callback:
            try:
                import inspect

                start_data = {
                    "models": models,
                    "use_proxy": use_proxy,
                    "total_tests": total_tests,
                    "timestamp": all_results["generated_at"],
                }
                if inspect.iscoroutinefunction(progress_callback):
                    await progress_callback("benchmark_start", start_data)
                else:
                    progress_callback("benchmark_start", start_data)
            except Exception as e:
                print(f"Callback error: {e}")

        # Local models share a single GPU + proxy hot-swap, so they must run one
        # at a time. Online models hit external provider APIs and can safely fan
        # out in parallel with whatever local model is occupying the GPU.
        local_sem = asyncio.Semaphore(1)

        async def _run_one_model(model: str) -> dict | None:
            if cancel_event and cancel_event.is_set():
                return None

            model_data: dict[str, Any] = {"model": model}

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
                prior_results: dict[str, dict] = {}
                try:
                    pm_file = self.MODELS_DIR / f"general_{self._sanitize_model_filename(model)}.json"
                    if pm_file.exists():
                        with open(pm_file) as f:
                            pm = json.load(f)
                        for mres in pm.get("results", []):
                            for cat in mres:
                                if cat.startswith("category_"):
                                    for t in mres[cat].get("tests", []):
                                        # Keep the most recent result per test id.
                                        prior_results[t["test_id"]] = t
                except Exception as e:
                    print(f"[benchmark] Warning: could not load prior results for resume ({model}): {e}")

                func_data = await self.benchmark_model_functional(
                    model,
                    use_proxy,
                    progress_callback,
                    cancel_event,
                    completed_container,
                    total_tests,
                    test_ids,
                    prior_results,
                    mode=mode,
                    generated_at=all_results["generated_at"],
                    resume=resume,
                    groups=groups,
                    tiers=tiers,
                )
                model_data.update(func_data)

            if mode in ("performance", "all"):
                perf_data = await self.benchmark_model_performance(
                    model,
                    use_proxy,
                    progress_callback,
                    cancel_event,
                    completed_container,
                    total_tests,
                    test_ids,
                )
                model_data.update(perf_data)

            # Roll per-group scores up into a single comparable overall score.
            overall = self._compute_overall(model_data)
            model_data["overall_score"] = overall["score"]
            model_data["overall_letter"] = overall["letter"]
            model_data["overall_stars"] = overall["stars"]
            model_data["group_scores"] = overall["groups"]

            all_results["results"].append(model_data)
            self.save_per_model_result(model_data, mode, use_proxy, all_results["generated_at"])

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
            return model_data

        async def _run_local_model(model: str) -> dict | None:
            async with local_sem:
                return await _run_one_model(model)

        tasks = []
        for model in models:
            if online_model_provider.is_online_model(model):
                tasks.append(asyncio.create_task(_run_one_model(model)))
            else:
                tasks.append(asyncio.create_task(_run_local_model(model)))
        if tasks:
            await asyncio.gather(*tasks)

        if all_results["results"]:
            run_only_results = list(all_results["results"])
            latest_file = self.RESULTS_DIR / f"{mode}_benchmarks_latest.json"
            merged_results = []
            if latest_file.exists():
                try:
                    with open(latest_file) as f:
                        prev_data = json.load(f)
                    merged_results = prev_data.get("results", [])
                except Exception as e:
                    print(f"[benchmark] Warning: failed to load previous latest file: {e}")

            current_time = time.strftime("%Y-%m-%dT%H:%M:%S")
            for new_model in all_results["results"]:
                model_found = False
                for prev_model in merged_results:
                    if prev_model.get("model") == new_model["model"]:
                        for cat in [
                            "category_coding",
                            "category_reasoning",
                            "category_instruction",
                            "category_creative",
                            "category_home_automation",
                            "category_knowledge",
                            "category_mmlu_pro",
                            "category_gpqa_diamond",
                            "category_hle",
                            "category_math_hard",
                            "category_ifeval",
                            "category_gamedev",
                            "category_appdev",
                            "category_linux_admin",
                            "category_webdev",
                            "category_database",
                            "category_cpp",
                            "category_java",
                            "category_debugging",
                            "category_logic",
                            "category_retrogames",
                            "category_threedprint",
                            "category_languages",
                            "category_tvdev",
                            "category_uiux",
                            "category_office",
                            "category_life",
                            "category_biblical",
                            "category_metacog",
                        ]:
                            if cat in new_model:
                                if cat not in prev_model:
                                    prev_model[cat] = new_model[cat]
                                    for t in prev_model[cat].get("tests", []):
                                        t["last_run"] = current_time
                                else:
                                    prev_tests = prev_model[cat].get("tests", [])
                                    new_tests = new_model[cat].get("tests", [])
                                    test_map = {t["test_id"]: t for t in prev_tests}
                                    for nt in new_tests:
                                        nt["last_run"] = current_time
                                        test_map[nt["test_id"]] = nt
                                    prev_model[cat]["tests"] = list(test_map.values())
                                    prev_model[cat]["tests_run"] = len(prev_model[cat]["tests"])
                                    prev_model[cat]["tests_passed"] = sum(
                                        1 for t in prev_model[cat]["tests"] if t.get("success")
                                    )
                        if "performance_metrics" in new_model:
                            prev_model["performance_metrics"] = new_model["performance_metrics"]
                            prev_model["performance_metrics"]["last_run"] = current_time
                        model_found = True
                        break
                if not model_found:
                    for cat in [
                        "category_coding",
                        "category_reasoning",
                        "category_instruction",
                        "category_creative",
                        "category_home_automation",
                        "category_knowledge",
                        "category_mmlu_pro",
                        "category_gpqa_diamond",
                        "category_hle",
                        "category_math_hard",
                        "category_ifeval",
                        "category_gamedev",
                        "category_appdev",
                        "category_linux_admin",
                        "category_webdev",
                        "category_database",
                        "category_cpp",
                        "category_java",
                        "category_debugging",
                        "category_logic",
                        "category_retrogames",
                        "category_threedprint",
                        "category_languages",
                        "category_tvdev",
                        "category_uiux",
                        "category_office",
                        "category_life",
                        "category_biblical",
                    ]:
                        if cat in new_model:
                            for nt in new_model[cat].get("tests", []):
                                nt["last_run"] = current_time
                    if "performance_metrics" in new_model:
                        new_model["performance_metrics"]["last_run"] = current_time
                    merged_results.append(new_model)

            all_results["results"] = merged_results

            # Timestamped run file records ONLY the models benchmarked in this run.
            run_results = run_only_results if run_only_results else list(all_results["results"])
            save_file = (
                self.RESULTS_DIR
                / f"benchmarks_{time.strftime('%Y%m%d_%H%M%S')}_{mode}_{'proxy' if use_proxy else 'direct'}.json"
            )
            run_snapshot = dict(all_results)
            run_snapshot["results"] = run_results
            with open(save_file, "w") as f:
                json.dump(run_snapshot, f, indent=2, default=str)

            with open(latest_file, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

            print(f"\n{'=' * 80}")
            print("BENCHMARKING COMPLETE!")
            print(f"Results saved to: {save_file}")
            print(f"Merged history saved to: {latest_file}")
            print(f"{'=' * 80}")
            all_results["saved_as"] = str(save_file)

        # Emit benchmark_complete event
        if progress_callback:
            try:
                import inspect

                all_results["status"] = "completed"
                if inspect.iscoroutinefunction(progress_callback):
                    await progress_callback("benchmark_complete", all_results)
                else:
                    progress_callback("benchmark_complete", all_results)
            except Exception as e:
                print(f"Callback error: {e}")

        return all_results

    def run_optimization_pipeline(self, models: list[str]):
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
    parser.add_argument("--type", choices=["proxy", "direct", "both"], default="both", help="Endpoint target type.")

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
