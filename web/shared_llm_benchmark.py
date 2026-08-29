#!/usr/bin/env python3
"""
SharedLLM validation and benchmarking suite.
Evaluates LLM models (both local GPU and online providers) based on Jarvis / SharedLLM task patterns:
1. FastPath Intent Routing (HA Smart Lights, Geo/Location, Music Assistant, Climate, Raven Mission Dispatch)
2. Librarian Tool Use & Structured JSON Parameters (Nextcloud, RAG Knowledge Search, Geo-fence, Media)
3. Raven Autonomous Coding (MultiTenantLock Redis distributed lock, Async FastAPI routers, Self-healing bug fixes)
4. Raven Mission Planning & Reflection (DAG plan generation, Music Assistant stream troubleshooting)
5. Context Retention & Needle-in-a-Haystack verification
"""

import ast
import json
import os
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

sys.path.append(str(Path(__file__).resolve().parent.parent))
import context_awareness

# Per-model / benchmark-level reasoning settings resolve from the model profile
# (models.ini section / {alias}.profile.json) and the [benchmark] default
# section. No budget configured -> thinking off, no reasoning_budget sent.
from llm_benchmark_suite import (
    _effective_reasoning_budget,
    _effective_thinking,
    _model_sampling_options,
    _model_temperature,
)
from web.thermal import ThermalAbortError, ThermalWatchdog

# Per-chunk stream timeouts: read applies to the gap BETWEEN stream lines, not
# total request time. Large prompts (15k+ tokens) plus slow generation can run
# for many minutes; a hard non-streaming deadline kills healthy requests while
# llama-server is still generating (observed as "Connection handling canceled").
_STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0)


async def _read_chat_stream(resp: httpx.Response, watchdog: Any = None) -> dict:
    """Accumulate an Ollama-style NDJSON /api/chat stream into text + final metrics."""
    parts: list[str] = []
    think_parts: list[str] = []
    final: dict[str, Any] = {}
    async for line in resp.aiter_lines():
        if watchdog is not None:
            try:
                await watchdog.heartbeat()
            except ThermalAbortError:
                break
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
        "thermal_abort": bool(getattr(watchdog, "aborted", False)) and watchdog is not None,
    }


async def _read_generate_stream(resp: httpx.Response, watchdog: Any = None) -> dict:
    """Accumulate an Ollama-style NDJSON /api/generate stream into text + final metrics."""
    parts: list[str] = []
    think_parts: list[str] = []
    final: dict[str, Any] = {}
    async for line in resp.aiter_lines():
        if watchdog is not None:
            try:
                await watchdog.heartbeat()
            except ThermalAbortError:
                break
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
        "thermal_abort": watchdog is not None and bool(getattr(watchdog, "aborted", False)),
    }


online_model_provider: Any = None
try:
    from online_providers import online_model_provider
except Exception:
    try:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from online_providers import online_model_provider
    except Exception:
        online_model_provider = None


_THINKING_PATTERNS = [
    re.compile(r"<(think|thinking)>[\s\S]*?</\1>", re.IGNORECASE | re.DOTALL),
    re.compile(r" thinking.*? response", re.DOTALL),
]

# Canonical downstream SharedLLM tool names (ALLOWED_TOOLS in the gateway's
# agent_loop). The benchmark grades whether a model's structured tool call
# resolves to one of these, mirroring the app's own resolution tiers.
_CANONICAL_TOOLS = {
    "lightcontrolrequest",
    "mediaplayrequest",
    "mediatransportrequest",
    "mediastatusrequest",
    "tvcastrequest",
    "videoplayrequest",
    "climaterequest",
    "securityrequest",
    "announcementrequest",
    "haservicerequest",
    "calendarrequest",
    "noterequest",
    "timerrequest",
    "talkrequest",
    "websearchrequest",
    "webreadrequest",
    "webscraperrequest",
    "codesearchrequest",
    "dockerlogsrequest",
    "gitoperationrequest",
    "deploymentrequest",
    "capabilityindexrequest",
    "volumeinventoryrequest",
    "workspacefilereadrequest",
    "workspacefilewriterequest",
    "workspacefilepatchrequest",
    "workspacelintrequest",
    "workspacesearchrequest",
    "workspaceshellrequest",
    "storagefilereadrequest",
    "storagefilewriterequest",
    "storagelistrequest",
    "workspacebootstraprequest",
    "workspacecreaterequest",
    "workspacesettingsupdaterequest",
    "systemlearningrequest",
    "discoverysyncrequest",
    "storageindexrequest",
    "dockercomposerequest",
    "identityrequest",
    "identitymanagerequest",
    "controlplanerequest",
    "restart_service",
    "audiobookshelfrequest",
    "llminforequest",
    "contextsearchrequest",
    "haconfigrequest",
    "entitysearchrequest",
    "logbookrequest",
    "executionlogrequest",
    "documentbroadcastrequest",
    "nightmoderequest",
    "ttsrequest",
    "sttrequest",
    "storagetexttorequest",
    "audiobookregeneraterequest",
    "ghrequest",
    "ravenrecallrequest",
    "workspaceportexposerequest",
    "imagegenerationrequest",
}

# Tier-2 regex aliases mirroring the app (agent_loop.py lines ~3690-3735).
_TOOL_REGEX_ALIASES = [
    (re.compile(r".*light.*control.*"), "lightcontrolrequest"),
    (re.compile(r".*media.*play.*"), "mediaplayrequest"),
    (re.compile(r".*media.*transport.*"), "mediatransportrequest"),
    (re.compile(r".*media.*status.*"), "mediastatusrequest"),
    (re.compile(r".*tv.*cast.*"), "tvcastrequest"),
    (re.compile(r".*video.*play.*"), "videoplayrequest"),
    (re.compile(r".*climate.*"), "climaterequest"),
    (re.compile(r".*announce.*"), "announcementrequest"),
    (re.compile(r".*ha.*service.*"), "haservicerequest"),
    (re.compile(r".*calendar.*"), "calendarrequest"),
    (re.compile(r".*note.*create.*"), "noterequest"),
    (re.compile(r".*timer.*"), "timerrequest"),
    (re.compile(r".*talk.*"), "talkrequest"),
    (re.compile(r".*web.*search.*"), "websearchrequest"),
    (re.compile(r".*web.*read.*"), "webreadrequest"),
    (re.compile(r".*web.*scrap.*"), "webscraperrequest"),
    (re.compile(r".*code.*search.*"), "codesearchrequest"),
    (re.compile(r".*docker.*log.*"), "dockerlogsrequest"),
    (re.compile(r".*git.*"), "gitoperationrequest"),
    (re.compile(r".*deploy.*"), "deploymentrequest"),
    (re.compile(r".*workspace.*read.*"), "workspacefilereadrequest"),
    (re.compile(r".*workspace.*write.*"), "workspacefilewriterequest"),
    (re.compile(r".*workspace.*patch.*"), "workspacefilepatchrequest"),
    (re.compile(r".*workspace.*lint.*"), "workspacelintrequest"),
    (re.compile(r".*workspace.*search.*"), "workspacesearchrequest"),
    (re.compile(r".*workspace.*shell.*"), "workspaceshellrequest"),
    (re.compile(r".*storage.*list.*"), "storagelistrequest"),
    (re.compile(r".*storage.*read.*"), "storagefilereadrequest"),
    (re.compile(r".*storage.*write.*"), "storagefilewriterequest"),
    (re.compile(r".*context.*search.*"), "contextsearchrequest"),
    (re.compile(r".*rag.*"), "contextsearchrequest"),
    (re.compile(r".*image.*gen.*"), "imagegenerationrequest"),
    (re.compile(r".*sd.*gen.*"), "imagegenerationrequest"),
    (re.compile(r".*control.*plane.*"), "controlplanerequest"),
    (re.compile(r".*restart.*service.*"), "restart_service"),
    (re.compile(r".*identity.*"), "identityrequest"),
    (re.compile(r".*tts.*"), "ttsrequest"),
    (re.compile(r".*stt.*"), "sttrequest"),
    (re.compile(r".*git.*status.*"), "gitoperationrequest"),
]


def _resolve_tool_name(raw: str) -> str:
    """Resolve a model-emitted tool name to a canonical <x>request tool.

    Mirrors the downstream app's tiered resolution: exact match against the
    canonical set, then regex aliases, then fuzzy ``difflib`` closest match.
    Returns the empty string when nothing resolves.
    """
    if not raw:
        return ""
    name = str(raw).strip().strip("'\"")
    lower = name.lower()
    if name in _CANONICAL_TOOLS:
        return name
    for pattern, canonical in _TOOL_REGEX_ALIASES:
        if pattern.match(lower):
            return canonical
    import difflib

    close = difflib.get_close_matches(name, sorted(_CANONICAL_TOOLS), n=1, cutoff=0.6)
    return close[0] if close else ""


class SharedLLMModelBenchmark:
    """
    Validation benchmark harness based on SharedLLM tasks, AST code parsing,
    structured JSON validation, and multi-provider execution.
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
            self.PROXY_SERVER_URLS = [
                "http://localhost:11434",
                "http://alpaca-proxy:11434",
                "http://host.docker.internal:11434",
            ]

        self.RESULTS_DIR = Path("data/shared_llm_benchmarks")
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR = self.RESULTS_DIR / "models"
        self.ARTIFACTS_DIR = Path("data/artifacts")
        self.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Resolved backend context windows (model -> n_ctx), cached per instance.
        self._ctx_cache: dict[str, int] = {}

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

    def save_per_model_result(
        self,
        model_record: dict,
        use_proxy: bool,
        generated_at: str | None = None,
    ) -> Path | None:
        """Persist a single model's SharedLLM results to a per-model file so results follow the model."""
        model = model_record.get("model")
        if not model:
            return None
        per_model = {
            "benchmark_version": "SharedLLM-v2",
            "generated_at": generated_at or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "benchmark_type": "proxy" if use_proxy else "direct",
            "models_tested": 1,
            "per_model": True,
            "model": model,
            "results": [model_record],
        }
        file_path = self.MODELS_DIR / f"shared_{self._sanitize_model_filename(model)}.json"
        with open(file_path, "w") as f:
            json.dump(per_model, f, indent=2, default=str)
        return file_path

    def delete_model_results(self, model: str) -> bool:
        """Remove a model's per-model result file and its saved artifacts. Returns True if anything was removed."""
        if not model:
            return False
        removed = False

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
            file_path = self.MODELS_DIR / f"shared_{sv}.json"
            if file_path.exists():
                try:
                    file_path.unlink()
                    removed = True
                except Exception:
                    pass

        # 2. Check all remaining per-model files by inspecting their model payload
        if self.MODELS_DIR.exists():
            for pm_path in list(self.MODELS_DIR.glob("shared_*.json")):
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

    def strip_thinking(self, text: str) -> str:
        """Remove thinking blocks from a response.

        Mirrors the downstream SharedLLM gateway ``THINKING_PATTERNS``: both
        XML-style ``<thinking>...</thinking>`` tags and the prose
        `` thinking... response`` marker used by reasoning models. Thinking
        models still emit these even when ``think: False`` is set.
        """
        if not text:
            return ""
        cleaned = text
        for pattern in _THINKING_PATTERNS:
            cleaned = pattern.sub("", cleaned)
        return cleaned.strip()

    def clean_code_block(self, text: str) -> str:
        """Extract clean code from markdown code fences or raw text."""
        cleaned = self.strip_thinking(text)
        if "```python" in cleaned:
            return cleaned.split("```python")[1].split("```")[0].strip()
        elif "```py" in cleaned:
            return cleaned.split("```py")[1].split("```")[0].strip()
        elif "```" in cleaned:
            return cleaned.split("```")[1].split("```")[0].strip()
        return cleaned.strip()

    @staticmethod
    def _repair_json_control_chars(text: str) -> str:
        """Escape raw control characters inside JSON string values.

        Mirrors the downstream SharedLLM gateway ``_repair_json_control_chars``.
        Live models often emit literal newlines/tabs inside JSON strings which
        would otherwise make ``json.loads`` fail. Existing valid escapes are
        preserved.
        """
        if not text:
            return text

        control_escape = {"\n": "\\n", "\t": "\\t", "\r": "\\r"}

        def _esc_quoted(m: "re.Match[str]") -> str:
            raw = m.group(0)
            out: list[str] = []
            i = 0
            n = len(raw)
            while i < n:
                ch = raw[i]
                if ch == "\\" and i + 1 < n:
                    # Preserve an existing escape sequence verbatim.
                    out.append(ch)
                    out.append(raw[i + 1])
                    i += 2
                    continue
                if ch in control_escape:
                    out.append(control_escape[ch])
                    i += 1
                    continue
                out.append(ch)
                i += 1
            return "".join(out)

        return re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', _esc_quoted, text)

    @staticmethod
    def _extract_json_with_brace_depth(text: str) -> str:
        """Find the first balanced JSON object/array by tracking brace depth.

        Mirrors the downstream SharedLLM gateway ``_extract_json_with_brace_depth``.
        Unlike a greedy ``{.*}`` regex this stops exactly at the matching close
        brace, so trailing prose after the JSON payload is not swallowed.
        """
        if not text:
            return ""
        depth = 0
        start = -1
        in_str = False
        esc = False
        for i, ch in enumerate(text):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch in "{[":
                if depth == 0:
                    start = i
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth == 0 and start >= 0:
                    return text[start : i + 1]
        return text[start:].strip() if start >= 0 else ""

    def clean_json_block(self, text: str) -> str:
        """Extract clean JSON substring from markdown fences or raw text.

        Fenced code is preferred; otherwise a brace-depth scan extracts the
        first balanced JSON payload (mirroring the downstream app) instead of
        a greedy regex that can swallow trailing prose.
        """
        cleaned = self.strip_thinking(text)
        if "```json" in cleaned:
            return cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            return cleaned.split("```")[1].split("```")[0].strip()
        return self._extract_json_with_brace_depth(cleaned).strip()

    def validate_code(self, code: str, task_type: str = "redis_lock") -> dict:
        """Parse generated code via AST to verify structural correctness."""
        try:
            clean_code = self.clean_code_block(code)
            if task_type == "ha_yaml":
                # YAML (not Python) - validate structurally before the AST parse below.
                text = clean_code.lower()
                has_trigger = "trigger:" in text or "trigger" in text
                has_action = "action:" in text or "action" in text
                has_service = "service:" in text or "service" in text
                try:
                    import yaml

                    parsed = yaml.safe_load(clean_code)
                    valid_parse = parsed is not None
                except Exception:
                    parsed = None
                    # PyYAML may be unavailable; fall back to a structural substring check.
                    valid_parse = has_trigger and has_action and has_service
                is_complete = valid_parse and has_trigger and has_action and has_service
                return {
                    "valid_syntax": valid_parse,
                    "has_trigger": has_trigger,
                    "has_action": has_action,
                    "has_service": has_service,
                    "is_complete": is_complete,
                    "error": None if valid_parse else "YAML parse error",
                }
            tree = ast.parse(clean_code)

            if task_type == "redis_lock":
                has_class = False
                has_acquire = False
                has_release = False
                has_redis_usage = False
                has_ttl_handling = False
                has_import_redis = False
                code_lower = clean_code.lower()

                redis_patterns = [
                    "redis.redis",
                    "strictredis",
                    "from redis import",
                    "import redis",
                    "redis.cluster",
                    "redis_client",
                    "self.redis",
                    ".redis",
                    "redis.from_url",
                    "redis.Redis",
                    "redis.StrictRedis",
                ]
                ttl_patterns = [".expire(", ".setex(", "ex=", ".ttl(", "timeout", "nx=", "px="]

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and "lock" in node.name.lower():
                        has_class = True
                        for item in node.body:
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                if item.name == "acquire":
                                    has_acquire = True
                                if item.name == "release":
                                    has_release = True
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if "redis" in alias.name.lower():
                                has_import_redis = True
                    if isinstance(node, ast.ImportFrom) and node.module and "redis" in node.module.lower():
                        has_import_redis = True

                has_redis_usage = has_import_redis or any(p in code_lower for p in redis_patterns)
                has_ttl_handling = any(p in code_lower for p in ttl_patterns)
                is_complete = has_class and has_acquire and has_release and has_redis_usage and has_ttl_handling
                return {
                    "valid_syntax": True,
                    "has_class": has_class,
                    "has_acquire": has_acquire,
                    "has_release": has_release,
                    "has_redis_usage": has_redis_usage,
                    "has_ttl_handling": has_ttl_handling,
                    "is_complete": is_complete,
                    "error": None,
                }

            elif task_type == "async_api":
                has_fastapi_import = False
                has_pydantic = False
                has_async_func = False
                has_health_mention = "health" in clean_code.lower()

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            n = alias.name.lower()
                            if n == "fastapi" or n.startswith("fastapi."):
                                has_fastapi_import = True
                            if n == "pydantic" or n.startswith("pydantic."):
                                has_pydantic = True
                    if isinstance(node, ast.ImportFrom):
                        m = (node.module or "").lower()
                        if m == "fastapi" or m.startswith("fastapi."):
                            has_fastapi_import = True
                        if m == "pydantic" or m.startswith("pydantic."):
                            has_pydantic = True
                    if isinstance(node, ast.AsyncFunctionDef):
                        has_async_func = True

                is_complete = has_fastapi_import and has_pydantic and has_async_func
                return {
                    "valid_syntax": True,
                    "has_fastapi_import": has_fastapi_import,
                    "has_pydantic": has_pydantic,
                    "has_async_func": has_async_func,
                    "has_health_mention": has_health_mention,
                    "is_complete": is_complete,
                    "error": None,
                }

            else:  # General python syntax & function validation
                has_func = False
                has_real_body = False
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        has_func = True
                        body = node.body
                        if body and not (len(body) == 1 and isinstance(body[0], ast.Pass)):
                            has_real_body = True
                return {
                    "valid_syntax": True,
                    "has_func": has_func,
                    "has_real_body": has_real_body,
                    "is_complete": has_func and has_real_body,
                    "error": None,
                }

        except Exception as e:
            return {
                "valid_syntax": False,
                "is_complete": False,
                "error": str(e),
            }

    def validate_json_payload(self, text: str, required_keys: list[str] | None = None) -> dict:
        """Validates that output parses into JSON and optionally checks for required keys."""
        try:
            cleaned = self.clean_json_block(text)
            parsed = json.loads(self._repair_json_control_chars(cleaned))
            if required_keys:
                missing = [k for k in required_keys if k not in parsed]
                return {
                    "valid_json": True,
                    "has_required_keys": len(missing) == 0,
                    "missing_keys": missing,
                    "parsed": parsed,
                    "is_complete": len(missing) == 0,
                    "error": f"Missing keys: {missing}" if missing else None,
                }
            return {
                "valid_json": True,
                "has_required_keys": True,
                "parsed": parsed,
                "is_complete": True,
                "error": None,
            }
        except Exception as e:
            return {
                "valid_json": False,
                "has_required_keys": False,
                "parsed": None,
                "is_complete": False,
                "error": str(e),
            }

    @staticmethod
    def _watchdog_stats(watchdog: Any) -> dict:
        """Peak temp/throttle stats to merge into a test result record."""
        if watchdog is None:
            return {}
        return {"temps": watchdog.stats()}

    @staticmethod
    def _thermal_abort_result(watchdog: Any, latency: float, content: str, eval_cnt: int) -> dict:
        stats = watchdog.stats() if watchdog is not None else {"aborted": True}
        return {
            "success": False,
            "latency": latency,
            "response": content,
            "tokens_generated": eval_cnt,
            "error": f"thermal watchdog abort: {stats.get('abort_reason')}",
            "temps": stats,
            "thermal_aborted": True,
        }

    async def query_model(
        self,
        model: str,
        use_proxy: bool,
        prompt: str,
        max_tokens: int = 4000,
        custom_keys: dict[str, str] | None = None,
        watchdog: Any = None,
    ) -> dict:
        """Execute request against online provider, local proxy, or direct llama-server."""
        # 1. Route to Online Provider if model is an online identifier
        if online_model_provider.is_online_model(model):
            return await online_model_provider.query_online_model(
                model_identifier=model,
                prompt=prompt,
                max_tokens=max_tokens,
                custom_keys=custom_keys,
            )

        # 2. Local GPU Inference (Proxy or direct llama-server)
        urls = self.PROXY_SERVER_URLS if use_proxy else self.OLLAMA_SERVER_URLS

        # Context-window guard: resolve the live backend window and clamp the
        # generation budget so prompt + generation fit. Without this,
        # llama-server silently truncates (`n_tokens = N, truncated = 1`) and
        # the benchmark records empty responses on small-context hosts.
        messages = [{"role": "user", "content": prompt}]
        try:
            ctx = await context_awareness.resolve_context_window(
                model, self.PROXY_SERVER_URLS, self._ctx_cache, "shared-llm/benchmark"
            )
        except RuntimeError as e:
            return {
                "success": False,
                "latency": 0.0,
                "response": None,
                "tokens_generated": 0,
                "error": str(e),
            }
        est_tokens = context_awareness.estimate_prompt_tokens(messages)
        # Per-model / benchmark-level reasoning settings (profile wins, then
        # settable [benchmark] default; nothing configured = thinking off).
        thinking = _effective_thinking(model)
        reasoning_budget = _effective_reasoning_budget(model)
        if thinking and reasoning_budget:
            # Reasoning headroom: thinking phase + full answer must fit.
            effective_max_tokens = context_awareness.turn_budget(ctx, est_tokens, max_tokens + 2 * reasoning_budget)
        else:
            effective_max_tokens = context_awareness.turn_budget(ctx, est_tokens, max_tokens)
        if effective_max_tokens <= 0:
            return {
                "success": False,
                "latency": 0.0,
                "response": None,
                "tokens_generated": 0,
                "error": (
                    f"Context window exhausted before generation (ctx={ctx}, ~{est_tokens} prompt "
                    "tokens). Reduce the prompt size or increase the backend ctx-size."
                ),
            }

        last_error = None

        for base_url in urls:
            try:
                start_t = time.time()
                async with httpx.AsyncClient(timeout=_STREAM_TIMEOUT) as client:
                    if use_proxy:
                        payload = {
                            "model": model,
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": True,
                            "think": thinking,
                            "options": {
                                "num_predict": effective_max_tokens,
                                "temperature": _model_temperature(model),
                                **_model_sampling_options(model),
                            },
                        }
                        if reasoning_budget is not None:
                            payload["reasoning_budget"] = reasoning_budget
                        headers = self._proxy_headers({"X-Request-Source": "shared-llm/benchmark"})
                        async with client.stream("POST", f"{base_url}/api/chat", json=payload, headers=headers) as resp:
                            if resp.status_code != 200:
                                body = (await resp.aread()).decode("utf-8", "replace")
                                last_error = f"HTTP {resp.status_code}: {body[:300]}"
                                continue
                            data = await _read_chat_stream(resp, watchdog=watchdog)
                        latency = time.time() - start_t
                        content = self.strip_thinking(data["content"])
                        eval_cnt = data["eval_count"]

                        if data.get("thermal_abort"):
                            return self._thermal_abort_result(watchdog, latency, content, eval_cnt)

                        if eval_cnt >= effective_max_tokens:
                            try:
                                payload2 = {
                                    "model": model,
                                    "messages": [
                                        {"role": "user", "content": prompt},
                                        {"role": "assistant", "content": content},
                                        {
                                            "role": "user",
                                            "content": "[System: You are halfway through your token budget. Please come up with an answer quickly.]",
                                        },
                                    ],
                                    "stream": True,
                                    "think": thinking,
                                    "options": {
                                        "num_predict": effective_max_tokens,
                                        "temperature": _model_temperature(model),
                                        **_model_sampling_options(model),
                                    },
                                }
                                if reasoning_budget is not None:
                                    payload2["reasoning_budget"] = reasoning_budget
                                start_t2 = time.time()
                                async with client.stream(
                                    "POST", f"{base_url}/api/chat", json=payload2, headers=headers
                                ) as resp2:
                                    if resp2.status_code == 200:
                                        data2 = await _read_chat_stream(resp2, watchdog=watchdog)
                                        content = content + "\n" + self.strip_thinking(data2["content"])
                                        eval_cnt += data2["eval_count"]
                                        if data2.get("thermal_abort"):
                                            return self._thermal_abort_result(watchdog, latency, content, eval_cnt)
                                latency += time.time() - start_t2
                            except Exception as e2:
                                print(f"Phase 2 proxy query error in SharedLLM: {e2}")

                        return {
                            "success": True,
                            "latency": latency,
                            "response": content,
                            "tokens_generated": eval_cnt,
                            "error": None,
                            **self._watchdog_stats(watchdog),
                        }
                    else:
                        payload = {
                            "model": model,
                            "prompt": prompt,
                            "stream": True,
                            "think": thinking,
                            "options": {
                                "num_predict": effective_max_tokens,
                                "temperature": _model_temperature(model),
                                **_model_sampling_options(model),
                            },
                        }
                        if reasoning_budget is not None:
                            payload["reasoning_budget"] = reasoning_budget
                        async with client.stream("POST", f"{base_url}/api/generate", json=payload) as resp:
                            if resp.status_code != 200:
                                body = (await resp.aread()).decode("utf-8", "replace")
                                last_error = f"HTTP {resp.status_code}: {body[:300]}"
                                continue
                            data = await _read_generate_stream(resp, watchdog=watchdog)
                        latency = time.time() - start_t
                        content = self.strip_thinking(data["content"])
                        eval_cnt = data["eval_count"]

                        if data.get("thermal_abort"):
                            return self._thermal_abort_result(watchdog, latency, content, eval_cnt)

                        if eval_cnt >= effective_max_tokens:
                            try:
                                new_prompt = (
                                    f"{prompt}\n{content}\n"
                                    f"[System: You are halfway through your token budget. Please come up with an answer quickly.]"
                                )
                                payload2 = {
                                    "model": model,
                                    "prompt": new_prompt,
                                    "stream": True,
                                    "think": thinking,
                                    "options": {
                                        "num_predict": effective_max_tokens,
                                        "temperature": _model_temperature(model),
                                        **_model_sampling_options(model),
                                    },
                                }
                                if reasoning_budget is not None:
                                    payload2["reasoning_budget"] = reasoning_budget
                                start_t2 = time.time()
                                async with client.stream("POST", f"{base_url}/api/generate", json=payload2) as resp2:
                                    if resp2.status_code == 200:
                                        data2 = await _read_generate_stream(resp2, watchdog=watchdog)
                                        content = content + "\n" + self.strip_thinking(data2["content"])
                                        eval_cnt += data2["eval_count"]
                                        if data2.get("thermal_abort"):
                                            return self._thermal_abort_result(watchdog, latency, content, eval_cnt)
                                latency += time.time() - start_t2
                            except Exception as e2:
                                print(f"Phase 2 direct query error in SharedLLM: {e2}")

                        return {
                            "success": True,
                            "latency": latency,
                            "response": content,
                            "tokens_generated": eval_cnt,
                            "error": None,
                            **self._watchdog_stats(watchdog),
                        }
            except Exception as e:
                last_error = str(e)
                continue

        return {
            "success": False,
            "latency": 0.0,
            "response": None,
            "tokens_generated": 0,
            "error": last_error or "Endpoint unavailable",
            **self._watchdog_stats(watchdog),
        }

    @classmethod
    def get_all_tasks(cls) -> list[dict]:
        """Return all available SharedLLM task definitions."""
        return [
            # --- 1. FastPath Intent Routing ---
            {
                "id": "fast_path_light",
                "category": "FastPath Intent",
                "label": "HA Intent: turn on lights",
                "prompt": "/no_think You are a smart home intent classifier. Classify this request: 'turn on the office lights and set brightness to 75%'. Options: [light_on, light_off, thermostat_temp, geo_locate, media_play, dispatch_raven, conversational]. Respond with EXACTLY one of the options, nothing else.",
                "max_tokens": 100,
                "task_type": "intent",
                "expected": "light_on",
            },
            {
                "id": "fast_path_geo",
                "category": "FastPath Intent",
                "label": "Geo Intent: locate person",
                "prompt": "/no_think You are a smart home intent classifier. Classify this request: 'Where is Jeremiah right now?'. Options: [light_on, light_off, thermostat_temp, geo_locate, media_play, dispatch_raven, conversational]. Respond with EXACTLY one of the options, nothing else.",
                "max_tokens": 100,
                "task_type": "intent",
                "expected": "geo_locate",
            },
            {
                "id": "fast_path_media",
                "category": "FastPath Intent",
                "label": "Media Intent: Music Assistant playback",
                "prompt": "/no_think You are a smart home intent classifier. Classify this request: 'Play Bohemian Rhapsody by Queen in the living room'. Options: [light_on, light_off, thermostat_temp, geo_locate, media_play, dispatch_raven, conversational]. Respond with EXACTLY one of the options, nothing else.",
                "max_tokens": 100,
                "task_type": "intent",
                "expected": "media_play",
            },
            {
                "id": "fast_path_climate",
                "category": "FastPath Intent",
                "label": "Climate Intent: set thermostat",
                "prompt": "/no_think You are a smart home intent classifier. Classify this request: 'Set the upstairs thermostat to 72 degrees'. Options: [light_on, light_off, thermostat_temp, geo_locate, media_play, dispatch_raven, conversational]. Respond with EXACTLY one of the options, nothing else.",
                "max_tokens": 100,
                "task_type": "intent",
                "expected": "thermostat_temp",
            },
            {
                "id": "fast_path_raven_dispatch",
                "category": "FastPath Intent",
                "label": "Autonomous Intent: dispatch Raven mission",
                "prompt": "/no_think You are a smart home router. A user says: 'Please write a new microservice that scrapes our energy dashboard, computes daily totals, and saves a PDF report in Nextcloud'. Should this be handled by an inline quick tool or dispatched to Raven? Options: [inline_quick, dispatch_raven]. Respond with EXACTLY one option, nothing else.",
                "max_tokens": 100,
                "task_type": "intent",
                "expected": "dispatch_raven",
            },
            # --- 2. Librarian Structured Tool Calling ---
            {
                "id": "tool_nextcloud_list",
                "category": "Librarian Tools",
                "label": "Tool Selection: list Nextcloud files",
                "prompt": '/no_think Determine which tool and parameter to call for: \'list my invoice files in /documents/invoices on Nextcloud\'. Available tools: [ha_light_control(entity_id), nextcloud_list_files(directory), nextcloud_download_file(file_path), rag_search(query)]. Output ONLY valid JSON: {"tool": "nextcloud_list_files", "args": {"directory": "/documents/invoices"}}',
                "max_tokens": 150,
                "task_type": "tool_json",
                "required_tool": "nextcloud_list_files",
                "required_key": "directory",
            },
            {
                "id": "tool_rag_search",
                "category": "Librarian Tools",
                "label": "Tool Selection: RAG knowledge search",
                "prompt": '/no_think Determine which tool to call for: \'Search our documentation for server backup schedules\'. Available tools: [rag_search(query), nextcloud_list_files(directory), ha_switch_toggle(entity_id)]. Output ONLY valid JSON: {"tool": "rag_search", "args": {"query": "server backup schedules"}}',
                "max_tokens": 150,
                "task_type": "tool_json",
                "required_tool": "rag_search",
                "required_key": "query",
            },
            {
                "id": "tool_geo_fence",
                "category": "Librarian Tools",
                "label": "Tool Selection: Geo fence proximity",
                "prompt": '/no_think Determine which tool to call for: \'Check if Jeremiah has entered the home zone\'. Available tools: [geo_check_zone(person, zone), nextcloud_list_files(directory), media_pause(player)]. Output ONLY valid JSON: {"tool": "geo_check_zone", "args": {"person": "jeremiah", "zone": "home"}}',
                "max_tokens": 150,
                "task_type": "tool_json",
                "required_tool": "geo_check_zone",
                "required_key": "person",
            },
            {
                "id": "tool_sd_image_gen",
                "category": "Librarian Tools",
                "label": "Tool Selection: Stable Diffusion image gen",
                "prompt": '/no_think Determine tool and JSON arguments to generate an image for: \'Create a cinematic render of a cyberpunk server room in 8k\'. Available tools: [sd_generate_image(prompt, steps, cfg_scale), nextcloud_download_file(file_path), ha_light_control(entity_id)]. Output ONLY valid JSON: {"tool": "sd_generate_image", "args": {"prompt": "cyberpunk server room in 8k", "steps": 25, "cfg_scale": 7.0}}',
                "max_tokens": 150,
                "task_type": "tool_json",
                "required_tool": "sd_generate_image",
                "required_key": "prompt",
            },
            # --- 2b. Canonical Tool Request (downstream <x>request names) ---
            {
                "id": "tool_request_light_control",
                "category": "Librarian Tools",
                "label": "Canonical Tool: Home Assistant light control",
                "prompt": '/no_think Call the tool for: \'turn on the living room lights at 80 percent\'. Available tools: [lightcontrolrequest(entity_id, brightness_pct), mediaplayrequest(player, media_id), climaterequest(entity_id, target_temp)]. Output ONLY valid JSON: {"tool": "lightcontrolrequest", "args": {"entity_id": "light.living_room", "brightness_pct": 80}}',
                "max_tokens": 150,
                "task_type": "tool_request",
                "required_tool": "lightcontrolrequest",
                "required_args": ["entity_id", "brightness_pct"],
            },
            {
                "id": "tool_request_media_play",
                "category": "Librarian Tools",
                "label": "Canonical Tool: Music Assistant playback",
                "prompt": '/no_think Call the tool for: \'play the new album by Laufey in the kitchen\'. Available tools: [lightcontrolrequest(entity_id, brightness_pct), mediaplayrequest(player, media_id), climaterequest(entity_id, target_temp)]. Output ONLY valid JSON: {"tool": "mediaplayrequest", "args": {"player": "kitchen", "media_id": "album:laufey:beverly-whale"}}',
                "max_tokens": 150,
                "task_type": "tool_request",
                "required_tool": "mediaplayrequest",
                "required_args": ["player", "media_id"],
            },
            {
                "id": "tool_request_rag_search",
                "category": "Librarian Tools",
                "label": "Canonical Tool: RAG knowledge search",
                "prompt": '/no_think Call the tool for: \'search my learnings for the deadlock fix from last month\'. Available tools: [contextsearchrequest(query), storagelistrequest(path), gitoperationrequest(repo_url, action)]. Output ONLY valid JSON: {"tool": "contextsearchrequest", "args": {"query": "deadlock fix last month"}}',
                "max_tokens": 150,
                "task_type": "tool_request",
                "required_tool": "contextsearchrequest",
                "required_args": ["query"],
            },
            {
                "id": "tool_request_git_commit",
                "category": "Librarian Tools",
                "label": "Canonical Tool: Git operation",
                "prompt": '/no_think Call the tool for: \'stage and commit the changes in my raven-workspace repo\'. Available tools: [contextsearchrequest(query), gitoperationrequest(repo_url, action, message), workspacefilewriterequest(path, content)]. Output ONLY valid JSON: {"tool": "gitoperationrequest", "args": {"repo_url": "github.com/user/raven-workspace", "action": "commit", "message": "fix: resolve deadlock"}}',
                "max_tokens": 150,
                "task_type": "tool_request",
                "required_tool": "gitoperationrequest",
                "required_args": ["action"],
            },
            # --- 3. Raven Autonomous Coding & AST Validation ---
            {
                "id": "code_raven_redis_lock",
                "category": "Raven Code Gen",
                "label": "Raven: Redis MultiTenantLock",
                "prompt": "/no_think Write a Python class 'MultiTenantLock' that uses Redis (redis-py) to implement a distributed lock. The lock MUST be scoped to a 'user_id' and a 'resource_id'. It MUST have 'acquire' and 'release' methods. Ensure it handles timeouts and uses a TTL to prevent deadlocks. Provide ONLY the Python code, no explanation, no markdown preamble outside of backticks.",
                "max_tokens": 800,
                "task_type": "ast_code",
                "code_type": "redis_lock",
            },
            {
                "id": "code_raven_async_api",
                "category": "Raven Code Gen",
                "label": "Raven: FastAPI Async Health Router",
                "prompt": "/no_think Write a Python module using FastAPI that defines a Pydantic model 'HealthStatus' (fields: status, uptime, database_connected) and an async router function 'get_health_status' returning this model. Provide ONLY the Python code, no explanation.",
                "max_tokens": 700,
                "task_type": "ast_code",
                "code_type": "async_api",
            },
            {
                "id": "code_raven_self_healing",
                "category": "Raven Code Gen",
                "label": "Raven: Self-Healing Recursion Bug Fix",
                "prompt": "/no_think Fix this broken Python function causing RecursionError:\n\n```python\ndef compute_factorial(n):\n    return n * compute_factorial(n - 1)\n```\nProvide ONLY the complete, working Python function with proper base case handling, no explanation.",
                "max_tokens": 400,
                "task_type": "ast_code",
                "code_type": "general",
            },
            {
                "id": "code_evalplus_lru_memo",
                "category": "Raven Code Gen",
                "label": "Raven: Type-Checked LRU Memoized Fibonacci",
                "prompt": "/no_think Write a Python function 'fibonacci_memo' that calculates the nth Fibonacci number using @functools.lru_cache. It MUST include Python type annotations (n: int) -> int, a docstring, and handle negative integers by raising a ValueError. Output ONLY the Python code.",
                "max_tokens": 500,
                "task_type": "ast_code",
                "code_type": "general",
            },
            {
                "id": "code_raven_plan_multi_step",
                "category": "Raven Code Gen",
                "label": "Raven: Multi-Step Plan with Lesson Citations",
                "prompt": "/no_think Produce a short numbered execution plan to refactor the shared auth module into per-service microservices. The plan must list concrete steps (each on its own numbered line, <=20 lines total, tool names in CAPS), and MUST end with an 'Apply: [lesson-id]' citation line referencing a learned lesson for at least one step. Format:\n1. STEP DESCRIPTION WITH TOOL NAME IN CAPS\n2. ...\nApply: [lesson-xxxxxxxxxx]\nOutput ONLY the plan, no preamble.",
                "max_tokens": 300,
                "task_type": "raven_plan",
            },
            # --- 4. Troubleshooting & Diagnostics ---
            {
                "id": "troubleshoot_deadlock_diagnosis",
                "category": "Troubleshooting",
                "label": "Troubleshooting: Database Transaction Deadlock",
                "prompt": '/no_think Analyze this PostgreSQL log error:\n"ERROR: deadlock detected. Process 1284 waits for ExclusiveLock on relation orders_seq; blocked by process 1289. Process 1289 waits for ExclusiveLock on relation inventory; blocked by process 1284."\nIn JSON format with keys "root_cause" and "recommended_action", explain why the deadlock happened and provide the mitigation. Output ONLY valid JSON.',
                "max_tokens": 400,
                "task_type": "schema_json",
                "required_keys": ["root_cause", "recommended_action"],
            },
            {
                "id": "troubleshoot_media_stream_error",
                "category": "Troubleshooting",
                "label": "Troubleshooting: Music Assistant Stream Error",
                "prompt": '/no_think Analyze this Music Assistant log error:\n"StreamConnectionError: 404 stream URL expired for provider Tidal stream_id=89102."\nIn JSON format with keys "root_cause" and "recommended_action", explain the cause and provide the exact fix. Output ONLY valid JSON.',
                "max_tokens": 400,
                "task_type": "schema_json",
                "required_keys": ["root_cause", "recommended_action"],
            },
            {
                "id": "troubleshoot_git_unified_diff",
                "category": "Troubleshooting",
                "label": "Troubleshooting: Git Patch Generation",
                "prompt": "/no_think The following Python pagination function has an off-by-one error where the last page item is omitted:\n\n```python\ndef get_page(items, page, page_size):\n    start = (page - 1) * page_size\n    end = start + page_size - 1\n    return items[start:end]\n```\nProvide a unified git diff patch (with --- a/utils.py, +++ b/utils.py, and @@ markers) that fixes the slicing bug. Output ONLY the patch block.",
                "max_tokens": 400,
                "task_type": "diff_patch",
            },
            # --- 5. Media & Generative Studio ---
            {
                "id": "media_sd_prompt_crafting",
                "category": "Media & Graphics",
                "label": "Media: Stable Diffusion Generation Spec",
                "prompt": '/no_think Construct a structured Stable Diffusion generation request for: "An ultra-realistic photograph of a cozy log cabin in the snowy mountains at sunset". Output a JSON object with keys "prompt", "negative_prompt", "steps", "cfg_scale", "sampler", "width", and "height". Output ONLY valid JSON.',
                "max_tokens": 350,
                "task_type": "schema_json",
                "required_keys": ["prompt", "negative_prompt", "steps", "cfg_scale"],
            },
            {
                "id": "media_music_assistant_queue",
                "category": "Media & Graphics",
                "label": "Media: Music Assistant Queue Orchestration",
                "prompt": '/no_think Create a Music Assistant playback payload to queue an evening jazz playlist on the "Living Room Speakers" group with 5-second crossfade. Output a JSON object with keys "action", "target_player", "playlist", "crossfade_seconds", and "shuffle". Output ONLY valid JSON.',
                "max_tokens": 300,
                "task_type": "schema_json",
                "required_keys": ["action", "target_player", "crossfade_seconds"],
            },
            # --- 6. Word Processing & Document Automation ---
            {
                "id": "wordproc_executive_markdown_report",
                "category": "Word Processing & Data",
                "label": "WordProc: Structured Markdown Executive Briefing",
                "prompt": "/no_think Compile this system telemetry into an executive markdown document with headings (# Executive Briefing, ## VRAM Telemetry, ## Action Items) and a markdown table:\nGPU VRAM: 14.2 GB / 16.0 GB (88% utilized)\nProxy Latency: 142ms\nActive Slots: 4/4\nDropped Requests: 0\nProvide ONLY the structured markdown text with table.",
                "max_tokens": 500,
                "task_type": "markdown_report",
            },
            {
                "id": "wordproc_pandas_aggregation",
                "category": "Word Processing & Data",
                "label": "WordProc: Pandas CSV Data Synthesis",
                "prompt": "/no_think Write a Python function 'aggregate_hourly_energy(df)' using pandas that takes a DataFrame with columns ['timestamp', 'sensor_id', 'kwh'], converts 'timestamp' to datetime, groups by hour, calculates total kwh, and returns the top 3 peak hours as a DataFrame. Output ONLY the Python code.",
                "max_tokens": 600,
                "task_type": "ast_code",
                "code_type": "general",
            },
            # --- 7. Composite Multi-Step Chaining ---
            {
                "id": "chaining_morning_executive_routine",
                "category": "Composite Chaining",
                "label": "Chaining: Morning Routine (HA + Docs + Music)",
                "prompt": '/no_think Plan an end-to-end autonomous morning routine workflow that chains: 1) Query Nextcloud calendar for daily agenda, 2) Query Home Assistant outdoor weather sensor, 3) Synthesize a brief Markdown morning briefing, 4) Set bedroom lights to scene "Morning Rise", 5) Queue morning acoustic playlist on Music Assistant living room speaker. Output a JSON object with keys "workflow_name", "steps" (an ordered list of step objects each with "step_id", "service", "action", and "depends_on"), and "error_handling". Output ONLY valid JSON.',
                "max_tokens": 600,
                "task_type": "schema_json",
                "required_keys": ["workflow_name", "steps"],
            },
            {
                "id": "chaining_security_incident_dispatch",
                "category": "Composite Chaining",
                "label": "Chaining: Security Incident & Visual Dispatch",
                "prompt": '/no_think Design a multi-agent incident response chain for: Front door motion detected while all residents are geofenced away. Steps must chain: 1) Verify Life360 resident zones, 2) Capture camera snapshot and generate Stable Diffusion incident overlay thumbnail, 3) Home Assistant lock all entry doors and turn on porch floodlights, 4) Write security incident entry to Nextcloud /security/incidents.json, 5) Dispatch push notification to mobile phones. Output a JSON object with keys "incident_type", "priority", "pipeline_steps" (array of steps with "step", "target_system", "action"), and "contingency_action". Output ONLY valid JSON.',
                "max_tokens": 600,
                "task_type": "schema_json",
                "required_keys": ["incident_type", "pipeline_steps"],
            },
            # --- 8. Context Retention ---
            {
                "id": "needle_context_retrieval",
                "category": "Context Retention",
                "label": "Needle in Haystack: Secret Token Retrieval",
                "prompt": "/no_think System Log Buffer:\n[12:00:01] Service gateway starting on port 11435\n[12:00:02] Initializing DNS sync daemon\n[12:00:03] Discovery ping: found 14 Home Assistant entities\n[12:00:04] Security secret token: ALPHA-9842-OMEGA\n[12:00:05] Caddy reverse proxy upstream healthy\n[12:00:06] Identity service token refresh completed\n\nQuestion: What is the exact security secret token recorded in the log? Output ONLY the token string.",
                "max_tokens": 100,
                "task_type": "needle",
                "expected": "ALPHA-9842-OMEGA",
            },
            # --- 9. Raven Home Automation (HA mock) ---
            {
                "id": "raven_home_light_brightness",
                "category": "raven_home",
                "label": "HA Mock: dim living room lights to 40%",
                "prompt": '/think Mock Home Assistant state:\n[{"entity_id":"light.living_room","state":"on","attributes":{"brightness_pct":100}}]\nUser request: \'Dim the living room lights to 40 percent.\'\nOutput ONLY a single JSON Home Assistant service call: {"domain":"light","service":"turn_on","target":{"entity_id":"light.living_room"},"service_data":{"brightness_pct":40}}',
                "max_tokens": 200,
                "task_type": "ha_service",
                "expected_domain": "light",
                "expected_service": "turn_on",
                "expected_entity": "light.living_room",
            },
            {
                "id": "raven_home_climate_temp",
                "category": "raven_home",
                "label": "HA Mock: set upstairs thermostat to 72F",
                "prompt": '/think Mock Home Assistant state:\n[{"entity_id":"climate.upstairs","state":"heat","attributes":{"temperature":68}}]\nUser request: \'Set the upstairs thermostat to 72 degrees.\'\nOutput ONLY a single JSON Home Assistant service call: {"domain":"climate","service":"set_temperature","target":{"entity_id":"climate.upstairs"},"service_data":{"temperature":72}}',
                "max_tokens": 200,
                "task_type": "ha_service",
                "expected_domain": "climate",
                "expected_service": "set_temperature",
                "expected_entity": "climate.upstairs",
            },
            {
                "id": "raven_home_media_pause",
                "category": "raven_home",
                "label": "HA Mock: pause living room media player",
                "prompt": '/think Mock Home Assistant state:\n[{"entity_id":"media_player.living_room","state":"playing"}]\nUser request: \'Pause the living room speaker.\'\nOutput ONLY a single JSON Home Assistant service call: {"domain":"media_player","service":"media_pause","target":{"entity_id":"media_player.living_room"}}',
                "max_tokens": 200,
                "task_type": "ha_service",
                "expected_domain": "media_player",
                "expected_service": "media_pause",
                "expected_entity": "media_player.living_room",
            },
            {
                "id": "raven_home_cover_garage",
                "category": "raven_home",
                "label": "HA Mock: open garage door",
                "prompt": '/think Mock Home Assistant state:\n[{"entity_id":"cover.garage","state":"closed"}]\nUser request: \'Open the garage door.\'\nOutput ONLY a single JSON Home Assistant service call: {"domain":"cover","service":"open_cover","target":{"entity_id":"cover.garage"}}',
                "max_tokens": 200,
                "task_type": "ha_service",
                "expected_domain": "cover",
                "expected_service": "open_cover",
                "expected_entity": "cover.garage",
            },
            {
                "id": "raven_home_script_goodnight",
                "category": "raven_home",
                "label": "HA Mock: run goodnight script",
                "prompt": '/think Mock Home Assistant state:\n[{"entity_id":"script.goodnight","state":"off"}]\nUser request: \'Run the goodnight routine.\'\nOutput ONLY a single JSON Home Assistant service call: {"domain":"script","service":"turn_on","target":{"entity_id":"script.goodnight"}}',
                "max_tokens": 200,
                "task_type": "ha_service",
                "expected_domain": "script",
                "expected_service": "turn_on",
                "expected_entity": "script.goodnight",
            },
            # --- 10. Raven Coding (HA / Raven flavored) ---
            {
                "id": "raven_coding_ha_automation_yaml",
                "category": "raven_coding",
                "label": "Raven Coding: HA automation YAML",
                "prompt": "/think Write a Home Assistant automation in YAML that triggers when 'binary_sensor.front_door' changes to 'on' and, as the action, turns on 'light.porch' and notifies via the mobile app. Provide ONLY the YAML.",
                "max_tokens": 400,
                "task_type": "ast_code",
                "code_type": "ha_yaml",
            },
            {
                "id": "raven_coding_ha_service_call",
                "category": "raven_coding",
                "label": "Raven Coding: call HA light service in Python",
                "prompt": "/think Write a Python async function 'toggle_living_room_lights(hass, state)' that calls hass.services.call to turn 'light.living_room' on or off based on the boolean 'state'. Provide ONLY the Python code.",
                "max_tokens": 400,
                "task_type": "ast_code",
                "code_type": "general",
            },
            {
                "id": "raven_coding_fastapi_raven_proxy",
                "category": "raven_coding",
                "label": "Raven Coding: FastAPI router proxying a Raven mission",
                "prompt": "/think Write a Python module using FastAPI that defines a Pydantic model 'RavenMission' (fields: mission_id, description, priority) and an async router function 'create_mission' that accepts the model and returns it. Provide ONLY the Python code.",
                "max_tokens": 600,
                "task_type": "ast_code",
                "code_type": "async_api",
            },
            {
                "id": "raven_coding_self_heal_loop",
                "category": "raven_coding",
                "label": "Raven Coding: fix infinite loop bug",
                "prompt": "/think Fix this broken Python function that never terminates:\n\n```python\ni = 0\nwhile i < 5:\n    print(i)\n```\nProvide ONLY the complete, working Python function with a proper increment, no explanation.",
                "max_tokens": 300,
                "task_type": "ast_code",
                "code_type": "general",
            },
            # --- 11. Raven Media Creation (mock) ---
            {
                "id": "raven_media_audiobook_voice",
                "category": "raven_media",
                "label": "Media: audiobook TTS pipeline config",
                "prompt": '/think Given chapter text and a voice profile (en-US, medium quality, rate 1.0), output ONLY a JSON TTS/audiobook pipeline config with keys: engine, voice, rate, chapter_split. Example: {"engine":"piper","voice":"en_US-lessac-medium","rate":1.0,"chapter_split":true}',
                "max_tokens": 200,
                "task_type": "media_config",
                "required_keys": ["engine", "voice", "rate", "chapter_split"],
                "value_hints": {"engine": ["piper", "coqui", "xtts", "openai"]},
            },
            {
                "id": "raven_media_audiobook_split",
                "category": "raven_media",
                "label": "Media: audiobook long-chapter split config",
                "prompt": '/think A 90-minute chapter must be split into <=10 minute segments with a high-quality female voice. Output ONLY JSON: {"engine":"coqui","voice":"en_female_01","rate":1.0,"chapter_split":true,"max_segment_minutes":10}',
                "max_tokens": 200,
                "task_type": "media_config",
                "required_keys": ["engine", "voice", "rate", "chapter_split"],
                "value_hints": {"engine": ["piper", "coqui", "xtts", "openai"]},
            },
            {
                "id": "raven_media_image_poster",
                "category": "raven_media",
                "label": "Media: Stable Diffusion poster params",
                "prompt": '/think Generate image params for a cinematic synthwave poster of a smart home dashboard. Output ONLY JSON with keys: prompt, negative_prompt, steps, sampler, cfg_scale. Example: {"prompt":"cinematic synthwave smart home dashboard","negative_prompt":"blurry text","steps":25,"sampler":"dpmpp_2m","cfg_scale":7.0}',
                "max_tokens": 250,
                "task_type": "media_config",
                "required_keys": ["prompt", "negative_prompt", "steps", "sampler", "cfg_scale"],
                "value_hints": {"sampler": ["dpmpp_2m", "euler", "ddim", "uni_pc", "heun"], "steps_max": 150},
            },
            {
                "id": "raven_media_image_product",
                "category": "raven_media",
                "label": "Media: product render image params",
                "prompt": '/think Generate image params for a photorealistic render of a wall-mounted tablet running Home Assistant. Output ONLY JSON: {"prompt":"photorealistic wall tablet home assistant","negative_prompt":"cartoon","steps":30,"sampler":"euler","cfg_scale":7.5}',
                "max_tokens": 250,
                "task_type": "media_config",
                "required_keys": ["prompt", "negative_prompt", "steps", "sampler", "cfg_scale"],
                "value_hints": {"sampler": ["dpmpp_2m", "euler", "ddim", "uni_pc", "heun"], "steps_max": 150},
            },
            # --- 12. Raven Learning (mock tutoring) ---
            {
                "id": "raven_learning_python_gil",
                "category": "raven_learning",
                "label": "Learning: explain the Python GIL",
                "prompt": "/think Explain the Python Global Interpreter Lock to a junior developer in 2-3 sentences. Keep it accurate.",
                "max_tokens": 250,
                "task_type": "fact",
                "expected": "global interpreter lock",
            },
            {
                "id": "raven_learning_ha_trigger",
                "category": "raven_learning",
                "label": "Learning: what triggers a HA automation",
                "prompt": "/think In Home Assistant, what element of an automation defines when it runs? Answer in one sentence.",
                "max_tokens": 150,
                "task_type": "fact",
                "expected": "trigger",
            },
            {
                "id": "raven_learning_http_method",
                "category": "raven_learning",
                "label": "Learning: HTTP GET purpose",
                "prompt": "/think Which HTTP method is used to retrieve data from a server? Answer with the single method name.",
                "max_tokens": 100,
                "task_type": "fact",
                "expected": "get",
            },
            {
                "id": "raven_learning_water_freeze",
                "category": "raven_learning",
                "label": "Learning: freezing point of water",
                "prompt": "/think What is the freezing point of water in degrees Celsius? Answer with the number.",
                "max_tokens": 100,
                "task_type": "fact",
                "expected": "0",
            },
            # --- 13. Raven RAG (mock corpus) ---
            {
                "id": "raven_rag_nextcloud_backup",
                "category": "raven_rag",
                "label": "RAG: backup schedule from docs",
                "prompt": "/think Corpus:\ndoc_1: The gardening club meets Thursdays at 6pm.\ndoc_2: Nightly Nextcloud backups run at 02:00 to /backups/nc.\ndoc_3: The oven self-clean cycle takes 2 hours.\nQuery: When and where are Nextcloud backups stored? Answer using ONLY the corpus and cite the source doc id (e.g. doc_2).",
                "max_tokens": 200,
                "task_type": "rag",
                "expected": "02:00",
                "expected_source": "doc_2",
            },
            {
                "id": "raven_rag_meeting_day",
                "category": "raven_rag",
                "label": "RAG: club meeting day",
                "prompt": "/think Corpus:\ndoc_1: The gardening club meets Thursdays at 6pm.\ndoc_2: Nightly Nextcloud backups run at 02:00 to /backups/nc.\ndoc_3: The oven self-clean cycle takes 2 hours.\nQuery: What day does the gardening club meet? Answer using ONLY the corpus and cite the source doc id.",
                "max_tokens": 200,
                "task_type": "rag",
                "expected": "thursday",
                "expected_source": "doc_1",
            },
            {
                "id": "raven_rag_oven_cycle",
                "category": "raven_rag",
                "label": "RAG: oven self-clean duration",
                "prompt": "/think Corpus:\ndoc_1: The gardening club meets Thursdays at 6pm.\ndoc_2: Nightly Nextcloud backups run at 02:00 to /backups/nc.\ndoc_3: The oven self-clean cycle takes 2 hours.\nQuery: How long is the oven self-clean cycle? Answer using ONLY the corpus and cite the source doc id.",
                "max_tokens": 200,
                "task_type": "rag",
                "expected": "2 hours",
                "expected_source": "doc_3",
            },
            {
                "id": "raven_rag_backup_location",
                "category": "raven_rag",
                "label": "RAG: backup destination path",
                "prompt": "/think Corpus:\ndoc_1: The gardening club meets Thursdays at 6pm.\ndoc_2: Nightly Nextcloud backups run at 02:00 to /backups/nc.\ndoc_3: The oven self-clean cycle takes 2 hours.\nQuery: What is the destination path of the Nextcloud backups? Answer using ONLY the corpus and cite the source doc id.",
                "max_tokens": 200,
                "task_type": "rag",
                "expected": "/backups/nc",
                "expected_source": "doc_2",
            },
        ]

    async def run_shared_llm_benchmarks(
        self,
        models: list[str],
        use_proxy: bool,
        progress_callback: Callable[..., Any] | None = None,
        cancel_event=None,
        task_ids: list[str] | None = None,
        custom_keys: dict[str, str] | None = None,
    ) -> dict:
        """Run tasks for FastPath, Tool Use, Code Gen, and Mission Planning validation.

        task_ids: optional list of task IDs to run. If None or empty, all tasks run.
        """
        all_results: dict[str, Any] = {
            "benchmark_version": "SharedLLM-v2",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "benchmark_type": "proxy" if use_proxy else "direct",
            "models_tested": len(models),
            "results": [],
        }

        all_tasks = self.get_all_tasks()
        TASKS = [t for t in all_tasks if t["id"] in task_ids] if task_ids else all_tasks

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
        thermal_stop = False
        for model in models:
            if cancel_event and cancel_event.is_set():
                break
            if thermal_stop:
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

            model_record: dict[str, Any] = {
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

                # Query endpoint (local or online)
                prompt_val = task["prompt"] if isinstance(task["prompt"], str) else ""
                tokens_val = task["max_tokens"] if isinstance(task["max_tokens"], int) else 4000
                watchdog = ThermalWatchdog()
                await watchdog.pre_test_wait()
                res = await self.query_model(
                    model, use_proxy, prompt_val, tokens_val, custom_keys=custom_keys, watchdog=watchdog
                )

                if res.get("thermal_aborted"):
                    # Machine protection: a test hit the abort temperature -
                    # stop this model and the whole run immediately.
                    print(f"[thermal] stopping benchmark run: {res.get('error')}")
                    all_results["status"] = "thermal_abort"
                    thermal_stop = True
                    model_record["tasks"].append(
                        {
                            "test_id": task["id"],
                            "test_category": task["category"],
                            "test_label": task["label"],
                            "success": False,
                            "latency": res.get("latency", 0.0),
                            "tokens_generated": res.get("tokens_generated", 0),
                            "prompt": task["prompt"],
                            "response": res.get("response"),
                            "error": res.get("error"),
                            "validation": {},
                            "temps": res.get("temps"),
                        }
                    )
                    break

                # Custom evaluations for SharedLLM tiers
                validation_results: dict[str, Any] = {}
                task_type = task.get("task_type", "general")

                if res["success"]:
                    response_text = self.strip_thinking(res["response"] or "").strip()

                    if task_type == "intent":
                        expected = task.get("expected", "")
                        lower_resp = response_text.lower()
                        known_intents = [
                            "light_on",
                            "light_off",
                            "geo_locate",
                            "media_play",
                            "thermostat_temp",
                            "dispatch_raven",
                            "conversational",
                        ]
                        matched = [i for i in known_intents if i in lower_resp]
                        # A response echoing multiple intent options is a copy of the
                        # prompt options list, not a routed decision.
                        passed = (
                            expected.lower() in lower_resp and len(matched) <= 1 and len(response_text.strip()) <= 200
                        )
                        validation_results = {
                            "correct_intent": passed,
                            "expected": expected,
                            "matched": matched,
                            "actual": response_text,
                        }
                        res["success"] = passed

                    elif task_type == "tool_json":
                        req_tool = task.get("required_tool", "")
                        req_key = task.get("required_key", "")
                        json_val = self.validate_json_payload(response_text)
                        parsed = json_val.get("parsed") or {}
                        tool_matched = req_tool.lower() in str(parsed.get("tool", "")).lower() if req_tool else True
                        args_obj = parsed.get("args") or {}
                        raw_val = args_obj.get(req_key) if req_key else None
                        key_matched = bool(req_key) and raw_val not in (None, "", [], {})
                        passed = json_val["valid_json"] and tool_matched and key_matched
                        validation_results = {
                            "valid_json": json_val["valid_json"],
                            "tool_match": tool_matched,
                            "key_match": key_matched,
                            "parsed": parsed,
                            "error": json_val.get("error"),
                        }
                        res["success"] = passed

                    elif task_type == "tool_request":
                        # Grade structured tool calls against the downstream
                        # app's canonical <x>request tool names, using the same
                        # tiered resolution the gateway applies (exact, regex
                        # alias, fuzzy). Required args must all be present and
                        # non-empty.
                        req_tool = task.get("required_tool", "")
                        req_args = task.get("required_args", [])
                        json_val = self.validate_json_payload(response_text)
                        parsed = json_val.get("parsed") or {}
                        emitted = str(parsed.get("tool", parsed.get("action", "")))
                        resolved = _resolve_tool_name(emitted)
                        tool_matched = bool(req_tool) and resolved == req_tool
                        args_obj = parsed.get("args") or parsed.get("payload") or {}
                        args_matched = all(args_obj.get(k) not in (None, "", [], {}) for k in req_args)
                        passed = json_val["valid_json"] and tool_matched and args_matched
                        validation_results = {
                            "valid_json": json_val["valid_json"],
                            "emitted_tool": emitted,
                            "resolved_tool": resolved,
                            "expected_tool": req_tool,
                            "tool_match": tool_matched,
                            "args_match": args_matched,
                            "parsed": parsed,
                            "error": json_val.get("error"),
                        }
                        res["success"] = passed

                    elif task_type == "raven_plan":
                        # Grade a Raven planning response in the downstream app's
                        # format: a short numbered prose plan (<=20 lines) with
                        # at least one required "Apply: [lesson-id]" citation.
                        lines = [ln.strip() for ln in response_text.splitlines() if ln.strip()]
                        numbered_steps = sum(1 for ln in lines if re.match(r"^\d+[\.\)]", ln))
                        apply_cites = [ln for ln in lines if re.match(r"^Apply:\s*\[?lesson-", ln, re.IGNORECASE)]
                        passed = numbered_steps >= 1 and len(lines) <= 20 and len(apply_cites) >= 1
                        validation_results = {
                            "line_count": len(lines),
                            "numbered_steps": numbered_steps,
                            "apply_citations": len(apply_cites),
                            "is_complete": passed,
                        }
                        res["success"] = passed

                    elif task_type == "schema_json":
                        req_keys = task.get("required_keys", [])
                        json_val = self.validate_json_payload(response_text, required_keys=req_keys)
                        parsed = json_val.get("parsed") or {}
                        values_non_empty = all(parsed.get(k) not in (None, "", [], {}) for k in req_keys)
                        json_val["values_non_empty"] = values_non_empty
                        json_val["is_complete"] = json_val["is_complete"] and values_non_empty
                        validation_results = json_val
                        res["success"] = json_val["is_complete"]

                    elif task_type == "ast_code":
                        code_type = task.get("code_type", "redis_lock")
                        validation_results = self.validate_code(response_text, task_type=code_type)
                        res["success"] = validation_results["is_complete"]

                    elif task_type == "diff_patch":
                        # A real unified diff has file headers AND at least one hunk
                        # marker (or a diff --git header for new files).
                        has_diff_headers = (
                            "---" in response_text
                            and "+++" in response_text
                            and ("@@" in response_text or "diff --git" in response_text)
                        )
                        validation_results = {
                            "valid_patch_format": has_diff_headers,
                            "is_complete": has_diff_headers,
                        }
                        res["success"] = has_diff_headers

                    elif task_type == "markdown_report":
                        lines = response_text.splitlines()
                        has_heading_line = any(line.strip().startswith("#") for line in lines)
                        has_table = "|" in response_text and ("---" in response_text or "--" in response_text)
                        passed = has_heading_line and has_table
                        validation_results = {
                            "has_headings": has_heading_line,
                            "has_table": has_table,
                            "is_complete": passed,
                        }
                        res["success"] = passed

                    elif task_type == "needle":
                        expected = task.get("expected", "")
                        # A genuine retrieval answer is a short extract; echoing the
                        # entire prompt log (which also contains the token) is not.
                        passed = expected in response_text and len(response_text.strip()) <= 1000
                        validation_results = {
                            "needle_found": passed,
                            "expected": expected,
                            "actual": response_text,
                        }
                        res["success"] = passed

                    elif task_type == "ha_service":
                        exp_domain = task.get("expected_domain", "").lower()
                        exp_service = task.get("expected_service", "").lower()
                        exp_entity = task.get("expected_entity", "").lower()
                        json_val = self.validate_json_payload(response_text)
                        parsed = json_val.get("parsed")
                        calls = parsed if isinstance(parsed, list) else ([parsed] if isinstance(parsed, dict) else [])
                        ha_matched = False
                        for call in calls:
                            if not isinstance(call, dict):
                                continue
                            dom = str(call.get("domain", "")).lower()
                            svc = str(call.get("service", "")).lower()
                            tgt = call.get("target") or {}
                            ent = str(tgt.get("entity_id", "")).lower() if isinstance(tgt, dict) else str(tgt).lower()
                            if dom == exp_domain and svc == exp_service and exp_entity in ent:
                                ha_matched = True
                                break
                        validation_results = {
                            "valid_json": json_val["valid_json"],
                            "domain_match": ha_matched,
                            "expected": {"domain": exp_domain, "service": exp_service, "entity": exp_entity},
                            "parsed": parsed,
                        }
                        res["success"] = json_val["valid_json"] and ha_matched

                    elif task_type == "media_config":
                        req_keys = task.get("required_keys", [])
                        json_val = self.validate_json_payload(response_text, required_keys=req_keys)
                        parsed = json_val.get("parsed") or {}
                        hints = task.get("value_hints", {})
                        value_ok = True

                        def _val_matches(val, allowed):
                            v = str(val).lower()
                            return any(v == a.lower() or v.startswith(a.lower()) or a.lower() in v for a in allowed)

                        if "engine" in hints:
                            value_ok = value_ok and _val_matches(parsed.get("engine", ""), hints["engine"])
                        if "sampler" in hints:
                            value_ok = value_ok and _val_matches(parsed.get("sampler", ""), hints["sampler"])
                        if "steps_max" in hints:
                            try:
                                value_ok = value_ok and 1 <= int(float(parsed.get("steps", 0))) <= int(
                                    hints["steps_max"]
                                )
                            except (TypeError, ValueError):
                                value_ok = False
                        json_val["value_ok"] = value_ok
                        json_val["is_complete"] = json_val["is_complete"] and value_ok
                        validation_results = json_val
                        res["success"] = json_val["is_complete"]

                    elif task_type == "fact":
                        expected = task.get("expected", "")
                        # Accept a case-insensitive phrase match, or the canonical
                        # acronym when the expected term is the Global Interpreter Lock
                        # (models often answer with "GIL" instead of the spelled-out phrase).
                        passed = expected.lower() in response_text.lower() or (
                            "gil" in response_text.lower() and "interpreter" in response_text.lower()
                        )
                        validation_results = {"expected": expected, "actual": response_text, "is_complete": passed}
                        res["success"] = passed

                    elif task_type == "rag":
                        expected = task.get("expected", "")
                        exp_source = task.get("expected_source", "").lower()
                        answer_ok = expected.lower() in response_text.lower()
                        source_ok = exp_source in response_text.lower()
                        passed = answer_ok and source_ok
                        validation_results = {
                            "expected": expected,
                            "expected_source": exp_source,
                            "answer_found": answer_ok,
                            "source_cited": source_ok,
                            "is_complete": passed,
                        }
                        res["success"] = passed

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
                    "temps": res.get("temps"),
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

            results_list = all_results["results"]
            if isinstance(results_list, list):
                results_list.append(model_record)
            try:
                self.save_per_model_result(model_record, use_proxy, all_results.get("generated_at"))
            except Exception as e:
                print(f"Callback error: failed to save per-model result for {model}: {e}")
                raise
            if progress_callback:
                try:
                    import inspect

                    if inspect.iscoroutinefunction(progress_callback):
                        await progress_callback("model_complete", {"model": model, "results": model_record})
                    else:
                        progress_callback("model_complete", {"model": model, "results": model_record})
                except Exception as e:
                    print(f"Callback error: {e}")

        if cancel_event and cancel_event.is_set():
            all_results["status"] = "cancelled"
        elif all_results.get("status") != "thermal_abort":
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
