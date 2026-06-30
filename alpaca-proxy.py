import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Precise logging setup with memory buffer
LOG_BUFFER = deque(maxlen=1000)


class DequeHandler(logging.Handler):
    def emit(self, record):
        LOG_BUFFER.append(self.format(record))


logger = logging.getLogger("alpaca-proxy")
logger.setLevel(logging.INFO)
if not logger.handlers:
    stream_handler = logging.StreamHandler()
    deque_handler = DequeHandler()
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - [%(request_id)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    stream_handler.setFormatter(formatter)
    deque_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(deque_handler)


# Custom filter to ensure request_id is always present
class RequestIDFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "N/A"
        return True


logger.addFilter(RequestIDFilter())

client_httpx = None
model_expires_at = {}
model_unload_tasks = {}
router_management_supported = None
router_model_lock = asyncio.Lock()

# Configuration
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://llama-server:8080")
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "/models")
MODEL_NAMESPACE = os.getenv("MODEL_NAMESPACE", "registry.ollama.ai")
ENGINE_STARTUP_TIMEOUT_SECONDS = int(os.getenv("ENGINE_STARTUP_TIMEOUT_SECONDS", "300"))
API_VERSION = os.getenv("API_VERSION", "0.3.1")
DEFAULT_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
MAX_LOADED_MODELS = int(os.getenv("MAX_LOADED_MODELS", "1"))
ROUTER_MODELS_URL = f"{LLAMA_SERVER_URL}/models"
ROUTER_MODELS_DIR = os.getenv("ROUTER_MODELS_DIR", "/router-models")
LLAMA_SERVER_CONNECT_TIMEOUT_SECONDS = float(
    os.getenv("LLAMA_SERVER_CONNECT_TIMEOUT_SECONDS", "60")
)
LLAMA_SERVER_READ_TIMEOUT_SECONDS = os.getenv(
    "LLAMA_SERVER_READ_TIMEOUT_SECONDS", "600"
).strip()  # Default to 600s
FOREVER_EXPIRES_AT = "9999-12-31T23:59:59Z"
LOADED_MODELS_STATE_FILE = os.path.join(ROUTER_MODELS_DIR, ".loaded-models.json")

# Reference counting for active requests to prevent unloading in-use models
active_requests = {}
active_requests_lock = asyncio.Condition()

# Thread-safe detailed request tracking (for user query/summary)
import threading

active_request_details = {}
completed_requests = []
active_request_details_lock = threading.Lock()

# Persistent request storage for resubmit (never rotates out)
resubmittable_requests = {}
resubmittable_requests_lock = threading.Lock()


def sanitize_prompt(text: str) -> str:
    if not text:
        return ""
    # Redact common patterns for passwords, tokens, API keys
    # 1. API Keys (e.g. sk-..., gpt-..., AIzaSy...)
    text = re.sub(r"(sk-[a-zA-Z0-9]{20,})", "[REDACTED_API_KEY]", text)
    text = re.sub(r"(AIzaSy[a-zA-Z0-9_-]{33})", "[REDACTED_API_KEY]", text)
    # 2. Key-value pairs for passwords/tokens/secrets in JSON or text
    text = re.sub(
        r'(?i)\b(pass|password|passwd|secret|key|token|auth|credentials)\s*[:=]\s*["\']([^"\']{4,})["\']',
        r'\1 = "[REDACTED]"',
        text,
    )
    return text


def register_active_request(request_id, model, req_type, payload, request_source="unknown", client_ip="unknown"):
    prompt_str = ""
    if "messages" in payload:
        msgs = payload["messages"]
        formatted_msgs = []
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            formatted_msgs.append(f"{role.upper()}: {content}")
        prompt_str = "\n".join(formatted_msgs)
    elif "prompt" in payload:
        prompt_str = str(payload["prompt"])

    prompt_str = sanitize_prompt(prompt_str)

    with active_request_details_lock:
        active_request_details[request_id] = {
            "request_id": request_id,
            "model": model,
            "type": req_type,
            "started_at": time.time(),
            "prompt": prompt_str,
            "thinking": "",
            "response": "",
            "request_source": request_source,
            "client_ip": client_ip,
        }


def update_active_request_progress(request_id, response_chunk=None, thinking_chunk=None):
    with active_request_details_lock:
        if request_id in active_request_details:
            req = active_request_details[request_id]
            if thinking_chunk:
                req["thinking"] += thinking_chunk
            if response_chunk:
                req["response"] += response_chunk
            # Calculate TTFT (Time to First Token) on first content/thinking chunk
            if (thinking_chunk or response_chunk) and "ttft_seconds" not in req:
                req["ttft_seconds"] = round(time.time() - req["started_at"], 3)


def complete_active_request(request_id, final_response=None, final_thinking=None, prompt_tokens=0, completion_tokens=0):
    req = None
    with active_request_details_lock:
        if request_id in active_request_details:
            req = active_request_details.pop(request_id)
            if final_response:
                req["response"] = final_response
            if final_thinking:
                req["thinking"] = final_thinking
            
            # calculate prompt and completion tokens if not provided
            if not completion_tokens and req["response"]:
                completion_tokens = max(1, int(len(req["response"]) / 4))
                if req["thinking"]:
                    completion_tokens += max(1, int(len(req["thinking"]) / 4))
            if not prompt_tokens and req.get("prompt"):
                prompt_tokens = max(1, int(len(req["prompt"]) / 4))
            
            req["completed_at"] = time.time()
            req["duration_seconds"] = round(req["completed_at"] - req["started_at"], 2)
            
            # calculate tps
            if req["duration_seconds"] > 0:
                req["tps"] = round(completion_tokens / req["duration_seconds"], 2)
            else:
                req["tps"] = 0.0
                
            # set ttft for non-streaming if not set
            if "ttft_seconds" not in req:
                req["ttft_seconds"] = req["duration_seconds"]
                
            req["completion_tokens"] = completion_tokens
            req["prompt_tokens"] = prompt_tokens
            
            completed_requests.append(req)
            if len(completed_requests) > 50:
                completed_requests.pop(0)

    if req is not None:
        with resubmittable_requests_lock:
            resubmittable_requests[request_id] = dict(req)
    return req


# Model loading state tracking (for OOM recovery)
model_loading = {}
MODEL_LOADING_TIMEOUT = 120  # seconds

# Track crashed models to prevent deadlock during recovery
# When a model crashes, this is set so other models don't try to unload it
crashed_models = {}  # {backend_model: True}

# Model usage tracking (load/unload frequency, popularity)
model_usage_log = []
MODEL_USAGE_LOG_MAX = 1000
model_usage_lock = asyncio.Lock()

# Persisted loaded models state (for autoload on recovery)
_loaded_models_state_lock = asyncio.Lock()


def handle_background_task_result(task: asyncio.Task):
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Background task '{task.get_name()}' failed: {e}", exc_info=True)


class RouterManagementUnsupported(RuntimeError):
    pass


def model_manifest_base():
    return os.path.join(OLLAMA_BASE, "manifests", MODEL_NAMESPACE)


def with_default_tag(model_name):
    if ":" not in model_name:
        return f"{model_name}:latest"
    return model_name


def public_model_name(model_name):
    resolved = with_default_tag(model_name)
    if resolved.endswith(":latest"):
        return resolved[:-7]
    return resolved


def model_manifest_paths(model_name):
    repo_tag = with_default_tag(model_name).replace(":", os.sep)
    return [
        os.path.join(model_manifest_base(), "library", repo_tag),
        os.path.join(model_manifest_base(), repo_tag),
    ]


def blob_path_for_digest(digest):
    return os.path.join(OLLAMA_BASE, "blobs", digest.replace(":", "-"))


def router_filename_for_model_name(model_name):
    normalized = with_default_tag(model_name)
    name, tag = normalized.rsplit(":", 1)
    flattened = f"{name}--{tag}".replace("/", "--")
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", flattened).strip("-.") or "model"
    return f"{sanitized}.gguf"


def router_model_id_for_name(model_name):
    # llama-server requires the exact filename (including .gguf) for lazy loading from --models-dir
    return router_filename_for_model_name(model_name)


def router_path_for_model_name(model_name):
    return os.path.join(ROUTER_MODELS_DIR, router_filename_for_model_name(model_name))


def normalize_digest(digest):
    return digest.split(":", 1)[1] if ":" in digest else digest


def modified_at_for_path(path):
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).astimezone().isoformat()
    except OSError:
        return utc_now()


def is_model_complete(manifest):
    """Verifies that all blobs in the manifest exist and are the correct size."""
    try:
        layers = manifest.get("layers", [])
        config = manifest.get("config", {})
        for layer in layers + [config]:
            digest = layer.get("digest")
            if not digest:
                continue
            blob_path = blob_path_for_digest(digest)
            if not os.path.exists(blob_path):
                return False
            if os.path.getsize(blob_path) != layer.get("size", 0):
                return False
        return True
    except:
        return False


def read_manifest(path):
    """Load a manifest file, returning None for partial or malformed files."""
    try:
        with open(path, "r") as f:
            manifest = json.load(f)
        if isinstance(manifest, dict):
            return manifest
    except (OSError, json.JSONDecodeError):
        return None
    return None


def iter_local_manifests():
    manifest_base = model_manifest_base()
    if not os.path.exists(manifest_base):
        return
    for root, dirs, files in os.walk(manifest_base):
        for file in files:
            if "sha256" in file:
                continue
            path = os.path.join(root, file)
            manifest = read_manifest(path)
            if manifest is None or not is_model_complete(manifest):
                continue
            yield manifest_base, path, manifest


def manifest_model_name(manifest_base, path):
    rel_path = os.path.relpath(path, manifest_base)
    parts = rel_path.split("/")
    if len(parts) < 2:
        return None
    tag = parts[-1]
    name = "/".join(parts[:-1])
    if name.startswith("library/"):
        name = name[8:]
    return public_model_name(f"{name}:{tag}")


def read_config_blob(manifest):
    config = manifest.get("config") or {}
    digest = config.get("digest")
    if not digest:
        return {}
    blob_path = blob_path_for_digest(digest)
    blob = read_manifest(blob_path)
    return blob or {}


def model_info_from_config(config_blob):
    if isinstance(config_blob.get("model_info"), dict):
        return config_blob["model_info"]

    model_info = {}
    for key, value in config_blob.items():
        if "." in key:
            model_info[key] = value
    return model_info


def context_length_from_config(config_blob):
    model_info = model_info_from_config(config_blob)
    if isinstance(model_info.get("general.context_length"), int):
        return model_info["general.context_length"]
    for key, value in model_info.items():
        if key.endswith(".context_length") and isinstance(value, int):
            return value
    if isinstance(config_blob.get("context_length"), int):
        return config_blob["context_length"]
    return None


def model_capabilities_from_manifest(manifest):
    config_blob = read_config_blob(manifest)
    capabilities = ["completion"]
    model_info = model_info_from_config(config_blob)
    if any(".vision." in key or key.endswith(".mm.tokens_per_image") for key in model_info):
        capabilities.append("vision")
    return capabilities


def model_details_from_manifest(manifest):
    config_blob = read_config_blob(manifest)
    details = {}

    for source_key, detail_key in (
        ("model_format", "format"),
        ("model_family", "family"),
        ("families", "families"),
        ("parameter_size", "parameter_size"),
        ("quantization_level", "quantization_level"),
        ("parent_model", "parent_model"),
    ):
        value = config_blob.get(source_key)
        if value not in (None, "", []):
            details[detail_key] = value

    if "format" not in details:
        media_types = [layer.get("mediaType", "") for layer in manifest.get("layers", [])]
        if any("gguf" in media_type for media_type in media_types):
            details["format"] = "gguf"

    return details


def manifest_layer(manifest, prefix):
    for layer in manifest.get("layers", []):
        if layer.get("mediaType", "").startswith(prefix):
            return layer
    return {}


def manifest_stats(manifest_path, manifest):
    model_layer = manifest_layer(manifest, "application/vnd.ollama.image.model")
    config_blob = read_config_blob(manifest)
    return {
        "size": model_layer.get("size", 0),
        "digest": normalize_digest(model_layer.get("digest", "")),
        "details": model_details_from_manifest(manifest),
        "modified_at": modified_at_for_path(manifest_path),
        "context_length": context_length_from_config(config_blob),
        "capabilities": model_capabilities_from_manifest(manifest),
        "model_info": model_info_from_config(config_blob),
        "template": config_blob.get("template", ""),
        "system": config_blob.get("system", ""),
        "license": config_blob.get("license", ""),
        "parameters": config_blob.get("parameters", ""),
    }


def manifest_path_for_model(model_name):
    for candidate in model_manifest_paths(model_name):
        if os.path.exists(candidate):
            return candidate
    return None


def load_local_manifest(model_name, require_complete=True):
    manifest_path = manifest_path_for_model(model_name)
    if not manifest_path:
        return None, None
    manifest = read_manifest(manifest_path)
    if manifest is None:
        return manifest_path, None
    if require_complete and not is_model_complete(manifest):
        return manifest_path, None
    return manifest_path, manifest


def router_model_candidates(model_name, manifest):
    candidates = []
    router_path = router_path_for_model_name(model_name)
    candidates.extend(
        [
            router_path,
            os.path.basename(router_path),
            os.path.splitext(os.path.basename(router_path))[0],
        ]
    )
    model_layer = manifest_layer(manifest, "application/vnd.ollama.image.model")
    digest = model_layer.get("digest", "")
    if digest:
        blob_path = blob_path_for_digest(digest)
        candidates.extend(
            [
                os.path.basename(blob_path),
                blob_path,
                digest,
                normalize_digest(digest),
            ]
        )
    candidates.extend(
        [
            public_model_name(model_name),
            with_default_tag(model_name),
        ]
    )
    return [
        candidate
        for i, candidate in enumerate(candidates)
        if candidate and candidate not in candidates[:i]
    ]


def router_entry_status(entry):
    return ((entry.get("status") or {}).get("value") or "").lower()


def router_entry_values(entry):
    values = []
    model_id = entry.get("id")
    path = entry.get("path")
    if model_id:
        values.append(model_id)
        values.append(os.path.basename(model_id))
    if path:
        values.append(path)
        values.append(os.path.basename(path))
    return {value for value in values if value}


def router_entry_matches(entry, candidates):
    entry_values = router_entry_values(entry)
    for candidate in candidates:
        if candidate in entry_values:
            return True
    return False


OLLAMA_OPTION_MAP = {
    "num_predict": "n_predict",
    "stop": "stop",
    "temperature": "temperature",
    "top_k": "top_k",
    "top_p": "top_p",
    "min_p": "min_p",
    "typical_p": "typical_p",
    "repeat_last_n": "repeat_last_n",
    "repeat_penalty": "repeat_penalty",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "mirostat": "mirostat",
    "mirostat_tau": "mirostat_tau",
    "mirostat_eta": "mirostat_eta",
    "seed": "seed",
    "num_ctx": "n_ctx",
    "num_keep": "n_keep",
    "tfs_z": "tfs_z",
    "grammar": "grammar",
    "json_schema": "json_schema",
    "grammar_lazy": "grammar_lazy",
    "cache_prompt": "cache_prompt",
    "image_data": "image_data",
    "enable_thinking": "thinking",  # qwen3-style: enable_thinking=False disables reasoning phase
}

DIRECT_LLAMA_FIELDS = {
    "grammar",
    "json_schema",
    "grammar_lazy",
    "response_format",
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "typical_p",
    "repeat_last_n",
    "repeat_penalty",
    "presence_penalty",
    "frequency_penalty",
    "mirostat",
    "mirostat_tau",
    "mirostat_eta",
    "seed",
    "n_predict",
    "n_ctx",
    "n_keep",
    "tfs_z",
    "cache_prompt",
    "image_data",
    "stop",
    "logprobs",
    "top_logprobs",
    "thinking",
    "reasoning_format",
    "enable_thinking",  # qwen3-style thinking control; mapped → thinking by OLLAMA_OPTION_MAP
}


def utc_now():
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def now_ns():
    return time.perf_counter_ns()


def should_stream(body):
    return body.get("stream", True) is not False


def parse_keep_alive(keep_alive):
    if keep_alive in (None, ""):
        return None
    if isinstance(keep_alive, (int, float)):
        return float(keep_alive)
    text = str(keep_alive).strip().lower()
    if text == "0":
        return 0
    if re.fullmatch(r"-?\d+(\.\d+)?", text):
        return float(text)
    match = re.fullmatch(r"(-?\d+)(ms|s|m|h|d)", text)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    return (
        value
        * {
            "ms": 0.001,
            "s": 1,
            "m": 60,
            "h": 3600,
            "d": 86400,
        }[unit]
    )


def expires_at_from_keep_alive(keep_alive):
    seconds = parse_keep_alive(keep_alive)
    if seconds is None:
        return "0001-01-01T00:00:00Z"
    if seconds <= 0:
        return utc_now()
    return datetime.fromtimestamp(time.time() + seconds).astimezone().isoformat()


def apply_format_parameter(payload, body):
    fmt = body.get("format")
    if fmt is None:
        return payload
    if isinstance(fmt, dict):
        payload.setdefault("json_schema", fmt)
        payload.setdefault("response_format", {"type": "json_object", "schema": fmt})
    elif fmt == "json":
        payload.setdefault("response_format", {"type": "json_object"})
    else:
        payload["format"] = fmt
    return payload


def apply_ollama_options(payload, options):
    if not isinstance(options, dict):
        return payload
    for key, value in options.items():
        mapped = OLLAMA_OPTION_MAP.get(key)
        if mapped:
            payload[mapped] = value
        else:
            payload[key] = value
    return payload


def apply_direct_llama_fields(payload, body):
    for key in DIRECT_LLAMA_FIELDS:
        if key in body:
            payload[key] = body[key]
    return payload


def build_chat_message(prompt, images=None):
    message = {"role": "user", "content": prompt}
    if images is not None:
        message["images"] = images
    return message


def should_generate_via_chat(body):
    return any(body.get(key) is not None for key in ("system", "think"))


def render_template_prompt(body):
    template = body.get("template")
    if not isinstance(template, str) or not template:
        return body.get("prompt", "")
    rendered = template
    rendered = rendered.replace("{{ .System }}", body.get("system", "") or "")
    rendered = rendered.replace("{{ .Prompt }}", body.get("prompt", "") or "")
    rendered = rendered.replace("{{ .Response }}", "")
    return rendered


def apply_thinking_override(payload, body):
    # We always keep thinking enabled downstream to prevent reasoning models
    # (like 35B MTP) from failing or crashing. Proxy will strip/filter
    # the thinking phase from the output returned to the client if think is False.
    payload["thinking"] = True



def build_generate_chat_payload(body, backend_model):
    prompt = body.get("prompt", "")
    payload = {
        "model": backend_model,
        "messages": [build_chat_message(prompt, body.get("images"))],
        "stream": should_stream(body),
    }
    if body.get("system"):
        payload["messages"].insert(0, {"role": "system", "content": body["system"]})
    apply_ollama_options(payload, body.get("options"))
    apply_direct_llama_fields(payload, body)
    apply_format_parameter(payload, body)
    if body.get("think") is not None:
        payload["thinking"] = body["think"]
    if body.get("enable_thinking") is not None:
        payload["thinking"] = body["enable_thinking"]
    apply_thinking_override(payload, body)
    return payload


def build_chat_payload(body, backend_model):
    payload = {
        "model": backend_model,
        "messages": body.get("messages", []),
        "stream": should_stream(body),
    }
    if body.get("tools") is not None:
        payload["tools"] = body["tools"]
    if body.get("tool_choice") is not None:
        payload["tool_choice"] = body["tool_choice"]
    apply_ollama_options(payload, body.get("options"))
    apply_direct_llama_fields(payload, body)
    apply_format_parameter(payload, body)
    if body.get("think") is not None:
        payload["thinking"] = body["think"]
    if body.get("enable_thinking") is not None:
        payload["thinking"] = body["enable_thinking"]
    apply_thinking_override(payload, body)
    return payload


def build_generate_payload(body, backend_model):
    payload = {
        "model": backend_model,
        "prompt": render_template_prompt(body),
        "stream": should_stream(body),
    }
    if body.get("images") is not None:
        payload["image_data"] = body["images"]
    if body.get("stop") is not None:
        payload["stop"] = body["stop"]
    if body.get("raw"):
        payload["raw"] = True
    apply_ollama_options(payload, body.get("options"))
    apply_direct_llama_fields(payload, body)
    apply_format_parameter(payload, body)
    if "num_predict" in body:
        payload["n_predict"] = body["num_predict"]
    if body.get("suffix") is not None:
        payload["input_suffix"] = body["suffix"]
    if isinstance(body.get("context"), list) and body["context"]:
        payload["prompt"] = list(body["context"]) + [payload["prompt"]]
    apply_thinking_override(payload, body)
    return payload


async def fetch_router_models(reload=False):
    # Handle case where client_httpx is not yet initialized (e.g., during tests)
    if client_httpx is None:
        logger.debug("client_httpx not yet initialized; returning empty router models list")
        return []

    params = {"reload": "1"} if reload else None
    try:
        resp = await client_httpx.get(ROUTER_MODELS_URL, params=params)
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        logger.warning(
            f"Connection to llama-server failed: {exc}. Waiting for server to become responsive..."
        )
        if await wait_for_llama_server():
            resp = await client_httpx.get(ROUTER_MODELS_URL, params=params)
        else:
            raise
    resp.raise_for_status()
    data = resp.json()
    return data.get("data") or []


async def resolve_router_model(model_name, reload=True):
    resolved_name = with_default_tag(model_name)
    manifest_path, manifest = load_local_manifest(resolved_name, require_complete=True)
    if not manifest_path:
        raise HTTPException(status_code=404, detail=f"Model {resolved_name} not found")
    if manifest is None:
        raise HTTPException(
            status_code=409, detail=f"Model {resolved_name} is still downloading or incomplete."
        )

    candidates = router_model_candidates(resolved_name, manifest)
    router_models = await fetch_router_models(reload=reload)
    for entry in router_models:
        if router_entry_matches(entry, candidates):
            return {
                "model_name": resolved_name,
                "backend_model": entry.get("id") or candidates[0],
                "entry": entry,
                "manifest_path": manifest_path,
                "manifest": manifest,
                "router_models": router_models,
            }

    fallback_id = router_model_id_for_name(resolved_name)
    fallback_path = router_path_for_model_name(resolved_name)
    # If the router service is unavailable (client_httpx not initialized or connection failed),
    # or the router returned no models, fall back to a deterministic entry.
    # This ensures resolve_router_model always returns a usable dict during tests or when the router is offline.
    if not router_models:
        return {
            "model_name": resolved_name,
            "backend_model": fallback_id,
            "entry": {"id": fallback_id, "path": fallback_path, "status": {"value": "unloaded"}},
            "manifest_path": manifest_path,
            "manifest": manifest,
            "router_models": router_models,
        }
    # Keep original fallback for when the file actually exists on disk.
    if os.path.exists(fallback_path):
        return {
            "model_name": resolved_name,
            "backend_model": fallback_id,
            "entry": {"id": fallback_id, "path": fallback_path, "status": {"value": "unloaded"}},
            "manifest_path": manifest_path,
            "manifest": manifest,
            "router_models": router_models,
        }

    raise HTTPException(
        status_code=404, detail=f"Router could not discover backend model for {resolved_name}"
    )


def effective_keep_alive(keep_alive):
    return DEFAULT_KEEP_ALIVE if keep_alive in (None, "") else keep_alive


def is_resident_status(status):
    return status in {"loaded", "loading"}


def upstream_timeout():
    read_timeout = None
    if LLAMA_SERVER_READ_TIMEOUT_SECONDS not in ("", "none", "None", "0", "0.0"):
        read_timeout = float(LLAMA_SERVER_READ_TIMEOUT_SECONDS)
    # Increase base timeout to 600s to match Raven expectations
    return httpx.Timeout(600.0, connect=LLAMA_SERVER_CONNECT_TIMEOUT_SECONDS, read=read_timeout)


async def post_router_model_action(action, payload):
    global router_management_supported
    url = f"{ROUTER_MODELS_URL}/{action}"

    # If payload is just a string, wrap it. If it's a dict, use it as is.
    if isinstance(payload, str):
        json_body = {"model": payload}
    else:
        json_body = payload

    try:
        resp = await client_httpx.post(url, json=json_body)
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        logger.warning(
            f"Connection to llama-server failed: {exc}. Waiting for server to become responsive..."
        )
        if await wait_for_llama_server():
            resp = await client_httpx.post(url, json=json_body)
        else:
            raise

    if resp.status_code == 404:
        router_management_supported = False
        raise RouterManagementUnsupported(url)
    router_management_supported = True

    if resp.status_code == 400 and action == "load":
        try:
            body = resp.json()
            msg = str(body.get("error", {}).get("message", ""))
            if "already running" in msg.lower() or "model is already running" in msg.lower():
                logger.info("Model already running on load action, treating as success.")
                return {"success": True}
        except Exception:
            pass

    resp.raise_for_status()
    return resp.json()


def is_ignorable_router_unload_error(exc):
    response = exc.response
    if response is None or response.status_code != 400:
        return False
    try:
        payload = response.json()
    except ValueError:
        payload = None
    message = ""
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = str(error.get("message", ""))
        elif error is not None:
            message = str(error)
    if not message:
        message = response.text
    message = message.lower()
    ignorable_patterns = [
        "model is not found",
        "model not found",
        "not found",
        "model is not loaded",
        "model is not resident",
        "model already unloaded",
        "model is unloaded",
        "model has been evicted",
        "model not loaded",
    ]
    return any(pattern in message for pattern in ignorable_patterns)


async def unload_model(model_name):
    try:
        resolved = await resolve_router_model(model_name, reload=False)
    except HTTPException:
        return
    backend_model = resolved["backend_model"]
    async with active_requests_lock:
        if active_requests.get(backend_model, 0) > 0:
            logger.warning(
                f"Aborting unload of {backend_model} because it currently has "
                f"{active_requests[backend_model]} active request(s)."
            )
            return
    async with router_model_lock:
        try:
            await post_router_model_action("unload", backend_model)
        except RouterManagementUnsupported:
            logger.warning("Router model unload endpoint unavailable; skipping explicit unload.")
            return
        except httpx.HTTPStatusError as exc:
            if is_ignorable_router_unload_error(exc) or not await router_model_is_still_resident(
                backend_model
            ):
                logger.info(f"Router ignored unload for {backend_model}: {exc}")
            else:
                raise
    model_expires_at.pop(public_model_name(model_name), None)
    await record_model_unloaded(model_name)


def cancel_model_unload(model_name):
    task = model_unload_tasks.pop(public_model_name(model_name), None)
    if task:
        task.cancel()


def begin_model_request(model_name):
    cancel_model_unload(model_name)
    model_expires_at[public_model_name(model_name)] = "0001-01-01T00:00:00Z"


def mark_model_loading(model_name):
    public = public_model_name(model_name)
    model_loading[public] = {"start_time": time.time(), "backend_model": None}
    logger.info(f"Marked model {public} as loading.")


def mark_model_loaded(model_name):
    public = public_model_name(model_name)
    model_loading.pop(public, None)
    logger.info(f"Marked model {public} as no longer loading.")


def is_model_loading(model_name):
    public = public_model_name(model_name)
    entry = model_loading.get(public)
    if entry is None:
        return False
    if time.time() - entry["start_time"] > MODEL_LOADING_TIMEOUT:
        logger.warning(f"Model {public} loading timed out ({MODEL_LOADING_TIMEOUT}s). Clearing.")
        model_loading.pop(public, None)
        return False
    return True


async def router_model_is_still_resident(backend_model):
    try:
        for entry in await fetch_router_models(reload=False):
            if entry.get("id") == backend_model:
                return is_resident_status(router_entry_status(entry))
    except Exception:
        return True
    return False


async def unload_model_later(model_name, delay_seconds):
    try:
        await asyncio.sleep(delay_seconds)
        await unload_model(model_name)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning(f"Deferred unload failed for {public_model_name(model_name)}: {exc}")
    finally:
        model_unload_tasks.pop(public_model_name(model_name), None)


async def apply_keep_alive_policy(model_name, keep_alive):
    model_name = with_default_tag(model_name)
    public_name = public_model_name(model_name)
    keep_alive = effective_keep_alive(keep_alive)
    seconds = parse_keep_alive(keep_alive)

    cancel_model_unload(model_name)

    if seconds is None:
        model_expires_at[public_name] = "0001-01-01T00:00:00Z"
        return
    if seconds < 0:
        model_expires_at[public_name] = FOREVER_EXPIRES_AT
        return
    if seconds == 0:
        model_expires_at[public_name] = utc_now()
        await unload_model(model_name)
        return

    model_expires_at[public_name] = (
        datetime.fromtimestamp(time.time() + seconds).astimezone().isoformat()
    )
    model_unload_tasks[public_name] = asyncio.create_task(unload_model_later(model_name, seconds))


def usage_stats(data):
    usage = data.get("usage") or {}
    prompt_eval_count = usage.get("prompt_tokens", data.get("tokens_evaluated"))
    eval_count = usage.get("completion_tokens", data.get("tokens_predicted"))
    if prompt_eval_count is None:
        prompt_eval_count = data.get("prompt_eval_count")
    if eval_count is None:
        eval_count = data.get("eval_count")
    return prompt_eval_count, eval_count


def timing_stats(data, fallback_total_duration=None, fallback_load_duration=0):
    total_duration = data.get("total_duration", fallback_total_duration)
    load_duration = data.get("load_duration", fallback_load_duration)
    prompt_eval_duration = data.get("prompt_eval_duration")
    eval_duration = data.get("eval_duration")

    timings = data.get("timings") or {}
    total_duration = total_duration if total_duration is not None else fallback_total_duration
    prompt_eval_duration = (
        prompt_eval_duration if prompt_eval_duration is not None else timings.get("prompt_ms")
    )
    eval_duration = eval_duration if eval_duration is not None else timings.get("predicted_ms")

    if isinstance(prompt_eval_duration, (int, float)) and prompt_eval_duration < 10_000_000:
        prompt_eval_duration = int(prompt_eval_duration * 1_000_000)
    if isinstance(eval_duration, (int, float)) and eval_duration < 10_000_000:
        eval_duration = int(eval_duration * 1_000_000)

    return total_duration, load_duration, prompt_eval_duration, eval_duration


def logprobs_from_choice(choice):
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return None

    tokens = logprobs.get("tokens") or []
    token_logprobs = logprobs.get("token_logprobs") or []
    top_logprobs = logprobs.get("top_logprobs") or []
    result = []
    for index, token in enumerate(tokens):
        entry = {
            "token": token,
            "logprob": token_logprobs[index] if index < len(token_logprobs) else None,
        }
        top_entries = []
        raw_top = top_logprobs[index] if index < len(top_logprobs) else {}
        if isinstance(raw_top, dict):
            for top_token, top_logprob in raw_top.items():
                top_entries.append({"token": top_token, "logprob": top_logprob})
        if top_entries:
            entry["top_logprobs"] = top_entries
        result.append(entry)
    return result or None


def chat_message_from_choice(choice):
    delta = choice.get("delta") or {}
    message = choice.get("message") or {}
    payload = {
        "role": message.get("role") or "assistant",
        "content": delta.get("content") or message.get("content") or "",
    }
    thinking = (
        delta.get("reasoning_content")
        or delta.get("thinking")
        or message.get("reasoning_content")
        or message.get("thinking")
    )
    if thinking:
        payload["thinking"] = thinking
    tool_calls = delta.get("tool_calls") or message.get("tool_calls")
    if tool_calls:
        payload["tool_calls"] = tool_calls
    images = message.get("images")
    if images:
        payload["images"] = images
    return payload


def apply_metrics(chunk, data, total_duration=None, load_duration=0):
    prompt_eval_count, eval_count = usage_stats(data)
    td, ld, ped, ed = timing_stats(data, total_duration, load_duration)
    if td is not None:
        chunk["total_duration"] = int(td)
    if ld is not None:
        chunk["load_duration"] = int(ld)
    if prompt_eval_count is not None:
        chunk["prompt_eval_count"] = int(prompt_eval_count)
    if ped is not None:
        chunk["prompt_eval_duration"] = int(ped)
    if eval_count is not None:
        chunk["eval_count"] = int(eval_count)
    if ed is not None:
        chunk["eval_duration"] = int(ed)
    return chunk


def ollama_chat_chunk(model_name, message=None, done=False, done_reason=None):
    chunk = {
        "model": public_model_name(model_name),
        "created_at": utc_now(),
        "message": message or {"role": "assistant", "content": ""},
        "done": done,
    }
    if done_reason is not None:
        chunk["done_reason"] = done_reason
    return chunk


def ollama_generate_chunk(model_name, content="", done=False, done_reason=None):
    chunk = {
        "model": public_model_name(model_name),
        "created_at": utc_now(),
        "response": content,
        "done": done,
    }
    if done_reason is not None:
        chunk["done_reason"] = done_reason
    return chunk


def get_model_info(model_name):
    manifest_path, manifest = load_local_manifest(model_name, require_complete=True)
    if not manifest_path or manifest is None:
        return None
    return manifest_stats(manifest_path, manifest)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client_httpx
    client_httpx = httpx.AsyncClient(timeout=upstream_timeout())
    await restore_models_on_recovery()
    yield
    for task in list(model_unload_tasks.values()):
        task.cancel()
    await client_httpx.aclose()


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    # Extract client IP and User-Agent
    client_ip = request.client.host if request.client else "unknown-ip"
    user_agent = request.headers.get("user-agent", "unknown-ua")

    # Extract explicit origin tracking header (case-insensitive)
    request_source = request.headers.get("x-request-source")
    if not request_source:
        # Fallback detection based on User-Agent patterns
        if "playwright" in user_agent.lower() or "python-httpx" in user_agent.lower():
            request_source = "agent/script"
        elif "mozilla" in user_agent.lower() or "chrome" in user_agent.lower():
            request_source = "browser/ui"
        else:
            request_source = "unknown-origin"

    request.state.request_source = request_source

    logger.info(
        f"Hit: {request.method} {request.url.path} | "
        f"Origin: {request_source} | IP: {client_ip} | UA: {user_agent}",
        extra={"request_id": request_id, "request_source": request_source},
    )

    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    logger.info(
        f"Finished: {request.method} {request.url.path} in {duration:.3f}s | "
        f"Origin: {request_source}",
        extra={"request_id": request_id, "request_source": request_source},
    )
    return response


# ─── Performance Metrics Tracking ─────────────────────────────────────────────
metrics = {
    "requests_total": 0,
    "requests_by_endpoint": {},
    "tokens_generated": 0,
    "tokens_prompted": 0,
    "errors_total": 0,
    "avg_latency_ms": 0.0,
    "latency_samples": [],
    "start_time": time.time(),
}
metrics_lock = asyncio.Lock()


async def record_metrics(endpoint, latency_ms, prompt_tokens=0, gen_tokens=0, error=False):
    async with metrics_lock:
        metrics["requests_total"] += 1
        metrics["requests_by_endpoint"][endpoint] = (
            metrics["requests_by_endpoint"].get(endpoint, 0) + 1
        )
        metrics["tokens_prompted"] += prompt_tokens
        metrics["tokens_generated"] += gen_tokens
        if error:
            metrics["errors_total"] += 1
        metrics["latency_samples"].append(latency_ms)
        if len(metrics["latency_samples"]) > 1000:
            metrics["latency_samples"] = metrics["latency_samples"][-500:]
        metrics["avg_latency_ms"] = sum(metrics["latency_samples"]) / len(
            metrics["latency_samples"]
        )


# ─── Grammar/Schema Registry ──────────────────────────────────────────────────
GRAMMAR_REGISTRY_DIR = os.getenv("GRAMMAR_REGISTRY_DIR", "/alpaca-data/grammars")
SCHEMA_REGISTRY_DIR = os.getenv("SCHEMA_REGISTRY_DIR", "/alpaca-data/schemas")


def _ensure_registry_dirs():
    os.makedirs(GRAMMAR_REGISTRY_DIR, exist_ok=True)
    os.makedirs(SCHEMA_REGISTRY_DIR, exist_ok=True)


_ensure_registry_dirs()


def _registry_path(registry_dir, name):
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")
    return os.path.join(registry_dir, f"{sanitized}.json")


@app.get("/admin/grammars")
async def list_grammars():
    _ensure_registry_dirs()
    grammars = []
    if os.path.exists(GRAMMAR_REGISTRY_DIR):
        for f in os.listdir(GRAMMAR_REGISTRY_DIR):
            if f.endswith(".json"):
                path = os.path.join(GRAMMAR_REGISTRY_DIR, f)
                try:
                    with open(path) as fh:
                        data = json.load(fh)
                    grammars.append(
                        {
                            "name": f[:-5],
                            "type": data.get("type", "grammar"),
                            "description": data.get("description", ""),
                            "created_at": data.get("created_at", ""),
                            "size": os.path.getsize(path),
                        }
                    )
                except Exception:
                    pass
    return {"grammars": grammars}


@app.post("/admin/grammars")
async def save_grammar(request: Request):
    _ensure_registry_dirs()
    body = await request.json()
    name = body.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    path = _registry_path(GRAMMAR_REGISTRY_DIR, name)
    data = {
        "name": name,
        "type": body.get("type", "grammar"),
        "description": body.get("description", ""),
        "grammar": body.get("grammar", ""),
        "json_schema": body.get("json_schema"),
        "created_at": utc_now(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return {"status": "saved", "name": name, "path": path}


@app.get("/admin/grammars/{name}")
async def get_grammar(name: str):
    path = _registry_path(GRAMMAR_REGISTRY_DIR, name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Grammar '{name}' not found")
    with open(path) as f:
        return json.load(f)


@app.delete("/admin/grammars/{name}")
async def delete_grammar(name: str):
    path = _registry_path(GRAMMAR_REGISTRY_DIR, name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Grammar '{name}' not found")
    os.remove(path)
    return {"status": "deleted", "name": name}


@app.get("/admin/schemas")
async def list_schemas():
    _ensure_registry_dirs()
    schemas = []
    if os.path.exists(SCHEMA_REGISTRY_DIR):
        for f in os.listdir(SCHEMA_REGISTRY_DIR):
            if f.endswith(".json"):
                path = os.path.join(SCHEMA_REGISTRY_DIR, f)
                try:
                    with open(path) as fh:
                        data = json.load(fh)
                    schemas.append(
                        {
                            "name": f[:-5],
                            "description": data.get("description", ""),
                            "created_at": data.get("created_at", ""),
                            "size": os.path.getsize(path),
                        }
                    )
                except Exception:
                    pass
    return {"schemas": schemas}


@app.post("/admin/schemas")
async def save_schema(request: Request):
    _ensure_registry_dirs()
    body = await request.json()
    name = body.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    schema = body.get("schema")
    if not schema:
        raise HTTPException(status_code=400, detail="schema is required")
    path = _registry_path(SCHEMA_REGISTRY_DIR, name)
    data = {
        "name": name,
        "description": body.get("description", ""),
        "schema": schema,
        "created_at": utc_now(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return {"status": "saved", "name": name, "path": path}


@app.get("/admin/schemas/{name}")
async def get_schema(name: str):
    path = _registry_path(SCHEMA_REGISTRY_DIR, name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Schema '{name}' not found")
    with open(path) as f:
        return json.load(f)


@app.delete("/admin/schemas/{name}")
async def delete_schema(name: str):
    path = _registry_path(SCHEMA_REGISTRY_DIR, name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Schema '{name}' not found")
    os.remove(path)
    return {"status": "deleted", "name": name}


# ─── Embedding Endpoints (Ollama-compatible) ──────────────────────────────────
@app.post("/api/embed")
async def embed(request: Request):
    """Ollama-compatible embedding endpoint. Proxies to llama-server /v1/embeddings."""
    body = await request.json()
    model_name = body.get("model")
    input_data = body.get("input")
    if not model_name:
        raise HTTPException(status_code=400, detail="model is required")
    if input_data is None:
        raise HTTPException(status_code=400, detail="input is required")

    started_ns = now_ns()
    resolved = await ensure_model(model_name)
    backend_model = resolved["backend_model"]

    # Build OpenAI-compatible embedding request
    llama_payload = {
        "model": backend_model,
        "input": input_data,
    }
    normalize = body.get("normalize", True)
    if not normalize:
        llama_payload["normalize"] = False

    try:
        resp = await client_httpx.post(f"{LLAMA_SERVER_URL}/v1/embeddings", json=llama_payload)
        resp.raise_for_status()
        data = resp.json()

        # Convert OpenAI format to Ollama format
        ollama_embeddings = []
        for item in data.get("data", []):
            ollama_embeddings.append(item.get("embedding", []))

        load_duration = data.get("load_duration", 0)
        total_duration = now_ns() - started_ns

        result = {
            "model": public_model_name(model_name),
            "embeddings": ollama_embeddings,
            "total_duration": total_duration,
            "load_duration": load_duration,
            "prompt_eval_count": data.get("usage", {}).get("prompt_tokens", 0),
        }
        if not body.get("truncate", True):
            result["truncated"] = False

        await record_metrics(
            "/api/embed", total_duration / 1e6, prompt_tokens=result["prompt_eval_count"]
        )
        return JSONResponse(result)
    except httpx.HTTPStatusError as e:
        await record_metrics("/api/embed", 0, error=True)
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.TimeoutException:
        await record_metrics("/api/embed", 0, error=True)
        raise HTTPException(status_code=504, detail="Upstream llama-server timed out")
    except httpx.RequestError as e:
        await record_metrics("/api/embed", 0, error=True)
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {e}")


@app.post("/api/embeddings")
async def embeddings_legacy(request: Request):
    """Legacy Ollama embedding endpoint (alias for /api/embed)."""
    return await embed(request)


# ─── OpenAI-Compatible Endpoints ──────────────────────────────────────────────
@app.get("/v1/models")
async def openai_models():
    """OpenAI-compatible model listing. Proxies to llama-server /v1/models or builds from local manifests."""
    try:
        resp = await client_httpx.get(f"{LLAMA_SERVER_URL}/v1/models")
        if resp.status_code == 200:
            data = resp.json()
            # Enrich with local manifest info
            local_info = {}
            for manifest_base, path, manifest in iter_local_manifests():
                mn = manifest_model_name(manifest_base, path)
                if mn:
                    info = manifest_stats(path, manifest)
                    local_info[mn] = info

            for obj in data.get("data", []):
                model_id = obj.get("id", "")
                for local_name, info in local_info.items():
                    if local_name.replace(":", "--") in model_id or model_id in local_name:
                        obj["alpaca"] = {
                            "size": info["size"],
                            "details": info["details"],
                            "capabilities": info["capabilities"],
                            "context_length": info["context_length"],
                        }
                        break
            return data
    except Exception:
        pass

    # Fallback: build from local manifests
    models = []
    for manifest_base, path, manifest in iter_local_manifests():
        mn = manifest_model_name(manifest_base, path)
        if mn:
            info = manifest_stats(path, manifest)
            models.append(
                {
                    "id": mn.replace(":", "--"),
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "alpaca",
                    "alpaca": {
                        "name": mn,
                        "size": info["size"],
                        "details": info["details"],
                        "capabilities": info["capabilities"],
                        "context_length": info["context_length"],
                    },
                }
            )
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    """OpenAI-compatible chat completions. Proxies directly to llama-server."""
    global client_httpx
    body = await request.json()
    started_ns = now_ns()
    model_name = body.get("model", "")
    backend_model = model_name
    stream = body.get("stream", False)
    request_id = getattr(request.state, "request_id", None)
    if not request_id or request_id == "N/A":
        import uuid

        request_id = str(uuid.uuid4())[:8]
    register_active_request(
        request_id,
        model_name,
        "openai_chat",
        body,
        request_source=getattr(request.state, "request_source", "unknown"),
        client_ip=(request.client.host if request.client else "unknown")
    )
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Resolve model if provided (inside the retry loop!)
            if model_name:
                try:
                    resolved = await ensure_model(model_name)
                    backend_model = resolved["backend_model"]
                    body["model"] = backend_model
                except HTTPException as e:
                    # Client errors (like 404 Not Found) should fail immediately
                    return JSONResponse(
                        status_code=e.status_code,
                        content={
                            "error": {
                                "message": f"Model resolution failed: {e.detail}",
                                "type": "invalid_request_error",
                                "param": "model",
                                "code": "model_not_found",
                            }
                        },
                    )
            if stream:

                async def stream_proxy():
                    stream_started = False
                    async with active_requests_lock:
                        active_requests[backend_model] = active_requests.get(backend_model, 0) + 1
                    try:
                        for s_attempt in range(max_retries):
                            try:
                                async with client_httpx.stream(
                                    "POST", f"{LLAMA_SERVER_URL}/v1/chat/completions", json=body
                                ) as resp:
                                    if resp.status_code != 200:
                                        err_body = await resp.aread()
                                        err_msg = err_body.decode(errors="ignore")
                                        if s_attempt < max_retries - 1 and (
                                            "loading" in err_msg
                                            or "error" in err_msg
                                            or resp.status_code in (502, 503, 504)
                                        ):
                                            logger.warning(
                                                f"Upstream stream returned status {resp.status_code}. Retrying recovery/load..."
                                            )
                                            await ensure_model(model_name)
                                            continue
                                        yield f"data: {json.dumps({'error': {'message': err_msg, 'type': 'invalid_request_error', 'code': resp.status_code}})}\n\n"
                                        return

                                    stream_started = True
                                    async for line in resp.aiter_lines():
                                        if line and line.startswith("data: "):
                                            try:
                                                payload_str = line[6:].strip()
                                                if payload_str and payload_str != "[DONE]":
                                                    payload_json = json.loads(payload_str)
                                                    choices = payload_json.get("choices")
                                                    if (
                                                        choices
                                                        and isinstance(choices, list)
                                                        and len(choices) > 0
                                                    ):
                                                        delta = choices[0].get("delta")
                                                        if delta:
                                                            update_active_request_progress(
                                                                request_id,
                                                                response_chunk=delta.get("content"),
                                                                thinking_chunk=delta.get(
                                                                    "reasoning_content"
                                                                )
                                                                or delta.get("thinking"),
                                                            )
                                            except Exception:
                                                pass
                                            yield f"{line}\n\n"
                                    return
                            except httpx.RequestError as exc:
                                if stream_started or s_attempt == max_retries - 1:
                                    yield f"data: {json.dumps({'error': {'message': f'Upstream connection lost: {exc}', 'type': 'api_error', 'code': None}})}\n\n"
                                    if stream_started:
                                        # Mid-stream crash: stream is lost but kick off background
                                        # recovery now so the next request doesn't pay cold-start cost
                                        logger.warning(
                                            f"Mid-stream crash detected ({exc}). Triggering background server recovery..."
                                        )
                                        task = asyncio.create_task(
                                            _ensure_model_skip_swap(model_name)
                                        )
                                        task.set_name(f"midstream-recovery-chat-{model_name}")
                                        task.add_done_callback(handle_background_task_result)
                                    return
                                logger.warning(
                                    f"Connection lost during stream init: {exc}. Retrying recovery/load..."
                                )
                                await ensure_model(model_name)
                                await asyncio.sleep(2.0)
                    finally:
                        async with active_requests_lock:
                            active_requests[backend_model] = max(0, active_requests.get(backend_model, 0) - 1)
                            active_requests_lock.notify_all()
                        req_data = complete_active_request(request_id)
                        p_toks = req_data.get("prompt_tokens", 0) if req_data else 0
                        c_toks = req_data.get("completion_tokens", 0) if req_data else 0
                        await record_metrics(
                            "/v1/chat/completions",
                            (now_ns() - started_ns) / 1e6,
                            prompt_tokens=p_toks,
                            gen_tokens=c_toks
                        )

                return StreamingResponse(stream_proxy(), media_type="text/event-stream")
            else:
                async with active_requests_lock:
                    active_requests[backend_model] = active_requests.get(backend_model, 0) + 1
                try:
                    resp = await client_httpx.post(
                        f"{LLAMA_SERVER_URL}/v1/chat/completions", json=body
                    )
                    if resp.status_code != 200:
                        err_msg = resp.text
                        if attempt < max_retries - 1 and resp.status_code in (502, 503, 504):
                            logger.warning(
                                f"Upstream returned error status {resp.status_code}. Retrying recovery/load..."
                            )
                            await ensure_model(model_name)
                            continue
                        try:
                            upstream_err = resp.json()
                            if "error" in upstream_err:
                                err_msg = upstream_err["error"].get("message", err_msg)
                        except Exception:
                            pass
                        await record_metrics("/v1/chat/completions", 0, error=True)
                        return JSONResponse(
                            status_code=resp.status_code,
                            content={
                                "error": {
                                    "message": err_msg,
                                    "type": "invalid_request_error",
                                    "code": resp.status_code,
                                }
                            },
                        )
                    data = resp.json()
                    prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
                    gen_tokens = data.get("usage", {}).get("completion_tokens", 0)
                    try:
                        choices = data.get("choices", [])
                        if choices and len(choices) > 0:
                            message_obj = choices[0].get("message", {})
                            content_val = message_obj.get("content")
                            thinking_val = message_obj.get("reasoning_content") or message_obj.get(
                                "thinking"
                            )
                            complete_active_request(
                                request_id, 
                                final_response=content_val, 
                                final_thinking=thinking_val,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=gen_tokens
                            )
                    except Exception:
                        pass
                    latency = (now_ns() - started_ns) / 1e6
                    await record_metrics("/v1/chat/completions", latency, prompt_tokens, gen_tokens)
                    return JSONResponse(data)
                finally:
                    async with active_requests_lock:
                        active_requests[backend_model] = max(0, active_requests.get(backend_model, 0) - 1)
                        active_requests_lock.notify_all()
                    complete_active_request(request_id)
        except httpx.RequestError as exc:
            if attempt == max_retries - 1:
                logger.error(
                    f"Chat completions proxy failed after {max_retries} attempts due to connection error: {exc}"
                )
                await record_metrics("/v1/chat/completions", 0, error=True)
                return JSONResponse(
                    status_code=502,
                    content={
                        "error": {
                            "message": f"Upstream request failed: {exc}",
                            "type": "api_error",
                            "code": "bad_gateway",
                        }
                    },
                )
            logger.warning(
                f"Connection lost during completions request: {exc}. Flushing client pool and waiting for llama-server recovery..."
            )
            old_client = client_httpx
            client_httpx = httpx.AsyncClient(timeout=upstream_timeout())
            task = asyncio.create_task(old_client.aclose())
            task.set_name("close-old-client-chat")
            task.add_done_callback(handle_background_task_result)

            await wait_for_llama_server_or_restart(timeout=60.0)
            await ensure_model(model_name)


@app.post("/v1/completions")
async def openai_completions(request: Request):
    """OpenAI-compatible text completions. Proxies directly to llama-server."""
    global client_httpx
    body = await request.json()
    started_ns = now_ns()
    model_name = body.get("model", "")
    backend_model = model_name
    stream = body.get("stream", False)
    request_id = getattr(request.state, "request_id", None)
    if not request_id or request_id == "N/A":
        import uuid

        request_id = str(uuid.uuid4())[:8]
    register_active_request(
        request_id,
        model_name,
        "openai_generate",
        body,
        request_source=getattr(request.state, "request_source", "unknown"),
        client_ip=(request.client.host if request.client else "unknown")
    )
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Resolve model if provided (inside the retry loop!)
            if model_name:
                try:
                    resolved = await ensure_model(model_name)
                    backend_model = resolved["backend_model"]
                    body["model"] = backend_model
                except HTTPException as e:
                    # Client errors (like 404 Not Found) should fail immediately
                    return JSONResponse(
                        status_code=e.status_code,
                        content={
                            "error": {
                                "message": f"Model resolution failed: {e.detail}",
                                "type": "invalid_request_error",
                                "param": "model",
                                "code": "model_not_found",
                            }
                        },
                    )
            if stream:

                async def stream_proxy():
                    stream_started = False
                    async with active_requests_lock:
                        active_requests[backend_model] = active_requests.get(backend_model, 0) + 1
                    try:
                        for s_attempt in range(max_retries):
                            try:
                                async with client_httpx.stream(
                                    "POST", f"{LLAMA_SERVER_URL}/v1/completions", json=body
                                ) as resp:
                                    if resp.status_code != 200:
                                        err_body = await resp.aread()
                                        err_msg = err_body.decode(errors="ignore")
                                        if s_attempt < max_retries - 1 and (
                                            "loading" in err_msg
                                            or "error" in err_msg
                                            or resp.status_code in (502, 503, 504)
                                        ):
                                            logger.warning(
                                                f"Upstream stream returned status {resp.status_code}. Retrying recovery/load..."
                                            )
                                            await ensure_model(model_name)
                                            continue
                                        yield f"data: {json.dumps({'error': {'message': err_msg, 'type': 'invalid_request_error', 'code': resp.status_code}})}\n\n"
                                        return

                                    stream_started = True
                                    async for line in resp.aiter_lines():
                                        if line and line.startswith("data: "):
                                            try:
                                                payload_str = line[6:].strip()
                                                if payload_str and payload_str != "[DONE]":
                                                    payload_json = json.loads(payload_str)
                                                    choices = payload_json.get("choices")
                                                    if (
                                                        choices
                                                        and isinstance(choices, list)
                                                        and len(choices) > 0
                                                    ):
                                                        text_val = choices[0].get("text")
                                                        thinking_val = choices[0].get(
                                                            "thinking"
                                                        ) or choices[0].get("reasoning_content")
                                                        update_active_request_progress(
                                                            request_id,
                                                            response_chunk=text_val,
                                                            thinking_chunk=thinking_val,
                                                        )
                                            except Exception:
                                                pass
                                            yield f"{line}\n\n"
                                    return
                            except httpx.RequestError as exc:
                                if stream_started or s_attempt == max_retries - 1:
                                    yield f"data: {json.dumps({'error': {'message': f'Upstream connection lost: {exc}', 'type': 'api_error', 'code': None}})}\n\n"
                                    if stream_started:
                                        # Mid-stream crash: stream is lost but kick off background
                                        # recovery now so the next request doesn't pay cold-start cost
                                        logger.warning(
                                            f"Mid-stream crash detected ({exc}). Triggering background server recovery..."
                                        )
                                        task = asyncio.create_task(
                                            _ensure_model_skip_swap(model_name)
                                        )
                                        task.set_name(
                                            f"midstream-recovery-completions-{model_name}"
                                        )
                                        task.add_done_callback(handle_background_task_result)
                                    return
                                logger.warning(
                                    f"Connection lost during stream init: {exc}. Retrying recovery/load..."
                                )
                                await ensure_model(model_name)
                                await asyncio.sleep(2.0)
                    finally:
                        async with active_requests_lock:
                            active_requests[backend_model] = max(0, active_requests.get(backend_model, 0) - 1)
                            active_requests_lock.notify_all()
                        req_data = complete_active_request(request_id)
                        p_toks = req_data.get("prompt_tokens", 0) if req_data else 0
                        c_toks = req_data.get("completion_tokens", 0) if req_data else 0
                        await record_metrics(
                            "/v1/completions",
                            (now_ns() - started_ns) / 1e6,
                            prompt_tokens=p_toks,
                            gen_tokens=c_toks
                        )

                return StreamingResponse(stream_proxy(), media_type="text/event-stream")
            else:
                async with active_requests_lock:
                    active_requests[backend_model] = active_requests.get(backend_model, 0) + 1
                try:
                    resp = await client_httpx.post(f"{LLAMA_SERVER_URL}/v1/completions", json=body)
                    if resp.status_code != 200:
                        err_msg = resp.text
                        if attempt < max_retries - 1 and resp.status_code in (502, 503, 504):
                            logger.warning(
                                f"Upstream returned error status {resp.status_code}. Retrying recovery/load..."
                            )
                            await ensure_model(model_name)
                            continue
                        try:
                            upstream_err = resp.json()
                            if "error" in upstream_err:
                                err_msg = upstream_err["error"].get("message", err_msg)
                        except Exception:
                            pass
                        await record_metrics("/v1/completions", 0, error=True)
                        return JSONResponse(
                            status_code=resp.status_code,
                            content={
                                "error": {
                                    "message": err_msg,
                                    "type": "invalid_request_error",
                                    "code": resp.status_code,
                                }
                            },
                        )
                    data = resp.json()
                    prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
                    gen_tokens = data.get("usage", {}).get("completion_tokens", 0)
                    try:
                        choices = data.get("choices", [])
                        if choices and len(choices) > 0:
                            content_val = choices[0].get("text")
                            thinking_val = choices[0].get("thinking") or choices[0].get(
                                "reasoning_content"
                            )
                            complete_active_request(
                                request_id, 
                                final_response=content_val, 
                                final_thinking=thinking_val,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=gen_tokens
                            )
                    except Exception:
                        pass
                    latency = (now_ns() - started_ns) / 1e6
                    await record_metrics("/v1/completions", latency, prompt_tokens, gen_tokens)
                    return JSONResponse(data)
                finally:
                    async with active_requests_lock:
                        active_requests[backend_model] = max(0, active_requests.get(backend_model, 0) - 1)
                        active_requests_lock.notify_all()
                    complete_active_request(request_id)
        except httpx.RequestError as exc:
            if attempt == max_retries - 1:
                logger.error(
                    f"Completions proxy failed after {max_retries} attempts due to connection error: {exc}"
                )
                await record_metrics("/v1/completions", 0, error=True)
                return JSONResponse(
                    status_code=502,
                    content={
                        "error": {
                            "message": f"Upstream request failed: {exc}",
                            "type": "api_error",
                            "code": "bad_gateway",
                        }
                    },
                )
            logger.warning(
                f"Connection lost during completions request: {exc}. Flushing client pool and waiting for llama-server recovery..."
            )
            old_client = client_httpx
            client_httpx = httpx.AsyncClient(timeout=upstream_timeout())
            task = asyncio.create_task(old_client.aclose())
            task.set_name("close-old-client-completions")
            task.add_done_callback(handle_background_task_result)

            await wait_for_llama_server_or_restart(timeout=60.0)
            await ensure_model(model_name)


@app.post("/v1/embeddings")
async def openai_embeddings(request: Request):
    """OpenAI-compatible embeddings. Proxies directly to llama-server."""
    body = await request.json()
    started_ns = now_ns()
    model_name = body.get("model", "")

    if model_name:
        try:
            resolved = await ensure_model(model_name)
            body["model"] = resolved["backend_model"]
        except HTTPException as e:
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": {
                        "message": f"Model resolution failed: {e.detail}",
                        "type": "invalid_request_error",
                        "param": "model",
                        "code": "model_not_found",
                    }
                },
            )

    try:
        resp = await client_httpx.post(f"{LLAMA_SERVER_URL}/v1/embeddings", json=body)
        if resp.status_code != 200:
            err_msg = resp.text
            try:
                upstream_err = resp.json()
                if "error" in upstream_err:
                    err_msg = upstream_err["error"].get("message", err_msg)
            except Exception:
                pass
            await record_metrics("/v1/embeddings", 0, error=True)
            return JSONResponse(
                status_code=resp.status_code,
                content={
                    "error": {
                        "message": err_msg,
                        "type": "invalid_request_error",
                        "code": resp.status_code,
                    }
                },
            )
        data = resp.json()
        latency = (now_ns() - started_ns) / 1e6
        prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
        await record_metrics("/v1/embeddings", latency, prompt_tokens)
        return JSONResponse(data)
    except httpx.TimeoutException:
        await record_metrics("/v1/embeddings", 0, error=True)
        return JSONResponse(
            status_code=504,
            content={
                "error": {
                    "message": "Upstream llama-server timed out",
                    "type": "api_error",
                    "code": "timeout",
                }
            },
        )
    except httpx.RequestError as e:
        await record_metrics("/v1/embeddings", 0, error=True)
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": f"Upstream request failed: {e}",
                    "type": "api_error",
                    "code": "bad_gateway",
                }
            },
        )


# ─── Management API ───────────────────────────────────────────────────────────
@app.get("/admin/health")
async def admin_health():
    """Comprehensive health check for Alpaca proxy, llama-server, and model inventory."""
    health = {
        "proxy": {"status": "ok", "uptime_seconds": round(time.time() - metrics["start_time"], 1)},
        "llama_server": {"status": "unknown", "url": LLAMA_SERVER_URL},
        "models": {"total_local": 0, "loaded": 0, "available": []},
    }

    # Check llama-server health
    try:
        resp = await client_httpx.get(f"{LLAMA_SERVER_URL}/health", timeout=httpx.Timeout(5.0))
        health["llama_server"]["status"] = "ok" if resp.status_code == 200 else "error"
        health["llama_server"]["http_status"] = resp.status_code
    except Exception as e:
        health["llama_server"]["status"] = "error"
        health["llama_server"]["error"] = str(e)

    # Count local models
    local_models = []
    for manifest_base, path, manifest in iter_local_manifests():
        mn = manifest_model_name(manifest_base, path)
        if mn:
            info = manifest_stats(path, manifest)
            local_models.append(
                {
                    "name": mn,
                    "size": info["size"],
                    "details": info["details"],
                    "capabilities": info["capabilities"],
                }
            )
    health["models"]["total_local"] = len(local_models)
    health["models"]["available"] = [m["name"] for m in local_models]

    # Count loaded models
    try:
        loaded = await loaded_models_from_router()
        health["models"]["loaded"] = len(loaded)
        health["models"]["loaded_names"] = [public_model_name(r["name"]) for r in loaded]
    except Exception:
        health["models"]["loaded"] = -1

    # Overall status
    health["overall"] = "ok" if health["llama_server"]["status"] == "ok" else "degraded"
    return JSONResponse(health)


@app.get("/admin/system")
async def admin_system():
    """System information: GPU memory, RAM, CPU, disk usage."""
    import platform
    import shutil

    try:
        import psutil
    except ImportError:
        psutil = None

    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }

    # RAM and CPU usage via psutil
    if psutil:
        try:
            vm = psutil.virtual_memory()
            info["ram_usage"] = {
                "total_gb": round(vm.total / 1e9, 2),
                "available_gb": round(vm.available / 1e9, 2),
                "used_gb": round(vm.used / 1e9, 2),
                "used_pct": vm.percent,
            }
            info["cpu_usage"] = {
                "percent": psutil.cpu_percent(interval=None),
                "load_avg": [round(x, 2) for x in os.getloadavg()]
                if hasattr(os, "getloadavg")
                else [],
            }
        except Exception as e:
            logger.warning(f"Failed to fetch psutil system metrics: {e}")

    # GPU information via docker exec into llama-server (which holds the GPU reservation).
    # The proxy has no GPU device access itself — it uses the Docker socket it already mounts.
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "llama-server",
            "nvidia-smi",
            "--query-gpu=gpu_name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        gpus = []
        if proc.returncode == 0:
            for line in stdout.decode().strip().split("\n"):
                if line:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 4:
                        name, total, used, free = parts
                        total_i, used_i, free_i = int(total), int(used), int(free)
                        gpus.append(
                            {
                                "name": name,
                                "total_mb": total_i,
                                "used_mb": used_i,
                                "free_mb": free_i,
                                "used_pct": round(used_i / total_i * 100, 1) if total_i > 0 else 0,
                            }
                        )
        info["gpu_info"] = gpus if gpus else {"error": "no GPU data from llama-server"}
    except Exception as e:
        logger.debug(f"GPU query via docker exec failed: {e}")
        info["gpu_info"] = {"error": "unavailable"}

    # Disk usage
    for path_name, path in [
        ("models_dir", OLLAMA_BASE),
        ("router_models_dir", ROUTER_MODELS_DIR),
    ]:
        try:
            total, used, free = shutil.disk_usage(path)
            info[f"{path_name}_disk"] = {
                "total_gb": round(total / 1e9, 2),
                "used_gb": round(used / 1e9, 2),
                "free_gb": round(free / 1e9, 2),
                "used_pct": round(used / total * 100, 1) if total > 0 else 0,
            }
        except Exception:
            info[f"{path_name}_disk"] = {"error": "unavailable"}

    # llama-server props (includes GPU info)
    try:
        resp = await client_httpx.get(f"{LLAMA_SERVER_URL}/props", timeout=httpx.Timeout(5.0))
        if resp.status_code == 200:
            props = resp.json()
            info["llama_server_props"] = {
                "model_path": props.get("model_path"),
                "chat_template": props.get("chat_template", "")[:100] + "..."
                if props.get("chat_template")
                else None,
                "n_ctx": props.get("n_ctx"),
                "n_gpu_layers": props.get("n_gpu_layers"),
                "flash_attn": props.get("flash_attn"),
                "build_info": props.get("build_info"),
                "slots": props.get("slots"),
            }
    except Exception:
        info["llama_server_props"] = {"error": "unavailable"}

    # GPU memory from llama-server metrics (if available)
    try:
        resp = await client_httpx.get(f"{LLAMA_SERVER_URL}/metrics", timeout=httpx.Timeout(5.0))
        if resp.status_code == 200:
            # Parse Prometheus metrics for GPU info
            text = resp.text
            gpu_info = {}
            for line in text.split("\n"):
                if "gpu" in line.lower() and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 2:
                        gpu_info[parts[0]] = parts[1]
            if gpu_info:
                info["gpu_metrics"] = gpu_info
    except Exception:
        pass

    return JSONResponse(info)


@app.get("/admin/logs/llama-server")
async def get_llama_server_logs(limit: int = 150):
    """Fetches the last N lines from the llama-server container logs using docker logs."""
    import subprocess

    try:
        res = subprocess.check_output(
            ["docker", "logs", "--tail", str(limit), "llama-server"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        return {"logs": res.split("\n")}
    except Exception as e:
        logger.warning(f"Failed to fetch llama-server container logs: {e}")
        return {
            "error": "Failed to fetch llama-server container logs.",
            "detail": str(e),
            "logs": [
                "Container logs unavailable. Verify that the docker socket/daemon is accessible or permission is granted."
            ],
        }


@app.get("/admin/metrics")
async def admin_metrics():
    """Performance metrics: request counts, token throughput, latency stats."""
    async with metrics_lock:
        uptime = time.time() - metrics["start_time"]
        result = {
            "uptime_seconds": round(uptime, 1),
            "requests_total": metrics["requests_total"],
            "requests_by_endpoint": dict(metrics["requests_by_endpoint"]),
            "tokens_generated": metrics["tokens_generated"],
            "tokens_prompted": metrics["tokens_prompted"],
            "errors_total": metrics["errors_total"],
            "avg_latency_ms": round(metrics["avg_latency_ms"], 2),
            "latency_samples": len(metrics["latency_samples"]),
        }
        # Calculate throughput
        if uptime > 0:
            result["requests_per_second"] = round(metrics["requests_total"] / uptime, 3)
            result["tokens_generated_per_second"] = round(metrics["tokens_generated"] / uptime, 3)
        return JSONResponse(result)


@app.get("/admin/models")
async def admin_models():
    """Extended model inventory with sizes, quantization, manifests, download status."""
    models = []
    for manifest_base, path, manifest in iter_local_manifests():
        mn = manifest_model_name(manifest_base, path)
        if not mn:
            continue
        info = manifest_stats(path, manifest)

        # Check router status
        router_status = "unknown"
        try:
            router_models = await fetch_router_models(reload=False)
            candidates = router_model_candidates(mn, manifest)
            for entry in router_models:
                if router_entry_matches(entry, candidates):
                    router_status = router_entry_status(entry)

                    break
        except Exception:
            pass

        # Calculate blob info
        blobs = []
        total_blob_size = 0
        for layer in manifest.get("layers", []):
            digest = layer.get("digest", "")
            size = layer.get("size", 0)
            blob_path = blob_path_for_digest(digest)
            exists = os.path.exists(blob_path)
            blobs.append(
                {
                    "digest": normalize_digest(digest),
                    "size": size,
                    "exists": exists,
                    "media_type": layer.get("mediaType", ""),
                }
            )
            if exists:
                total_blob_size += size

        models.append(
            {
                "name": mn,
                "size": info["size"],
                "total_blob_size": total_blob_size,
                "digest": info["digest"],
                "details": info["details"],
                "capabilities": info["capabilities"],
                "context_length": info["context_length"],
                "modified_at": info["modified_at"],
                "router_status": router_status,
                "blobs": blobs,
                "blob_count": len(blobs),
                "complete": is_model_complete(manifest),
            }
        )

    return {"models": models, "total": len(models)}


@app.get("/admin/usage")
async def admin_usage():
    """Model usage statistics: load/unload frequency, popularity, recent events."""
    async with model_usage_lock:
        recent_events = list(model_usage_log[-50:])

    # Calculate per-model statistics
    model_stats = {}
    for event in model_usage_log:
        model = event["model"]
        if model not in model_stats:
            model_stats[model] = {
                "loads": 0,
                "unloads": 0,
                "last_loaded": None,
                "last_unloaded": None,
            }
        if event["event"] == "loaded":
            model_stats[model]["loads"] += 1
            model_stats[model]["last_loaded"] = event["timestamp"]
        elif event["event"] == "unloaded":
            model_stats[model]["unloads"] += 1
            model_stats[model]["last_unloaded"] = event["timestamp"]

    # Calculate popularity (total loads per model)
    popular_models = sorted(model_stats.items(), key=lambda x: x[1]["loads"], reverse=True)

    return {
        "recent_events": recent_events,
        "model_stats": model_stats,
        "popular_models": [{"model": m, "stats": s} for m, s in popular_models],
        "total_events": len(model_usage_log),
    }


@app.get("/admin/requests")
async def admin_requests():
    """Retrieve detailed state of currently active and recently completed requests."""
    with active_request_details_lock:
        active = list(active_request_details.values())
        completed = list(completed_requests)
    return {"active_requests": active, "completed_requests": completed}


@app.post("/admin/requests/clear")
async def admin_requests_clear():
    """Clear recently completed requests buffer."""
    with active_request_details_lock:
        completed_requests.clear()
    return {"status": "success", "message": "Completed requests history cleared."}


@app.delete("/admin/requests/{request_id}")
async def cancel_request(request_id: str):
    """Cancel a stuck/active request by ID."""
    with active_request_details_lock:
        if request_id in active_request_details:
            req = active_request_details.pop(request_id)
            return {
                "status": "cancelled",
                "request_id": request_id,
                "model": req.get("model", "unknown"),
            }
        else:
            return {"status": "not_found", "message": f"Request {request_id} not found"}


@app.get("/admin/resubmit/all")
async def get_all_resubmit_data():
    """Get all stored requests for resubmission (keyed by request_id)."""
    with resubmittable_requests_lock:
        return resubmittable_requests


@app.get("/admin/resubmit/{request_id}")
async def get_resubmit_data(request_id: str):
    """Get stored request data for resubmission (persists indefinitely)."""
    with resubmittable_requests_lock:
        if request_id in resubmittable_requests:
            return resubmittable_requests[request_id]
        else:
            return JSONResponse(
                status_code=404,
                content={"error": f"Request {request_id} not found in persistent storage"},
            )


@app.get("/admin/runtime")
async def admin_runtime():
    """Runtime state: loaded models, active requests, keep-alive timers, queue depth."""
    # Active requests
    async with active_requests_lock:
        active = dict(active_requests)

    # Model expiry timers
    expiry_info = {}
    for name, expires in model_expires_at.items():
        expiry_info[name] = {
            "expires_at": expires,
            "has_unload_task": name in model_unload_tasks,
        }

    # Fetch live llama-server props once (n_ctx, n_gpu_layers, flash_attn etc.)
    live_props = {}
    try:
        resp = await client_httpx.get(f"{LLAMA_SERVER_URL}/props", timeout=httpx.Timeout(3.0))
        if resp.status_code == 200:
            live_props = resp.json()
    except Exception:
        pass

    # Loaded models — enriched with running_settings from .profile.json + live props
    loaded = []
    try:
        router_models = await fetch_router_models(reload=False)
        for manifest_base, path, manifest in iter_local_manifests():
            model_name = manifest_model_name(manifest_base, path)
            if not model_name:
                continue
            candidates = router_model_candidates(model_name, manifest)
            matched_entry = None
            for entry in router_models:
                if router_entry_status(entry) == "loaded" and router_entry_matches(
                    entry, candidates
                ):
                    matched_entry = entry
                    break
            if not matched_entry:
                continue

            info = manifest_stats(path, manifest)
            public_name = public_model_name(model_name)
            backend_model = matched_entry.get("id", "")

            # Load running settings: models.ini is canonical, .profile.json fills gaps
            running_settings = {}
            try:
                import configparser

                ini_path = os.path.join(ROUTER_MODELS_DIR, "models.ini")
                if os.path.exists(ini_path):
                    cfg = configparser.ConfigParser(delimiters=("=",))
                    cfg.read(ini_path)
                    for section_key in [backend_model, public_name]:
                        if cfg.has_section(section_key):
                            running_settings = dict(cfg[section_key])
                            break

                # .profile.json only supplies fields not already set by models.ini
                profile_path = os.path.join(ROUTER_MODELS_DIR, f"{backend_model}.profile.json")
                if os.path.exists(profile_path):
                    with open(profile_path) as f:
                        profile_settings = json.load(f)
                        for k, v in profile_settings.items():
                            running_settings.setdefault(k, v)
            except Exception:
                pass

            # Overlay live llama-server values (authoritative for current run)
            if live_props.get("n_ctx"):
                running_settings["ctx-size"] = str(live_props["n_ctx"])
            if live_props.get("n_gpu_layers") is not None:
                running_settings["n-gpu-layers"] = str(live_props["n_gpu_layers"])
            if live_props.get("flash_attn") is not None:
                running_settings["flash-attn"] = "on" if live_props["flash_attn"] else "off"

            loaded.append(
                {
                    "name": public_name,
                    "backend_model": backend_model,
                    "size": info["size"],
                    "digest": info["digest"],
                    "details": info["details"],
                    "context_length": info["context_length"],
                    "expires_at": model_expires_at.get(public_name, "0001-01-01T00:00:00Z"),
                    "active_requests": active.get(backend_model, 0),
                    "running_settings": running_settings,
                    "peak_active_requests": 0,
                    "total_requests_processed": 0,
                }
            )
    except Exception:
        pass

    # Loading models
    loading = []
    current_time = time.time()
    for name, entry in list(model_loading.items()):
        if current_time - entry["start_time"] <= MODEL_LOADING_TIMEOUT:
            loading.append({
                "name": name,
                "elapsed_seconds": int(current_time - entry["start_time"])
            })

    return {
        "loaded_models": loaded,
        "loading_models": loading,
        "active_requests": active,
        "model_expiry_timers": expiry_info,
        "max_loaded_models": MAX_LOADED_MODELS,
        "default_keep_alive": DEFAULT_KEEP_ALIVE,
        "router_management_supported": router_management_supported,
    }


@app.get("/admin/slots")
async def admin_slots(fail_on_no_slot: int = 0):
    """llama-server slot status with enhanced Alpaca metadata."""
    try:
        resp = await client_httpx.get(
            f"{LLAMA_SERVER_URL}/slots",
            params={"fail_on_no_slot": fail_on_no_slot},
            timeout=httpx.Timeout(5.0),
        )
        resp.raise_for_status()
        slots = resp.json()

        # Enhance with Alpaca metadata
        for slot in slots:
            slot["alpaca"] = {
                "is_busy": slot.get("is_processing", False),
                "has_prompt_cache": bool(slot.get("prompt", [])),
                "context_used_pct": round(
                    slot.get("n_ctx", 0) / max(slot.get("n_ctx", 1), 1) * 100, 1
                )
                if slot.get("n_ctx")
                else 0,
            }

        return {"slots": slots, "total": len(slots)}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch slot info: {e}")


@app.get("/admin/lora")
async def admin_lora():
    """LoRA adapter status and scale control."""
    try:
        resp = await client_httpx.get(
            f"{LLAMA_SERVER_URL}/lora-adapters", timeout=httpx.Timeout(5.0)
        )
        resp.raise_for_status()
        adapters = resp.json()
        return {"adapters": adapters, "total": len(adapters)}
    except Exception as e:
        return {
            "adapters": [],
            "error": str(e),
            "note": "LoRA adapters not configured or endpoint unavailable",
        }


@app.post("/admin/lora")
async def admin_lora_update(request: Request):
    """Update LoRA adapter scales. Body: [{\"id\": 0, \"scale\": 0.5}, ...]"""
    body = await request.json()
    if not isinstance(body, list):
        raise HTTPException(status_code=400, detail="Body must be a list of {id, scale} objects")
    try:
        resp = await client_httpx.post(
            f"{LLAMA_SERVER_URL}/lora-adapters", json=body, timeout=httpx.Timeout(5.0)
        )
        resp.raise_for_status()
        return {"status": "updated", "adapters": body}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to update LoRA adapters: {e}")


@app.get("/admin/config")
async def admin_config():
    """Current runtime configuration."""
    return {
        "llama_server_url": LLAMA_SERVER_URL,
        "ollama_base": OLLAMA_BASE,
        "model_namespace": MODEL_NAMESPACE,
        "api_version": API_VERSION,
        "default_keep_alive": DEFAULT_KEEP_ALIVE,
        "max_loaded_models": MAX_LOADED_MODELS,
        "router_models_dir": ROUTER_MODELS_DIR,
        "engine_startup_timeout": ENGINE_STARTUP_TIMEOUT_SECONDS,
        "llama_server_connect_timeout": LLAMA_SERVER_CONNECT_TIMEOUT_SECONDS,
        "llama_server_read_timeout": LLAMA_SERVER_READ_TIMEOUT_SECONDS,
        "router_management_supported": router_management_supported,
        "grammar_registry_dir": GRAMMAR_REGISTRY_DIR,
        "schema_registry_dir": SCHEMA_REGISTRY_DIR,
    }


@app.post("/admin/config")
async def admin_config_update(request: Request):
    """Update runtime configuration. Only mutable fields: default_keep_alive, max_loaded_models."""
    body = await request.json()
    global DEFAULT_KEEP_ALIVE, MAX_LOADED_MODELS

    if "default_keep_alive" in body:
        # In a real implementation, you'd use a mutable config object
        # For now, this is informational
        pass
    if "max_loaded_models" in body:
        pass

    return {
        "status": "config_update_requested",
        "note": "Runtime config changes require restart for full effect",
        "requested": body,
    }


@app.post("/admin/models/pull")
async def admin_model_pull(request: Request):
    """Trigger model pull via alpaca-puller. Body: {\"model\": \"name:tag\", \"source\": \"ollama|huggingface\"}"""
    body = await request.json()
    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    # Check if model already exists locally
    if manifest_path_for_model(model):
        return {"status": "already_exists", "model": model}

    # For now, return instructions. In production, this would trigger the puller
    return {
        "status": "pull_requested",
        "model": model,
        "note": "Run alpaca-puller.py to download: python alpaca-puller.py pull {model}",
        "source": body.get("source", "ollama"),
    }


@app.post("/admin/models/delete")
async def admin_model_delete(request: Request):
    """Delete a model and clean up blobs. Body: {\"model\": \"name:tag\"}"""
    body = await request.json()
    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    model = with_default_tag(model)
    manifest_path = manifest_path_for_model(model)
    if not manifest_path:
        raise HTTPException(status_code=404, detail=f"Model {model} not found")

    # Read manifest to find blobs
    manifest = read_manifest(manifest_path)
    if not manifest:
        raise HTTPException(status_code=500, detail=f"Model {model} manifest is corrupted")

    # Unload from router if loaded
    try:
        resolved = await resolve_router_model(model, reload=False)
        backend_model = resolved["backend_model"]
        try:
            await post_router_model_action("unload", backend_model)
        except RouterManagementUnsupported:
            pass
    except HTTPException:
        pass
    await record_model_unloaded(model)

    # Remove manifest
    os.remove(manifest_path)
    deleted_blobs = []

    # Clean up orphaned blobs
    for layer in manifest.get("layers", []) + [manifest.get("config", {})]:
        digest = layer.get("digest")
        if digest:
            blob_path = blob_path_for_digest(digest)
            if os.path.exists(blob_path):
                # Check if any other manifest references this blob
                referenced = False
                for mb, mp, m in iter_local_manifests():
                    if mp == manifest_path:
                        continue
                    for l in m.get("layers", []) + [m.get("config", {})]:
                        if l.get("digest") == digest:
                            referenced = True
                            break
                    if referenced:
                        break
                if not referenced:
                    os.remove(blob_path)
                    deleted_blobs.append(normalize_digest(digest))

    # Clean up router symlink
    router_path = router_path_for_model_name(model)
    if os.path.islink(router_path):
        os.remove(router_path)
    elif os.path.exists(router_path):
        os.remove(router_path)

    return {
        "status": "deleted",
        "model": model,
        "manifest": manifest_path,
        "deleted_blobs": deleted_blobs,
    }


@app.post("/admin/models/switch")
async def admin_model_switch(request: Request):
    """Switch to a different model. Body: {\"model\": \"name:tag\"}. Loads the model (unloads current if needed)."""
    body = await request.json()
    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    model = with_default_tag(model)

    # Check if model exists
    manifest_path = manifest_path_for_model(model)
    if not manifest_path or not os.path.exists(manifest_path):
        raise HTTPException(status_code=404, detail=f"Model {model} not found on disk")

    # Check if already loading
    public = public_model_name(model)
    if is_model_loading(model):
        return {"status": "already_loading", "model": public}

    # Check if already loaded
    try:
        router_models = await fetch_router_models(reload=False)
        candidates = [model, with_default_tag(model)]
        for entry in router_models:
            if router_entry_matches(entry, candidates):
                current_status = router_entry_status(entry)
                if is_resident_status(current_status):
                    return {
                        "status": "already_loaded",
                        "model": public,
                        "backend_model": entry.get("id"),
                    }
    except Exception:
        pass

    # Mark as loading
    mark_model_loading(model)

    try:
        # Load the model using the full ensure_model flow (resolve + load)
        resolved = await ensure_model(model)

        return {"status": "loaded", "model": public, "backend_model": resolved.get("backend_model")}
    except Exception as e:
        mark_model_loaded(model)  # Clear loading state even on failure
        logger.error(f"Failed to switch to model {model}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.post("/admin/models/unload")
async def admin_model_unload(request: Request):
    """Unload a model. Body: {\"model\": \"name:tag\"}."""
    body = await request.json()
    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    model = with_default_tag(model)

    # Find the backend_model for this model
    try:
        router_models = await fetch_router_models(reload=False)
        candidates = [model, with_default_tag(model)]
        for entry in router_models:
            if router_entry_matches(entry, candidates):
                current_status = router_entry_status(entry)
                if is_resident_status(current_status):
                    backend_model = entry.get("id")
                    
                    # Check active requests before unloading
                    async with active_requests_lock:
                        if active_requests.get(backend_model, 0) > 0:
                            raise HTTPException(
                                status_code=409,
                                detail=f"Cannot unload model {model} because it currently has {active_requests[backend_model]} active request(s)."
                            )
                            
                    await post_router_model_action("unload", backend_model)
                    public = public_model_name(model)
                    await record_model_unloaded(model)
                    return {"status": "unloaded", "model": public, "backend_model": backend_model}

        # Model not found or not loaded
        return {"status": "not_loaded", "model": model}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to unload model {model}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")


@app.post("/admin/vram/clear")
async def admin_vram_clear():
    """Force clear all VRAM by unloading all active models and restarting llama-server."""
    try:
        # 1. Unload all resident models via router
        router_models = await fetch_router_models(reload=True)
        unloaded_count = 0
        for entry in router_models:
            current_status = router_entry_status(entry)
            if is_resident_status(current_status):
                backend_model = entry.get("id")
                try:
                    await post_router_model_action("unload", backend_model)
                    await record_model_unloaded_by_backend_id(backend_model)
                    unloaded_count += 1
                except Exception as unload_err:
                    logger.warning(f"Failed to unload {backend_model} during VRAM clear: {unload_err}")

        # 2. Force reset active requests counter to prevent stuck locks
        async with active_requests_lock:
            active_requests.clear()
            active_requests_lock.notify_all()

        # 3. Always trigger a docker restart on llama-server to guarantee 100% VRAM release
        logger.info("Restarting llama-server container to guarantee 100% VRAM cleanup...")
        await restart_llama_server()
        
        # 4. Wait for it to come back online
        if not await wait_for_llama_server_or_restart(timeout=30.0):
            raise HTTPException(status_code=502, detail="llama-server failed to recover after VRAM clear restart")

        return {
            "status": "success",
            "message": f"Successfully cleared VRAM. Unloaded {unloaded_count} model(s) and restarted llama-server.",
            "unloaded_models_count": unloaded_count
        }
    except Exception as e:
        logger.error(f"Failed to clear VRAM: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear VRAM: {str(e)}")


@app.post("/admin/models/copy")
async def admin_model_copy(request: Request):
    """Copy a model to a new name/tag. Body: {\"source\": \"name:tag\", \"target\": \"newname:newtag\"}"""
    body = await request.json()
    source = body.get("source")
    target = body.get("target")
    if not source or not target:
        raise HTTPException(status_code=400, detail="source and target are required")

    source = with_default_tag(source)
    target = with_default_tag(target)

    source_manifest_path = manifest_path_for_model(source)
    if not source_manifest_path:
        raise HTTPException(status_code=404, detail=f"Source model {source} not found")

    target_manifest_path = manifest_path_for_model(target)
    if target_manifest_path:
        raise HTTPException(status_code=409, detail=f"Target model {target} already exists")

    # Create target directory and copy manifest
    target_dir = os.path.dirname(target_manifest_path)
    os.makedirs(target_dir, exist_ok=True)

    # Read source manifest and update model name references
    with open(source_manifest_path) as f:
        manifest = json.load(f)

    with open(target_manifest_path, "w") as f:
        json.dump(manifest, f)

    return {
        "status": "copied",
        "source": source,
        "target": target,
        "target_manifest": target_manifest_path,
    }


@app.get("/admin/tokenize")
async def admin_tokenize(text: str, add_special: bool = True, parse_special: bool = False):
    """Tokenize text using llama-server. Returns token IDs."""
    try:
        resp = await client_httpx.post(
            f"{LLAMA_SERVER_URL}/tokenize",
            json={"content": text, "add_special": add_special, "parse_special": parse_special},
            timeout=httpx.Timeout(5.0),
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Tokenization failed: {e}")


@app.get("/admin/props")
async def admin_props():
    """Get llama-server properties (model info, build, slots config)."""
    try:
        resp = await client_httpx.get(f"{LLAMA_SERVER_URL}/props", timeout=httpx.Timeout(5.0))
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch props: {e}")


@app.get("/api/logs")
async def get_logs(limit: int = 100):
    """Returns the last N lines from the in-memory log buffer."""
    logs = list(LOG_BUFFER)
    return {"logs": logs[-limit:]}


MTP_INCOMPATIBLE_FILE = os.path.join(ROUTER_MODELS_DIR, ".mtp_incompatible_models.json")
MTP_INCOMPATIBLE_MODELS = set()

SAFE_SETTINGS_FILE = os.path.join(ROUTER_MODELS_DIR, ".safe_settings_models.json")
SAFE_SETTINGS_MODELS = set()


def load_mtp_incompatible_models():
    global MTP_INCOMPATIBLE_MODELS
    try:
        if os.path.exists(MTP_INCOMPATIBLE_FILE):
            with open(MTP_INCOMPATIBLE_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    MTP_INCOMPATIBLE_MODELS = set(data)
                    logger.info(f"Loaded MTP incompatible models: {MTP_INCOMPATIBLE_MODELS}")
    except Exception as e:
        logger.warning(f"Failed to load MTP incompatible models file: {e}")


def save_mtp_incompatible_models():
    try:
        with open(MTP_INCOMPATIBLE_FILE, "w") as f:
            json.dump(list(MTP_INCOMPATIBLE_MODELS), f, indent=2)
            logger.info("Saved MTP incompatible models list.")
        try:
            from alpaca_puller import update_models_ini

            update_models_ini()
        except Exception as e:
            logger.warning(f"Failed to regenerate models.ini: {e}")
    except Exception as e:
        logger.warning(f"Failed to save MTP incompatible models file: {e}")


def load_safe_settings_models():
    global SAFE_SETTINGS_MODELS
    try:
        if os.path.exists(SAFE_SETTINGS_FILE):
            with open(SAFE_SETTINGS_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    SAFE_SETTINGS_MODELS = set(data)
                    logger.info(f"Loaded safe settings models: {SAFE_SETTINGS_MODELS}")
    except Exception as e:
        logger.warning(f"Failed to load safe settings models file: {e}")


def save_safe_settings_models():
    try:
        with open(SAFE_SETTINGS_FILE, "w") as f:
            json.dump(list(SAFE_SETTINGS_MODELS), f, indent=2)
            logger.info("Saved safe settings models list.")
        try:
            from alpaca_puller import update_models_ini

            update_models_ini()
        except Exception as e:
            logger.warning(f"Failed to regenerate models.ini: {e}")
    except Exception as e:
        logger.warning(f"Failed to save safe settings models file: {e}")


# Load initially on import
load_mtp_incompatible_models()
load_safe_settings_models()


async def save_loaded_models_state(loaded_models):
    try:
        os.makedirs(os.path.dirname(LOADED_MODELS_STATE_FILE), exist_ok=True)
        with open(LOADED_MODELS_STATE_FILE, "w") as f:
            json.dump(loaded_models, f)
        logger.info(f"Saved loaded models state: {loaded_models}")
    except Exception as e:
        logger.warning(f"Failed to save loaded models state: {e}")


async def load_loaded_models_state():
    try:
        if os.path.exists(LOADED_MODELS_STATE_FILE):
            with open(LOADED_MODELS_STATE_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load loaded models state: {e}")
    return []


async def record_model_loaded(model_name):
    async with _loaded_models_state_lock:
        current_loaded = await load_loaded_models_state()
        public = public_model_name(model_name)
        if public in current_loaded:
            current_loaded.remove(public)
        current_loaded.append(public)
        await save_loaded_models_state(current_loaded)

    async with model_usage_lock:
        model_usage_log.append({"event": "loaded", "model": public, "timestamp": time.time()})
        while len(model_usage_log) > MODEL_USAGE_LOG_MAX:
            model_usage_log.pop(0)


async def record_model_unloaded(model_name):
    async with _loaded_models_state_lock:
        current_loaded = await load_loaded_models_state()
        public = public_model_name(model_name)
        if public in current_loaded:
            current_loaded.remove(public)
            await save_loaded_models_state(current_loaded)

    async with model_usage_lock:
        model_usage_log.append({"event": "unloaded", "model": public, "timestamp": time.time()})
        while len(model_usage_log) > MODEL_USAGE_LOG_MAX:
            model_usage_log.pop(0)


async def record_model_unloaded_by_backend_id(backend_id):
    async with _loaded_models_state_lock:
        current_loaded = await load_loaded_models_state()
        modified = False
        for name in list(current_loaded):
            if router_model_id_for_name(name) == backend_id:
                current_loaded.remove(name)
                modified = True
        if modified:
            await save_loaded_models_state(current_loaded)


async def restore_models_on_recovery():
    loaded_models = await load_loaded_models_state()
    if not loaded_models:
        logger.info("No persisted loaded models to restore.")
        return
    # Only preload the last loaded model to conserve VRAM and avoid concurrent loading OOMs
    last_model = loaded_models[-1]
    logger.info(
        f"Restoring only the last loaded model after recovery: {last_model} (out of {loaded_models})"
    )
    try:
        logger.info(f"Auto-loading model on recovery: {last_model}")
        task = asyncio.create_task(ensure_model(last_model))
        task.set_name(f"restore-model-{last_model}")
        task.add_done_callback(handle_background_task_result)
    except Exception as e:
        logger.error(f"Failed to auto-load {last_model} on recovery: {e}")


async def wait_for_llama_server(timeout=300.0):
    start_time = time.time()
    logger.info("Waiting for llama-server to become responsive...")
    while time.time() - start_time < timeout:
        try:
            resp = await client_httpx.get(f"{LLAMA_SERVER_URL}/health", timeout=httpx.Timeout(2.0))
            if resp.status_code == 200:
                logger.info("llama-server is responsive and healthy.")
                return True
        except Exception:
            pass
        await asyncio.sleep(0.5)
    logger.error(f"llama-server failed to become responsive within {timeout} seconds.")
    return False


llama_server_restart_lock = asyncio.Lock()
last_llama_server_restart_time = 0.0


async def restart_llama_server():
    global last_llama_server_restart_time
    async with llama_server_restart_lock:
        now = time.time()
        # 15 seconds cooldown to prevent rapid consecutive restarts under concurrent loads
        if now - last_llama_server_restart_time < 15.0:
            logger.info("llama-server was restarted very recently. Skipping redundant restart.")
            return True

        logger.info("Initiating single synchronized restart of llama-server...")
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "restart",
                "llama-server",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                logger.info("llama-server restart command succeeded.")
                last_llama_server_restart_time = time.time()
                return True
            else:
                logger.error(
                    f"llama-server restart command failed (exit {proc.returncode}): {stderr.decode().strip()}"
                )
                return False
        except FileNotFoundError:
            logger.error("docker command not found — cannot restart llama-server.")
            return False
        except Exception as e:
            logger.error(f"Failed to restart llama-server: {e}")
            return False


async def _fetch_llama_server_logs(tail=30) -> str:
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "logs",
            f"--tail={tail}",
            "llama-server",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return (stdout.decode("utf-8", errors="replace") + "\n" + stderr.decode("utf-8", errors="replace"))
    except Exception as e:
        return f"Failed to get container logs: {e}"


async def raise_model_load_failure_exception(model_name: str, backend_model: str, original_error: str):
    logs = await _fetch_llama_server_logs(tail=30)
    logs_lower = logs.lower()
    
    suggested_fix = ""
    if "out of memory" in logs_lower or "cudamalloc failed" in logs_lower or "unable to allocate" in logs_lower:
        suggested_fix = (
            "Suggested Fixes:\n"
            "1. Decrease the KV cache quantization level (e.g. from q8_0 to q4_0) in Model Profiles.\n"
            "2. Reduce n-gpu-layers to offload fewer layers to GPU (freeing up VRAM).\n"
            "3. Reduce ctx-size in the Model Profiles tab."
        )
    elif "no such file" in logs_lower or "failed to load model" in logs_lower:
        suggested_fix = (
            "Suggested Fixes:\n"
            "1. Make sure the model files exist in the router models directory (/router-models).\n"
            "2. Verify the model file format is a valid and uncorrupted GGUF file."
        )
    
    detailed_msg = f"Failed to load model {model_name} ({backend_model}): {original_error}"
    if suggested_fix:
        detailed_msg += f"\n\n[Diagnosis] VRAM/resource limitation detected.\n{suggested_fix}"
    
    logger.error(f"Model load failure exception raised: {detailed_msg}\nLast logs:\n{logs}")
    raise HTTPException(status_code=500, detail=detailed_msg)


async def wait_for_llama_server_or_restart(timeout=300.0):
    start_time = time.time()
    restart_triggered = False
    while time.time() - start_time < timeout:
        try:
            resp = await client_httpx.get(f"{LLAMA_SERVER_URL}/health", timeout=httpx.Timeout(2.0))
            if resp.status_code == 200:
                # Active verification: Check if any loaded models are actually healthy/responsive
                try:
                    models_resp = await client_httpx.get(
                        f"{LLAMA_SERVER_URL}/models", timeout=httpx.Timeout(2.0)
                    )
                    if models_resp.status_code == 200:
                        loaded_models = []
                        data = models_resp.json().get("data", [])
                        for model in data:
                            if model.get("status", {}).get("value") == "loaded":
                                loaded_models.append(model.get("id"))

                        all_healthy = True
                        for m_id in loaded_models:
                            if not await is_child_model_healthy(m_id):
                                logger.warning(
                                    f"Active health check failed: loaded model {m_id} is unresponsive."
                                )
                                all_healthy = False
                                break

                        if all_healthy:
                            logger.info(
                                "llama-server and all loaded models are responsive and healthy."
                            )
                            return True
                    else:
                        logger.warning(
                            f"Failed to fetch models list during health check: {models_resp.status_code}"
                        )
                except Exception as check_exc:
                    logger.warning(
                        f"Error checking child model health during recovery: {check_exc}"
                    )
        except Exception:
            pass

        if not restart_triggered and time.time() - start_time > 15.0:
            logger.info(
                "llama-server or child model not responding after 15s — attempting docker restart..."
            )
            restart_triggered = True
            if await restart_llama_server():
                logger.info(
                    "llama-server restart command succeeded, waiting 5s for GPU memory release..."
                )
                await asyncio.sleep(5.0)
                # Reset start_time to give it a full timeout window to boot up
                start_time = time.time()
                continue
            else:
                logger.error("llama-server restart failed.")
                return False
        await asyncio.sleep(1.0)
    logger.error(f"llama-server failed to become responsive within {timeout} seconds.")
    return False


async def is_child_model_healthy(backend_model: str) -> bool:
    """Check if the child model status on the router is 'loaded' or 'loading' and verified functional."""
    if os.getenv("PYTEST_CURRENT_TEST"):
        return True
    try:
        resp = await client_httpx.get(f"{LLAMA_SERVER_URL}/models")
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            for model in data:
                if model.get("id") == backend_model:
                    status = model.get("status", {}).get("value")
                    if status == "loading":
                        return True
                    if status == "loaded":
                        # Perform active direct port probe to ensure the child process is alive and responsive
                        args = model.get("status", {}).get("args", [])
                        try:
                            port_idx = args.index("--port")
                            port = args[port_idx + 1]
                        except (ValueError, IndexError):
                            logger.warning(
                                f"Could not find port in status args for model {backend_model}"
                            )
                            return False

                        try:
                            proc = await asyncio.create_subprocess_exec(
                                "docker",
                                "exec",
                                "llama-server",
                                "curl",
                                "-s",
                                "-m",
                                "3",
                                f"http://127.0.0.1:{port}/health",
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE,
                            )
                            stdout, stderr = await proc.communicate()
                            if proc.returncode == 0:
                                res = json.loads(stdout.decode().strip())
                                if res.get("status") == "ok":
                                    return True
                            logger.warning(
                                f"Health probe to child port {port} failed with returncode {proc.returncode}: {stdout.decode().strip()} {stderr.decode().strip()}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Health probe to child port {port} raised exception: {e}"
                            )
                        return False
    except Exception as e:
        logger.warning(f"Failed to check child model health for {backend_model}: {e}")
    return False


def is_model_over_9b(model_name: str, manifest: dict = None) -> bool:
    """Check if a model's size/parameter count is higher than 9B."""
    import re

    # 1. Parse name for size indicators like "35b", "14b", "32b", "70b"
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", model_name)
    if m:
        try:
            val = float(m.group(1))
            if val > 9.0:
                return True
        except ValueError:
            pass

    # 2. Check manifest parameter count metadata if available
    if manifest:
        layers = manifest.get("layers", [])
        for layer in layers:
            if layer.get("mediaType", "").startswith("application/vnd.ollama.image.model"):
                size_bytes = layer.get("size", 0)
                # If the GGUF blob is > 8.5 GB, it is definitely > 9B parameters
                if size_bytes > 8.5 * 1024 * 1024 * 1024:
                    return True
    return False


def is_model_moe(model_name: str, meta: dict = None) -> bool:
    """Check if the model is a Mixture of Experts (MoE) model."""
    if meta:
        return _is_moe(meta)
    return "moe" in model_name.lower() or "mixtral" in model_name.lower()


_FA_UNSUPPORTED_ARCHS = {
    "mamba",
    "rwkv",
    "rwkv6",
    "wavtokenizer",
}


def _read_gguf_metadata(path: str) -> dict:
    """Parse only the metadata header from a GGUF file (fast, no full load)."""
    import struct

    meta: dict = {}
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return meta
            struct.unpack("<I", f.read(4))
            struct.unpack("<Q", f.read(8))
            kv_count = struct.unpack("<Q", f.read(8))[0]

            for _ in range(kv_count):
                key_len = struct.unpack("<Q", f.read(8))[0]  # uint64, not uint32
                key = f.read(key_len).decode("utf-8", errors="replace")
                val_type = struct.unpack("<I", f.read(4))[0]

                # GGUF value types: 0=uint8,1=int8,2=uint16,3=int16,4=uint32,
                # 5=int32,6=float32,7=bool,8=str,9=array,10=uint64,11=int64,12=float64
                if val_type == 8:  # string
                    str_len = struct.unpack("<Q", f.read(8))[0]
                    val = f.read(str_len).decode("utf-8", errors="replace")
                    meta[key] = val
                elif val_type in (0, 1, 4, 5, 10, 11):  # integer types
                    fmt = {0: "<B", 1: "<b", 4: "<I", 5: "<i", 10: "<Q", 11: "<q"}
                    val = struct.unpack(fmt[val_type], f.read(struct.calcsize(fmt[val_type])))[0]
                    meta[key] = val
                elif val_type == 7:  # bool
                    meta[key] = struct.unpack("<?", f.read(1))[0]
                elif val_type == 6:  # float32
                    meta[key] = struct.unpack("<f", f.read(4))[0]
                elif val_type == 12:  # float64
                    meta[key] = struct.unpack("<d", f.read(8))[0]
                elif val_type == 9:  # array
                    arr_type = struct.unpack("<I", f.read(4))[0]
                    arr_len = struct.unpack("<Q", f.read(8))[0]
                    # Skip array contents — we only care about scalar metadata
                    if arr_type == 8:  # array of strings
                        for _ in range(arr_len):
                            sl = struct.unpack("<Q", f.read(8))[0]
                            f.read(sl)
                    else:
                        sizes = {
                            0: 1,
                            1: 1,
                            2: 2,
                            3: 2,
                            4: 4,
                            5: 4,
                            6: 4,
                            7: 1,
                            10: 8,
                            11: 8,
                            12: 8,
                        }
                        skip = arr_len * sizes.get(arr_type, 0)
                        f.read(skip)
                else:
                    # Unknown type — skip conservatively
                    break
    except Exception:
        pass
    return meta


def _is_moe(meta: dict) -> bool:
    # Check all keys for expert_count/expert_used_count (architecture-prefixed)
    for key, val in meta.items():
        if key.endswith(".expert_count") and isinstance(val, int) and val > 0:
            return True
        if key.endswith(".expert_used_count") and isinstance(val, int) and val > 0:
            return True
    arch = meta.get("general.architecture", "")
    if isinstance(arch, str) and "moe" in arch.lower():
        return True
    return False


def _supports_flash_attn(meta: dict) -> bool:
    arch = meta.get("general.architecture", "").lower()
    if not arch:
        return False  # unknown arch — don't risk it
    return arch not in _FA_UNSUPPORTED_ARCHS


# ---------------------------------------------------------------------------
# VRAM-aware n_gpu_layers computation (metadata-driven, no hardcoded model names)
#
# The llama.cpp router reads n_gpu_layers ONLY from models.ini at startup,
# ignoring the /models/load payload. So to override a bad preset we write the
# corrected value to the INI, then restart llama-server (which the recovery
# path already does). After restart the router picks up the new value.
# ---------------------------------------------------------------------------

_KV_CACHE_BYTES = {"f16": 2, "q8_0": 1, "q4_0": 0.5, "bf16": 2}
_VRAM_OVERHEAD_MIB = 256


def _get_model_arch_meta(meta: dict) -> tuple:
    """Extract (arch, n_layers, n_embd) from GGUF metadata."""
    arch = meta.get("general.architecture", "")
    n_layers = meta.get(f"{arch}.block_count", 0) or 0
    n_embd = meta.get(f"{arch}.embedding_length", 0) or 0
    return arch, n_layers, n_embd


async def _get_available_vram_mib() -> int | None:
    """Query free GPU VRAM in MiB via nvidia-smi in the llama-server container."""
    if os.getenv("PYTEST_CURRENT_TEST"):
        return None
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "llama-server",
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0:
            lines = stdout.decode().strip().split("\n")
            if lines:
                return int(lines[0].strip())
    except Exception as e:
        logger.debug(f"nvidia-smi failed: {e}")
    return None


def _estimate_vram_mib(model_path, meta, n_ctx, cache_type, n_parallel, n_gpu_layers):
    """Estimate GPU VRAM usage in MiB for the given parameters.

    Only layers on GPU (n_gpu_layers) consume VRAM for weights, KV cache,
    and compute buffers. Layers on CPU don't contribute to VRAM usage.
    """
    _, n_layers, n_embd = _get_model_arch_meta(meta)
    file_size_mib = os.path.getsize(model_path) / (1024 * 1024)
    if n_gpu_layers <= 0:
        return _VRAM_OVERHEAD_MIB
    # Weights: proportional to n_gpu_layers / n_layers
    weights_mib = (
        file_size_mib * (n_gpu_layers / n_layers)
        if n_layers > 0 and n_gpu_layers < n_layers
        else file_size_mib
    )
    # KV cache: only for layers on GPU (proportional to n_gpu_layers / n_layers)
    kv_cache_mib = 0
    if n_layers and n_embd:
        kv_bytes = _KV_CACHE_BYTES.get(cache_type, 2)
        kv_cache_mib = (2 * n_gpu_layers * n_ctx * n_embd * kv_bytes * n_parallel) / (1024 * 1024)
    # Compute buffers: proportional to weights on GPU
    compute_mib = max(weights_mib * 0.12, 128)
    return int(weights_mib + kv_cache_mib + compute_mib + _VRAM_OVERHEAD_MIB)


def _compute_safe_n_gpu_layers(
    model_path, meta, n_ctx, cache_type, n_parallel, available_vram_mib, requested_n_gpu_layers
):
    """Binary-search for the max n_gpu_layers that fits in available VRAM (90% margin)."""
    _, n_layers, _ = _get_model_arch_meta(meta)
    if n_layers == 0:
        return requested_n_gpu_layers
    target_vram = int(available_vram_mib * 0.90)
    if requested_n_gpu_layers < 0:
        candidate = n_layers
    elif requested_n_gpu_layers == 0:
        candidate = 0
    else:
        candidate = min(requested_n_gpu_layers, n_layers)
    estimated = _estimate_vram_mib(model_path, meta, n_ctx, cache_type, n_parallel, candidate)
    if estimated <= target_vram:
        logger.info(
            f"VRAM check passed: ~{estimated}MiB needed, ~{available_vram_mib}MiB available (n_gpu_layers={candidate})"
        )
        return requested_n_gpu_layers
    lo, hi, best = 0, candidate, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if _estimate_vram_mib(model_path, meta, n_ctx, cache_type, n_parallel, mid) <= target_vram:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    logger.warning(
        f"VRAM budgeting: n_gpu_layers={requested_n_gpu_layers}→{best} (~{estimated}MiB needed, ~{available_vram_mib}MiB available)"
    )
    return best


def _write_ini_model_setting(backend_model, key, value):
    """Write a setting to a model section in models.ini. No-op if already correct."""
    ini_path = os.path.join(ROUTER_MODELS_DIR, "models.ini")
    if not os.path.exists(ini_path):
        return
    try:
        import configparser

        config = configparser.ConfigParser()
        config.read(ini_path)
        if config.has_section(backend_model) and config[backend_model].get(key) == str(value):
            return
        if not config.has_section(backend_model):
            config.add_section(backend_model)
        config[backend_model][key] = str(value)
        with open(ini_path, "w") as f:
            config.write(f)
        logger.info(f"Updated models.ini: [{backend_model}] {key} = {value}")
    except Exception as e:
        logger.warning(f"Failed to write {key}={value} to models.ini: {e}")


def _read_ini_model_setting(backend_model, key, default=""):
    """Read a setting for a model from models.ini, checking [model] then [*] defaults."""
    ini_path = os.path.join(ROUTER_MODELS_DIR, "models.ini")
    if not os.path.exists(ini_path):
        return default
    try:
        import configparser

        config = configparser.ConfigParser()
        config.read(ini_path)
        if config.has_section(backend_model) and key in config[backend_model]:
            return config[backend_model][key]
        if config.has_section("*") and key in config["*"]:
            return config["*"][key]
    except Exception as e:
        logger.warning(f"Failed to read {key} from models.ini: {e}")
    return default


async def get_child_model_props(backend_model: str) -> dict:
    """Retrieve /props directly from the spawned llama-server child process container."""
    if os.getenv("PYTEST_CURRENT_TEST"):
        return {}
    try:
        resp = await client_httpx.get(f"{LLAMA_SERVER_URL}/models")
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            for model in data:
                if model.get("id") == backend_model:
                    args = model.get("status", {}).get("args", [])
                    try:
                        port_idx = args.index("--port")
                        port = args[port_idx + 1]
                    except (ValueError, IndexError):
                        continue

                    # Run docker exec to query the child's /props directly inside the container
                    proc = await asyncio.create_subprocess_exec(
                        "docker",
                        "exec",
                        "llama-server",
                        "curl",
                        "-s",
                        f"http://127.0.0.1:{port}/props",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await proc.communicate()
                    if proc.returncode == 0:
                        return json.loads(stdout.decode().strip())
    except Exception as e:
        logger.warning(f"Failed to get child model props: {e}")
    return {}


def get_model_preset_info(backend_model: str) -> str:
    ini_path = os.path.join(ROUTER_MODELS_DIR, "models.ini")
    if not os.path.exists(ini_path):
        return ""
    import configparser

    try:
        config = configparser.ConfigParser()
        config.read(ini_path)
        if config.has_section(backend_model):
            sec = config[backend_model]
            ctx = sec.get("ctx-size", "unknown")
            cache_k = sec.get("cache-type-k", "unknown")
            cache_v = sec.get("cache-type-v", "unknown")
            fa = sec.get("flash-attn", "unknown")
            return f" (n_ctx={ctx}, cache={cache_k}/{cache_v}, flash_attn={fa})"
    except Exception as e:
        logger.warning(f"Failed to read preset info for {backend_model}: {e}")
    return ""


async def _ensure_model_skip_swap(model_name: str):
    """Ensure model during mid-stream crash recovery without swapping models.

    Used when a model crashes mid-stream and we need to fix the server without
    unloading the user's currently-loaded model. The crashed model will autoload
    naturally when its keep_alive triggers or when a new request comes in.
    """
    await ensure_model(model_name, skip_swap=True)


async def ensure_model(model_name: str, options: dict = None, skip_swap: bool = False):
    model_name = with_default_tag(model_name)
    begin_model_request(model_name)

    # Lock-free check for loaded/healthy or loading models to prevent lock starvation
    try:
        resolved = await resolve_router_model(model_name, reload=True)
        backend_model = resolved["backend_model"]
        entry = resolved["entry"]
        status = router_entry_status(entry)

        if status == "loaded":
            if await is_child_model_healthy(backend_model):
                mark_model_loaded(model_name)
                await record_model_loaded(model_name)
                return {
                    "model_name": model_name,
                    "backend_model": backend_model,
                    "manifest_path": resolved["manifest_path"],
                    "manifest": resolved["manifest"],
                }

        if status == "loading":
            logger.info(
                f"Model {backend_model} is currently loading. Waiting for load to finish..."
            )
            for _ in range(120):
                await asyncio.sleep(1.0)
                try:
                    resolved = await resolve_router_model(model_name, reload=True)
                    backend_model = resolved["backend_model"]
                    entry = resolved["entry"]
                    if router_entry_status(entry) == "loaded":
                        if await is_child_model_healthy(backend_model):
                            logger.info(f"Model {backend_model} finished loading successfully.")
                            mark_model_loaded(model_name)
                            await record_model_loaded(model_name)
                            return {
                                "model_name": model_name,
                                "backend_model": backend_model,
                                "manifest_path": resolved["manifest_path"],
                                "manifest": resolved["manifest"],
                            }
                except Exception as poll_exc:
                    logger.warning(f"Error polling model loading status: {poll_exc}")
    except Exception as e:
        logger.warning(f"Lock-free status check failed: {e}")

    async with router_model_lock:
        mark_model_loading(model_name)
        resolved = await resolve_router_model(model_name, reload=True)
        backend_model = resolved["backend_model"]

        async with active_requests_lock:
            active_requests[backend_model] = active_requests.get(backend_model, 0) + 1
        try:
            return await _ensure_model_impl(model_name, options, resolved, skip_swap)
        finally:
            async with active_requests_lock:
                active_requests[backend_model] = max(0, active_requests.get(backend_model, 0) - 1)
                active_requests_lock.notify_all()


async def _ensure_model_impl(
    model_name: str, options: dict = None, resolved: dict = None, skip_swap: bool = False
):
    if skip_swap:
        # During mid-stream crash recovery, don't try to load the crashed model.
        # Another model may be loaded (user's current session) and MAX_LOADED_MODELS
        # may prevent loading the crashed model anyway. Just ensure the server is
        # running — the crashed model will autoload naturally when needed.
        logger.info(
            f"Mid-stream crash recovery: ensuring llama-server is running "
            f"(model {public_model_name(model_name)} will autoload naturally)."
        )
        await wait_for_llama_server_or_restart(timeout=60.0)
        backend = resolved["backend_model"] if resolved else model_name
        return {
            "model_name": model_name,
            "backend_model": backend,
            "manifest_path": None,
            "manifest": None,
        }
    if resolved is None:
        resolved = await resolve_router_model(model_name, reload=True)
    backend_model = resolved["backend_model"]
    router_models = resolved["router_models"]
    entry = resolved["entry"]

    # Read n-gpu-layers preset from models.ini if available, otherwise default to -1
    n_gpu_layers_preset = -1
    ini_path = os.path.join(ROUTER_MODELS_DIR, "models.ini")
    if os.path.exists(ini_path):
        try:
            import configparser

            config = configparser.ConfigParser()
            config.read(ini_path)
            if config.has_section(backend_model):
                sec = config[backend_model]
                if "n-gpu-layers" in sec:
                    n_gpu_layers_preset = int(sec["n-gpu-layers"])
        except Exception as e:
            logger.warning(f"Failed to read n-gpu-layers preset from models.ini: {e}")

    if MAX_LOADED_MODELS == 1 and not skip_swap:
        for other in router_models:
            other_id = other.get("id")
            other_status = router_entry_status(other)
            if other_id != backend_model and other_status == "loaded":
                # Skip active request wait for crashed models — they're dead and
                # need to be force-unloaded immediately (not wait forever)
                is_crashed = crashed_models.get(other_id, False)

                if not is_crashed:
                    # Wait for active requests on the model we are about to unload
                    async with active_requests_lock:
                        deadline = asyncio.get_event_loop().time() + 180.0  # 180s max wait
                        while active_requests.get(other_id, 0) > 0:
                            remaining = deadline - asyncio.get_event_loop().time()
                            if remaining <= 0:
                                logger.warning(
                                    f"Timeout waiting for {active_requests[other_id]} active requests on {other_id} to finish. Forcing unload."
                                )
                                break
                            logger.info(
                                f"Waiting for {active_requests[other_id]} active requests on {other_id} to finish before unloading ({remaining:.0f}s remaining)."
                            )
                            try:
                                await asyncio.wait_for(
                                    active_requests_lock.wait(), timeout=remaining
                                )
                            except asyncio.TimeoutError:
                                if active_requests.get(other_id, 0) > 0:
                                    logger.warning(
                                        f"Timeout waiting for {active_requests[other_id]} active requests on {other_id} to finish. Forcing unload."
                                    )
                                break

                logger.info(f"Unloading backend model {other_id} before loading {backend_model}")
                try:
                    await post_router_model_action("unload", other_id)
                    await record_model_unloaded_by_backend_id(other_id)
                except RouterManagementUnsupported:
                    logger.warning(
                        "Router unload endpoint unavailable; relying on router autoload behavior."
                    )
                    break
                except httpx.HTTPStatusError as exc:
                    if is_ignorable_router_unload_error(
                        exc
                    ) or not await router_model_is_still_resident(other_id):
                        logger.info(f"Router ignored unload for {other_id}: {exc}")
                        await record_model_unloaded_by_backend_id(other_id)
                        continue
                    raise

    # If already loading, wait transparently for it to finish loading
    status = router_entry_status(entry)
    if status == "loading":
        logger.info(f"Model {backend_model} is currently loading. Waiting for load to finish...")
        for _ in range(120):
            await asyncio.sleep(1.0)
            try:
                resolved = await resolve_router_model(model_name, reload=True)
                entry = resolved["entry"]
                if router_entry_status(entry) == "loaded":
                    logger.info(f"Model {backend_model} finished loading successfully.")
                    status = "loaded"
                    break
            except Exception as poll_exc:
                logger.warning(f"Error polling model loading status: {poll_exc}")
        else:
            logger.warning(
                f"Model {backend_model} did not finish loading after 120s. "
                "Force-unloading and retrying..."
            )
            try:
                await post_router_model_action("unload", backend_model)
            except Exception:
                pass
            await asyncio.sleep(2.0)
            resolved = await resolve_router_model(model_name, reload=True)
            entry = resolved["entry"]
            backend_model = resolved["backend_model"]
            status = router_entry_status(entry)

    # If already loaded, verify health and skip load entirely if healthy
    if status == "loaded":
        if await is_child_model_healthy(backend_model):
            logger.info(
                f"Model {backend_model} already loaded (status=loaded) and active health check passed, proceeding."
            )
            mark_model_loaded(model_name)
            await record_model_loaded(model_name)
            return {
                "model_name": model_name,
                "backend_model": backend_model,
                "manifest_path": resolved["manifest_path"],
                "manifest": resolved["manifest"],
            }
        else:
            logger.warning(
                f"Model {backend_model} is marked as loaded, but active health check failed! Triggering restart and reload recovery..."
            )
            await restart_llama_server()
            if not await wait_for_llama_server_or_restart(timeout=60.0):
                raise HTTPException(
                    status_code=502, detail="Failed to restore llama-server after child crash"
                )
            status = "unloaded"

    # Need to load — attempt load with optimized parameters
    logger.info(f"Loading backend model {backend_model} for {public_model_name(model_name)}")

    # Optimization: Pass n_ctx and other flags to the load call if provided in options
    load_payload = {"model": backend_model}
    if options:
        n_ctx = options.get("num_ctx") or options.get("n_ctx")
        if n_ctx:
            load_payload["n_ctx"] = int(n_ctx)
            logger.info(f"Setting n_ctx={n_ctx} for model load.")

    # Use configured n-gpu-layers preset
    load_payload["n_gpu_layers"] = n_gpu_layers_preset
    load_payload["use_mmap"] = True

    # Resolve GGUF path and read metadata to dynamically determine capabilities without hardcoding
    model_path = None
    if entry and entry.get("path"):
        model_path = entry.get("path")
    elif backend_model:
        bm_filename = (
            backend_model if backend_model.endswith(".gguf") else (backend_model + ".gguf")
        )
        model_path = os.path.join(ROUTER_MODELS_DIR, bm_filename)

    meta = None
    if model_path and os.path.exists(model_path):
        try:
            meta = _read_gguf_metadata(model_path)
        except Exception as e:
            logger.warning(f"Failed to read GGUF metadata at {model_path}: {e}")

    if meta:
        is_moe = _is_moe(meta)
        flash_attn_supported = _supports_flash_attn(meta)
        load_payload["flash_attn"] = flash_attn_supported

        if is_moe and backend_model not in MTP_INCOMPATIBLE_MODELS:
            load_payload["spec_type"] = "draft-mtp"
            load_payload["spec_draft_n_max"] = 3
            logger.info(
                f"Detected MoE model {backend_model} supporting MTP. Enabling speculative decoding."
            )
        else:
            load_payload["spec_type"] = "none"
            load_payload["spec_draft_n_max"] = 0
            if is_moe:
                logger.info(
                    f"Model {backend_model} is marked as MTP incompatible. Disabling speculative decoding."
                )
            else:
                logger.info(
                    f"Detected dense model {backend_model}. Disabling speculative decoding."
                )
    else:
        # Fallback when GGUF file is not found (e.g. in unit tests)
        load_payload["flash_attn"] = True
        if backend_model in MTP_INCOMPATIBLE_MODELS:
            load_payload["spec_type"] = "none"
            load_payload["spec_draft_n_max"] = 0
            logger.info(
                f"Model {backend_model} is marked as MTP incompatible. Disabling speculative decoding."
            )

    if backend_model in SAFE_SETTINGS_MODELS:
        load_payload["flash_attn"] = False
        load_payload["n_ctx"] = 8192
        logger.info(
            f"Model {backend_model} is marked for safe settings. Disabling flash attention, capping n_ctx to 8192."
        )

    # Cap concurrent inference slots to 2 for large models to prevent multiple
    # simultaneous full-context prefills from exhausting host DRAM. Small models
    # can use auto (typically 4) since their KV cache footprint is minimal.
    if is_model_over_9b(model_name, resolved.get("manifest")):
        load_payload["n_parallel"] = 2
        logger.info(
            f"Model {backend_model} is >9B — capping n_parallel=2 to prevent concurrent prefill OOM while --kv-unified handles unified dynamic context."
        )

    # Proactive VRAM budgeting for dense models: if the model won't fit in GPU
    # VRAM with the current preset, write a safe n_gpu_layers to models.ini and
    # restart llama-server BEFORE attempting the load. This avoids the OOM crash
    # + recovery cycle entirely. Metadata-driven (no hardcoded model names).
    is_dense = meta is not None and not _is_moe(meta)
    if is_dense and model_path and os.path.exists(model_path):
        available_vram = await _get_available_vram_mib()
        if available_vram is not None:
            eff_ctx = int(
                load_payload.get("n_ctx")
                or _read_ini_model_setting(backend_model, "ctx-size", "32768")
            )
            eff_cache = _read_ini_model_setting(backend_model, "cache-type-k", "f16")
            safe_ngl = _compute_safe_n_gpu_layers(
                model_path, meta, eff_ctx, eff_cache, 2, available_vram, n_gpu_layers_preset
            )
            if safe_ngl != n_gpu_layers_preset:
                _write_ini_model_setting(backend_model, "n-gpu-layers", str(safe_ngl))
                logger.info("Restarting llama-server to pick up VRAM-safe n_gpu_layers preset...")
                await restart_llama_server()
                if not await wait_for_llama_server_or_restart(timeout=60.0):
                    raise HTTPException(
                        status_code=502, detail="Failed to restart llama-server for VRAM budgeting"
                    )
                n_gpu_layers_preset = safe_ngl

    try:
        await post_router_model_action("load", load_payload)
    except RouterManagementUnsupported:
        logger.warning("Router load endpoint unavailable; relying on request-time model autoload.")
        await record_model_loaded(model_name)
        return {
            "model_name": model_name,
            "backend_model": backend_model,
            "manifest_path": resolved["manifest_path"],
            "manifest": resolved["manifest"],
        }
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        # Skip changing settings for MoE models, but allow for dense models
        is_moe = is_model_moe(backend_model, meta) or is_model_over_9b(
            model_name, resolved.get("manifest") if resolved else None
        )
        if is_moe:
            logger.warning(
                f"Failed to load MoE model {backend_model} "
                f"with original settings: {exc}. Performing clean "
                "restart of llama-server and retrying..."
            )
            await restart_llama_server()
            if await wait_for_llama_server_or_restart(timeout=60.0):
                # Re-resolve and retry load once with original settings
                resolved = await resolve_router_model(model_name, reload=True)
                backend_model = resolved["backend_model"]
                entry = resolved["entry"]

                status = router_entry_status(entry)
                if status == "loading":
                    logger.info(
                        f"Model {backend_model} is loading after restart. Waiting for load to finish..."
                    )
                    for _ in range(120):
                        await asyncio.sleep(1.0)
                        try:
                            resolved = await resolve_router_model(model_name, reload=True)
                            entry = resolved["entry"]
                            if router_entry_status(entry) == "loaded":
                                logger.info(f"Model {backend_model} finished loading successfully.")
                                status = "loaded"
                                break
                        except Exception as poll_exc:
                            logger.warning(f"Error polling model loading status: {poll_exc}")
                    else:
                        logger.warning(f"Model {backend_model} did not finish loading after 120s.")

                if status == "loaded":
                    logger.info(f"Model {backend_model} successfully loaded on recovery restart.")
                    mark_model_loaded(model_name)
                    await record_model_loaded(model_name)
                    return {
                        "model_name": model_name,
                        "backend_model": backend_model,
                        "manifest_path": resolved["manifest_path"],
                        "manifest": resolved["manifest"],
                    }

                try:
                    await post_router_model_action("load", load_payload)
                    logger.info(f"Model {backend_model} successfully loaded after clean restart.")
                    mark_model_loaded(model_name)
                    await record_model_loaded(model_name)
                    return {
                        "model_name": model_name,
                        "backend_model": backend_model,
                        "manifest_path": resolved["manifest_path"],
                        "manifest": resolved["manifest"],
                    }
                except Exception as retry_exc:
                    logger.error(f"Failed to load MoE model even after clean restart: {retry_exc}")
                    raise
            raise

        # Tier 1: Check if speculative decoding (MTP) was enabled and caused the crash
        if load_payload.get("spec_type") != "none":
            logger.warning(
                f"Failed to load model {backend_model} with default options: {exc}. "
                "Retrying without speculative decoding (MTP)..."
            )
            MTP_INCOMPATIBLE_MODELS.add(backend_model)
            save_mtp_incompatible_models()

            # Wait for llama-server to recover/restart
            logger.info("Waiting for llama-server to restart...")
            await asyncio.sleep(2.0)
            if await wait_for_llama_server_or_restart(timeout=60.0):
                # Re-resolve router model to get updated status and entry
                resolved = await resolve_router_model(model_name, reload=True)
                backend_model = resolved["backend_model"]

                # Re-construct payload with MTP disabled (applying safe settings if registered)
                load_payload = {"model": backend_model}
                if options:
                    n_ctx = options.get("num_ctx") or options.get("n_ctx")
                    if n_ctx:
                        load_payload["n_ctx"] = int(n_ctx)
                load_payload["n_gpu_layers"] = n_gpu_layers_preset
                load_payload["use_mmap"] = True
                load_payload["flash_attn"] = True
                load_payload["spec_type"] = "none"
                load_payload["spec_draft_n_max"] = 0

                if backend_model in SAFE_SETTINGS_MODELS:
                    load_payload["flash_attn"] = False
                    if "n_ctx" not in load_payload:
                        load_payload["n_ctx"] = 8192

                logger.info(f"Retrying load of {backend_model} with spec_type='none'...")
                try:
                    await post_router_model_action("load", load_payload)
                    logger.info(
                        f"Model {backend_model} loaded successfully after disabling speculative decoding."
                    )
                    mark_model_loaded(model_name)
                    await record_model_loaded(model_name)
                    return {
                        "model_name": model_name,
                        "backend_model": backend_model,
                        "manifest_path": resolved["manifest_path"],
                        "manifest": resolved["manifest"],
                    }
                except (httpx.RequestError, httpx.HTTPStatusError) as retry_exc:
                    # Escalation Tier 2: Failed even without MTP -> Apply Safe Settings!
                    is_moe = is_model_moe(backend_model, meta) or is_model_over_9b(
                        model_name, resolved.get("manifest") if resolved else None
                    )
                    if is_moe:
                        logger.info(
                            f"Model {model_name} is an MoE model. Skipping Safe Settings escalation."
                        )
                        raise
                    if backend_model not in SAFE_SETTINGS_MODELS:
                        logger.warning(
                            f"Failed to load model {backend_model} even without speculative decoding: {retry_exc}. "
                            "Escalating to Safe Settings (disabling flash attention, capping n_ctx to 8192)..."
                        )
                        SAFE_SETTINGS_MODELS.add(backend_model)
                        save_safe_settings_models()

                        # Wait for llama-server to recover/restart
                        logger.info("Waiting for llama-server to restart...")
                        await asyncio.sleep(2.0)
                        if await wait_for_llama_server_or_restart(timeout=60.0):
                            resolved = await resolve_router_model(model_name, reload=True)
                            backend_model = resolved["backend_model"]

                            load_payload = {"model": backend_model}
                            if options:
                                n_ctx = options.get("num_ctx") or options.get("n_ctx")
                                if n_ctx:
                                    load_payload["n_ctx"] = int(n_ctx)
                            load_payload["n_gpu_layers"] = n_gpu_layers_preset
                            load_payload["use_mmap"] = True
                            load_payload["flash_attn"] = False
                            if "n_ctx" not in load_payload:
                                load_payload["n_ctx"] = 8192
                            load_payload["spec_type"] = "none"
                            load_payload["spec_draft_n_max"] = 0

                            logger.info(f"Retrying load of {backend_model} with Safe Settings...")
                            try:
                                await post_router_model_action("load", load_payload)
                                logger.info(
                                    f"Model {backend_model} loaded successfully with Safe Settings."
                                )
                                mark_model_loaded(model_name)
                                await record_model_loaded(model_name)
                                return {
                                    "model_name": model_name,
                                    "backend_model": backend_model,
                                    "manifest_path": resolved["manifest_path"],
                                    "manifest": resolved["manifest"],
                                }
                            except Exception as safe_exc:
                                logger.error(
                                    f"Failed to load model even with Safe Settings: {safe_exc}"
                                )
                                raise
                    raise
            else:
                logger.error("llama-server did not recover/restart in time.")
                raise

        # Tier 2: If we already disabled MTP but load failed, escalate directly to Safe Settings
        elif backend_model not in SAFE_SETTINGS_MODELS:
            is_moe = is_model_moe(backend_model, meta) or is_model_over_9b(
                model_name, resolved.get("manifest") if resolved else None
            )
            if is_moe:
                logger.info(
                    f"Model {model_name} is an MoE model. Skipping Safe Settings escalation."
                )
                raise
            logger.warning(
                f"Failed to load model {backend_model} under spec_type='none': {exc}. "
                "Escalating to Safe Settings (disabling flash attention, capping n_ctx to 8192)..."
            )
            SAFE_SETTINGS_MODELS.add(backend_model)
            save_safe_settings_models()

            # Wait for llama-server to recover/restart
            logger.info("Waiting for llama-server to restart...")
            await asyncio.sleep(2.0)
            if await wait_for_llama_server_or_restart(timeout=60.0):
                resolved = await resolve_router_model(model_name, reload=True)
                backend_model = resolved["backend_model"]

                load_payload = {"model": backend_model}
                if options:
                    n_ctx = options.get("num_ctx") or options.get("n_ctx")
                    if n_ctx:
                        load_payload["n_ctx"] = int(n_ctx)
                load_payload["n_gpu_layers"] = n_gpu_layers_preset
                load_payload["use_mmap"] = True
                load_payload["flash_attn"] = False
                if "n_ctx" not in load_payload:
                    load_payload["n_ctx"] = 8192
                load_payload["spec_type"] = "none"
                load_payload["spec_draft_n_max"] = 0

                logger.info(f"Retrying load of {backend_model} with Safe Settings...")
                try:
                    await post_router_model_action("load", load_payload)
                    logger.info(f"Model {backend_model} loaded successfully with Safe Settings.")
                    mark_model_loaded(model_name)
                    await record_model_loaded(model_name)
                    return {
                        "model_name": model_name,
                        "backend_model": backend_model,
                        "manifest_path": resolved["manifest_path"],
                        "manifest": resolved["manifest"],
                    }
                except Exception as safe_exc:
                    logger.error(f"Failed to load model even with Safe Settings: {safe_exc}")
                    raise

        # Harmful errors: if it's HTTPStatusError 400 "already running", ignore it
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 400:
            try:
                payload = exc.response.json()
                msg = (
                    str(payload.get("error", {}).get("message", ""))
                    if isinstance(payload, dict)
                    else ""
                )
            except Exception:
                msg = exc.response.text
            if "already running" in msg.lower() or "model is already running" in msg.lower():
                logger.info(f"Model {backend_model} already loaded (detected via 400), proceeding.")
                mark_model_loaded(model_name)
                await record_model_loaded(model_name)
                return {
                    "model_name": model_name,
                    "backend_model": backend_model,
                    "manifest_path": resolved["manifest_path"],
                    "manifest": resolved["manifest"],
                }
        raise

    # Load succeeded — check for OOM or startup crash (post-load verification)
    preset_info = get_model_preset_info(backend_model)
    logger.info(
        f"Model {backend_model} loaded successfully{preset_info}. Checking for OOM or startup crash..."
    )
    await asyncio.sleep(1.0)
    oom_detected = False
    crash_detected = False

    # Check health for all models to detect startup crashes
    if not await is_child_model_healthy(backend_model):
        crash_detected = True
        logger.warning(
            f"Crash detected: model {backend_model} is not "
            "running or failed after load. Triggering recovery."
        )
    elif not is_model_moe(backend_model, meta):
        # For <= 9B models, if it's running, we also check if it OOM'ed to CPU (n_gpu_layers=0)
        try:
            if client_httpx is None:
                logger.debug("client_httpx not initialized; skipping OOM check")
                oom_detected = False
            else:
                # Query the child process's props directly inside the llama-server container
                props = await get_child_model_props(backend_model)
                actual_gpu_layers = props.get("n_gpu_layers", -1)
                if actual_gpu_layers == 0:
                    oom_detected = True
                    logger.warning(
                        f"OOM detected: model {backend_model} loaded with n_gpu_layers=0 "
                        f"(expected -1). Triggering OOM recovery."
                    )
        except Exception:
            logger.exception("OOM detection failed")
            oom_detected = False

    if oom_detected or crash_detected:
        try:
            # Mark model as crashed so other models don't try to unload it (prevents deadlock)
            crashed_models[backend_model] = True
            logger.info(
                f"Model {backend_model} marked as crashed. Other models will skip unload wait."
            )
            # Clear active requests for this model — the requests are dead (server crashed)
            # and need to be retried, not waited on forever
            async with active_requests_lock:
                active_requests[backend_model] = 0
                active_requests_lock.notify_all()
                logger.info(f"Cleared active_requests for crashed model {backend_model}")

            logger.info(
                f"Recovery: unloading {backend_model} and "
                "restarting llama-server to release GPU memory..."
            )
            try:
                await post_router_model_action("unload", backend_model)
            except Exception:
                pass

            # Before restarting, write VRAM-safe n_gpu_layers to models.ini.
            # The router only reads n_gpu_layers from the INI at startup, so
            # we must write the corrected value BEFORE the restart for it to
            # take effect. Use metadata-driven VRAM budgeting (no hardcoding).
            is_moe = is_model_moe(backend_model, meta) or is_model_over_9b(
                model_name, resolved.get("manifest") if resolved else None
            )
            if not is_moe and model_path and os.path.exists(model_path) and meta:
                available_vram = await _get_available_vram_mib()
                if available_vram is not None:
                    eff_ctx = int(_read_ini_model_setting(backend_model, "ctx-size", "32768"))
                    eff_cache = _read_ini_model_setting(backend_model, "cache-type-k", "f16")
                    safe_ngl = _compute_safe_n_gpu_layers(
                        model_path, meta, eff_ctx, eff_cache, 2, available_vram, n_gpu_layers_preset
                    )
                    if safe_ngl != n_gpu_layers_preset:
                        _write_ini_model_setting(backend_model, "n-gpu-layers", str(safe_ngl))

            # Incorporate Telemetry & Auto-Tuning suggestions to prevent recurring memory exhaustion.
            # NOTE: n-gpu-layers is intentionally excluded here — it is already computed above
            # using live VRAM state and GGUF metadata, which is more accurate than heuristics.
            _ANALYZER_SKIP_KEYS = {"n-gpu-layers"}
            try:
                from analyzer import analyze_telemetry

                analysis = analyze_telemetry(backend_model, performance_first=True)
                if analysis.get("recommendations"):
                    applied = {}
                    for key, val in analysis["recommendations"].items():
                        if key in _ANALYZER_SKIP_KEYS:
                            logger.debug(
                                f"Auto-tuning: skipping '{key}' (managed by VRAM budget logic)"
                            )
                            continue
                        _write_ini_model_setting(backend_model, key, val)
                        applied[key] = val
                    if applied:
                        logger.info(f"Auto-tuning recovery applied configurations: {applied}")
            except Exception as auto_err:
                logger.warning(f"Auto-tuning engine failed during recovery: {auto_err}")

            # Force restart of llama-server to guarantee GPU memory release
            # and pick up any INI overrides written above
            await restart_llama_server()

            logger.info("Waiting for llama-server to become responsive after restart...")
            if not await wait_for_llama_server_or_restart(timeout=60.0):
                raise RuntimeError("llama-server did not recover after OOM/crash unload.")

            # Re-resolve and retry loading
            resolved = await resolve_router_model(model_name, reload=True)
            backend_model = resolved["backend_model"]

            # Skip changing settings for MoE models, but allow for dense models
            is_moe = is_model_moe(backend_model, meta) or is_model_over_9b(
                model_name, resolved.get("manifest") if resolved else None
            )
            if is_moe:
                logger.info(f"Retrying load of MoE model {backend_model} with original settings...")
                load_payload = {"model": backend_model}
                if options:
                    n_ctx = options.get("num_ctx") or options.get("n_ctx")
                    if n_ctx:
                        load_payload["n_ctx"] = int(n_ctx)
                await post_router_model_action("load", load_payload)
                logger.info(
                    f"Model {backend_model} successfully loaded "
                    "with original settings after clean restart."
                )
            else:
                # Tiered recovery/escalation for <= 9B models
                if oom_detected:
                    # OOM recovery: cap n_gpu_layers=0 immediately
                    logger.warning(
                        f"Model {backend_model} OOM'ed. Falling back to OOM-safe CPU settings (n_gpu_layers=0)..."
                    )
                    load_payload = {"model": backend_model}
                    if options:
                        n_ctx = options.get("num_ctx") or options.get("n_ctx")
                        if n_ctx:
                            load_payload["n_ctx"] = int(n_ctx)
                    load_payload["n_gpu_layers"] = 0
                    load_payload["n_thread"] = 8
                    load_payload["n_batch"] = 256
                    load_payload["n_ubatch"] = 512
                    load_payload["use_mmap"] = True
                    load_payload["flash_attn"] = False
                    load_payload["spec_type"] = "none"
                    load_payload["spec_draft_n_max"] = 0

                    logger.info(f"Retrying load of {backend_model} with OOM-safe settings...")
                    await post_router_model_action("load", load_payload)
                    logger.info(
                        f"Model {backend_model} loaded successfully with OOM-safe settings."
                    )
                else:
                    # crash_detected: Check if speculative decoding (MTP) was active and caused the crash
                    if load_payload.get("spec_type") != "none":
                        logger.warning(
                            f"Model {backend_model} failed load with MTP enabled. "
                            "Retrying without speculative decoding (MTP)..."
                        )
                        MTP_INCOMPATIBLE_MODELS.add(backend_model)
                        save_mtp_incompatible_models()

                        load_payload = {"model": backend_model}
                        if options:
                            n_ctx = options.get("num_ctx") or options.get("n_ctx")
                            if n_ctx:
                                load_payload["n_ctx"] = int(n_ctx)
                        load_payload["n_gpu_layers"] = n_gpu_layers_preset
                        load_payload["use_mmap"] = True
                        load_payload["flash_attn"] = True
                        load_payload["spec_type"] = "none"
                        load_payload["spec_draft_n_max"] = 0

                        if backend_model in SAFE_SETTINGS_MODELS:
                            load_payload["flash_attn"] = False
                            if "n_ctx" not in load_payload:
                                load_payload["n_ctx"] = 8192

                        logger.info(f"Retrying load of {backend_model} with spec_type='none'...")
                        await post_router_model_action("load", load_payload)
                        logger.info(
                            f"Model {backend_model} loaded successfully after disabling speculative decoding."
                        )
                    # If MTP is already disabled, check if we can escalate to Safe Settings
                    elif backend_model not in SAFE_SETTINGS_MODELS:
                        logger.warning(
                            f"Failed to load model {backend_model} even without speculative decoding. "
                            "Escalating to Safe Settings (disabling flash attention, capping n_ctx to 8192)..."
                        )
                        SAFE_SETTINGS_MODELS.add(backend_model)
                        save_safe_settings_models()

                        load_payload = {"model": backend_model}
                        if options:
                            n_ctx = options.get("num_ctx") or options.get("n_ctx")
                            if n_ctx:
                                load_payload["n_ctx"] = int(n_ctx)
                        load_payload["n_gpu_layers"] = n_gpu_layers_preset
                        load_payload["use_mmap"] = True
                        load_payload["flash_attn"] = False
                        if "n_ctx" not in load_payload:
                            load_payload["n_ctx"] = 8192
                        load_payload["spec_type"] = "none"
                        load_payload["spec_draft_n_max"] = 0

                        logger.info(f"Retrying load of {backend_model} with Safe Settings...")
                        await post_router_model_action("load", load_payload)
                        logger.info(
                            f"Model {backend_model} loaded successfully with Safe Settings."
                        )
                    # If already using Safe Settings, fall back to OOM-safe settings (n_gpu_layers=0)
                    else:
                        logger.warning(
                            f"Model {backend_model} failed even with Safe Settings. "
                            "Falling back to OOM-safe CPU settings (n_gpu_layers=0)..."
                        )
                        load_payload = {"model": backend_model}
                        if options:
                            n_ctx = options.get("num_ctx") or options.get("n_ctx")
                            if n_ctx:
                                load_payload["n_ctx"] = int(n_ctx)
                        load_payload["n_gpu_layers"] = 0
                        load_payload["n_thread"] = 8
                        load_payload["n_batch"] = 256
                        load_payload["n_ubatch"] = 512
                        load_payload["use_mmap"] = True
                        load_payload["flash_attn"] = False
                        load_payload["spec_type"] = "none"
                        load_payload["spec_draft_n_max"] = 0

                        logger.info(f"Retrying load of {backend_model} with OOM-safe settings...")
                        await post_router_model_action("load", load_payload)
                        logger.info(
                            f"Model {backend_model} loaded successfully with OOM-safe settings."
                        )
        except Exception as recovery_err:
            logger.error(f"Recovery failed for {model_name}: {recovery_err}. Raising.")
            try:
                failed_config = {
                    "model": backend_model,
                    "cache-type-k": _read_ini_model_setting(backend_model, "cache-type-k", "f16"),
                    "cache-type-v": _read_ini_model_setting(backend_model, "cache-type-v", "f16"),
                    "n-gpu-layers": str(_read_ini_model_setting(backend_model, "n-gpu-layers", "-1")),
                    "ctx-size": str(_read_ini_model_setting(backend_model, "ctx-size", "4096")),
                    "timestamp": time.time()
                }
                failed_configs_file = os.path.join("data", "failed_configs.json")
                failed_list = []
                if os.path.exists(failed_configs_file):
                    try:
                        with open(failed_configs_file, "r") as f:
                            failed_list = json.load(f)
                            if not isinstance(failed_list, list):
                                failed_list = []
                    except Exception:
                        pass
                if not any(f.get("model") == failed_config["model"] and
                           f.get("cache-type-k") == failed_config["cache-type-k"] and
                           f.get("cache-type-v") == failed_config["cache-type-v"] and
                           f.get("n-gpu-layers") == failed_config["n-gpu-layers"] and
                           f.get("ctx-size") == failed_config["ctx-size"] for f in failed_list):
                    failed_list.append(failed_config)
                    failed_list = failed_list[-50:]
                    os.makedirs("data", exist_ok=True)
                    with open(failed_configs_file, "w") as f:
                        json.dump(failed_list, f, indent=2)
                    logger.info(f"Recorded failed model configuration in failed_configs.json: {failed_config}")
            except Exception as rec_err:
                logger.warning(f"Failed to record failed configuration: {rec_err}")
            await raise_model_load_failure_exception(model_name, backend_model, str(recovery_err))
        await record_model_loaded(model_name)
        return {
            "model_name": model_name,
            "backend_model": backend_model,
            "manifest_path": resolved["manifest_path"],
            "manifest": resolved["manifest"],
        }

    # Mark as loaded on successful load
    mark_model_loaded(model_name)
    await record_model_loaded(model_name)
    return {
        "model_name": model_name,
        "backend_model": backend_model,
        "manifest_path": resolved["manifest_path"],
        "manifest": resolved["manifest"],
    }


async def wait_for_slot(timeout: float = 120.0) -> bool:
    """Wait for a llama-server slot to become available.

    Returns True if a slot opened, False on timeout.
    """
    start = asyncio.get_event_loop().time()
    while True:
        slot_info = await get_llama_server_slots()
        if slot_info and slot_info["available"] > 0:
            return True
        elapsed = asyncio.get_event_loop().time() - start
        if elapsed >= timeout:
            logger.warning(f"Timeout waiting for slot after {elapsed:.0f}s")
            return False
        await asyncio.sleep(0.5)


SLOTS_CACHE_DIR = os.getenv("SLOTS_CACHE_DIR", "/slots-cache")


def get_prefix_hash(model_name: str, messages: list, options: dict = None) -> str:
    import hashlib

    if not messages:
        return ""
    payload = {
        "model": model_name,
        "messages": messages,
    }
    if options:
        payload["options"] = options
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


async def find_idle_slot(backend_model: str) -> int:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(
                f"{LLAMA_SERVER_URL}/slots", params={"model": backend_model}, timeout=3.0
            )
            if resp.status_code == 200:
                slots = resp.json()
                for slot in slots:
                    if not slot.get("is_processing", False):
                        return slot.get("id") or slot.get("id_slot", 0)
    except Exception as e:
        logger.warning(f"Failed to find idle slot: {e}")
    return 0


async def restore_slot_cache(backend_model: str, prefix_hash: str) -> bool:
    if not prefix_hash:
        return False
    filename = f"conv_{prefix_hash}.bin"
    filepath = os.path.join(SLOTS_CACHE_DIR, filename)
    if not os.path.exists(filepath):
        return False

    try:
        target_slot_id = await find_idle_slot(backend_model)
        logger.info(f"Restoring KV Cache checkpoint {filename} into slot {target_slot_id}...")
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{LLAMA_SERVER_URL}/slots/{target_slot_id}",
                params={"action": "restore"},
                json={"filename": filename},
                timeout=10.0,
            )
            if resp.status_code == 200:
                logger.info(f"Successfully restored KV Cache {filename} into slot {target_slot_id}")
                return True
            else:
                logger.warning(
                    f"Failed to restore slot {target_slot_id}: status {resp.status_code} {resp.text}"
                )
    except Exception as e:
        logger.warning(f"Error during slot restore: {e}")
    return False


async def find_slot_for_request(backend_model: str, messages: list) -> int:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(
                f"{LLAMA_SERVER_URL}/slots", params={"model": backend_model}, timeout=3.0
            )
            if resp.status_code == 200:
                slots = resp.json()
                best_slot_id = None
                max_matches = -1

                check_texts = [m.get("content", "") for m in messages if m.get("content")]
                check_texts = check_texts[-3:]

                for slot in slots:
                    slot_id = slot.get("id") or slot.get("id_slot", 0)
                    slot_prompt = slot.get("prompt", "")
                    if not slot_prompt:
                        continue

                    matches = 0
                    for text in check_texts:
                        if text in slot_prompt:
                            matches += 1

                    if matches > max_matches and matches > 0:
                        max_matches = matches
                        best_slot_id = slot_id

                if best_slot_id is not None:
                    return best_slot_id
    except Exception as e:
        logger.warning(f"Error finding slot for request: {e}")
    return 0


async def save_slot_cache(backend_model: str, new_prefix_hash: str, slot_id: int) -> bool:
    if not new_prefix_hash:
        return False
    filename = f"conv_{new_prefix_hash}.bin"
    try:
        os.makedirs(SLOTS_CACHE_DIR, exist_ok=True)
        logger.info(f"Saving KV Cache checkpoint {filename} from slot {slot_id}...")
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{LLAMA_SERVER_URL}/slots/{slot_id}",
                params={"action": "save"},
                json={"filename": filename},
                timeout=10.0,
            )
            if resp.status_code == 200:
                logger.info(f"Successfully saved KV Cache {filename} from slot {slot_id}")
                prune_slots_cache()
                return True
            else:
                logger.warning(
                    f"Failed to save slot {slot_id}: status {resp.status_code} {resp.text}"
                )
    except Exception as e:
        logger.warning(f"Error during slot save: {e}")
    return False


def prune_slots_cache(max_files: int = 15):
    try:
        if not os.path.exists(SLOTS_CACHE_DIR):
            return
        files = [
            os.path.join(SLOTS_CACHE_DIR, f)
            for f in os.listdir(SLOTS_CACHE_DIR)
            if f.startswith("conv_") and f.endswith(".bin")
        ]
        if len(files) <= max_files:
            return
        files.sort(key=os.path.getmtime)
        to_delete = files[: len(files) - max_files]
        for f in to_delete:
            try:
                os.remove(f)
                logger.info(f"Deleted old slot cache file: {f}")
            except Exception as e:
                logger.warning(f"Failed to delete {f}: {e}")
    except Exception as e:
        logger.warning(f"Error pruning slots cache: {e}")


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    model_name = body.get("model")
    keep_alive = effective_keep_alive(body.get("keep_alive"))

    request_id = getattr(request.state, "request_id", None)
    if not request_id or request_id == "N/A":
        import uuid

        request_id = str(uuid.uuid4())[:8]
    register_active_request(
        request_id,
        model_name,
        "ollama_chat",
        body,
        request_source=getattr(request.state, "request_source", "unknown"),
        client_ip=(request.client.host if request.client else "unknown")
    )

    # Ensure model is loaded and healthy (triggers recovery if crashed/dead)
    resolved = await ensure_model(model_name, options=body.get("options"))
    resolved_backend = resolved["backend_model"]

    # Restore slot cache if available (Turn K prefix is up to Turn K-1 assistant response)
    prefix_hash = None
    messages = body.get("messages", [])
    if len(messages) > 1:
        prefix_hash = get_prefix_hash(model_name, messages[:-1], options=body.get("options"))
        await restore_slot_cache(resolved_backend, prefix_hash)

    # Queue-and-wait: if all llama-server slots are busy, wait for one to open
    wait_timeout = body.get("queue_timeout", 120.0)
    slot_available = await wait_for_slot(timeout=wait_timeout)
    if not slot_available:
        return JSONResponse(
            {"error": "No llama-server slots available within timeout", "status": "queue_timeout"},
            status_code=503,
        )

    async def stream_proxy():
        resolved_backend = None
        try:
            started_ns = now_ns()
            load_started_ns = now_ns()
            resolved = await ensure_model(model_name, options=body.get("options"))
            resolved_backend = resolved["backend_model"]

            # Increment reference count
            async with active_requests_lock:
                active_requests[resolved_backend] = active_requests.get(resolved_backend, 0) + 1
                logger.info(
                    f"In-flight request started for {resolved_backend}. Active: {active_requests[resolved_backend]}"
                )

            load_duration = now_ns() - load_started_ns

            think_val = body.get("think")
            if think_val is None:
                think_val = body.get("enable_thinking")
            if think_val is None and isinstance(body.get("options"), dict):
                opts = body["options"]
                think_val = opts.get("think") if opts.get("think") is not None else opts.get("enable_thinking")

            full_response_content = ""
            current_payload = build_chat_payload(body, resolved_backend)
            attempt = 0
            max_attempts = 2

            while attempt < max_attempts:
                try:
                    request_id = getattr(request.state, "request_id", "N/A")
                    request_source = getattr(request.state, "request_source", "unknown")
                    logger.info(
                        f"[CHAT] Sending payload to llama-server (attempt {attempt}): model={current_payload.get('model')}, stream=True | "
                        f"Origin: {request_source}",
                        extra={"request_id": request_id, "request_source": request_source},
                    )

                    async with client_httpx.stream(
                        "POST", f"{LLAMA_SERVER_URL}/v1/chat/completions", json=current_payload
                    ) as resp:
                        resp.raise_for_status()
                        in_thinking = False
                        async for line in resp.aiter_lines():
                            if not line or "[DONE]" in line:
                                continue
                            if line.startswith("data: "):
                                line = line[6:]
                            try:
                                data = json.loads(line)
                                choice = (data.get("choices") or [{}])[0]
                                message = chat_message_from_choice(choice)
                                if message:
                                    content_chunk = message.get("content") or ""
                                    thinking_chunk = message.get("thinking") or ""
                                    
                                    # Blend/wrap or separate based on think_val
                                    out_content = ""
                                    if think_val is None:
                                        # Default: blend/wrap thinking inside content field
                                        if thinking_chunk:
                                            if not in_thinking:
                                                out_content = "<think>\n" + thinking_chunk
                                                in_thinking = True
                                            else:
                                                out_content = thinking_chunk
                                        elif content_chunk:
                                            if in_thinking:
                                                out_content = "</think>\n" + content_chunk
                                                in_thinking = False
                                            else:
                                                out_content = content_chunk
                                    else:
                                        # Explicit True or False
                                        if in_thinking:
                                            in_thinking = False
                                        out_content = content_chunk
                                            
                                    if out_content:
                                        full_response_content += out_content
                                        
                                    update_active_request_progress(
                                        request_id,
                                        response_chunk=content_chunk,
                                        thinking_chunk=thinking_chunk,
                                    )
                                    
                                    client_message = {
                                        "role": message.get("role", "assistant"),
                                        "content": out_content
                                    }
                                    if thinking_chunk and think_val is True:
                                        client_message["thinking"] = thinking_chunk
                                    elif thinking_chunk and think_val is None:
                                        client_message["thinking"] = thinking_chunk
                                else:
                                    client_message = {"role": "assistant", "content": ""}
                                    
                                done = choice.get("finish_reason") is not None
                                if done and in_thinking and think_val is None:
                                    client_message["content"] = (client_message.get("content") or "") + "\n</think>"
                                    in_thinking = False
                                    
                                chunk = ollama_chat_chunk(
                                    model_name, client_message, done, choice.get("finish_reason")
                                )
                                if done:
                                    apply_metrics(chunk, data, now_ns() - started_ns, load_duration)
                                yield json.dumps(chunk) + "\n"
                            except Exception as e:
                                logger.exception(f"Error processing chat stream line: {e}")
                                continue
                    break
                except (
                    httpx.RemoteProtocolError,
                    httpx.ReadError,
                    httpx.HTTPStatusError,
                    httpx.RequestError,
                ) as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(f"Stream failed after {attempt} attempts: {e}")
                        raise

                    logger.warning(
                        f"Upstream stream error (attempt {attempt}): {e}. Attempting seamless recovery..."
                    )
                    await restart_llama_server()
                    if not await wait_for_llama_server_or_restart(timeout=60.0):
                        raise HTTPException(
                            status_code=502,
                            detail="Failed to restore llama-server after mid-stream crash",
                        )

                    resolved = await ensure_model(model_name, options=body.get("options"))
                    resolved_backend = resolved["backend_model"]
                    if prefix_hash:
                        await restore_slot_cache(resolved_backend, prefix_hash)

                    if full_response_content:
                        current_payload["messages"] = body.get("messages", []) + [
                            {"role": "assistant", "content": full_response_content}
                        ]
                    else:
                        current_payload["messages"] = body.get("messages", [])

            # Save the new slot cache checkpoint
            if full_response_content:
                new_messages = list(messages) + [
                    {"role": "assistant", "content": full_response_content}
                ]
                new_prefix_hash = get_prefix_hash(
                    model_name, new_messages, options=body.get("options")
                )
                slot_id = await find_slot_for_request(resolved_backend, new_messages)
                await save_slot_cache(resolved_backend, new_prefix_hash, slot_id)

            await apply_keep_alive_policy(model_name, keep_alive)
        except httpx.HTTPStatusError as e:
            try:
                await e.response.aread()
                body_text = e.response.text
            except Exception:
                body_text = "<unreadable>"
            error_msg = f"Upstream error {e.response.status_code}: {body_text}"
            logger.error(error_msg)
            yield json.dumps({"error": error_msg}) + "\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield json.dumps({"error": str(e)}) + "\n"
        finally:
            if resolved_backend:
                async with active_requests_lock:
                    active_requests[resolved_backend] = max(
                        0, active_requests.get(resolved_backend, 0) - 1
                    )
                    logger.info(
                        f"In-flight request finished for {resolved_backend}. Active: {active_requests[resolved_backend]}"
                    )
                    active_requests_lock.notify_all()
            complete_active_request(request_id)

    if should_stream(body):
        return StreamingResponse(stream_proxy(), media_type="application/x-ndjson")

    resolved_backend = None
    try:
        started_ns = now_ns()
        load_started_ns = now_ns()
        resolved = await ensure_model(model_name, options=body.get("options"))
        resolved_backend = resolved["backend_model"]

        async with active_requests_lock:
            active_requests[resolved_backend] = active_requests.get(resolved_backend, 0) + 1
            logger.info(
                f"In-flight request started for {resolved_backend}. Active: {active_requests[resolved_backend]}"
            )

        load_duration = now_ns() - load_started_ns
        payload = build_chat_payload(body, resolved_backend)

        attempt = 0
        max_attempts = 2
        resp = None
        while attempt < max_attempts:
            try:
                request_id = getattr(request.state, "request_id", "N/A")
                request_source = getattr(request.state, "request_source", "unknown")
                logger.info(
                    f"[CHAT] Sending payload to llama-server (attempt {attempt}): model={payload.get('model')}, stream=False | "
                    f"Origin: {request_source}",
                    extra={"request_id": request_id, "request_source": request_source},
                )
                resp = await client_httpx.post(
                    f"{LLAMA_SERVER_URL}/v1/chat/completions", json=payload
                )
                resp.raise_for_status()
                break
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                attempt += 1
                if attempt >= max_attempts:
                    raise
                logger.warning(
                    f"Upstream request failed (attempt {attempt}): {e}. Retrying recovery..."
                )
                await restart_llama_server()
                await wait_for_llama_server_or_restart()
                resolved = await ensure_model(model_name, options=body.get("options"))
                resolved_backend = resolved["backend_model"]
                if prefix_hash:
                    await restore_slot_cache(resolved_backend, prefix_hash)

        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        message = chat_message_from_choice(choice)
        
        final_content = message.get("content") if message else ""
        final_thinking = message.get("thinking") if message else None
        
        # Determine think_val
        think_val = body.get("think")
        if think_val is None:
            think_val = body.get("enable_thinking")
        if think_val is None and isinstance(body.get("options"), dict):
            opts = body["options"]
            think_val = opts.get("think") if opts.get("think") is not None else opts.get("enable_thinking")

        client_message = {
            "role": message.get("role", "assistant") if message else "assistant",
            "content": final_content
        }
        if final_thinking:
            if think_val is None:
                # Default: blend/wrap
                if "<think>" not in final_content:
                    client_message["content"] = f"<think>\n{final_thinking}\n</think>\n{final_content}"
                client_message["thinking"] = final_thinking
            elif think_val is True:
                # Explicit True: separate
                client_message["thinking"] = final_thinking
            else:
                # Explicit False: strip/filter out thinking entirely
                pass
                
        chunk = ollama_chat_chunk(model_name, client_message, True, choice.get("finish_reason"))
        apply_metrics(chunk, data, now_ns() - started_ns, load_duration)
        logprobs = logprobs_from_choice(choice)
        if logprobs is not None:
            chunk["logprobs"] = logprobs

        # Complete request with final content and thinking
        prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
        gen_tokens = data.get("usage", {}).get("completion_tokens", 0)
        complete_active_request(
            request_id, 
            final_response=final_content, 
            final_thinking=final_thinking,
            prompt_tokens=prompt_tokens,
            completion_tokens=gen_tokens
        )

        # Save the new slot cache checkpoint
        if client_message and client_message.get("content"):
            new_messages = list(messages) + [client_message]
            new_prefix_hash = get_prefix_hash(model_name, new_messages, options=body.get("options"))
            slot_id = await find_slot_for_request(resolved_backend, new_messages)
            await save_slot_cache(resolved_backend, new_prefix_hash, slot_id)

        await apply_keep_alive_policy(model_name, keep_alive)
        return JSONResponse(chunk)
    except httpx.HTTPStatusError as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.TimeoutException as e:
        logger.error(f"Chat upstream timeout: {e}")
        raise HTTPException(status_code=504, detail="Upstream llama-server timed out")
    except httpx.RequestError as e:
        logger.error(f"Chat upstream request error: {e}")
        raise HTTPException(status_code=502, detail=f"Upstream llama-server request failed: {e}")
    finally:
        if resolved_backend:
            async with active_requests_lock:
                active_requests[resolved_backend] = max(
                    0, active_requests.get(resolved_backend, 0) - 1
                )
                logger.info(
                    f"In-flight request finished for {resolved_backend}. Active: {active_requests[resolved_backend]}"
                )
                active_requests_lock.notify_all()
        complete_active_request(request_id)


@app.post("/api/generate")
async def generate(request: Request):
    body = await request.json()
    model_name = body.get("model")
    use_chat_backend = should_generate_via_chat(body)
    endpoint = "/v1/chat/completions" if use_chat_backend else "/completion"
    keep_alive = effective_keep_alive(body.get("keep_alive"))

    request_id = getattr(request.state, "request_id", None)
    if not request_id or request_id == "N/A":
        import uuid

        request_id = str(uuid.uuid4())[:8]
    register_active_request(
        request_id,
        model_name,
        "ollama_generate",
        body,
        request_source=getattr(request.state, "request_source", "unknown"),
        client_ip=(request.client.host if request.client else "unknown")
    )

    async def stream_proxy():
        resolved_backend = None
        try:
            started_ns = now_ns()
            load_started_ns = now_ns()
            resolved = await ensure_model(model_name, options=body.get("options"))
            resolved_backend = resolved["backend_model"]

            # Increment reference count
            async with active_requests_lock:
                active_requests[resolved_backend] = active_requests.get(resolved_backend, 0) + 1
                logger.info(
                    f"In-flight request started for {resolved_backend}. Active: {active_requests[resolved_backend]}"
                )

            load_duration = now_ns() - load_started_ns
            
            think_val = body.get("think")
            if think_val is None:
                think_val = body.get("enable_thinking")
            if think_val is None and isinstance(body.get("options"), dict):
                opts = body["options"]
                think_val = opts.get("think") if opts.get("think") is not None else opts.get("enable_thinking")

            full_response_content = ""
            current_payload = (
                build_generate_chat_payload(body, resolved_backend)
                if use_chat_backend
                else build_generate_payload(body, resolved_backend)
            )

            attempt = 0
            max_attempts = 2

            while attempt < max_attempts:
                try:
                    request_id = getattr(request.state, "request_id", "N/A")
                    request_source = getattr(request.state, "request_source", "unknown")
                    logger.info(
                        f"[GENERATE] Sending payload to llama-server (attempt {attempt}): model={current_payload.get('model')} | "
                        f"Origin: {request_source}",
                        extra={"request_id": request_id, "request_source": request_source},
                    )

                    async with client_httpx.stream(
                        "POST", f"{LLAMA_SERVER_URL}{endpoint}", json=current_payload
                    ) as resp:
                        resp.raise_for_status()
                        in_thinking = False
                        async for line in resp.aiter_lines():
                            if not line or "[DONE]" in line:
                                continue
                            if line.startswith("data: "):
                                line = line[6:]
                            try:
                                data = json.loads(line)
                                response_chunk = ""
                                thinking_chunk = None
                                if use_chat_backend:
                                    choice = (data.get("choices") or [{}])[0]
                                    message = chat_message_from_choice(choice)
                                    response_chunk = message.get("content", "") if message else ""
                                    thinking_chunk = message.get("thinking") if message else None
                                    done = choice.get("finish_reason") is not None
                                    
                                    # Blend/wrap or separate based on think_val
                                    out_content = ""
                                    if think_val is None:
                                        # Default: blend/wrap thinking inside response/content field
                                        if thinking_chunk:
                                            if not in_thinking:
                                                out_content = "<think>\n" + thinking_chunk
                                                in_thinking = True
                                            else:
                                                out_content = thinking_chunk
                                        elif response_chunk:
                                            if in_thinking:
                                                out_content = "</think>\n" + response_chunk
                                                in_thinking = False
                                            else:
                                                out_content = response_chunk
                                        if done and in_thinking:
                                            out_content += "\n</think>"
                                            in_thinking = False
                                    else:
                                        # Explicit True or False
                                        if in_thinking:
                                            in_thinking = False
                                        out_content = response_chunk
                                        
                                    chunk = ollama_generate_chunk(
                                        model_name,
                                        out_content,
                                        done,
                                        choice.get("finish_reason"),
                                    )
                                    if thinking_chunk is not None and think_val is True:
                                        chunk["thinking"] = thinking_chunk
                                    if done:
                                        apply_metrics(
                                            chunk, data, now_ns() - started_ns, load_duration
                                        )
                                else:
                                    done = data.get("stop", False)
                                    done_reason = data.get("finish_reason") or (
                                        "stop" if done else None
                                    )
                                    response_chunk = data.get("content") or ""
                                    thinking_chunk = data.get("thinking")
                                    
                                    # Blend/wrap or separate based on think_val
                                    out_content = ""
                                    if think_val is None:
                                        # Default: blend/wrap thinking inside response/content field
                                        if thinking_chunk:
                                            if not in_thinking:
                                                out_content = "<think>\n" + thinking_chunk
                                                in_thinking = True
                                            else:
                                                out_content = thinking_chunk
                                        elif response_chunk:
                                            if in_thinking:
                                                out_content = "</think>\n" + response_chunk
                                                in_thinking = False
                                            else:
                                                out_content = response_chunk
                                        if done and in_thinking:
                                            out_content += "\n</think>"
                                            in_thinking = False
                                    else:
                                        # Explicit True or False
                                        if in_thinking:
                                            in_thinking = False
                                        out_content = response_chunk
                                        
                                    chunk = ollama_generate_chunk(
                                        model_name, out_content, done, done_reason
                                    )
                                    if thinking_chunk is not None and think_val is True:
                                        chunk["thinking"] = thinking_chunk
                                    if done:
                                        apply_metrics(
                                            chunk, data, now_ns() - started_ns, load_duration
                                        )
                                if out_content:
                                    full_response_content += out_content
                                update_active_request_progress(
                                    request_id,
                                    response_chunk=response_chunk,
                                    thinking_chunk=thinking_chunk,
                                )
                                yield json.dumps(chunk) + "\n"
                            except Exception as e:
                                logger.exception(f"Error processing generate stream line: {e}")
                                continue
                    break
                except (
                    httpx.RemoteProtocolError,
                    httpx.ReadError,
                    httpx.HTTPStatusError,
                    httpx.RequestError,
                ) as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(f"Stream failed after {attempt} attempts: {e}")
                        raise

                    logger.warning(
                        f"Upstream stream error (attempt {attempt}): {e}. Attempting seamless recovery..."
                    )
                    await restart_llama_server()
                    if not await wait_for_llama_server_or_restart(timeout=60.0):
                        raise HTTPException(
                            status_code=502,
                            detail="Failed to restore llama-server after mid-stream crash",
                        )

                    resolved = await ensure_model(model_name, options=body.get("options"))
                    resolved_backend = resolved["backend_model"]

                    if use_chat_backend:
                        if full_response_content:
                            current_payload["messages"] = build_generate_chat_payload(
                                body, resolved_backend
                            )["messages"] + [
                                {"role": "assistant", "content": full_response_content}
                            ]
                        else:
                            current_payload["messages"] = build_generate_chat_payload(
                                body, resolved_backend
                            )["messages"]
                    else:
                        if full_response_content:
                            current_payload["prompt"] = (
                                build_generate_payload(body, resolved_backend)["prompt"]
                                + full_response_content
                            )
                        else:
                            current_payload["prompt"] = build_generate_payload(
                                body, resolved_backend
                            )["prompt"]

            await apply_keep_alive_policy(model_name, keep_alive)
        except httpx.HTTPStatusError as e:
            try:
                await e.response.aread()
                body_text = e.response.text
            except Exception:
                body_text = "<unreadable>"
            error_msg = f"Upstream error {e.response.status_code}: {body_text}"
            logger.error(error_msg)
            yield json.dumps({"error": error_msg}) + "\n"
        except Exception as e:
            logger.error(f"Generate stream error: {e}")
            yield json.dumps({"error": str(e)}) + "\n"
        finally:
            if resolved_backend:
                async with active_requests_lock:
                    active_requests[resolved_backend] = max(
                        0, active_requests.get(resolved_backend, 0) - 1
                    )
                    logger.info(
                        f"In-flight request finished for {resolved_backend}. Active: {active_requests[resolved_backend]}"
                    )
                    active_requests_lock.notify_all()
            complete_active_request(request_id)

    if should_stream(body):
        return StreamingResponse(stream_proxy(), media_type="application/x-ndjson")

    resolved_backend = None
    try:
        started_ns = now_ns()
        load_started_ns = now_ns()
        resolved = await ensure_model(model_name, options=body.get("options"))
        resolved_backend = resolved["backend_model"]

        async with active_requests_lock:
            active_requests[resolved_backend] = active_requests.get(resolved_backend, 0) + 1
            logger.info(
                f"In-flight request started for {resolved_backend}. Active: {active_requests[resolved_backend]}"
            )

        load_duration = now_ns() - load_started_ns
        
        think_val = body.get("think")
        if think_val is None:
            think_val = body.get("enable_thinking")
        if think_val is None and isinstance(body.get("options"), dict):
            opts = body["options"]
            think_val = opts.get("think") if opts.get("think") is not None else opts.get("enable_thinking")

        payload = (
            build_generate_chat_payload(body, resolved_backend)
            if use_chat_backend
            else build_generate_payload(body, resolved_backend)
        )

        attempt = 0
        max_attempts = 2
        resp = None
        while attempt < max_attempts:
            try:
                request_id = getattr(request.state, "request_id", "N/A")
                request_source = getattr(request.state, "request_source", "unknown")
                logger.info(
                    f"[GENERATE] Sending payload to llama-server (attempt {attempt}): model={payload.get('model')} | "
                    f"Origin: {request_source}",
                    extra={"request_id": request_id, "request_source": request_source},
                )
                resp = await client_httpx.post(f"{LLAMA_SERVER_URL}{endpoint}", json=payload)
                resp.raise_for_status()
                break
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                attempt += 1
                if attempt >= max_attempts:
                    raise
                logger.warning(
                    f"Upstream request failed (attempt {attempt}): {e}. Retrying recovery..."
                )
                await restart_llama_server()
                await wait_for_llama_server_or_restart()
                resolved = await ensure_model(model_name, options=body.get("options"))
                resolved_backend = resolved["backend_model"]
                payload["model"] = resolved_backend

        data = resp.json()
        final_response = ""
        final_thinking = None
        if use_chat_backend:
            choice = (data.get("choices") or [{}])[0]
            message = chat_message_from_choice(choice)
            final_response = message.get("content", "") if message else ""
            final_thinking = message.get("thinking") if message else None
            
            client_content = final_response
            if final_thinking:
                if think_val is None:
                    # Default: blend/wrap
                    if "<think>" not in final_response:
                        client_content = f"<think>\n{final_thinking}\n</think>\n{final_response}"
                elif think_val is True:
                    # Explicit True: separate
                    client_content = final_response
                else:
                    # Explicit False: strip thinking
                    client_content = final_response
                    final_thinking = None
                
            chunk = ollama_generate_chunk(
                model_name, client_content, True, choice.get("finish_reason")
            )
            if final_thinking is not None and think_val is not False:
                chunk["thinking"] = final_thinking
        else:
            done = data.get("stop", True)
            done_reason = data.get("finish_reason") or ("stop" if done else None)
            final_response = data.get("content") or ""
            final_thinking = data.get("thinking")
            
            client_content = final_response
            if final_thinking:
                if think_val is None:
                    # Default: blend/wrap
                    if "<think>" not in final_response:
                        client_content = f"<think>\n{final_thinking}\n</think>\n{final_response}"
                elif think_val is True:
                    # Explicit True: separate
                    client_content = final_response
                else:
                    # Explicit False: strip thinking
                    client_content = final_response
                    final_thinking = None
                
            chunk = ollama_generate_chunk(model_name, client_content, done, done_reason)
            if final_thinking is not None and think_val is not False:
                chunk["thinking"] = final_thinking
        apply_metrics(chunk, data, now_ns() - started_ns, load_duration)

        # Complete active request details
        prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
        gen_tokens = data.get("usage", {}).get("completion_tokens", 0)
        complete_active_request(
            request_id, 
            final_response=final_response, 
            final_thinking=final_thinking,
            prompt_tokens=prompt_tokens,
            completion_tokens=gen_tokens
        )

        await apply_keep_alive_policy(model_name, keep_alive)
        return JSONResponse(chunk)
    except httpx.HTTPStatusError as e:
        logger.error(f"Generate error: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.TimeoutException as e:
        logger.error(f"Generate upstream timeout: {e}")
        raise HTTPException(status_code=504, detail="Upstream llama-server timed out")
    except httpx.RequestError as e:
        logger.error(f"Generate upstream request error: {e}")
        raise HTTPException(status_code=502, detail=f"Upstream llama-server request failed: {e}")
    finally:
        if resolved_backend:
            async with active_requests_lock:
                active_requests[resolved_backend] = max(
                    0, active_requests.get(resolved_backend, 0) - 1
                )
                logger.info(
                    f"In-flight request finished for {resolved_backend}. Active: {active_requests[resolved_backend]}"
                )
                active_requests_lock.notify_all()
        complete_active_request(request_id)


@app.get("/api/tags")
async def tags():
    models = []
    for manifest_base, path, manifest in iter_local_manifests():
        model_name = manifest_model_name(manifest_base, path)
        if model_name:
            info = manifest_stats(path, manifest)
            models.append(
                {
                    "name": model_name,
                    "model": model_name,
                    "modified_at": info["modified_at"],
                    "size": info["size"],
                    "digest": info["digest"],
                    "details": info["details"],
                }
            )
    return {"models": models}


async def loaded_models_from_router():
    router_models = await fetch_router_models(reload=False)
    loaded = []
    local_records = []
    for manifest_base, path, manifest in iter_local_manifests():
        model_name = manifest_model_name(manifest_base, path)
        if model_name:
            local_records.append(
                {
                    "name": model_name,
                    "info": manifest_stats(path, manifest),
                    "candidates": router_model_candidates(model_name, manifest),
                }
            )

    for entry in router_models:
        if router_entry_status(entry) != "loaded":
            continue
        for record in local_records:
            if router_entry_matches(entry, record["candidates"]):
                loaded.append(record)
                break
    return loaded


async def get_llama_server_slots():
    """Fetch slot status from llama-server /slots endpoint."""
    try:
        # Get router model ID for loaded models - /slots requires the router model ID (with -- separators)
        router_models = await fetch_router_models(reload=False)
        for entry in router_models:
            if router_entry_status(entry) != "loaded":
                continue
            model_id = entry.get("id")
            if model_id:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.get(
                        f"{LLAMA_SERVER_URL}/slots", params={"model": model_id}, timeout=3.0
                    )
                    if resp.status_code == 200:
                        slots_data = resp.json()
                        total = len(slots_data) if isinstance(slots_data, list) else 0
                        busy = sum(1 for s in (slots_data or []) if s.get("is_processing", False))
                        return {"total": total, "busy": busy, "available": total - busy}
    except Exception:
        pass
    return None


@app.get("/api/ps")
async def ps():
    models = []
    for record in await loaded_models_from_router():
        info = record["info"]
        public_name = public_model_name(record["name"])
        models.append(
            {
                "name": public_name,
                "model": public_name,
                "size": info["size"],
                "digest": info["digest"],
                "details": info["details"],
                "expires_at": model_expires_at.get(public_name, "0001-01-01T00:00:00Z"),
                "size_vram": info["size"],
                "context_length": info["context_length"],
            }
        )

    # Add slot/busy info from llama-server
    slot_info = await get_llama_server_slots()

    # Add active request counts per model
    async with active_requests_lock:
        current_active = dict(active_requests)

    result = {"models": models}
    if slot_info:
        result["slots"] = slot_info
    if current_active:
        result["active_requests"] = current_active
    return result


@app.post("/api/queue/wait")
async def queue_wait(request: Request):
    """Wait for a llama-server slot to become available.

    Body: {"timeout": 300.0} — max seconds to wait (default 300)
    Returns: {"status": "ready", "slots": {...}} when a slot opens
    Or: {"status": "timeout"} if no slot available within timeout
    """
    body = await request.json()
    timeout = body.get("timeout", 300.0)

    start = asyncio.get_event_loop().time()
    poll_interval = 0.5

    while True:
        slot_info = await get_llama_server_slots()
        if slot_info and slot_info["available"] > 0:
            return {
                "status": "ready",
                "slots": slot_info,
                "wait_time": round(asyncio.get_event_loop().time() - start, 2),
            }

        elapsed = asyncio.get_event_loop().time() - start
        if elapsed >= timeout:
            return {"status": "timeout", "slots": slot_info, "waited": round(elapsed, 2)}

        await asyncio.sleep(poll_interval)


@app.post("/api/show")
async def show(request: Request):
    body = await request.json()
    model_name = body.get("model")
    verbose = body.get("verbose", False)
    if not model_name:
        raise HTTPException(status_code=400, detail="model is required")

    manifest_path = None
    for candidate in model_manifest_paths(model_name):
        if os.path.exists(candidate):
            manifest_path = candidate
            break
    if not manifest_path:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    manifest = read_manifest(manifest_path)
    if manifest is None or not is_model_complete(manifest):
        raise HTTPException(
            status_code=409, detail=f"Model {model_name} is still downloading or incomplete."
        )

    info = manifest_stats(manifest_path, manifest)
    response = {
        "parameters": info["parameters"],
        "license": info["license"],
        "template": info["template"],
        "system": info["system"],
        "capabilities": info["capabilities"],
        "modified_at": info["modified_at"],
        "details": info["details"],
        "model_info": info["model_info"] if verbose else info["model_info"],
    }
    return JSONResponse(response)


@app.get("/api/version")
async def version():
    return {"version": API_VERSION}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=11434)
