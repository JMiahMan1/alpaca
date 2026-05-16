import asyncio
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from collections import deque
import uuid

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
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - [%(request_id)s] %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')
    
    stream_handler.setFormatter(formatter)
    deque_handler.setFormatter(formatter)
    
    logger.addHandler(stream_handler)
    logger.addHandler(deque_handler)

# Custom filter to ensure request_id is always present
class RequestIDFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'request_id'):
            record.request_id = 'N/A'
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
LLAMA_SERVER_CONNECT_TIMEOUT_SECONDS = float(os.getenv("LLAMA_SERVER_CONNECT_TIMEOUT_SECONDS", "60"))
LLAMA_SERVER_READ_TIMEOUT_SECONDS = os.getenv("LLAMA_SERVER_READ_TIMEOUT_SECONDS", "600").strip() # Default to 600s
FOREVER_EXPIRES_AT = "9999-12-31T23:59:59Z"

# Reference counting for active requests to prevent unloading in-use models
active_requests = {}
active_requests_lock = asyncio.Condition()


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
        layers = manifest.get('layers', [])
        config = manifest.get('config', {})
        for layer in layers + [config]:
            digest = layer.get('digest')
            if not digest: continue
            blob_path = blob_path_for_digest(digest)
            if not os.path.exists(blob_path):
                return False
            if os.path.getsize(blob_path) != layer.get('size', 0):
                return False
        return True
    except:
        return False

def read_manifest(path):
    """Load a manifest file, returning None for partial or malformed files."""
    try:
        with open(path, 'r') as f:
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
    candidates.extend([
        router_path,
        os.path.basename(router_path),
        os.path.splitext(os.path.basename(router_path))[0],
    ])
    model_layer = manifest_layer(manifest, "application/vnd.ollama.image.model")
    digest = model_layer.get("digest", "")
    if digest:
        blob_path = blob_path_for_digest(digest)
        candidates.extend([
            os.path.basename(blob_path),
            blob_path,
            digest,
            normalize_digest(digest),
        ])
    candidates.extend([
        public_model_name(model_name),
        with_default_tag(model_name),
    ])
    return [candidate for i, candidate in enumerate(candidates) if candidate and candidate not in candidates[:i]]

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
    return value * {
        "ms": 0.001,
        "s": 1,
        "m": 60,
        "h": 3600,
        "d": 86400,
    }[unit]

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
    return payload

async def fetch_router_models(reload=False):
    params = {"reload": "1"} if reload else None
    resp = await client_httpx.get(ROUTER_MODELS_URL, params=params)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data") or []

async def resolve_router_model(model_name, reload=True):
    resolved_name = with_default_tag(model_name)
    manifest_path, manifest = load_local_manifest(resolved_name, require_complete=True)
    if not manifest_path:
        raise HTTPException(status_code=404, detail=f"Model {resolved_name} not found")
    if manifest is None:
        raise HTTPException(status_code=409, detail=f"Model {resolved_name} is still downloading or incomplete.")

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
    if os.path.exists(fallback_path):
        return {
            "model_name": resolved_name,
            "backend_model": fallback_id,
            "entry": {"id": fallback_id, "path": fallback_path, "status": {"value": "unloaded"}},
            "manifest_path": manifest_path,
            "manifest": manifest,
            "router_models": router_models,
        }

    raise HTTPException(status_code=404, detail=f"Router could not discover backend model for {resolved_name}")

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

    resp = await client_httpx.post(url, json=json_body)
    if resp.status_code == 404:
        router_management_supported = False
        raise RouterManagementUnsupported(url)
    router_management_supported = True
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
    return "model is not found" in message or "model not found" in message

async def unload_model(model_name):
    try:
        resolved = await resolve_router_model(model_name, reload=False)
    except HTTPException:
        return
    async with router_model_lock:
        backend_model = resolved["backend_model"]
        try:
            await post_router_model_action("unload", backend_model)
        except RouterManagementUnsupported:
            logger.warning("Router model unload endpoint unavailable; skipping explicit unload.")
            return
        except httpx.HTTPStatusError as exc:
            if is_ignorable_router_unload_error(exc) or not await router_model_is_still_resident(backend_model):
                logger.info(f"Router ignored unload for {backend_model}: {exc}")
                return
            raise
    model_expires_at.pop(public_model_name(model_name), None)

def cancel_model_unload(model_name):
    task = model_unload_tasks.pop(public_model_name(model_name), None)
    if task:
        task.cancel()

def begin_model_request(model_name):
    cancel_model_unload(model_name)
    model_expires_at[public_model_name(model_name)] = "0001-01-01T00:00:00Z"

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

    model_expires_at[public_name] = datetime.fromtimestamp(time.time() + seconds).astimezone().isoformat()
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
    prompt_eval_duration = prompt_eval_duration if prompt_eval_duration is not None else timings.get("prompt_ms")
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
    thinking = delta.get("reasoning_content") or delta.get("thinking") or message.get("reasoning_content") or message.get("thinking")
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
    yield
    for task in list(model_unload_tasks.values()):
        task.cancel()
    await client_httpx.aclose()

app = FastAPI(lifespan=lifespan)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    logger.info(f"Hit: {request.method} {request.url.path}", extra={"request_id": request_id})
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"Finished: {request.method} {request.url.path} in {duration:.3f}s", extra={"request_id": request_id})
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
        metrics["requests_by_endpoint"][endpoint] = metrics["requests_by_endpoint"].get(endpoint, 0) + 1
        metrics["tokens_prompted"] += prompt_tokens
        metrics["tokens_generated"] += gen_tokens
        if error:
            metrics["errors_total"] += 1
        metrics["latency_samples"].append(latency_ms)
        if len(metrics["latency_samples"]) > 1000:
            metrics["latency_samples"] = metrics["latency_samples"][-500:]
        metrics["avg_latency_ms"] = sum(metrics["latency_samples"]) / len(metrics["latency_samples"])

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
                    grammars.append({
                        "name": f[:-5],
                        "type": data.get("type", "grammar"),
                        "description": data.get("description", ""),
                        "created_at": data.get("created_at", ""),
                        "size": os.path.getsize(path),
                    })
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
                    schemas.append({
                        "name": f[:-5],
                        "description": data.get("description", ""),
                        "created_at": data.get("created_at", ""),
                        "size": os.path.getsize(path),
                    })
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

        await record_metrics("/api/embed", total_duration / 1e6, prompt_tokens=result["prompt_eval_count"])
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
            models.append({
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
            })
    return {"object": "list", "data": models}

@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    """OpenAI-compatible chat completions. Proxies directly to llama-server."""
    body = await request.json()
    started_ns = now_ns()
    model_name = body.get("model", "")
    stream = body.get("stream", False)

    # Resolve model if provided
    if model_name:
        try:
            resolved = await ensure_model(model_name)
            body["model"] = resolved["backend_model"]
        except HTTPException:
            pass  # Let llama-server handle unknown models

    try:
        if stream:
            async def stream_proxy():
                try:
                    async with client_httpx.stream("POST", f"{LLAMA_SERVER_URL}/v1/chat/completions", json=body) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if line and line.startswith("data: "):
                                yield f"{line}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                finally:
                    await record_metrics("/v1/chat/completions", (now_ns() - started_ns) / 1e6)
            return StreamingResponse(stream_proxy(), media_type="text/event-stream")
        else:
            resp = await client_httpx.post(f"{LLAMA_SERVER_URL}/v1/chat/completions", json=body)
            resp.raise_for_status()
            data = resp.json()
            latency = (now_ns() - started_ns) / 1e6
            prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
            gen_tokens = data.get("usage", {}).get("completion_tokens", 0)
            await record_metrics("/v1/chat/completions", latency, prompt_tokens, gen_tokens)
            return JSONResponse(data)
    except httpx.HTTPStatusError as e:
        await record_metrics("/v1/chat/completions", 0, error=True)
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.TimeoutException:
        await record_metrics("/v1/chat/completions", 0, error=True)
        raise HTTPException(status_code=504, detail="Upstream llama-server timed out")
    except httpx.RequestError as e:
        await record_metrics("/v1/chat/completions", 0, error=True)
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {e}")

@app.post("/v1/completions")
async def openai_completions(request: Request):
    """OpenAI-compatible text completions. Proxies directly to llama-server."""
    body = await request.json()
    started_ns = now_ns()
    model_name = body.get("model", "")
    stream = body.get("stream", False)

    if model_name:
        try:
            resolved = await ensure_model(model_name)
            body["model"] = resolved["backend_model"]
        except HTTPException:
            pass

    try:
        if stream:
            async def stream_proxy():
                try:
                    async with client_httpx.stream("POST", f"{LLAMA_SERVER_URL}/v1/completions", json=body) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if line and line.startswith("data: "):
                                yield f"{line}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                finally:
                    await record_metrics("/v1/completions", (now_ns() - started_ns) / 1e6)
            return StreamingResponse(stream_proxy(), media_type="text/event-stream")
        else:
            resp = await client_httpx.post(f"{LLAMA_SERVER_URL}/v1/completions", json=body)
            resp.raise_for_status()
            data = resp.json()
            latency = (now_ns() - started_ns) / 1e6
            prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
            gen_tokens = data.get("usage", {}).get("completion_tokens", 0)
            await record_metrics("/v1/completions", latency, prompt_tokens, gen_tokens)
            return JSONResponse(data)
    except httpx.HTTPStatusError as e:
        await record_metrics("/v1/completions", 0, error=True)
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.TimeoutException:
        await record_metrics("/v1/completions", 0, error=True)
        raise HTTPException(status_code=504, detail="Upstream llama-server timed out")
    except httpx.RequestError as e:
        await record_metrics("/v1/completions", 0, error=True)
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {e}")

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
        except HTTPException:
            pass

    try:
        resp = await client_httpx.post(f"{LLAMA_SERVER_URL}/v1/embeddings", json=body)
        resp.raise_for_status()
        data = resp.json()
        latency = (now_ns() - started_ns) / 1e6
        prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
        await record_metrics("/v1/embeddings", latency, prompt_tokens)
        return JSONResponse(data)
    except httpx.HTTPStatusError as e:
        await record_metrics("/v1/embeddings", 0, error=True)
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.TimeoutException:
        await record_metrics("/v1/embeddings", 0, error=True)
        raise HTTPException(status_code=504, detail="Upstream llama-server timed out")
    except httpx.RequestError as e:
        await record_metrics("/v1/embeddings", 0, error=True)
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {e}")

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
            local_models.append({
                "name": mn,
                "size": info["size"],
                "details": info["details"],
                "capabilities": info["capabilities"],
            })
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
    import shutil
    import platform

    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }

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
                "chat_template": props.get("chat_template", "")[:100] + "..." if props.get("chat_template") else None,
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
        router_entry = None
        try:
            router_models = await fetch_router_models(reload=False)
            candidates = router_model_candidates(mn, manifest)
            for entry in router_models:
                if router_entry_matches(entry, candidates):
                    router_status = router_entry_status(entry)
                    router_entry = entry
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
            blobs.append({
                "digest": normalize_digest(digest),
                "size": size,
                "exists": exists,
                "media_type": layer.get("mediaType", ""),
            })
            if exists:
                total_blob_size += size

        models.append({
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
        })

    return {"models": models, "total": len(models)}

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

    # Loaded models
    loaded = []
    try:
        for record in await loaded_models_from_router():
            info = record["info"]
            public_name = public_model_name(record["name"])
            loaded.append({
                "name": public_name,
                "size": info["size"],
                "digest": info["digest"],
                "details": info["details"],
                "context_length": info["context_length"],
                "expires_at": model_expires_at.get(public_name, "0001-01-01T00:00:00Z"),
                "active_requests": active.get(record.get("backend_model", ""), 0),
            })
    except Exception:
        pass

    return {
        "loaded_models": loaded,
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
            timeout=httpx.Timeout(5.0)
        )
        resp.raise_for_status()
        slots = resp.json()

        # Enhance with Alpaca metadata
        for slot in slots:
            slot_id = slot.get("id", -1)
            slot["alpaca"] = {
                "is_busy": slot.get("is_processing", False),
                "has_prompt_cache": bool(slot.get("prompt", [])),
                "context_used_pct": round(slot.get("n_ctx", 0) / max(slot.get("n_ctx", 1), 1) * 100, 1) if slot.get("n_ctx") else 0,
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
        resp = await client_httpx.get(f"{LLAMA_SERVER_URL}/lora-adapters", timeout=httpx.Timeout(5.0))
        resp.raise_for_status()
        adapters = resp.json()
        return {"adapters": adapters, "total": len(adapters)}
    except Exception as e:
        return {"adapters": [], "error": str(e), "note": "LoRA adapters not configured or endpoint unavailable"}

@app.post("/admin/lora")
async def admin_lora_update(request: Request):
    """Update LoRA adapter scales. Body: [{\"id\": 0, \"scale\": 0.5}, ...]"""
    body = await request.json()
    if not isinstance(body, list):
        raise HTTPException(status_code=400, detail="Body must be a list of {id, scale} objects")
    try:
        resp = await client_httpx.post(f"{LLAMA_SERVER_URL}/lora-adapters", json=body, timeout=httpx.Timeout(5.0))
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

    return {"status": "config_update_requested", "note": "Runtime config changes require restart for full effect", "requested": body}

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
            timeout=httpx.Timeout(5.0)
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

async def ensure_model(model_name: str, options: dict = None):
    model_name = with_default_tag(model_name)
    begin_model_request(model_name)
    
    async with router_model_lock:
        resolved = await resolve_router_model(model_name, reload=True)
        backend_model = resolved["backend_model"]
        router_models = resolved["router_models"]
        entry = resolved["entry"]

        if MAX_LOADED_MODELS == 1:
            for other in router_models:
                other_id = other.get("id")
                if (other_id != backend_model) and is_resident_status(router_entry_status(other)):
                    # Wait for active requests on the model we are about to unload
                    async with active_requests_lock:
                        while active_requests.get(other_id, 0) > 0:
                            logger.info(f"Waiting for {active_requests[other_id]} active requests on {other_id} to finish before unloading.")
                            await active_requests_lock.wait()
                    
                    logger.info(f"Unloading backend model {other_id} before loading {backend_model}")
                    try:
                        await post_router_model_action("unload", other_id)
                    except RouterManagementUnsupported:
                        logger.warning("Router unload endpoint unavailable; relying on router autoload behavior.")
                        break
                    except httpx.HTTPStatusError as exc:
                        if is_ignorable_router_unload_error(exc) or not await router_model_is_still_resident(other_id):
                            logger.info(f"Router ignored unload for {other_id}: {exc}")
                            continue
                        raise

        # If already loaded, skip load entirely
        if router_entry_status(entry) == "loaded":
            logger.info(f"Model {backend_model} already loaded (status=loaded), proceeding.")
            return {
                "model_name": model_name,
                "backend_model": backend_model,
                "manifest_path": resolved["manifest_path"],
                "manifest": resolved["manifest"],
            }

        # Need to load — attempt load with optimized parameters
        logger.info(f"Loading backend model {backend_model} for {public_model_name(model_name)}")
        
        # Optimization: Pass n_ctx and other flags to the load call if provided in options
        load_payload = {"model": backend_model}
        if options:
            n_ctx = options.get("num_ctx") or options.get("n_ctx")
            if n_ctx:
                load_payload["n_ctx"] = int(n_ctx)
                logger.info(f"Setting n_ctx={n_ctx} for model load.")
        
        # Always use acceleration flags
        load_payload["n_gpu_layers"] = -1
        load_payload["use_mmap"] = True
        load_payload["flash_attn"] = True

        try:
            await post_router_model_action("load", load_payload)
        except RouterManagementUnsupported:
            logger.warning("Router load endpoint unavailable; relying on request-time model autoload.")
            return {
                "model_name": model_name,
                "backend_model": backend_model,
                "manifest_path": resolved["manifest_path"],
                "manifest": resolved["manifest"],
            }
        except httpx.HTTPStatusError as exc:
            # 400 "model is already running" is harmless
            if exc.response.status_code == 400:
                try:
                    payload = exc.response.json()
                    msg = str(payload.get("error", {}).get("message", "")) if isinstance(payload, dict) else ""
                except Exception:
                    msg = exc.response.text
                if "already running" in msg.lower() or "model is already running" in msg.lower():
                    logger.info(f"Model {backend_model} already loaded (detected via 400), proceeding.")
                    return {
                        "model_name": model_name,
                        "backend_model": backend_model,
                        "manifest_path": resolved["manifest_path"],
                        "manifest": resolved["manifest"],
                    }
                else:
                    raise
            else:
                raise

        # Load succeeded
        logger.info(f"Model {backend_model} loaded successfully.")
        return {
            "model_name": model_name,
            "backend_model": backend_model,
            "manifest_path": resolved["manifest_path"],
            "manifest": resolved["manifest"],
        }

@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    model_name = body.get("model")
    keep_alive = effective_keep_alive(body.get("keep_alive"))

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
                logger.info(f"In-flight request started for {resolved_backend}. Active: {active_requests[resolved_backend]}")

            load_duration = now_ns() - load_started_ns
            payload = build_chat_payload(body, resolved_backend)
            
            request_id = getattr(request.state, "request_id", "N/A")
            logger.info(f"[CHAT] Sending payload to llama-server: model={payload.get('model')}, stream={payload.get('stream')}", extra={"request_id": request_id})
            
            async with client_httpx.stream("POST", f"{LLAMA_SERVER_URL}/v1/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or "[DONE]" in line: continue
                    if line.startswith("data: "): line = line[6:]
                    try:
                        data = json.loads(line)
                        choice = (data.get("choices") or [{}])[0]
                        message = chat_message_from_choice(choice)
                        done = choice.get("finish_reason") is not None
                        chunk = ollama_chat_chunk(model_name, message, done, choice.get("finish_reason"))
                        if done:
                            apply_metrics(chunk, data, now_ns() - started_ns, load_duration)
                        yield json.dumps(chunk) + "\n"
                    except: continue
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
                    active_requests[resolved_backend] -= 1
                    logger.info(f"In-flight request finished for {resolved_backend}. Active: {active_requests[resolved_backend]}")
                    active_requests_lock.notify_all()
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
            logger.info(f"In-flight request started for {resolved_backend}. Active: {active_requests[resolved_backend]}")

        load_duration = now_ns() - load_started_ns
        payload = build_chat_payload(body, resolved_backend)
        resp = await client_httpx.post(f"{LLAMA_SERVER_URL}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        chunk = ollama_chat_chunk(model_name, chat_message_from_choice(choice), True, choice.get("finish_reason"))
        apply_metrics(chunk, data, now_ns() - started_ns, load_duration)
        logprobs = logprobs_from_choice(choice)
        if logprobs is not None:
            chunk["logprobs"] = logprobs
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
                active_requests[resolved_backend] -= 1
                logger.info(f"In-flight request finished for {resolved_backend}. Active: {active_requests[resolved_backend]}")
                active_requests_lock.notify_all()

@app.post("/api/generate")
async def generate(request: Request):
    body = await request.json()
    model_name = body.get("model")
    use_chat_backend = should_generate_via_chat(body)
    endpoint = "/v1/chat/completions" if use_chat_backend else "/completion"
    keep_alive = effective_keep_alive(body.get("keep_alive"))

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
                logger.info(f"In-flight request started for {resolved_backend}. Active: {active_requests[resolved_backend]}")

            load_duration = now_ns() - load_started_ns
            payload = build_generate_chat_payload(body, resolved_backend) if use_chat_backend else build_generate_payload(body, resolved_backend)
            
            async with client_httpx.stream("POST", f"{LLAMA_SERVER_URL}{endpoint}", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or "[DONE]" in line: continue
                    if line.startswith("data: "): line = line[6:]
                    try:
                        data = json.loads(line)
                        if use_chat_backend:
                            choice = (data.get("choices") or [{}])[0]
                            message = chat_message_from_choice(choice)
                            done = choice.get("finish_reason") is not None
                            chunk = ollama_generate_chunk(model_name, message.get("content", ""), done, choice.get("finish_reason"))
                            if "thinking" in message:
                                chunk["thinking"] = message["thinking"]
                            if done:
                                apply_metrics(chunk, data, now_ns() - started_ns, load_duration)
                        else:
                            done = data.get("stop", False)
                            done_reason = data.get("finish_reason") or ("stop" if done else None)
                            chunk = ollama_generate_chunk(model_name, data.get("content") or "", done, done_reason)
                            if data.get("thinking") is not None:
                                chunk["thinking"] = data.get("thinking")
                            if done:
                                apply_metrics(chunk, data, now_ns() - started_ns, load_duration)
                        yield json.dumps(chunk) + "\n"
                    except: continue
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
                    active_requests[resolved_backend] -= 1
                    logger.info(f"In-flight request finished for {resolved_backend}. Active: {active_requests[resolved_backend]}")
                    active_requests_lock.notify_all()
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
            logger.info(f"In-flight request started for {resolved_backend}. Active: {active_requests[resolved_backend]}")

        load_duration = now_ns() - load_started_ns
        payload = build_generate_chat_payload(body, resolved_backend) if use_chat_backend else build_generate_payload(body, resolved_backend)
        resp = await client_httpx.post(f"{LLAMA_SERVER_URL}{endpoint}", json=payload)
        resp.raise_for_status()
        data = resp.json()
        if use_chat_backend:
            choice = (data.get("choices") or [{}])[0]
            message = chat_message_from_choice(choice)
            chunk = ollama_generate_chunk(model_name, message.get("content", ""), True, choice.get("finish_reason"))
            if "thinking" in message:
                chunk["thinking"] = message["thinking"]
        else:
            done = data.get("stop", True)
            done_reason = data.get("finish_reason") or ("stop" if done else None)
            chunk = ollama_generate_chunk(model_name, data.get("content") or "", done, done_reason)
            if data.get("thinking") is not None:
                chunk["thinking"] = data.get("thinking")
        apply_metrics(chunk, data, now_ns() - started_ns, load_duration)
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
                active_requests[resolved_backend] -= 1
                logger.info(f"In-flight request finished for {resolved_backend}. Active: {active_requests[resolved_backend]}")
                active_requests_lock.notify_all()

@app.get("/api/tags")
async def tags():
    models = []
    for manifest_base, path, manifest in iter_local_manifests():
        model_name = manifest_model_name(manifest_base, path)
        if model_name:
            info = manifest_stats(path, manifest)
            models.append({
                "name": model_name,
                "model": model_name,
                "modified_at": info["modified_at"],
                "size": info["size"],
                "digest": info["digest"],
                "details": info["details"]
            })
    return {"models": models}

async def loaded_models_from_router():
    router_models = await fetch_router_models(reload=False)
    loaded = []
    local_records = []
    for manifest_base, path, manifest in iter_local_manifests():
        model_name = manifest_model_name(manifest_base, path)
        if model_name:
            local_records.append({
                "name": model_name,
                "info": manifest_stats(path, manifest),
                "candidates": router_model_candidates(model_name, manifest),
            })

    for entry in router_models:
        if router_entry_status(entry) != "loaded":
            continue
        for record in local_records:
            if router_entry_matches(entry, record["candidates"]):
                loaded.append(record)
                break
    return loaded

@app.get("/api/ps")
async def ps():
    models = []
    for record in await loaded_models_from_router():
        info = record["info"]
        public_name = public_model_name(record["name"])
        models.append({
            "name": public_name,
            "model": public_name,
            "size": info["size"],
            "digest": info["digest"],
            "details": info["details"],
            "expires_at": model_expires_at.get(public_name, "0001-01-01T00:00:00Z"),
            "size_vram": info["size"],
            "context_length": info["context_length"],
        })
    return {"models": models}

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
        raise HTTPException(status_code=409, detail=f"Model {model_name} is still downloading or incomplete.")

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
