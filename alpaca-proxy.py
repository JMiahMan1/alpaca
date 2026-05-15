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
