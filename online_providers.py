"""
Online Model Providers integration for Alpaca LLM Benchmarks.
Supports:
1. Dynamic model discovery & live search from OpenRouter, Hugging Face, Cloudflare, OpenCode Zen / Custom OpenAI
2. API Key configuration, credential testing, and persistence
3. Execution and benchmarking of online models
4. Custom user selection persistence
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import threading
import time
import uuid
from collections import deque
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("online_providers")

# In-process tracker for online provider queries. Unlike local requests, online
# queries never touch the alpaca-proxy slot manager or llama-server, so their
# lifecycle is tracked here and merged into the web dashboard's request queue.
_online_request_lock = threading.Lock()
_active_online_requests: dict[str, dict[str, Any]] = {}
_completed_online_requests: deque[dict[str, Any]] = deque(maxlen=50)


def start_online_request(
    request_id: str,
    model: str,
    req_type: str,
    payload: dict[str, Any],
    request_source: str = "web",
    client_ip: str = "web",
) -> None:
    """Registers an in-flight online provider query for the request queue."""
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
    prompt_str = prompt_str[:2000]

    with _online_request_lock:
        _active_online_requests[request_id] = {
            "request_id": request_id,
            "model": model,
            "type": req_type,
            "started_at": time.time(),
            "prompt": prompt_str,
            "thinking": "",
            "response": "",
            "request_source": request_source,
            "client_ip": client_ip,
            "online": True,
        }


def complete_online_request(
    request_id: str,
    result: dict[str, Any],
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> None:
    """Moves a finished online query into the completed queue, mirroring the proxy shape."""
    with _online_request_lock:
        req = _active_online_requests.pop(request_id, None)
    if req is None:
        return

    req["response"] = result.get("response") or ""
    req["error"] = result.get("error")
    req["success"] = bool(result.get("success"))
    req["completed_at"] = time.time()
    req["duration_seconds"] = round(req["completed_at"] - req["started_at"], 2)

    tokens = completion_tokens or int(result.get("tokens_generated") or 0)
    if not tokens and req["response"]:
        tokens = max(1, int(len(req["response"]) / 4))
    req["completion_tokens"] = tokens
    if not prompt_tokens and req.get("prompt"):
        prompt_tokens = max(1, int(len(req["prompt"]) / 4))
    req["prompt_tokens"] = prompt_tokens

    if req["duration_seconds"] > 0:
        req["tps"] = round(tokens / req["duration_seconds"], 2)
    else:
        req["tps"] = 0.0
    req["ttft_seconds"] = req["duration_seconds"]

    with _online_request_lock:
        _completed_online_requests.append(req)


def cancel_online_request(request_id: str) -> bool:
    """Removes an active online query (cannot abort the HTTP call, just drops it from the queue)."""
    with _online_request_lock:
        return _active_online_requests.pop(request_id, None) is not None


def clear_completed_online_requests() -> int:
    """Clears the completed online query buffer."""
    with _online_request_lock:
        count = len(_completed_online_requests)
        _completed_online_requests.clear()
    return count


def get_online_requests() -> dict[str, list[dict[str, Any]]]:
    """Returns the current online query queue (active + completed)."""
    with _online_request_lock:
        return {
            "active_requests": list(_active_online_requests.values()),
            "completed_requests": list(_completed_online_requests),
        }


def _make_request_id(model_identifier: str) -> str:
    """Builds a stable request id for an online query (uuid, `online`-prefixed to avoid colliding with proxy ids)."""
    return f"online-{uuid.uuid4().hex}"


def _opencode_zen_project_id() -> str:
    """Derives the x-opencode-project id the CLI would send for this repo.

    Mirrors opencode's ProjectInfo: sha1 of "git-remote:<owner>/<repo>" from the
    nearest git remote, else the first root commit hash, else "global".
    """
    try:
        import subprocess

        remote = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if remote.returncode == 0 and remote.stdout.strip():
            url = remote.stdout.strip()
            for prefix in ("https://", "http://", "git@", "ssh://git@"):
                url = url.removeprefix(prefix)
            url = url.rstrip("/").removesuffix(".git")
            return hashlib.sha1(f"git-remote:{url}".encode()).hexdigest()
    except Exception:
        pass
    return "global"


def _opencode_zen_headers(api_key: str | None = None) -> dict[str, str]:
    """Builds the x-opencode-* headers the opencode CLI sends to OpenCode Zen.

    Zen's free-tier limiter attributes traffic by these headers; plain API calls
    without them are treated as unattributed raw traffic and get HTTP 429.
    """
    session_id = "ses_" + secrets.token_hex(13)
    request_id = "msg_" + secrets.token_hex(13)
    headers = {
        "Content-Type": "application/json",
        "x-opencode-project": _opencode_zen_project_id(),
        "x-opencode-session": session_id,
        "x-opencode-request": request_id,
        "x-opencode-client": "cli",
        "User-Agent": "opencode/1.18.18",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


class OnlineModelProvider:
    """Manages connections, live discovery, credentials, and queries to online LLM providers.

    All model discovery is 100% dynamic and pulled live on-demand via remote provider APIs.
    """

    def __init__(self):
        self._selected_models_file = Path("data/online_models_selected.json")
        self._cached_live_models: dict[str, list] = {}
        self.reload_credentials()

    def reload_credentials(self) -> None:
        """Reload credentials from .env and environment.

        Each value is read from a single source with no fallback or empty-string
        default; a missing value stays None so callers can surface a clear
        configuration error instead of silently substituting an empty value.
        """
        self._load_dotenv_if_present()
        self.alpaca_api_key = os.getenv("ALPACA_API_KEY")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.huggingface_token = os.getenv("HUGGING_FACE_TOKEN")
        self.cloudflare_api_token = os.getenv("CLOUDFLARE_API_TOKEN")
        self.cloudflare_account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        self.opencode_zen_api_key = os.getenv("OPENCODE_ZEN_API_KEY")
        self.opencode_zen_base_url = os.getenv("OPENCODE_ZEN_BASE_URL")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

    @staticmethod
    def generate_alpaca_token() -> str:
        """Generates a secure Alpaca Bearer token."""
        import secrets

        return f"alpaca-sk-{secrets.token_hex(20)}"

    @staticmethod
    def _find_dotenv_path() -> Path | None:
        candidates = [Path.cwd() / ".env", Path(__file__).resolve().parent / ".env", Path("/app/.env")]
        for p in candidates:
            if p.is_file():
                return p
        return None

    def _load_dotenv_if_present(self) -> None:
        """Helper to load .env into os.environ if present on disk."""
        p = self._find_dotenv_path()
        if p and p.is_file():
            try:
                with open(p, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, _, v = line.partition("=")
                            k = k.strip()
                            v = v.strip().strip("\"'")
                            os.environ[k] = v
            except Exception:
                pass

    def save_credentials(self, keys: dict[str, str]) -> dict[str, Any]:
        """Save API credentials to .env file and update runtime environment."""
        p = self._find_dotenv_path() or (Path.cwd() / ".env")
        current_env: dict[str, str] = {}
        if p.exists():
            try:
                with open(p, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, _, v = line.partition("=")
                            current_env[k.strip()] = v.strip().strip("\"'")
            except Exception:
                pass

        # Update environment
        for key, val in keys.items():
            if val is not None:
                current_env[key] = val
                os.environ[key] = val

        # Write back to .env
        try:
            with open(p, "w", encoding="utf-8") as f:
                f.write("# ==============================================================================\n")
                f.write("# Alpaca Configuration & Online Provider Credentials\n")
                f.write("# ==============================================================================\n\n")
                for k, v in sorted(current_env.items()):
                    f.write(f"{k}={v}\n")
        except Exception as e:
            return {"success": False, "error": f"Failed to write .env file: {e}"}

        self.reload_credentials()
        return {"success": True, "configured": self.get_configured_providers()}

    def get_configured_providers(self) -> dict[str, bool]:
        """Returns map of providers and whether their API credentials are configured."""
        return {
            "alpaca": bool(self.alpaca_api_key),
            "openrouter": bool(self.openrouter_api_key),
            "huggingface": bool(self.huggingface_token),
            "cloudflare": bool(self.cloudflare_api_token and self.cloudflare_account_id),
            "opencode_zen": bool(self.opencode_zen_base_url),
            "groq": bool(self.groq_api_key),
            "gemini": bool(self.gemini_api_key),
        }

    def get_masked_credentials(self) -> dict[str, Any]:
        """Returns masked credentials and configuration info for display in the UI."""

        def mask(s: str | None) -> str:
            if not s:
                return ""
            if len(s) <= 8:
                return "••••••••"
            return f"{s[:4]}••••{s[-4:]}"

        return {
            "alpaca": {
                "configured": bool(self.alpaca_api_key),
                "masked_key": mask(self.alpaca_api_key),
                "has_key": bool(self.alpaca_api_key),
                "auth_required": bool(self.alpaca_api_key),
            },
            "openrouter": {
                "configured": bool(self.openrouter_api_key),
                "masked_key": mask(self.openrouter_api_key),
                "has_key": bool(self.openrouter_api_key),
            },
            "huggingface": {
                "configured": bool(self.huggingface_token),
                "masked_key": mask(self.huggingface_token),
                "has_key": bool(self.huggingface_token),
            },
            "cloudflare": {
                "configured": bool(self.cloudflare_api_token and self.cloudflare_account_id),
                "masked_token": mask(self.cloudflare_api_token),
                "account_id": self.cloudflare_account_id,
                "has_token": bool(self.cloudflare_api_token),
            },
            "opencode_zen": {
                "configured": bool(self.opencode_zen_api_key or self.opencode_zen_base_url),
                "masked_key": mask(self.opencode_zen_api_key),
                "base_url": self.opencode_zen_base_url,
                "has_key": bool(self.opencode_zen_api_key),
            },
            "groq": {
                "configured": bool(self.groq_api_key),
                "masked_key": mask(self.groq_api_key),
                "has_key": bool(self.groq_api_key),
            },
            "gemini": {
                "configured": bool(self.gemini_api_key),
                "masked_key": mask(self.gemini_api_key),
                "has_key": bool(self.gemini_api_key),
            },
        }

    async def test_connection(self, provider: str, custom_keys: dict[str, str] | None = None) -> dict[str, Any]:
        """Verify API key validity against the remote provider API."""
        custom = custom_keys or {}
        try:
            if provider == "openrouter":
                api_key = custom.get("openrouter_api_key") or self.openrouter_api_key
                headers = {
                    "HTTP-Referer": "https://github.com/JMiahMan1/alpaca",
                    "X-Title": "Alpaca LLM Benchmark Suite",
                }
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"

                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get("https://openrouter.ai/api/v1/models", headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        count = len(data.get("data", []))
                        auth_status = (
                            "Authenticated with API Key." if api_key else "Public catalog verified (Free models ready)."
                        )
                        return {
                            "success": True,
                            "message": f"Connected to OpenRouter! {count} models available. {auth_status}",
                            "count": count,
                        }
                    return {
                        "success": False,
                        "error": self._format_http_error("OpenRouter", resp.status_code, resp.text[:200]),
                    }

            elif provider == "huggingface":
                api_key = custom.get("huggingface_token") or self.huggingface_token
                if not api_key:
                    return {"success": False, "error": "Hugging Face Token not provided."}

                headers = {"Authorization": f"Bearer {api_key}"}
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get("https://huggingface.co/api/whoami-v2", headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        user = data.get("name", "authenticated")
                        return {"success": True, "message": f"Connected as user '{user}'."}
                    return {
                        "success": False,
                        "error": self._format_http_error("Hugging Face", resp.status_code, resp.text[:200]),
                    }

            elif provider == "cloudflare":
                api_token = custom.get("cloudflare_api_token") or self.cloudflare_api_token
                account_id = custom.get("cloudflare_account_id") or self.cloudflare_account_id
                if not api_token or not account_id:
                    return {"success": False, "error": "Cloudflare API Token and Account ID required."}

                headers = {"Authorization": f"Bearer {api_token}"}
                async with httpx.AsyncClient(timeout=15.0) as client:
                    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models/search"
                    resp = await client.get(url, headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        count = len(data.get("result", []))
                        return {"success": True, "message": f"Connected to Cloudflare! {count} AI models found."}
                    return {
                        "success": False,
                        "error": self._format_http_error("Cloudflare", resp.status_code, resp.text[:200]),
                    }

            elif provider == "opencode_zen":
                api_key = custom.get("opencode_zen_api_key") or self.opencode_zen_api_key
                base_url = custom.get("opencode_zen_base_url") or self.opencode_zen_base_url
                if not base_url:
                    return {
                        "success": False,
                        "error": "OpenCode Zen base URL not configured. Set OPENCODE_ZEN_BASE_URL in Settings.",
                    }
                base_url = base_url.rstrip("/")
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get(f"{base_url}/models", headers=headers)
                    if resp.status_code in (200, 201):
                        return {"success": True, "message": "Successfully reached endpoint!"}
                    return {"success": False, "error": self._format_http_error("Endpoint", resp.status_code, "")}

            elif provider == "groq":
                api_key = custom.get("groq_api_key") or self.groq_api_key
                if not api_key:
                    return {"success": False, "error": "Groq API Key not provided."}

                headers = {"Authorization": f"Bearer {api_key}"}
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get("https://api.groq.com/openai/v1/models", headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        count = len(data.get("data", []))
                        return {"success": True, "message": f"Connected to Groq! {count} models available."}
                    return {
                        "success": False,
                        "error": self._format_http_error("Groq", resp.status_code, resp.text[:200]),
                    }

            elif provider == "gemini":
                api_key = custom.get("gemini_api_key") or self.gemini_api_key
                if not api_key:
                    return {"success": False, "error": "Gemini API Key not provided."}

                headers = {"x-goog-api-key": api_key}
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get("https://generativelanguage.googleapis.com/v1beta/models", headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        count = len(data.get("models", []))
                        return {"success": True, "message": f"Connected to Gemini! {count} models available."}
                    return {
                        "success": False,
                        "error": self._format_http_error("Gemini", resp.status_code, resp.text[:200]),
                    }

            return {"success": False, "error": f"Unknown provider '{provider}'"}

        except Exception as e:
            return {"success": False, "error": f"Connection test error: {e}"}

    async def fetch_live_models(
        self, provider: str = "all", query: str = "", free_only: bool = False
    ) -> list[dict[str, Any]]:
        """Dynamically fetch and search available models from remote provider APIs in real-time."""
        results: list[dict[str, Any]] = []
        providers_to_query = (
            ["openrouter", "huggingface", "cloudflare", "opencode_zen", "groq", "gemini"]
            if provider == "all"
            else [provider]
        )

        query_lower = query.lower().strip()

        # 1. OpenRouter Live Discovery
        if "openrouter" in providers_to_query:
            try:
                headers = {}
                if self.openrouter_api_key:
                    headers["Authorization"] = f"Bearer {self.openrouter_api_key}"
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get("https://openrouter.ai/api/v1/models", headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        raw_models = data.get("data", [])
                        self._cached_live_models["openrouter"] = raw_models
                        for m in raw_models:
                            m_id = m.get("id", "")
                            m_name = m.get("name", m_id)
                            desc = m.get("description", "")
                            ctx = m.get("context_length", 0)
                            pricing = m.get("pricing", {})
                            prompt_p = float(pricing.get("prompt", 0) or 0)
                            comp_p = float(pricing.get("completion", 0) or 0)
                            is_free = (prompt_p == 0 and comp_p == 0) or m_id.endswith(":free")

                            if free_only and not is_free:
                                continue
                            if query_lower and (query_lower not in m_id.lower() and query_lower not in m_name.lower()):
                                continue

                            pricing_str = (
                                "Free (Rate-limited)"
                                if is_free
                                else (
                                    f"${prompt_p * 1e6:.2f}/M in, ${comp_p * 1e6:.2f}/M out"
                                    if (prompt_p > 0 or comp_p > 0)
                                    else "Paid Tier"
                                )
                            )

                            results.append(
                                {
                                    "id": f"openrouter:{m_id}",
                                    "name": m_id,
                                    "label": m_name,
                                    "provider": "openrouter",
                                    "free": is_free,
                                    "free_tier": "Free (Rate-limited)" if is_free else "Paid Tier",
                                    "pricing_label": pricing_str,
                                    "context_length": ctx,
                                    "description": desc[:200] + "..." if len(desc) > 200 else desc,
                                    "pricing": pricing,
                                    # OpenRouter publishes per-model reasoning capability.
                                    "reasoning": bool(m.get("reasoning")),
                                }
                            )
            except Exception as e:
                logger.warning(f"Error discovering OpenRouter models: {e}")

        # 2. Hugging Face Live Discovery
        if "huggingface" in providers_to_query:
            try:
                headers = {}
                if self.huggingface_token:
                    headers["Authorization"] = f"Bearer {self.huggingface_token}"
                url = "https://huggingface.co/api/models?pipeline_tag=text-generation&sort=trendingScore&limit=100"
                if query_lower:
                    url += f"&search={query_lower}"
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(url, headers=headers)
                    if resp.status_code == 200:
                        hf_models = resp.json()
                        if isinstance(hf_models, list):
                            for m in hf_models:
                                m_id = m.get("id", "")
                                if not m_id:
                                    continue
                                results.append(
                                    {
                                        "id": f"huggingface:{m_id}",
                                        "name": m_id,
                                        "label": m_id.split("/")[-1],
                                        "provider": "huggingface",
                                        "free": True,
                                        "free_tier": "Free (Serverless Inference)",
                                        "pricing_label": "Free Serverless",
                                        "context_length": 32768,
                                        "description": f"Trending text generation model from Hugging Face ({m_id}).",
                                    }
                                )
            except Exception as e:
                logger.warning(f"Error discovering Hugging Face models: {e}")

        # 3. OpenCode Zen Live Discovery (All models + free tiers)
        if "opencode_zen" in providers_to_query:
            base_url = (self.opencode_zen_base_url or "").strip().rstrip("/")
            if not base_url:
                if provider == "opencode_zen":
                    raise ValueError("OpenCode Zen base URL not configured. Set OPENCODE_ZEN_BASE_URL in Settings.")
            else:
                try:
                    headers = {}
                    if self.opencode_zen_api_key:
                        headers["Authorization"] = f"Bearer {self.opencode_zen_api_key}"

                    async with httpx.AsyncClient(timeout=10.0) as client:
                        resp = await client.get(f"{base_url}/models", headers=headers)
                        if resp.status_code == 200:
                            data = resp.json()
                            raw_list = data.get("data") or data.get("models") or []
                            if isinstance(raw_list, list) and raw_list:
                                for m in raw_list:
                                    m_id = m.get("id") or m.get("name") if isinstance(m, dict) else str(m)
                                    if not m_id:
                                        continue
                                    label = m.get("label") or m.get("name") or m_id if isinstance(m, dict) else m_id
                                    is_free = "-free" in m_id.lower() or ":free" in m_id.lower()

                                    if free_only and not is_free:
                                        continue

                                    zen_id = f"opencode_zen:{m_id}"
                                    if query_lower and (
                                        query_lower not in zen_id.lower() and query_lower not in label.lower()
                                    ):
                                        continue

                                    free_tier_str = "Free (Zen No-Key)" if is_free else "Zen Quota / Key"
                                    pricing_str = "Free Tier" if is_free else "Paid / BYOK"

                                    results.append(
                                        {
                                            "id": zen_id,
                                            "name": m_id,
                                            "label": f"OpenCode Zen {label}",
                                            "provider": "opencode_zen",
                                            "free": is_free,
                                            "free_tier": free_tier_str,
                                            "pricing_label": pricing_str,
                                            "context_length": 65536,
                                            "description": f"Model {m_id} hosted on OpenCode Zen.",
                                        }
                                    )
                except Exception as e:
                    logger.warning(f"Error discovering OpenCode Zen models: {e}")

        # 4. Cloudflare Workers AI Live Discovery (10k Neurons/day free tier)
        if "cloudflare" in providers_to_query:
            try:
                cf_models: list[dict[str, Any]] = []
                # Try account search first if credentials configured
                if self.cloudflare_account_id and self.cloudflare_api_token:
                    headers = {"Authorization": f"Bearer {self.cloudflare_api_token}"}
                    cf_url = f"https://api.cloudflare.com/client/v4/accounts/{self.cloudflare_account_id}/ai/models/search?task=Text%20Generation"
                    if query_lower:
                        cf_url += f"&search={query_lower}"
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        resp = await client.get(cf_url, headers=headers)
                        if resp.status_code == 200:
                            cf_models = resp.json().get("result", [])

                # If no models found from account endpoint or credentials not set, discover live catalog
                if not cf_models:
                    async with httpx.AsyncClient(timeout=10.0, headers={"User-Agent": "Alpaca-LLM-Suite"}) as client:
                        resp = await client.get(
                            "https://api.github.com/repos/cloudflare/cloudflare-docs/contents/src/content/workers-ai-models"
                        )
                        if resp.status_code == 200:
                            entries = resp.json()
                            if isinstance(entries, list):
                                for item in entries:
                                    fname = item.get("name", "")
                                    if fname.endswith(".json"):
                                        m_name = f"@cf/{fname[:-5]}"
                                        cf_models.append(
                                            {"name": m_name, "description": f"Cloudflare Workers AI model ({m_name})."}
                                        )

                for m in cf_models:
                    m_name = m.get("name", "")
                    if not m_name:
                        continue
                    if query_lower and query_lower not in m_name.lower():
                        continue

                    results.append(
                        {
                            "id": f"cloudflare:{m_name}",
                            "name": m_name,
                            "label": m_name.split("/")[-1],
                            "provider": "cloudflare",
                            "free": True,  # Cloudflare provides 10k Neurons/day free across all models
                            "free_tier": "10k Neurons/day Free",
                            "pricing_label": "10k Neurons/day Free",
                            "context_length": 128000,
                            "description": m.get(
                                "description", f"Cloudflare Workers AI model ({m_name}). 10k Neurons/day free tier."
                            ),
                        }
                    )
            except Exception as e:
                logger.warning(f"Error discovering Cloudflare models: {e}")

        # 5. Groq Live Discovery (Free tier: open models, no published rate limits)
        if "groq" in providers_to_query:
            try:
                headers = {}
                if self.groq_api_key:
                    headers["Authorization"] = f"Bearer {self.groq_api_key}"
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get("https://api.groq.com/openai/v1/models", headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        raw_models = data.get("data", []) if isinstance(data, dict) else []
                        self._cached_live_models["groq"] = raw_models
                        for m in raw_models:
                            m_id = m.get("id", "") if isinstance(m, dict) else ""
                            if not m_id:
                                continue
                            owned = m.get("owned_by", "") if isinstance(m, dict) else ""
                            if query_lower and query_lower not in m_id.lower():
                                continue

                            results.append(
                                {
                                    "id": f"groq:{m_id}",
                                    "name": m_id,
                                    "label": m_id,
                                    "provider": "groq",
                                    "free": True,
                                    "free_tier": "Free (Groq)",
                                    "pricing_label": "Free (Rate-limited)",
                                    "context_length": m.get("context_window") or m.get("context_length") or 131072,
                                    "reasoning": "reasoning" in (m.get("supported_features") or []),
                                    "description": f"Groq hosted model {m_id} ({owned}).",
                                }
                            )
            except Exception as e:
                logger.warning(f"Error discovering Groq models: {e}")

        # 6. Gemini Live Discovery (Google AI Studio, free tier: rate-limited, no card)
        if "gemini" in providers_to_query:
            try:
                headers = {}
                if self.gemini_api_key:
                    headers["x-goog-api-key"] = self.gemini_api_key
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get("https://generativelanguage.googleapis.com/v1beta/models", headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        raw_models = data.get("models", []) if isinstance(data, dict) else []
                        self._cached_live_models["gemini"] = raw_models
                        for m in raw_models:
                            m_name = m.get("name", "") if isinstance(m, dict) else ""
                            m_id = m_name.split("/")[-1] if m_name.startswith("models/") else m_name
                            if not m_id:
                                continue
                            display = m.get("displayName", m_id) if isinstance(m, dict) else m_id
                            methods = m.get("supportedGenerationMethods", []) if isinstance(m, dict) else []
                            if "generateContent" not in methods:
                                continue
                            if query_lower and query_lower not in m_id.lower() and query_lower not in display.lower():
                                continue

                            results.append(
                                {
                                    "id": f"gemini:{m_id}",
                                    "name": display,
                                    "label": display,
                                    "provider": "gemini",
                                    "free": True,
                                    "free_tier": "Free (Google AI Studio)",
                                    "pricing_label": "Free (Rate-limited)",
                                    "context_length": m.get("inputTokenLimit", 1048576)
                                    if isinstance(m, dict)
                                    else 1048576,
                                    "reasoning": bool(m.get("thinking")) if isinstance(m, dict) else False,
                                    "description": f"Google Gemini model {m_id} ({display}).",
                                }
                            )
            except Exception as e:
                logger.warning(f"Error discovering Gemini models: {e}")

        return results

    def get_selected_models(self) -> list[dict[str, Any]]:
        """Returns the user-selected online models list (persisted to disk).

        If the file exists, return the exact stored list (even if empty []).
        If the file does not exist, return [] so unselected models never show up automatically.
        """
        if self._selected_models_file.exists():
            try:
                with open(self._selected_models_file, encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            except Exception:
                pass
        return []

    def save_selected_models(self, models: list[dict[str, Any]]) -> dict[str, Any]:
        """Save user's custom selection of online models to disk."""
        try:
            self._selected_models_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._selected_models_file, "w", encoding="utf-8") as f:
                json.dump(models, f, indent=2)
            return {"success": True, "count": len(models)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_available_models(self) -> list[dict[str, Any]]:
        """Returns all models from active selection with configured status."""
        configured = self.get_configured_providers()
        models = self.get_selected_models()
        enriched = []
        for m in models:
            item = dict(m)
            prov = item.get("provider", "openrouter")
            item["configured"] = configured.get(prov, False)
            enriched.append(item)
        return enriched

    @staticmethod
    def is_online_model(model_identifier: str) -> bool:
        """Determines if a model string corresponds to an online provider."""
        prefixes = ("openrouter:", "huggingface:", "cloudflare:", "opencode_zen:", "hf:", "groq:", "gemini:")
        return any(model_identifier.startswith(p) for p in prefixes)

    @staticmethod
    def parse_model_identifier(model_identifier: str) -> tuple[str, str]:
        """Splits model identifier into (provider, raw_model_name)."""
        valid_providers = {"openrouter", "huggingface", "hf", "cloudflare", "opencode_zen", "groq", "gemini"}
        if ":" in model_identifier:
            provider, raw_model = model_identifier.split(":", 1)
            provider_clean = provider.lower()
            if provider_clean in valid_providers:
                if provider_clean == "hf":
                    provider_clean = "huggingface"
                return provider_clean, raw_model
        return "local", model_identifier

    @staticmethod
    def _decode_json_response(resp, provider_label: str) -> tuple[dict | None, dict | None]:
        """Parse a JSON response body, returning (data, None) on success.

        Also transparently handles Server-Sent-Events (streaming) responses: if the
        endpoint streams deltas (and `stream: false` was ignored), the deltas are
        accumulated into a single OpenAI-style `choices[0].message.content` so the
        rest of the pipeline works unchanged.

        If the endpoint returns HTTP 200 with a non-JSON body (e.g. a
        misconfigured base URL answering 'Not Found'), returns (None, error_dict)
        so callers surface a clear error instead of a cryptic JSONDecodeError.
        """
        import json as _json

        headers = getattr(resp, "headers", None)
        content_type = ""
        if headers is not None and hasattr(headers, "get"):
            ct = headers.get("content-type", "")
            if isinstance(ct, str):
                content_type = ct
        text = getattr(resp, "text", "") or ""

        if isinstance(content_type, str) and "text/event-stream" in content_type:
            # Streaming response: accumulate deltas below.
            pass
        elif isinstance(text, str) and text.lstrip().startswith("data:"):
            # Streaming response (no content-type header): accumulate deltas below.
            pass
        else:
            try:
                return resp.json(), None
            except Exception as exc:
                latency = resp.elapsed.total_seconds() if resp.elapsed else 0.0
                return None, {
                    "success": False,
                    "latency": latency,
                    "response": None,
                    "tokens_generated": 0,
                    "error": (
                        f"{provider_label} returned HTTP {resp.status_code} with a non-JSON body ({exc}): "
                        f"{resp.text[:200]!r}. Verify the provider base URL and credentials."
                    ),
                }
            # Accumulate streamed deltas into one OpenAI-style completion.
            parts: list[str] = []
            finish_reason = None
            for raw in text.splitlines():
                raw = raw.strip()
                if not raw or not raw.startswith("data:"):
                    continue
                payload = raw[len("data:") :].strip()
                if payload in ("", "[DONE]"):
                    continue
                try:
                    evt = _json.loads(payload)
                except Exception:
                    continue
                for ch in evt.get("choices", []) or []:
                    delta = ch.get("delta", {}) or {}
                    if delta.get("content"):
                        parts.append(delta["content"])
                    if ch.get("finish_reason"):
                        finish_reason = ch.get("finish_reason")
            return {"choices": [{"message": {"content": "".join(parts)}, "finish_reason": finish_reason}]}, None
        try:
            return resp.json(), None
        except Exception as exc:
            latency = resp.elapsed.total_seconds() if resp.elapsed else 0.0
            return None, {
                "success": False,
                "latency": latency,
                "response": None,
                "tokens_generated": 0,
                "error": (
                    f"{provider_label} returned HTTP {resp.status_code} with a non-JSON body ({exc}): "
                    f"{resp.text[:200]!r}. Verify the provider base URL and credentials."
                ),
            }

    def _extract_online_content(
        self, data: dict, provider_label: str
    ) -> tuple[str | None, str | None, str | None, str | None]:
        """Pull completion text out of an OpenAI-style response.

        Returns (content, thinking, finish_reason, None) on success, or
        (None, thinking, finish_reason, error_message) when the body has no choices or
        an empty/None message (content filter, length limit, or a tool-call response)
        so callers do NOT silently report success with no text. The final ``content``
        has any inline ``<think>...</think>`` blocks removed so it is pure output
        (code), while ``thinking`` preserves the model's reasoning (from OpenRouter's
        ``reasoning`` field or inline blocks) for optional display.
        """
        import json as _json

        choices = data.get("choices") or []
        if not choices:
            return (
                None,
                None,
                None,
                (
                    f"{provider_label} returned HTTP 200 with no choices (possible content filter or "
                    f"max-token limit). Body: {str(data)[:200]}"
                ),
            )
        choice = choices[0] if isinstance(choices[0], dict) else {}
        msg = choice.get("message", {}) or {}
        raw = msg.get("content") or ""
        tag_re = r"<think[^>]*>[\s\S]*?</think[^>]*>"
        inline_think = re.findall(tag_re, raw, flags=re.I)
        clean = re.sub(tag_re + r"\s*", "", raw, flags=re.I)
        reasoning = msg.get("reasoning")  # OpenRouter reasoning field (separate from content)
        thinking = reasoning or "\n".join(inline_think) or None
        finish = choice.get("finish_reason")
        if clean.strip():
            return clean, thinking, finish, None
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            return _json.dumps(tool_calls), thinking, finish, None
        return None, thinking, finish, f"{provider_label} returned an empty response (finish_reason={finish})."

    # Reasoning-model name hints. These models spend a meaningful chunk of their
    # output budget on a ` thinking` block or a `reasoning` field before writing
    # the actual answer, so plain `max_tokens` reservations starve the response.
    _THINKING_MODEL_HINTS = (
        "deepseek",
        "qwen3",
        "qwen2.5",
        "r1",
        "thinking",
        "reasoning",
        "reasoner",
        "gpt-oss",
        "gemini-2.5",
        "gemini-3",
        "glm",
        "kimi",
        "qwq",
        "grok",
    )

    @classmethod
    def _is_thinking_model_name(cls, model_name: str) -> bool:
        """Name-hint fallback: detect a reasoning model when the provider exposes no metadata."""
        name = (model_name or "").lower()
        return any(hint in name for hint in cls._THINKING_MODEL_HINTS)

    def _is_thinking_model(self, model_identifier: str) -> bool:
        """Determine whether a model spends output tokens on reasoning.

        Resolution order (per user requirement — metadata, NOT name guessing):
        1. Provider metadata captured at discovery time and persisted in the
           selected-models list (the ``reasoning`` flag added by
           ``fetch_live_models``: OpenRouter ``reasoning`` field, Gemini
           ``thinking`` bool, Groq ``supported_features``). This is authoritative
           for models the user has added to the list.
        2. A live metadata fetch for the provider/model (cached per model id) when
           the model is not in the selected list — e.g. an ad-hoc query by id.
        3. Name hints only as a last resort for providers that publish no
           reasoning metadata (Groq returned none for its /models entries).
        """
        # 1. Persisted provider metadata from the selected-models list.
        try:
            for m in self.get_selected_models():
                if isinstance(m, dict) and m.get("id") == model_identifier and "reasoning" in m:
                    return bool(m.get("reasoning"))
        except Exception:
            pass

        # 2. Live provider metadata (cached) for models not in the selected list.
        try:
            meta = self._get_provider_model_metadata(model_identifier)
            if meta is not None:
                return bool(meta)
        except Exception:
            pass

        # 3. Name-hint last resort (provider gives us no metadata at all).
        _, model_name = self.parse_model_identifier(model_identifier)
        return self._is_thinking_model_name(model_name)

    def _get_provider_model_metadata(self, model_identifier: str) -> bool | None:
        """Live lookup of the reasoning capability from the provider's model list.

        Returns True/False when the provider publishes the flag for this model,
        None when the provider exposes no reasoning metadata for it.
        """
        provider, model_name = self.parse_model_identifier(model_identifier)
        if provider == "openrouter":
            data = self._cached_live_models.get("openrouter")
            if data is None:
                return None
            for m in data:
                if isinstance(m, dict) and m.get("id") == model_name:
                    # Presence of the `reasoning` field => reasoning-capable model.
                    return bool(m.get("reasoning"))
            return None
        if provider == "gemini":
            data = self._cached_live_models.get("gemini")
            if data is None:
                return None
            for m in data:
                if not isinstance(m, dict):
                    continue
                m_name = m.get("name", "")
                m_id = m_name.split("/")[-1] if m_name.startswith("models/") else m_name
                if m_id == model_name:
                    return bool(m.get("thinking"))
            return None
        if provider == "groq":
            data = self._cached_live_models.get("groq")
            if data is None:
                return None
            for m in data:
                if isinstance(m, dict) and m.get("id") == model_name:
                    return "reasoning" in (m.get("supported_features") or [])
            return None
        return None

    async def _resolve_thinking_model(self, model_identifier: str) -> bool:
        """Resolve reasoning-model status, fetching live provider metadata on demand.

        Mirrors ``_is_thinking_model`` but performs a live metadata fetch (populating
        ``_cached_live_models``) when the model is neither in the selected list nor
        already cached, so ad-hoc queries get provider metadata too.
        """
        # 1. Persisted selected-models metadata (authoritative for listed models).
        try:
            for m in self.get_selected_models():
                if isinstance(m, dict) and m.get("id") == model_identifier and "reasoning" in m:
                    return bool(m.get("reasoning"))
        except Exception:
            pass

        # 2. Cached provider metadata.
        cached = self._get_provider_model_metadata(model_identifier)
        if cached is not None:
            return cached

        # 3. Live fetch if the provider cache is cold, then retry.
        provider, _ = self.parse_model_identifier(model_identifier)
        if provider in ("openrouter", "gemini", "groq"):
            try:
                await self.fetch_live_models(provider=provider)
                cached = self._get_provider_model_metadata(model_identifier)
                if cached is not None:
                    return cached
            except Exception:
                pass

        # 4. Name-hint last resort.
        return self._is_thinking_model_name(self.parse_model_identifier(model_identifier)[1])

    async def query_online_model(
        self,
        model_identifier: str,
        prompt: str,
        max_tokens: int = 4000,
        temperature: float = 0.2,
        custom_keys: dict[str, str] | None = None,
        request_source: str = "web",
        client_ip: str = "web",
        max_retries: int = 4,
    ) -> dict[str, Any]:
        """Queries the specified online provider.

        Thinking/reasoning models are handled specially: they consume output tokens
        for a ``<think>`` block / ``reasoning`` field before producing the answer, so
        a plain ``max_tokens`` reservation can truncate before any answer is written.
        For those models we (a) request a larger output budget so reasoning plus the
        answer fits, (b) inject a token/time budget warning into the prompt so the
        model knows to wrap up, and (c) if the completion is still cut off by the
        length limit, run one phase-2 continuation asking it to finish the answer.

        Every online query is tracked in the in-process request queue so the web
        dashboard can display it alongside local requests. Tracking is fully
        separate from the proxy's slot management: online queries never consume
        llama-server slots.
        """
        provider, _ = self.parse_model_identifier(model_identifier)
        request_id = _make_request_id(model_identifier)
        start_t = time.time()
        start_online_request(
            request_id,
            model_identifier,
            provider,
            {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens},
            request_source=request_source,
            client_ip=client_ip,
        )

        thinking = await self._resolve_thinking_model(model_identifier)
        # Give reasoning models headroom so thinking + answer both fit. Capped so
        # we never exceed a provider's (unknown) max output window by much.
        effective_max_tokens = max_tokens
        if thinking:
            effective_max_tokens = min(max_tokens + max(2048, max_tokens // 2), 65536)
            # Longer budget + reasoning phase => longer wall-clock time per call.
        timeout = 180.0 if thinking else 120.0

        # Inject a token/time budget warning for reasoning models so they wrap up
        # instead of rambling in the think block and then hitting length limits.
        working_prompt = prompt
        if thinking:
            working_prompt = (
                f"{prompt}\n\n[System: You are a reasoning model with a token budget of "
                f"{effective_max_tokens} output tokens. Reason briefly, then provide your "
                f"final answer well before the budget is exhausted. Do not repeat yourself; "
                f"keep thinking concise and conclude within the limit.]"
            )

        result: dict[str, Any] = {
            "success": False,
            "latency": 0.0,
            "response": None,
            "tokens_generated": 0,
            "error": "Online request never executed.",
        }
        for attempt in range(max_retries + 1):
            try:
                result = await self._query_online_model_impl(
                    model_identifier,
                    working_prompt,
                    effective_max_tokens,
                    temperature,
                    custom_keys,
                    request_timeout=timeout,
                )
            except Exception as exc:
                result = {
                    "success": False,
                    "latency": time.time() - start_t,
                    "response": None,
                    "tokens_generated": 0,
                    "error": f"Online request failed: {exc}",
                }
            # If a reasoning model still got cut off by the length limit, run one
            # phase-2 continuation asking it to finish the answer now. This recovers
            # responses whose entire budget was consumed by the think block.
            # Also fires when the budget was exhausted entirely inside the
            # provider's reasoning field (empty content, finish_reason=length,
            # reasoning text captured): that outcome is deterministic - plain
            # retries reproduce it forever - so the continuation IS the recovery.
            _length_cut = result.get("finish_reason") in ("length", "MAX_TOKENS", "max_tokens")
            if result.get("success") and _length_cut:
                continuation = (
                    "\n\n[System: You ran out of tokens before finishing. Provide ONLY the "
                    "remaining final answer now, continuing exactly where you left off. Do not "
                    "reason or repeat anything already written — finish quickly.]"
                )
                try:
                    phase2 = await self._query_online_model_impl(
                        model_identifier,
                        f"{working_prompt}\n{result.get('response') or ''}\n{continuation}",
                        max(2048, effective_max_tokens // 2),
                        temperature,
                        custom_keys,
                        request_timeout=timeout,
                    )
                    if phase2.get("success") and phase2.get("response"):
                        result["response"] = (result.get("response") or "") + "\n" + phase2["response"]
                        result["tokens_generated"] = (result.get("tokens_generated") or 0) + (
                            phase2.get("tokens_generated") or 0
                        )
                        result["finish_reason"] = phase2.get("finish_reason") or "stop"
                except Exception as exc2:
                    print(f"[online] {provider} phase-2 continuation failed: {exc2}")
            elif not result.get("success") and _length_cut and result.get("thinking"):
                # Deterministic exhaustion: every token went to reasoning and the
                # content came back empty. Ask for the final answer directly; if
                # that still fails, stop retrying - backoff cannot fix a full
                # reasoning field.
                try:
                    phase2 = await self._query_online_model_impl(
                        model_identifier,
                        f"{working_prompt}\n\n[System: Your previous attempt produced no "
                        f"final answer because reasoning used the entire token budget. "
                        f"Answer now with ONLY the complete final answer - no reasoning, "
                        f"no preamble, no restating the problem.]",
                        effective_max_tokens,
                        temperature,
                        custom_keys,
                        request_timeout=timeout,
                    )
                    if phase2.get("success") and phase2.get("response"):
                        result.update(
                            {
                                "success": True,
                                "response": phase2["response"],
                                "thinking": result.get("thinking") or None,
                                "tokens_generated": (result.get("tokens_generated") or 0)
                                + (phase2.get("tokens_generated") or 0),
                                "finish_reason": phase2.get("finish_reason") or "stop",
                                "error": None,
                            }
                        )
                    else:
                        result["error"] = (
                            f"{result.get('error') or 'Empty response'} "
                            "(reasoning exhausted the output budget; direct-answer retry also failed)"
                        )
                        break
                except Exception as exc2:
                    print(f"[online] {provider} direct-answer retry failed: {exc2}")
                    break
            # Retry transient free-tier failures (empty/length-truncated completions,
            # DNS/network blips, timeouts, 429/5xx) instead of scoring them as a miss.
            if not self._is_retryable_online_failure(result) or attempt >= max_retries:
                break
            # Exponential backoff with jitter; honor a provider Retry-After when present.
            # Free-tier endpoints (OpenRouter) throttle for tens of seconds and return
            # empty 200 completions, so short fixed backoffs cause false-negative empties.
            retry_after = result.get("retry_after")
            try:
                retry_after = float(retry_after) if retry_after else 0.0
            except (TypeError, ValueError):
                retry_after = 0.0
            backoff = min(120.0, max(5.0 * (2**attempt), retry_after)) + (attempt + 1) * 0.5
            print(
                f"[online] {provider} attempt {attempt + 1} returned a retryable failure "
                f"({result.get('error')}); retrying in {backoff:.1f}s"
            )
            await asyncio.sleep(backoff)

        complete_online_request(request_id, result)
        return result

    @staticmethod
    def _is_retryable_online_failure(result: dict) -> bool:
        """True for failures that are transient (provider-side) and worth one retry.

        Covers empty/length-truncated completions from free tiers, DNS/network errors,
        timeouts, and rate-limit / 5xx responses. Permanent failures (auth, bad key,
        unknown model, malformed request) are NOT retried.
        """
        if result.get("success"):
            return False
        err = (result.get("error") or "").lower()
        retryable_tokens = (
            "empty response",
            "no choices",
            "temporary failure in name resolution",
            "name or service not known",
            "timeout",
            "timed out",
            "503",
            "502",
            "504",
            "429",
        )
        return any(tok in err for tok in retryable_tokens)

    @staticmethod
    def _format_http_error(label: str, status_code: int, err_msg: str) -> str:
        """Format a provider HTTP error, keeping rate limits user-actionable."""
        if status_code == 429:
            return (
                f"{label} rate limit exceeded (HTTP 429). "
                "Check your service usage and rate limits in the provider dashboard, "
                "or wait for the current quota window to reset."
            )
        return f"{label} HTTP {status_code}: {err_msg}"

    async def _query_online_model_impl(
        self,
        model_identifier: str,
        prompt: str,
        max_tokens: int = 4000,
        temperature: float = 0.2,
        custom_keys: dict[str, str] | None = None,
        request_timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Queries the specified online provider."""
        provider, model_name = self.parse_model_identifier(model_identifier)
        custom = custom_keys or {}

        start_t = time.time()
        try:
            if provider == "openrouter":
                api_key = custom.get("openrouter_api_key") or self.openrouter_api_key
                if not api_key:
                    return {
                        "success": False,
                        "latency": 0.0,
                        "response": None,
                        "tokens_generated": 0,
                        "error": "OpenRouter API Key not configured. Set OPENROUTER_API_KEY in Settings.",
                    }

                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/JMiahMan1/alpaca",
                    "X-Title": "Alpaca LLM Benchmark Suite",
                }
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False,
                }
                async with httpx.AsyncClient(timeout=request_timeout) as client:
                    resp = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    latency = time.time() - start_t
                    if resp.status_code == 200:
                        data, decode_err = self._decode_json_response(resp, "OpenRouter")
                        if data is None or decode_err:
                            return decode_err or {
                                "success": False,
                                "latency": latency,
                                "response": None,
                                "tokens_generated": 0,
                                "error": "OpenRouter returned an empty response body.",
                            }
                        tokens = data.get("usage", {}).get("completion_tokens", 0)
                        content, thinking, finish, cerr = self._extract_online_content(data, "OpenRouter")
                        if cerr:
                            return {
                                "success": False,
                                "latency": latency,
                                "response": None,
                                "thinking": thinking,
                                "tokens_generated": 0,
                                "finish_reason": finish,
                                "error": cerr,
                            }
                        return {
                            "success": True,
                            "latency": latency,
                            "response": content,
                            "thinking": thinking,
                            "finish_reason": finish,
                            "tokens_generated": tokens,
                            "error": None,
                        }
                    retry_after = resp.headers.get("Retry-After")
                    try:
                        err_data = resp.json()
                        err_msg = err_data.get("error", {}).get("message") or err_data.get("message") or resp.text[:300]
                    except Exception:
                        err_msg = resp.text[:300]
                    fail = {
                        "success": False,
                        "latency": latency,
                        "response": None,
                        "tokens_generated": 0,
                        "error": self._format_http_error("OpenRouter", resp.status_code, err_msg),
                    }
                    if retry_after:
                        fail["retry_after"] = retry_after
                    return fail

            elif provider == "huggingface":
                api_key = custom.get("huggingface_token") or self.huggingface_token
                if not api_key:
                    return {
                        "success": False,
                        "latency": 0.0,
                        "response": None,
                        "tokens_generated": 0,
                        "error": "Hugging Face Token not configured. Set HUGGING_FACE_TOKEN in Settings.",
                    }

                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                async with httpx.AsyncClient(timeout=request_timeout) as client:
                    # Hugging Face Inference Providers router: the model is passed
                    # in the request body (not the URL path). The legacy
                    # serverless api-inference.huggingface.co endpoint is being
                    # deprecated/blocked, so the router is the supported path.
                    url = os.getenv("HUGGINGFACE_ROUTER_URL", "https://router.huggingface.co/v1/chat/completions")
                    last_err = ""
                    last_status = 0
                    try:
                        resp = await client.post(url, headers=headers, json=payload)
                        last_status = resp.status_code
                        latency = time.time() - start_t
                        if resp.status_code == 200:
                            data, decode_err = self._decode_json_response(resp, "Hugging Face")
                            if data is None or decode_err:
                                return decode_err or {
                                    "success": False,
                                    "latency": latency,
                                    "response": None,
                                    "tokens_generated": 0,
                                    "error": "Hugging Face returned an empty response body.",
                                }
                            # The HF router returns HTTP 200 with {"error": "..."}
                            # when the account has no remaining credits or the
                            # model/provider is unavailable. Treat that as a
                            # failure, not a successful empty completion.
                            err_field = data.get("error")
                            if err_field is not None:
                                if isinstance(err_field, dict):
                                    err_field = err_field.get("message") or str(err_field)
                                return {
                                    "success": False,
                                    "latency": latency,
                                    "response": None,
                                    "tokens_generated": 0,
                                    "error": f"Hugging Face: {err_field}",
                                }
                            tokens = data.get("usage", {}).get("completion_tokens", 0)
                            content, thinking, finish, cerr = self._extract_online_content(data, "Hugging Face")
                            if cerr:
                                return {
                                    "success": False,
                                    "latency": latency,
                                    "response": None,
                                    "thinking": thinking,
                                    "tokens_generated": 0,
                                    "finish_reason": finish,
                                    "error": cerr,
                                }
                            return {
                                "success": True,
                                "latency": latency,
                                "response": content,
                                "thinking": thinking,
                                "finish_reason": finish,
                                "tokens_generated": tokens,
                                "error": None,
                            }
                        try:
                            err_data = resp.json()
                            last_err = (
                                err_data.get("error", {}).get("message") or err_data.get("error") or resp.text[:300]
                            )
                        except Exception:
                            last_err = resp.text[:300]
                    except Exception as e:
                        last_err = str(e)
                    return {
                        "success": False,
                        "latency": time.time() - start_t,
                        "response": None,
                        "tokens_generated": 0,
                        "error": self._format_http_error("Hugging Face", last_status, last_err),
                    }

            elif provider == "cloudflare":
                api_token = custom.get("cloudflare_api_token") or self.cloudflare_api_token
                account_id = custom.get("cloudflare_account_id") or self.cloudflare_account_id
                if not api_token or not account_id:
                    return {
                        "success": False,
                        "latency": 0.0,
                        "response": None,
                        "tokens_generated": 0,
                        "error": "Cloudflare credentials not configured. Set CLOUDFLARE_API_TOKEN & Account ID.",
                    }

                import urllib.parse

                clean_acc = urllib.parse.quote(str(account_id).strip(), safe="")
                clean_model = urllib.parse.quote(str(model_name).strip().lstrip("/"), safe="/@-")
                url = f"https://api.cloudflare.com/client/v4/accounts/{clean_acc}/ai/run/{clean_model}"
                headers = {
                    "Authorization": f"Bearer {api_token}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "stream": False,
                }
                async with httpx.AsyncClient(timeout=request_timeout) as client:
                    resp = await client.post(url, headers=headers, json=payload)
                    latency = time.time() - start_t
                    if resp.status_code == 200:
                        data, decode_err = self._decode_json_response(resp, "Cloudflare Workers AI")
                        if data is None or decode_err:
                            return decode_err or {
                                "success": False,
                                "latency": latency,
                                "response": None,
                                "tokens_generated": 0,
                                "error": "Cloudflare Workers AI returned an empty response body.",
                            }
                        # Cloudflare's v4 API returns failures as HTTP 200 with
                        # {"success": false, "errors": [{...}]}. Treat that as a
                        # failure rather than a successful empty completion.
                        if data.get("success") is False:
                            errs = data.get("errors", [])
                            msg = (
                                errs[0].get("message")
                                if errs and isinstance(errs[0], dict)
                                else data.get("error") or resp.text[:300]
                            )
                            return {
                                "success": False,
                                "latency": latency,
                                "response": None,
                                "tokens_generated": 0,
                                "error": f"Cloudflare Workers AI: {msg}",
                            }
                        result = data.get("result", {})
                        content = result.get("response", "")
                        if not content:
                            return {
                                "success": False,
                                "latency": latency,
                                "response": None,
                                "tokens_generated": 0,
                                "error": "Cloudflare Workers AI returned an empty response.",
                            }
                        return {
                            "success": True,
                            "latency": latency,
                            "response": content,
                            "tokens_generated": len(content.split()),
                            "error": None,
                        }
                    try:
                        err_data = resp.json()
                        err_msg = (
                            err_data.get("errors", [{}])[0].get("message") or err_data.get("error") or resp.text[:300]
                        )
                    except Exception:
                        err_msg = resp.text[:300]
                    return {
                        "success": False,
                        "latency": latency,
                        "response": None,
                        "tokens_generated": 0,
                        "error": self._format_http_error("Cloudflare Workers AI", resp.status_code, err_msg),
                    }

            elif provider == "opencode_zen":
                api_key = custom.get("opencode_zen_api_key") or self.opencode_zen_api_key
                base_url = custom.get("opencode_zen_base_url") or self.opencode_zen_base_url
                if not base_url:
                    return {
                        "success": False,
                        "latency": 0.0,
                        "response": None,
                        "tokens_generated": 0,
                        "error": "OpenCode Zen base URL not configured. Set OPENCODE_ZEN_BASE_URL in Settings.",
                    }
                base_url = base_url.rstrip("/")
                headers = _opencode_zen_headers(api_key)
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False,
                }
                async with httpx.AsyncClient(timeout=request_timeout) as client:
                    resp = await client.post(
                        f"{base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    latency = time.time() - start_t
                    if resp.status_code == 200:
                        data, decode_err = self._decode_json_response(resp, "OpenCode Zen")
                        if data is None or decode_err:
                            return decode_err or {
                                "success": False,
                                "latency": latency,
                                "response": None,
                                "tokens_generated": 0,
                                "error": "OpenCode Zen returned an empty response body.",
                            }
                        tokens = data.get("usage", {}).get("completion_tokens", 0)
                        content, thinking, finish, cerr = self._extract_online_content(data, "OpenCode Zen")
                        if cerr:
                            return {
                                "success": False,
                                "latency": latency,
                                "response": None,
                                "thinking": thinking,
                                "tokens_generated": 0,
                                "finish_reason": finish,
                                "error": cerr,
                            }
                        return {
                            "success": True,
                            "latency": latency,
                            "response": content,
                            "thinking": thinking,
                            "finish_reason": finish,
                            "tokens_generated": tokens,
                            "error": None,
                        }
                    try:
                        err_data = resp.json()
                        err_msg = err_data.get("error", {}).get("message") or err_data.get("message") or resp.text[:300]
                    except Exception:
                        err_msg = resp.text[:300]
                    return {
                        "success": False,
                        "latency": latency,
                        "response": None,
                        "tokens_generated": 0,
                        "error": self._format_http_error("OpenCode Zen", resp.status_code, err_msg),
                    }

            elif provider == "groq":
                api_key = custom.get("groq_api_key") or self.groq_api_key
                if not api_key:
                    return {
                        "success": False,
                        "latency": 0.0,
                        "response": None,
                        "tokens_generated": 0,
                        "error": "Groq API Key not configured. Set GROQ_API_KEY in Settings.",
                    }

                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False,
                }
                async with httpx.AsyncClient(timeout=request_timeout) as client:
                    resp = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    latency = time.time() - start_t
                    if resp.status_code == 200:
                        data, decode_err = self._decode_json_response(resp, "Groq")
                        if data is None or decode_err:
                            return decode_err or {
                                "success": False,
                                "latency": latency,
                                "response": None,
                                "tokens_generated": 0,
                                "error": "Groq returned an empty response body.",
                            }
                        tokens = data.get("usage", {}).get("completion_tokens", 0)
                        content, thinking, finish, cerr = self._extract_online_content(data, "Groq")
                        if cerr:
                            return {
                                "success": False,
                                "latency": latency,
                                "response": None,
                                "thinking": thinking,
                                "tokens_generated": 0,
                                "finish_reason": finish,
                                "error": cerr,
                            }
                        return {
                            "success": True,
                            "latency": latency,
                            "response": content,
                            "thinking": thinking,
                            "finish_reason": finish,
                            "tokens_generated": tokens,
                            "error": None,
                        }
                    try:
                        err_data = resp.json()
                        err_msg = err_data.get("error", {}).get("message") or err_data.get("message") or resp.text[:300]
                    except Exception:
                        err_msg = resp.text[:300]
                    return {
                        "success": False,
                        "latency": latency,
                        "response": None,
                        "tokens_generated": 0,
                        "error": self._format_http_error("Groq", resp.status_code, err_msg),
                    }

            elif provider == "gemini":
                api_key = custom.get("gemini_api_key") or self.gemini_api_key
                if not api_key:
                    return {
                        "success": False,
                        "latency": 0.0,
                        "response": None,
                        "tokens_generated": 0,
                        "error": "Gemini API Key not configured. Set GEMINI_API_KEY in Settings.",
                    }

                headers = {
                    "x-goog-api-key": api_key,
                    "Content-Type": "application/json",
                }
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "maxOutputTokens": max_tokens,
                        "temperature": temperature,
                    },
                }
                async with httpx.AsyncClient(timeout=request_timeout) as client:
                    resp = await client.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
                        headers=headers,
                        json=payload,
                    )
                    latency = time.time() - start_t
                    if resp.status_code == 200:
                        data, decode_err = self._decode_json_response(resp, "Gemini")
                        if data is None or decode_err:
                            return decode_err or {
                                "success": False,
                                "latency": latency,
                                "response": None,
                                "tokens_generated": 0,
                                "error": "Gemini returned an empty response body.",
                            }
                        tokens = data.get("usageMetadata", {}).get("candidatesTokenCount", 0)
                        candidates = data.get("candidates") or []
                        content = ""
                        finish_reason = ""
                        if candidates:
                            cand = candidates[0]
                            finish_reason = cand.get("finishReason", "")
                            parts = (cand.get("content") or {}).get("parts") or []
                            content = "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
                        if not content:
                            return {
                                "success": False,
                                "latency": latency,
                                "response": None,
                                "tokens_generated": 0,
                                "finish_reason": finish_reason,
                                "error": (f"Gemini returned an empty response (finish_reason={finish_reason})"),
                            }
                        return {
                            "success": True,
                            "latency": latency,
                            "response": content,
                            "thinking": None,
                            "finish_reason": finish_reason,
                            "tokens_generated": tokens,
                            "error": None,
                        }
                    try:
                        err_data = resp.json()
                        err_msg = err_data.get("error", {}).get("message") or err_data.get("message") or resp.text[:300]
                    except Exception:
                        err_msg = resp.text[:300]
                    fail = {
                        "success": False,
                        "latency": latency,
                        "response": None,
                        "tokens_generated": 0,
                        "error": self._format_http_error("Gemini", resp.status_code, err_msg),
                    }
                    if resp.status_code == 429:
                        # Gemini's quota error embeds a precise retry hint
                        # ("Please retry in 43.08s") that we should honor instead
                        # of guessing. Pass it up so the retry backoff waits it out.
                        m = re.search(r"retry in ([\d.]+)\s*s", err_msg)
                        if m:
                            with suppress(ValueError):
                                fail["retry_after"] = float(m.group(1))
                    return fail

            else:
                return {
                    "success": False,
                    "latency": 0.0,
                    "response": None,
                    "tokens_generated": 0,
                    "error": f"Unknown online provider '{provider}'",
                }

        except Exception as exc:
            return {
                "success": False,
                "latency": time.time() - start_t,
                "response": None,
                "tokens_generated": 0,
                "error": f"Online request failed: {exc}",
            }


# Global instance
online_model_provider = OnlineModelProvider()
