#!/usr/bin/env python3
"""Shared context-window awareness for all benchmark suites.

Every suite (general, SharedLLM, multi-step) must fit its prompt plus
generation budget inside the backend's live context window. On hosts where
`num_predict` plus the prompt exceeds `n_ctx`, llama-server silently
truncates generation (`stop processing: n_tokens = N, truncated = 1`),
which surfaces downstream as empty or cut-off benchmark responses.

This module is the single source of truth for:
- resolving the live context window via the proxy's /admin/runtime endpoint
  (the component that launches llama-server with its configured ctx-size),
- warming an unloaded model so the runtime entry exists,
- estimating prompt token counts,
- clamping a requested generation budget to the available headroom.

Resolution failure is loud (RuntimeError) — no silent defaults.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable

import httpx

_CTX_TIMEOUT = 10.0

# How long resolve_context_window waits for a warm-up-triggered model load to
# finish before giving up. Configurable via CONTEXT_RESOLVE_TIMEOUT_S because
# cold-load times vary from seconds (small models, page-cached weights) to
# minutes (large models spinning up on GPU).
RESOLVE_TIMEOUT_S = float(os.getenv("CONTEXT_RESOLVE_TIMEOUT_S", "") or 300)
POLL_INTERVAL_S = 2.0


def _proxy_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = dict(extra or {})
    key = os.getenv("ALPACA_API_KEY", "").strip()
    if key:
        headers.setdefault("Authorization", f"Bearer {key}")
        headers.setdefault("X-API-Key", key)
    return headers


def estimate_prompt_tokens(messages: list[dict[str, str]]) -> int:
    """Rough token estimate for a chat transcript, plus per-message overhead.

    Dense JavaScript/HTML tokenizes closer to ~3 chars/token than the
    prose-typical ~4, so use the conservative divisor: underestimating the
    prompt makes the clamp hand out generation headroom the backend does not
    actually have (observed as silent truncation on 8K hosts).
    """
    return sum(len(m.get("content", "")) for m in messages) // 3 + 16 * len(messages)


async def scan_runtime_ctx(
    client: httpx.AsyncClient,
    model: str,
    candidates: tuple[str, ...],
    proxy_urls: list[str],
) -> int | None:
    """Look up ctx-size for the model across proxy URLs; None if absent."""
    headers = _proxy_headers()
    for base_url in proxy_urls:
        try:
            resp = await client.get(f"{base_url}/admin/runtime", headers=headers)
            if resp.status_code != 200:
                continue
            for loaded in resp.json().get("loaded_models", []):
                ident = loaded.get("backend_model") or loaded.get("name") or ""
                if ident not in candidates and loaded.get("name") != model:
                    continue
                raw = (loaded.get("running_settings") or {}).get("ctx-size")
                ctx = int(str(raw).strip())
                if ctx > 0:
                    return ctx
        except Exception:
            continue
    return None


async def warm_model(client: httpx.AsyncClient, model: str, proxy_urls: list[str], source_tag: str) -> None:
    """Send a minimal streamed chat through each proxy so it loads the model."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word OK."}],
        "stream": True,
        "think": False,
        "options": {"num_predict": 1},
    }
    headers = _proxy_headers({"X-Request-Source": source_tag})
    for base_url in proxy_urls:
        try:
            async with client.stream("POST", f"{base_url}/api/chat", json=payload, headers=headers) as resp:
                if resp.status_code == 200:
                    await resp.aread()
                    return
        except Exception:
            continue


async def resolve_context_window(
    model: str,
    proxy_urls: list[str],
    cache: dict[str, int] | None = None,
    source_tag: str = "benchmark",
) -> int:
    """Resolve the live context window for `model` from the proxy runtime.

    The proxy exposes each loaded model's launch flags under /admin/runtime
    (`loaded_models[].running_settings["ctx-size"]`). If the model is not
    currently loaded (keep-alive expired or fresh boot), a warm-up request is
    sent once — which makes the proxy start loading the model — and the
    runtime is then polled until the entry appears (model loads can take from
    seconds to minutes). Failure raises RuntimeError loudly — budgets computed
    from a guessed window would reintroduce silent truncation.
    """
    if cache is not None and model in cache:
        return cache[model]

    candidates = (model, f"{model}--latest")
    deadline = time.monotonic() + RESOLVE_TIMEOUT_S
    warmed = False
    ctx = None
    async with httpx.AsyncClient(timeout=_CTX_TIMEOUT) as client:
        while True:
            ctx = await scan_runtime_ctx(client, model, candidates, proxy_urls)
            if ctx:
                break
            if not warmed:
                # One warm-up is enough: the proxy queues it and loads the
                # model. Its result does not matter; the polling below waits
                # for the load to finish.
                await warm_model(client, model, proxy_urls, source_tag)
                warmed = True
                continue
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(POLL_INTERVAL_S)
    if not ctx:
        raise RuntimeError(
            f"Cannot determine context window for '{model}': no /admin/runtime entry with a "
            f"valid ctx-size was reachable on {proxy_urls} within {RESOLVE_TIMEOUT_S:.0f}s of "
            "warm-up polling. Verify the alpaca-proxy is running and can load this model."
        )
    if cache is not None:
        cache[model] = ctx
    return ctx


def turn_budget(ctx: int, estimated_prompt_tokens: int, requested: int, reserve: int = 128) -> int:
    """Clamp a requested generation budget so prompt + generation fit the window."""
    avail = ctx - estimated_prompt_tokens - reserve
    return max(0, min(requested, avail))


def compact_messages(
    messages: list[dict[str, str]],
    ctx: int,
    manifest_extractor: Callable[[str], str] | None = None,
) -> tuple[list[dict[str, str]], bool]:
    """Shrink a transcript when it crowds out generation headroom.

    The most recent assistant message and the most recent user message are
    preserved verbatim (they define what must be produced next). Older
    assistant messages collapse to their extracted state via
    `manifest_extractor` when provided; older user messages collapse to a
    short summary of their contract.
    """
    limit = int(ctx * 0.55)
    if estimate_prompt_tokens(messages) <= limit:
        return messages, False

    user_positions = [i for i, m in enumerate(messages) if m["role"] == "user"]
    last_user = user_positions[-1] if user_positions else None
    assistant_positions = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    keep_assistant = set(assistant_positions[-1:])

    compacted = []
    changed = False
    for i, m in enumerate(messages):
        if i == last_user or i in keep_assistant or len(m.get("content", "")) <= 400:
            compacted.append(m)
            continue
        if m["role"] == "assistant":
            manifests = manifest_extractor(m["content"]) if manifest_extractor else ""
            if manifests:
                note = (
                    "[System note: your full code from this earlier turn is omitted to fit the "
                    f"context window. The authoritative state you must preserve:\n{manifests}\n]"
                )
            else:
                note = "[System note: your full output from this earlier turn is omitted to fit the context window.]"
        else:
            head = m["content"][:400].strip()
            note = (
                "[System note: this earlier instruction is summarized to fit the context window. "
                f"In brief it required:\n{head}\n... Its completed work carries forward.]"
            )
        compacted.append({"role": m["role"], "content": note})
        changed = True
    return compacted, changed
