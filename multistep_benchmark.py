#!/usr/bin/env python3
"""
Multi-step agentic benchmark harness.

The existing suites (llm_benchmark_suite.py, web/shared_llm_benchmark.py) are
single-turn: exactly one user message per test. This module adds a *multi-turn*
harness for long agentic workflows: each workflow is an ordered list of steps,
and every step is sent together with the accumulated conversation so the model
must retain, extend, and refactor its own previous output across turns. Step
gates (static content checks) run between turns, and the final artifact is
executed in the headless sandbox (grade_code ui=True) like other UI benchmarks.

The first workflow recreates the classic Macintosh game "Glider"
(John Calhoun, Soft Dorothy Software, 1988) as a modern HTML5 canvas game
built across four conversational turns, with easter eggs honoring the
original game throughout the house.

Results follow SharedLLM-suite conventions and are written to
data/multistep_benchmarks/.
"""

import asyncio
import contextlib
import json
import os
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

# Shared streaming/config infrastructure lives in the SharedLLM suite; reuse it
# so stream parsing, thinking-strip rules, timeouts, and temperature resolution
# behave identically across suites.
sys.path.append(str(Path(__file__).resolve().parent))
import context_awareness
from web.shared_llm_benchmark import (
    _STREAM_TIMEOUT,
    _THINKING_PATTERNS,
    _model_temperature,
    _read_chat_stream,
    _read_generate_stream,
)

online_model_provider: Any = None
try:
    from online_providers import online_model_provider
except Exception:  # pragma: no cover - optional dependency
    online_model_provider = None

extract_clean_code: Any = None
grade_code: Any = None
with contextlib.suppress(Exception):  # sandbox optional (CI containers)
    from sandbox_exec import extract_clean_code, grade_code

# Zero-token turns are transient backend failures (e.g. cgroup OOM kill
# respawning llama-server mid-generation). Retry a couple of times with a
# short backoff before recording the turn as failed.
_EMPTY_TURN_RETRIES = 2
_EMPTY_TURN_RETRY_DELAY_S = 15
_UI_RENDER_RETRIES = 2
_UI_RENDER_RETRY_DELAY_S = 5


def strip_thinking(text: str | None) -> str:
    """Strip <think> blocks and reasoning-prose headers (same rules as SharedLLM suite)."""
    if not text:
        return ""
    cleaned = text
    for pattern in _THINKING_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned


def extract_html_document(text: str | None) -> str | None:
    """Extract a complete HTML document from a model response.

    Prefers fenced ```html blocks containing a real document, falls back to a
    raw <!DOCTYPE html>...</html> span, and finally tolerates a truncated final
    fence (token-budget cut) as long as an <html> element started.
    """
    if not text:
        return None
    for fence in re.findall(r"```(?:html|web)?[^\n]*\n(.*?)```", text, re.DOTALL | re.IGNORECASE):
        low = fence.lower()
        if "<!doctype html" in low or "<html" in low:
            return fence.strip()
    match = re.search(r"<!doctype html.*</html>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0).strip()
    match = re.search(r"<html.*</html>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0).strip()
    # Truncated final fence: take everything after the last ```html opener.
    lowered = text.lower()
    pos = lowered.rfind("```html")
    if pos >= 0:
        tail = text[pos:]
        tail = tail.split("\n", 1)[1] if "\n" in tail else tail
        if "<html" in tail.lower():
            return tail.replace("```", "").strip()
    return None


def document_is_complete(doc: str | None) -> bool:
    """Heuristically decide whether an extracted HTML document is untruncated.

    A document counts as complete only when it closes its <html> element and
    every <script> it opened. Token-budget cuts leave documents that extract
    "successfully" but end mid-script; those must not be treated as delivered.
    """
    if not doc or not doc.strip():
        return False
    if "</html>" not in doc.lower():
        return False
    opened = len(re.findall(r"<script\b", doc, re.IGNORECASE))
    closed = len(re.findall(r"</script\s*>", doc, re.IGNORECASE))
    return closed >= opened


def serialize_messages(messages: list[dict[str, str]]) -> str:
    """Flatten a chat transcript for APIs without native multi-turn state."""
    parts = [f"[{m['role'].upper()}]\n{m['content']}" for m in messages]
    return "\n\n".join(parts) + "\n\n[ASSISTANT]"


def extract_egg_ids(doc: str | None) -> list[str]:
    """Read unique easter-egg ids from the GLIDER_EASTER_EGGS/GLIDER_EGGS manifest."""
    if not doc:
        return []
    for name in ("GLIDER_EASTER_EGGS", "GLIDER_EGGS"):
        m = re.search(name + r"\s*=\s*\[(.*?)\]", doc, re.DOTALL)
        if m:
            ids = re.findall(r"""["']?id["']?\s*:\s*["']([^"']+)["']""", m.group(1))
            seen: list[str] = []
            for i in ids:
                if i not in seen:
                    seen.append(i)
            return seen
    return []


def extract_manifest_state(doc: str | None) -> str:
    """Extract GLIDER_* manifest array literals (rooms, easter eggs, ...).

    Used to compact conversation history on small-context hosts: earlier
    assistant documents are replaced by their authoritative manifest state
    so later turns can still honor regressions without the full source.
    """
    if not doc:
        return ""
    parts = re.findall(r"(GLIDER_[A-Z_]+)\s*=\s*(\[.*?\])\s*;", doc, re.DOTALL)
    seen: list[str] = []
    for name, body in parts:
        entry = f"{name} = {body}"
        if entry not in seen:
            seen.append(entry)
    return "\n".join(seen)


def has_persistent_scoreboard_js(doc: str | None) -> bool:
    """JS/web analogue of the suite's _has_persistent_scoreboard gate.

    Requires ALL four signals: a score, persistence (localStorage), player
    name/initials entry, and a reset-to-zero on new game.
    """
    if not doc:
        return False
    low = doc.lower()
    has_score = "score" in low
    has_persistence = "localstorage" in low
    has_name_entry = bool(re.search(r"prompt\s*\(|initials|enter[^.\n]{0,20}name", low))
    has_reset = bool(re.search(r"newscore|newgame|new game|resetscore|resetgame|score\s*=\s*0\b", low))
    return has_score and has_persistence and has_name_entry and has_reset


def evaluate_check(check: dict[str, Any], doc: str | None) -> tuple[bool, dict[str, Any]]:
    """Evaluate a single content check against the current document.

    Supported types: regex (one pattern), any_of (list of patterns),
    min_eggs (manifest count >= count).
    Returns (passed, detail).
    """
    ctype = check.get("type", "regex")
    if ctype == "regex":
        pat = check.get("pattern", "")
        found = bool(doc and re.search(pat, doc, re.IGNORECASE | re.DOTALL))
        return found, {"pattern": pat}
    if ctype == "any_of":
        pats = check.get("patterns", [])
        hits = [p for p in pats if doc and re.search(p, doc, re.IGNORECASE | re.DOTALL)]
        return bool(hits), {"patterns": pats, "matched": hits[:3]}
    if ctype == "min_eggs":
        need = int(check.get("count", 1))
        got = len(extract_egg_ids(doc))
        return got >= need, {"required": need, "found": got}
    return False, {"error": f"unknown check type '{ctype}'"}


def evaluate_checks(checks: list[dict[str, Any]], doc: str | None) -> dict[str, Any]:
    """Run a weighted check group; returns totals plus per-check detail."""
    details: list[dict[str, Any]] = []
    weight_total = 0
    weight_passed = 0
    for check in checks:
        w = int(check.get("weight", 1))
        passed, detail = evaluate_check(check, doc)
        weight_total += w
        weight_passed += w if passed else 0
        details.append({"name": check.get("name", ""), "passed": passed, "weight": w, **detail})
    return {
        "weight_total": weight_total,
        "weight_passed": weight_passed,
        "all_passed": weight_passed == weight_total,
        "checks": details,
    }


class MultiStepBenchmark:
    """Harness driving multi-turn agentic workflows against local/online models."""

    def __init__(self) -> None:
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

        self.RESULTS_DIR = Path("data/multistep_benchmarks")
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR = self.RESULTS_DIR / "models"
        self.ARTIFACTS_DIR = Path("data/artifacts")
        self.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Resolved backend context windows (model -> n_ctx), cached per run.
        self._ctx_cache: dict[str, int] = {}

    async def _resolve_context_window(self, model: str) -> int:
        """Resolve the live context window for `model` (delegates to shared module)."""
        return await context_awareness.resolve_context_window(
            model, self.PROXY_SERVER_URLS, self._ctx_cache, "multistep/benchmark"
        )

    async def _scan_runtime_ctx(
        self, client: httpx.AsyncClient, model: str, candidates: tuple[str, ...]
    ) -> int | None:
        """Look up ctx-size across proxy URLs (delegates to shared module)."""
        return await context_awareness.scan_runtime_ctx(client, model, candidates, self.PROXY_SERVER_URLS)

    async def _warm_model(self, client: httpx.AsyncClient, model: str) -> None:
        """Send a minimal streamed chat so the proxy loads the model."""
        await context_awareness.warm_model(client, model, self.PROXY_SERVER_URLS, "multistep/benchmark")

    @staticmethod
    def _estimate_tokens(messages: list[dict[str, str]]) -> int:
        """Rough token estimate for a transcript (delegates to shared module)."""
        return context_awareness.estimate_prompt_tokens(messages)

    def _compact_messages(self, messages: list[dict[str, str]], ctx: int) -> tuple[list[dict[str, str]], bool]:
        """Shrink history when it crowds out generation headroom.

        The most recent assistant document and the most recent user prompt are
        preserved verbatim; older assistant documents collapse to their
        extracted GLIDER_* manifest state (authoritative rooms/eggs data).
        """
        return context_awareness.compact_messages(messages, ctx, extract_manifest_state)

    def _turn_budget(self, ctx: int, messages: list[dict[str, str]], requested: int) -> int:
        """Clamp requested num_predict so prompt + generation fit the window."""
        return context_awareness.turn_budget(ctx, context_awareness.estimate_prompt_tokens(messages), requested)

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

    # ------------------------------------------------------------------ #
    # Workflow definitions                                                #
    # ------------------------------------------------------------------ #

    def get_all_workflows(self) -> list[dict[str, Any]]:
        """All registered multi-step workflows. Currently: Glider 2026."""
        return [_glider_2026_workflow()]

    # ------------------------------------------------------------------ #
    # Query layer                                                         #
    # ------------------------------------------------------------------ #

    async def query_model_turn(
        self,
        model: str,
        use_proxy: bool,
        messages: list[dict[str, str]],
        max_tokens: int = 4000,
        custom_keys: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Send the full conversation so far; returns one assistant turn."""
        if online_model_provider and online_model_provider.is_online_model(model):
            return await online_model_provider.query_online_model(
                model_identifier=model,
                prompt=serialize_messages(messages),
                max_tokens=max_tokens,
                custom_keys=custom_keys,
            )

        urls = self.PROXY_SERVER_URLS if use_proxy else self.OLLAMA_SERVER_URLS
        last_error = None

        for base_url in urls:
            try:
                start_t = time.time()
                async with httpx.AsyncClient(timeout=_STREAM_TIMEOUT) as client:
                    if use_proxy:
                        payload = {
                            "model": model,
                            "messages": messages,
                            "stream": True,
                            "think": False,
                            "options": {
                                "num_predict": max_tokens,
                                "temperature": _model_temperature(model),
                            },
                        }
                        headers = self._proxy_headers({"X-Request-Source": "multistep/benchmark"})
                        async with client.stream("POST", f"{base_url}/api/chat", json=payload, headers=headers) as resp:
                            if resp.status_code != 200:
                                body = (await resp.aread()).decode("utf-8", "replace")
                                last_error = f"HTTP {resp.status_code}: {body[:300]}"
                                continue
                            data = await _read_chat_stream(resp)
                    else:
                        payload = {
                            "model": model,
                            "prompt": serialize_messages(messages),
                            "stream": True,
                            "think": False,
                            "options": {
                                "num_predict": max_tokens,
                                "temperature": _model_temperature(model),
                            },
                        }
                        async with client.stream("POST", f"{base_url}/api/generate", json=payload) as resp:
                            if resp.status_code != 200:
                                body = (await resp.aread()).decode("utf-8", "replace")
                                last_error = f"HTTP {resp.status_code}: {body[:300]}"
                                continue
                            data = await _read_generate_stream(resp)

                latency = time.time() - start_t
                content = strip_thinking(data["content"])
                eval_cnt = data["eval_count"]

                # Phase-2 continuation when the turn exhausted its budget
                # (same policy as the SharedLLM suite).
                if eval_cnt >= max_tokens:
                    try:
                        nudge = (
                            "[System: You are nearly out of your token budget. Close your HTML document cleanly NOW.]"
                        )
                        if use_proxy:
                            payload2 = dict(payload)
                            payload2["messages"] = [
                                *messages,
                                {"role": "assistant", "content": content},
                                {"role": "user", "content": nudge},
                            ]
                            start_t2 = time.time()
                            async with (
                                httpx.AsyncClient(timeout=_STREAM_TIMEOUT) as client2,
                                client2.stream("POST", f"{base_url}/api/chat", json=payload2, headers=headers) as resp2,
                            ):
                                if resp2.status_code == 200:
                                    data2 = await _read_chat_stream(resp2)
                                    content = content + "\n" + strip_thinking(data2["content"])
                                    eval_cnt += data2["eval_count"]
                            latency += time.time() - start_t2
                        else:
                            payload2 = dict(payload)
                            payload2["prompt"] = serialize_messages(messages) + f"\n{content}\n{nudge}"
                            start_t2 = time.time()
                            async with (
                                httpx.AsyncClient(timeout=_STREAM_TIMEOUT) as client2,
                                client2.stream("POST", f"{base_url}/api/generate", json=payload2) as resp2,
                            ):
                                if resp2.status_code == 200:
                                    data2 = await _read_generate_stream(resp2)
                                    content = content + "\n" + strip_thinking(data2["content"])
                                    eval_cnt += data2["eval_count"]
                            latency += time.time() - start_t2
                    except Exception as e2:
                        print(f"Phase 2 continuation error in multistep harness: {e2}")

                return {
                    "success": True,
                    "latency": latency,
                    "response": content,
                    "tokens_generated": eval_cnt,
                    "error": None,
                }
            except Exception as e:
                last_error = str(e)
                continue

        return {
            "success": False,
            "latency": 0.0,
            "response": None,
            "tokens_generated": 0,
            "error": last_error or "no endpoint reachable",
        }

    # ------------------------------------------------------------------ #
    # Verification                                                        #
    # ------------------------------------------------------------------ #

    def verify_ui_render(self, doc: str | None) -> dict[str, Any]:
        """Render the final document headless via the sandbox (like ui-type tests).

        Headless Chromium is flaky under container resource pressure (dbus
        noise, OOM), so a *lint-passing* render that comes back ``ran=False``
        is retried: a syntax/lint error would never succeed on a second try,
        so only transient failures are worth another shot.
        """
        if not doc:
            return {"ran": False, "skipped": "no final document"}
        if grade_code is None:
            return {"ran": None, "skipped": "sandbox unavailable"}
        code = doc
        if extract_clean_code is not None:
            code = extract_clean_code(doc, "html") or doc

        def _render() -> dict[str, Any]:
            res = grade_code(code, lang="html", expected_output=None, timeout=45, ui=True)
            return {
                "ran": res.get("ran"),
                "score": res.get("score"),
                "error": res.get("error", ""),
                "screenshot": bool(res.get("screenshot")),
            }

        try:
            result = _render()
        except Exception as e:
            return {"ran": None, "skipped": f"sandbox error: {e}"}
        # Only retry when the lint gate passed and the render itself failed --
        # a syntax/lint error is deterministic and would fail again.
        attempts = 1
        while (
            result["ran"] is False
            and not result["error"].startswith("syntax/lint error")
            and attempts < _UI_RENDER_RETRIES
        ):
            attempts += 1
            time.sleep(_UI_RENDER_RETRY_DELAY_S)
            try:
                result = _render()
            except Exception as e:
                return {"ran": None, "skipped": f"sandbox error: {e}"}
        if attempts > 1:
            result["render_attempts"] = attempts
        return result

    def score_workflow(
        self,
        workflow: dict[str, Any],
        step_results: list[dict[str, Any]],
        final_doc: str | None,
        ui_render: dict[str, Any] | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Weighted 0-100 score over step gates, delivery chain, eggs, rooms, render."""
        breakdown: dict[str, float] = {}

        orig_results = [sr for sr in step_results if sr.get("step_id") != "assembly_pass"]
        total_w = 0
        passed_w = 0
        for sr in step_results:
            val = sr.get("validation", {})
            total_w += val.get("weight_total", 0)
            passed_w += val.get("weight_passed", 0)
        breakdown["step_content_checks"] = round(40.0 * (passed_w / total_w) if total_w else 0.0, 1)

        # The assembly pass exists to repair interrupted turns; when it
        # delivers, the shipped artifact is complete again, so the delivery
        # chain is credited back (the process penalty stays in content checks).
        delivered = sum(1 for sr in orig_results if sr.get("doc_extracted"))
        healed = any(sr.get("step_id") == "assembly_pass" and sr.get("doc_extracted") for sr in step_results)
        if healed and delivered < len(workflow["steps"]):
            delivered = len(workflow["steps"])
        breakdown["delivery_chain"] = round(15.0 * delivered / max(1, len(workflow["steps"])), 1)

        eggs = extract_egg_ids(final_doc)
        need_eggs = int(workflow.get("min_easter_eggs", 0))
        breakdown["easter_eggs"] = round(15.0 * min(1.0, len(eggs) / need_eggs) if need_eggs else 15.0, 1)

        room_pats = workflow.get("rooms_required_patterns", [])
        found_rooms = sum(1 for p in room_pats if final_doc and re.search(p, final_doc, re.IGNORECASE))
        breakdown["rooms"] = round(10.0 * found_rooms / len(room_pats) if room_pats else 10.0, 1)

        breakdown["persistent_scoreboard"] = 5.0 if has_persistent_scoreboard_js(final_doc) else 0.0

        if ui_render is None:
            ui_render = self.verify_ui_render(final_doc)
        if ui_render.get("ran") is True:
            breakdown["ui_render"] = 15.0 if ui_render.get("screenshot") else 7.0
        elif ui_render.get("ran") is None:
            breakdown["ui_render"] = 7.0
        else:
            breakdown["ui_render"] = 0.0

        score = round(min(100.0, sum(breakdown.values())), 1)
        return score, {"breakdown": breakdown, "egg_ids": eggs, "rooms_found": f"{found_rooms}/{len(room_pats)}"}

    # ------------------------------------------------------------------ #
    # Execution                                                           #
    # ------------------------------------------------------------------ #

    async def run_workflow(
        self,
        model: str,
        use_proxy: bool,
        workflow: dict[str, Any],
        cancel_event=None,
        custom_keys: dict[str, str] | None = None,
        progress_callback: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        """Drive one workflow (conversation) against one model."""

        def _emit(event: str, data: dict[str, Any]) -> None:
            if progress_callback:
                try:
                    progress_callback(event, data)
                except Exception as e:
                    print(f"Callback error: {e}")

        messages: list[dict[str, str]] = []
        final_doc: str | None = None
        final_doc_complete = False
        step_results: list[dict[str, Any]] = []
        total_latency = 0.0
        total_tokens = 0
        error: str | None = None

        def _absorb_doc(doc: str | None) -> None:
            """Adopt the best document so far.

            A COMPLETE document always outranks a longer truncated one; among
            documents of equal completeness the longer one wins. Without this,
            a token-cut fragment from a late turn silently replaces an earlier
            full delivery (and vice versa: the longest fragment wins over no
            artifact at all).
            """
            nonlocal final_doc, final_doc_complete
            if not doc:
                return
            complete = document_is_complete(doc)
            if final_doc is None:
                final_doc, final_doc_complete = doc, complete
                return
            if complete and not final_doc_complete:
                final_doc, final_doc_complete = doc, True
            elif complete == final_doc_complete and len(doc) > len(final_doc):
                final_doc, final_doc_complete = doc, complete

        is_online = bool(online_model_provider and online_model_provider.is_online_model(model))
        ctx_window = await self._resolve_context_window(model) if not is_online else None

        # Small-window hosts need an explicit brevity contract, or later turns
        # cannot re-emit the full document without blowing the context ceiling.
        brevity_note = ""
        if ctx_window and ctx_window < 16384:
            doc_cap = max(8000, min(14000, int(ctx_window * 1.75)))
            brevity_note = (
                f"\n\n[System host constraint: this machine has a {ctx_window}-token context window. "
                f"Keep your COMPLETE single-file document under {doc_cap} characters total: terse JavaScript, "
                "minimal whitespace and comments, compact helpers. Never leave the document truncated mid-tag.]"
            )

        async def _query_with_recovery(msgs: list[dict[str, str]], budget: int):
            """One model turn with transient-failure recovery.

            Zero-token responses (backend stream died, e.g. cgroup OOM kill
            respawning llama-server mid-generation) are retried while the
            backend recovers; prose-without-document gets one redo.
            """
            r = await self.query_model_turn(model, use_proxy, msgs, budget, custom_keys=custom_keys)
            c = strip_thinking(r.get("response") or "")
            retries = 0
            while (
                r["success"]
                and not c.strip()
                and (r.get("tokens_generated") or 0) == 0
                and retries < _EMPTY_TURN_RETRIES
                and not (cancel_event and cancel_event.is_set())
            ):
                retries += 1
                if _EMPTY_TURN_RETRY_DELAY_S:
                    await asyncio.sleep(_EMPTY_TURN_RETRY_DELAY_S)
                r = await self.query_model_turn(model, use_proxy, msgs, budget, custom_keys=custom_keys)
                c = strip_thinking(r.get("response") or "")
            d = extract_html_document(c) if r["success"] else None
            if (
                r["success"]
                and c.strip()
                and d is None
                and not (cancel_event and cancel_event.is_set())
            ):
                r = await self.query_model_turn(model, use_proxy, msgs, budget, custom_keys=custom_keys)
                c = strip_thinking(r.get("response") or "")
                d = extract_html_document(c) if r["success"] else None
            err = None
            if not r["success"]:
                err = r.get("error") or "turn failed"
            elif not c.strip():
                err = (
                    f"empty generation after {retries} retries "
                    "(backend stream died mid-turn; server restart/OOM suspected)"
                )
            return r, c, d, err

        for idx, step in enumerate(workflow["steps"], start=1):
            if cancel_event and cancel_event.is_set():
                error = "cancelled"
                break

            num_predict = step["max_tokens"]
            # Append this turn's prompt first so compaction and the budget
            # clamp account for its tokens too.
            messages.append({"role": "user", "content": step["prompt"] + brevity_note})
            if ctx_window:
                messages, _compacted = self._compact_messages(messages, ctx_window)
                num_predict = self._turn_budget(ctx_window, messages, step["max_tokens"])
                if num_predict < 512:
                    error = (
                        f"Context window exhausted before turn {idx} "
                        f"(ctx={ctx_window}, ~{self._estimate_tokens(messages)} prompt tokens). "
                        "Cannot generate a meaningful room on this host."
                    )
                    step_results.append(
                        {
                            "step_id": step["id"],
                            "step_label": step["label"],
                            "latency": 0.0,
                            "tokens_generated": 0,
                            "response_chars": 0,
                            "doc_extracted": False,
                            "num_predict": num_predict,
                            "error": error,
                            "validation": evaluate_checks(step.get("checks", []), None),
                        }
                    )
                    break

            _emit(
                "test_step",
                {
                    "model": model,
                    "workflow": workflow["id"],
                    "workflow_label": workflow["label"],
                    "category": workflow["category"],
                    "step": idx,
                    "total": len(workflow["steps"]),
                    "label": step["label"],
                    "num_predict": num_predict,
                },
            )

            res, content, doc, error = await _query_with_recovery(messages, num_predict)
            messages.append({"role": "assistant", "content": content})
            # Documents are cumulative by contract; a later turn that only
            # manages a truncated fragment must not overwrite a fuller earlier
            # delivery, so the most substantial COMPLETE extraction becomes
            # the artifact.
            _absorb_doc(doc)

            validation = evaluate_checks(step.get("checks", []), doc)
            total_latency += res["latency"]
            total_tokens += res["tokens_generated"]

            # Persist EVERY turn's raw output. Older turns are compacted out of
            # the live conversation and only the most substantial document
            # becomes the playable artifact; without per-turn files, all other
            # code the model generated (including truncated turns) is lost.
            turn_path = None
            try:
                self.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
                turn_path = (
                    self.ARTIFACTS_DIR / f"{self._sanitize_model_filename(model)}__{workflow['id']}__turn{idx}.txt"
                )
                turn_path.write_text(content or "", encoding="utf-8")
            except Exception as e:
                print(f"Turn artifact save failed: {e}")

            step_results.append(
                {
                    "step_id": step["id"],
                    "step_label": step["label"],
                    "latency": round(res["latency"], 2),
                    "tokens_generated": res["tokens_generated"],
                    "response_chars": len(content),
                    "doc_extracted": doc is not None,
                    "num_predict": num_predict,
                    "error": error,
                    "response_path": str(turn_path) if turn_path else None,
                    "validation": validation,
                }
            )

        # Healing assembly pass: if any turn failed to deliver its code, the
        # shipped artifact would be missing whole rooms/features. One extra
        # turn re-emits ONE complete runnable document from the best state
        # recovered so far — the graded result must be runnable end-to-end.
        assembly_cfg = workflow.get("assembly")
        missing = [sr["step_label"] for sr in step_results if not sr.get("doc_extracted")]
        # A turn can "deliver" a document that is still truncated mid-tag (token
        # cap hit inside a <script>). The graded artifact must be runnable, so
        # truncation triggers the healing pass exactly like a missing delivery.
        if assembly_cfg and not missing and final_doc and not final_doc_complete:
            missing = ["an untruncated document (current one was cut off mid-tag)"]
        if assembly_cfg and missing and not (cancel_event and cancel_event.is_set()):
            asm_idx = len(workflow["steps"]) + 1
            budget = int(assembly_cfg.get("max_tokens") or workflow["steps"][-1].get("max_tokens", 11000))
            doc_block = (
                ("\n\nCURRENT MOST COMPLETE DOCUMENT (re-emit it in full with the missing parts integrated):\n"
                 f"```html\n{final_doc}\n```\n")
                if final_doc
                else "\n\nNo complete document exists yet; produce the full game now.\n"
            )
            asm_prompt = (
                assembly_cfg["prompt"].replace("{missing}", ", ".join(missing))
                + doc_block
                + brevity_note
            )
            messages.append({"role": "user", "content": asm_prompt})
            if ctx_window:
                messages, _compacted = self._compact_messages(messages, ctx_window)
                budget = self._turn_budget(ctx_window, messages, budget)
            if budget < 512:
                # No headroom to re-emit anything meaningful; keep the honest
                # failed-turn records instead of sending a doomed request.
                messages.pop()
            else:
                _emit(
                    "test_step",
                    {
                        "model": model,
                        "workflow": workflow["id"],
                        "workflow_label": workflow["label"],
                        "category": workflow["category"],
                        "step": asm_idx,
                        "total": asm_idx,
                        "label": assembly_cfg.get("label", "Assembly pass"),
                        "num_predict": budget,
                    },
                )
                res, content, doc, error = await _query_with_recovery(messages, budget)
                messages.append({"role": "assistant", "content": content})
                total_latency += res["latency"]
                total_tokens += res["tokens_generated"]
                _absorb_doc(doc)

                turn_path = None
                try:
                    self.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
                    turn_path = (
                        self.ARTIFACTS_DIR / f"{self._sanitize_model_filename(model)}__{workflow['id']}__turn{asm_idx}.txt"
                    )
                    turn_path.write_text(content or "", encoding="utf-8")
                except Exception as e:
                    print(f"Assembly artifact save failed: {e}")

                step_results.append(
                    {
                        "step_id": "assembly_pass",
                        "step_label": assembly_cfg.get("label", "Assembly pass"),
                        "latency": round(res["latency"], 2),
                        "tokens_generated": res["tokens_generated"],
                        "response_chars": len(content),
                        "doc_extracted": doc is not None,
                        "num_predict": budget,
                        "error": error,
                        "response_path": str(turn_path) if turn_path else None,
                        "validation": evaluate_checks(assembly_cfg.get("checks", []), doc),
                        "healed_passes": missing,
                    }
                )

        # Render once; the same verdict feeds scoring and the result payload.
        ui_render = self.verify_ui_render(final_doc)
        score, extra = self.score_workflow(workflow, step_results, final_doc, ui_render=ui_render)

        # Persist the final playable artifact for inspection/serving.
        artifact_path = None
        if final_doc:
            try:
                artifact_path = self.ARTIFACTS_DIR / f"{self._sanitize_model_filename(model)}__{workflow['id']}.html"
                artifact_path.write_text(final_doc, encoding="utf-8")
            except Exception as e:
                print(f"Artifact save failed: {e}")

        ran = ui_render.get("ran")
        success = score >= 60.0 if ran is None else bool(ran) and score >= 60.0

        return {
            "test_id": workflow["id"],
            "test_category": workflow["category"],
            "test_label": workflow["label"],
            "success": success,
            "score": score,
            "steps_completed": sum(1 for sr in step_results if sr["doc_extracted"] and sr.get("step_id") != "assembly_pass"),
            "steps_total": len(workflow["steps"]),
            "latency": round(total_latency, 2),
            "tokens_generated": total_tokens,
            "prompt_steps": [s["id"] for s in workflow["steps"]],
            "steps": step_results,
            "artifact": str(artifact_path) if artifact_path else None,
            "validation": {
                **extra,
                "ui_render": ui_render,
                "persistent_scoreboard": has_persistent_scoreboard_js(final_doc),
                "final_document_chars": len(final_doc or ""),
            },
            # Store the COMPLETE final document so every viewer (payload
            # modal, exports) shows the full generated code, not an excerpt.
            "response": final_doc,
            "error": error,
        }

    def save_per_model_result(
        self,
        model_record: dict[str, Any],
        use_proxy: bool,
        generated_at: str | None = None,
    ) -> Path | None:
        model = model_record.get("model")
        if not model:
            return None
        per_model = {
            "benchmark_version": "MultiStep-v1",
            "generated_at": generated_at or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "benchmark_type": "proxy" if use_proxy else "direct",
            "models_tested": 1,
            "per_model": True,
            "model": model,
            "results": [model_record],
        }
        file_path = self.MODELS_DIR / f"multistep_{self._sanitize_model_filename(model)}.json"
        with open(file_path, "w") as f:
            json.dump(per_model, f, indent=2, default=str)
        return file_path

    async def run_multistep_benchmarks(
        self,
        models: list[str],
        use_proxy: bool,
        progress_callback: Callable[..., Any] | None = None,
        cancel_event=None,
        workflow_ids: list[str] | None = None,
        custom_keys: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Run workflows for each model, emitting the standard progress protocol."""
        all_results: dict[str, Any] = {
            "benchmark_version": "MultiStep-v1",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "benchmark_type": "proxy" if use_proxy else "direct",
            "models_tested": len(models),
            "results": [],
        }

        all_workflows = self.get_all_workflows()
        WORKFLOWS = [w for w in all_workflows if w["id"] in set(workflow_ids)] if workflow_ids else all_workflows

        total_tests = len(models) * len(WORKFLOWS)
        generated_at = all_results["generated_at"]

        def _emit(event: str, data: dict[str, Any]) -> None:
            if progress_callback:
                try:
                    progress_callback(event, data)
                except Exception as e:
                    print(f"Callback error: {e}")

        _emit(
            "benchmark_start",
            {
                "models": models,
                "use_proxy": use_proxy,
                "total_models": len(models),
                "total_tests": total_tests,
                "timestamp": generated_at,
            },
        )

        completed_count = 0
        for model in models:
            if cancel_event and cancel_event.is_set():
                break
            _emit("model_start", {"model": model})

            model_record: dict[str, Any] = {
                "model": model,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "tasks": [],
            }

            for workflow in WORKFLOWS:
                if cancel_event and cancel_event.is_set():
                    break
                _emit(
                    "test_start",
                    {
                        "model": model,
                        "category": workflow["category"],
                        "test_id": workflow["id"],
                        "test_label": workflow["label"],
                    },
                )

                result = await self.run_workflow(
                    model=model,
                    use_proxy=use_proxy,
                    workflow=workflow,
                    cancel_event=cancel_event,
                    custom_keys=custom_keys,
                    progress_callback=progress_callback,
                )
                model_record["tasks"].append(result)
                completed_count += 1
                _emit(
                    "test_complete",
                    {
                        "model": model,
                        "category": workflow["category"],
                        "test_id": workflow["id"],
                        "test_label": workflow["label"],
                        "result": result,
                        "progress": {
                            "completed": completed_count,
                            "total": total_tests,
                            "percentage": round((completed_count / total_tests) * 100) if total_tests else 100,
                        },
                    },
                )

            results_list = all_results["results"]
            results_list.append(model_record)
            try:
                self.save_per_model_result(model_record, use_proxy, generated_at)
            except Exception as e:
                print(f"Failed to save per-model multistep result for {model}: {e}")
                raise
            _emit("model_complete", {"model": model, "results": model_record})

        if cancel_event and cancel_event.is_set():
            all_results["status"] = "cancelled"
        else:
            all_results["status"] = "completed"

        if all_results["results"]:
            save_file = (
                self.RESULTS_DIR
                / f"multistep_benchmarks_{time.strftime('%Y%m%d_%H%M%S')}_{'proxy' if use_proxy else 'direct'}.json"
            )
            with open(save_file, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            all_results["saved_as"] = str(save_file)

        _emit("benchmark_complete", all_results)
        return all_results


# ---------------------------------------------------------------------- #
# Glider 2026 workflow                                                    #
# ---------------------------------------------------------------------- #


def _glider_2026_workflow() -> dict[str, Any]:
    """Four-turn agentic build of a modern Glider homage with easter eggs."""
    category = "multistep_gamedev"

    contracts = (
        "DATA CONTRACTS (keep these exact identifier names in the code and extend them every turn):\n"
        'const GLIDER_ROOMS = [{ id: "den", name: "The Den", exits: ["kitchen"] }, ...]; '
        "one entry per room, with door/exit wiring between rooms;\n"
        'const GLIDER_EASTER_EGGS = [{ id: "sleeping-cat", room: "den", description: "..." }, ...]; '
        "one entry per easter egg with a UNIQUE lowercase-kebab id, the room id it lives in, and a "
        "short description. Every egg must be triggerable through gameplay (fly over/touch an object, "
        "a timed event, or a keyboard secret) and show visible in-game feedback (banner/toast, particle "
        "burst, or screen effect). Pressing E opens an EGG TRACKER overlay listing found vs unfound eggs."
    )

    output_contract = (
        "OUTPUT FORMAT: Respond with the COMPLETE, self-contained HTML document in ONE single ```html "
        "code block — the entire document (doctype through </html>) including ALL previously completed "
        "work, so it can be extracted and rendered verbatim. Do NOT split the code across multiple blocks, "
        "do NOT output diffs/patches/snippets, and include NO prose outside the block. The page must "
        "auto-start gameplay without any user gesture so a headless renderer captures a non-blank frame."
    )

    physics_spec = (
        "PHYSICS (honor the original's feel): constant gravity pulls the plane down; horizontal arrow keys "
        "(and A/D) give gentle forward/backward thrust with momentum and drag; vertical control comes ONLY "
        "from air sources — heat-vent updraft columns (animated particle streams), candle thermals near "
        "flames, ceiling-fan downdrafts/crosswinds, outdoor gusts. Touching floor/furniture/flames kills a "
        "plane and costs a life; the plane respawns at the room entrance."
    )

    step1_prompt = (
        'You are starting a 4-turn engineering session building "GLIDER 2026" — a modern, creative '
        "HTML5-canvas reimagining of the classic 1988 Macintosh game Glider by John Calhoun "
        "(Soft Dorothy Software). In the original, a paper airplane drifts through the rooms of a house, "
        "riding warm updrafts from heat vents while household objects (tables, candles, bouncing balls, "
        "popping toast) destroy it; collectibles were clocks (points) and sheets of paper (extra planes). "
        "Honor that heritage with a fresh, polished look of your own.\n\n"
        "TECH BASE: ONE single-file HTML document, vanilla JavaScript + Canvas 2D only (no libraries, no CDN, "
        "no external assets/fonts/images/audio files). Fixed 960x600 canvas, crisp retro-modern styling, "
        "requestAnimationFrame loop, keyboard input via keydown/keyup.\n"
        + physics_spec
        + '\n\nTHIS TURN — ENGINE + ROOM 1 "The Den":\n'
        "1. Core engine: gravity/thrust/lift integration, collision system, lives (start 3), score, room "
        "transition through doorways (walking/flying through a door edge loads the next room).\n"
        "2. The Den: a floor vent whose updraft column lifts the plane past a tall bookshelf; a coffee table "
        "and lamp as obstacles; a clock collectible (+points); a sheet-of-paper collectible (+1 life); a "
        "doorway on the right leading to the Kitchen (next turn); window with rain outside (foreshadowing).\n"
        "3. Easter egg #1 — sleeping cat: a cat napping on the rug; flying gently over it makes it purr "
        "(visual heart particles + banner) and registers an egg.\n"
        "4. Title splash referencing the lineage (paper plane + subtle nod to 1988).\n"
        "5. HUD: room name, score, lives, speed.\n" + contracts + "\n\n" + output_contract
    )

    step2_prompt = (
        'CONTINUING the 4-turn "GLIDER 2026" build (turn 2 of 4). Extend your existing game. Do not '
        "regress anything from turn 1: the Den, its vent/cat/clock/paper, physics and HUD must remain intact "
        "and reachable.\n\n"
        'ADD ROOM 2 "The Kitchen":\n'
        "- A toaster on the counter pops toast on a periodic timer; toast mid-pop is lethal.\n"
        "- An electrical wall outlet that sparks/zaps on a visible timer arc — lethal during the zap.\n"
        "- A goldfish bowl: the goldfish periodically leaps out in an arc (lethal to touch mid-air, funny).\n"
        "- A greasy countertop section: touching it makes the plane slide along it smoothly until it ends "
        "(slide-along-grease mechanic from Glider PRO).\n"
        "- Fridge-magnet letters that spell a hidden word (easter egg #2: fly close to read it — banner "
        "reveals the message).\n\n"
        'ADD ROOM 3 "The Living Room":\n'
        "- Candles on the mantel: bright thermal updraft above each flame (rideable) but lethal flame hitbox.\n"
        "- A basketball bouncing across the floor in a predictable arc (lethal on contact).\n"
        "- A ceiling fan creating a crosswind band that pushes the plane sideways.\n"
        "- Fireplace with a subtle draft: flying INTO the fireplace flue reveals a hidden passage "
        "(easter egg #3: secret nook with a framed photo labeled 'Soft Dorothy').\n"
        "- Bookshelf with a clickable-looking glowing book (easter egg #4: flying past it slowly flips pages "
        "showing pixel art of a paper plane).\n\n"
        "Wire doors: den <-> kitchen <-> living_room, and update BOTH manifests (rooms + eggs, now >= 4 eggs).\n"
        + output_contract
    )

    step3_prompt = (
        'CONTINUING the 4-turn "GLIDER 2026" build (turn 3 of 4). Keep every previous room working '
        "(regression matters more than new polish).\n\n"
        'ADD ROOM 4 "The Attic":\n'
        "- Stacked box maze with narrow gaps; cobwebs that slow the plane while passing through.\n"
        "- Draft gusts whistling through wall cracks (periodic horizontal wind bands, telegraphed by dust "
        "particles).\n"
        "- A pull-chain lightbulb: flying through the chain toggles a dim/bright lighting mode "
        "(easter egg #5).\n"
        "- Hidden dedication plaque in a far corner reading 'For John Calhoun — Soft Dorothy Software, 1988' "
        "(easter egg #6, discovered by flying into the corner).\n\n"
        'ADD ROOM 5 "The Basement":\n'
        "- Dripping water pipes: drops fall on a cycle, splashing puddles below (lethal drops).\n"
        "- A long grease-covered pipe you can slide along over a hazard gap (reuse the grease mechanic).\n"
        "- Rubber-band power-up pickup: press SPACE to fire a rubber band that can pop balloon enemies.\n"
        "- Balloons drifting upward from a party stash (lethal on contact unless shot down).\n\n"
        'ADD ROOM 6 "The Backyard" (outdoors, sunny Glider PRO homage):\n'
        "- Open sky with wind gusts, a clothesline with flapping sheets, tree branches, and birds crossing "
        "(lethal).\n"
        "- A chimney updraft column that lets you re-enter the house; crossing into the attic triggers a "
        "storm-cloud weather shift (rain + darker palette) honoring Glider 4.0's moody theme "
        "(easter egg #7: experiencing both sunny and storm states in one flight).\n\n"
        "Update BOTH manifests: rooms now 6 with full door graph; eggs now >= 7.\n" + output_contract
    )

    step4_prompt = (
        'FINAL TURN (4 of 4) of the "GLIDER 2026" build: polish, persistence, and ship. All six rooms and '
        "every mechanic must still work after this turn.\n\n"
        "REQUIRED THIS TURN:\n"
        "1. SCORING & PERSISTENT HIGH-SCORE BOARD: award points for clocks, eggs discovered, and rooms "
        "cleared. On game over (or victory), prompt for player name/initials and record it with the score; "
        "persist the TOP-5 table in localStorage under a key like 'glider2026_highscores'; display it on the "
        "title and game-over screens; scores survive reload; every NEW GAME resets the current score to 0 "
        "without wiping the high-score table.\n"
        "2. Victory condition: reaching the Backyard flag with any plane left shows a victory screen with "
        "stats (time, eggs found, rooms visited).\n"
        "3. House map (TAB): overlay minimap showing the six rooms, doors, and which have been visited.\n"
        "4. Egg tracker (E): lists found vs unfound egg descriptions (ids from GLIDER_EASTER_EGGS).\n"
        "5. Pause (P or Esc).\n"
        "6. Easter egg #8 (minimum total is now 8, all declared in the manifest): typing D-O-R-O-T-H-Y on "
        "the keyboard toggles a '1988 MODE' — monochrome black-and-white rendering with chunkier pixels, a "
        "playable homage to the original Mac release. Show a banner when unlocked.\n"
        "7. Credits line on the title screen honoring John Calhoun's original 1988 shareware Glider.\n\n"
        "Final regression checklist (verify in-code): 6 rooms connected via doors, vents/candles/gusts lift, "
        "toast/outlet/goldfish/basketball/balloons/birds/drips hazards, grease sliding, rubber band, "
        "clocks/paper/battery collectibles, storm transition, high-score board, egg tracker, map, pause, "
        ">= 8 manifest eggs.\n" + output_contract
    )

    def _complete_doc_check(weight: int = 3) -> dict[str, Any]:
        return {
            "name": "delivered_complete_html_document",
            "type": "regex",
            "pattern": r"<!doctype html|<html[\s>]",
            "weight": weight,
        }

    return {
        "id": "glider_2026_house",
        "category": category,
        "label": "Multi-Step Agentic: Glider 2026 — paper-plane house odyssey (4 turns)",
        "description": (
            "Recreates the 1988 Macintosh classic Glider (John Calhoun, Soft Dorothy Software) as a modern "
            "HTML5 canvas game built across a 4-turn agentic conversation: engine + Den, Kitchen + Living "
            "Room, Attic + Basement + Backyard, then polish/high-scores/easter-egg hunt. Graded on per-turn "
            "content gates, context retention across turns, >= 8 discoverable easter eggs honoring the "
            "original, persistent top-5 high-score board, and a successful headless render of the final build."
        ),
        "lang": "html",
        "min_easter_eggs": 8,
        "rooms_required_patterns": [
            r"\bden\b",
            r"\bkitchen\b",
            r"\bliving(\s*_|\s)?room\b",
            r"\battic\b",
            r"\b(basement|sewer)\b",
            r"\b(backyard|outdoor|garden)\b",
        ],
        "steps": [
            {
                "id": "turn1_engine_den",
                "label": "Turn 1: Engine + The Den",
                "prompt": step1_prompt,
                "max_tokens": 9000,
                "checks": [
                    _complete_doc_check(),
                    {"name": "canvas_rendering", "type": "regex", "pattern": r"getContext\(['\"]2d['\"]", "weight": 2},
                    {"name": "raf_game_loop", "type": "regex", "pattern": r"requestAnimationFrame", "weight": 2},
                    {"name": "keyboard_input", "type": "regex", "pattern": r"keydown", "weight": 2},
                    {"name": "gravity_physics", "type": "regex", "pattern": r"gravit|gravity", "weight": 2},
                    {"name": "vent_updraft", "type": "regex", "pattern": r"\bvent", "weight": 2},
                    {"name": "den_room", "type": "regex", "pattern": r"\bden\b", "weight": 2},
                    {"name": "door_transition", "type": "any_of", "patterns": [r"\bdoor", r"exits"], "weight": 2},
                    {"name": "rooms_manifest", "type": "regex", "pattern": r"GLIDER_ROOMS\s*=", "weight": 2},
                    {"name": "eggs_manifest", "type": "regex", "pattern": r"GLIDER_(EASTER_)?EGGS\s*=", "weight": 2},
                    {"name": "first_egg_declared", "type": "min_eggs", "count": 1, "weight": 2},
                    {"name": "lives_and_score_hud", "type": "regex", "pattern": r"live|life", "weight": 1},
                ],
            },
            {
                "id": "turn2_kitchen_livingroom",
                "label": "Turn 2: Kitchen + Living Room",
                "prompt": step2_prompt,
                "max_tokens": 11000,
                "checks": [
                    _complete_doc_check(),
                    {"name": "kitchen_room", "type": "regex", "pattern": r"kitchen", "weight": 2},
                    {"name": "toaster_hazard", "type": "regex", "pattern": r"toast", "weight": 2},
                    {"name": "outlet_zap", "type": "any_of", "patterns": [r"outlet", r"socket", r"\bzap"], "weight": 2},
                    {
                        "name": "goldfish_hazard",
                        "type": "regex",
                        "pattern": r"goldfish|fish bowl|fishbowl",
                        "weight": 1,
                    },
                    {"name": "grease_slide_mechanic", "type": "regex", "pattern": r"grease", "weight": 2},
                    {"name": "living_room", "type": "regex", "pattern": r"living(\s*_|\s)?room", "weight": 2},
                    {"name": "candle_thermal", "type": "regex", "pattern": r"candle", "weight": 2},
                    {
                        "name": "basketball_hazard",
                        "type": "any_of",
                        "patterns": [r"basketball", r"\bball\b"],
                        "weight": 1,
                    },
                    {"name": "ceiling_fan_crosswind", "type": "regex", "pattern": r"fan", "weight": 1},
                    {"name": "fireplace_secret_hint", "type": "regex", "pattern": r"fireplace", "weight": 1},
                    {"name": "den_still_present", "type": "regex", "pattern": r"\bden\b", "weight": 2},
                    {"name": "four_eggs_by_now", "type": "min_eggs", "count": 4, "weight": 3},
                ],
            },
            {
                "id": "turn3_attic_basement_backyard",
                "label": "Turn 3: Attic + Basement + Backyard",
                "prompt": step3_prompt,
                "max_tokens": 11000,
                "checks": [
                    _complete_doc_check(),
                    {"name": "attic_room", "type": "regex", "pattern": r"attic", "weight": 2},
                    {"name": "cobweb_slowdown", "type": "regex", "pattern": r"cobweb|spider.?web|web", "weight": 1},
                    {"name": "draft_gusts", "type": "any_of", "patterns": [r"gust", r"draft"], "weight": 2},
                    {"name": "basement_or_sewer", "type": "regex", "pattern": r"basement|sewer", "weight": 2},
                    {"name": "dripping_water_hazard", "type": "regex", "pattern": r"drip", "weight": 2},
                    {"name": "balloon_enemies", "type": "regex", "pattern": r"balloon", "weight": 1},
                    {"name": "rubber_band_weapon", "type": "regex", "pattern": r"rubber.?band", "weight": 2},
                    {"name": "outdoor_backyard", "type": "regex", "pattern": r"backyard|outdoor|garden", "weight": 2},
                    {"name": "birds_hazard", "type": "regex", "pattern": r"\bbird", "weight": 1},
                    {"name": "storm_weather_shift", "type": "any_of", "patterns": [r"storm", r"rain"], "weight": 2},
                    {
                        "name": "calhoun_dorothy_tribute_egg",
                        "type": "any_of",
                        "patterns": [r"calhoun", r"dorothy"],
                        "weight": 2,
                    },
                    {"name": "seven_eggs_by_now", "type": "min_eggs", "count": 7, "weight": 3},
                ],
            },
            {
                "id": "turn4_polish_ship",
                "label": "Turn 4: High-scores + Easter-egg hunt + Ship",
                "prompt": step4_prompt,
                "max_tokens": 11000,
                "checks": [
                    _complete_doc_check(),
                    {"name": "localstorage_persistence", "type": "regex", "pattern": r"localstorage", "weight": 3},
                    {
                        "name": "top5_highscores",
                        "type": "any_of",
                        "patterns": [r"top.?5", r"high.?scores?", r"leaderboard"],
                        "weight": 2,
                    },
                    {
                        "name": "name_entry_on_gameover",
                        "type": "regex",
                        "pattern": r"prompt\s*\(|initials|enter[^.\n]{0,20}name",
                        "weight": 2,
                    },
                    {
                        "name": "score_reset_new_game",
                        "type": "any_of",
                        "patterns": [r"newgame", r"new game", r"resetscore", r"resetgame", r"score\s*=\s*0\b"],
                        "weight": 2,
                    },
                    {
                        "name": "victory_flag_state",
                        "type": "any_of",
                        "patterns": [r"victory", r"\bwin\b", r"flag"],
                        "weight": 1,
                    },
                    {"name": "house_map_overlay", "type": "any_of", "patterns": [r"minimap", r"\bmap\b"], "weight": 1},
                    {
                        "name": "egg_tracker_overlay",
                        "type": "any_of",
                        "patterns": [r"tracker", r"eggs?\s+found", r"found\s+\d+\s*/\s*\d+"],
                        "weight": 2,
                    },
                    {"name": "pause_support", "type": "regex", "pattern": r"pause", "weight": 1},
                    {
                        "name": "dorothy_retro_mode_secret",
                        "type": "any_of",
                        "patterns": [r"dorothy", r"1988"],
                        "weight": 2,
                    },
                    {
                        "name": "credits_calhoun",
                        "type": "any_of",
                        "patterns": [r"john calhoun", r"soft dorothy", r"calhoun"],
                        "weight": 1,
                    },
                    {"name": "eight_eggs_final", "type": "min_eggs", "count": 8, "weight": 3},
                    {
                        "name": "backyard_still_present_final",
                        "type": "any_of",
                        "patterns": [r"backyard|outdoor|garden"],
                        "weight": 1,
                    },
                ],
            },
        ],
        "assembly": {
            "label": "Assembly pass: one complete runnable build",
            "max_tokens": 16000,
            "prompt": (
                "[System recovery pass] One or more earlier passes did not deliver their code "
                "({missing}). The graded result must be ONE complete, runnable, single-file HTML5 "
                "document — a human will open it and play the whole game, and a headless browser "
                "will render it for scoring. Re-emit the ENTIRE game now as a single ```html block: "
                "every room from GLIDER_ROOMS (den, kitchen, living_room, attic, basement/sewer, "
                "backyard/outdoors) joined by doors with working exits, EVERY easter egg from "
                "GLIDER_EASTER_EGGS (all 8, triggerable in gameplay with visible feedback), the full "
                "physics set (gravity, arrow-key thrust + A/D trim, momentum/drag; lift ONLY from "
                "vents/candle thermals/fans/gusts), hazards (toaster, outlets, goldfish, basketballs, "
                "dripping pipes, balloon enemies poppable with SPACE rubber band), collectibles "
                "(clocks=points, paper=lives, batteries=speed), persistent localStorage top-5 "
                "high-score board with initials entry and reset-to-0 on new game, victory flag "
                "screen, TAB minimap, E egg tracker, P/Esc pause, DOROTHY retro mode secret, credits "
                "honoring John Calhoun / Soft Dorothy Software 1988. Integrate the current document "
                "below as your starting point; do not drop any feature already present. Output ONLY "
                "the final complete document in one fenced block."
            ),
            "checks": [
                _complete_doc_check(),
                {"name": "canvas_rendering", "type": "regex", "pattern": r"getContext\(['\"]2d['\"]", "weight": 2},
                {"name": "raf_game_loop", "type": "regex", "pattern": r"requestAnimationFrame", "weight": 2},
                {
                    "name": "all_rooms_present",
                    "type": "any_of",
                    "patterns": [r"\bden\b", r"\bkitchen\b", r"\battic\b"],
                    "weight": 3,
                },
                {"name": "localstorage_persistence", "type": "regex", "pattern": r"localstorage", "weight": 3},
                {"name": "eight_eggs_assembled", "type": "min_eggs", "count": 8, "weight": 3},
            ],
        },
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    bench = MultiStepBenchmark()
    print([w["id"] for w in bench.get_all_workflows()])
    asyncio.run(
        bench.run_multistep_benchmarks(
            models=[m.strip() for m in (os.getenv("MULTISTEP_MODEL", "") or "qwen3:8b").split(",")],
            use_proxy=os.getenv("USE_PROXY", "1") == "1",
        )
    )
