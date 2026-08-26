"""Tests for context-window awareness shared by all benchmark suites."""

import json
from unittest.mock import patch

import httpx
import pytest

import context_awareness as ca
from context_awareness import (
    compact_messages,
    estimate_prompt_tokens,
    resolve_context_window,
    turn_budget,
)

RUNTIME_BODY = {
    "loaded_models": [
        {
            "name": "test-model--latest",
            "backend_model": "test-model--latest",
            "running_settings": {"ctx-size": "8192"},
        }
    ]
}


def _runtime_handler(seen):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/admin/runtime":
            seen.append({"url": str(request.url)})
            return httpx.Response(200, json=RUNTIME_BODY)
        if request.url.path == "/api/chat":
            seen.append({"url": str(request.url), "warmup": True})
            raw = json.dumps({"message": {"role": "assistant", "content": "OK"}, "done": True}).encode() + b"\n"
            return httpx.Response(200, content=raw)
        return httpx.Response(404)

    return handler


@pytest.mark.asyncio
async def test_resolve_context_window_scans_and_caches():
    seen = []
    real_client = httpx.AsyncClient
    cache = {}

    def factory(*a, **k):
        k.pop("timeout", None)
        return real_client(transport=httpx.MockTransport(_runtime_handler(seen)))

    with patch("context_awareness.httpx.AsyncClient", side_effect=factory):
        ctx = await resolve_context_window("test-model", ["http://mock-proxy"], cache)

    assert ctx == 8192
    assert cache["test-model"] == 8192
    runtime_hits = len(seen)

    with patch("context_awareness.httpx.AsyncClient", side_effect=factory):
        await resolve_context_window("test-model", ["http://mock-proxy"], cache)
    assert len(seen) == runtime_hits  # served from cache, no extra HTTP


@pytest.mark.asyncio
async def test_resolve_context_window_warms_unloaded_model():
    """Model absent from runtime -> warm-up chat fires, then resolution succeeds."""
    state = {"scans": 0}
    real_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/admin/runtime":
            state["scans"] += 1
            if state["scans"] == 1:
                return httpx.Response(200, json={"loaded_models": []})
            return httpx.Response(200, json=RUNTIME_BODY)
        if request.url.path == "/api/chat":
            body = json.loads(request.content.decode())
            assert body["options"]["num_predict"] == 1
            raw = json.dumps({"message": {"role": "assistant", "content": "OK"}, "done": True}).encode() + b"\n"
            return httpx.Response(200, content=raw)
        return httpx.Response(404)

    def factory(*a, **k):
        k.pop("timeout", None)
        return real_client(transport=httpx.MockTransport(handler))

    with patch("context_awareness.httpx.AsyncClient", side_effect=factory):
        ctx = await resolve_context_window("test-model", ["http://mock-proxy"], {}, "unit/test")
    assert ctx == 8192


@pytest.mark.asyncio
async def test_resolve_context_window_raises_loudly_when_unknown(monkeypatch):
    monkeypatch.setattr(ca, "RESOLVE_TIMEOUT_S", 0.2)
    monkeypatch.setattr(ca, "POLL_INTERVAL_S", 0.01)
    scans = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/admin/runtime":
            scans.append(1)
        return httpx.Response(404)

    real_client = httpx.AsyncClient

    def factory(*a, **k):
        k.pop("timeout", None)
        return real_client(transport=httpx.MockTransport(handler))

    with patch("context_awareness.httpx.AsyncClient", side_effect=factory), pytest.raises(RuntimeError, match="context window"):
        await resolve_context_window("ghost-model", ["http://dead-proxy"], {})
    assert len(scans) > 2  # it polled until the deadline, not just once


@pytest.mark.asyncio
async def test_resolve_context_window_warms_once_then_polls_until_loaded():
    """Model absent from runtime -> ONE warm-up fires, then polling waits for the load."""
    state = {"scans": 0, "chats": 0}
    real_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/admin/runtime":
            state["scans"] += 1
            if state["scans"] < 4:
                return httpx.Response(200, json={"loaded_models": []})
            return httpx.Response(200, json=RUNTIME_BODY)
        if request.url.path == "/api/chat":
            state["chats"] += 1
            body = json.loads(request.content.decode())
            assert body["options"]["num_predict"] == 1
            raw = json.dumps({"message": {"role": "assistant", "content": "OK"}, "done": True}).encode() + b"\n"
            return httpx.Response(200, content=raw)
        return httpx.Response(404)

    def factory(*a, **k):
        k.pop("timeout", None)
        return real_client(transport=httpx.MockTransport(handler))

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ca, "POLL_INTERVAL_S", 0.01)
    try:
        with patch("context_awareness.httpx.AsyncClient", side_effect=factory):
            ctx = await resolve_context_window("test-model", ["http://mock-proxy"], {}, "unit/test")
    finally:
        monkeypatch.undo()
    assert ctx == 8192
    assert state["chats"] == 1  # exactly one warm-up despite three empty scans
    assert state["scans"] >= 4


def test_estimate_and_turn_budget():
    msgs = [{"role": "user", "content": "abcdefghij"}]  # 10 chars -> 3 + 16 = 19
    est = estimate_prompt_tokens(msgs)
    assert est == 19
    assert turn_budget(8192, est, 7000) == 7000  # plenty of headroom: unchanged
    assert turn_budget(1000, est, 7000) == 1000 - 19 - 128  # clamped to headroom
    assert turn_budget(50, est, 7000) == 0  # prompt alone overflows


def test_compact_messages_generic():
    big_doc = "assistant code " * 4000  # far above the 400-char keep threshold
    messages = [
        {"role": "user", "content": "turn one instructions " * 60},
        {"role": "assistant", "content": big_doc},
        {"role": "user", "content": "turn two instructions " * 60},
        {"role": "assistant", "content": big_doc},
        {"role": "user", "content": "final instruction"},
    ]

    untouched, changed = compact_messages(messages, 1_000_000)
    assert untouched is messages and changed is False

    def extractor(content: str) -> str:
        return "MANIFEST-STATE"

    compacted, changed = compact_messages(messages, 2048, extractor)
    assert changed
    # Last user message and last assistant message preserved verbatim.
    assert compacted[-1]["content"] == "final instruction"
    assert compacted[-2]["content"] == big_doc
    # Older assistant content collapsed through the extractor.
    assert "MANIFEST-STATE" in compacted[1]["content"]
    # Older user prompts collapsed to a summary.
    assert "[System note" in compacted[0]["content"] and "[System note" in compacted[2]["content"]
    assert len(compacted) == len(messages)


@pytest.mark.asyncio
async def test_shared_llm_query_fails_fast_when_context_exhausted(tmp_path, monkeypatch):
    """A prompt that fills the window must fail fast with a clear error."""
    from web.shared_llm_benchmark import SharedLLMModelBenchmark

    ini = tmp_path / "models.ini"
    ini.write_text("[*]\ntemperature = 0.5\n")
    monkeypatch.setenv("MODELS_INI_PATH", str(ini))

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/admin/runtime":
            return httpx.Response(
                200,
                json={"loaded_models": [
                    {"name": "m--latest", "backend_model": "m--latest",
                     "running_settings": {"ctx-size": "512"}}
                ]},
            )
        calls.append(str(request.url))
        return httpx.Response(200, content=b"")

    real_client = httpx.AsyncClient

    def factory(*a, **k):
        k.pop("timeout", None)
        return real_client(transport=httpx.MockTransport(handler))

    bench = SharedLLMModelBenchmark()
    with patch("web.shared_llm_benchmark.httpx.AsyncClient", side_effect=factory):
        res = await bench.query_model(model="m", use_proxy=True, prompt="x" * 60000, max_tokens=4000)

    assert res["success"] is False
    assert "Context window exhausted" in res["error"]
    assert calls == []  # no doomed generation request was ever sent


@pytest.mark.asyncio
async def test_general_suite_clamps_num_predict_on_small_context(tmp_path, monkeypatch):
    """llm_benchmark_suite must clamp num_predict into the live context window."""
    import llm_benchmark_suite as suite_mod

    ini = tmp_path / "models.ini"
    ini.write_text("[*]\ntemperature = 0.5\n")
    monkeypatch.setenv("MODELS_INI_PATH", str(ini))

    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/admin/runtime":
            return httpx.Response(
                200,
                json={"loaded_models": [
                    {"name": "local-model--latest", "backend_model": "local-model--latest",
                     "running_settings": {"ctx-size": "2048"}}
                ]},
            )
        body = json.loads(request.content.decode())
        sent.append(body)
        raw = json.dumps({"message": {"role": "assistant", "content": "ok"}, "done": True,
                          "eval_count": 10, "eval_duration": 100, "prompt_eval_duration": 10}).encode() + b"\n"
        return httpx.Response(200, content=raw)

    real_client = httpx.AsyncClient

    def factory(*a, **k):
        k.pop("timeout", None)
        return real_client(transport=httpx.MockTransport(handler))

    s = suite_mod.LLMModelBenchmark()
    # ~1000 chars -> ~349 estimated tokens; fits a 2048 window but leaves less
    # headroom than the requested cap, so the clamp must engage.
    test = {"id": "t1", "prompt": "word " * 200, "reasoning_budget": 512}
    with patch.object(suite_mod.httpx, "AsyncClient", side_effect=factory):
        res = await s.test_model_proxy("local-model", test)

    assert res["success"] is True
    requested = suite_mod._test_num_predict(test)
    assert sent[0]["options"]["num_predict"] == 2048 - ca.estimate_prompt_tokens(
        [{"role": "user", "content": test["prompt"]}]
    ) - 128
    assert sent[0]["options"]["num_predict"] <= requested


@pytest.mark.asyncio
async def test_general_suite_reports_unresolvable_context(tmp_path, monkeypatch):
    """Unknown context window -> loud per-test failure, not a silent send."""
    import llm_benchmark_suite as suite_mod

    ini = tmp_path / "models.ini"
    ini.write_text("[*]\ntemperature = 0.5\n")
    monkeypatch.setenv("MODELS_INI_PATH", str(ini))
    monkeypatch.setattr(ca, "RESOLVE_TIMEOUT_S", 0.2)
    monkeypatch.setattr(ca, "POLL_INTERVAL_S", 0.01)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    real_client = httpx.AsyncClient

    def factory(*a, **k):
        k.pop("timeout", None)
        return real_client(transport=httpx.MockTransport(handler))

    s = suite_mod.LLMModelBenchmark()
    test = {"id": "t1", "prompt": "Hello", "reasoning_budget": 512}
    with patch.object(suite_mod.httpx, "AsyncClient", side_effect=factory):
        res = await s.test_model_proxy("ghost-model", test)

    assert res["success"] is False
    assert "Cannot determine context window" in res["error"]


def test_module_reexports_stay_in_sync():
    """multistep delegates to this module; keep the API surface stable."""
    from multistep_benchmark import MultiStepBenchmark

    b = MultiStepBenchmark()
    assert b._estimate_tokens([{"role": "user", "content": "abcdefghij"}]) == 19
    assert b._turn_budget(8192, [{"role": "user", "content": "hi"}], 9000) < 9000
