"""Concurrency, priority-FIFO queueing, backpressure, and cancellation tests for alpaca-proxy."""

import asyncio
import importlib.util
import json
import os
import pathlib
import tempfile
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException, Request

with tempfile.TemporaryDirectory() as _tmpdir:
    os.environ.setdefault("GRAMMAR_REGISTRY_DIR", os.path.join(_tmpdir, "grammars"))
    os.environ.setdefault("SCHEMA_REGISTRY_DIR", os.path.join(_tmpdir, "schemas"))
    _MODULE_PATH = pathlib.Path(__file__).resolve().parent.parent / "alpaca-proxy.py"
    _SPEC = importlib.util.spec_from_file_location("alpaca_proxy_queueing", _MODULE_PATH)
    alpaca_proxy = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(alpaca_proxy)


def _make_request(path: str, payload: dict, headers: dict[str, str] | None = None, source: str = "unknown"):
    body = json.dumps(payload).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    raw_headers = [(b"content-type", b"application/json")]
    for k, v in (headers or {}).items():
        raw_headers.append((k.lower().encode(), v.encode()))

    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "query_string": b"",
        "headers": raw_headers,
        "client": ("127.0.0.1", 12345),
    }
    req = Request(scope, receive)
    req.state.request_source = source
    return req


@pytest.fixture(autouse=True)
def _clean_proxy_state(tmp_path, monkeypatch):
    alpaca_proxy.active_requests.clear()
    alpaca_proxy.queued_requests.clear()
    alpaca_proxy._slot_waiters.clear()
    alpaca_proxy._slot_capacity_cache.clear()
    alpaca_proxy._waiting_for_router_lock.clear()
    with alpaca_proxy.active_request_details_lock:
        alpaca_proxy.active_request_details.clear()
        alpaca_proxy.completed_requests.clear()
    monkeypatch.setattr(alpaca_proxy, "_COMPLETED_REQUESTS_PATH", str(tmp_path / "completed.json"))
    monkeypatch.setattr(alpaca_proxy, "MODEL_ERRORS_FILE", str(tmp_path / "errors.jsonl"))
    monkeypatch.setattr(alpaca_proxy, "restart_llama_server", AsyncMock(return_value=True))
    monkeypatch.setattr(alpaca_proxy, "wait_for_llama_server_or_restart", AsyncMock(return_value=True))
    monkeypatch.setattr(alpaca_proxy, "restore_slot_cache", AsyncMock(return_value=True))
    monkeypatch.setattr(alpaca_proxy, "save_slot_cache", AsyncMock(return_value=True))
    monkeypatch.setattr(alpaca_proxy, "find_slot_for_request", AsyncMock(return_value=0))
    monkeypatch.setattr(alpaca_proxy, "apply_keep_alive_policy", AsyncMock())
    monkeypatch.setattr(alpaca_proxy, "record_metrics", AsyncMock())
    yield
    alpaca_proxy.active_requests.clear()
    alpaca_proxy.queued_requests.clear()
    alpaca_proxy._slot_waiters.clear()
    alpaca_proxy._slot_capacity_cache.clear()
    alpaca_proxy._waiting_for_router_lock.clear()
    with alpaca_proxy.active_request_details_lock:
        alpaca_proxy.active_request_details.clear()


@pytest.mark.asyncio
async def test_burst_never_exceeds_single_slot_even_with_stale_slots_probe(monkeypatch):
    """15 concurrent requests on a 1-slot model must serialize strictly (max in-flight == 1).

    Even when llama-server's /slots always reports is_processing=False (simulating
    the TOCTOU window before the upstream request reaches llama-server's loop),
    atomic proxy reservation must prevent any stampede.
    """
    backend = "kat-coder--iq4"
    monkeypatch.setattr(alpaca_proxy, "resolve_router_model", AsyncMock(return_value={"backend_model": backend}))
    monkeypatch.setattr(alpaca_proxy, "ensure_model", AsyncMock(return_value={"backend_model": backend}))
    monkeypatch.setattr(alpaca_proxy, "router_model_state", AsyncMock(return_value="loaded"))
    # Deliberately always report slot 0 idle to stress the proxy's own atomic reservation!
    monkeypatch.setattr(
        alpaca_proxy,
        "_fetch_model_slots",
        AsyncMock(return_value=[{"id": 0, "is_processing": False}]),
    )

    in_flight = 0
    max_seen = 0

    class SlowUpstream:
        async def post(self, url, json):
            nonlocal in_flight, max_seen
            in_flight += 1
            max_seen = max(max_seen, in_flight)
            await asyncio.sleep(0.015)
            in_flight -= 1
            resp = MagicMock(status_code=200)
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            }
            return resp

    monkeypatch.setattr(alpaca_proxy, "client_httpx", SlowUpstream())

    tasks = []
    for i in range(15):
        if i % 2 == 0:
            req = _make_request(
                "/v1/chat/completions",
                {"model": "kat-coder:iq4", "messages": [{"role": "user", "content": f"q{i}"}], "stream": False},
            )
            tasks.append(asyncio.create_task(alpaca_proxy.openai_chat_completions(req)))
        else:
            req = _make_request(
                "/api/chat",
                {"model": "kat-coder:iq4", "messages": [{"role": "user", "content": f"q{i}"}], "stream": False},
            )
            tasks.append(asyncio.create_task(alpaca_proxy.chat(req)))

    responses = await asyncio.gather(*tasks)
    assert all(r.status_code == 200 for r in responses)
    assert max_seen == 1, f"Stampede detected: {max_seen} concurrent upstream calls on a 1-slot model"
    assert alpaca_proxy.active_requests.get(backend, 0) == 0
    assert alpaca_proxy.queued_requests.get(backend, 0) == 0
    assert alpaca_proxy._slot_waiters == {}


@pytest.mark.asyncio
async def test_two_slot_model_caps_concurrency_at_two(monkeypatch):
    """A 2-slot model runs up to 2 requests in parallel and never 3."""
    backend = "qwen--2slot"
    monkeypatch.setattr(alpaca_proxy, "resolve_router_model", AsyncMock(return_value={"backend_model": backend}))
    monkeypatch.setattr(alpaca_proxy, "ensure_model", AsyncMock(return_value={"backend_model": backend}))
    monkeypatch.setattr(alpaca_proxy, "router_model_state", AsyncMock(return_value="loaded"))
    monkeypatch.setattr(
        alpaca_proxy,
        "_fetch_model_slots",
        AsyncMock(return_value=[{"id": 0, "is_processing": False}, {"id": 1, "is_processing": False}]),
    )

    in_flight = 0
    max_seen = 0

    class TwoSlotUpstream:
        async def post(self, url, json):
            nonlocal in_flight, max_seen
            in_flight += 1
            max_seen = max(max_seen, in_flight)
            await asyncio.sleep(0.025)
            in_flight -= 1
            resp = MagicMock(status_code=200)
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }
            return resp

    monkeypatch.setattr(alpaca_proxy, "client_httpx", TwoSlotUpstream())

    reqs = [
        _make_request(
            "/v1/chat/completions",
            {"model": "qwen:2slot", "messages": [{"role": "user", "content": f"m{i}"}], "stream": False},
        )
        for i in range(10)
    ]
    responses = await asyncio.gather(*(alpaca_proxy.openai_chat_completions(r) for r in reqs))
    assert all(r.status_code == 200 for r in responses)
    assert max_seen == 2
    assert alpaca_proxy.active_requests.get(backend, 0) == 0
    assert alpaca_proxy.queued_requests.get(backend, 0) == 0


@pytest.mark.asyncio
async def test_high_priority_voice_jumps_ahead_of_low_priority_coding_queue(monkeypatch):
    """High-priority voice traffic runs immediately after the active job, ahead of queued low-priority jobs."""
    backend = "kat--prio"
    monkeypatch.setattr(alpaca_proxy, "resolve_router_model", AsyncMock(return_value={"backend_model": backend}))
    monkeypatch.setattr(alpaca_proxy, "ensure_model", AsyncMock(return_value={"backend_model": backend}))
    monkeypatch.setattr(alpaca_proxy, "router_model_state", AsyncMock(return_value="loaded"))
    monkeypatch.setattr(
        alpaca_proxy,
        "_fetch_model_slots",
        AsyncMock(return_value=[{"id": 0, "is_processing": False}]),
    )

    started_order: list[str] = []
    first_entered = asyncio.Event()
    release_first = asyncio.Event()

    class OrderedUpstream:
        async def post(self, url, json):
            tag = json["messages"][0]["content"]
            started_order.append(tag)
            if tag == "first-active":
                first_entered.set()
                await release_first.wait()
            else:
                await asyncio.sleep(0.01)
            resp = MagicMock(status_code=200)
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {
                "choices": [{"message": {"role": "assistant", "content": tag}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            }
            return resp

    monkeypatch.setattr(alpaca_proxy, "client_httpx", OrderedUpstream())

    t_first = asyncio.create_task(
        alpaca_proxy.openai_chat_completions(
            _make_request(
                "/v1/chat/completions",
                {"model": "kat", "messages": [{"role": "user", "content": "first-active"}], "stream": False},
                source="opencode",
            )
        )
    )
    await first_entered.wait()

    # Queue two low-priority coding jobs first, then a high-priority voice job.
    t_low1 = asyncio.create_task(
        alpaca_proxy.openai_chat_completions(
            _make_request(
                "/v1/chat/completions",
                {"model": "kat", "messages": [{"role": "user", "content": "low-1"}], "stream": False},
                headers={"X-Priority": "low"},
                source="opencode",
            )
        )
    )
    await asyncio.sleep(0.02)
    t_low2 = asyncio.create_task(
        alpaca_proxy.openai_chat_completions(
            _make_request(
                "/v1/chat/completions",
                {"model": "kat", "messages": [{"role": "user", "content": "low-2"}], "stream": False},
                headers={"X-Priority": "low"},
                source="cline",
            )
        )
    )
    await asyncio.sleep(0.02)
    t_voice = asyncio.create_task(
        alpaca_proxy.chat(
            _make_request(
                "/api/chat",
                {"model": "kat", "messages": [{"role": "user", "content": "voice-urgent"}], "stream": False},
                source="voice/assistant",
            )
        )
    )
    await asyncio.sleep(0.02)

    release_first.set()
    results = await asyncio.gather(t_first, t_low1, t_low2, t_voice)
    # The coding job already running is LOW, so the voice request pre-empts it (503 with
    # Retry-After, a "preempted" error) instead of waiting behind it; the queued coding
    # jobs then run after the voice job.
    first, *rest = results
    assert first.status_code == 503
    assert first.headers.get("Retry-After") == str(alpaca_proxy.PROXY_PREEMPT_RETRY_AFTER_S)
    assert json.loads(first.body)["error"]["type"] == "preempted"
    assert all(r.status_code == 200 for r in rest)
    assert started_order == ["first-active", "voice-urgent", "low-1", "low-2"]


@pytest.mark.asyncio
async def test_bounded_queue_rejects_excess_normal_but_reserves_headroom_for_high_priority(monkeypatch):
    """When queue reaches MAX_QUEUE_DEPTH, normal/low requests get 503 queue_full while voice still enters."""
    monkeypatch.setattr(alpaca_proxy, "MAX_QUEUE_DEPTH", 2)
    monkeypatch.setattr(alpaca_proxy, "HIGH_PRIORITY_QUEUE_RESERVE", 2)
    alpaca_proxy.queued_requests["kat--q"] = 2

    normal_req = _make_request(
        "/v1/chat/completions",
        {"model": "kat", "messages": [{"role": "user", "content": "hi"}], "stream": False},
        source="opencode",
    )
    resp = await alpaca_proxy.openai_chat_completions(normal_req)
    assert resp.status_code == 503
    assert resp.headers.get("retry-after") == "5"
    assert json.loads(resp.body)["error"]["code"] == "queue_full"

    # High priority (100) still fits inside HIGH_PRIORITY_QUEUE_RESERVE
    ok, reason = alpaca_proxy._check_queue_admission(alpaca_proxy.PRIORITY_HIGH)
    assert ok is True and reason == ""
    # Once HIGH reserve is exhausted (2 + 2 = 4), HIGH is bounded while CRITICAL (200) still has emergency reserve
    alpaca_proxy.queued_requests["kat--q"] = 4
    ok2, reason2 = alpaca_proxy._check_queue_admission(alpaca_proxy.PRIORITY_HIGH)
    assert ok2 is False and reason2 == "queue_full"
    ok_crit, reason_crit = alpaca_proxy._check_queue_admission(alpaca_proxy.PRIORITY_CRITICAL)
    assert ok_crit is True and reason_crit == ""
    # Once CRITICAL reserve (2 + 4 = 6) is exhausted, even CRITICAL is cleanly bounded
    alpaca_proxy.queued_requests["kat--q"] = 6
    ok_crit2, reason_crit2 = alpaca_proxy._check_queue_admission(alpaca_proxy.PRIORITY_CRITICAL)
    assert ok_crit2 is False and reason_crit2 == "queue_full"


@pytest.mark.asyncio
async def test_sharedllm_numeric_priority_scale_orders_critical_high_normal_low(monkeypatch):
    """Canonical SharedLLM scale (higher = more urgent: 200 > 100 > 50 > 0) orders queued work."""
    backend = "kat--scale"
    monkeypatch.setattr(alpaca_proxy, "resolve_router_model", AsyncMock(return_value={"backend_model": backend}))
    monkeypatch.setattr(alpaca_proxy, "ensure_model", AsyncMock(return_value={"backend_model": backend}))
    monkeypatch.setattr(alpaca_proxy, "router_model_state", AsyncMock(return_value="loaded"))
    monkeypatch.setattr(
        alpaca_proxy,
        "_fetch_model_slots",
        AsyncMock(return_value=[{"id": 0, "is_processing": False}]),
    )

    started_order: list[str] = []
    first_entered = asyncio.Event()
    release_first = asyncio.Event()

    class ScaleUpstream:
        async def post(self, url, json):
            tag = json["messages"][0]["content"]
            started_order.append(tag)
            if tag == "active":
                first_entered.set()
                await release_first.wait()
            else:
                await asyncio.sleep(0.01)
            resp = MagicMock(status_code=200)
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {
                "choices": [{"message": {"role": "assistant", "content": tag}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            }
            return resp

    monkeypatch.setattr(alpaca_proxy, "client_httpx", ScaleUpstream())

    t0 = asyncio.create_task(
        alpaca_proxy.openai_chat_completions(
            _make_request(
                "/v1/chat/completions",
                {"model": "kat", "messages": [{"role": "user", "content": "active"}], "stream": False},
            )
        )
    )
    await first_entered.wait()

    # Enqueue LOW (0), NORMAL (50), HIGH (100), CRITICAL (200) in reverse urgency order.
    queued = []
    for tag, prio in (("low-0", "0"), ("normal-50", "50"), ("high-100", "100"), ("critical-200", "200")):
        queued.append(
            asyncio.create_task(
                alpaca_proxy.chat(
                    _make_request(
                        "/api/chat",
                        {"model": "kat", "messages": [{"role": "user", "content": tag}], "stream": False},
                        headers={"X-Priority": prio},
                    )
                )
            )
        )
        await asyncio.sleep(0.015)

    release_first.set()
    responses = await asyncio.gather(t0, *queued)
    assert all(r.status_code == 200 for r in responses)
    assert started_order == ["active", "critical-200", "high-100", "normal-50", "low-0"]


def test_high_priority_swap_does_not_block_behind_low_priority_other_model_waiters():
    """HIGH/CRITICAL swap for another model only waits for >= priority waiters, not LOW background work."""
    alpaca_proxy.queued_requests["vl--7b"] = 2
    alpaca_proxy._enqueue_slot_waiter("vl--7b", alpaca_proxy.PRIORITY_LOW)
    alpaca_proxy._enqueue_slot_waiter("vl--7b", alpaca_proxy.PRIORITY_LOW)

    # Normal/low caller respects queued waiters on vl--7b to prevent thrashing
    assert alpaca_proxy._queued_blocking_swap("vl--7b", alpaca_proxy.PRIORITY_NORMAL) == 2
    assert alpaca_proxy._queued_blocking_swap("vl--7b", alpaca_proxy.PRIORITY_LOW) == 2
    # High/critical caller does NOT stall behind LOW background jobs on vl--7b
    assert alpaca_proxy._queued_blocking_swap("vl--7b", alpaca_proxy.PRIORITY_HIGH) == 0
    assert alpaca_proxy._queued_blocking_swap("vl--7b", alpaca_proxy.PRIORITY_CRITICAL) == 0

    # If a HIGH waiter is queued on vl--7b, HIGH waits for it while CRITICAL still preempts
    alpaca_proxy._enqueue_slot_waiter("vl--7b", alpaca_proxy.PRIORITY_HIGH)
    assert alpaca_proxy._queued_blocking_swap("vl--7b", alpaca_proxy.PRIORITY_HIGH) == 1
    assert alpaca_proxy._queued_blocking_swap("vl--7b", alpaca_proxy.PRIORITY_CRITICAL) == 0


@pytest.mark.asyncio
async def test_ps_endpoint_reports_zero_available_when_proxy_has_active_or_queued_work(monkeypatch):
    """/api/ps must not report available slots while proxy-local requests hold or wait on them."""
    monkeypatch.setattr(alpaca_proxy, "loaded_models_from_router", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        alpaca_proxy,
        "get_llama_server_slots",
        AsyncMock(return_value={"total": 1, "busy": 0, "available": 1}),
    )
    alpaca_proxy.active_requests["kat--ps"] = 1
    res = await alpaca_proxy.ps()
    assert res["slots"] == {"total": 1, "busy": 1, "available": 0}

    alpaca_proxy.active_requests.clear()
    alpaca_proxy.queued_requests["kat--ps"] = 1
    res2 = await alpaca_proxy.ps()
    assert res2["slots"] == {"total": 1, "busy": 1, "available": 0}

    alpaca_proxy.queued_requests.clear()
    res3 = await alpaca_proxy.ps()
    assert res3["slots"] == {"total": 1, "busy": 0, "available": 1}


@pytest.mark.asyncio
async def test_prompt_keeper_defers_immediately_when_real_work_is_active_or_queued(monkeypatch):
    """Prompt-keeper warmups must never queue ahead of or compete with real requests."""
    alpaca_proxy.queued_requests["kat--k"] = 1
    keeper_req = _make_request(
        "/api/chat",
        {"model": "kat", "messages": [{"role": "user", "content": "."}], "stream": False},
        headers={alpaca_proxy.PROMPT_KEEPER_HEADER: "1"},
    )
    resp = await alpaca_proxy.chat(keeper_req)
    assert resp.status_code == 503
    assert json.loads(resp.body)["status"] == "prompt_keeper_deferred"

    # Background loops also see queued_requests as busy
    monkeypatch.setattr(alpaca_proxy, "HOME_MODEL", "kat")
    assert await alpaca_proxy.home_model_return_once(now=9999999999.0) == "busy"
    alpaca_proxy._kept_prompts["k1"] = {"model": "kat", "stale": True, "path": "/api/chat", "body": {}, "chars": 7000, "source": ""}
    try:
        assert await alpaca_proxy.prompt_keeper_once(now=9999999999.0) == "busy"
    finally:
        alpaca_proxy._kept_prompts.clear()


@pytest.mark.asyncio
async def test_cancelled_waiter_releases_queue_and_wakes_next_request(monkeypatch):
    """A client disconnect (CancelledError) while queued must not leak counters or stall peers."""
    backend = "kat--cancel"
    monkeypatch.setattr(alpaca_proxy, "resolve_router_model", AsyncMock(return_value={"backend_model": backend}))
    monkeypatch.setattr(alpaca_proxy, "ensure_model", AsyncMock(return_value={"backend_model": backend}))
    monkeypatch.setattr(alpaca_proxy, "router_model_state", AsyncMock(return_value="loaded"))
    monkeypatch.setattr(
        alpaca_proxy,
        "_fetch_model_slots",
        AsyncMock(return_value=[{"id": 0, "is_processing": False}]),
    )

    # Hold slot 0 busy initially
    alpaca_proxy._slot_capacity_cache[backend] = 1
    alpaca_proxy.active_requests[backend] = 1

    cancelled_task = asyncio.create_task(
        alpaca_proxy.chat(
            _make_request(
                "/api/chat",
                {"model": "kat", "messages": [{"role": "user", "content": "drop-me"}], "stream": False},
            )
        )
    )
    await asyncio.sleep(0.03)
    assert alpaca_proxy.queued_requests.get(backend, 0) == 1

    cancelled_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_task

    assert alpaca_proxy.queued_requests.get(backend, 0) == 0
    assert alpaca_proxy.active_requests.get(backend, 0) == 1
    assert alpaca_proxy.active_request_details == {}
    assert alpaca_proxy._slot_waiters == {}


@pytest.mark.asyncio
async def test_chat_4xx_does_not_restart_llama_server_and_releases_active_slot(monkeypatch):
    """An upstream 400 error (e.g. context length exceeded) must not restart llama-server or leak slots."""
    backend = "kat--400"
    monkeypatch.setattr(alpaca_proxy, "resolve_router_model", AsyncMock(return_value={"backend_model": backend}))
    monkeypatch.setattr(alpaca_proxy, "ensure_model", AsyncMock(return_value={"backend_model": backend}))
    monkeypatch.setattr(alpaca_proxy, "router_model_state", AsyncMock(return_value="loaded"))
    monkeypatch.setattr(
        alpaca_proxy,
        "_fetch_model_slots",
        AsyncMock(return_value=[{"id": 0, "is_processing": False}]),
    )

    restart_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(alpaca_proxy, "restart_llama_server", restart_mock)

    class BadRequestUpstream:
        async def post(self, url, json):
            req = httpx.Request("POST", url)
            resp = httpx.Response(400, request=req, text='{"error":{"message":"exceeds n_ctx"}}')
            return resp

    monkeypatch.setattr(alpaca_proxy, "client_httpx", BadRequestUpstream())

    with pytest.raises(HTTPException) as exc:
        await alpaca_proxy.chat(
            _make_request(
                "/api/chat",
                {"model": "kat", "messages": [{"role": "user", "content": "too long"}], "stream": False},
            )
        )
    assert exc.value.status_code == 400
    restart_mock.assert_not_awaited()
    assert alpaca_proxy.active_requests.get(backend, 0) == 0
    assert alpaca_proxy.queued_requests.get(backend, 0) == 0


@pytest.mark.asyncio
async def test_queue_wait_endpoint_respects_local_queued_and_active_counts(monkeypatch):
    """/api/queue/wait must not tell background callers 'ready' while proxy work is queued or active."""
    monkeypatch.setattr(
        alpaca_proxy,
        "get_llama_server_slots",
        AsyncMock(return_value={"total": 1, "busy": 0, "available": 1}),
    )
    alpaca_proxy.queued_requests["kat--q"] = 1
    res = await alpaca_proxy.queue_wait(_make_request("/api/queue/wait", {"timeout": 0.05}))
    assert res["status"] == "timeout"

    alpaca_proxy.queued_requests.clear()
    alpaca_proxy.active_requests["kat--q"] = 1
    res2 = await alpaca_proxy.queue_wait(_make_request("/api/queue/wait", {"timeout": 0.05}))
    assert res2["status"] == "timeout"

    alpaca_proxy.active_requests.clear()
    res3 = await alpaca_proxy.queue_wait(_make_request("/api/queue/wait", {"timeout": 0.5}))
    assert res3["status"] == "ready"


@pytest.mark.asyncio
async def test_is_child_model_healthy_accepts_503_no_slot_and_busy_loaded_model(monkeypatch):
    """HTTP 503 'no slot available' or active in-flight traffic must never look unhealthy."""
    backend = "kat--busy"
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    models_resp = MagicMock(status_code=200)
    models_resp.json.return_value = {
        "data": [{"id": backend, "status": {"value": "loaded", "args": ["--port", "18080"]}}]
    }
    fake_client = MagicMock()
    fake_client.get = AsyncMock(return_value=models_resp)
    monkeypatch.setattr(alpaca_proxy, "client_httpx", fake_client)

    # 1. With active_requests > 0, loaded child is immediately healthy without probing /health.
    alpaca_proxy.active_requests[backend] = 2
    assert await alpaca_proxy.is_child_model_healthy(backend) is True
    alpaca_proxy.active_requests.clear()

    # 2. With active_requests == 0, HTTP 503 'no slot available' from /health is still healthy.
    health_503 = MagicMock(status_code=503)
    health_503.json.return_value = {"error": {"code": 503, "message": "no slot available"}}

    class _ProbeClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None):
            return health_503

    monkeypatch.setattr(alpaca_proxy.httpx, "AsyncClient", _ProbeClient)
    assert await alpaca_proxy.is_child_model_healthy(backend) is True


@pytest.mark.asyncio
async def test_ensure_model_impl_skips_restart_when_other_requests_are_active(monkeypatch):
    """_ensure_model_impl must never restart llama-server while other requests hold slots."""
    backend = "kat--in-flight"
    alpaca_proxy.active_requests[backend] = 2  # 1 from ensure_model + 1 real active caller
    monkeypatch.setattr(alpaca_proxy, "is_child_model_healthy", AsyncMock(return_value=False))
    restart_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(alpaca_proxy, "restart_llama_server", restart_mock)
    monkeypatch.setattr(alpaca_proxy, "record_model_loaded", AsyncMock())

    resolved = {
        "model_name": "kat",
        "backend_model": backend,
        "entry": {"id": backend, "status": {"value": "loaded"}},
        "router_models": [{"id": backend, "status": {"value": "loaded"}}],
        "manifest_path": None,
        "manifest": None,
    }
    res = await alpaca_proxy._ensure_model_impl("kat", resolved=resolved)
    assert res["backend_model"] == backend
    restart_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_wait_for_slot_evicts_disconnected_client_immediately(monkeypatch):
    """A disconnected client must leave the queue immediately instead of sitting for 120s."""
    backend = "kat--disc"
    alpaca_proxy._slot_capacity_cache[backend] = 1
    alpaca_proxy.active_requests[backend] = 1
    alpaca_proxy.queued_requests[backend] = 1

    disconnected = False

    async def is_disconnected():
        return disconnected

    async def flip():
        nonlocal disconnected
        await asyncio.sleep(0.05)
        disconnected = True

    flipper = asyncio.create_task(flip())
    t0 = asyncio.get_running_loop().time()
    ok = await alpaca_proxy.wait_for_slot(
        backend,
        timeout=30.0,
        queued_backend=backend,
        is_disconnected=is_disconnected,
    )
    elapsed = asyncio.get_running_loop().time() - t0
    await flipper

    assert ok is False
    assert elapsed < 2.0
    assert not alpaca_proxy._prune_slot_waiters(backend)


@pytest.mark.asyncio
async def test_wait_for_slot_reconciles_drifted_active_requests(monkeypatch):
    """If active_requests ever drifts >= slot_capacity while llama-server is idle, wait_for_slot self-heals."""
    backend = "kat--drift"
    alpaca_proxy._slot_capacity_cache[backend] = 2
    alpaca_proxy.active_requests[backend] = 2  # Simulated permanent leak!
    alpaca_proxy._last_slot_admit_ts[backend] = 0.0  # Old admission (past grace window)
    alpaca_proxy.queued_requests[backend] = 1

    monkeypatch.setattr(alpaca_proxy, "router_model_state", AsyncMock(return_value="loaded"))
    monkeypatch.setattr(
        alpaca_proxy,
        "_fetch_model_slots",
        AsyncMock(return_value=[{"id": 0, "is_processing": False}, {"id": 1, "is_processing": False}]),
    )

    reservation = {"activated": False, "backend_model": backend}
    ok = await alpaca_proxy.wait_for_slot(
        backend,
        timeout=5.0,
        queued_backend=backend,
        reservation=reservation,
        reconcile_interval=0.02,
        reconcile_grace=0.0,
    )
    assert ok is True
    assert reservation["activated"] is True
    assert alpaca_proxy.active_requests[backend] == 1
    assert alpaca_proxy.queued_requests[backend] == 0


@pytest.mark.asyncio
async def test_slot_lease_background_task_is_idempotent_and_prevents_handoff_leak():
    """Calling _SlotLease.release from BackgroundTask and/or generator finally releases exactly once."""
    backend = "kat--lease"
    alpaca_proxy.active_requests[backend] = 1
    lease = alpaca_proxy._SlotLease("req-1", active_backend=backend)

    # First release (e.g. client disconnected before stream_proxy started -> BackgroundTask runs).
    await lease.release()
    assert alpaca_proxy.active_requests[backend] == 0

    # Second release (e.g. generator finally ALSO runs) must be a no-op (never underflows or double-releases).
    alpaca_proxy.active_requests[backend] = 1  # Another peer took the slot
    await lease.release()
    assert alpaca_proxy.active_requests[backend] == 1
