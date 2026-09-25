import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from online_providers import (
    AGENT_HARNESS_SPECS,
    OnlineModelProvider,
    _query_harness_cli,
    build_agent_harness_cmd,
)
from web.shared_llm_benchmark import SharedLLMModelBenchmark


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [{"content-type": "text/event-stream"}, {}])
async def test_openrouter_accumulates_sse_deltas(headers):
    import httpx

    provider = OnlineModelProvider()
    provider.openrouter_api_key = "test-key"
    body = (
        'data: {"choices":[{"delta":{"content":"hello "}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"world"}}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )
    response = httpx.Response(200, headers=headers, content=body)
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=response):
        result = await provider._query_online_model_impl("openrouter:test-model", "hi")
    assert result["success"] is True
    assert result["response"] == "hello world"
    assert result["finish_reason"] == "stop"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "embedded",
    [
        {"message": "Provider timed out after 301s", "code": 504, "metadata": {"error_type": "timeout"}},
        {"message": "Upstream unavailable", "code": 504, "status": "GATEWAY_TIMEOUT"},
        {"message": "Upstream unavailable", "metadata": {"error_type": "timeout"}},
    ],
)
async def test_openrouter_embedded_timeout_diagnosis_and_retry_cap(embedded):
    import httpx

    provider = OnlineModelProvider()
    provider.openrouter_api_key = "test-key"
    response = httpx.Response(200, json={"error": embedded})
    with (
        patch.object(provider, "_resolve_thinking_model", new_callable=AsyncMock, return_value=False),
        patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=response) as post,
        patch("online_providers.asyncio.sleep", new_callable=AsyncMock) as sleep,
    ):
        result = await provider.query_online_model("openrouter:test-model", "hi")
    assert not result["success"]
    assert "HTTP 200 with embedded provider error" in result["error"]
    assert embedded["message"] in result["error"]
    if "code" in embedded:
        assert f"code={embedded['code']}" in result["error"]
    if "metadata" in embedded:
        assert "error_type=timeout" in result["error"]
    if "status" in embedded:
        assert "status=GATEWAY_TIMEOUT" in result["error"]
    assert "possible content filter" not in result["error"]
    assert post.await_count == 2
    sleep.assert_awaited_once()
    assert set(result) == {"success", "latency", "response", "thinking", "finish_reason", "tokens_generated", "error"}


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["code", "status"])
@pytest.mark.parametrize("as_string", [False, True])
@pytest.mark.parametrize("code, attempts", [(500, 5), (502, 5), (503, 5), (504, 2), (401, 1), (403, 1)])
async def test_embedded_http_error_retry_budget(field, as_string, code, attempts):
    import httpx

    provider = OnlineModelProvider()
    provider.openrouter_api_key = "test-key"
    embedded = {field: str(code) if as_string else code, "message": "Provider request failed"}
    response = httpx.Response(200, json={"error": embedded})
    with (
        patch.object(provider, "_resolve_thinking_model", new_callable=AsyncMock, return_value=False),
        patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=response) as post,
        patch("online_providers.asyncio.sleep", new_callable=AsyncMock) as sleep,
    ):
        result = await provider.query_online_model("openrouter:test-model", "hi")
    assert result == {
        "success": False,
        "latency": result["latency"],
        "response": None,
        "thinking": None,
        "finish_reason": None,
        "tokens_generated": 0,
        "error": (
            "OpenRouter returned HTTP 200 with embedded provider error "
            f"(code={embedded.get('code')}, status={embedded.get('status')}, "
            "error_type=None: Provider request failed)"
        ),
    }
    assert post.await_count == attempts
    assert sleep.await_count == attempts - 1


@pytest.mark.parametrize("code", [401, 403])
def test_permanent_error_does_not_retry_transient_message(code):
    provider = OnlineModelProvider()
    error = provider._extract_online_content(
        {"error": {"code": code, "message": "timeout contacting upstream 503"}}, "OpenRouter"
    )[3]
    assert not provider._is_retryable_online_failure({"success": False, "error": error})


@pytest.mark.parametrize("code", [500, 502, 503, 504])
def test_retry_status_requires_explicit_code(code):
    assert OnlineModelProvider._is_retryable_online_failure({"success": False, "error": f"OpenRouter HTTP {code}"})
    assert not OnlineModelProvider._is_retryable_online_failure(
        {"success": False, "error": f"Invalid request: identifier {code}"}
    )


def test_online_content_embedded_error_precedes_choices():
    provider = OnlineModelProvider()
    content, thinking, finish, error = provider._extract_online_content(
        {"error": {"code": 504, "message": "Provider timeout"}, "choices": [{"message": {"content": "partial"}}]},
        "OpenRouter",
    )
    assert content is thinking is finish is None
    assert "code=504" in error
    assert "embedded provider error" in error
    assert "no choices" in provider._extract_online_content({}, "OpenRouter")[3]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", ["OpenRouter HTTP 429", "Online request failed: network timeout"])
async def test_online_rate_limit_and_network_retry_budget_unchanged(error):
    provider = OnlineModelProvider()
    failure = {"success": False, "latency": 1, "response": None, "tokens_generated": 0, "error": error}
    with (
        patch.object(provider, "_resolve_thinking_model", new_callable=AsyncMock, return_value=False),
        patch.object(provider, "_query_online_model_impl", new_callable=AsyncMock, return_value=failure) as query,
        patch("online_providers.asyncio.sleep", new_callable=AsyncMock) as sleep,
    ):
        result = await provider.query_online_model("openrouter:test-model", "hi")
    assert not result["success"]
    assert query.await_count == 5
    assert sleep.await_count == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("exhausted_thinking", [False, True])
async def test_online_latency_includes_retries_backoff_and_continuation(exhausted_thinking):
    import time
    from types import SimpleNamespace

    provider = OnlineModelProvider()
    clock = [100.0]
    attempts = iter(
        [
            {"success": False, "response": None, "finish_reason": None, "error": "HTTP 429"},
            {
                "success": not exhausted_thinking,
                "response": None if exhausted_thinking else "part",
                "finish_reason": "length",
                "error": "empty response" if exhausted_thinking else None,
            },
            {"success": True, "response": "done", "finish_reason": "stop", "error": None},
        ]
    )

    async def query(*args, **kwargs):
        clock[0] += 10
        return {"latency": 10, "thinking": "reasoning", "tokens_generated": 5, **next(attempts)}

    async def sleep(delay):
        clock[0] += delay

    with (
        patch("online_providers.time", SimpleNamespace(time=time.time, monotonic=lambda: clock[0])),
        patch.object(provider, "_resolve_thinking_model", new_callable=AsyncMock, return_value=True),
        patch.object(provider, "_query_online_model_impl", side_effect=query) as impl,
        patch("online_providers.asyncio.sleep", side_effect=sleep),
    ):
        result = await provider.query_online_model("openrouter:test-model", "hi")
    assert result["success"]
    assert result["response"].endswith("done")
    assert result["latency"] == 35.5
    assert result["latency"] > 10
    assert impl.await_count == 3


def test_online_decode_preserves_json_and_malformed_body():
    import httpx

    data, error = OnlineModelProvider._decode_json_response(httpx.Response(200, json={"choices": []}), "OpenRouter")
    assert data == {"choices": []}
    assert error is None
    data, error = OnlineModelProvider._decode_json_response(httpx.Response(200, text="Not Found"), "OpenRouter")
    assert data is None
    assert not error["success"]
    assert "non-JSON body" in error["error"]


@pytest.mark.parametrize("measured", [False, True])
def test_online_tracker_marks_estimates_and_unknown_ttft(monkeypatch, measured):
    from collections import deque

    import online_providers as online

    monkeypatch.setattr(online, "_active_online_requests", {})
    monkeypatch.setattr(online, "_completed_online_requests", deque(maxlen=50))
    online.start_online_request("tracker-test", "openrouter:test", "openrouter", {"prompt": "x" * 4000})
    online.complete_online_request(
        "tracker-test",
        {"success": True, "response": "a response", "tokens_generated": 0, "error": None},
        prompt_tokens=1000 if measured else 0,
        completion_tokens=10 if measured else 0,
    )
    tracked = online.get_online_requests()
    assert tracked["active_requests"] == []
    record = tracked["completed_requests"][0]
    assert record["ttft_seconds"] is None
    assert record["prompt_tokens_estimated"] is not measured
    assert record["completion_tokens_estimated"] is not measured
    assert record["prompt_tokens"] == (1000 if measured else 500)
    assert record["completion_tokens"] == (10 if measured else 2)
    assert record["duration_seconds"] >= 0


def test_online_model_provider_detection():
    provider = OnlineModelProvider()
    assert provider.is_online_model("openrouter:google/gemini-2.0-flash-exp:free") is True
    assert provider.is_online_model("huggingface:meta-llama/Llama-3.1-8B-Instruct") is True
    assert provider.is_online_model("cloudflare:@cf/meta/llama-3.1-8b-instruct") is True
    assert provider.is_online_model("opencode_zen:zen-coder-v1") is True
    assert provider.is_online_model("groq:llama-3.3-70b-versatile") is True
    assert provider.is_online_model("orcarouter:openai/gpt-4o-mini") is True
    assert provider.is_online_model("gemini:gemini-2.5-flash") is True
    assert provider.is_online_model("opencode:opencode/mimo-v2.6-flash-free") is True
    assert provider.is_online_model("qwen2.5-coder:7b") is False


@pytest.mark.asyncio
async def test_shared_benchmark_online_query_bypasses_local_slot_path(monkeypatch):
    import web.shared_llm_benchmark as benchmark_module

    bench = SharedLLMModelBenchmark()
    online_query = AsyncMock(
        return_value={
            "success": True,
            "latency": 0.01,
            "response": "online",
            "tokens_generated": 1,
            "error": None,
        }
    )
    monkeypatch.setattr(benchmark_module.online_model_provider, "is_online_model", lambda model: True)
    monkeypatch.setattr(benchmark_module.online_model_provider, "query_online_model", online_query)
    monkeypatch.setattr(
        benchmark_module.context_awareness,
        "resolve_context_window",
        AsyncMock(side_effect=AssertionError("local context path was entered")),
    )

    result = await bench.query_model("openrouter:test/model", use_proxy=True, prompt="hello")

    assert result["success"] is True
    online_query.assert_awaited_once()
    assert online_query.await_args.kwargs["model_identifier"] == "openrouter:test/model"


def test_online_model_provider_parse():
    provider = OnlineModelProvider()
    p, m = provider.parse_model_identifier("openrouter:meta-llama/llama-3.3-70b-instruct:free")
    assert p == "openrouter"
    assert m == "meta-llama/llama-3.3-70b-instruct:free"

    p, m = provider.parse_model_identifier("hf:meta-llama/Llama-3.1-8B-Instruct")
    assert p == "huggingface"
    assert m == "meta-llama/Llama-3.1-8B-Instruct"

    p, m = provider.parse_model_identifier("llama3:latest")
    assert p == "local"
    assert m == "llama3:latest"

    p, m = provider.parse_model_identifier("groq:llama-3.3-70b-versatile")
    assert p == "groq"
    assert m == "llama-3.3-70b-versatile"

    p, m = provider.parse_model_identifier("orcarouter:openai/gpt-4o-mini")
    assert p == "orcarouter"
    assert m == "openai/gpt-4o-mini"

    p, m = provider.parse_model_identifier("gemini:gemini-2.5-flash")
    assert p == "gemini"
    assert m == "gemini-2.5-flash"

    p, m = provider.parse_model_identifier("opencode:opencode/mimo-v2.6-flash-free")
    assert p == "opencode"
    assert m == "opencode/mimo-v2.6-flash-free"


@pytest.mark.asyncio
async def test_online_model_query_opencode_cli_mock():
    provider = OnlineModelProvider()

    # Simulate `opencode run --format json` NDJSON stdout.
    stdout = (
        b'{"type":"text","part":{"text":"BENCH_OK"}}\n'
        b'{"type":"step_finish","part":{"reason":"stop","tokens":{"output":5}}}\n'
    )

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(stdout, b""))
    mock_proc.kill = MagicMock()
    mock_proc.wait = AsyncMock(return_value=0)

    with (
        patch("shutil.which", return_value="/usr/local/bin/opencode"),
        patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc) as exec_mock,
    ):
        res = await provider.query_online_model(
            "opencode:opencode/mimo-v2.6-flash-free",
            prompt="Reply with exactly: BENCH_OK",
        )
        assert res["success"] is True
        assert res["response"] == "BENCH_OK"
        assert res["tokens_generated"] == 5
        assert res["finish_reason"] == "stop"
        argv = exec_mock.await_args[0]
        assert argv[argv.index("-m") + 1] == "opencode/mimo-v2.6-flash-free"


@pytest.mark.asyncio
async def test_online_model_query_opencode_cli_missing_binary():
    provider = OnlineModelProvider()
    with patch("shutil.which", return_value=None):
        res = await provider.query_online_model(
            "opencode:opencode/mimo-v2.6-flash-free",
            prompt="hi",
        )
    assert res["success"] is False
    assert "opencode CLI" in res.get("error", "")


@pytest.mark.asyncio
async def test_zen_free_tier_403_falls_back_to_opencode_cli():
    """Zen's free tier 403s raw HTTP; the provider must retry via `opencode run`."""
    provider = OnlineModelProvider()
    provider.opencode_zen_base_url = "https://opencode.ai/zen/v1"

    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.text = (
        '{"error":{"message":"OpenCode\'s free tier can only be used from within OpenCode"}}'
    )
    mock_resp.json.return_value = {
        "error": {"message": "OpenCode's free tier can only be used from within OpenCode"}
    }

    stdout = (
        b'{"type":"text","part":{"text":"FALLBACK_OK"}}\n'
        b'{"type":"step_finish","part":{"reason":"stop","tokens":{"output":3}}}\n'
    )
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(stdout, b""))
    mock_proc.kill = MagicMock()
    mock_proc.wait = AsyncMock(return_value=0)

    with (
        patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp),
        patch("shutil.which", return_value="/usr/local/bin/opencode"),
        patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc) as exec_mock,
    ):
        res = await provider.query_online_model(
            "opencode_zen:mimo-v2.6-flash-free",
            prompt="hi",
        )

    assert res["success"] is True
    assert res["response"] == "FALLBACK_OK"
    assert mock_proc.communicate.called
    # Bare Zen ids must be namespace-qualified or `opencode run -m` 500s.
    argv = exec_mock.await_args[0]
    assert argv[argv.index("-m") + 1] == "opencode/mimo-v2.6-flash-free"


@pytest.mark.asyncio
async def test_online_model_query_openrouter_mock():
    provider = OnlineModelProvider()
    provider.openrouter_api_key = "test-key"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "light_on"}}],
        "usage": {"completion_tokens": 5},
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        res = await provider.query_online_model(
            "openrouter:google/gemini-2.0-flash-exp:free",
            prompt="Classify intent",
        )
        assert res["success"] is True
        assert res["response"] == "light_on"
        assert res["tokens_generated"] == 5


@pytest.mark.asyncio
async def test_online_model_query_groq_mock():
    provider = OnlineModelProvider()
    provider.groq_api_key = "test-key"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "42"}}],
        "usage": {"completion_tokens": 3},
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        res = await provider.query_online_model(
            "groq:llama-3.3-70b-versatile",
            prompt="What is the answer?",
        )
        assert res["success"] is True
        assert res["response"] == "42"
        assert res["tokens_generated"] == 3


@pytest.mark.asyncio
async def test_online_model_query_groq_no_key():
    provider = OnlineModelProvider()
    provider.groq_api_key = ""

    res = await provider.query_online_model("groq:llama-3.3-70b-versatile", prompt="hi")
    assert res["success"] is False
    assert "GROQ_API_KEY" in res.get("error", "")


@pytest.mark.asyncio
async def test_online_model_query_orcarouter_mock():
    provider = OnlineModelProvider()
    provider.orcarouter_api_key = "test-key"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "42"}}],
        "usage": {"completion_tokens": 3},
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        res = await provider.query_online_model(
            "orcarouter:openai/gpt-4o-mini",
            prompt="What is the answer?",
        )
        assert res["success"] is True
        assert res["response"] == "42"
        assert res["tokens_generated"] == 3


@pytest.mark.asyncio
async def test_online_headroom_uses_run_budget_floor():
    """Headroom math uses caller-supplied UI values only: max(2*est, budget, base//2)."""
    provider = OnlineModelProvider()

    async def fake_impl(model_id, prompt, max_tokens, *args, **kwargs):
        return {
            "success": True,
            "latency": 0.1,
            "response": "done",
            "thinking": None,
            "finish_reason": "stop",
            "tokens_generated": 5,
            "error": None,
        }

    captured = {}

    async def spy_impl(model_id, prompt, max_tokens, *args, **kwargs):
        captured["max_tokens"] = max_tokens
        return await fake_impl(model_id, prompt, max_tokens, *args, **kwargs)

    with (
        patch.object(provider, "_query_online_model_impl", side_effect=spy_impl),
        patch.object(provider, "_resolve_thinking_model", new_callable=AsyncMock, return_value=True),
    ):
        # base 8000, estimate 0, run budget 2048 -> headroom max(0, 2048, 4000) = 4000
        res = await provider.query_online_model(
            "openrouter:deepseek/deepseek-r1", prompt="hi", max_tokens=8000, reasoning_budget=2048
        )
        assert res["success"] is True
        assert captured["max_tokens"] == 8000 + 4000

        # base 8000, estimate 4096, budget 1024 -> headroom max(8192, 1024, 4000) = 8192
        res = await provider.query_online_model(
            "openrouter:deepseek/deepseek-r1",
            prompt="hi",
            max_tokens=8000,
            reasoning_estimate=4096,
            reasoning_budget=1024,
        )
        assert res["success"] is True
        assert captured["max_tokens"] == 8000 + 8192

        # no UI values at all -> headroom max(0, 0, 4000) = base//2, no literals
        res = await provider.query_online_model("openrouter:deepseek/deepseek-r1", prompt="hi", max_tokens=8000)
        assert res["success"] is True
        assert captured["max_tokens"] == 8000 + 4000


@pytest.mark.asyncio
async def test_online_model_query_orcarouter_no_key():
    provider = OnlineModelProvider()
    provider.orcarouter_api_key = ""

    res = await provider.query_online_model("orcarouter:openai/gpt-4o-mini", prompt="hi")
    assert res["success"] is False
    assert "ORCAROUTER_API_KEY" in res.get("error", "")


@pytest.mark.asyncio
async def test_orcarouter_test_connection_mock():
    provider = OnlineModelProvider()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"data": [{"id": "openai/gpt-4o-mini"}, {"id": "deepseek/deepseek-chat"}]}

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
        res = await provider.test_connection("orcarouter", {"orcarouter_api_key": "test-key"})
        assert res["success"] is True
        assert "2 models" in res.get("message", "")


@pytest.mark.asyncio
async def test_orcarouter_test_connection_no_key():
    provider = OnlineModelProvider()
    provider.orcarouter_api_key = ""

    res = await provider.test_connection("orcarouter", {})
    assert res["success"] is False
    assert "not provided" in res.get("error", "")


@pytest.mark.asyncio
async def test_orcarouter_fetch_live_models_mock():
    provider = OnlineModelProvider()
    provider.orcarouter_api_key = "test-key"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": [
            {"id": "openai/gpt-4o-mini", "owned_by": "openai"},
            {"id": "deepseek/deepseek-chat", "owned_by": "deepseek"},
        ]
    }

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
        results = await provider.fetch_live_models(provider="orcarouter")
        ids = [r["id"] for r in results]
        assert "orcarouter:openai/gpt-4o-mini" in ids
        assert "orcarouter:deepseek/deepseek-chat" in ids
        assert all(r["provider"] == "orcarouter" for r in results)


@pytest.mark.asyncio
async def test_online_model_query_gemini_mock():
    provider = OnlineModelProvider()
    provider.gemini_api_key = "test-key"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [
            {
                "content": {"parts": [{"text": "light_on"}], "role": "model"},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"candidatesTokenCount": 5},
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        res = await provider.query_online_model(
            "gemini:gemini-2.5-flash",
            prompt="Classify intent",
        )
        assert res["success"] is True
        assert res["response"] == "light_on"
        assert res["tokens_generated"] == 5


@pytest.mark.asyncio
async def test_online_model_query_gemini_empty():
    provider = OnlineModelProvider()
    provider.gemini_api_key = "test-key"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [{"content": {"parts": [], "role": "model"}, "finishReason": "MAX_TOKENS"}],
        "usageMetadata": {"candidatesTokenCount": 0},
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        res = await provider.query_online_model(
            "gemini:gemini-2.5-flash",
            prompt="Say hi",
            max_retries=0,
        )
        assert res["success"] is False
        assert "empty response" in res.get("error", "").lower()


@pytest.mark.asyncio
async def test_online_model_query_gemini_no_key():
    provider = OnlineModelProvider()
    provider.gemini_api_key = ""

    res = await provider.query_online_model("gemini:gemini-2.5-flash", prompt="hi")
    assert res["success"] is False
    assert "GEMINI_API_KEY" in res.get("error", "")


@pytest.mark.asyncio
async def test_online_model_query_gemini_429_rate_limit():
    provider = OnlineModelProvider()
    provider.gemini_api_key = "test-key"

    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.json.return_value = {
        "error": {
            "message": "You exceeded your current quota, please check your plan and billing details.",
            "status": "RESOURCE_EXHAUSTED",
        }
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        res = await provider.query_online_model(
            "gemini:gemini-3.7-flash",
            prompt="Say hi",
            max_retries=0,
        )
        assert res["success"] is False
        err = res.get("error", "")
        assert "rate limit exceeded" in err.lower()
        assert "service usage" in err.lower()
        assert "429" in err
        assert "You exceeded your current quota" not in err


@pytest.mark.asyncio
async def test_shared_llm_benchmark_tasks_and_validation():
    bench = SharedLLMModelBenchmark()
    tasks = bench.get_all_tasks()
    assert len(tasks) >= 10

    # Test code validation
    valid_redis_code = """
class MultiTenantLock:
    def __init__(self, redis_client, user_id, resource_id):
        self.redis = redis_client
        self.key = f"lock:{user_id}:{resource_id}"
    def acquire(self, timeout=10):
        return True
    def release(self):
        return True
"""
    val = bench.validate_code(valid_redis_code, task_type="redis_lock")
    assert val["valid_syntax"] is True
    assert val["is_complete"] is True

    # Test JSON validation
    valid_json = '```json\n{"tool": "nextcloud_list_files", "args": {"directory": "/documents"}}\n```'
    j_val = bench.validate_json_payload(valid_json, required_keys=["tool", "args"])
    assert j_val["valid_json"] is True
    assert j_val["is_complete"] is True


def test_shared_llm_strip_thinking_handles_prose_blocks():
    bench = SharedLLMModelBenchmark()
    # XML-style thinking block.
    assert (
        bench.strip_thinking("Let me think<thinking>carefully</thinking> answer is 42") == "Let me think answer is 42"
    )
    # Prose thinking marker used by reasoning models.
    text = " thinking step by step about the request we need to classify response light_on"
    assert bench.strip_thinking(text) == "light_on"
    assert "thinking" not in bench.strip_thinking(text)


def test_shared_llm_clean_json_block_brace_depth():
    bench = SharedLLMModelBenchmark()
    # Trailing prose after the JSON payload must NOT be swallowed by a greedy regex.
    text = 'The answer is {"tool": "lightcontrolrequest", "args": {"entity_id": "light.living_room"}} hope that helps'
    cleaned = bench.clean_json_block(text)
    import json as _json

    parsed = _json.loads(cleaned)
    assert parsed["tool"] == "lightcontrolrequest"
    # Nested braces inside a JSON string value stay balanced.
    nested = '{"note": "like {really} nested", "tool": "contextsearchrequest"}'
    parsed_nested = _json.loads(bench.clean_json_block(nested))
    assert parsed_nested["tool"] == "contextsearchrequest"


def test_shared_llm_json_control_char_repair():
    bench = SharedLLMModelBenchmark()
    # Raw newline inside a JSON string value (common model artifact).
    text = '{"tool": "lightcontrolrequest", "args": {"entity_id": "light\nliving_room"}}'
    val = bench.validate_json_payload(text)
    assert val["valid_json"] is True
    assert val["parsed"]["args"]["entity_id"] == "light\nliving_room"


def test_shared_llm_tool_resolution():
    from web.shared_llm_benchmark import _resolve_tool_name

    # Exact canonical match.
    assert _resolve_tool_name("lightcontrolrequest") == "lightcontrolrequest"
    # Regex alias tier (mirrors downstream app resolution).
    assert _resolve_tool_name("light_control") == "lightcontrolrequest"
    assert _resolve_tool_name("media play request") == "mediaplayrequest"
    assert _resolve_tool_name("context search") == "contextsearchrequest"
    # Fuzzy tier.
    assert _resolve_tool_name("lightcntrolreq") == "lightcontrolrequest"
    # Unresolvable returns empty.
    assert _resolve_tool_name("") == ""
    assert _resolve_tool_name("zzzz_nonsense") == ""
    # Home-Assistant domain__Service names are never canonical request tools:
    # neither the regex tier ("play" inside "player") nor the fuzzy tier may
    # resolve them, matching the gateway's Unknown-tool rejection.
    assert _resolve_tool_name("media_player__HassSetVolume") == ""
    assert _resolve_tool_name("light__HassTurnOn") == ""


def test_shared_llm_tool_request_tasks_and_validation():
    bench = SharedLLMModelBenchmark()
    tool_tasks = [t for t in bench.get_all_tasks() if t["task_type"] == "tool_request"]
    assert len(tool_tasks) >= 4
    ids = {t["id"] for t in tool_tasks}
    assert {
        "tool_request_light_control",
        "tool_request_media_play",
        "tool_request_rag_search",
        "tool_request_git_commit",
    } <= ids

    # Valid canonical tool call passes.
    valid = '{"tool": "lightcontrolrequest", "args": {"entity_id": "light.living_room", "brightness_pct": 80}}'
    val = bench.validate_json_payload(valid)
    assert val["valid_json"] is True

    # A hallucinated/alias name that resolves to the wrong tool must fail.
    wrong = '{"tool": "mediaplayrequest", "args": {"player": "kitchen", "media_id": "x"}}'
    val_wrong = bench.validate_json_payload(wrong)
    assert val_wrong["valid_json"] is True  # parses fine, resolution check rejects it


def test_shared_llm_raven_plan_tasks_present():
    bench = SharedLLMModelBenchmark()
    plan_tasks = [t for t in bench.get_all_tasks() if t["task_type"] == "raven_plan"]
    assert len(plan_tasks) >= 1
    assert plan_tasks[0]["id"] == "code_raven_plan_multi_step"


@pytest.mark.asyncio
async def test_shared_llm_benchmark_run_with_mock(tmp_path):
    bench = SharedLLMModelBenchmark()
    real_results_dir = bench.RESULTS_DIR
    real_models_dir = bench.MODELS_DIR

    def _real_files() -> set[str]:
        snap_names = (
            {f.name for f in real_results_dir.glob("shared_llm_benchmarks_*.json")}
            if real_results_dir.exists()
            else set()
        )
        model_names = {f.name for f in real_models_dir.glob("shared_*.json")} if real_models_dir.exists() else set()
        return snap_names | model_names

    real_files_before = _real_files()

    # Isolate writes: never pollute the real data dirs with mock-run artifacts.
    bench.RESULTS_DIR = tmp_path / "results"
    bench.MODELS_DIR = tmp_path / "results" / "models"
    bench.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bench.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    mock_query = AsyncMock(
        return_value={
            "success": True,
            "latency": 0.15,
            "response": "light_on",
            "tokens_generated": 10,
            "error": None,
        }
    )

    with patch.object(bench, "query_model", side_effect=mock_query):
        results = await bench.run_shared_llm_benchmarks(
            models=["openrouter:google/gemini-2.0-flash-exp:free"],
            use_proxy=True,
            task_ids=["fast_path_light"],
        )
        assert results["status"] == "completed"
        assert len(results["results"]) == 1
        model_tasks = results["results"][0]["tasks"]
        assert model_tasks[0]["test_id"] == "fast_path_light"
    # No artifacts may leak into the real benchmark data dirs.
    assert _real_files() == real_files_before


@pytest.mark.asyncio
async def test_online_providers_discovery_all_providers():
    provider = OnlineModelProvider()

    # 1. OpenCode Zen Discovery
    provider.opencode_zen_base_url = "https://opencode.ai/zen/v1"
    mock_zen_resp = MagicMock()
    mock_zen_resp.status_code = 200
    mock_zen_resp.json.return_value = {
        "data": [
            {"id": "deepseek-v4-flash-free", "name": "deepseek-v4-flash-free"},
            {"id": "claude-opus-5", "name": "claude-opus-5"},
        ]
    }

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_zen_resp):
        zen_models = await provider.fetch_live_models(provider="opencode_zen", free_only=False)
        assert len(zen_models) == 2
        free_m = [m for m in zen_models if m["free"] is True]
        paid_m = [m for m in zen_models if m["free"] is False]
        assert len(free_m) == 1
        assert free_m[0]["name"] == "deepseek-v4-flash-free"
        assert free_m[0]["id"] == "opencode:opencode/deepseek-v4-flash-free"
        assert free_m[0]["provider"] == "opencode"
        assert free_m[0]["free_tier"] == "Free (Zen No-Key)"
        assert len(paid_m) == 1
        assert paid_m[0]["name"] == "claude-opus-5"

    # 2. Cloudflare Discovery with Free Tier (10k Neurons/day)
    mock_cf_resp = MagicMock()
    mock_cf_resp.status_code = 200
    mock_cf_resp.json.return_value = {
        "result": [{"name": "@cf/meta/llama-3.3-70b-instruct", "description": "Llama 3.3 70B Instruct"}]
    }
    provider.cloudflare_account_id = "test-acc"
    provider.cloudflare_api_token = "test-tok"

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_cf_resp):
        cf_models = await provider.fetch_live_models(provider="cloudflare", free_only=False)
        assert len(cf_models) >= 1
        assert cf_models[0]["provider"] == "cloudflare"
        assert cf_models[0]["free"] is True
        assert "10k Neurons/day Free" in cf_models[0]["free_tier"]


def test_online_model_thinking_detection():
    provider = OnlineModelProvider()
    # Name-hint fallback (last resort when a provider publishes no metadata).
    assert provider._is_thinking_model_name("deepseek-v4-flash-free") is True
    assert provider._is_thinking_model_name("qwen/qwen3.6-27b") is True
    assert provider._is_thinking_model_name("gpt-oss-20b") is True
    assert provider._is_thinking_model_name("gemini-2.5-flash") is True
    assert provider._is_thinking_model_name("llama-3.3-70b-versatile") is False
    assert provider._is_thinking_model_name("") is False


def test_online_model_thinking_metadata_from_selected_list():
    """The reasoning flag persisted at selection time is authoritative."""
    provider = OnlineModelProvider()
    selected = [
        {
            "id": "groq:qwen/qwen3.6-27b",
            "provider": "groq",
            "reasoning": True,
        },
        {
            "id": "groq:llama-3.3-70b-versatile",
            "provider": "groq",
            "reasoning": False,
        },
    ]
    with patch.object(provider, "get_selected_models", return_value=selected):
        assert provider._is_thinking_model("groq:qwen/qwen3.6-27b") is True
        assert provider._is_thinking_model("groq:llama-3.3-70b-versatile") is False


def test_online_model_thinking_metadata_from_provider_cache():
    """Provider metadata captured during live discovery drives detection."""
    provider = OnlineModelProvider()
    provider._cached_live_models["groq"] = [
        {"id": "qwen/qwen3.6-27b", "supported_features": ["reasoning"]},
        {"id": "llama-3.3-70b-versatile", "supported_features": ["json_mode"]},
    ]
    assert provider._get_provider_model_metadata("groq:qwen/qwen3.6-27b") is True
    assert provider._get_provider_model_metadata("groq:llama-3.3-70b-versatile") is False


def test_online_model_thinking_gemini_metadata_from_provider_cache():
    provider = OnlineModelProvider()
    provider._cached_live_models["gemini"] = [
        {"name": "models/gemini-2.5-flash", "thinking": True},
        {"name": "models/gemma-2-27b", "thinking": False},
    ]
    assert provider._get_provider_model_metadata("gemini:gemini-2.5-flash") is True
    assert provider._get_provider_model_metadata("gemini:gemma-2-27b") is False


@pytest.mark.asyncio
async def test_online_model_query_thinking_model_inflates_budget():
    """Thinking models get a larger max_tokens budget and a budget warning injected."""
    provider = OnlineModelProvider()
    provider.groq_api_key = "test-key"
    # Simulate metadata already discovered from the Groq API.
    provider._cached_live_models["groq"] = [{"id": "qwen/qwen3.6-27b", "supported_features": ["reasoning"]}]

    captured = {}

    def _post(url, **kwargs):
        captured.update(kwargs)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "42"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 3},
        }
        return mock_resp

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=_post):
        res = await provider.query_online_model(
            "groq:qwen/qwen3.6-27b",
            prompt="What is the answer?",
            max_tokens=2000,
        )
        assert res["success"] is True
        assert res["response"] == "42"
        assert res["tokens_generated"] == 3
        assert captured["json"]["max_tokens"] > 2000
        assert "[System:" in captured["json"]["messages"][0]["content"]
        assert "token budget" in captured["json"]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_online_model_query_non_thinking_uses_exact_budget():
    """Non-thinking models must keep the caller's max_tokens and no injected warning."""
    provider = OnlineModelProvider()
    provider.groq_api_key = "test-key"
    # Simulate metadata already discovered from the Groq API (no reasoning feature).
    provider._cached_live_models["groq"] = [{"id": "llama-3.3-70b-versatile", "supported_features": ["json_mode"]}]

    captured = {}

    def _post(url, **kwargs):
        captured.update(kwargs)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "42"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 3},
        }
        return mock_resp

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=_post):
        res = await provider.query_online_model(
            "groq:llama-3.3-70b-versatile",
            prompt="What is the answer?",
            max_tokens=2000,
        )
        assert res["success"] is True
        assert captured["json"]["max_tokens"] == 2000
        assert captured["json"]["messages"][0]["content"] == "What is the answer?"


@pytest.mark.asyncio
async def test_online_model_query_thinking_model_phase2_continuation():
    """A length-truncated thinking-model completion is continued in phase 2 and merged."""
    provider = OnlineModelProvider()
    provider.groq_api_key = "test-key"
    # Metadata cached so detection uses provider data, not name hints / live fetch.
    provider._cached_live_models["groq"] = [{"id": "qwen/qwen3.6-27b", "supported_features": ["reasoning"]}]

    calls = []

    def _post(url, **kwargs):
        calls.append(kwargs)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        if len(calls) == 1:
            # First call: thinking block ate the budget, partial answer, length cutoff.
            mock_resp.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": "<think>thinking...</think>\nHere is the code:",
                        },
                        "finish_reason": "length",
                    }
                ],
                "usage": {"completion_tokens": 2000},
            }
        else:
            # Phase-2 continuation returns the rest.
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 50},
            }
        return mock_resp

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=_post):
        res = await provider.query_online_model(
            "groq:qwen/qwen3.6-27b",
            prompt="Write code",
            max_tokens=2000,
        )
        assert res["success"] is True
        assert res["response"].endswith("done")
        assert "Here is the code:" in res["response"]
        assert res["tokens_generated"] == 2050
        assert len(calls) == 2
        # Phase-2 prompt asks to finish and includes the prior partial output.
        assert "finish" in calls[1]["json"]["messages"][0]["content"].lower()


# ---------------------------------------------------------------------------
# Streaming benchmark requests (regression: non-streaming hard timeouts killed
# healthy long generations on large-prompt models - llama-server logged
# "Connection handling canceled" and benchmarks scored empty results).
# ---------------------------------------------------------------------------


def _chat_ndjson(chunks, final_metrics):
    lines = []
    for c in chunks:
        lines.append(json.dumps({"message": {"role": "assistant", **c}, "done": False}))
    lines.append(json.dumps({"message": {"role": "assistant", "content": ""}, "done": True, **final_metrics}))
    return ("\n".join(lines) + "\n").encode()


def _generate_ndjson(text_parts, final_metrics):
    lines = [json.dumps({"response": t, "done": False}) for t in text_parts]
    lines.append(json.dumps({"done": True, **final_metrics}))
    return ("\n".join(lines) + "\n").encode()


@pytest.mark.asyncio
@pytest.mark.parametrize("module_path", ["web.shared_llm_benchmark", "llm_benchmark_suite"])
async def test_read_chat_stream_accumulates_content_thinking_metrics(module_path):
    from importlib import import_module

    import httpx

    mod = import_module(module_path)
    payload = _chat_ndjson(
        [{"content": "def foo():"}, {"thinking": "step"}, {"content": "\n    return 1"}],
        {"eval_count": 42, "eval_duration": 9000, "prompt_eval_count": 10, "prompt_eval_duration": 300},
    )
    data = await mod._read_chat_stream(httpx.Response(200, content=payload))
    assert data["content"] == "def foo():\n    return 1"
    assert data["thinking"] == "step"
    assert data["eval_count"] == 42
    assert data["eval_duration"] == 9000
    assert data["prompt_eval_count"] == 10
    assert data["prompt_eval_duration"] == 300


@pytest.mark.asyncio
async def test_read_chat_stream_no_done_frame_yields_zero_metrics():
    import httpx

    from web.shared_llm_benchmark import _read_chat_stream

    raw = json.dumps({"message": {"role": "assistant", "content": "partial"}}).encode() + b"\n"
    data = await _read_chat_stream(httpx.Response(200, content=raw))
    assert data["content"] == "partial"
    assert data["eval_count"] == 0


@pytest.mark.asyncio
async def test_query_model_impl_streams_via_proxy_and_continues_phase2(tmp_path, monkeypatch):
    """Regression: proxy benchmark requests must use stream=True so slow
    generations are not killed by a hard client timeout."""
    from unittest.mock import patch

    import httpx

    from web.shared_llm_benchmark import SharedLLMModelBenchmark

    ini = tmp_path / "models.ini"
    ini.write_text("[*]\ntemperature = 0.5\n")
    monkeypatch.setenv("MODELS_INI_PATH", str(ini))

    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/admin/runtime":
            return httpx.Response(
                200,
                json={"loaded_models": [
                    {"name": "local-model--latest", "backend_model": "local-model--latest",
                     "running_settings": {"ctx-size": "8192"}}
                ]},
            )
        body = json.loads(request.content.decode())
        seen.append({"url": str(request.url), "json": body})
        if len(seen) == 1:
            # First attempt exhausts the token budget (7 >= max_tokens 4).
            raw = _chat_ndjson(
                [{"content": "part1"}, {"content": "-end"}],
                {"eval_count": 7, "eval_duration": 5000, "prompt_eval_duration": 100},
            )
            return httpx.Response(200, content=raw)
        raw = _chat_ndjson([{"content": "+part2"}], {"eval_count": 3, "eval_duration": 10})
        return httpx.Response(200, content=raw)

    factory_calls = []

    real_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        kwargs.pop("timeout", None)
        factory_calls.append(kwargs)
        return real_client(transport=httpx.MockTransport(handler))

    bench = SharedLLMModelBenchmark()
    with patch("web.shared_llm_benchmark.httpx.AsyncClient", side_effect=client_factory):
        res = await bench.query_model(model="local-model", use_proxy=True, prompt="hi", max_tokens=4)

    assert len(seen) == 2
    assert seen[0]["url"].endswith("/api/chat")
    assert seen[0]["json"]["stream"] is True
    assert seen[1]["json"]["stream"] is True
    assert seen[1]["json"]["messages"][1]["role"] == "assistant"
    assert res["success"] is True
    assert res["response"].replace("\n", "").endswith("part2")
    assert res["tokens_generated"] == 10


@pytest.mark.asyncio
async def test_query_model_impl_direct_generate_streams(tmp_path, monkeypatch):
    from unittest.mock import patch

    import httpx

    from web.shared_llm_benchmark import SharedLLMModelBenchmark

    ini = tmp_path / "models.ini"
    ini.write_text("[*]\ntemperature = 0.5\n")
    monkeypatch.setenv("MODELS_INI_PATH", str(ini))

    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/admin/runtime":
            return httpx.Response(
                200,
                json={"loaded_models": [
                    {"name": "local-model--latest", "backend_model": "local-model--latest",
                     "running_settings": {"ctx-size": "8192"}}
                ]},
            )
        body = json.loads(request.content.decode())
        seen.append({"url": str(request.url), "json": body})
        raw = _generate_ndjson(["answer"], {"eval_count": 2, "eval_duration": 20})
        return httpx.Response(200, content=raw)

    real_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        kwargs.pop("timeout", None)
        return real_client(transport=httpx.MockTransport(handler))

    bench = SharedLLMModelBenchmark()
    monkeypatch.setattr(bench, "OLLAMA_SERVER_URLS", ["http://direct-mock"])
    with patch("web.shared_llm_benchmark.httpx.AsyncClient", side_effect=client_factory):
        res = await bench.query_model(model="local-model", use_proxy=False, prompt="hi", max_tokens=100)

    assert seen[0]["url"].endswith("/api/generate")
    assert seen[0]["json"]["stream"] is True
    assert res["success"] is True
    assert res["response"] == "answer"
    assert res["tokens_generated"] == 2


@pytest.mark.asyncio
async def test_suite_test_model_proxy_streams_payload(monkeypatch, tmp_path):
    """llm_benchmark_suite.test_model must stream too (General Benchmarks path)."""
    from unittest.mock import patch

    import httpx

    import llm_benchmark_suite as suite_mod

    ini = tmp_path / "models.ini"
    ini.write_text("[*]\ntemperature = 0.5\n")
    monkeypatch.setenv("MODELS_INI_PATH", str(ini))

    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/admin/runtime":
            return httpx.Response(
                200,
                json={"loaded_models": [
                    {"name": "local-model--latest", "backend_model": "local-model--latest",
                     "running_settings": {"ctx-size": "8192"}}
                ]},
            )
        body = json.loads(request.content.decode())
        seen.append(body)
        raw = _chat_ndjson([{"content": "42"}], {"eval_count": 3, "eval_duration": 60})
        return httpx.Response(200, content=raw)

    real_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        kwargs.pop("timeout", None)
        return real_client(transport=httpx.MockTransport(handler))

    s = suite_mod.LLMModelBenchmark()
    test = {"id": "t1", "prompt": "life?", "reasoning_budget": 512}
    with patch.object(suite_mod.httpx, "AsyncClient", side_effect=client_factory):
        res = await s.test_model_proxy("local-model", test)

    assert seen and seen[0]["stream"] is True
    assert res["success"] is True
    assert res["response"] == "42"
    assert res["tokens_generated"] == 3


@pytest.mark.asyncio
async def test_shared_llm_online_code_task_disables_reasoning(monkeypatch):
    bench = SharedLLMModelBenchmark()
    captured = {}

    async def fake_query(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "latency": 0.1,
            "response": "code",
            "tokens_generated": 12,
            "error": None,
            "reasoning_enabled": False,
            "provider_thinking": True,
            "effective_max_tokens": kwargs["max_tokens"],
            "continuation_count": 0,
            "retry_count": 0,
        }

    monkeypatch.setattr("web.shared_llm_benchmark.online_model_provider.query_online_model", fake_query)
    result = await bench.query_model(
        "openrouter:stealth/space-bunny-alpha",
        use_proxy=True,
        prompt="Write code",
        max_tokens=700,
        reasoning_estimate=2048,
        benchmark_mode="code",
    )

    assert result["success"] is True
    assert captured["max_tokens"] == 700
    assert captured["benchmark_mode"] == "code"
    assert captured["thinking_override"] is False
    assert captured["allow_continuation"] is False
    assert captured["max_retries"] == 1
    assert result["effective_max_tokens"] == 700
    assert result["reasoning_enabled"] is False
    assert result["provider_thinking"] is True


def test_build_provenance_and_learning_record_contract():
    from online_providers import build_learning_record, build_provenance

    prov = build_provenance(
        model="openrouter:vendor/model",
        source="alpaca",
        harness="shared_llm_benchmark",
        transport="api",
        run_id="run-1",
    )
    assert prov["schema_version"] == 1
    assert prov["run_id"] == "run-1"
    assert prov["provider"] == "openrouter"
    assert prov["model_revision"] == "vendor/model"
    assert prov["learning_policy"] == "excluded_by_default"

    shared = build_provenance(
        model="openrouter:vendor/model",
        source="alpaca",
        harness="shared_llm_benchmark",
        transport="api",
        run_id="run-2",
        learning_policy="shared_llm_success_only",
    )
    eligible = build_learning_record({"test_id": "t1", "success": True, "response": "ok", "provenance": shared})
    assert eligible["training_eligible"] is True
    assert eligible["exclusion_reason"] is None
    assert eligible["source_record_id"] == "run-2:t1"

    failed = build_learning_record({"test_id": "t2", "success": False, "response": "bad", "provenance": shared})
    assert failed["training_eligible"] is False
    assert failed["exclusion_reason"] == "failed_grader"

    repaired = build_learning_record(
        {"test_id": "t3", "success": True, "response": "ok", "repaired": True, "provenance": shared}
    )
    assert repaired["training_eligible"] is False
    assert repaired["exclusion_reason"] == "repaired_output"

    excluded = build_learning_record({"test_id": "t4", "success": True, "response": "ok", "provenance": prov})
    assert excluded["training_eligible"] is False
    assert excluded["exclusion_reason"] == "policy_excludes_source"


@pytest.mark.asyncio
async def test_shared_llm_run_records_provenance(tmp_path):
    bench = SharedLLMModelBenchmark()
    bench.RESULTS_DIR = tmp_path / "results"
    bench.MODELS_DIR = bench.RESULTS_DIR / "models"
    bench.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bench.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mock_query = AsyncMock(
        return_value={
            "success": True,
            "latency": 0.1,
            "response": "light_on",
            "tokens_generated": 5,
            "error": None,
        }
    )
    with patch.object(bench, "query_model", side_effect=mock_query):
        results = await bench.run_shared_llm_benchmarks(
            models=["openrouter:vendor/model"],
            use_proxy=True,
            task_ids=["fast_path_light"],
        )

    assert results["run_id"].startswith("sharedllm-")
    assert results["provenance"]["harness"] == "shared_llm_benchmark"
    assert results["provenance"]["transport"] == "api"
    assert results["provenance"]["run_id"] == results["run_id"]
    assert results["provenance"]["finished_at"]

    model_record = results["results"][0]
    assert model_record["run_id"] == results["run_id"]
    assert model_record["provenance"]["model"] == "openrouter:vendor/model"
    task = model_record["tasks"][0]
    assert task["source_record_id"] == f"{results['run_id']}:fast_path_light"
    assert task["provenance"]["run_id"] == results["run_id"]
    assert task["training_eligible"] is True
    assert task["exclusion_reason"] is None

    per_model = json.loads((bench.MODELS_DIR / "shared_openrouter_vendor_model.json").read_text())
    assert per_model["run_id"] == results["run_id"]
    assert per_model["provenance"]["run_id"] == results["run_id"]


def test_cline_model_identifier_detection_and_parsing():
    provider = OnlineModelProvider()
    assert provider.is_online_model("cline_pass:cline-pass/qwen3.7-max") is True
    assert provider.is_online_model("cline:cline-pass/qwen3.7-max") is True
    assert provider.is_online_model("cline-pass:glm-5.3") is True

    provider_name, model_name = provider.parse_model_identifier("cline_pass:cline-pass/qwen3.7-max")
    assert provider_name == "cline_pass"
    assert model_name == "cline-pass/qwen3.7-max"

    provider_name, model_name = provider.parse_model_identifier("cline:cline-pass/qwen3.7-max")
    assert provider_name == "cline"
    assert model_name == "cline-pass/qwen3.7-max"

    provider_name, model_name = provider.parse_model_identifier("cline-pass:glm-5.3")
    assert provider_name == "cline_pass"
    assert model_name == "glm-5.3"


@pytest.mark.asyncio
async def test_online_model_query_cline_pass_mock():
    provider = OnlineModelProvider()
    provider.cline_api_key = "test-key"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "66"}}],
        "usage": {"completion_tokens": 3},
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        res = await provider.query_online_model("cline_pass:cline-pass/qwen3.7-max", prompt="What is 6*11?")
        assert res["success"] is True
        assert res["response"] == "66"
        assert res["tokens_generated"] == 3


@pytest.mark.asyncio
async def test_online_model_query_cline_pass_no_key():
    provider = OnlineModelProvider()
    provider.cline_api_key = ""

    res = await provider.query_online_model("cline_pass:cline-pass/qwen3.7-max", prompt="hi")
    assert res["success"] is False
    assert "CLINE_API_KEY" in res.get("error", "")


@pytest.mark.asyncio
async def test_cline_test_connection_mock():
    provider = OnlineModelProvider()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"data": [{"id": "cline-pass/glm-5.3"}, {"id": "cline-pass/kimi-k3"}]}

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
        res = await provider.test_connection("cline_pass", {"cline_api_key": "test-key"})
        assert res["success"] is True
        assert "2 models" in res.get("message", "")


@pytest.mark.asyncio
async def test_cline_test_connection_no_key():
    provider = OnlineModelProvider()
    provider.cline_api_key = ""

    res = await provider.test_connection("cline_pass", {})
    assert res["success"] is False
    assert "not provided" in res.get("error", "")


@pytest.mark.asyncio
async def test_cline_fetch_live_models_mock(monkeypatch):
    provider = OnlineModelProvider()
    provider.cline_api_key = "test-key"
    monkeypatch.setattr("online_providers._cline_cli_available", lambda: True)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": [
            {"id": "cline-pass/glm-5.3", "context_length": 200000},
            {"id": "cline-pass/kimi-k3"},
        ]
    }

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
        results = await provider.fetch_live_models(provider="cline_pass")
        ids = [r["id"] for r in results]
        assert "cline_pass:cline-pass/glm-5.3" in ids
        assert "cline_pass:cline-pass/kimi-k3" in ids
        assert all(r["provider"] == "cline_pass" for r in results)

        harness_results = await provider.fetch_live_models(provider="cline")
        harness_ids = [r["id"] for r in harness_results]
        assert "cline:default" in harness_ids
        assert "cline:cline-pass/glm-5.3" in harness_ids
        assert all(r["provider"] == "cline" for r in harness_results)


@pytest.mark.asyncio
async def test_cline_harness_missing_binary(monkeypatch):
    provider = OnlineModelProvider()
    monkeypatch.delenv("CLINE_DOCKER_IMAGE", raising=False)
    provider.cline_bin = ""

    with patch("shutil.which", return_value=None):
        res = await provider.query_online_model("cline:cline-pass/qwen3.7-max", prompt="hi")
        assert res["success"] is False
        assert "not found" in res.get("error", "").lower()
        assert "CLINE_DOCKER_IMAGE" in res.get("error", "")


def test_agent_harness_identifier_detection_and_parsing():
    provider = OnlineModelProvider()
    for harness in ("claude", "codex", "deepseek", "pi"):
        assert provider.is_online_model(f"{harness}:default") is True
    assert provider.is_online_model("claude-code:default") is True
    assert provider.is_online_model("codex-cli:gpt-5-codex") is True
    assert provider.is_online_model("dsh:deepseek-chat") is True
    assert provider.is_online_model("pi-cli:default") is True
    assert provider.is_online_model("qwen2.5-coder:7b") is False

    assert provider.parse_model_identifier("claude:claude-sonnet-4-5") == ("claude", "claude-sonnet-4-5")
    assert provider.parse_model_identifier("codex:gpt-5-codex") == ("codex", "gpt-5-codex")
    assert provider.parse_model_identifier("deepseek:deepseek-chat") == ("deepseek", "deepseek-chat")
    assert provider.parse_model_identifier("pi:default") == ("pi", "default")
    assert provider.parse_model_identifier("claude-code:default") == ("claude", "default")
    assert provider.parse_model_identifier("codex-cli:gpt-5") == ("codex", "gpt-5")
    assert provider.parse_model_identifier("dsh:deepseek-reasoner") == ("deepseek", "deepseek-reasoner")
    assert provider.parse_model_identifier("pi-cli:model-x") == ("pi", "model-x")


def test_agent_harness_specs_expose_verified_defaults():
    claude = AGENT_HARNESS_SPECS["claude"]
    assert (claude["binary"], claude["docker_image_env"], claude["key_env"]) == (
        "claude",
        "CLAUDE_DOCKER_IMAGE",
        "ANTHROPIC_API_KEY",
    )
    assert claude["build_args"]("default", "hi", False) == [
        "--bare",
        "-p",
        "hi",
        "--output-format",
        "json",
        "--permission-mode",
        "dontAsk",
    ]
    assert claude["build_args"]("claude-sonnet-4-5", "hi", False)[-2:] == ["--model", "claude-sonnet-4-5"]

    codex = AGENT_HARNESS_SPECS["codex"]
    assert (codex["binary"], codex["docker_image_env"], codex["key_env"]) == (
        "codex",
        "CODEX_DOCKER_IMAGE",
        "CODEX_API_KEY",
    )
    assert codex["build_args"]("default", "hi", False) == [
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--sandbox",
        "workspace-write",
        "--",
        "hi",
    ]

    deepseek = AGENT_HARNESS_SPECS["deepseek"]
    assert (deepseek["binary"], deepseek["docker_image_env"], deepseek["key_env"]) == (
        "dsh",
        "DEEPSEEK_DOCKER_IMAGE",
        "DEEPSEEK_API_KEY",
    )
    assert deepseek["extra_env"] == ("DEEPSEEK_BASE_URL",)
    assert deepseek["build_args"]("default", "hi", False) == ["--profile", "headless", "--", "hi"]

    pi = AGENT_HARNESS_SPECS["pi"]
    assert (pi["binary"], pi["docker_image_env"]) == ("pi", "PI_DOCKER_IMAGE")
    assert pi["build_args"]("pi-mini", "hi", False) == ["--mode", "json", "--print", "hi", "--model", "pi-mini"]


def test_build_agent_harness_cmd_docker_prefix_and_env():
    cmd = build_agent_harness_cmd(
        "claude",
        "claude-sonnet-4-5",
        "Say hi",
        container=True,
        docker_image="alpaca-claude:latest",
        data_dir="/tmp/claude-data",
        env={"ANTHROPIC_API_KEY": "sk-ant-test"},
    )
    assert cmd[:6] == ["docker", "run", "--rm", "--network", "host", "-i"]
    assert cmd[6:10] == ["-e", "ANTHROPIC_API_KEY=sk-ant-test", "-v", "/tmp/claude-data:/data"]
    assert cmd[10:12] == ["-e", "HOME=/data"]
    image_idx = cmd.index("alpaca-claude:latest")
    assert cmd[image_idx + 1] == "claude"
    assert cmd[image_idx + 2 :] == [
        "--bare",
        "-p",
        "Say hi",
        "--output-format",
        "json",
        "--permission-mode",
        "dontAsk",
        "--model",
        "claude-sonnet-4-5",
    ]

    native = build_agent_harness_cmd("deepseek", "default", "hi", container=False, env={"DEEPSEEK_BIN": "/opt/dsh"})
    assert native == ["/opt/dsh", "--profile", "headless", "--", "hi"]


def test_agent_harness_claude_json_output_parse():
    parsed = AGENT_HARNESS_SPECS["claude"]["parse"](
        '{"type":"result","subtype":"success","is_error":false,"result":"FINAL",'
        '"usage":{"output_tokens":12},"stop_reason":"end_turn"}',
        "",
    )
    assert parsed["response"] == "FINAL"
    assert parsed["tokens_generated"] == 12
    assert parsed["finish_reason"] == "end_turn"
    assert parsed["error"] is None

    failed = AGENT_HARNESS_SPECS["claude"]["parse"](
        '{"type":"result","subtype":"error_max_turns","is_error":true,"result":"hit max turns"}',
        "",
    )
    assert "max turns" in failed["error"]


def test_agent_harness_codex_jsonl_output_parse():
    stdout = "\n".join(
        [
            '{"type":"item.completed","item":{"type":"reasoning","text":"thinking"}}',
            '{"type":"item.completed","item":{"type":"agent_message","text":"FIRST"}}',
            '{"type":"item.completed","item":{"type":"agent_message","text":"LAST"}}',
            '{"type":"turn.completed","usage":{"output_tokens":9}}',
        ]
    )
    parsed = AGENT_HARNESS_SPECS["codex"]["parse"](stdout, "")
    assert parsed["response"] == "LAST"
    assert parsed["tokens_generated"] == 9
    assert parsed["error"] is None

    failed = AGENT_HARNESS_SPECS["codex"]["parse"]('{"type":"turn.failed","error":{"message":"rate limited"}}', "")
    assert "rate limited" in failed["error"]


@pytest.mark.asyncio
async def test_agent_harness_claude_docker_command_and_parse(monkeypatch):
    monkeypatch.setenv("CLAUDE_DOCKER_IMAGE", "alpaca-claude:latest")
    monkeypatch.setenv("CLAUDE_DATA_DIR", "/tmp/claude-data")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("CLAUDE_BIN", raising=False)

    stdout = json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "HARNESS_OK",
            "usage": {"output_tokens": 7},
            "stop_reason": "end_turn",
        }
    ).encode()
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(stdout, b""))
    mock_proc.kill = MagicMock()
    mock_proc.wait = AsyncMock(return_value=0)

    with (
        patch("shutil.which", side_effect=lambda binary: None if binary == "claude" else f"/usr/local/bin/{binary}"),
        patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc) as exec_mock,
    ):
        res = await _query_harness_cli("claude", "claude-sonnet-4-5", "Say hi")

    assert res == {
        "success": True,
        "latency": res["latency"],
        "response": "HARNESS_OK",
        "thinking": None,
        "finish_reason": "end_turn",
        "tokens_generated": 7,
        "error": None,
    }
    argv = list(exec_mock.await_args[0])
    assert argv[:6] == ["docker", "run", "--rm", "--network", "host", "-i"]
    assert "ANTHROPIC_API_KEY=sk-ant-test" in argv
    assert "/tmp/claude-data:/data" in argv
    assert "HOME=/data" in argv
    image_idx = argv.index("alpaca-claude:latest")
    assert argv[image_idx + 1] == "claude"
    assert argv[image_idx + 2 :] == [
        "--bare",
        "-p",
        "Say hi",
        "--output-format",
        "json",
        "--permission-mode",
        "dontAsk",
        "--model",
        "claude-sonnet-4-5",
    ]


@pytest.mark.asyncio
async def test_agent_harness_codex_docker_command_and_parse(monkeypatch):
    monkeypatch.setenv("CODEX_DOCKER_IMAGE", "alpaca-codex:latest")
    monkeypatch.setenv("CODEX_API_KEY", "sk-codex")
    monkeypatch.delenv("CODEX_BIN", raising=False)
    monkeypatch.delenv("CODEX_DATA_DIR", raising=False)

    stdout = (
        json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "first"}})
        + "\n"
        + json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "CODEX_OK"}})
        + "\n"
        + json.dumps({"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 11}})
        + "\n"
    ).encode()
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(stdout, b""))
    mock_proc.kill = MagicMock()
    mock_proc.wait = AsyncMock(return_value=0)

    with (
        patch("shutil.which", side_effect=lambda binary: None if binary == "codex" else f"/usr/local/bin/{binary}"),
        patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc) as exec_mock,
    ):
        res = await _query_harness_cli("codex", "gpt-5-codex", "Do it")

    assert res["success"] is True
    assert res["response"] == "CODEX_OK"
    assert res["tokens_generated"] == 11
    argv = list(exec_mock.await_args[0])
    assert "CODEX_API_KEY=sk-codex" in argv
    image_idx = argv.index("alpaca-codex:latest")
    assert argv[image_idx + 1] == "codex"
    assert argv[image_idx + 2 :] == [
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--sandbox",
        "workspace-write",
        "--model",
        "gpt-5-codex",
        "--",
        "Do it",
    ]


@pytest.mark.asyncio
async def test_agent_harness_missing_binary_clear_error(monkeypatch):
    provider = OnlineModelProvider()
    monkeypatch.delenv("CLAUDE_DOCKER_IMAGE", raising=False)
    monkeypatch.delenv("CLAUDE_BIN", raising=False)

    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="CLAUDE_DOCKER_IMAGE"):
            await _query_harness_cli("claude", "default", "hi")
        res = await provider.query_online_model("claude:default", prompt="hi")

    assert res["success"] is False
    assert "CLAUDE_BIN" in res["error"]
    assert "CLAUDE_DOCKER_IMAGE" in res["error"]


@pytest.mark.asyncio
async def test_agent_harness_fetch_live_models_and_configured(monkeypatch):
    monkeypatch.setenv("PI_DOCKER_IMAGE", "alpaca-pi:latest")
    monkeypatch.setenv("PI_MODELS", "pi-mini, pi-pro")
    provider = OnlineModelProvider()

    assert provider.get_configured_providers()["pi"] is True
    assert provider.get_masked_credentials()["pi"]["configured"] is True

    results = await provider.fetch_live_models(provider="pi")
    assert sorted(r["id"] for r in results) == ["pi:default", "pi:pi-mini", "pi:pi-pro"]
    assert all(r["provider"] == "pi" for r in results)


@pytest.mark.asyncio
async def test_agent_harness_test_connection_docker(monkeypatch):
    provider = OnlineModelProvider()
    monkeypatch.delenv("DEEPSEEK_BIN", raising=False)
    monkeypatch.setenv("DEEPSEEK_DOCKER_IMAGE", "alpaca-dsh:latest")

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    with (
        patch("shutil.which", side_effect=lambda binary: None if binary == "dsh" else f"/usr/local/bin/{binary}"),
        patch("subprocess.run", return_value=mock_proc) as run_mock,
    ):
        res = await provider.test_connection("deepseek")

    assert res["success"] is True
    assert run_mock.call_args[0][0] == ["docker", "image", "inspect", "alpaca-dsh:latest"]
    assert "alpaca-dsh:latest" in res["message"]
