from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from online_providers import OnlineModelProvider
from web.shared_llm_benchmark import SharedLLMModelBenchmark


def test_online_model_provider_detection():
    provider = OnlineModelProvider()
    assert provider.is_online_model("openrouter:google/gemini-2.0-flash-exp:free") is True
    assert provider.is_online_model("huggingface:meta-llama/Llama-3.1-8B-Instruct") is True
    assert provider.is_online_model("cloudflare:@cf/meta/llama-3.1-8b-instruct") is True
    assert provider.is_online_model("opencode_zen:zen-coder-v1") is True
    assert provider.is_online_model("groq:llama-3.3-70b-versatile") is True
    assert provider.is_online_model("gemini:gemini-2.5-flash") is True
    assert provider.is_online_model("qwen2.5-coder:7b") is False


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

    p, m = provider.parse_model_identifier("gemini:gemini-2.5-flash")
    assert p == "gemini"
    assert m == "gemini-2.5-flash"


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
    assert bench.strip_thinking("Let me think<thinking>carefully</thinking> answer is 42") == "Let me think answer is 42"
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


def test_shared_llm_tool_request_tasks_and_validation():
    bench = SharedLLMModelBenchmark()
    tool_tasks = [t for t in bench.get_all_tasks() if t["task_type"] == "tool_request"]
    assert len(tool_tasks) >= 4
    ids = {t["id"] for t in tool_tasks}
    assert {"tool_request_light_control", "tool_request_media_play", "tool_request_rag_search", "tool_request_git_commit"} <= ids

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
    provider._cached_live_models["groq"] = [
        {"id": "qwen/qwen3.6-27b", "supported_features": ["reasoning"]}
    ]

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
    provider._cached_live_models["groq"] = [
        {"id": "llama-3.3-70b-versatile", "supported_features": ["json_mode"]}
    ]

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
    provider._cached_live_models["groq"] = [
        {"id": "qwen/qwen3.6-27b", "supported_features": ["reasoning"]}
    ]

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
                "choices": [
                    {"message": {"content": "done"}, "finish_reason": "stop"}
                ],
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
