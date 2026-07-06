import os
import json
import pytest
from unittest.mock import patch
from llm_benchmark_suite import LLMModelBenchmark

@pytest.mark.asyncio
async def test_dynamic_tests_loading(tmp_path):
    tests_file = tmp_path / "custom_tests.json"
    custom_tests = {
        "coding": [{"id": "custom_coding", "label": "Custom Coding", "prompt": "Coding prompt", "num_predict": 100}],
        "reasoning": [{"id": "custom_reasoning", "label": "Custom Reasoning", "prompt": "Reasoning prompt", "num_predict": 100}],
        "instruction": [{"id": "custom_inst", "label": "Custom Inst", "prompt": "Inst prompt", "num_predict": 100}],
        "creative": [{"id": "custom_creative", "label": "Custom Creative", "prompt": "Creative prompt", "num_predict": 100}],
        "home_automation": [{"id": "custom_ha", "label": "Custom HA", "prompt": "HA prompt", "num_predict": 100}]
    }
    with open(tests_file, "w") as f:
        json.dump(custom_tests, f)
        
    with patch.dict(os.environ, {"BENCHMARK_TESTS_JSON": str(tests_file)}):
        benchmark = LLMModelBenchmark()
        assert benchmark.tests_config["coding"][0]["id"] == "custom_coding"
        assert benchmark._coding_tests("")[0]["id"] == "custom_coding"
        assert benchmark._reasoning_tests("")[0]["id"] == "custom_reasoning"
        
        assert benchmark.get_total_tests_per_model("functional") == 5
        assert benchmark.get_total_tests_per_model("all") == 7

@pytest.mark.asyncio
async def test_fallback_tests_loading():
    with patch.dict(os.environ, {"BENCHMARK_TESTS_JSON": "nonexistent_file_path_123.json"}):
        benchmark = LLMModelBenchmark()
        assert len(benchmark._coding_tests("")) > 0
        assert benchmark._coding_tests("")[0]["id"] == "debug_fix"

@pytest.mark.asyncio
async def test_test_ids_filtering(tmp_path):
    benchmark = LLMModelBenchmark()
    benchmark.RESULTS_DIR = tmp_path
    
    async def dummy_test(model, test, sampler=None):
        return {"success": True, "tokens_generated": 50, "latency": 1.0, "response": "Dummy response"}
    
    benchmark.test_model_proxy = dummy_test
    
    results = await benchmark.run_model_benchmarks(
        models=["qwen3:8b"],
        use_proxy=True,
        mode="functional",
        test_ids=["debug_fix"]
    )
    
    assert len(results["results"]) == 1
    model_res = results["results"][0]
    assert "category_coding" in model_res
    coding_tests = model_res["category_coding"]["tests"]
    assert len(coding_tests) == 1
    assert coding_tests[0]["test_id"] == "debug_fix"
    assert "last_run" in coding_tests[0]
    assert "category_reasoning" not in model_res

@pytest.mark.asyncio
async def test_incremental_merging(tmp_path):
    benchmark = LLMModelBenchmark()
    benchmark.RESULTS_DIR = tmp_path
    
    async def dummy_test(model, test, sampler=None):
        return {"success": True, "tokens_generated": 50, "latency": 1.0, "response": f"Response for {test['id']}"}
    
    benchmark.test_model_proxy = dummy_test
    
    await benchmark.run_model_benchmarks(
        models=["qwen3:8b"],
        use_proxy=True,
        mode="functional",
        test_ids=["debug_fix"]
    )
    
    await benchmark.run_model_benchmarks(
        models=["qwen3:8b"],
        use_proxy=True,
        mode="functional",
        test_ids=["logic_puzzle"]
    )
    
    latest_file = tmp_path / "functional_benchmarks_latest.json"
    assert latest_file.exists()
    with open(latest_file, "r") as f:
        latest_data = json.load(f)
        
    results = latest_data["results"]
    assert len(results) == 1
    model_res = results[0]
    
    assert "category_coding" in model_res
    assert "category_reasoning" in model_res
    
    coding_tests = model_res["category_coding"]["tests"]
    reasoning_tests = model_res["category_reasoning"]["tests"]
    assert len(coding_tests) == 1
    assert coding_tests[0]["test_id"] == "debug_fix"
    assert "last_run" in coding_tests[0]
    
    assert len(reasoning_tests) == 1
    assert reasoning_tests[0]["test_id"] == "logic_puzzle"
    assert "last_run" in reasoning_tests[0]


@pytest.mark.asyncio
async def test_two_phase_token_generation_and_nudge_injection():
    """Test that both test_model_proxy and test_model_direct trigger a two-phase request when tokens reach 4000."""
    benchmark = LLMModelBenchmark()
    
    # We mock Client.post to simulate hitting the 4000 token limit in phase 1,
    # followed by a successful phase 2.
    calls = []
    
    class MockResponse:
        def __init__(self, data, status_code=200):
            self.data = data
            self.status_code = status_code
        def json(self):
            return self.data

    async def mock_post(client_self, url, json, **kwargs):
        calls.append((url, json))
        # Determine if it's phase 1 or phase 2 based on json payload structure
        options = json.get("options", {})
        assert options.get("num_predict") == 4000
        
        if "messages" in json:
            # Chat endpoint (proxy)
            messages = json["messages"]
            if len(messages) == 1:
                # Phase 1
                return MockResponse({
                    "eval_count": 4000,
                    "eval_duration": 4000000000,
                    "prompt_eval_duration": 1000000000,
                    "message": {"content": "Phase 1 chat output"}
                })
            else:
                # Phase 2
                assert messages[-1]["role"] == "user"
                assert "halfway through your token budget" in messages[-1]["content"]
                return MockResponse({
                    "eval_count": 2500,
                    "eval_duration": 2000000000,
                    "prompt_eval_duration": 500000000,
                    "message": {"content": "Phase 2 chat output"}
                })
        else:
            # Generate endpoint (direct)
            prompt = json["prompt"]
            if "halfway through your token budget" not in prompt:
                # Phase 1
                return MockResponse({
                    "eval_count": 4000,
                    "eval_duration": 4000000000,
                    "prompt_eval_duration": 1000000000,
                    "response": "Phase 1 direct output"
                })
            else:
                # Phase 2
                assert "Phase 1 direct output" in prompt
                return MockResponse({
                    "eval_count": 1500,
                    "eval_duration": 1500000000,
                    "prompt_eval_duration": 300000000,
                    "response": "Phase 2 direct output"
                })

    with patch("httpx.AsyncClient.post", new=mock_post):
        # 1. Test Proxy Path
        res_proxy = await benchmark.test_model_proxy("test-model", {"id": "test", "prompt": "Hello"})
        assert res_proxy["success"] is True
        assert "Phase 1 chat output" in res_proxy["response"]
        assert "Phase 2 chat output" in res_proxy["response"]
        assert res_proxy["tokens_generated"] == 6500  # 4000 + 2500
        # 2 proxy calls made
        proxy_calls = [c for c in calls if "api/chat" in c[0]]
        assert len(proxy_calls) == 2

        calls.clear()

        # 2. Test Direct Path
        res_direct = await benchmark.test_model_direct("test-model", {"id": "test", "prompt": "Hello"})
        assert res_direct["success"] is True
        assert "Phase 1 direct output" in res_direct["response"]
        assert "Phase 2 direct output" in res_direct["response"]
        assert res_direct["tokens_generated"] == 5500  # 4000 + 1500
        # 2 direct calls made
        direct_calls = [c for c in calls if "api/generate" in c[0]]
        assert len(direct_calls) == 2


@pytest.mark.asyncio
async def test_shared_llm_two_phase_query_model():
    """Test that SharedLLMModelBenchmark.query_model triggers two-phase generation for both proxy and direct paths."""
    from web.shared_llm_benchmark import SharedLLMModelBenchmark
    shared_bench = SharedLLMModelBenchmark()

    calls = []

    class MockResponse:
        def __init__(self, data, status_code=200):
            self.data = data
            self.status_code = status_code
        def json(self):
            return self.data

    async def mock_post(client_self, url, json, **kwargs):
        calls.append((url, json))
        options = json.get("options", {})
        assert options.get("num_predict") == 4000

        if "messages" in json:
            # Proxy /api/chat path
            messages = json["messages"]
            if len(messages) == 1:
                return MockResponse({
                    "eval_count": 4000,
                    "message": {"content": "SharedLLM Phase 1 Chat"}
                })
            else:
                assert messages[-1]["role"] == "user"
                assert "halfway through your token budget" in messages[-1]["content"]
                return MockResponse({
                    "eval_count": 1800,
                    "message": {"content": "SharedLLM Phase 2 Chat"}
                })
        else:
            # Direct /api/generate path
            prompt = json["prompt"]
            if "halfway through your token budget" not in prompt:
                return MockResponse({
                    "eval_count": 4000,
                    "response": "SharedLLM Phase 1 Direct"
                })
            else:
                assert "SharedLLM Phase 1 Direct" in prompt
                return MockResponse({
                    "eval_count": 1200,
                    "response": "SharedLLM Phase 2 Direct"
                })

    with patch("httpx.AsyncClient.post", new=mock_post):
        # 1. Test Proxy Path
        res_proxy = await shared_bench.query_model("test-model", use_proxy=True, prompt="HA lights")
        assert res_proxy["success"] is True
        assert "SharedLLM Phase 1 Chat" in res_proxy["response"]
        assert "SharedLLM Phase 2 Chat" in res_proxy["response"]
        assert res_proxy["tokens_generated"] == 5800  # 4000 + 1800
        assert len([c for c in calls if "api/chat" in c[0]]) == 2

        calls.clear()

        # 2. Test Direct Path
        res_direct = await shared_bench.query_model("test-model", use_proxy=False, prompt="HA lights")
        assert res_direct["success"] is True
        assert "SharedLLM Phase 1 Direct" in res_direct["response"]
        assert "SharedLLM Phase 2 Direct" in res_direct["response"]
        assert res_direct["tokens_generated"] == 5200  # 4000 + 1200
        assert len([c for c in calls if "api/generate" in c[0]]) == 2
