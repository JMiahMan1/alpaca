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
