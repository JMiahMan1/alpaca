import json
import os
from unittest.mock import patch

import pytest

from llm_benchmark_suite import LLMModelBenchmark


@pytest.mark.asyncio
async def test_dynamic_tests_loading(tmp_path):
    tests_file = tmp_path / "custom_tests.json"
    custom_tests = {
        "coding": [{"id": "custom_coding", "label": "Custom Coding", "prompt": "Coding prompt", "num_predict": 100}],
        "reasoning": [
            {"id": "custom_reasoning", "label": "Custom Reasoning", "prompt": "Reasoning prompt", "num_predict": 100}
        ],
        "instruction": [{"id": "custom_inst", "label": "Custom Inst", "prompt": "Inst prompt", "num_predict": 100}],
        "creative": [
            {"id": "custom_creative", "label": "Custom Creative", "prompt": "Creative prompt", "num_predict": 100}
        ],
        "home_automation": [{"id": "custom_ha", "label": "Custom HA", "prompt": "HA prompt", "num_predict": 100}],
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
        models=["qwen3:8b"], use_proxy=True, mode="functional", test_ids=["debug_fix"]
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
async def test_tier_filtering_excludes_advanced_tests(tmp_path):
    benchmark = LLMModelBenchmark()
    benchmark.RESULTS_DIR = tmp_path

    async def dummy_test(model, test, sampler=None):
        return {"success": True, "tokens_generated": 50, "latency": 1.0, "response": f"Response for {test['id']}"}

    benchmark.test_model_proxy = dummy_test

    # Inject one advanced-tier test into the coding category so we can prove
    # the tier filter removes it from the run without running the live suite.
    adv_test = {
        "id": "agentic_long_running",
        "label": "Agentic: Long Running",
        "prompt": "Long running agentic task",
        "num_predict": 16000,
        "tier": "advanced",
    }
    benchmark.tests_config["coding"] = [*benchmark._coding_tests(""), adv_test]

    results = await benchmark.run_model_benchmarks(
        models=["qwen3:8b"],
        use_proxy=True,
        mode="functional",
        test_ids=["debug_fix", "agentic_long_running"],
        tiers=["standard"],
    )

    assert len(results["results"]) == 1
    model_res = results["results"][0]
    coding_tests = model_res["category_coding"]["tests"]
    ids = [t["test_id"] for t in coding_tests]
    assert "debug_fix" in ids
    assert "agentic_long_running" not in ids

    # get_total_tests_per_model honors tiers the same way.
    assert benchmark.get_total_tests_per_model("functional", test_ids=["agentic_long_running"], tiers=["standard"]) == 0
    assert benchmark.get_total_tests_per_model("functional", test_ids=["agentic_long_running"], tiers=["standard", "advanced"]) == 1


@pytest.mark.asyncio
async def test_incremental_merging(tmp_path):
    benchmark = LLMModelBenchmark()
    benchmark.RESULTS_DIR = tmp_path

    async def dummy_test(model, test, sampler=None):
        return {"success": True, "tokens_generated": 50, "latency": 1.0, "response": f"Response for {test['id']}"}

    benchmark.test_model_proxy = dummy_test

    await benchmark.run_model_benchmarks(models=["qwen3:8b"], use_proxy=True, mode="functional", test_ids=["debug_fix"])

    await benchmark.run_model_benchmarks(
        models=["qwen3:8b"], use_proxy=True, mode="functional", test_ids=["logic_puzzle"]
    )

    latest_file = tmp_path / "functional_benchmarks_latest.json"
    assert latest_file.exists()
    with open(latest_file) as f:
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
                return MockResponse(
                    {
                        "eval_count": 4000,
                        "eval_duration": 4000000000,
                        "prompt_eval_duration": 1000000000,
                        "message": {"content": "Phase 1 chat output"},
                    }
                )
            else:
                # Phase 2
                assert messages[-1]["role"] == "user"
                assert "halfway through your token budget" in messages[-1]["content"]
                return MockResponse(
                    {
                        "eval_count": 2500,
                        "eval_duration": 2000000000,
                        "prompt_eval_duration": 500000000,
                        "message": {"content": "Phase 2 chat output"},
                    }
                )
        else:
            # Generate endpoint (direct)
            prompt = json["prompt"]
            if "halfway through your token budget" not in prompt:
                # Phase 1
                return MockResponse(
                    {
                        "eval_count": 4000,
                        "eval_duration": 4000000000,
                        "prompt_eval_duration": 1000000000,
                        "response": "Phase 1 direct output",
                    }
                )
            else:
                # Phase 2
                assert "Phase 1 direct output" in prompt
                return MockResponse(
                    {
                        "eval_count": 1500,
                        "eval_duration": 1500000000,
                        "prompt_eval_duration": 300000000,
                        "response": "Phase 2 direct output",
                    }
                )

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
                return MockResponse({"eval_count": 4000, "message": {"content": "SharedLLM Phase 1 Chat"}})
            else:
                assert messages[-1]["role"] == "user"
                assert "halfway through your token budget" in messages[-1]["content"]
                return MockResponse({"eval_count": 1800, "message": {"content": "SharedLLM Phase 2 Chat"}})
        else:
            # Direct /api/generate path
            prompt = json["prompt"]
            if "halfway through your token budget" not in prompt:
                return MockResponse({"eval_count": 4000, "response": "SharedLLM Phase 1 Direct"})
            else:
                assert "SharedLLM Phase 1 Direct" in prompt
                return MockResponse({"eval_count": 1200, "response": "SharedLLM Phase 2 Direct"})

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


@pytest.mark.asyncio
async def test_rate_limited_results_discarded_and_pre_429_kept(tmp_path):
    """A 429 rate-limit mid-run must discard rate-limited results and abort the model's
    run, while results completed before the 429 remain persisted incrementally."""
    benchmark = LLMModelBenchmark()
    benchmark.RESULTS_DIR = tmp_path
    benchmark.MODELS_DIR = tmp_path / "models"
    benchmark.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    rate_limited = False
    call_count = 0

    async def proxy_call(model, test, sampler=None):
        nonlocal rate_limited, call_count
        call_count += 1
        if call_count > 1:
            rate_limited = True
        if rate_limited:
            return {
                "success": False,
                "tokens_generated": 0,
                "latency": 1.0,
                "response": None,
                "error": "Groq rate limit exceeded (HTTP 429). Check your service usage and rate limits.",
            }
        # First test completes fine (pre-429).
        return {"success": True, "tokens_generated": 50, "latency": 1.0, "response": f"Response for {test['id']}"}

    benchmark.test_model_proxy = proxy_call

    results = await benchmark.run_model_benchmarks(
        models=["qwen3:8b"], use_proxy=True, mode="functional", test_ids=["debug_fix", "logic_puzzle"]
    )

    # The rate-limited model's results are NOT merged into the aggregated results.
    assert results["results"] == []

    # The pre-429 result was persisted incrementally and must survive.
    pm_file = benchmark.MODELS_DIR / "general_qwen3_8b.json"
    assert pm_file.exists()
    with open(pm_file) as f:
        pm = json.load(f)
    saved_tests = []
    for mres in pm.get("results", []):
        for cat in mres:
            if cat.startswith("category_"):
                saved_tests.extend(mres[cat].get("tests", []))
    ids = {t["test_id"] for t in saved_tests}
    assert "logic_puzzle" in ids  # pre-429 result kept
    assert "debug_fix" not in ids  # rate-limited test discarded

    # No merged/aggregated latest file was written for this run.
    assert not (tmp_path / "functional_benchmarks_latest.json").exists()


@pytest.mark.asyncio
async def test_rate_limit_helper_detection():
    benchmark = LLMModelBenchmark()
    assert benchmark._is_rate_limited_result({"error": "Groq rate limit exceeded (HTTP 429). Check usage"}) is True
    assert benchmark._is_rate_limited_result({"error": "HTTP 429: Too Many Requests"}) is True
    assert benchmark._is_rate_limited_result({"error": "Gemini HTTP 500: Internal server error"}) is False
    assert benchmark._is_rate_limited_result({"error": None}) is False


def test_persistent_scoreboard_detection():
    benchmark = LLMModelBenchmark()
    good = (
        "class Game:\n"
        "    def __init__(self):\n"
        "        self.score = 0\n"
        "        self.high_scores = self.load_scores()\n"
        "    def load_scores(self):\n"
        "        with open('high_scores.json') as f:\n"
        "            return json.load(f)\n"
        "    def save_score(self):\n"
        "        self.high_scores.append(self.score)\n"
        "        with open('high_scores.json', 'w') as f:\n"
        "            json.dump(self.high_scores, f)\n"
        "    def game_over(self):\n"
        "        name = input('Enter your name: ')\n"
        "        self.save_score()\n"
        "    def new_game(self):\n"
        "        self.score = 0\n"
    )
    bad = "draw paddle, draw ball, keep the score in a variable"
    assert benchmark._has_persistent_scoreboard(good) is True
    assert benchmark._has_persistent_scoreboard(bad) is False
    assert benchmark._has_persistent_scoreboard("score and json.dump") is False
    assert benchmark._has_persistent_scoreboard("record high score to file") is False


def test_retro_verify_requires_scoreboard():
    benchmark = LLMModelBenchmark()
    test = {"id": "retro_pacman"}
    with_scoreboard = (
        "import pygame\n"
        "class PacMan:\n"
        "    def eat(self):\n"
        "        self.score += 10\n"
        "    def game_over(self):\n"
        "        name = input('Enter your name: ')\n"
        "        high_scores.append((name, self.score))\n"
        "    def new_game(self):\n"
        "        self.score = 0\n"
        "maze = []\n"
        "pellets = []\n"
        "high_scores = []\n"
        "with open('high_scores.json') as f:\n"
        "    high_scores = json.load(f)\n"
    )
    no_scoreboard = (
        "import pygame\n"
        "class PacMan:\n"
        "    def eat(self):\n"
        "        self.score += 10\n"
        "maze = []\n"
        "pellets = []\n"
    )
    assert benchmark._verify_functional_response(test, with_scoreboard) is True
    assert benchmark._verify_functional_response(test, no_scoreboard) is False


def test_gamedev_alt_code_counterpart_verify():
    benchmark = LLMModelBenchmark()
    good = (
        "<canvas id='c'></canvas>\n"
        "<script>\n"
        "const ctx = document.getElementById('c').getContext('2d');\n"
        "let snake = [];\n"
        "let food = {x: 10, y: 10};\n"
        "let score = 0;\n"
        "function tick() {\n"
        "    if (score > 0) {\n"
        "        localStorage.setItem('score', score);\n"
        "    }\n"
        "}\n"
        "document.addEventListener('keydown', function(e) {\n"
        "    arrow = e.key;\n"
        "});\n"
        "function gameOver() {\n"
        "    let name = prompt('Enter your name:');\n"
        "    score = 0;\n"
        "}\n"
        "</script>\n"
    )
    bad = "package main\nfunc main() {\n    fmt.Println(\"hello\")\n}\n"
    assert benchmark._verify_functional_response({"id": "game_snake_canvas"}, good) is True
    assert benchmark._verify_functional_response({"id": "game_snake_canvas"}, bad) is False
    assert benchmark._verify_functional_response({"id": "game_3d_asteroid_go"}, bad) is False


def test_gamedev_alt_web_ui_verify():
    benchmark = LLMModelBenchmark()
    good_three = (
        "<script src=\"three.min.js\"></script>\n"
        "<script>\n"
        "const scene = new THREE.Scene();\n"
        "const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 100);\n"
        "const paddle = new THREE.Mesh();\n"
        "const ball = new THREE.Mesh();\n"
        "function animate() {\n"
        "    requestAnimationFrame(animate);\n"
        "    localStorage.setItem('score', 42);\n"
        "}\n"
        "function gameOver() {\n"
        "    let name = prompt('Enter your name:');\n"
        "    localStorage.setItem(name, score);\n"
        "    score = 0;\n"
        "}\n"
        "</script>\n"
    )
    good_webgl = (
        "<canvas id='c'></canvas>\n"
        "const gl = document.getElementById('c').getContext('webgl');\n"
        "const vs = gl.createShader(gl.VERTEX_SHADER);\n"
        "const buf = gl.createBuffer();\n"
        "// procedural terrain, height coloring, perspective matrix\n"
        "matrix = perspective(fov, aspect, near, far);\n"
    )
    assert benchmark._verify_functional_response({"id": "game_pong_threejs"}, good_three) is True
    assert benchmark._verify_functional_response({"id": "game_3d_terrain_webgl"}, good_webgl) is True
    assert benchmark._verify_functional_response({"id": "game_3d_terrain_webgl"}, "<canvas></canvas>") is False


def test_youtuber_verify():
    benchmark = LLMModelBenchmark()
    sand_good = (
        "import pygame\n"
        "sand = []\n"
        "grid = [[0]*100 for _ in range(100)]\n"
        "water = []\n"
        "fire = []\n"
        "wood = []\n"
        "lava = []\n"
        "smoke = []\n"
        "with open('sim_stats.json') as f:\n"
        "    stats = json.load(f)\n"
        "json.dump(stats, f)\n"
    )
    life_good = (
        "import pygame\n"
        "cells = [[0]*50 for _ in range(50)]\n"
        "neighbors = count_neighbors(cells)\n"
        "glider = [(1,2),(2,3),(3,1),(3,2),(3,3)]\n"
        "alive = rules(alive, neighbors)\n"
        "with open('life_stats.json') as f:\n"
        "    json.dump(gen, f)\n"
    )
    boids_good = (
        "import pygame\n"
        "class Boid:\n"
        "    def separation(self): pass\n"
        "    def alignment(self): pass\n"
        "    def cohesion(self): pass\n"
        "flock = [Boid() for _ in range(50)]\n"
        "radius = 40\n"
        "with open('boids_stats.json') as f:\n"
        "    json.dump(cohesion, f)\n"
    )
    assert benchmark._verify_functional_response({"id": "game_falling_sand"}, sand_good) is True
    assert benchmark._verify_functional_response({"id": "game_conway_life"}, life_good) is True
    assert benchmark._verify_functional_response({"id": "game_boids"}, boids_good) is True
    assert benchmark._verify_functional_response({"id": "game_falling_sand"}, "import pygame\nsand = []\n") is False


def test_agentic_verify_requires_needle():
    benchmark = LLMModelBenchmark()
    incident_good = (
        "## Root cause\n"
        "The origin-5 registration failure is caused by transcoder_pool=12.\n"
        "## Timeline\n"
        "seq=1021 propagated error.\n"
        "## Rollback plan\n"
        "Roll back the config change.\n"
        "The embedded secret is ALPHA-9842-OMEGA.\n"
    )
    incident_bad = incident_good.replace("ALPHA-9842-OMEGA", "alpha-beta-gamma")
    migration_good = (
        "## Inventory\n"
        "Legacy PHP modules found.\n"
        "## Architecture\n"
        "## Phases\n"
        "Phase 1: extract services.\n"
        "## Risks\n"
        "The embedded secret is KAPPA-7713-ZETA.\n"
    )
    assert benchmark._verify_functional_response({"id": "agentic_incident_forensics"}, incident_good) is True
    assert benchmark._verify_functional_response({"id": "agentic_incident_forensics"}, incident_bad) is False
    assert benchmark._verify_functional_response({"id": "agentic_codebase_migration"}, migration_good) is True
