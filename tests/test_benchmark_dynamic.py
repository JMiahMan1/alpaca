import json
import os
from unittest.mock import patch

import pytest

from llm_benchmark_suite import LLMModelBenchmark
from online_providers import OnlineModelProvider


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
    assert (
        benchmark.get_total_tests_per_model(
            "functional", test_ids=["agentic_long_running"], tiers=["standard", "advanced"]
        )
        == 1
    )


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
async def test_two_phase_token_generation_and_nudge_injection(tmp_path, monkeypatch):
    """Both test_model_proxy and test_model_direct must run two streamed phases when tokens reach the per-test cap."""
    import httpx

    benchmark = LLMModelBenchmark()

    # Temperature must resolve from models.ini (suite raises ValueError otherwise).
    ini = tmp_path / "models.ini"
    ini.write_text("[*]\ntemperature = 0.5\n[test-model]\nreasoning-budget = 2048\n")
    monkeypatch.setenv("MODELS_INI_PATH", str(ini))

    # Model profile budget: think ON for this model, num_predict = base + 2 x headroom.
    bench_test = {"id": "test", "prompt": "Hello"}
    from llm_benchmark_suite import _test_num_predict

    expected_cap = _test_num_predict(bench_test, "test-model")

    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/admin/runtime":
            # Big-context host: the context guard must be a transparent no-op here.
            return httpx.Response(
                200,
                json={"loaded_models": [
                    {"name": "test-model--latest", "backend_model": "test-model--latest",
                     "running_settings": {"ctx-size": "131072"}}
                ]},
            )
        body = json.loads(request.content.decode())
        seen.append({"url": str(request.url), "json": body})
        assert body["stream"] is True
        assert body["options"]["num_predict"] == expected_cap
        if request.url.path.endswith("/api/chat"):
            messages = body["messages"]
            if len(messages) == 1:
                # Phase 1 - exhaust the cap to trigger the nudge retry.
                raw = _chat_ndjson(
                    [{"content": "Phase 1 chat output"}],
                    {"eval_count": expected_cap, "eval_duration": 400, "prompt_eval_duration": 100},
                )
                return httpx.Response(200, content=raw)
            assert messages[-1]["role"] == "user"
            assert "halfway through your token budget" in messages[-1]["content"]
            raw = _chat_ndjson(
                [{"content": "Phase 2 chat output"}],
                {"eval_count": 2500, "eval_duration": 200, "prompt_eval_duration": 50},
            )
            return httpx.Response(200, content=raw)
        prompt = body["prompt"]
        if "halfway through your token budget" not in prompt:
            raw = _generate_ndjson(
                ["Phase 1 direct output"],
                {"eval_count": expected_cap, "eval_duration": 400, "prompt_eval_duration": 100},
            )
            return httpx.Response(200, content=raw)
        assert "Phase 1 direct output" in prompt
        raw = _generate_ndjson(
            ["Phase 2 direct output"],
            {"eval_count": 1500, "eval_duration": 150, "prompt_eval_duration": 30},
        )
        return httpx.Response(200, content=raw)

    real_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        kwargs.pop("timeout", None)
        return real_client(transport=httpx.MockTransport(handler))

    with patch("httpx.AsyncClient", side_effect=client_factory):
        # 1. Proxy path (/api/chat)
        res_proxy = await benchmark.test_model_proxy("test-model", bench_test)
        assert res_proxy["success"] is True
        assert res_proxy["think"] is True
        assert "Phase 1 chat output" in res_proxy["response"]
        assert "Phase 2 chat output" in res_proxy["response"]
        assert res_proxy["tokens_generated"] == expected_cap + 2500
        assert len([c for c in seen if "/api/chat" in c["url"]]) == 2

        seen.clear()

        # 2. Direct path (/api/generate)
        res_direct = await benchmark.test_model_direct("test-model", bench_test)
        assert res_direct["success"] is True
        assert res_direct["think"] is True
        assert "Phase 1 direct output" in res_direct["response"]
        assert "Phase 2 direct output" in res_direct["response"]
        assert res_direct["tokens_generated"] == expected_cap + 1500
        assert len([c for c in seen if "/api/generate" in c["url"]]) == 2


@pytest.mark.asyncio
async def test_shared_llm_two_phase_query_model(tmp_path, monkeypatch):
    """Test that SharedLLMModelBenchmark.query_model triggers two-phase generation for both proxy and direct paths."""
    import httpx

    from web.shared_llm_benchmark import SharedLLMModelBenchmark

    ini = tmp_path / "models.ini"
    ini.write_text("[*]\ntemperature = 0.5\n")
    monkeypatch.setenv("MODELS_INI_PATH", str(ini))

    shared_bench = SharedLLMModelBenchmark()

    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/admin/runtime":
            return httpx.Response(
                200,
                json={"loaded_models": [
                    {"name": "test-model--latest", "backend_model": "test-model--latest",
                     "running_settings": {"ctx-size": "8192"}}
                ]},
            )
        body = json.loads(request.content.decode())
        seen.append({"url": str(request.url), "json": body})
        assert body["stream"] is True
        assert body["options"]["num_predict"] == 4000
        if request.url.path.endswith("/api/chat"):
            messages = body["messages"]
            if len(messages) == 1:
                raw = _chat_ndjson(
                    [{"content": "SharedLLM Phase 1 Chat"}],
                    {"eval_count": 4000, "eval_duration": 300, "prompt_eval_duration": 100},
                )
                return httpx.Response(200, content=raw)
            assert messages[-1]["role"] == "user"
            assert "halfway through your token budget" in messages[-1]["content"]
            raw = _chat_ndjson(
                [{"content": "SharedLLM Phase 2 Chat"}],
                {"eval_count": 1800, "eval_duration": 90, "prompt_eval_duration": 20},
            )
            return httpx.Response(200, content=raw)
        prompt = body["prompt"]
        if "halfway through your token budget" not in prompt:
            raw = _generate_ndjson(
                ["SharedLLM Phase 1 Direct"],
                {"eval_count": 4000, "eval_duration": 300, "prompt_eval_duration": 100},
            )
            return httpx.Response(200, content=raw)
        assert "SharedLLM Phase 1 Direct" in prompt
        raw = _generate_ndjson(
            ["SharedLLM Phase 2 Direct"],
            {"eval_count": 1200, "eval_duration": 70, "prompt_eval_duration": 10},
        )
        return httpx.Response(200, content=raw)

    real_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        kwargs.pop("timeout", None)
        return real_client(transport=httpx.MockTransport(handler))

    with patch("web.shared_llm_benchmark.httpx.AsyncClient", side_effect=client_factory):
        # 1. Test Proxy Path
        res_proxy = await shared_bench.query_model(model="test-model", use_proxy=True, prompt="HA lights")
        assert res_proxy["success"] is True
        assert "SharedLLM Phase 1 Chat" in res_proxy["response"]
        assert "SharedLLM Phase 2 Chat" in res_proxy["response"]
        assert res_proxy["tokens_generated"] == 5800  # 4000 + 1800
        assert len([c for c in seen if "/api/chat" in c["url"]]) == 2

        seen.clear()

        # 2. Test Direct Path
        res_direct = await shared_bench.query_model(model="test-model", use_proxy=False, prompt="HA lights")
        assert res_direct["success"] is True
        assert "SharedLLM Phase 1 Direct" in res_direct["response"]
        assert "SharedLLM Phase 2 Direct" in res_direct["response"]
        assert res_direct["tokens_generated"] == 5200  # 4000 + 1200
        assert len([c for c in seen if "/api/generate" in c["url"]]) == 2


@pytest.mark.asyncio
async def test_rate_limited_results_discarded_and_pre_429_kept(tmp_path):
    """A 429 rate-limit mid-run must NOT abort the model's run: the rate-limited
    test is recorded as failed, the run continues, and the model's aggregated
    results (including pre-429 successes) are still saved."""
    benchmark = LLMModelBenchmark()
    benchmark.RESULTS_DIR = tmp_path
    benchmark.MODELS_DIR = tmp_path / "models"
    benchmark.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    benchmark._rate_limit_retry_floor = 0.0  # don't sleep during the test

    call_count = 0

    async def proxy_call(model, test, sampler=None):
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            return {
                "success": False,
                "tokens_generated": 0,
                "latency": 1.0,
                "response": None,
                "error": "Groq rate limit exceeded (HTTP 429). Check your service usage and rate limits.",
                "retry_after": 0.0,
            }
        # First test completes fine (pre-429).
        return {"success": True, "tokens_generated": 50, "latency": 1.0, "response": f"Response for {test['id']}"}

    benchmark.test_model_proxy = proxy_call

    results = await benchmark.run_model_benchmarks(
        models=["qwen3:8b"], use_proxy=True, mode="functional", test_ids=["debug_fix", "logic_puzzle"]
    )

    # The run completed (did not abort): both tests were attempted and the
    # model's aggregated results are saved (pre-429 success + rate-limited failure).
    assert len(results["results"]) == 1
    model_result = results["results"][0]
    assert model_result["model"] == "qwen3:8b"
    assert "rate_limited" not in model_result

    # Pre-429 result persisted incrementally and survives.
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
    assert "debug_fix" in ids  # pre-429 result kept
    assert "logic_puzzle" in ids  # rate-limited test still recorded (as a failure)

    # Aggregated latest file was written for this run.
    assert (tmp_path / "functional_benchmarks_latest.json").exists()


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
        "import pygame\nclass PacMan:\n    def eat(self):\n        self.score += 10\nmaze = []\npellets = []\n"
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
    bad = 'package main\nfunc main() {\n    fmt.Println("hello")\n}\n'
    assert benchmark._verify_functional_response({"id": "game_snake_canvas"}, good) is True
    assert benchmark._verify_functional_response({"id": "game_snake_canvas"}, bad) is False
    assert benchmark._verify_functional_response({"id": "game_3d_asteroid_go"}, bad) is False


def test_gamedev_alt_web_ui_verify():
    benchmark = LLMModelBenchmark()
    good_three = (
        '<script src="three.min.js"></script>\n'
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


def test_instruction_adherence_checker():
    benchmark = LLMModelBenchmark()
    good = (
        "- **Failures:** Google's AI Overview misspells words with repeated letters in answers.\n"
        "- **Pattern:** Earlier incidents included The Onion pranks about rocks and glue on pizza.\n"
        "- **Cause:** Tokenization splits words so transformers see subwords, not letters.\n"
    )
    assert benchmark._verify_functional_response({"id": "instruction_adherence"}, good) is True

    two_bullets = (
        "- **Failures:** Google's AI Overview misspells words with repeated letters in answers.\n"
        "- **Cause:** Tokenization splits words so transformers see subwords, not letters.\n"
    )
    assert benchmark._verify_functional_response({"id": "instruction_adherence"}, two_bullets) is False

    wrong_order = (
        "- **Failures:** Google's AI Overview misspells words with repeated letters in answers.\n"
        "- **Cause:** Tokenization splits words so transformers see subwords, not letters.\n"
        "- **Pattern:** Earlier incidents included The Onion pranks about rocks and glue on pizza.\n"
    )
    assert benchmark._verify_functional_response({"id": "instruction_adherence"}, wrong_order) is False

    long_bullet = (
        "- **Failures:** Google's AI Overview has repeatedly misspelled very common English words by confidently miscounting repeated letters in short answers to everyday user spelling questions asked all across the web.\n"
        "- **Pattern:** Earlier incidents included The Onion pranks about rocks and glue on pizza.\n"
        "- **Cause:** Tokenization splits words so transformers see subwords, not letters.\n"
    )
    assert benchmark._verify_functional_response({"id": "instruction_adherence"}, long_bullet) is False

    forbidden = (
        "- **Failures:** Google's AI Overview misspells words like strawberry with repeated letters in answers.\n"
        "- **Pattern:** Earlier incidents included The Onion pranks about rocks and glue on pizza.\n"
        "- **Cause:** Tokenization splits words so transformers see subwords, not letters.\n"
    )
    assert benchmark._verify_functional_response({"id": "instruction_adherence"}, forbidden) is False

    extra_prose = (
        "Here is my analysis of the article:\n"
        "- **Failures:** Google's AI Overview misspells words with repeated letters in answers.\n"
        "- **Pattern:** Earlier incidents included The Onion pranks about rocks and glue on pizza.\n"
        "- **Cause:** Tokenization splits words so transformers see subwords, not letters.\n"
    )
    assert benchmark._verify_functional_response({"id": "instruction_adherence"}, extra_prose) is False

    missing_terms = (
        "- **Failures:** Google's overview feature misspells words with repeated letters in answers.\n"
        "- **Pattern:** Earlier incidents included The Onion pranks about rocks and glue on pizza.\n"
        "- **Cause:** Encoding splits words so models see subwords, not letters.\n"
    )
    assert benchmark._verify_functional_response({"id": "instruction_adherence"}, missing_terms) is False


def test_new_lukes_tests_present_in_suite():
    tests = LLMModelBenchmark()._load_tests_config()
    by_id = {t.get("id"): t for cat in tests.values() for t in (cat if isinstance(cat, list) else [cat])}
    for tid, typ, np_, est in [
        ("game_falling_sand", "ui", 9000, 4096),
        ("game_slime_maze", "ui", 9000, 4096),
        ("game_dungeon_fog", "ui", 9000, 4096),
        ("game_memory_match", "ui", 9000, 4096),
        ("game_driving_2d", "ui", 9000, 4096),
        ("app_fake_desktop", "ui", 9000, 4096),
        ("app_kanban_board", "ui", 9000, 4096),
        ("app_expense_tracker", "ui", 9000, 4096),
        ("instruction_adherence", None, 800, 1024),
        ("logic_bridge_torch", None, 400, 2048),
        ("logic_twins_birthday", None, 400, 2048),
    ]:
        t = by_id.get(tid)
        assert t, f"missing test {tid}"
        assert t.get("type") == typ, f"{tid} type {t.get('type')} != {typ}"
        assert t.get("num_predict") == np_, f"{tid} num_predict {t.get('num_predict')} != {np_}"
        assert t.get("reasoning_estimate") == est, f"{tid} estimate mismatch"

    logic = {t["id"]: t for t in tests["logic"]}
    assert logic["logic_bridge_torch"].get("expected") == "17"
    assert logic["logic_twins_birthday"].get("expected") == "1"


def test_strip_thinking_removes_prose_thinking_preamble_before_code():
    benchmark = LLMModelBenchmark()
    raw = (
        "Here's a thinking process:\n"
        "1. **Analyze the Request:**\n"
        "   ```python\n"
        "   import pygame  # draft\n"
        "   ```\n"
        "2. Plan the layout.\n"
        "\n"
        "```python\n"
        "import pygame\n"
        "import sys\n"
        "def main():\n"
        "    pygame.init()\n"
        "main()\n"
        "```"
    )
    clean, thinking = benchmark.strip_thinking(raw)
    assert "thinking process" not in clean
    assert "import pygame" in clean
    assert "draft" not in clean
    assert thinking and "thinking process" in thinking


def test_strip_thinking_keeps_short_intros_and_plain_text():
    benchmark = LLMModelBenchmark()
    intro = "Here's the game:\n```python\nx = 1\n```"
    clean, thinking = benchmark.strip_thinking(intro)
    assert "Here's the game" in clean
    assert thinking is None
    plain = "The sky is blue and the ocean is deep."
    assert benchmark.strip_thinking(plain)[0] == plain


def test_score_code_quality_uses_extracted_code_for_syntax():
    benchmark = LLMModelBenchmark()
    # A leaked thinking preamble with indented draft snippets must not cause a
    # false "python syntax invalid" alarm: only the real final block is graded.
    resp = (
        "Here's a thinking process:\n"
        "   ```python\n"
        "   import pygame  # draft\n"
        "   ```\n"
        "```python\n"
        "import pygame\n"
        "def main():\n"
        "    print('ok')\n"
        "main()\n"
        "```"
    )
    qual = benchmark._score_code_quality(resp, {"category": "retrogames"})
    assert qual["syntax_valid"] is True


def test_score_test_graded_not_binary():
    benchmark = LLMModelBenchmark()
    test = {"id": "retro_space_invaders", "category": "retrogames", "type": "ui"}
    # Ran, screenshot, but failed prompt-expectation verification -> below passing.
    failed_fp = {
        "response": "code...",
        "code_ran": True,
        "code_score": 100,
        "success": False,
        "functional_pass": False,
        "code_quality": {"score": 80},
    }
    assert benchmark._score_test(test, failed_fp) <= 45
    # Ran + passed verification + clean quality -> high graded score, not always 100.
    good = {
        "response": "code...",
        "code_ran": True,
        "code_score": 100,
        "success": True,
        "functional_pass": True,
        "code_quality": {"score": 80},
    }
    assert 70 <= benchmark._score_test(test, good) <= 100
    # Clean run (base 60) with good quality -> meaningful intermediate score.
    clean = {
        "response": "code...",
        "code_ran": True,
        "code_score": 60,
        "success": True,
        "functional_pass": True,
        "code_quality": {"score": 70},
    }
    assert 55 <= benchmark._score_test(test, clean) <= 75
    # Did not run -> 0.
    not_ran = {"response": "code...", "code_ran": False, "code_score": 0, "success": False}
    assert benchmark._score_test(test, not_ran) == 0


def test_score_test_knowledge_partial_credit():
    benchmark = LLMModelBenchmark()
    test = {"id": "k1", "expected": "the sky is blue and the grass is green"}
    result = {"response": "The sky is blue today", "success": True}
    score = benchmark._score_test(test, result)
    assert 0 < score < 100


def test_merge_run_history_tracks_run_and_fail_counts():
    benchmark = LLMModelBenchmark()
    # First attempt fails -> run_count 1, fail_count 1
    first = benchmark._merge_run_history(None, {"test_id": "t1", "success": False, "score": 0})
    assert first["run_count"] == 1
    assert first["fail_count"] == 1
    # Second attempt passes -> run_count 2, fail_count stays 1 (latest wins)
    second = benchmark._merge_run_history(first, {"test_id": "t1", "success": True, "score": 100})
    assert second["run_count"] == 2
    assert second["fail_count"] == 1
    assert second["score"] == 100
    # Third attempt fails -> run_count 3, fail_count 2
    third = benchmark._merge_run_history(second, {"test_id": "t1", "success": False, "score": 0})
    assert third["run_count"] == 3
    assert third["fail_count"] == 2


@pytest.mark.asyncio
async def test_incremental_save_preserves_run_history(tmp_path):
    benchmark = LLMModelBenchmark()
    benchmark.MODELS_DIR = tmp_path
    benchmark.RESULTS_DIR = tmp_path
    now = "2026-08-20T12:00:00"
    base = {
        "test_id": "debug_fix",
        "test_category": "coding",
        "test_label": "Debug Fix",
        "success": False,
        "score": 0,
        "last_run": now,
    }
    benchmark.save_test_result_incremental("model_x", "coding", base, "functional", True, now)
    f = tmp_path / "general_model_x.json"
    assert f.exists()
    data = json.loads(f.read_text())
    tests = data["results"][0]["category_coding"]["tests"]
    assert len(tests) == 1
    assert tests[0]["run_count"] == 1
    assert tests[0]["fail_count"] == 1

    passing = dict(base)
    passing["success"] = True
    passing["score"] = 100
    benchmark.save_test_result_incremental("model_x", "coding", passing, "functional", True, now)
    data = json.loads(f.read_text())
    tests = data["results"][0]["category_coding"]["tests"]
    assert len(tests) == 1
    assert tests[0]["run_count"] == 2
    assert tests[0]["fail_count"] == 1
    assert tests[0]["success"] is True


def test_fence_lang_honors_explicit_tag_over_sql_sniffing():
    """Models answering SQL prompts with a tagged ```python sqlite3 program must
    be executed as Python, not piped into sqlite3 (regression: db_* tests all
    scored 0 with 'Parse error near line 1: near "import"')."""
    benchmark = LLMModelBenchmark()
    py_resp = (
        "```python\nimport sqlite3\nconn = sqlite3.connect(':memory:')\n"
        "cursor = conn.cursor()\ncursor.execute('SELECT c.name FROM customers c "
        "JOIN orders o ON c.customer_id = o.customer_id')\n```"
    )
    assert benchmark._fence_lang(py_resp) == "python"
    # lang selection chain: fence tag beats content sniffing
    lang = benchmark._fence_lang(py_resp) or benchmark._infer_lang(py_resp)
    assert lang == "python"


def test_infer_lang_python_with_embedded_sql_is_not_sql():
    """Content sniffing: raw SQL is 'sql', but a Python program embedding SQL
    string literals must stay 'python'."""
    benchmark = LLMModelBenchmark()
    raw_sql = "SELECT c.name FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.name;"
    assert benchmark._infer_lang(raw_sql) == "sql"
    py_prog = (
        "import sqlite3\nconn = sqlite3.connect(':memory:')\n"
        "cursor = conn.cursor()\n"
        "cursor.execute('SELECT name FROM customers JOIN orders ON id = customer_id')\n"
        "print(cursor.fetchall())"
    )
    assert benchmark._infer_lang(py_prog) == "python"


@pytest.mark.asyncio
async def test_db_join_orders_grades_python_sqlite_solution(tmp_path):
    """End-to-end regression for the database category: a correct Python
    sqlite3 answer runs in the sandbox and passes instead of failing with a
    sqlite parse error."""
    benchmark = LLMModelBenchmark()
    benchmark.RESULTS_DIR = tmp_path
    resp = (
        "Here's the query:\n```python\nimport sqlite3\n\n"
        "conn = sqlite3.connect(':memory:')\ncursor = conn.cursor()\n"
        "cursor.execute('''CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, "
        "customer_name TEXT NOT NULL)''')\n"
        "cursor.execute('''CREATE TABLE orders (order_id INTEGER PRIMARY KEY, "
        "customer_id INTEGER REFERENCES customers(customer_id))''')\n"
        'cursor.execute("SELECT c.customer_name, COUNT(o.order_id) AS order_count FROM customers c '
        "JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.customer_name "
        'HAVING COUNT(o.order_id) > 3")\n'
        "print(cursor.fetchall())\n```"
    )
    lang = benchmark._fence_lang(resp) or benchmark._infer_lang(resp)
    assert lang == "python"
    from sandbox_exec import extract_clean_code

    code = extract_clean_code(resp, lang)
    assert code.startswith("import sqlite3")


def test_reasoning_estimate_explicit_field_wins_and_tiers_fallback():
    """Per-test reasoning_estimate: explicit field wins; task-shape tiers fill in."""
    from llm_benchmark_suite import _test_reasoning_estimate

    # Explicit field always wins
    assert _test_reasoning_estimate({"reasoning_estimate": 4096, "category": "coding"}) == 4096
    # Tier fallbacks by task shape
    assert _test_reasoning_estimate({"type": "ui"}) == 4096
    assert _test_reasoning_estimate({"category": "retrogames"}) == 4096
    assert _test_reasoning_estimate({"category": "gamedev_alt"}) == 4096
    assert _test_reasoning_estimate({"category": "coding"}) == 3072
    assert _test_reasoning_estimate({"category": "linux_driver"}) == 3072
    assert _test_reasoning_estimate({"category": "gpqa_diamond"}) == 2048
    assert _test_reasoning_estimate({"category": "metacog"}) == 2048
    assert _test_reasoning_estimate({"category": "office"}) == 1024
    # Heavy-reasoning flag never estimates below its threshold
    assert _test_reasoning_estimate({"reasoning_budget": 3072, "category": "office"}) == 3072


def test_num_predict_doubles_reasoning_headroom_when_thinking_on(tmp_path, monkeypatch):
    """A model profile budget enables thinking and doubles headroom
    (base + 2 x estimate) so think + full answer fit without truncation;
    no budget anywhere -> plain base cap, thinking off."""
    from llm_benchmark_suite import _effective_thinking, _test_num_predict

    ini = tmp_path / "models.ini"
    ini.write_text("[*]\ntemperature = 0.5\n[test-model]\nreasoning-budget = 2048\n")
    monkeypatch.setenv("MODELS_INI_PATH", str(ini))

    heavy = {"num_predict": 8000, "reasoning_estimate": 4096}
    assert _effective_thinking("test-model") is True
    assert _test_num_predict(heavy, "test-model") == 8000 + 2 * 4096

    # Legacy per-test reasoning_budget values no longer toggle thinking
    assert _effective_thinking("missing-model") is False
    legacy = {"num_predict": 8000, "reasoning_budget": 4096, "reasoning_estimate": 4096}
    assert _test_num_predict(legacy) == 8000

    # Any defined budget enables thinking; explicit thinking=off keeps the
    # plain base cap even when a budget is defined.
    ini.write_text("[*]\ntemperature = 0.5\n[test-model]\nreasoning-budget = 512\n")
    light = {"num_predict": 900}
    assert _effective_thinking("test-model") is True
    assert _test_num_predict(light, "test-model") == 900 + 2 * _default_estimate()
    ini.write_text("[*]\ntemperature = 0.5\n[test-model]\nreasoning-budget = 512\nthinking = off\n")
    assert _effective_thinking("test-model") is False
    assert _test_num_predict(light, "test-model") == 900


def _default_estimate():
    from llm_benchmark_suite import _test_reasoning_estimate

    return _test_reasoning_estimate({})


def test_model_profile_reasoning_budget_thinking_and_benchmark_default(tmp_path, monkeypatch):
    """Reasoning comes from the model profile: profile budget wins, then the
    settable [benchmark] default, and 'no budget anywhere' means thinking off.
    explicit thinking=off must beat a profile budget; benchmark default must
    apply to models without their own budget."""
    from llm_benchmark_suite import (
        _effective_reasoning_budget,
        _effective_thinking,
        _model_reasoning_budget,
        _test_reasoning_budget,
    )

    ini = tmp_path / "models.ini"
    ini.write_text(
        "[*]\ntemperature = 0.5\n"
        "[profiled]\nreasoning-budget = 4096\nthinking = off\n"
        "[benchmark]\nreasoning-budget = 1024\n"
    )
    monkeypatch.setenv("MODELS_INI_PATH", str(ini))

    # Model profile budget wins over benchmark default
    assert _model_reasoning_budget("profiled") == 4096
    assert _effective_reasoning_budget("profiled") == 4096
    assert _test_reasoning_budget({}, "profiled") == 4096

    # Explicit thinking=off beats the profile budget
    assert _effective_thinking("profiled") is False

    # Benchmark default applies to models without their own budget
    assert _model_reasoning_budget("unprofiled") is None
    assert _effective_reasoning_budget("unprofiled") == 1024
    assert _effective_thinking("unprofiled") is True

    # No budget anywhere -> no budget, no thinking
    ini.write_text("[*]\ntemperature = 0.5\n[probably-empty]\nseed = 42\n")
    assert _effective_reasoning_budget("probably-empty") is None
    assert _effective_thinking("probably-empty") is False
    assert _test_reasoning_budget({"reasoning_budget": 4096}, "probably-empty") is None


def test_model_profile_sampling_options_parse(tmp_path, monkeypatch):
    """Per-model sampling knobs surface as typed llama.cpp option names and
    only when actually defined in the profile."""
    from llm_benchmark_suite import _model_sampling_options

    ini = tmp_path / "models.ini"
    ini.write_text(
        "[*]\ntemperature = 0.5\n"
        "[sampling]\ntop-k = 40\ntop-p = 0.9\nmin-p = 0.05\nrepeat-last-n = 64\n"
        "repeat-penalty = 1.15\npresence-penalty = 0.4\nfrequency-penalty = 0.2\nseed = 7\n"
        "[plain]\ntemperature = 0.5\n"
    )
    monkeypatch.setenv("MODELS_INI_PATH", str(ini))

    opts = _model_sampling_options("sampling")
    assert opts == {
        "top_k": 40,
        "top_p": 0.9,
        "min_p": 0.05,
        "repeat_last_n": 64,
        "repeat_penalty": 1.15,
        "presence_penalty": 0.4,
        "frequency_penalty": 0.2,
        "seed": 7,
    }
    assert isinstance(opts["top_k"], int)
    assert isinstance(opts["top_p"], float)
    assert opts["top_p"] == 0.9

    # A profile without sampling keys returns {}; a malformed value is skipped
    # instead of raising.
    ini.write_text(
        "[*]\ntemperature = 0.5\n"
        "[broken]\ntop-k = not-a-number\n[plain]\ntemperature = 0.5\n"
    )
    assert _model_sampling_options("broken") == {}
    assert _model_sampling_options("plain") == {}


@pytest.mark.asyncio
async def test_online_empty_length_thinking_gets_direct_answer_recovery():
    """Deterministic exhaustion (empty content + finish_reason=length + captured
    thinking) must fire the phase-2 direct-answer continuation instead of
    retrying forever - retries reproduce it identically."""
    provider = OnlineModelProvider()

    call = {"n": 0}

    async def fake_impl(model_identifier, prompt, max_tokens, temperature, custom_keys, request_timeout=None):
        call["n"] += 1
        if call["n"] == 1:
            return {
                "success": False,
                "latency": 0.1,
                "response": None,
                "thinking": "step 1: plan the pacman maze layout...",
                "tokens_generated": 16192,
                "finish_reason": "length",
                "error": "Empty completion",
            }
        return {
            "success": True,
            "latency": 0.2,
            "response": "```python\nimport pygame\n# full game\n```",
            "tokens_generated": 500,
            "finish_reason": "stop",
        }

    with patch.object(provider, "_query_online_model_impl", side_effect=fake_impl):
        res = await provider.query_online_model("openrouter:stealth/ox-alpha", prompt="Build pacman", max_tokens=8000)

    assert res["success"] is True
    assert res["response"].startswith("```python")
    assert res["finish_reason"] == "stop"
    assert call["n"] == 2  # phase-2 fired exactly once; no retry loop


@pytest.mark.asyncio
async def test_online_deterministic_exhaustion_stops_retry_loop():
    """When the direct-answer retry also fails, the loop must break immediately
    instead of burning max_retries x backoff on a deterministic outcome."""
    provider = OnlineModelProvider()
    call = {"n": 0}

    async def fake_impl(model_identifier, prompt, max_tokens, temperature, custom_keys, request_timeout=None):
        call["n"] += 1
        if "previous attempt produced no final answer" in prompt:
            return {
                "success": False,
                "latency": 0.1,
                "response": None,
                "tokens_generated": 0,
                "finish_reason": "length",
                "error": "still empty",
            }
        return {
            "success": False,
            "latency": 0.1,
            "response": None,
            "thinking": "deep reasoning...",
            "tokens_generated": 8000,
            "finish_reason": "length",
            "error": "Empty completion",
        }

    with patch.object(provider, "_query_online_model_impl", side_effect=fake_impl):
        res = await provider.query_online_model("openrouter:stealth/ox-alpha", prompt="Build pacman", max_tokens=8000)

    assert res["success"] is False
    assert "reasoning exhausted" in res["error"]
    assert call["n"] == 2  # initial + one direct-answer retry, then stop


def test_get_fallback_models_requires_env_when_nothing_discovered():
    """No hardcoded model fallbacks: without BENCHMARK_MODELS the suite fails fast with a clear error."""
    suite = LLMModelBenchmark()
    with patch.dict(os.environ, {"BENCHMARK_MODELS": ""}), pytest.raises(RuntimeError, match="BENCHMARK_MODELS"):
        suite._get_fallback_models()
    with patch.dict(os.environ, {"BENCHMARK_MODELS": "m1:latest, m2:8b"}):
        assert suite._get_fallback_models() == ["m1:latest", "m2:8b"]
