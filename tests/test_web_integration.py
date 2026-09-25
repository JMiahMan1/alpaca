import json
from unittest.mock import MagicMock, mock_open, patch

import pytest

from web.app import active_run, active_run_lock, app
from web.model_tracker import ModelTracker


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        # Reset state before each test
        with active_run_lock:
            active_run["status"] = "idle"
            active_run["current_model"] = None
            active_run["current_test"] = None
            active_run["current_category"] = None
            active_run["tests_completed"] = 0
            active_run["total_tests"] = 0
            active_run["models"] = []
            active_run["use_proxy"] = True
            active_run["results"] = []
            active_run["test_results"] = []
            active_run["start_time"] = None
            active_run["saved_as"] = None
        yield client


@pytest.mark.parametrize("next_model,next_workflow", [("m", "next"), ("m2", "house")])
def test_multistep_assembly_progress_adopts_larger_total(client, next_model, next_workflow):
    from web.app import get_progress_callback

    callback = get_progress_callback("multistep")
    step = {"model": "m", "workflow": "house", "label": "Turn", "category": "c", "step": 1, "total": 4}
    completion = {"model": "m", "test_id": "house", "test_label": "House", "category": "c", "result": {}}
    with patch("web.app.socketio.emit") as emit:
        callback("benchmark_start", {"models": ["m"], "use_proxy": True, "total_tests": 1, "timestamp": "t"})
        callback("test_step", step)
        callback("test_step", {**step, "step": 5, "total": 5})
        assert (active_run["tests_completed"], active_run["total_tests"]) == (4, 5)
        callback("test_complete", completion)
        assert emit.call_args.args[1]["progress"] == {"completed": 5, "total": 5, "percentage": 100}
        callback("test_step", {**step, "model": next_model, "workflow": next_workflow})
        assert (active_run["tests_completed"], active_run["total_tests"]) == (5, 9)
        callback("test_step", {**step, "model": next_model, "workflow": next_workflow, "step": 4})
        callback("test_complete", {**completion, "model": next_model, "test_id": next_workflow})
        assert emit.call_args.args[1]["progress"] == {"completed": 9, "total": 9, "percentage": 100}


@pytest.mark.parametrize("run_type", ["general", "shared_llm"])
def test_non_multistep_progress_denominator_unchanged(client, run_type):
    from web.app import get_progress_callback

    callback = get_progress_callback(run_type)
    with patch("web.app.socketio.emit") as emit:
        callback("benchmark_start", {"models": ["m"], "use_proxy": True, "total_tests": 4, "timestamp": "t"})
        callback("test_complete", {"model": "m", "category": "c", "test_id": "t", "test_label": "T", "result": {}})
        assert emit.call_args.args[1]["progress"] == {"completed": 1, "total": 4, "percentage": 25}
        assert active_run["test_results"][0]["test_id"] == "t"
        assert "response" not in active_run["test_results"][0]


@pytest.mark.parametrize("custom_keys", [None, {"openrouter_api_key": "test-key"}])
def test_multistep_route_forwards_custom_keys_to_harness(client, custom_keys):
    from unittest.mock import AsyncMock

    import web.app as web_app

    body = {"models": ["openrouter:test-model"], "workflow_ids": ["house"], "use_proxy": False}
    if custom_keys is not None:
        body["custom_keys"] = custom_keys
    with (
        patch("web.app.threading.Thread") as thread,
        patch.object(web_app.multistep_benchmark, "run_multistep_benchmarks", new_callable=AsyncMock) as run,
        patch("web.app.socketio.emit"),
    ):
        response = client.post("/api/run/multistep", json=body)
        assert response.status_code == 200
        assert "custom_keys" not in response.get_json()["active_run"]
        thread.return_value.start.assert_called_once()
        call = thread.call_args.kwargs
        call["target"](*call["args"])
        run.assert_awaited_once_with(
            models=body["models"],
            use_proxy=False,
            progress_callback=call["args"][3],
            cancel_event=call["args"][2],
            workflow_ids=["house"],
            custom_keys=custom_keys,
        )


def test_requests_preserves_online_estimate_labels_and_null_ttft(client, monkeypatch):
    from collections import deque

    import online_providers as online

    monkeypatch.setattr(online, "_active_online_requests", {})
    monkeypatch.setattr(online, "_completed_online_requests", deque(maxlen=50))
    online.start_online_request("online-test", "openrouter:test", "openrouter", {"prompt": "hello"})
    online.complete_online_request("online-test", {"success": True, "response": "answer", "tokens_generated": 2})
    response = MagicMock(status_code=200)
    response.json.return_value = {"active_requests": [], "completed_requests": []}
    with (
        patch("web.app._find_proxy_url", return_value="http://proxy-test"),
        patch("httpx.Client.get", return_value=response),
    ):
        result = client.get("/api/requests")
    assert result.status_code == 200
    record = result.get_json()["completed_requests"][0]
    assert record["ttft_seconds"] is None
    assert record["prompt_tokens_estimated"] is True
    assert record["completion_tokens_estimated"] is False
    assert record["completion_tokens"] == 2


@pytest.fixture
def arcade_source(tmp_path, monkeypatch):
    import web.arcade_publish as arcade

    games_dir = tmp_path / "games"
    source = tmp_path / "game.html"
    source.write_text("<!doctype html><html><body>Original game</body></html>")
    monkeypatch.setattr(arcade, "GAMES_DIR", games_dir)
    monkeypatch.setattr(arcade, "find_artifact_file", lambda model, test_id: source)
    return games_dir, source


@pytest.mark.parametrize("run_type", ["general", "shared_llm", "multistep"])
@pytest.mark.parametrize("threshold", [None, "0", "80"])
@pytest.mark.parametrize("published", [False, True])
def test_benchmark_completion_never_publishes_arcade(client, arcade_source, monkeypatch, run_type, threshold, published):
    import web.arcade_publish as arcade
    from web.app import get_progress_callback

    if threshold is None:
        monkeypatch.delenv("ARCADE_AUTO_PUBLISH_SCORE", raising=False)
    else:
        monkeypatch.setenv("ARCADE_AUTO_PUBLISH_SCORE", threshold)
    games_dir, source = arcade_source
    body = {"model": "demo-model", "test_id": "demo_game", "benchmark_score": 90}
    if published:
        assert client.post("/api/arcade/publish", json=body).status_code == 200
    before = {p.relative_to(games_dir): p.read_bytes() for p in games_dir.rglob("*") if p.is_file()}
    source.write_text("<!doctype html><html><body>New benchmark game</body></html>")
    callback = get_progress_callback(run_type)
    event_data = {
        "model": body["model"],
        "test_id": body["test_id"],
        "test_label": "Demo game",
        "category": "gamedev",
        "result": {"score": 100, "max_score": 100, "response": source.read_text()},
    }
    with (
        patch("web.arcade_publish.publish_game", wraps=arcade.publish_game) as publish,
        patch("web.app.socketio.emit") as emit,
    ):
        callback(
            "benchmark_start",
            {"models": [body["model"]], "use_proxy": True, "total_tests": 1, "timestamp": "2026-09-16"},
        )
        callback("test_complete", event_data)
        publish.assert_not_called()
        emit.assert_called_with(
            "test_complete",
            {**event_data, "progress": {"completed": 1, "total": 1, "percentage": 100}},
        )
    assert active_run["tests_completed"] == 1
    assert {p.relative_to(games_dir): p.read_bytes() for p in games_dir.rglob("*") if p.is_file()} == before
    assert games_dir.exists() is published


@pytest.mark.parametrize("lang", ["html", "python"])
def test_arcade_manual_publish_republish_and_unpublish(client, arcade_source, monkeypatch, lang):
    import web.arcade_publish as arcade

    games_dir, source = arcade_source
    if lang == "python":
        monkeypatch.setattr(arcade, "find_artifact_file", lambda model, test_id: None)
        monkeypatch.setattr(
            arcade,
            "find_model_response",
            lambda model, test_id: {"response": "```python\nprint('game')\n```", "score": 10},
        )
    body = {"model": "demo-model", "test_id": "demo_game", "benchmark_score": 10}
    res = client.post("/api/arcade/publish", json=body)
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["republished"] is False
    slug = data["slug"]
    game_dir = games_dir / slug
    assert (game_dir / ("game.html" if lang == "html" else "game.py")).is_file()
    assert json.loads((game_dir / "meta.json").read_text())["auto_published"] is False
    assert client.get("/api/arcade/published").get_json()["slugs"] == [slug]
    scores = '[{"initials": "ABC", "score": 123}]'
    ratings = '{"ABC": 5}'
    (game_dir / "scores.json").write_text(scores)
    (game_dir / "ratings.json").write_text(ratings)
    source.write_text("<!doctype html><html><body>Manually updated game</body></html>")
    res = client.post("/api/arcade/publish", json={**body, "benchmark_score": 20})
    assert res.status_code == 200
    assert res.get_json()["republished"] is True
    assert (game_dir / "scores.json").read_text() == scores
    assert (game_dir / "ratings.json").read_text() == ratings
    assert json.loads((game_dir / "meta.json").read_text())["benchmark_score"] == 20
    if lang == "html":
        assert (game_dir / "game.html").read_text() == source.read_text()
    assert client.post("/api/arcade/unpublish", json={"slug": slug}).get_json()["removed"] is True
    assert client.get("/api/arcade/published").get_json()["slugs"] == []


def test_arcade_legacy_settings_cannot_enable_publishing(client, arcade_source, monkeypatch):
    import web.arcade_publish as arcade
    from web.app import get_progress_callback

    monkeypatch.setenv("ARCADE_AUTO_PUBLISH_SCORE", "80")
    monkeypatch.setattr(arcade, "AUTO_PUBLISH_SCORE", 80)
    with (
        patch("online_providers.online_model_provider.save_credentials", return_value={"success": True}),
        patch("web.app._read_dotenv_values", return_value={"ARCADE_AUTO_PUBLISH_SCORE": "0"}),
        patch("web.arcade_publish.publish_game") as publish,
        patch("web.app.socketio.emit"),
    ):
        assert client.get("/api/arcade/settings").get_json()["auto_publish_enabled"] is False
        assert client.post("/api/arcade/settings", json={"auto_publish_score": 0}).status_code == 200
        assert client.get("/api/arcade/settings").get_json()["auto_publish_enabled"] is False
        active_run["total_tests"] = 1
        get_progress_callback("general")(
            "test_complete",
            {
                "model": "demo-model",
                "test_id": "demo_game",
                "test_label": "Demo game",
                "category": "gamedev",
                "result": {"score": 100},
            },
        )
        publish.assert_not_called()
    assert not arcade_source[0].exists()


def test_arcade_dashboard_manual_only(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"Games are never published automatically" in res.data
    assert b"input-arcade-threshold" not in res.data
    assert b"btn-save-arcade" not in res.data


def test_index_route(client):
    """Test that the index route serves the template and returns 200 OK"""
    res = client.get("/")
    assert res.status_code == 200
    assert b"Alpaca Benchmarks v2" in res.data
    assert b"Pipeline Controls" in res.data


def test_benchmark_groups_include_frontier_diagnostics(client):
    response = client.get("/api/benchmark/groups")
    assert response.status_code == 200
    assert "frontier_diagnostics" in response.get_json()["groups"]


def test_frontier_all_button_and_static_handler(client):
    index = client.get("/")
    assert index.status_code == 200
    assert b'id="btn-run-frontier-all"' in index.data
    assert b"Run Frontier All" in index.data
    assert b'id="btn-run"' in index.data

    response = client.get("/static/js/dashboard.js")
    assert response.status_code == 200
    source = response.get_data(as_text=True)
    assert "frontier_diagnostics: 'Frontier Diagnostics'" in source
    handler_start = source.index("btnRunFrontierAll.addEventListener('click'")
    handler_end = source.index("async function triggerBenchmark", handler_start)
    handler = source[handler_start:handler_end]
    assert "Promise.all([loadModels(), loadTests(), loadBenchmarkGroups()])" in handler
    assert "selectAllBtn.click()" in handler
    assert "selectAllTestsBtn.click()" in handler
    assert "box.checked = box.value === 'frontier_diagnostics'" in handler
    assert "await triggerBenchmark('/api/run')" in handler


@pytest.mark.parametrize("download", [False, True])
def test_artifact_preview_serves_bundled_three_js(client, tmp_path, monkeypatch, download):
    from pathlib import Path

    from web.app import benchmark

    monkeypatch.setattr(benchmark, "ARTIFACTS_DIR", tmp_path)
    html = '<html><script src="three.min.js"></script></html>'
    (tmp_path / "game.html").write_text(html)
    (tmp_path / "three.min.js").write_text("artifact-local script")
    assert client.get("/api/artifacts/game.html").data == html.encode()
    url = "/api/artifacts/three.min.js" + ("?download=1" if download else "")
    res = client.get(url)
    assert res.status_code == 200
    assert res.data == (Path(__file__).resolve().parent.parent / "three.min.js").read_bytes()
    assert res.mimetype == "application/javascript"
    assert res.headers["Content-Disposition"].startswith("attachment" if download else "inline")
    assert "no-store" in res.headers["Cache-Control"]
    assert client.head(url).status_code == 200
    assert client.head(url).data == b""
    assert (tmp_path / "three.min.js").read_text() == "artifact-local script"
    assert client.get("/api/artifacts").get_json()["artifacts"][0]["filename"] == "game.html"


def test_artifact_three_js_delete_does_not_touch_bundle(client, tmp_path, monkeypatch):
    from web.app import BUNDLED_THREE_JS, benchmark

    monkeypatch.setattr(benchmark, "ARTIFACTS_DIR", tmp_path)
    bundled = BUNDLED_THREE_JS.read_bytes()
    assert client.delete("/api/artifacts/three.min.js").status_code == 404
    local = tmp_path / "three.min.js"
    local.write_text("artifact-local script")
    assert client.delete("/api/artifacts/three.min.js").status_code == 200
    assert not local.exists()
    assert BUNDLED_THREE_JS.read_bytes() == bundled
    assert client.get("/api/artifacts/three.min.js").data == bundled


@pytest.mark.parametrize("prefix", ["/api/artifacts/", "/api/multistep/artifact/"])
def test_artifact_three_js_missing_bundle(client, tmp_path, monkeypatch, prefix):
    import web.app as web_app

    monkeypatch.setattr(web_app, "BUNDLED_THREE_JS", tmp_path / "missing-three.min.js")
    monkeypatch.setattr(web_app.benchmark, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(web_app.multistep_benchmark, "ARTIFACTS_DIR", tmp_path)
    (tmp_path / "three.min.js").write_text("artifact-local script")
    assert client.get(prefix + "three.min.js").status_code == 404


@pytest.mark.parametrize("prefix", ["/api/artifacts/", "/api/multistep/artifact/"])
def test_artifact_three_js_requires_authentication(client, monkeypatch, prefix):
    monkeypatch.setenv("ALPACA_API_KEY", "test-artifact-key")
    with patch("web.app.is_flask_client_local", return_value=False):
        assert client.get(prefix + "three.min.js").status_code == 401
        assert client.get(prefix + "three.min.js", headers={"X-API-Key": "test-artifact-key"}).status_code == 200


@pytest.mark.parametrize("filename", ["three.js", "three.min.js.map", "Dockerfile.web", "web/app.py", "../three.min.js"])
def test_artifact_dependency_does_not_expose_other_project_files(client, tmp_path, monkeypatch, filename):
    from web.app import benchmark

    monkeypatch.setattr(benchmark, "ARTIFACTS_DIR", tmp_path)
    assert client.get("/api/artifacts/" + filename).status_code == 404


def test_artifact_download_and_delete_unchanged(client, tmp_path, monkeypatch):
    from web.app import benchmark

    monkeypatch.setattr(benchmark, "ARTIFACTS_DIR", tmp_path)
    artifact = tmp_path / "game.html"
    artifact.write_text("<html>game</html>")
    res = client.get("/api/artifacts/game.html?download=1")
    assert res.status_code == 200
    assert res.data == artifact.read_bytes()
    assert res.headers["Content-Disposition"].startswith("attachment")
    assert client.delete("/api/artifacts/game.html").status_code == 200
    assert not artifact.exists()
    assert client.get("/api/artifacts/game.html").status_code == 404


def test_api_status_route(client):
    """Test that the status route returns the active run structure"""
    res = client.get("/api/status")
    assert res.status_code == 200
    data = json.loads(res.data.decode("utf-8"))
    assert data["status"] == "idle"
    assert "current_model" in data
    assert "tests_completed" in data


@patch("llm_benchmark_suite.LLMModelBenchmark.discover_all_models")
@patch("llm_benchmark_suite.LLMModelBenchmark.discover_all_proxy_models")
def test_api_models_route(mock_discover_proxy, mock_discover_all, client):
    """Test that the models discovery endpoint returns combined lists of models"""
    # Setup mocks
    mock_discover_all.return_value = ["model1", "model2"]
    mock_discover_proxy.return_value = ["model2", "model3"]

    res = client.get("/api/models")
    assert res.status_code == 200
    data = json.loads(res.data.decode("utf-8"))

    assert "models" in data
    # Unique combined list: model1, model2, model3
    assert set(data["models"]) == {"model1", "model2", "model3"}
    assert data["direct_models"] == ["model1", "model2"]
    assert data["proxy_models"] == ["model2", "model3"]


def test_api_run_missing_models(client):
    """Test starting benchmark fails if models list is missing"""
    res = client.post("/api/run", json={})
    assert res.status_code == 400
    data = json.loads(res.data.decode("utf-8"))
    assert "error" in data


@patch("threading.Thread.start")
def test_api_run_success(mock_thread_start, client):
    """Test starting benchmark succeeds and spawns the background worker thread"""
    res = client.post("/api/run", json={"models": ["qwen3:8b"], "use_proxy": True})
    assert res.status_code == 200
    data = json.loads(res.data.decode("utf-8"))
    assert data["status"] == "Benchmark started"

    # Verify status is running
    status_res = client.get("/api/status")
    status_data = json.loads(status_res.data.decode("utf-8"))
    assert status_data["status"] == "running"
    assert status_data["models"] == ["qwen3:8b"]
    assert status_data["use_proxy"] is True

    # Assert thread was triggered
    mock_thread_start.assert_called_once()


def test_api_cancel_no_active_run(client):
    """Test cancel fails when no run is active"""
    res = client.post("/api/cancel")
    assert res.status_code == 400
    data = json.loads(res.data.decode("utf-8"))
    assert "error" in data


@patch("threading.Event.set")
def test_api_cancel_success(mock_event_set, client):
    """Test cancel sets event and transitions status back to cancelled"""
    # Artificially set status to running
    with active_run_lock:
        active_run["status"] = "running"
        # Create a mock cancel event
        import web.app

        web.app.cancel_event = MagicMock()

    res = client.post("/api/cancel")
    assert res.status_code == 200
    data = json.loads(res.data.decode("utf-8"))
    assert data["status"] == "Cancellation requested"

    # Check updated status
    status_res = client.get("/api/status")
    status_data = json.loads(status_res.data.decode("utf-8"))
    assert status_data["status"] == "cancelled"


def test_api_results_list(client):
    """Test retrieving results lists empty list when directory is empty or returns list of files"""
    with patch("pathlib.Path.exists", return_value=False):
        res = client.get("/api/results")
        assert res.status_code == 200
        data = json.loads(res.data.decode("utf-8"))
        assert data["results"] == []


@patch("httpx.Client.get")
def test_api_proxy_status(mock_get, client):
    """Test proxy status parsing endpoint returns aggregated metrics when proxy is online"""
    # Setup mock responses
    mock_version = MagicMock()
    mock_version.status_code = 200
    mock_version.json.return_value = {"version": "0.3.1"}

    mock_system = MagicMock()
    mock_system.status_code = 200
    mock_system.json.return_value = {"hostname": "node1", "cpu_usage": {"percent": 15}}

    mock_metrics = MagicMock()
    mock_metrics.status_code = 200
    mock_metrics.json.return_value = {"requests_total": 42}

    mock_runtime = MagicMock()
    mock_runtime.status_code = 200
    mock_runtime.json.return_value = {"loaded_models": []}

    mock_slots = MagicMock()
    mock_slots.status_code = 200
    mock_slots.json.return_value = {"slots": []}

    mock_logs = MagicMock()
    mock_logs.status_code = 200
    mock_logs.json.return_value = {"logs": []}

    mock_get.side_effect = [
        mock_version,
        mock_system,
        mock_metrics,
        mock_runtime,
        mock_slots,
        mock_logs,
    ]

    res = client.get("/api/proxy/status")
    assert res.status_code == 200
    data = json.loads(res.data.decode("utf-8"))
    assert data["online"] is True
    assert data["system"]["hostname"] == "node1"
    assert data["metrics"]["requests_total"] == 42


@patch("threading.Thread.start")
def test_api_run_shared_llm_success(mock_thread_start, client):
    """Test starting SharedLLM benchmark succeeds and triggers background worker thread"""
    res = client.post("/api/run/shared_llm", json={"models": ["qwen3:8b"], "use_proxy": True})
    assert res.status_code == 200
    data = json.loads(res.data.decode("utf-8"))
    assert "SharedLLM Benchmark started" in data["status"]

    # Check status state
    status_res = client.get("/api/status")
    status_data = json.loads(status_res.data.decode("utf-8"))
    assert status_data["status"] == "running"
    assert status_data["type"] == "shared_llm"
    assert status_data["models"] == ["qwen3:8b"]

    mock_thread_start.assert_called_once()


def test_api_profiles_route(client, tmp_path, monkeypatch):
    """Test that the profiles route successfully reads models.ini settings"""
    mock_ini = tmp_path / "models.ini"
    mock_ini.write_text("""[*]
mlock = true
ctx-size = 8192

[test-model]
ctx-size = 16384
""")
    monkeypatch.setattr("web.app.get_models_ini_path", lambda: mock_ini)

    res = client.get("/api/profiles")
    assert res.status_code == 200
    data = json.loads(res.data.decode("utf-8"))
    assert "profiles" in data
    assert data["profiles"]["*"]["mlock"] == "true"
    assert data["profiles"]["test-model"]["ctx-size"] == "16384"


def test_api_profiles_save_route(client, tmp_path, monkeypatch):
    """Test that saving a profile correctly writes to the mock models.ini"""
    mock_ini = tmp_path / "models.ini"
    mock_ini.write_text("""[*]
mlock = true
""")
    monkeypatch.setattr("web.app.get_models_ini_path", lambda: mock_ini)

    payload = {"section": "new-model", "settings": {"ctx-size": 4096, "flash-attn": "on"}}

    res = client.post("/api/profiles/save", json=payload)
    assert res.status_code == 200
    data = json.loads(res.data.decode("utf-8"))
    assert "success" in data["status"]

    # Verify profile.json is created
    profile_json_path = tmp_path / "new-model.profile.json"
    assert profile_json_path.exists()
    with open(profile_json_path) as pf:
        pf_data = json.load(pf)
    assert pf_data["ctx-size"] == 4096
    assert pf_data["flash-attn"] == "on"

    import configparser

    config = configparser.ConfigParser(delimiters=("=",))
    config.read(str(mock_ini))
    assert config.has_section("new-model")
    assert config["new-model"]["ctx-size"] == "4096"
    assert config["new-model"]["flash-attn"] == "on"


def test_profile_save_keeps_harness_keys_out_of_models_ini(client, tmp_path, monkeypatch):
    """llama-server's preset parser rejects keys that are not its own arguments and
    the container crash-loops, so harness-only settings must stay in the overlay."""
    mock_ini = tmp_path / "models.ini"
    mock_ini.write_text("[*]\nmlock = true\n\n[thinker]\nthinking = on\n")
    monkeypatch.setattr("web.app.get_models_ini_path", lambda: mock_ini)

    payload = {
        "section": "thinker",
        "settings": {"ctx-size": 8192, "thinking": "off", "reasoning-budget": 2048, "cache-reuse": 256},
    }
    res = client.post("/api/profiles/save", json=payload)
    assert res.status_code == 200

    import configparser

    config = configparser.ConfigParser(delimiters=("=",))
    config.read(str(mock_ini))
    assert config["thinker"]["ctx-size"] == "8192"
    assert config["thinker"]["cache-reuse"] == "256"
    assert config["thinker"]["reasoning-budget"] == "2048"
    # Not a llama-server argument: dropped here, and the stale one is swept out.
    assert "thinking" not in config["thinker"]

    # ...but every setting still reaches the harness through the overlay.
    overlay = json.loads((tmp_path / "thinker.profile.json").read_text())
    assert overlay["thinking"] == "off"
    assert overlay["cache-reuse"] == 256


def test_profile_save_rejected_key_still_reaches_benchmark_suite(client, tmp_path, monkeypatch):
    """The benchmark suite reads thinking from the overlay, so filtering the ini
    must not change how a model is benchmarked."""
    import llm_benchmark_suite

    mock_ini = tmp_path / "models.ini"
    mock_ini.write_text("[*]\ntemperature = 0.6\n")
    monkeypatch.setattr("web.app.get_models_ini_path", lambda: mock_ini)
    client.post("/api/profiles/save", json={"section": "thinker", "settings": {"thinking": "off"}})

    monkeypatch.setenv("MODELS_INI_PATH", str(mock_ini))
    assert llm_benchmark_suite._model_thinking("thinker") is False


@patch("httpx.Client.post")
def test_api_proxy_restart_route(mock_post, client):
    """Test that restarting proxy triggers proxy endpoints or fallback subprocess"""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "success"}
    mock_post.return_value = mock_resp

    res = client.post("/api/proxy/restart")
    assert res.status_code == 200
    data = json.loads(res.data.decode("utf-8"))
    assert "success" in data["status"]


def test_api_profiles_delete_route(client, tmp_path, monkeypatch):
    """Test that deleting a profile correctly removes it from models.ini"""
    mock_ini = tmp_path / "models.ini"
    mock_ini.write_text("""[*]
mlock = true

[to-delete]
ctx-size = 4096
""")
    monkeypatch.setattr("web.app.get_models_ini_path", lambda: mock_ini)

    # Try deleting [*] defaults (should fail)
    res = client.post("/api/profiles/delete", json={"section": "*"})
    assert res.status_code == 400

    # Try deleting invalid section (should fail)
    res = client.post("/api/profiles/delete", json={"section": "non-existent"})
    assert res.status_code == 404

    # Write a mock profile.json to verify deletion
    mock_profile_json = tmp_path / "to-delete.profile.json"
    mock_profile_json.write_text('{"ctx-size": 4096}')
    assert mock_profile_json.exists()

    # Try deleting to-delete (should succeed)
    res = client.post("/api/profiles/delete", json={"section": "to-delete"})
    assert res.status_code == 200
    data = json.loads(res.data.decode("utf-8"))
    assert "success" in data["status"]

    # Verify profile.json was cleaned up
    assert not mock_profile_json.exists()

    import configparser

    config = configparser.ConfigParser(delimiters=("=",))
    config.read(str(mock_ini))
    assert not config.has_section("to-delete")
    assert config.has_section("*")


@patch("httpx.Client.get")
def test_api_logs_download_route(mock_get, client):
    """Test that downloading logs successfully streams data from proxy logs buffer"""
    # Mock proxy version check (online)
    mock_version = MagicMock()
    mock_version.status_code = 200

    # Mock logs response
    mock_logs = MagicMock()
    mock_logs.status_code = 200
    mock_logs.json.return_value = {"logs": ["line 1", "line 2"]}

    mock_get.side_effect = [mock_version, mock_logs]

    res = client.get("/api/logs/download")
    assert res.status_code == 200
    assert res.data == b"line 1\nline 2"
    assert res.headers["Content-Disposition"] == "attachment; filename=alpaca_proxy_system.log"


@patch("httpx.Client.post")
@patch("httpx.Client.get")
def test_api_models_delete_route(mock_get, mock_post, client):
    """Test that model deletion correctly proxies the request to the proxy server"""
    # Mock proxy version check (online)
    mock_version = MagicMock()
    mock_version.status_code = 200
    mock_get.return_value = mock_version

    # Mock proxy delete response
    mock_delete = MagicMock()
    mock_delete.status_code = 200
    mock_delete.json.return_value = {"status": "deleted", "model": "qwen:7b"}
    mock_post.return_value = mock_delete

    res = client.post("/api/models/delete", json={"model": "qwen:7b"})
    assert res.status_code == 200
    data = json.loads(res.data.decode("utf-8"))
    assert data["status"] == "deleted"
    assert data["model"] == "qwen:7b"


@patch("httpx.get")
def test_api_models_search_route(mock_get, client):
    """Test searching models from Ollama and Hugging Face"""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = '<html>href="/library/llama3.1" class="group w-full"><div class="flex flex-col mb-1" title="llama3.1"><h2 class="truncate text-xl font-medium underline-offset-2 group-hover:underline md:text-2xl"><span x-test-search-response-title>llama3.1</span></h2><p class="max-w-lg break-words text-neutral-800 text-md">Llama description</p></div></html>'
    mock_resp.json.return_value = [{"id": "Qwen/Qwen2.5-Coder-7B", "author": "Qwen"}]
    mock_get.return_value = mock_resp

    res = client.post("/api/models/search", json={"query": "llama", "source": "all"})
    assert res.status_code == 200
    data = json.loads(res.data.decode("utf-8"))
    assert "results" in data
    assert len(data["results"]) >= 2
    assert data["results"][0]["name"] == "llama3.1"
    assert data["results"][1]["name"] == "Qwen/Qwen2.5-Coder-7B"

    # Test precise repo lookup
    mock_resp_precise = MagicMock()
    mock_resp_precise.status_code = 200
    mock_resp_precise.json.return_value = {
        "id": "Precise/Repo",
        "author": "Precise",
        "downloads": 10,
        "likes": 5,
        "tags": ["gguf"],
    }
    mock_get.return_value = mock_resp_precise

    res = client.post("/api/models/search", json={"query": "Precise/Repo", "source": "huggingface"})
    assert res.status_code == 200
    data = json.loads(res.data.decode("utf-8"))
    assert len(data["results"]) == 1
    assert data["results"][0]["name"] == "Precise/Repo"
    assert "Direct Match" in data["results"][0]["description"]


@patch("httpx.get")
def test_api_models_hf_files_route(mock_get, client):
    """Test listing GGUF files in Hugging Face repository"""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "siblings": [
            {"rfilename": "model-q4_k_m.gguf", "size": 4829102910},
            {"rfilename": "readme.md"},
        ]
    }
    mock_get.return_value = mock_resp

    res = client.get("/api/models/huggingface/files?repo=Qwen/Qwen2.5")
    assert res.status_code == 200
    data = json.loads(res.data.decode("utf-8"))
    assert "files" in data
    assert len(data["files"]) == 1
    assert data["files"][0]["filename"] == "model-q4_k_m.gguf"
    assert "GB" in data["files"][0]["size"]


def test_api_models_active_pulls_route(client):
    """Test retrieving active pulls and logs"""
    res = client.get("/api/models/pulls/active")
    assert res.status_code == 200
    data = json.loads(res.data.decode("utf-8"))
    assert "active_pulls" in data
    assert len(data["active_pulls"]) == 0

    from web.app import active_pulls

    active_pulls["test-model"] = {
        "model": "test-model",
        "source": "huggingface",
        "local_name": "test-alias",
        "logs": ["Downloading...", "10% completed"],
    }

    try:
        res = client.get("/api/models/pulls/active")
        assert res.status_code == 200
        data = json.loads(res.data.decode("utf-8"))
        assert "active_pulls" in data
        assert "test-model" in data["active_pulls"]
        assert data["active_pulls"]["test-model"]["model"] == "test-model"
        assert len(data["active_pulls"]["test-model"]["logs"]) == 2
        assert data["active_pulls"]["test-model"]["logs"][0] == "Downloading..."
    finally:
        active_pulls.pop("test-model", None)


@patch("httpx.get")
def test_api_models_ollama_tags_route(mock_get, client):
    """Test retrieving tags for an Ollama library model"""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = '<html>href="/library/llama3:latest" ... href="/library/llama3:8b"</html>'
    mock_get.return_value = mock_resp

    res = client.get("/api/models/ollama/tags?model=llama3")
    assert res.status_code == 200
    data = json.loads(res.data.decode("utf-8"))
    assert "tags" in data
    assert len(data["tags"]) == 2
    # Tags are enriched objects {tag, size, size_bytes}; sizes are best-effort
    # from registry manifests and stay None when the lookup fails.
    assert data["tags"][0]["tag"] == "latest"
    assert data["tags"][1]["tag"] == "8b"
    assert "size" in data["tags"][0]
    assert "size_bytes" in data["tags"][0]


def test_api_pull_stop_returns_404_for_missing_pull(client):
    """Test that stopping a non-existent pull returns 404"""
    res = client.post("/api/models/pulls/nonexistent/stop")
    assert res.status_code == 404
    data = json.loads(res.data.decode("utf-8"))
    assert "error" in data


def test_api_pull_cancel_returns_404_for_missing_pull(client):
    """Test that cancelling a non-existent pull returns 404"""
    res = client.post("/api/models/pulls/nonexistent/cancel")
    assert res.status_code == 404
    data = json.loads(res.data.decode("utf-8"))
    assert "error" in data


@patch("pathlib.Path.mkdir")
@patch("pathlib.Path.write_text")
def test_api_pull_stop_success(mock_write, mock_mkdir, client):
    """Test that stopping an active pull sets status and creates marker file"""
    from web.app import active_pulls

    active_pulls["test-model"] = {
        "model": "test-model",
        "source": "ollama",
        "local_name": "",
        "status": "running",
        "logs": [],
    }

    try:
        res = client.post("/api/models/pulls/test-model/stop")
        assert res.status_code == 200
        data = json.loads(res.data.decode("utf-8"))
        assert data["status"] == "stopping"
        assert "Stopping" in data["message"]
        assert active_pulls["test-model"]["status"] == "stopping"
    finally:
        active_pulls.pop("test-model", None)


@patch("pathlib.Path.mkdir")
@patch("pathlib.Path.write_text")
def test_api_pull_stop_fails_if_not_running(mock_write, mock_mkdir, client):
    """Test that stopping a non-running pull returns an error"""
    from web.app import active_pulls

    active_pulls["test-model"] = {
        "model": "test-model",
        "source": "ollama",
        "local_name": "",
        "status": "success",
        "logs": [],
    }

    try:
        res = client.post("/api/models/pulls/test-model/stop")
        assert res.status_code == 400
        data = json.loads(res.data.decode("utf-8"))
        assert "error" in data
    finally:
        active_pulls.pop("test-model", None)


def test_api_pull_cancel_success(client):
    """Test that cancelling an active pull sets status to cancelled"""
    from web.app import active_pulls

    active_pulls["test-model"] = {
        "model": "test-model",
        "source": "ollama",
        "local_name": "",
        "status": "running",
        "logs": [],
    }

    try:
        res = client.post("/api/models/pulls/test-model/cancel")
        assert res.status_code == 200
        data = json.loads(res.data.decode("utf-8"))
        assert data["status"] == "cancelled"
        assert "cancelled" in data["message"].lower()
        assert active_pulls["test-model"]["status"] == "cancelled"
    finally:
        active_pulls.pop("test-model", None)


def test_api_pull_cancel_double_returns_error(client):
    """Test that cancelling an already-cancelled pull returns an error"""
    from web.app import active_pulls

    active_pulls["test-model"] = {
        "model": "test-model",
        "source": "ollama",
        "local_name": "",
        "status": "cancelled",
        "logs": [],
    }

    try:
        res = client.post("/api/models/pulls/test-model/cancel")
        assert res.status_code == 400
        data = json.loads(res.data.decode("utf-8"))
        assert "error" in data
        assert "already cancelled" in data["error"].lower()
    finally:
        active_pulls.pop("test-model", None)


def test_api_pull_trigger_with_no_resume(client):
    """Test that pull trigger accepts and passes no_resume flag"""
    with patch("threading.Thread.start"):
        res = client.post(
            "/api/models/pull",
            json={
                "model": "test-model",
                "source": "huggingface",
                "no_resume": True,
            },
        )
        assert res.status_code == 200
        data = json.loads(res.data.decode("utf-8"))
        assert data["status"] == "pulling_started"


def test_api_pull_trigger_duplicate_returns_409(client):
    """Test that triggering a pull for an already-active model returns 409"""
    from web.app import active_pulls

    active_pulls["test-model"] = {
        "model": "test-model",
        "source": "ollama",
        "local_name": "",
        "status": "running",
        "logs": [],
    }

    try:
        with patch("threading.Thread.start"):
            res = client.post("/api/models/pull", json={"model": "test-model"})
            assert res.status_code == 409
            data = json.loads(res.data.decode("utf-8"))
            assert "already being downloaded" in data["error"].lower()
    finally:
        active_pulls.pop("test-model", None)


def test_api_pull_trigger_missing_model_returns_400(client):
    """Test that triggering a pull without a model returns 400"""
    res = client.post("/api/models/pull", json={})
    assert res.status_code == 400
    data = json.loads(res.data.decode("utf-8"))
    assert "model is required" in data["error"]


@patch("pathlib.Path.exists", return_value=False)
def test_get_telemetry_history_no_model_returns_empty(mock_exists, client):
    """Test telemetry history with no model parameter returns empty list"""
    res = client.get("/api/telemetry/history")
    assert res.status_code == 200
    data = json.loads(res.data.decode("utf-8"))
    assert "history" in data
    assert data["model"] == "system_idle"
    assert data["history"] == []


@patch("pathlib.Path.exists", return_value=True)
@patch("builtins.open", new_callable=lambda: MagicMock())
def test_get_telemetry_history_with_file(mock_open, mock_exists, client):
    """Test telemetry history returns data when file exists"""
    import json as json_mod

    mock_open.return_value.__enter__.return_value.read.return_value = json_mod.dumps({"epoch_time": 1})
    res = client.get("/api/telemetry/history?model=test-model")
    assert res.status_code == 200
    data = json.loads(res.data.decode("utf-8"))
    assert "history" in data


def test_api_pulls_active_includes_status_field(client):
    """Test that active pulls endpoint includes the status field"""
    from web.app import active_pulls

    active_pulls["test-model"] = {
        "model": "test-model",
        "source": "ollama",
        "local_name": "alias",
        "status": "running",
        "logs": ["log entry"],
    }

    try:
        res = client.get("/api/models/pulls/active")
        assert res.status_code == 200
        data = json.loads(res.data.decode("utf-8"))
        assert data["active_pulls"]["test-model"]["status"] == "running"
        assert data["active_pulls"]["test-model"]["local_name"] == "alias"
    finally:
        active_pulls.pop("test-model", None)


def test_api_result_detail_get_and_delete(client):
    """Test retrieving and deleting result files via API"""
    mock_data = {"benchmark_version": "3.0.0", "results": []}
    mock_file_content = json.dumps(mock_data)

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=mock_file_content)),
        patch("os.remove") as mock_remove,
    ):
        # Test GET
        res = client.get("/api/results/benchmarks_12345_all_proxy.json")
        assert res.status_code == 200
        data = json.loads(res.data.decode("utf-8"))
        assert data["benchmark_version"] == "3.0.0"

        # Test DELETE
        res_del = client.delete("/api/results/benchmarks_12345_all_proxy.json")
        assert res_del.status_code == 200
        del_data = json.loads(res_del.data.decode("utf-8"))
        assert del_data["status"] == "deleted"
        mock_remove.assert_called_once()


def test_api_errors_get_fallback_to_file(client):
    """Test that /api/errors falls back to reading the JSONL file directly if the proxy is offline"""
    # Mock Path.exists to return True and open to return sample errors file content
    mock_file_content = (
        '{"timestamp": "2026-07-06T09:00:00Z", "model": "qwen", "error_type": "oom", "message": "OOM"}\n'
        '{"timestamp": "2026-07-06T09:05:00Z", "model": "llama", "error_type": "context_overflow", "message": "overflow"}\n'
    )
    # We force the proxy connection check to fail by mocking Client.get to raise an error
    with (
        patch("httpx.Client.get", side_effect=Exception("Proxy offline")),
        patch("pathlib.Path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=mock_file_content)),
    ):
        res = client.get("/api/errors")
        assert res.status_code == 200
        data = json.loads(res.data.decode("utf-8"))
        assert data["total"] == 2
        assert data["error_type_counts"]["oom"] == 1
        assert data["error_type_counts"]["context_overflow"] == 1
        assert data["errors"][0]["model"] == "llama"  # most recent (last line) is returned first

        # Test model filter in fallback mode
        res_filtered = client.get("/api/errors?model=qwen")
        data_filtered = json.loads(res_filtered.data.decode("utf-8"))
        assert data_filtered["total"] == 1
        assert data_filtered["errors"][0]["model"] == "qwen"


def test_api_errors_get_and_clear_proxy(client):
    """Test that /api/errors and /api/errors/clear successfully proxy when proxy is online"""
    mock_proxy_res = MagicMock()
    mock_proxy_res.status_code = 200
    mock_proxy_res.json.return_value = {
        "total": 1,
        "error_type_counts": {"oom": 1},
        "errors": [{"model": "qwen", "error_type": "oom", "message": "OOM"}],
    }

    mock_clear_res = MagicMock()
    mock_clear_res.status_code = 200
    mock_clear_res.json.return_value = {"status": "cleared"}

    def mock_client_request(client_self, method, url, **kwargs):
        # First call is to /api/version to check if proxy is online
        if "/api/version" in url:
            mock_ver = MagicMock()
            mock_ver.status_code = 200
            return mock_ver
        if "/admin/errors/clear" in url:
            return mock_clear_res
        if "/admin/errors" in url:
            return mock_proxy_res
        raise ValueError(f"Unexpected url: {url}")

    with patch("httpx.Client.request", new=mock_client_request):
        # Test proxy GET
        res = client.get("/api/errors")
        assert res.status_code == 200
        data = json.loads(res.data.decode("utf-8"))
        assert data["total"] == 1
        assert data["error_type_counts"]["oom"] == 1

        # Test proxy POST clear
        res_clear = client.post("/api/errors/clear")
        assert res_clear.status_code == 200
        data_clear = json.loads(res_clear.data.decode("utf-8"))
        assert data_clear["status"] == "cleared"


# Image Studio proxy adapter tests


def test_sd_presets_api_forwards_proxy_presets(client):
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"qwen_image_21.identity": {"version": 1}}
    with patch("httpx.Client.get", return_value=mock_resp):
        response = client.get("/api/sd/presets")
    assert response.status_code == 200
    assert response.get_json()["qwen_image_21.identity"]["version"] == 1


def test_sd_edit_api_forwards_named_reference_files_in_order(client):
    from io import BytesIO

    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"data": []}
    with patch("httpx.Client.post", return_value=mock_resp) as post:
        response = client.post(
            "/api/sd/edit",
            data={
                "model": "Qwen-Image-2.1-GGUF/qwen_image_2.1-Q4_K",
                "preset": "qwen_image_21.identity",
                "reference_roles": '["scene", "person", "face"]',
                "image__scene": (BytesIO(b"scene"), "scene.png"),
                "image__person": (BytesIO(b"person"), "person.png"),
                "image__face": (BytesIO(b"face"), "face.png"),
            },
            content_type="multipart/form-data",
        )
    assert response.status_code == 200
    files = post.call_args.kwargs["files"]
    assert [key for key, _file in files] == [
        "image__scene",
        "image__person",
        "image__face",
    ]


# Vision & Image-to-Prompt Assistant API Tests


def test_vision_describe_api_missing_file(client):
    """Test that /api/vision/describe returns 400 when no file is uploaded"""
    res = client.post("/api/vision/describe", data={})
    assert res.status_code == 400
    data = json.loads(res.data.decode("utf-8"))
    assert "error" in data
    assert "No image file uploaded" in data["error"]


def test_vision_describe_api_success(client):
    """Test /api/vision/describe with a valid uploaded image file"""
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    img = Image.new("RGB", (100, 100), color="blue")
    img.save(buf, format="JPEG")
    buf.seek(0)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "A high-resolution photo of a blue studio background with soft lighting."}}]
    }

    with patch("httpx.Client.post", return_value=mock_resp):
        res = client.post("/api/vision/describe", data={"image": (buf, "test.jpg")}, content_type="multipart/form-data")
        assert res.status_code == 200
        data = json.loads(res.data.decode("utf-8"))
        assert data["status"] == "success"
        assert "blue studio background" in data["image_description"]


def test_vision_synthesize_edit_prompt_missing_params(client):
    """Test that /api/vision/synthesize_edit_prompt returns 400 when required params are missing"""
    res = client.post("/api/vision/synthesize_edit_prompt", json={})
    assert res.status_code == 400
    data = json.loads(res.data.decode("utf-8"))
    assert "error" in data
    assert "required" in data["error"]


def test_vision_synthesize_edit_prompt_success(client):
    """Test /api/vision/synthesize_edit_prompt with valid base description and desired changes"""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Master Edit Prompt: A person with purple neon hair in a cyberpunk city, 8k resolution, raw photo."
                }
            }
        ]
    }
    mock_text_models = MagicMock()
    mock_text_models.get_json.return_value = {"models": ["qwen-test:7b"]}

    payload = {
        "base_description": "A woman wearing a white shirt in a studio",
        "desired_changes": "Change her hair to purple neon and add cyberpunk city background",
        "style_preset": "Cyberpunk Sci-Fi",
    }

    with (
        patch("web.app.get_text_models", return_value=mock_text_models),
        patch("httpx.Client.post", return_value=mock_resp),
    ):
        res = client.post("/api/vision/synthesize_edit_prompt", json=payload)
        assert res.status_code == 200
        data = json.loads(res.data.decode("utf-8"))
        assert data["status"] == "success"
        assert "purple neon" in data["master_prompt"]
        assert "suggested_strength" in data
        assert "suggested_negative" in data


def test_vision_synthesize_edit_prompt_fallback(client):
    """Test /api/vision/synthesize_edit_prompt fallback logic when LLM call raises exception"""
    mock_text_models = MagicMock()
    mock_text_models.get_json.return_value = {"models": ["qwen-test:7b"]}
    payload = {
        "base_description": "A portrait of a dog",
        "desired_changes": "Add superhero cape",
        "style_preset": "Anime Fantasy",
    }

    with (
        patch("web.app.get_text_models", return_value=mock_text_models),
        patch("httpx.Client.post", side_effect=Exception("Connection error")),
    ):
        res = client.post("/api/vision/synthesize_edit_prompt", json=payload)
        assert res.status_code == 200
        data = json.loads(res.data.decode("utf-8"))
        assert data["status"] == "success"
        assert "superhero cape" in data["master_prompt"]


def test_vision_synthesize_edit_prompt_no_models(client):
    """Test /api/vision/synthesize_edit_prompt fails fast when no text models exist (no hardcoded fallback)."""
    mock_text_models = MagicMock()
    mock_text_models.get_json.return_value = {"models": []}
    payload = {
        "base_description": "A portrait of a dog",
        "desired_changes": "Add superhero cape",
    }

    with patch("web.app.get_text_models", return_value=mock_text_models):
        res = client.post("/api/vision/synthesize_edit_prompt", json=payload)
        assert res.status_code == 503
        data = json.loads(res.data.decode("utf-8"))
        assert "No text models available" in data["error"]


def test_api_online_providers_get(client):
    """Test GET /api/online/providers returns provider configuration status."""
    res = client.get("/api/online/providers")
    assert res.status_code == 200
    data = json.loads(res.data.decode("utf-8"))
    assert "providers" in data
    assert "alpaca" in data["providers"]
    assert "openrouter" in data["providers"]
    assert "huggingface" in data["providers"]
    assert "cloudflare" in data["providers"]
    assert "opencode_zen" in data["providers"]
    assert "groq" in data["providers"]
    assert "gemini" in data["providers"]


def test_api_online_providers_alpaca_generate(client):
    """Test POST /api/online/providers/alpaca/generate produces a valid alpaca-sk- token."""
    res = client.post("/api/online/providers/alpaca/generate")
    assert res.status_code == 200
    data = json.loads(res.data.decode("utf-8"))
    assert data["success"] is True
    assert data["token"].startswith("alpaca-sk-")
    assert len(data["token"]) >= 20


def test_api_online_providers_save_and_test(client):
    """Test saving credentials and testing provider connection."""
    with patch(
        "online_providers.online_model_provider.save_credentials",
        return_value={"success": True, "configured": {"alpaca": True}},
    ):
        res = client.post("/api/online/providers/save", json={"alpaca_api_key": "alpaca-sk-test-token"})
        assert res.status_code == 200
        data = json.loads(res.data.decode("utf-8"))
        assert data["success"] is True

    with patch(
        "online_providers.online_model_provider.test_connection",
        return_value={"success": True, "message": "Test successful"},
    ):
        res = client.post(
            "/api/online/providers/test", json={"provider": "openrouter", "keys": {"openrouter_api_key": "sk-or-test"}}
        )
        assert res.status_code == 200
        data = json.loads(res.data.decode("utf-8"))
        assert data["success"] is True
        assert data["message"] == "Test successful"


def test_api_online_models_search_and_selected(client):
    """Test searching online models and persisting user selection."""
    mock_models = [
        {
            "id": "openrouter:google/gemini-2.0-flash-exp:free",
            "name": "gemini-2.0-flash",
            "label": "Gemini 2.0 Flash",
            "provider": "openrouter",
            "free": True,
        }
    ]
    with patch("online_providers.online_model_provider.fetch_live_models", return_value=mock_models):
        res = client.get("/api/online/models/search?provider=openrouter&free_only=true")
        assert res.status_code == 200
        data = json.loads(res.data.decode("utf-8"))
        assert data["success"] is True
        assert len(data["models"]) == 1
        assert data["models"][0]["id"] == "openrouter:google/gemini-2.0-flash-exp:free"

    # Test saving selection
    with patch(
        "online_providers.online_model_provider.save_selected_models", return_value={"success": True, "count": 1}
    ):
        res = client.post("/api/online/models/selected", json={"models": mock_models})
        assert res.status_code == 200
        data = json.loads(res.data.decode("utf-8"))
        assert data["success"] is True
        assert data["count"] == 1


def test_api_online_models_selected_clears_hidden(client):
    """Re-adding models to the selection clears any removal-hiding flag."""
    mock_models = [
        {
            "id": "openrouter:deepseek/deepseek-v4.1-flash",
            "name": "deepseek-v4.1-flash",
            "provider": "openrouter",
        }
    ]
    with (
        patch(
            "online_providers.online_model_provider.save_selected_models",
            return_value={"success": True, "count": 1},
        ),
        patch("web.app.model_tracker") as mock_tracker,
    ):
        res = client.post("/api/online/models/selected", json={"models": mock_models})
        assert res.status_code == 200
        mock_tracker.set_hidden.assert_called_once_with("openrouter:deepseek/deepseek-v4.1-flash", False)


def _online_selection():
    return [
        {"id": "openrouter:google/gemini-2.0-flash-exp:free", "name": "gemini-2.0-flash", "provider": "openrouter"},
        {"id": "openrouter:deepseek/deepseek-v4.1-flash", "name": "deepseek-v4.1-flash", "provider": "openrouter"},
    ]


def test_api_online_models_remove_model_only(client):
    """Remove selection entry, keep benchmarks: selection filtered, purge untouched."""
    saved = {}
    with (
        patch(
            "online_providers.online_model_provider.get_selected_models",
            return_value=_online_selection(),
        ),
        patch(
            "online_providers.online_model_provider.save_selected_models",
            side_effect=lambda models: saved.update(models=models) or {"success": True, "count": len(models)},
        ),
        patch("web.app._purge_model_benchmarks") as mock_purge,
        patch("web.app.model_tracker") as mock_tracker,
    ):
        res = client.post(
            "/api/online/models/remove",
            json={"model": "openrouter:deepseek/deepseek-v4.1-flash", "remove_model": True},
        )
        assert res.status_code == 200
        data = json.loads(res.data.decode("utf-8"))
        assert data["success"] is True
        assert data["model_removed"] is True
        assert [m["id"] for m in saved["models"]] == ["openrouter:google/gemini-2.0-flash-exp:free"]
        mock_purge.assert_not_called()
        # History is kept, so the entry must be hidden to stay out of the run list.
        mock_tracker.set_hidden.assert_called_once_with("openrouter:deepseek/deepseek-v4.1-flash", True)


def test_api_online_models_remove_benchmarks_only(client):
    """Purge benchmarks, keep selection: purge called, selection file untouched."""
    with (
        patch(
            "online_providers.online_model_provider.get_selected_models",
            return_value=_online_selection(),
        ),
        patch(
            "online_providers.online_model_provider.save_selected_models",
            side_effect=AssertionError("selection must not be rewritten"),
        ),
        patch(
            "web.app._purge_model_benchmarks",
            return_value={"removed": True, "general": False, "shared": True, "snapshots_pruned": 1},
        ) as mock_purge,
        patch("web.app.model_tracker") as mock_tracker,
    ):
        res = client.post(
            "/api/online/models/remove",
            json={
                "model": "openrouter:deepseek/deepseek-v4.1-flash",
                "remove_model": False,
                "remove_benchmarks": True,
            },
        )
        assert res.status_code == 200
        data = json.loads(res.data.decode("utf-8"))
        assert data["success"] is True
        assert data["model_removed"] is False
        assert data["benchmark_results_removed"]["removed"] is True
        mock_purge.assert_called_once_with("openrouter:deepseek/deepseek-v4.1-flash")
        # Selection untouched, so nothing to hide.
        mock_tracker.set_hidden.assert_not_called()


def test_api_online_models_remove_both(client):
    """Remove entry and purge benchmarks in one call."""
    saved = {}
    with (
        patch(
            "online_providers.online_model_provider.get_selected_models",
            return_value=_online_selection(),
        ),
        patch(
            "online_providers.online_model_provider.save_selected_models",
            side_effect=lambda models: saved.update(models=models) or {"success": True, "count": len(models)},
        ),
        patch(
            "web.app._purge_model_benchmarks",
            return_value={"removed": True, "general": True, "shared": True, "snapshots_pruned": 2},
        ) as mock_purge,
        patch("web.app.model_tracker") as mock_tracker,
    ):
        res = client.post(
            "/api/online/models/remove",
            json={
                "model": "openrouter:deepseek/deepseek-v4.1-flash",
                "remove_model": True,
                "remove_benchmarks": True,
            },
        )
        assert res.status_code == 200
        data = json.loads(res.data.decode("utf-8"))
        assert data["success"] is True
        assert data["model_removed"] is True
        assert len(saved["models"]) == 1
        assert data["benchmark_results_removed"]["snapshots_pruned"] == 2
        mock_purge.assert_called_once()
        # History is purged too, so the tracker entry is gone — no hiding needed.
        mock_tracker.set_hidden.assert_not_called()


def test_api_online_models_remove_validation(client):
    """Missing model, local model, and neither-flag requests are rejected."""
    res = client.post("/api/online/models/remove", json={})
    assert res.status_code == 400

    res = client.post("/api/online/models/remove", json={"model": "qwen3:8b", "remove_model": True})
    assert res.status_code == 400
    assert "not an online model" in json.loads(res.data.decode("utf-8"))["error"]

    res = client.post(
        "/api/online/models/remove",
        json={"model": "openrouter:deepseek/deepseek-v4.1-flash", "remove_model": False, "remove_benchmarks": False},
    )
    assert res.status_code == 400


def test_model_tracker_set_hidden_roundtrip(tmp_path):
    """Hidden flag persists on the entry and is exposed via the summary."""
    tracker = ModelTracker(data_dir=tmp_path)
    model = "openrouter:deepseek/deepseek-v4.1-flash"
    tracker.record_model_seen(model, source="openrouter")
    tracker.record_benchmark_result(model, 82.5, "shared", "shared_llm_benchmarks_x.json")

    assert tracker.set_hidden(model, True) is True
    summary = tracker.get_tracking_summary(current_online_models=[])
    assert summary["all_tracked"][model]["hidden_from_selection"] is True
    # Score history is untouched by hiding.
    assert summary["all_tracked"][model]["latest_score"] == 82.5

    assert tracker.set_hidden(model, False) is True
    summary = tracker.get_tracking_summary(current_online_models=[])
    assert summary["all_tracked"][model]["hidden_from_selection"] is False


def test_model_tracker_hidden_survives_history_rescan(tmp_path):
    """Later benchmark-result writes (history rescans) must not clear the flag."""
    tracker = ModelTracker(data_dir=tmp_path)
    model = "openrouter:deepseek/deepseek-v4.1-flash"
    tracker.record_model_seen(model, source="openrouter")
    assert tracker.set_hidden(model, True) is True

    tracker.record_benchmark_result(model, 90.0, "shared", "shared_llm_benchmarks_y.json")
    summary = tracker.get_tracking_summary(current_online_models=[])
    assert summary["all_tracked"][model]["hidden_from_selection"] is True
    assert summary["all_tracked"][model]["latest_score"] == 90.0


def test_llm_server_settings_get_reports_saved_vs_live(client):
    with (
        patch(
            "web.app._read_dotenv_values",
            return_value={"LLAMA_REASONING_BUDGET": "2048", "LLAMA_REASONING_FORMAT": "deepseek"},
        ),
        patch("web.app._llama_server_live_env", return_value={"LLAMA_REASONING_BUDGET": "1024"}),
    ):
        res = client.get("/api/settings/llm-server")
        assert res.status_code == 200
        data = json.loads(res.data.decode("utf-8"))
        assert data["budget"] == "2048"
        assert data["live_budget"] == "1024"
        assert data["needs_apply"] is True


def test_llm_server_settings_post_rejects_bad_values(client):
    res = client.post("/api/settings/llm-server", json={"budget": "lots", "format": "deepseek"})
    assert res.status_code == 400
    res = client.post("/api/settings/llm-server", json={"budget": -5, "format": "deepseek"})
    assert res.status_code == 400
    res = client.post("/api/settings/llm-server", json={"budget": 2048, "format": "telepathy"})
    assert res.status_code == 400


def test_llm_server_settings_post_saves_and_recreates(client):
    from online_providers import online_model_provider

    fake_old = MagicMock()
    fake_old.attrs = {"Config": {"Env": ["LLAMA_REASONING_BUDGET=1024", "OTHER=1"]}, "HostConfig": {}}
    fake_client = MagicMock()
    fake_client.containers.get.return_value = fake_old
    fake_new = MagicMock()
    fake_client.containers.run.return_value = fake_new

    with (
        patch.object(online_model_provider, "save_credentials", return_value={"success": True}),
        patch("web.app.docker") as mock_docker,
    ):
        mock_docker.from_env.return_value = fake_client
        mock_docker.types.DeviceRequest.side_effect = lambda **kw: kw
        mock_docker.types.Ulimit.side_effect = lambda **kw: kw
        mock_docker.types.LogConfig.side_effect = lambda **kw: kw
        res = client.post("/api/settings/llm-server", json={"budget": 2048, "format": "deepseek"})
        assert res.status_code == 200
        data = json.loads(res.data.decode("utf-8"))
        assert data["success"] is True and data["applied"] is True
        _, kwargs = fake_client.containers.run.call_args
        assert kwargs["environment"]["LLAMA_REASONING_BUDGET"] == "2048"
        assert kwargs["environment"]["LLAMA_REASONING_FORMAT"] == "deepseek"
        assert kwargs["environment"]["OTHER"] == "1"
        assert kwargs["name"] == "llama-server"


def test_llama_recreate_kwargs_preserves_config():
    from web.app import _llama_recreate_kwargs

    attrs = {
        "Config": {
            "Image": "img:tag",
            "Cmd": None,
            "Entrypoint": ["/bin/sh", "entry.sh"],
            "Env": ["A=1", "LLAMA_REASONING_BUDGET=1024"],
            "Labels": {"com.docker.compose.project": "alpaca"},
        },
        "HostConfig": {
            "Binds": ["/host/models:/models:ro", "/host/data:/data:rw"],
            "PortBindings": {"8080/tcp": [{"HostPort": "8080"}]},
            "Memory": 100,
            "CapAdd": ["IPC_LOCK"],
            "RestartPolicy": {"Name": "always"},
            "DeviceRequests": [
                {"Driver": "nvidia", "Count": -1, "DeviceIDs": [], "Capabilities": [["gpu"]], "Options": {}}
            ],
        },
        "NetworkSettings": {"Networks": {"alpaca_default": {}}},
    }
    kwargs, networks = _llama_recreate_kwargs(attrs, {"LLAMA_REASONING_BUDGET": "2048"})
    assert kwargs["image"] == "img:tag"
    assert kwargs["environment"] == {"A": "1", "LLAMA_REASONING_BUDGET": "2048"}
    assert kwargs["volumes"] == {
        "/host/models": {"bind": "/models", "mode": "ro"},
        "/host/data": {"bind": "/data", "mode": "rw"},
    }
    assert kwargs["ports"] == {"8080": "8080"}
    assert networks == ["alpaca_default"]


@patch("llm_benchmark_suite.LLMModelBenchmark.discover_all_models")
@patch("llm_benchmark_suite.LLMModelBenchmark.discover_all_proxy_models")
def test_api_models_tracking(mock_discover_proxy, mock_discover_all, client):
    """Test GET /api/models/tracking returns structured new vs benchmarked lists."""
    mock_discover_all.return_value = ["model1"]
    mock_discover_proxy.return_value = ["model2"]
    res = client.get("/api/models/tracking")
    assert res.status_code == 200
    data = json.loads(res.data.decode("utf-8"))
    assert data["success"] is True
    assert "newly_added" in data
    assert "previously_benchmarked" in data
    assert "counts" in data
    assert "all_tracked" in data
    assert data["counts"]["total"] >= 0


def test_api_models_tracking_discovery_failure(client, monkeypatch):
    """Genuine discovery errors surface an explicit error instead of hardcoded fallback models."""
    from web.app import benchmark

    def _boom():
        raise RuntimeError("No models discovered and BENCHMARK_MODELS is not set.")

    monkeypatch.setattr(benchmark, "discover_all_models", _boom)
    res = client.get("/api/models/tracking")
    assert res.status_code == 500
    data = json.loads(res.data.decode("utf-8"))
    assert data["success"] is False
    assert "BENCHMARK_MODELS" in data["error"]


def test_api_models_tracking_empty_discovery_uses_router_models(client):
    """Discovery returning [] (no raise) must not 500 - empty lists are a valid result."""
    res = client.get("/api/models/tracking")
    assert res.status_code == 200
    data = json.loads(res.data.decode("utf-8"))
    assert data["success"] is True


def test_delete_model_benchmarks_comprehensive(client, tmp_path):
    """Test DELETE /api/benchmarks/model/<model> completely purges per-model files, shared files, and snapshots."""
    from web.app import benchmark, shared_llm_benchmark

    # Setup mock dirs
    gen_models_dir = tmp_path / "general_models"
    gen_results_dir = tmp_path / "general_results"
    shared_models_dir = tmp_path / "shared_models"
    shared_results_dir = tmp_path / "shared_results"

    gen_models_dir.mkdir(parents=True, exist_ok=True)
    gen_results_dir.mkdir(parents=True, exist_ok=True)
    shared_models_dir.mkdir(parents=True, exist_ok=True)
    shared_results_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy general and shared per-model files
    (gen_models_dir / "general_qwen_test.json").write_text(json.dumps({"model": "qwen_test", "results": []}))
    (shared_models_dir / "shared_qwen_test.json").write_text(json.dumps({"model": "qwen_test", "results": []}))

    # Create dummy snapshots
    (gen_results_dir / "benchmarks_1.json").write_text(
        json.dumps(
            {
                "models_tested": ["qwen_test", "other_model"],
                "results": [{"model": "qwen_test"}, {"model": "other_model"}],
            }
        )
    )
    (shared_results_dir / "shared_llm_benchmarks_1.json").write_text(
        json.dumps({"models_tested": ["qwen_test"], "results": [{"model": "qwen_test"}]})
    )

    with (
        patch.object(benchmark, "MODELS_DIR", gen_models_dir),
        patch.object(benchmark, "RESULTS_DIR", gen_results_dir),
        patch.object(shared_llm_benchmark, "MODELS_DIR", shared_models_dir),
        patch.object(shared_llm_benchmark, "RESULTS_DIR", shared_results_dir),
    ):
        res = client.delete("/api/benchmarks/model/qwen_test")
        assert res.status_code == 200
        data = json.loads(res.data.decode("utf-8"))
        assert data["status"] == "deleted"
        assert data["removed"] is True

        # Verify per-model files deleted
        assert not (gen_models_dir / "general_qwen_test.json").exists()
        assert not (shared_models_dir / "shared_qwen_test.json").exists()

        # Verify multi-model snapshot was updated (qwen_test removed, other_model kept)
        assert (gen_results_dir / "benchmarks_1.json").exists()
        updated_snap = json.loads((gen_results_dir / "benchmarks_1.json").read_text())
        assert updated_snap["models_tested"] == ["other_model"]
        assert len(updated_snap["results"]) == 1

        # Verify single-model snapshot was unlinked
        assert not (shared_results_dir / "shared_llm_benchmarks_1.json").exists()


def test_clear_all_benchmarks(client, tmp_path):
    """Test POST /api/benchmarks/clear removes all benchmark data, models, and resets tracker."""
    from web.app import benchmark, model_tracker, shared_llm_benchmark

    gen_models_dir = tmp_path / "g_models"
    gen_results_dir = tmp_path / "g_results"
    gen_art_dir = tmp_path / "g_artifacts"
    shared_models_dir = tmp_path / "s_models"
    shared_results_dir = tmp_path / "s_results"
    shared_art_dir = tmp_path / "s_artifacts"

    for d in [gen_models_dir, gen_results_dir, gen_art_dir, shared_models_dir, shared_results_dir, shared_art_dir]:
        d.mkdir(parents=True, exist_ok=True)

    (gen_models_dir / "general_m1.json").write_text("{}")
    (gen_results_dir / "benchmarks_1.json").write_text("{}")
    (shared_models_dir / "shared_m1.json").write_text("{}")
    (shared_results_dir / "shared_llm_benchmarks_1.json").write_text("{}")

    with (
        patch.object(benchmark, "MODELS_DIR", gen_models_dir),
        patch.object(benchmark, "RESULTS_DIR", gen_results_dir),
        patch.object(benchmark, "ARTIFACTS_DIR", gen_art_dir),
        patch.object(shared_llm_benchmark, "MODELS_DIR", shared_models_dir),
        patch.object(shared_llm_benchmark, "RESULTS_DIR", shared_results_dir),
        patch.object(shared_llm_benchmark, "ARTIFACTS_DIR", shared_art_dir),
        patch.object(model_tracker, "clear_all", return_value=True) as mock_clear,
    ):
        res = client.post("/api/benchmarks/clear")
        assert res.status_code == 200
        data = json.loads(res.data.decode("utf-8"))
        assert data["status"] == "cleared"
        assert data["files_removed"] == 4
        mock_clear.assert_called_once()


@pytest.mark.parametrize("lint_fields", [{"lint_passed": None}, {"lint_passed": True}, {"lint_passed": False}, {}])
@pytest.mark.parametrize("code_ran", [None, True, False])
def test_api_tests_preserves_unknown_lint(client, tmp_path, lint_fields, code_ran):
    from web.app import benchmark

    record = {
        "model": "basic-model",
        "results": [
            {
                "category_coding": {
                    "tests": [
                        {
                            "test_id": "debug_fix",
                            "score": 0,
                            "code_ran": code_ran,
                            "code_quality": {"language": "basic"},
                            **lint_fields,
                        }
                    ]
                }
            }
        ],
    }
    (tmp_path / "general_basic-model.json").write_text(json.dumps(record))
    with patch.object(benchmark, "MODELS_DIR", tmp_path), patch.object(benchmark, "RESULTS_DIR", tmp_path):
        response = client.get("/api/tests")
    assert response.status_code == 200
    test = next(t for t in response.get_json()["tests"] if t["id"] == "debug_fix")
    assert test["models_lint"]["basic-model"] is lint_fields.get("lint_passed", code_ran is not False)
    assert test["models_breakdown"]["basic-model"]["lint_passed"] is lint_fields.get("lint_passed")


@pytest.mark.parametrize("value,label", [(None, "not run / unknown"), (True, "passed"), (False, "FAILED")])
def test_dashboard_lint_rendering(client, value, label):
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for dashboard rendering tests")
    response = client.get("/static/js/dashboard.js")
    assert response.status_code == 200
    source = response.get_data(as_text=True)
    start = source.index("const lp = lint[m];")
    end = source.index("const lr = lastRun[m];", start)
    script = f"const m = 'basic-model'; const lint = {{[m]: {json.dumps(value)}}};\n" + source[start:end]
    result = subprocess.run([node, "-e", script + "\nconsole.log(lintCell);"], capture_output=True, text=True, check=True)
    assert label in result.stdout
    if value is None:
        assert "FAILED" not in result.stdout
        assert "test-stats-lint-fail" not in result.stdout


def test_api_tests_run_stats_and_currency(client, tmp_path):
    """Test GET /api/tests returns model run counts and accurately detects out-of-date tests."""
    from web.app import _compute_test_hash, benchmark

    gen_models_dir = tmp_path / "models"
    gen_models_dir.mkdir(parents=True, exist_ok=True)

    cur_coding_test = next(t for t in benchmark.tests_config.get("coding", []) if t.get("id") == "debug_fix")
    cur_hash = _compute_test_hash(cur_coding_test)

    m1_data = {
        "model": "model1",
        "results": [
            {
                "category_coding": {
                    "tests": [
                        {
                            "test_id": "debug_fix",
                            "success": True,
                            "score": 100,
                            "test_hash": cur_hash,
                            "last_run": "2026-08-17T01:00:00",
                        },
                        {
                            "test_id": "guess_game",
                            "success": False,
                            "score": 0,
                            "test_hash": "old_stale_hash",
                            "last_run": "2026-08-16T01:00:00",
                        },
                    ]
                }
            }
        ],
    }
    (gen_models_dir / "general_model1.json").write_text(json.dumps(m1_data))

    with patch.object(benchmark, "MODELS_DIR", gen_models_dir), patch.object(benchmark, "RESULTS_DIR", tmp_path):
        res = client.get("/api/tests")
        assert res.status_code == 200
        data = json.loads(res.data.decode("utf-8"))
        assert "tests" in data
        tests_by_id = {t["id"]: t for t in data["tests"]}

        # debug_fix has 1 run, passed, up to date
        assert "debug_fix" in tests_by_id
        df = tests_by_id["debug_fix"]
        assert df["models_tested_count"] == 1
        assert df["models_passed_count"] == 1
        assert df["models_failed_count"] == 0
        assert df["is_out_of_date"] is False
        assert df["models_lint"].get("model1") is True
        assert df["models_last_run"].get("model1") == "2026-08-17T01:00:00"
        assert df["last_run"] == "2026-08-17T01:00:00"
        assert "model1" in df["models_passed"]
        assert df["models_run_count"].get("model1") == 1
        assert df["models_fail_count"].get("model1") == 0
        assert df["models_latency"].get("model1") == 0.0
        assert df["models_tokens"].get("model1") == 0
        assert df["models_speed"].get("model1") == 0.0

        # guess_game has 1 run, failed, out of date
        assert "guess_game" in tests_by_id
        gg = tests_by_id["guess_game"]
        assert gg["models_tested_count"] == 1
        assert gg["models_passed_count"] == 0
        assert gg["models_failed_count"] == 1
        assert "model1" not in gg.get("models_passed", [])
        assert gg["is_out_of_date"] is True
        assert "model1" in gg["out_of_date_models"]

        # logic_puzzle has 0 runs
        assert "logic_puzzle" in tests_by_id
        lp = tests_by_id["logic_puzzle"]
        assert lp["models_tested_count"] == 0
        assert lp["is_out_of_date"] is False


def test_sandbox_serve_proxy_forwards_to_upstream(client):
    """Serve proxy tunnels through the dashboard origin, never assuming localhost."""
    from unittest.mock import Mock

    with (
        patch("web.app._serve_container_host_port", return_value="39876"),
        patch("web.app.httpx.Client") as mock_client_cls,
    ):
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.content = b"<html><body>hello sandbox</body></html>"
        mock_resp.headers = {"content-type": "text/html", "content-length": "44"}
        mock_client = Mock()
        mock_client.request.return_value = mock_resp
        mock_client_cls.return_value.__enter__.return_value = mock_client

        res = client.get("/serve/abc123def/")

    assert res.status_code == 200
    assert b"hello sandbox" in res.data
    # Request must go to host.docker.internal (NOT localhost) with the container port
    call_url = mock_client.request.call_args[0][1]
    assert call_url.startswith("http://host.docker.internal:39876/")
    # Method is preserved
    assert mock_client.request.call_args[0][0] == "GET"


def test_sandbox_serve_proxy_preserves_subpath_and_method(client):
    from unittest.mock import Mock

    with (
        patch("web.app._serve_container_host_port", return_value="50001"),
        patch("web.app.httpx.Client") as mock_client_cls,
    ):
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.content = b"ok"
        mock_resp.headers = {"content-type": "text/plain"}
        mock_client = Mock()
        mock_client.request.return_value = mock_resp
        mock_client_cls.return_value.__enter__.return_value = mock_client

        res = client.post("/serve/cid123/static/style.css?x=1", json={"a": 1})

    assert res.status_code == 200
    call_url = mock_client.request.call_args[0][1]
    assert call_url == "http://host.docker.internal:50001/static/style.css?x=1"
    assert mock_client.request.call_args[0][0] == "POST"


def test_sandbox_serve_proxy_injects_vd_guard_in_vnc_html(client):
    """vnc.html gets the VideoDecoder wedge guard; other pages untouched.

    noVNC top-level-awaits a VideoDecoder capability check: on a phone with
    a cold media stack that call never settles and the viewer sits at its
    spinner with zero errors. The guard races it with an 8s timeout.
    """
    from unittest.mock import Mock

    page = b"<html><head><title>t</title></head><body>hi</body></html>"
    with (
        patch("web.app._serve_container_host_port", return_value="39876"),
        patch("web.app.httpx.Client") as mock_client_cls,
    ):
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.content = page
        mock_resp.headers = {"content-type": "text/html"}
        mock_client = Mock()
        mock_client.request.return_value = mock_resp
        mock_client_cls.return_value.__enter__.return_value = mock_client

        vnc = client.get("/serve/abc123def/vnc.html?autoconnect=true")
        other = client.get("/serve/abc123def/index.html")

    assert vnc.status_code == 200
    assert b"__vncTrace" in vnc.data
    assert b"VideoDecoder" in vnc.data
    assert b"isConfigSupported" in vnc.data
    assert b"supported:false" in vnc.data
    assert b"8000" in vnc.data
    assert b"hi" in vnc.data  # page itself intact
    assert b"VideoDecoder" not in other.data
    assert b"__vncTrace" not in other.data


def test_sandbox_serve_proxy_races_browser_js_tla(client):
    """browser.js top-level await gets a 10s whole-check race; others pass.

    The vendored capability check can wedge silently on a cold phone media
    stack, suspending the viewer module with zero errors. The rewrite keeps
    H.264 for healthy clients (race losers fall back to false) and refuses
    to touch the file unless the anchor matches exactly once.
    """
    from unittest.mock import Mock

    vendored = (
        b"import * as Log from '../util/logging.js';\n"
        b"export const supportsWebCodecsH264Decode = await _checkWebCodecsH264DecodeSupport();\n"
        b"async function _checkWebCodecsH264DecodeSupport(){return true;}\n"
    )

    def _get(subpath, content):
        with (
            patch("web.app._serve_container_host_port", return_value="39876"),
            patch("web.app.httpx.Client") as mock_client_cls,
        ):
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_resp.content = content
            mock_resp.headers = {"content-type": "application/javascript"}
            mock_client = Mock()
            mock_client.request.return_value = mock_resp
            mock_client_cls.return_value.__enter__.return_value = mock_client
            return client.get(f"/serve/abc123def/{subpath}")

    hit = _get("core/util/browser.js", vendored)
    assert hit.status_code == 200
    assert b"Promise.race" in hit.data
    assert b"10000" in hit.data
    assert b"catch(__e){return false;}" in hit.data
    assert b"_checkWebCodecsH264DecodeSupport()" in hit.data  # check itself kept
    assert hit.data.count(b"await _checkWebCodecsH264DecodeSupport()") == 0
    assert hit.headers.get("Cache-Control") == "no-store"

    # Anchor mismatch (vendored code changed shape): file passes untouched.
    other = _get("core/util/browser.js", b"export const x = 1;\n")
    assert other.status_code == 200
    assert b"Promise.race" not in other.data
    assert "Cache-Control" not in other.headers

    # Non-JS and other JS files untouched.
    plain = _get("app/ui.js", b"await _checkWebCodecsH264DecodeSupport();\n")
    assert b"Promise.race" not in plain.data


def test_sandbox_serve_proxy_unknown_container(client):
    with patch("web.app._serve_container_host_port", return_value=None):
        res = client.get("/serve/nope/")
    assert res.status_code == 404
    assert b"Serving container not found" in res.data


def test_sandbox_serve_proxy_upstream_unreachable(client):
    from unittest.mock import Mock

    import httpx

    with (
        patch("web.app._serve_container_host_port", return_value="50002"),
        patch("web.app.httpx.Client") as mock_client_cls,
    ):
        mock_client = Mock()
        mock_client.request.side_effect = httpx.ConnectError("connect failed")
        mock_client_cls.return_value.__enter__.return_value = mock_client

        res = client.get("/serve/cid123/")

    assert res.status_code == 502


def test_sandbox_serve_ws_proxy_unknown_container(client):
    """WS tunnel returns 404 when the container is gone or has no published port."""
    with patch("web.app._serve_container_host_port", return_value=None):
        res = client.get("/serve/ws/nope/websockify")
    assert res.status_code == 404
    assert b"Serving container not found" in res.data


def test_sandbox_serve_ws_proxy_handshake_failure(client):
    """WS tunnel fails with 400 when the browser handshake cannot be accepted."""
    with patch("web.app._serve_container_host_port", return_value="39876"):
        # Send a WebSocket upgrade request. The Flask test client does not
        # provide a real socket (no werkzeug.socket in environ), so
        # simple_websocket.Server raises during the handshake.
        res = client.get(
            "/serve/ws/abc123/websockify",
            headers={
                "Connection": "Upgrade",
                "Upgrade": "websocket",
                "Sec-WebSocket-Version": "13",
                "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ==",
            },
        )
    assert res.status_code == 400
    assert b"WebSocket handshake failed" in res.data


def test_serve_container_host_port_prefers_requested_port():
    """Port lookup pins the requested in-container port, falls back otherwise."""
    from unittest.mock import MagicMock

    from web.app import _serve_container_host_port

    with patch("web.app.docker.DockerClient") as mock_docker:
        mock_container = MagicMock()
        mock_container.ports = {
            "6080/tcp": [{"HostPort": "39781"}],
            "8090/tcp": [{"HostPort": "39782"}],
        }
        mock_docker.return_value.containers.get.return_value = mock_container
        assert _serve_container_host_port("cid123") == "39781"
        assert _serve_container_host_port("cid123", "8090") == "39782"

        # Pre-audio container (video port only): audio lookup falls back.
        mock_container.ports = {"6080/tcp": [{"HostPort": "39781"}]}
        assert _serve_container_host_port("cid123", "8090") == "39781"


def test_sandbox_serve_audio_unknown_container(client):
    with (
        patch("web.app.ensure_audio_encoder") as mock_ensure,
        patch("web.app._serve_container_host_port", return_value=None),
    ):
        res = client.get("/serve/audio/nope")
    assert res.status_code == 404
    assert b"no audio stream" in res.data
    mock_ensure.assert_called_once_with("nope")


def test_sandbox_serve_audio_upstream_unreachable(client):
    from unittest.mock import Mock

    import httpx

    with (
        patch("web.app.ensure_audio_encoder") as mock_ensure,
        patch("web.app._serve_container_host_port", return_value="50003") as mock_port,
        patch("web.app.httpx.Client") as mock_client_cls,
    ):
        mock_client = Mock()
        mock_client.send.side_effect = httpx.ConnectError("connect failed")
        mock_client_cls.return_value = mock_client

        res = client.get("/serve/audio/cid123")

    assert res.status_code == 502
    mock_ensure.assert_called_once_with("cid123")
    mock_port.assert_called_once_with("cid123", "8090")


def test_sandbox_serve_audio_streams_mp3(client):
    """Audio route relays the upstream MP3 bytes with the right content type."""
    from unittest.mock import Mock

    with (
        patch("web.app.ensure_audio_encoder") as mock_ensure,
        patch("web.app._serve_container_host_port", return_value="50003"),
        patch("web.app.httpx.Client") as mock_client_cls,
    ):
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.iter_bytes.return_value = iter([b"ID3\x04chunk1", b"chunk2"])
        mock_client = Mock()
        mock_client.send.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        res = client.get("/serve/audio/cid123")

    assert res.status_code == 200
    assert res.content_type == "audio/mpeg"
    assert res.data == b"ID3\x04chunk1chunk2"
    assert res.headers["X-Accel-Buffering"] == "no"
    mock_resp.iter_bytes.assert_called_once_with()
    mock_resp.close.assert_called_once()
    mock_client.close.assert_called_once()
    mock_ensure.assert_called_once_with("cid123")


def test_sandbox_audio_forwards_frames_without_waiting_for_batch(client):
    import httpx

    received = []

    def frames():
        yield b"frame1"
        assert received == [b"frame1"]
        yield b"frame2"

    upstream = httpx.Response(200, content=frames())
    with (
        patch("web.app.ensure_audio_encoder"),
        patch("web.app._serve_container_host_port", return_value="50003"),
        patch("web.app.httpx.Client") as mock_client_cls,
    ):
        mock_client_cls.return_value.send.return_value = upstream
        res = client.get("/serve/audio/cid123", buffered=False)
        try:
            for chunk in res.response:
                received.append(chunk)
        finally:
            res.close()

    assert received == [b"frame1", b"frame2"]
    assert upstream.is_closed


def test_sandbox_serve_ui_timeout_passthrough(client):
    """Client-requested session lifetime reaches serve_ui (clamped).

    The container's PID1 sleep expires at timeout+60 s and reaps the
    session, so a capped default would kill long play sessions mid-game.
    """
    from unittest.mock import patch

    with patch("web.app.serve_ui", return_value={"container_id": "abc"}) as mock_serve:
        res = client.post("/api/sandbox/serve_ui", json={"code": "print('hi')", "timeout": 7200})
        assert res.status_code == 200
        assert mock_serve.call_args.kwargs["timeout"] == 7200
    with patch("web.app.serve_ui", return_value={"container_id": "abc"}) as mock_serve:
        res = client.post("/api/sandbox/serve_ui", json={"code": "print('hi')"})
        assert res.status_code == 200
        assert mock_serve.call_args.kwargs["timeout"] == 600
    with patch("web.app.serve_ui", return_value={"container_id": "abc"}) as mock_serve:
        res = client.post("/api/sandbox/serve_ui", json={"code": "print('hi')", "timeout": 999999})
        assert res.status_code == 200
        assert mock_serve.call_args.kwargs["timeout"] == 28800


def _write_multistep_model_file(ms_models_dir, model="openrouter:poolside/laguna-s-2.1:free"):
    payload = {
        "model": model,
        "generated_at": "2026-08-25T01:43:47",
        "results": [
            {
                "model": model,
                "timestamp": "2026-08-25T01:43:47",
                "tasks": [
                    {
                        "test_id": "glider_2026_house",
                        "test_category": "multistep_gamedev",
                        "test_label": "Multi-Step Agentic: Glider 2026",
                        "success": "False",
                        "score": "84.2",
                        "latency": "547.0",
                        "tokens_generated": "23501",
                        "response": "<!DOCTYPE html><html><body>game</body></html>",
                        "error": None,
                    }
                ],
            }
        ],
    }
    (ms_models_dir / "multistep_openrouter_poolside_laguna-s-2_1_free.json").write_text(json.dumps(payload))


def test_get_tests_includes_multistep_workflows(client, tmp_path):
    """Test Browser lists multistep workflows with stats, not just file tests."""
    ms_models_dir = tmp_path / "ms_models"
    ms_models_dir.mkdir()
    _write_multistep_model_file(ms_models_dir)
    with patch("web.app.multistep_benchmark.MODELS_DIR", ms_models_dir):
        res = client.get("/api/tests")
    assert res.status_code == 200
    tests_by_id = {t["id"]: t for t in json.loads(res.data.decode("utf-8"))["tests"]}
    assert "glider_2026_house" in tests_by_id
    g = tests_by_id["glider_2026_house"]
    assert g["type"] == "multistep"
    assert g["kind"] == "multistep"
    assert g["models_tested"] == ["openrouter:poolside/laguna-s-2.1:free"]
    assert g["models_scores"]["openrouter:poolside/laguna-s-2.1:free"] == 84.2
    # 84.2 >= 50 counts as passed even though success is the string "False".
    assert g["models_passed_count"] == 1
    assert g["models_failed_count"] == 0
    assert g["last_run"] == "2026-08-25T01:43:47"
    assert g["is_out_of_date"] is False


def test_multistep_success_normalization():
    from web.app import _multistep_success

    assert _multistep_success(True) is True
    assert _multistep_success("True") is True
    assert _multistep_success("False") is False
    assert _multistep_success(False) is False
    assert _multistep_success(None) is False


def test_test_responses_multistep(client, tmp_path):
    """Winning-response replay serves the multistep game HTML per model."""
    ms_models_dir = tmp_path / "ms_models"
    ms_models_dir.mkdir()
    _write_multistep_model_file(ms_models_dir)
    with patch("web.app.multistep_benchmark.MODELS_DIR", ms_models_dir):
        res = client.get("/api/tests/glider_2026_house/responses")
    assert res.status_code == 200
    data = json.loads(res.data.decode("utf-8"))
    assert data["test_id"] == "glider_2026_house"
    assert len(data["responses"]) == 1
    r = data["responses"][0]
    assert r["model"] == "openrouter:poolside/laguna-s-2.1:free"
    assert r["is_html"] is True
    assert "game" in r["response"]


def test_ui_launcher_page_no_store(client):
    """Launcher HTML must not be cached (phones held stale toolbar without keys)."""
    res = client.get("/ui/launcher/does-not-exist")
    assert res.status_code == 404
    assert res.headers.get("Cache-Control", "").startswith("no-store")


def test_ui_inner_status_accepts_beacon(client):
    """Phone-side noVNC states must land in the server log path (200, no crash)."""
    res = client.post(
        "/api/sandbox/ui_inner_status",
        json={"container_id": "abc123", "state": "inner-status", "detail": "Connecting..."},
    )
    assert res.status_code == 200
    assert json.loads(res.data.decode("utf-8")) == {"ok": True}
    res = client.post("/api/sandbox/ui_inner_status", json={})
    assert res.status_code == 200


def test_learning_export_reports_eligibility_and_filters(client, tmp_path):
    from online_providers import build_provenance
    from web.app import benchmark, shared_llm_benchmark

    gen_models = tmp_path / "g_models"
    gen_results = tmp_path / "g_results"
    shared_models = tmp_path / "s_models"
    shared_results = tmp_path / "s_results"
    for d in (gen_models, gen_results, shared_models, shared_results):
        d.mkdir(parents=True, exist_ok=True)

    gen_prov = build_provenance(
        model="qwen_local",
        source="alpaca",
        harness="llm_benchmark_suite",
        transport="proxy",
        run_id="run-g",
    )
    shared_prov = build_provenance(
        model="openrouter:vendor/model",
        source="alpaca",
        harness="shared_llm_benchmark",
        transport="api",
        run_id="run-s",
        learning_policy="shared_llm_success_only",
    )

    (gen_models / "general_qwen_local.json").write_text(
        json.dumps(
            {
                "model": "qwen_local",
                "run_id": "run-g",
                "provenance": gen_prov,
                "results": [
                    {
                        "model": "qwen_local",
                        "run_id": "run-g",
                        "provenance": gen_prov,
                        "category_coding": {
                            "tests": [
                                {
                                    "test_id": "code_1",
                                    "test_category": "coding",
                                    "model": "qwen_local",
                                    "success": True,
                                    "response": "print(1)",
                                    "run_id": "run-g",
                                    "provenance": gen_prov,
                                }
                            ]
                        },
                    }
                ],
            }
        )
    )
    (shared_models / "shared_openrouter_vendor_model.json").write_text(
        json.dumps(
            {
                "model": "openrouter:vendor/model",
                "run_id": "run-s",
                "provenance": shared_prov,
                "results": [
                    {
                        "model": "openrouter:vendor/model",
                        "run_id": "run-s",
                        "provenance": shared_prov,
                        "tasks": [
                            {
                                "test_id": "fast_path_light",
                                "test_category": "fast_path",
                                "model": "openrouter:vendor/model",
                                "success": True,
                                "response": "light_on",
                                "run_id": "run-s",
                                "provenance": shared_prov,
                                "training_eligible": True,
                                "exclusion_reason": None,
                            }
                        ],
                    }
                ],
            }
        )
    )

    with (
        patch.object(benchmark, "MODELS_DIR", gen_models),
        patch.object(benchmark, "RESULTS_DIR", gen_results),
        patch.object(shared_llm_benchmark, "MODELS_DIR", shared_models),
        patch.object(shared_llm_benchmark, "RESULTS_DIR", shared_results),
    ):
        res = client.get("/api/benchmarks/export?format=learning")
        assert res.status_code == 200
        data = json.loads(res.data.decode("utf-8"))
        assert data["record_count"] == 2
        assert data["eligible_count"] == 1
        assert data["exclusion_reasons"]["eligible"] == 1
        assert data["exclusion_reasons"]["policy_excludes_source"] == 1

        res = client.get("/api/benchmarks/export?format=learning&eligible_only=1")
        data = json.loads(res.data.decode("utf-8"))
        assert data["record_count"] == 1
        assert data["records"][0]["test_id"] == "fast_path_light"
        assert data["records"][0]["training_eligible"] is True

        res = client.get("/api/benchmarks/export?format=learning&harness=shared_llm_benchmark")
        data = json.loads(res.data.decode("utf-8"))
        assert data["record_count"] == 1
        assert data["records"][0]["test_id"] == "fast_path_light"

        res = client.get("/api/benchmarks/export?format=learning&model=qwen_local")
        data = json.loads(res.data.decode("utf-8"))
        assert data["record_count"] == 1
        assert data["records"][0]["test_id"] == "code_1"
