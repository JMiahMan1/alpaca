import json
from unittest.mock import MagicMock, mock_open, patch

import pytest

from web.app import active_run, active_run_lock, app


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
            active_run["start_time"] = None
            active_run["saved_as"] = None
        yield client


def test_index_route(client):
    """Test that the index route serves the template and returns 200 OK"""
    res = client.get("/")
    assert res.status_code == 200
    assert b"Alpaca Benchmarks v2" in res.data
    assert b"Pipeline Controls" in res.data


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
    with open(profile_json_path, "r") as pf:
        pf_data = json.load(pf)
    assert pf_data["ctx-size"] == 4096
    assert pf_data["flash-attn"] == "on"

    import configparser

    config = configparser.ConfigParser(delimiters=("=",))
    config.read(str(mock_ini))
    assert config.has_section("new-model")
    assert config["new-model"]["ctx-size"] == "4096"
    assert config["new-model"]["flash-attn"] == "on"


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
    assert data["tags"][0] == "latest"
    assert data["tags"][1] == "8b"


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

    mock_open.return_value.__enter__.return_value.read.return_value = json_mod.dumps(
        {"epoch_time": 1}
    )
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
