import json
from unittest.mock import MagicMock, patch

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

    # Try deleting to-delete (should succeed)
    res = client.post("/api/profiles/delete", json={"section": "to-delete"})
    assert res.status_code == 200
    data = json.loads(res.data.decode("utf-8"))
    assert "success" in data["status"]

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
