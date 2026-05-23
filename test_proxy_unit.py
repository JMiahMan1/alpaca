import importlib.util
import json
import os
import pathlib
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException, Request

with tempfile.TemporaryDirectory() as tmpdir:
    os.environ["GRAMMAR_REGISTRY_DIR"] = os.path.join(tmpdir, "grammars")
    os.environ["SCHEMA_REGISTRY_DIR"] = os.path.join(tmpdir, "schemas")
    MODULE_PATH = pathlib.Path(__file__).with_name("alpaca-proxy.py")
    SPEC = importlib.util.spec_from_file_location("alpaca_proxy", MODULE_PATH)
    alpaca_proxy = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(alpaca_proxy)
REAL_POST_ROUTER_MODEL_ACTION = alpaca_proxy.post_router_model_action
REAL_RESOLVE_ROUTER_MODEL = alpaca_proxy.resolve_router_model
REAL_ENSURE_MODEL = alpaca_proxy.ensure_model
REAL_FETCH_ROUTER_MODELS = alpaca_proxy.fetch_router_models

def make_manifest(digest="sha256:abcd", size=4):
    return {
        "schemaVersion": 2,
        "config": {"digest": "sha256:cfg", "size": 2},
        "layers": [
            {
                "mediaType": "application/vnd.ollama.image.model",
                "digest": digest,
                "size": size,
            }
        ],
    }


class MockResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class MockHTTPClient:
    def __init__(self, response_payload):
        self.response_payload = response_payload
        self.calls = []

    async def post(self, url, json):
        self.calls.append({"method": "POST", "url": url, "json": json})
        return MockResponse(self.response_payload)

    async def aclose(self):
        return None


def make_request(path, payload):
    body = json.dumps(payload).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
    )


def test_parse_keep_alive_variants():
    assert alpaca_proxy.parse_keep_alive("5m") == 300
    assert alpaca_proxy.parse_keep_alive(60) == 60
    assert alpaca_proxy.parse_keep_alive("-1") == -1
    assert alpaca_proxy.parse_keep_alive("0") == 0
    assert alpaca_proxy.parse_keep_alive("250ms") == 0.25


def test_router_model_candidates_include_blob_and_public_names():
    manifest = make_manifest(digest="sha256:deadbeef", size=1)
    candidates = alpaca_proxy.router_model_candidates("tinyllama:latest", manifest)
    assert "/router-models/tinyllama--latest.gguf" in candidates
    assert "tinyllama--latest.gguf" in candidates
    assert "tinyllama--latest" in candidates
    assert "sha256-deadbeef" in candidates
    assert "sha256:deadbeef" in candidates
    assert "deadbeef" in candidates
    assert "tinyllama" in candidates
    assert "tinyllama:latest" in candidates


def test_router_entry_matches_path_and_id():
    entry = {
        "id": "sha256-deadbeef",
        "path": "/models/blobs/sha256-deadbeef",
        "status": {"value": "loaded"},
    }
    assert alpaca_proxy.router_entry_matches(entry, ["tinyllama", "sha256-deadbeef"])
    assert not alpaca_proxy.router_entry_matches(entry, ["other-model"])


def test_load_local_manifest_rejects_incomplete_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = pathlib.Path(tmpdir)
        alpaca_proxy.OLLAMA_BASE = str(base)

        manifest_path = base / "manifests" / "registry.ollama.ai" / "library" / "tinyllama" / "latest"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text('{"layers":[{"digest":"sha256:deadbeef","size":4}],"config":{"digest":"sha256:cfg","size":2}}')

        found_path, manifest = alpaca_proxy.load_local_manifest("tinyllama", require_complete=True)
        assert str(manifest_path) == found_path
        assert manifest is None


@pytest.mark.asyncio
async def test_apply_keep_alive_policy_zero_unloads_immediately():
    alpaca_proxy.model_expires_at.clear()
    alpaca_proxy.model_unload_tasks.clear()
    alpaca_proxy.unload_model = AsyncMock()

    await alpaca_proxy.apply_keep_alive_policy("tinyllama", 0)

    alpaca_proxy.unload_model.assert_awaited_once_with("tinyllama:latest")
    assert "tinyllama" in alpaca_proxy.model_expires_at


@pytest.mark.asyncio
async def test_apply_keep_alive_policy_negative_keeps_forever():
    alpaca_proxy.model_expires_at.clear()
    alpaca_proxy.model_unload_tasks.clear()
    alpaca_proxy.unload_model = AsyncMock()

    await alpaca_proxy.apply_keep_alive_policy("tinyllama", -1)

    alpaca_proxy.unload_model.assert_not_called()
    assert alpaca_proxy.model_expires_at["tinyllama"] == alpaca_proxy.FOREVER_EXPIRES_AT


def test_begin_model_request_cancels_pending_unload_and_clears_expiry():
    class DummyTask:
        def __init__(self):
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

    task = DummyTask()
    alpaca_proxy.model_unload_tasks.clear()
    alpaca_proxy.model_expires_at.clear()
    alpaca_proxy.model_unload_tasks["tinyllama"] = task
    alpaca_proxy.model_expires_at["tinyllama"] = "2026-05-13T00:00:00Z"

    alpaca_proxy.begin_model_request("tinyllama")

    assert task.cancelled is True
    assert "tinyllama" not in alpaca_proxy.model_unload_tasks
    assert alpaca_proxy.model_expires_at["tinyllama"] == "0001-01-01T00:00:00Z"


@pytest.mark.asyncio
async def test_ensure_model_unloads_other_loaded_model_before_load():
    alpaca_proxy.fetch_router_models = AsyncMock(side_effect=[
        [
            {"id": "other-model", "status": {"value": "loaded"}},
            {"id": "sha256-deadbeef", "status": {"value": "unloaded"}},
        ],
        [
            {"id": "other-model", "status": {"value": "unloaded"}},
            {"id": "sha256-deadbeef", "status": {"value": "loaded"}},
        ],
    ])
    alpaca_proxy.resolve_router_model = AsyncMock(return_value={
        "model_name": "tinyllama:latest",
        "backend_model": "sha256-deadbeef",
        "entry": {"id": "sha256-deadbeef", "status": {"value": "unloaded"}},
        "manifest_path": "/tmp/manifest",
        "manifest": make_manifest(digest="sha256:deadbeef"),
        "router_models": [
            {"id": "other-model", "status": {"value": "loaded"}},
            {"id": "sha256-deadbeef", "status": {"value": "unloaded"}},
        ],
    })
    alpaca_proxy.post_router_model_action = AsyncMock()

    # Mock client_httpx for OOM check after successful load
    props_resp = AsyncMock()
    props_resp.status_code = 200
    props_resp.json = MagicMock(return_value={"n_gpu_layers": -1})
    alpaca_proxy.client_httpx = AsyncMock()
    alpaca_proxy.client_httpx.get = AsyncMock(return_value=props_resp)

    resolved = await alpaca_proxy.ensure_model("tinyllama")

    assert resolved["backend_model"] == "sha256-deadbeef"
    alpaca_proxy.post_router_model_action.assert_any_await("unload", "other-model")
    alpaca_proxy.post_router_model_action.assert_any_await("load", {
        "model": "sha256-deadbeef",
        "n_gpu_layers": -1,
        "use_mmap": True,
        "flash_attn": True,
    })


@pytest.mark.asyncio
async def test_post_router_model_action_marks_management_unsupported_when_router_endpoint_missing():
    response = httpx.Response(404, request=httpx.Request("POST", "http://llama-server:8080/models/load"))

    alpaca_proxy.post_router_model_action = REAL_POST_ROUTER_MODEL_ACTION
    alpaca_proxy.router_management_supported = None
    alpaca_proxy.client_httpx = AsyncMock()
    alpaca_proxy.client_httpx.post = AsyncMock(return_value=response)

    with pytest.raises(alpaca_proxy.RouterManagementUnsupported):
        await alpaca_proxy.post_router_model_action("load", "sha256-deadbeef")

    assert alpaca_proxy.router_management_supported is False


@pytest.mark.asyncio
async def test_ensure_model_falls_back_to_router_autoload_when_load_endpoint_missing():
    alpaca_proxy.post_router_model_action = AsyncMock(
        side_effect=alpaca_proxy.RouterManagementUnsupported("http://llama-server:8080/models/load")
    )
    alpaca_proxy.resolve_router_model = AsyncMock(return_value={
        "model_name": "tinyllama:latest",
        "backend_model": "sha256-deadbeef",
        "entry": {"id": "sha256-deadbeef", "status": {"value": "unloaded"}},
        "manifest_path": "/tmp/manifest",
        "manifest": make_manifest(digest="sha256:deadbeef"),
        "router_models": [
            {"id": "sha256-deadbeef", "status": {"value": "unloaded"}},
        ],
    })
    resolved = await alpaca_proxy.ensure_model("tinyllama")

    assert resolved["backend_model"] == "sha256-deadbeef"
    alpaca_proxy.post_router_model_action.assert_awaited_once_with("load", {
        "model": "sha256-deadbeef",
        "n_gpu_layers": -1,
        "use_mmap": True,
        "flash_attn": True,
    })


@pytest.mark.asyncio
async def test_resolve_router_model_falls_back_to_router_alias_when_symlink_exists(tmp_path):
    alpaca_proxy.resolve_router_model = REAL_RESOLVE_ROUTER_MODEL
    alpaca_proxy.OLLAMA_BASE = str(tmp_path / "models")
    alpaca_proxy.ROUTER_MODELS_DIR = str(tmp_path / "router-models")
    manifest_path = tmp_path / "models" / "manifests" / "registry.ollama.ai" / "library" / "tinyllama" / "latest"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(make_manifest(digest="sha256:deadbeef", size=4)))
    blob_path = tmp_path / "models" / "blobs" / "sha256-deadbeef"
    blob_path.parent.mkdir(parents=True)
    blob_path.write_bytes(b"gguf")
    config_path = tmp_path / "models" / "blobs" / "sha256-cfg"
    config_path.write_bytes(b"{}")
    router_path = tmp_path / "router-models" / "tinyllama--latest.gguf"
    router_path.parent.mkdir(parents=True)
    router_path.write_text("stub")
    alpaca_proxy.fetch_router_models = AsyncMock(return_value=[])

    resolved = await alpaca_proxy.resolve_router_model("tinyllama", reload=False)

    assert resolved["backend_model"] == "tinyllama--latest.gguf"
    assert resolved["entry"]["path"] == str(router_path)


@pytest.mark.asyncio
async def test_unload_model_ignores_router_400_model_not_found():
    response = httpx.Response(
        400,
        request=httpx.Request("POST", "http://llama-server:8080/models/unload"),
        json={"error": {"message": "model is not found"}},
    )
    alpaca_proxy.resolve_router_model = AsyncMock(return_value={
        "backend_model": "tinyllama--latest",
    })
    alpaca_proxy.post_router_model_action = AsyncMock(
        side_effect=httpx.HTTPStatusError("bad request", request=response.request, response=response)
    )

    await alpaca_proxy.unload_model("tinyllama")


@pytest.mark.asyncio
async def test_unload_model_ignores_router_400_when_backend_is_already_not_resident():
    response = httpx.Response(
        400,
        request=httpx.Request("POST", "http://llama-server:8080/models/unload"),
        json={"error": {"message": "cannot unload model"}},
    )
    alpaca_proxy.resolve_router_model = AsyncMock(return_value={
        "backend_model": "tinyllama--latest",
    })
    alpaca_proxy.post_router_model_action = AsyncMock(
        side_effect=httpx.HTTPStatusError("bad request", request=response.request, response=response)
    )
    alpaca_proxy.fetch_router_models = AsyncMock(return_value=[
        {"id": "tinyllama--latest", "status": {"value": "unloaded"}},
    ])

    await alpaca_proxy.unload_model("tinyllama")


@pytest.mark.asyncio
async def test_loaded_models_from_router_returns_only_loaded_models():
    alpaca_proxy.fetch_router_models = AsyncMock(return_value=[
        {"id": "sha256-deadbeef", "path": "/models/blobs/sha256-deadbeef", "status": {"value": "loaded"}},
        {"id": "sha256-other", "path": "/models/blobs/sha256-other", "status": {"value": "unloaded"}},
    ])
    alpaca_proxy.iter_local_manifests = lambda: iter([
        ("base", "/tmp/tinyllama", make_manifest(digest="sha256:deadbeef")),
        ("base", "/tmp/other", make_manifest(digest="sha256:other")),
    ])
    alpaca_proxy.manifest_model_name = lambda base, path: "tinyllama" if "tinyllama" in path else "other"
    alpaca_proxy.manifest_stats = lambda path, manifest: {"size": 1, "digest": "d", "details": {}, "context_length": 4096}

    loaded = await alpaca_proxy.loaded_models_from_router()

    assert [item["name"] for item in loaded] == ["tinyllama"]


@pytest.mark.asyncio
async def test_chat_endpoint_maps_request_and_returns_ollama_shape():
    alpaca_proxy.ensure_model = AsyncMock(return_value={"backend_model": "router-backend"})
    alpaca_proxy.apply_keep_alive_policy = AsyncMock()
    mock_http = MockHTTPClient({
        "choices": [
            {
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    })
    alpaca_proxy.client_httpx = mock_http

    response = await alpaca_proxy.chat(make_request("/api/chat", {
        "model": "tinyllama",
        "messages": [{"role": "user", "content": "hi"}],
        "format": "json",
        "options": {"num_predict": 12},
        "stream": False,
    }))

    assert response.status_code == 200
    body = json.loads(response.body)
    assert body["model"] == "tinyllama"
    assert body["done"] is True
    assert body["message"]["content"] == "hello"
    assert mock_http.calls[0]["url"].endswith("/v1/chat/completions")
    assert mock_http.calls[0]["json"]["model"] == "router-backend"
    assert mock_http.calls[0]["json"]["n_predict"] == 12
    assert mock_http.calls[0]["json"]["response_format"] == {"type": "json_object"}
    alpaca_proxy.apply_keep_alive_policy.assert_awaited_once_with("tinyllama", "5m")


@pytest.mark.asyncio
async def test_chat_endpoint_returns_504_on_upstream_timeout():
    alpaca_proxy.ensure_model = AsyncMock(return_value={"backend_model": "router-backend"})
    alpaca_proxy.client_httpx = AsyncMock()
    alpaca_proxy.client_httpx.post = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))

    with pytest.raises(HTTPException) as excinfo:
        await alpaca_proxy.chat(make_request("/api/chat", {
            "model": "tinyllama",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }))

    assert excinfo.value.status_code == 504
    assert "timed out" in excinfo.value.detail.lower()


@pytest.mark.asyncio
async def test_generate_endpoint_uses_chat_backend_for_system_and_respects_keep_alive_zero():
    alpaca_proxy.ensure_model = AsyncMock(return_value={"backend_model": "router-backend"})
    alpaca_proxy.apply_keep_alive_policy = AsyncMock()
    mock_http = MockHTTPClient({
        "choices": [
            {
                "message": {"role": "assistant", "content": "done", "thinking": "trace"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    })
    alpaca_proxy.client_httpx = mock_http

    response = await alpaca_proxy.generate(make_request("/api/generate", {
        "model": "tinyllama",
        "prompt": "hi",
        "system": "be terse",
        "think": True,
        "keep_alive": 0,
        "stream": False,
    }))

    assert response.status_code == 200
    body = json.loads(response.body)
    assert body["model"] == "tinyllama"
    assert body["response"] == "done"
    assert body["thinking"] == "trace"
    assert mock_http.calls[0]["url"].endswith("/v1/chat/completions")
    assert mock_http.calls[0]["json"]["model"] == "router-backend"
    assert mock_http.calls[0]["json"]["thinking"] is True
    alpaca_proxy.apply_keep_alive_policy.assert_awaited_once_with("tinyllama", 0)


@pytest.mark.asyncio
async def test_generate_endpoint_returns_504_on_upstream_timeout():
    alpaca_proxy.ensure_model = AsyncMock(return_value={"backend_model": "router-backend"})
    alpaca_proxy.client_httpx = AsyncMock()
    alpaca_proxy.client_httpx.post = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))

    with pytest.raises(HTTPException) as excinfo:
        await alpaca_proxy.generate(make_request("/api/generate", {
            "model": "tinyllama",
            "prompt": "hi",
            "stream": False,
        }))

    assert excinfo.value.status_code == 504
    assert "timed out" in excinfo.value.detail.lower()


@pytest.mark.asyncio
async def test_ps_endpoint_returns_loaded_router_models():
    alpaca_proxy.model_expires_at.clear()
    alpaca_proxy.model_expires_at["tinyllama"] = "2026-05-11T00:00:00Z"
    alpaca_proxy.loaded_models_from_router = AsyncMock(return_value=[
        {
            "name": "tinyllama",
            "info": {
                "size": 1,
                "digest": "deadbeef",
                "details": {"format": "gguf"},
                "context_length": 4096,
            },
        }
    ])

    body = await alpaca_proxy.ps()
    assert body["models"][0]["name"] == "tinyllama"
    assert body["models"][0]["expires_at"] == "2026-05-11T00:00:00Z"
    assert body["models"][0]["context_length"] == 4096


@pytest.mark.asyncio
async def test_ensure_model_escalates_to_safe_settings_when_load_fails_completely():
    alpaca_proxy.ensure_model = REAL_ENSURE_MODEL
    alpaca_proxy.MTP_INCOMPATIBLE_MODELS.clear()
    alpaca_proxy.SAFE_SETTINGS_MODELS.clear()
    
    alpaca_proxy.wait_for_llama_server_or_restart = AsyncMock(return_value=True)
    alpaca_proxy.resolve_router_model = AsyncMock(return_value={
        "model_name": "qwen3.5:9b",
        "backend_model": "qwen3.5--9b.gguf",
        "entry": {"id": "qwen3.5--9b.gguf", "status": {"value": "unloaded"}},
        "manifest_path": "/tmp/manifest",
        "manifest": make_manifest(digest="sha256:qwen35"),
        "router_models": [
            {"id": "qwen3.5--9b.gguf", "status": {"value": "unloaded"}},
        ],
    })
    
    alpaca_proxy.post_router_model_action = AsyncMock(side_effect=[
        httpx.RequestError("Default load crashed"),
        httpx.RequestError("Retry without MTP crashed"),
        None,
    ])

    resolved = await alpaca_proxy.ensure_model("qwen3.5:9b")
    
    assert resolved["backend_model"] == "qwen3.5--9b.gguf"
    assert "qwen3.5--9b.gguf" in alpaca_proxy.MTP_INCOMPATIBLE_MODELS
    assert "qwen3.5--9b.gguf" in alpaca_proxy.SAFE_SETTINGS_MODELS

    # Verify calls
    assert alpaca_proxy.post_router_model_action.await_count == 3
    
    # 1st call: default optimization
    first_call = alpaca_proxy.post_router_model_action.await_args_list[0]
    assert first_call[0][0] == "load"
    assert first_call[0][1]["model"] == "qwen3.5--9b.gguf"
    assert first_call[0][1].get("spec_type") is None
    assert first_call[0][1]["flash_attn"] is True

    # 2nd call: disabled MTP
    second_call = alpaca_proxy.post_router_model_action.await_args_list[1]
    assert second_call[0][0] == "load"
    assert second_call[0][1]["spec_type"] == "none"
    assert second_call[0][1]["flash_attn"] is True

    # 3rd call: Safe Settings (flash_attn=False, n_ctx=8192)
    third_call = alpaca_proxy.post_router_model_action.await_args_list[2]
    assert third_call[0][0] == "load"
    assert third_call[0][1]["spec_type"] == "none"
    assert third_call[0][1]["flash_attn"] is False
    assert third_call[0][1]["n_ctx"] == 8192


@pytest.mark.asyncio
async def test_ensure_model_uses_saved_safe_settings_immediately():
    alpaca_proxy.ensure_model = REAL_ENSURE_MODEL
    alpaca_proxy.MTP_INCOMPATIBLE_MODELS.clear()
    alpaca_proxy.SAFE_SETTINGS_MODELS.clear()
    
    alpaca_proxy.MTP_INCOMPATIBLE_MODELS.add("qwen3.5--9b.gguf")
    alpaca_proxy.SAFE_SETTINGS_MODELS.add("qwen3.5--9b.gguf")
    
    alpaca_proxy.resolve_router_model = AsyncMock(return_value={
        "model_name": "qwen3.5:9b",
        "backend_model": "qwen3.5--9b.gguf",
        "entry": {"id": "qwen3.5--9b.gguf", "status": {"value": "unloaded"}},
        "manifest_path": "/tmp/manifest",
        "manifest": make_manifest(digest="sha256:qwen35"),
        "router_models": [
            {"id": "qwen3.5--9b.gguf", "status": {"value": "unloaded"}},
        ],
    })
    
    alpaca_proxy.post_router_model_action = AsyncMock()

    resolved = await alpaca_proxy.ensure_model("qwen3.5:9b")
    
    assert resolved["backend_model"] == "qwen3.5--9b.gguf"
    
    # Must immediately load with MTP off and Safe Settings applied
    alpaca_proxy.post_router_model_action.assert_awaited_once_with("load", {
        "model": "qwen3.5--9b.gguf",
        "n_gpu_layers": -1,
        "use_mmap": True,
        "flash_attn": False,
        "n_ctx": 8192,
        "spec_type": "none",
        "spec_draft_n_max": 0,
    })


@pytest.mark.asyncio
async def test_admin_system_endpoint_returns_metrics():
    # Mock psutil
    mock_psutil = MagicMock()
    mock_psutil.virtual_memory.return_value = MagicMock(
        total=32000000000,
        available=16000000000,
        used=16000000000,
        percent=50.0
    )
    mock_psutil.cpu_percent.return_value = 25.0
    
    # Mock subprocess.check_output
    mock_subprocess = MagicMock()
    mock_subprocess.check_output.return_value = "NVIDIA GeForce RTX 4060, 8188, 1241, 6557\n"
    
    with patch.dict("sys.modules", {"psutil": mock_psutil}), patch("subprocess.check_output", mock_subprocess.check_output):
        response = await alpaca_proxy.admin_system()
        assert response.status_code == 200
        body = json.loads(response.body)
        assert body["ram_usage"]["total_gb"] == 32.0
        assert body["ram_usage"]["used_pct"] == 50.0
        assert body["cpu_usage"]["percent"] == 25.0
        assert body["gpu_info"][0]["name"] == "NVIDIA GeForce RTX 4060"
        assert body["gpu_info"][0]["total_mb"] == 8188
        assert body["gpu_info"][0]["used_pct"] == 15.2


@pytest.mark.asyncio
async def test_get_llama_server_logs_endpoint():
    mock_subprocess = MagicMock()
    mock_subprocess.check_output.return_value = "line 1\nline 2\n"
    with patch("subprocess.check_output", mock_subprocess.check_output):
        response = await alpaca_proxy.get_llama_server_logs(limit=2)
        assert response["logs"] == ["line 1", "line 2", ""]


@pytest.mark.asyncio
async def test_ensure_model_recovers_from_oom_by_restarting_server():
    alpaca_proxy.ensure_model = REAL_ENSURE_MODEL
    alpaca_proxy.resolve_router_model = AsyncMock(return_value={
        "model_name": "tinyllama:latest",
        "backend_model": "sha256-deadbeef",
        "entry": {"id": "sha256-deadbeef", "status": {"value": "unloaded"}},
        "manifest_path": "/tmp/manifest",
        "manifest": make_manifest(digest="sha256:deadbeef"),
        "router_models": [
            {"id": "sha256-deadbeef", "status": {"value": "unloaded"}},
        ],
    })
    alpaca_proxy.post_router_model_action = AsyncMock()
    alpaca_proxy.restart_llama_server = AsyncMock(return_value=True)
    alpaca_proxy.wait_for_llama_server_or_restart = AsyncMock(return_value=True)

    # Mock get_child_model_props so check returns n_gpu_layers=0 (OOM!)
    alpaca_proxy.get_child_model_props = AsyncMock(return_value={"n_gpu_layers": 0})

    resolved = await alpaca_proxy.ensure_model("tinyllama")

    assert resolved["backend_model"] == "sha256-deadbeef"
    # Verify that we called restart_llama_server and wait_for_llama_server_or_restart
    alpaca_proxy.restart_llama_server.assert_awaited_once()
    alpaca_proxy.wait_for_llama_server_or_restart.assert_awaited_once()
    # Verify we retried loading with OOM safe settings (n_gpu_layers=0)
    alpaca_proxy.post_router_model_action.assert_any_await("load", {
        "model": "sha256-deadbeef",
        "n_gpu_layers": 0,
        "n_thread": 8,
        "n_batch": 256,
        "n_ubatch": 512,
        "use_mmap": True,
        "flash_attn": False,
        "spec_type": "none",
        "spec_draft_n_max": 0,
    })


@pytest.mark.asyncio
async def test_fetch_router_models_retries_on_connection_error():
    alpaca_proxy.fetch_router_models = REAL_FETCH_ROUTER_MODELS
    alpaca_proxy.wait_for_llama_server = AsyncMock(return_value=True)

    resp_mock = AsyncMock()
    resp_mock.status_code = 200
    resp_mock.json = MagicMock(return_value={"data": [{"id": "model1"}]})
    resp_mock.raise_for_status = MagicMock()

    alpaca_proxy.client_httpx = AsyncMock()
    # First call raises ConnectError, second call returns resp_mock
    alpaca_proxy.client_httpx.get = AsyncMock(side_effect=[
        httpx.ConnectError("Could not establish connection"),
        resp_mock,
    ])

    try:
        models = await alpaca_proxy.fetch_router_models(reload=True)
        assert models == [{"id": "model1"}]
        alpaca_proxy.wait_for_llama_server.assert_awaited_once_with()
        assert alpaca_proxy.client_httpx.get.call_count == 2
    finally:
        pass


@pytest.mark.asyncio
async def test_post_router_model_action_retries_on_connection_error():
    alpaca_proxy.post_router_model_action = REAL_POST_ROUTER_MODEL_ACTION
    alpaca_proxy.wait_for_llama_server = AsyncMock(return_value=True)

    resp_mock = AsyncMock()
    resp_mock.status_code = 200
    resp_mock.json = MagicMock(return_value={"status": "loaded"})
    resp_mock.raise_for_status = MagicMock()

    alpaca_proxy.client_httpx = AsyncMock()
    # First call raises ConnectError, second call returns resp_mock
    alpaca_proxy.client_httpx.post = AsyncMock(side_effect=[
        httpx.ConnectError("Could not establish connection"),
        resp_mock,
    ])

    try:
        res = await alpaca_proxy.post_router_model_action("load", "some-model")
        assert res == {"status": "loaded"}
        alpaca_proxy.wait_for_llama_server.assert_awaited_once_with()
        assert alpaca_proxy.client_httpx.post.call_count == 2
    finally:
        pass


def test_is_model_over_9b():
    # 1. Names with size tags
    assert alpaca_proxy.is_model_over_9b("qwen3.6-35b-a3b:q4_k_m") is True
    assert alpaca_proxy.is_model_over_9b("qwen3:8b") is False
    assert alpaca_proxy.is_model_over_9b("llama3:70b") is True
    assert alpaca_proxy.is_model_over_9b("deepseek-coder:1.3b") is False

    # 2. Manifest file size checks (8.5 GB threshold)
    small_manifest = {
        "layers": [
            {
                "mediaType": "application/vnd.ollama.image.model",
                "size": 5 * 1024 * 1024 * 1024,
            }
        ]
    }
    large_manifest = {
        "layers": [
            {
                "mediaType": "application/vnd.ollama.image.model",
                "size": 10 * 1024 * 1024 * 1024,
            }
        ]
    }
    
    assert alpaca_proxy.is_model_over_9b(
        "unknown_model", small_manifest
    ) is False
    assert alpaca_proxy.is_model_over_9b(
        "unknown_model", large_manifest
    ) is True


@pytest.mark.asyncio
async def test_ensure_model_skips_escalation_for_large_models():
    alpaca_proxy.ensure_model = REAL_ENSURE_MODEL
    alpaca_proxy.resolve_router_model = AsyncMock(return_value={
        "model_name": "qwen3.6-35b-a3b:q4_k_m",
        "backend_model": "sha256-large",
        "entry": {"id": "sha256-large", "status": {"value": "unloaded"}},
        "manifest_path": "/tmp/manifest",
        "manifest": make_manifest(digest="sha256:large"),
        "router_models": [],
    })
    alpaca_proxy.post_router_model_action = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "Load failed", request=MagicMock(), response=MagicMock()
        )
    )
    alpaca_proxy.restart_llama_server = AsyncMock(return_value=True)
    alpaca_proxy.wait_for_llama_server_or_restart = AsyncMock(
        return_value=True
    )

    # Loading a >9B model should immediately restart and retry once
    with pytest.raises(httpx.HTTPStatusError):
        await alpaca_proxy.ensure_model("qwen3.6-35b-a3b:q4_k_m")
    
    # Assert we called post_router_model_action twice
    assert alpaca_proxy.post_router_model_action.call_count == 2
    alpaca_proxy.restart_llama_server.assert_awaited_once()
    alpaca_proxy.wait_for_llama_server_or_restart.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_model_recovers_from_crash_for_large_models():
    alpaca_proxy.ensure_model = REAL_ENSURE_MODEL
    alpaca_proxy.resolve_router_model = AsyncMock(return_value={
        "model_name": "qwen3.6-35b-a3b:q4_k_m",
        "backend_model": "sha256-large",
        "entry": {"id": "sha256-large", "status": {"value": "unloaded"}},
        "manifest_path": "/tmp/manifest",
        "manifest": make_manifest(digest="sha256:large"),
        "router_models": [],
    })
    alpaca_proxy.post_router_model_action = AsyncMock()
    alpaca_proxy.restart_llama_server = AsyncMock(return_value=True)
    alpaca_proxy.wait_for_llama_server_or_restart = AsyncMock(
        return_value=True
    )

    # Health check returns False (crash), then True on recovery reload
    alpaca_proxy.is_child_model_healthy = AsyncMock(side_effect=[False, True])

    resolved = await alpaca_proxy.ensure_model("qwen3.6-35b-a3b:q4_k_m")
    
    assert resolved["backend_model"] == "sha256-large"
    alpaca_proxy.restart_llama_server.assert_awaited_once()
    alpaca_proxy.wait_for_llama_server_or_restart.assert_awaited_once()
    # Verify it retried with original load parameters
    alpaca_proxy.post_router_model_action.assert_any_await(
        "load", {"model": "sha256-large"}
    )



