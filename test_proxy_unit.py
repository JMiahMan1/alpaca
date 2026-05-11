import importlib.util
import json
import pathlib
import tempfile
from unittest.mock import AsyncMock

import pytest
from fastapi import Request

MODULE_PATH = pathlib.Path(__file__).with_name("alpaca-proxy.py")
SPEC = importlib.util.spec_from_file_location("alpaca_proxy", MODULE_PATH)
alpaca_proxy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(alpaca_proxy)


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

    resolved = await alpaca_proxy.ensure_model("tinyllama")

    assert resolved["backend_model"] == "sha256-deadbeef"
    alpaca_proxy.post_router_model_action.assert_any_await("unload", "other-model")
    alpaca_proxy.post_router_model_action.assert_any_await("load", "sha256-deadbeef")


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
