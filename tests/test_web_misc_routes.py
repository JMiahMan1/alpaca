"""Coverage for the remaining untested /api/* routes.

Grouped by surface:

* Request Monitor mutations  - /api/requests/{clear,cancel,resubmit/<id>}
* Telemetry tuning           - /api/telemetry/recommendations{,/apply}
* Capability routing         - /api/routing/{matrix,optimal}
* Model lifecycle            - /api/models/{switch,unload,text,vision,online},
                               /api/vram/clear, /api/usage
* Test Browser reads         - /api/tests/shared_llm, /api/tests/<id>/{download,
                               responses,attachment/<name>}
* Human ratings              - /api/ratings (GET + POST)
* Thermal watchdog config    - /api/thermal/watchdog (GET + POST)
* Pull control               - /api/models/pulls/<id>/{stop,cancel}
* OCR                        - /api/vision/ocr

The recurring hazard across all of them is the same one the audio and SD
suites already hit: a route that reports success without the upstream
having answered, or that silently rewrites the caller's body on the way
out. Every test here asserts the body or the status that actually reaches
the caller.
"""

import io
import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, Mock, patch

import pytest
from PIL import Image

from web.app import PROXY_URL, app, benchmark, multistep_benchmark

PROXY = "http://proxy.test:11434"


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# --------------------------------------------------------------------------
# Shared httpx mock plumbing (same shape as test_web_audio_routes.py)
# --------------------------------------------------------------------------


class _Rec:
    def __init__(self):
        self.timeout = None


class _Ctx:
    def __init__(self, http):
        self._http = http

    def __enter__(self):
        return self._http

    def __exit__(self, *exc):
        return False


def _upstream(payload, status_code=200):
    resp = Mock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.text = json.dumps(payload)
    http = Mock()
    for verb in ("get", "post", "patch", "delete", "put"):
        getattr(http, verb).return_value = resp
    return http


def _resp_only(payload, status_code=200):
    """A bare response Mock, for use as a side_effect *return* value.

    _upstream() hands back the client; when you need the response itself
    (because you are controlling which call gets which answer) use this.
    """
    resp = Mock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.text = json.dumps(payload)
    return resp


def _dead(side_effect):
    http = Mock()
    for verb in ("get", "post", "patch", "delete", "put"):
        getattr(http, verb).side_effect = side_effect
    return http


@contextmanager
def _patched(http):
    rec = _Rec()

    def _ctor(*args, **kwargs):
        rec.timeout = kwargs.get("timeout", args[0] if args else None)
        return _Ctx(http)

    with patch("web.app.httpx.Client", side_effect=_ctor):
        yield rec


@contextmanager
def _no_proxy():
    """Every httpx call fails, so _find_proxy_url() returns None."""
    with _patched(_dead(OSError("no route"))):
        yield


@contextmanager
def _proxy_urls(urls):
    """Pin the proxy endpoint list the request-mutation routes iterate.

    web.app reads ``benchmark.PROXY_SERVER_URLS`` off the module-level
    LLMModelBenchmark instance. That list is a *class* attribute, so the
    patch has to land on the instance (or the class) — patching the module
    name would silently do nothing and the route would walk the real list.
    """
    import web.app as web_app

    with patch.object(web_app.benchmark, "PROXY_SERVER_URLS", list(urls)):
        yield


# --------------------------------------------------------------------------
# /api/usage
# --------------------------------------------------------------------------


def test_usage_reads_admin_usage(client):
    with _patched(_upstream({"events": [{"model": "m"}]})) as rec, patch("web.app._find_proxy_url", return_value=PROXY):
        res = client.get("/api/usage")
    assert res.get_json() == {"events": [{"model": "m"}]}
    assert rec.timeout == 3.0


def test_usage_503_when_no_proxy_is_reachable(client):
    with _no_proxy():
        assert client.get("/api/usage").status_code == 503


def test_usage_500_when_the_call_itself_fails(client):
    with _patched(_dead(RuntimeError("boom"))), patch("web.app._find_proxy_url", return_value=PROXY):
        res = client.get("/api/usage")
    assert res.status_code == 500
    assert "boom" in res.get_json()["error"]


def test_usage_passes_the_proxy_status_through(client):
    with _patched(_upstream({}, status_code=502)), patch("web.app._find_proxy_url", return_value=PROXY):
        res = client.get("/api/usage")
    assert res.status_code == 502
    assert "502" in res.get_json()["error"]


# --------------------------------------------------------------------------
# /api/models/switch, /api/models/unload, /api/vram/clear
# --------------------------------------------------------------------------


@pytest.mark.parametrize("route", ["/api/models/switch", "/api/models/unload"])
def test_model_lifecycle_requires_a_model(client, route):
    assert client.post(route, json={}).status_code == 400
    assert client.post(route, json={"model": ""}).status_code == 400


@pytest.mark.parametrize(
    "route,upstream_suffix",
    [
        ("/api/models/switch", "/admin/models/switch"),
        ("/api/models/unload", "/admin/models/unload"),
    ],
)
def test_model_lifecycle_forwards_only_the_model(client, route, upstream_suffix):
    http = _upstream({"status": "ok"})
    with _patched(http) as rec, patch("web.app._find_proxy_url", return_value=PROXY):
        res = client.post(route, json={"model": "qwen3:8b", "keep_alive": 999})
    assert res.get_json() == {"status": "ok"}
    assert http.post.call_args.args[0] == f"{PROXY}{upstream_suffix}"
    # A stray key would be forwarded verbatim to the proxy, where it is
    # ignored at best and misread at worst.
    assert http.post.call_args.kwargs["json"] == {"model": "qwen3:8b"}
    assert rec.timeout == 30.0


@pytest.mark.parametrize("route", ["/api/models/switch", "/api/models/unload"])
def test_model_lifecycle_503_without_a_proxy(client, route):
    with _no_proxy():
        assert client.post(route, json={"model": "m"}).status_code == 503


@pytest.mark.parametrize("route", ["/api/models/switch", "/api/models/unload"])
def test_model_lifecycle_500_on_transport_failure(client, route):
    with _patched(_dead(RuntimeError("boom"))), patch("web.app._find_proxy_url", return_value=PROXY):
        res = client.post(route, json={"model": "m"})
    assert res.status_code == 500


def test_model_switch_surfaces_the_proxys_own_error_text(client):
    """The proxy answers 400 with a reason (e.g. online_model); keep it."""
    resp = Mock()
    resp.status_code = 400
    resp.text = "online model prefixes are not served locally"
    http = Mock()
    http.get.return_value = resp
    http.post.return_value = resp
    with _patched(http), patch("web.app._find_proxy_url", return_value=PROXY):
        res = client.post("/api/models/switch", json={"model": "openrouter:x"})
    assert res.status_code == 400
    assert "online model" in res.get_json()["error"]


def test_vram_clear_needs_no_body_and_uses_a_longer_timeout(client):
    """It drains active + queued LLM and SD requests, so 3 s is too short."""
    http = _upstream({"status": "cleared"})
    with _patched(http) as rec, patch("web.app._find_proxy_url", return_value=PROXY):
        res = client.post("/api/vram/clear", json={})
    assert res.get_json() == {"status": "cleared"}
    assert http.post.call_args.args[0] == f"{PROXY}/admin/vram/clear"
    assert rec.timeout == 45.0


def test_vram_clear_503_without_a_proxy(client):
    with _no_proxy():
        assert client.post("/api/vram/clear", json={}).status_code == 503


def test_vram_clear_500_on_transport_failure(client):
    with _patched(_dead(RuntimeError("boom"))), patch("web.app._find_proxy_url", return_value=PROXY):
        assert client.post("/api/vram/clear", json={}).status_code == 500


# --------------------------------------------------------------------------
# /api/models/text, /api/models/vision, /api/models/online
# --------------------------------------------------------------------------


def test_text_models_excludes_image_models_and_dedupes(client):
    tags = _upstream({"models": [{"name": "a:q4", "type": "model"}, {"name": "sd:fp16", "type": "image"}]})
    with _patched(tags), patch("web.app._get_router_text_models", return_value=["a", "b:latest"]):
        res = client.get("/api/models/text")
    models = res.get_json()["models"]
    assert "sd:fp16" not in models
    # A router entry and an Ollama tag naming the same model collapse to
    # one entry once ':latest' is canonicalised away.
    assert models == ["a:q4", "a", "b"]


def test_text_models_survives_a_dead_proxy(client):
    with _patched(_dead(OSError("no route"))), patch("web.app._get_router_text_models", return_value=["solo"]):
        res = client.get("/api/models/text")
    assert res.get_json() == {"models": ["solo"]}


def test_text_models_reads_proxy_api_tags_with_auth(client):
    tags = _upstream({"models": []})
    with _patched(tags), patch("web.app._get_router_text_models", return_value=[]), patch.dict(
        "os.environ", {"ALPACA_API_KEY": "secret"}
    ):
        client.get("/api/models/text")
    assert tags.get.call_args.args[0] == f"{PROXY_URL}/api/tags"
    assert tags.get.call_args.kwargs["headers"]["X-API-Key"] == "secret"


@pytest.mark.parametrize("name,expected_first", [("qwen2.5-vl-7b", "qwen2.5-vl-7b"), ("qwen2.5-vl-70b", "qwen2.5-vl-70b")])
def test_vision_models_filters_and_sorts_largest_first(client, name, expected_first):
    with patch("web.app._get_router_text_models", return_value=[name, "plain-8b"]):
        res = client.get("/api/models/vision")
    models = res.get_json()["models"]
    assert "plain-8b" not in models
    assert models[0] == expected_first


def test_vision_models_orders_70b_before_7b(client):
    with patch("web.app._get_router_text_models", return_value=["x-vl-7b", "y-vl-70b", "z-llava-3b"]):
        assert client.get("/api/models/vision").get_json()["models"] == ["y-vl-70b", "x-vl-7b", "z-llava-3b"]


def test_vision_models_empty_is_not_an_error(client):
    with patch("web.app._get_router_text_models", return_value=[]):
        res = client.get("/api/models/vision")
    assert res.status_code == 200
    assert res.get_json() == {"models": []}


def test_online_models_returns_providers_and_catalog(client):
    fake = Mock()
    fake.get_configured_providers.return_value = {"openrouter": True}
    fake.get_available_models.return_value = ["openrouter:a"]
    with patch("online_providers.online_model_provider", fake):
        res = client.get("/api/models/online")
    assert res.get_json() == {"providers": {"openrouter": True}, "models": ["openrouter:a"]}


def test_online_models_500_when_the_provider_raises(client):
    with patch("online_providers.online_model_provider", Mock(**{"get_configured_providers.side_effect": OSError("x")})):
        assert client.get("/api/models/online").status_code == 500


# --------------------------------------------------------------------------
# /api/requests/clear, /cancel, /resubmit/<id>
# --------------------------------------------------------------------------


def test_requests_clear_hits_the_proxy_and_the_online_tracker(client):
    http = _upstream({"status": "cleared"})
    with _patched(http) as rec, patch("web.app._find_proxy_url", return_value=PROXY), patch(
        "online_providers.clear_completed_online_requests"
    ) as clear_online:
        res = client.post("/api/requests/clear", json={})
    assert res.get_json() == {"status": "cleared"}
    assert http.post.call_args.args[0] == f"{PROXY}/admin/requests/clear"
    assert clear_online.called is True
    assert rec.timeout == 3.0


def test_requests_clear_503_without_a_proxy(client):
    with _no_proxy():
        assert client.post("/api/requests/clear", json={}).status_code == 503


def test_requests_clear_500_on_transport_failure(client):
    with _patched(_dead(RuntimeError("boom"))), patch("web.app._find_proxy_url", return_value=PROXY):
        res = client.post("/api/requests/clear", json={})
    assert res.status_code == 500
    assert "boom" in res.get_json()["error"]


def test_requests_clear_does_not_clear_online_when_the_proxy_fails(client):
    """A failed proxy clear must not leave the online buffer half-reset."""
    with _patched(_upstream({}, status_code=500)), patch("web.app._find_proxy_url", return_value=PROXY), patch(
        "online_providers.clear_completed_online_requests"
    ) as clear_online:
        res = client.post("/api/requests/clear", json={})
    assert res.status_code == 500
    assert clear_online.called is False


def test_requests_cancel_requires_a_request_id(client):
    assert client.post("/api/requests/cancel", json={}).status_code == 400
    assert client.post("/api/requests/cancel", json={"request_id": ""}).status_code == 400


def test_requests_cancel_handles_online_ids_in_process(client):
    with patch("online_providers.cancel_online_request", return_value=True) as cancel:
        res = client.post("/api/requests/cancel", json={"request_id": "online-42"})
    assert res.get_json() == {"status": "cancelled", "request_id": "online-42", "model": "online"}
    cancel.assert_called_once_with("online-42")


def test_requests_cancel_never_sends_an_online_id_to_a_proxy(client):
    """An online request is not in any proxy's buffer; a DELETE there is
    a silent no-op that would look like a successful cancel."""
    with patch("online_providers.cancel_online_request", return_value=False), _no_proxy():
        res = client.post("/api/requests/cancel", json={"request_id": "online-42"})
    assert res.status_code == 503


def test_requests_cancel_deletes_from_the_first_proxy_that_answers(client):
    urls = ["http://p1:11434", "http://p2:11434"]
    http = Mock()
    http.get.return_value = _resp_only({"version": "x"})
    http.delete.return_value = _resp_only({"status": "cancelled"})

    with (
        _patched(http),
        _proxy_urls(urls),
        patch("online_providers.cancel_online_request", return_value=False),
    ):
        res = client.post("/api/requests/cancel", json={"request_id": "abc"})
    assert res.get_json() == {"status": "cancelled"}
    assert http.delete.call_args_list[0].args[0] == f"{urls[0]}/admin/requests/abc"
    # It stops at the first proxy that answers rather than spraying DELETEs.
    assert len(http.delete.call_args_list) == 1


def test_requests_cancel_503_when_no_proxy_answers(client):
    with (
        _patched(_dead(RuntimeError("down"))),
        _proxy_urls(["http://p1:11434"]),
        patch("online_providers.cancel_online_request", return_value=False),
    ):
        res = client.post("/api/requests/cancel", json={"request_id": "abc"})
    assert res.status_code == 503


@pytest.fixture
def _resubmit_env():
    with patch("online_providers.get_online_requests", return_value={"active_requests": [], "completed_requests": []}):
        yield


def test_resubmit_unknown_id_is_404(client, _resubmit_env):
    with _patched(_upstream({"active_requests": [], "completed_requests": []})), _proxy_urls(["http://p1:11434"]):
        res = client.post("/api/requests/resubmit/nope")
    assert res.status_code == 404
    assert res.get_json()["error"] == "Request not found in any proxy"


def test_resubmit_uses_the_persistent_store_first(client, _resubmit_env):
    """GET /admin/resubmit/{id} never rotates out, so it is tried first."""
    http = Mock()
    http.get.side_effect = [
        _resp_only({"type": "ollama_generate", "model": "m", "prompt": "hi"}),
        _resp_only({"done": True, "response": "there"}),
    ]
    http.post.return_value = _resp_only({"done": True, "response": "there"})
    http.delete.return_value = _resp_only({})
    with _patched(http), _proxy_urls(["http://p1:11434"]):
        res = client.post("/api/requests/resubmit/abc")
    assert res.get_json()["status"] == "resubmitted"
    assert http.get.call_args_list[0].args[0] == "http://p1:11434/admin/resubmit/abc"
    assert http.post.call_args.args[0] == "http://p1:11434/api/generate"
    assert http.post.call_args.kwargs["json"]["model"] == "m"
    # The stuck request is dropped once it has been replayed.
    assert http.delete.called is True


@pytest.mark.parametrize(
    "req_type,endpoint",
    [
        ("ollama_generate", "/api/generate"),
        ("openai_generate", "/v1/completions"),
        ("ollama_chat", "/api/chat"),
        ("openai_chat", "/api/chat"),
        ("something_else", "/api/chat"),
    ],
)
def test_resubmit_routes_each_request_type_to_its_endpoint(client, _resubmit_env, req_type, endpoint):
    http = Mock()
    http.get.side_effect = [
        _resp_only({"type": req_type, "model": "m", "prompt": "USER: hello"}),
        _resp_only({"done": True}),
    ]
    http.post.return_value = _resp_only({"done": True})
    http.delete.return_value = _resp_only({})
    with _patched(http), _proxy_urls(["http://p1:11434"]):
        res = client.post("/api/requests/resubmit/abc")
    assert res.status_code == 200
    assert http.post.call_args.args[0] == f"http://p1:11434{endpoint}"


def test_resubmit_strips_the_stored_assistant_turn(client, _resubmit_env):
    """The stored prompt carries the model's own last answer; replaying it
    verbatim would make the model continue its own output."""
    http = Mock()
    http.get.side_effect = [
        _resp_only(
            {
                "type": "ollama_chat",
                "model": "m",
                "prompt": "USER: what is 2+2\nASSISTANT: four\nASSISTANT: (it is four)",
            }
        ),
        _resp_only({"done": True}),
    ]
    http.post.return_value = _resp_only({"done": True})
    http.delete.return_value = _resp_only({})
    with _patched(http), _proxy_urls(["http://p1:11434"]):
        client.post("/api/requests/resubmit/abc")
    messages = http.post.call_args.kwargs["json"]["messages"]
    assert [m["role"] for m in messages] == ["user"]


def test_resubmit_falls_back_to_the_active_request_buffer(client, _resubmit_env):
    http = Mock()
    http.get.side_effect = [
        _resp_only({}, status_code=404),
        _resp_only(
            {
                "active_requests": [{"request_id": "abc", "type": "ollama_generate", "model": "m", "prompt": "hi"}],
                "completed_requests": [],
            }
        ),
        _resp_only({"done": True}),
    ]
    http.post.return_value = _resp_only({"done": True})
    http.delete.return_value = _resp_only({})
    with _patched(http), _proxy_urls(["http://p1:11434"]):
        res = client.post("/api/requests/resubmit/abc")
    assert res.get_json()["status"] == "resubmitted"
    assert http.get.call_args_list[1].args[0] == "http://p1:11434/admin/requests"


def test_resubmit_keeps_the_model_resident(client, _resubmit_env):
    http = Mock()
    http.get.side_effect = [
        _resp_only({"type": "ollama_generate", "model": "m", "prompt": "hi"}),
        _resp_only({"done": True}),
    ]
    http.post.return_value = _resp_only({"done": True})
    http.delete.return_value = _resp_only({})
    with _patched(http), _proxy_urls(["http://p1:11434"]):
        client.post("/api/requests/resubmit/abc")
    assert http.post.call_args.kwargs["json"]["keep_alive"] == -1


def test_resubmit_surfaces_a_replay_failure(client, _resubmit_env):
    http = Mock()
    http.get.side_effect = [
        _resp_only({"type": "ollama_generate", "model": "m", "prompt": "hi"}),
    ]
    http.post.return_value = _resp_only({}, status_code=500)
    with _patched(http), _proxy_urls(["http://p1:11434"]):
        res = client.post("/api/requests/resubmit/abc")
    assert res.status_code == 500
    assert "500" in res.get_json()["error"]


def test_resubmit_replays_an_online_request_in_process(client):
    online = {
        "active_requests": [{"request_id": "online-1", "prompt": "why is the sky blue", "model": "openrouter:x"}],
        "completed_requests": [],
    }
    provider = Mock()
    provider.query_online_model = AsyncMock(return_value={"response": "rayleigh scattering"})
    with patch("online_providers.get_online_requests", return_value=online), patch(
        "online_providers.online_model_provider", provider
    ), _no_proxy():
        res = client.post("/api/requests/resubmit/online-1")
    assert res.get_json()["status"] == "resubmitted"
    assert provider.query_online_model.call_args.kwargs["model_identifier"] == "openrouter:x"
    assert provider.query_online_model.call_args.kwargs["max_tokens"] == 4000


def test_resubmit_online_request_without_a_prompt_is_400(client):
    online = {"active_requests": [{"request_id": "online-1", "prompt": "", "model": "openrouter:x"}], "completed_requests": []}
    with patch("online_providers.get_online_requests", return_value=online):
        res = client.post("/api/requests/resubmit/online-1")
    assert res.status_code == 400
    assert "no prompt" in res.get_json()["error"]


def test_resubmit_online_failure_is_500(client):
    online = {"active_requests": [{"request_id": "online-1", "prompt": "q", "model": "openrouter:x"}], "completed_requests": []}
    provider = Mock()
    provider.query_online_model = AsyncMock(side_effect=RuntimeError("rate limited"))
    with patch("online_providers.get_online_requests", return_value=online), patch(
        "online_providers.online_model_provider", provider
    ):
        res = client.post("/api/requests/resubmit/online-1")
    assert res.status_code == 500
    assert "rate limited" in res.get_json()["error"]


# --------------------------------------------------------------------------
# /api/routing/matrix
# --------------------------------------------------------------------------


@pytest.fixture
def _matrix_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path / "data" / "routing_matrix.json"


def test_routing_matrix_get_returns_the_default_template(client, _matrix_path):
    matrix = client.get("/api/routing/matrix").get_json()
    assert set(matrix) == {"fast_chat", "complex_coding", "reasoning", "summarization"}
    # No model is hardcoded: an unconfigured task falls back to whatever
    # is loaded, so the template must ship null.
    assert all(v["model"] is None for v in matrix.values())
    assert matrix["reasoning"]["reasoning_required"] is True
    assert matrix["fast_chat"]["reasoning_required"] is False


def test_routing_matrix_get_reads_the_saved_file(client, _matrix_path):
    _matrix_path.parent.mkdir(parents=True)
    _matrix_path.write_text(json.dumps({"fast_chat": {"model": "m1", "min_tps": 99}}))
    assert client.get("/api/routing/matrix").get_json() == {"fast_chat": {"model": "m1", "min_tps": 99}}


def test_routing_matrix_get_falls_back_when_the_file_is_corrupt(client, _matrix_path):
    _matrix_path.parent.mkdir(parents=True)
    _matrix_path.write_text("{not json")
    assert "fast_chat" in client.get("/api/routing/matrix").get_json()


def test_routing_matrix_post_persists_and_echoes(client, _matrix_path):
    payload = {"fast_chat": {"model": "m1", "min_tps": 40.0, "max_ttft_ms": 250, "reasoning_required": False}}
    res = client.post("/api/routing/matrix", json=payload)
    assert res.get_json()["status"] == "success"
    assert res.get_json()["matrix"] == payload
    assert json.loads(_matrix_path.read_text()) == payload


def test_routing_matrix_post_replaces_wholesale(client, _matrix_path):
    """The editor posts the whole matrix; a merge would keep dead tasks."""
    _matrix_path.parent.mkdir(parents=True)
    _matrix_path.write_text(json.dumps({"fast_chat": {"model": "a"}, "stale": {"model": "b"}}))
    client.post("/api/routing/matrix", json={"fast_chat": {"model": "c"}})
    assert json.loads(_matrix_path.read_text()) == {"fast_chat": {"model": "c"}}


def test_routing_matrix_post_500_when_the_write_fails(client, _matrix_path):
    with patch("builtins.open", side_effect=OSError("read-only fs")):
        res = client.post("/api/routing/matrix", json={"fast_chat": {}})
    assert res.status_code == 500
    assert "read-only fs" in res.get_json()["error"]


# --------------------------------------------------------------------------
# /api/routing/optimal
# --------------------------------------------------------------------------


def test_optimal_prefers_the_configured_model(client, _matrix_path):
    _matrix_path.parent.mkdir(parents=True)
    _matrix_path.write_text(json.dumps({"fast_chat": {"model": "m1"}}))
    with patch("web.app._get_currently_loaded_model", return_value="loaded"):
        res = client.get("/api/routing/optimal?task=fast_chat")
    assert res.get_json()["optimal_model"] == "m1"
    assert "configured model" in res.get_json()["explanation"]


def test_optimal_falls_back_to_the_loaded_model(client, _matrix_path):
    with patch("web.app._get_currently_loaded_model", return_value="loaded"):
        res = client.get("/api/routing/optimal?task=fast_chat")
    assert res.get_json()["optimal_model"] == "loaded"
    assert "falling back" in res.get_json()["explanation"]


def test_optimal_is_null_when_nothing_is_configured_or_loaded(client, _matrix_path):
    with patch("web.app._get_currently_loaded_model", return_value=None):
        res = client.get("/api/routing/optimal?task=fast_chat")
    assert res.get_json()["optimal_model"] is None
    # fallback_model must still be a usable answer for the caller.
    assert res.get_json()["fallback_model"] is None


def test_optimal_defaults_to_the_fast_chat_task(client, _matrix_path):
    with patch("web.app._get_currently_loaded_model", return_value="loaded"):
        assert client.get("/api/routing/optimal").get_json()["task"] == "fast_chat"


def test_optimal_keeps_a_model_that_meets_its_constraints(client, _matrix_path):
    _matrix_path.parent.mkdir(parents=True)
    _matrix_path.write_text(json.dumps({"fast_chat": {"model": "m1"}}))
    bench = {"avg_tokens_per_sec": 55.0, "avg_ttft_ms": 120}
    with (
        patch("web.app._get_currently_loaded_model", return_value="loaded"),
        patch("analyzer.load_latest_benchmark", return_value=bench),
    ):
        res = client.get("/api/routing/optimal?task=fast_chat&min_tps=40&max_ttft_ms=250")
    assert res.get_json()["optimal_model"] == "m1"
    assert "did not meet constraints" not in res.get_json()["explanation"]


def test_optimal_searches_alternatives_when_constraints_fail(client, _matrix_path, tmp_path):
    _matrix_path.parent.mkdir(parents=True)
    _matrix_path.write_text(json.dumps({"fast_chat": {"model": "slow"}}))
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    (bench_dir / "b.json").write_text(
        json.dumps(
            {
                "results": [
                    {"model": "slow", "avg_tokens_per_sec": 5.0, "avg_ttft_ms": 900},
                    {"model": "fast", "avg_tokens_per_sec": 90.0, "avg_ttft_ms": 100},
                ]
            }
        )
    )
    with (
        patch("web.app._get_currently_loaded_model", return_value="loaded"),
        patch("analyzer.load_latest_benchmark", return_value={"avg_tokens_per_sec": 5.0, "avg_ttft_ms": 900}),
        patch("analyzer.BENCHMARK_DIR", bench_dir),
    ):
        res = client.get("/api/routing/optimal?task=fast_chat&min_tps=40&max_ttft_ms=250")
    assert res.get_json()["optimal_model"] == "fast"
    assert "optimal alternative" in res.get_json()["explanation"]


def test_optimal_survives_a_broken_benchmark_store(client, _matrix_path):
    with (
        patch("web.app._get_currently_loaded_model", return_value="loaded"),
        patch("analyzer.load_latest_benchmark", side_effect=OSError("corrupt")),
    ):
        res = client.get("/api/routing/optimal?task=fast_chat&min_tps=40")
    assert res.status_code == 200
    assert "Benchmark validation skipped" in res.get_json()["explanation"]


def test_optimal_does_not_benchmark_when_no_model_resolves(client, _matrix_path):
    with patch("web.app._get_currently_loaded_model", return_value=None), patch(
        "analyzer.load_latest_benchmark"
    ) as bench:
        res = client.get("/api/routing/optimal?task=fast_chat&min_tps=40")
    assert res.get_json()["optimal_model"] is None
    assert bench.called is False


@pytest.mark.parametrize("raw,expected", [("true", True), ("1", True), ("yes", True), ("false", False), ("0", False)])
def test_optimal_parses_reasoning_required(client, _matrix_path, raw, expected):
    with patch("web.app._get_currently_loaded_model", return_value="loaded"), patch(
        "analyzer.load_latest_benchmark", return_value=None
    ):
        assert client.get(f"/api/routing/optimal?reasoning_required={raw}").status_code == 200


# --------------------------------------------------------------------------
# /api/telemetry/recommendations and /apply
# --------------------------------------------------------------------------


def test_recommendations_without_a_model_fall_back_to_the_active_one(client):
    runtime = _upstream({"active_model": "m1"})
    analysis = {"status": "ok", "model_alias": "m1", "recommendations": {"ctx-size": 8192}}
    with _patched(runtime), patch("web.app._find_proxy_url", return_value=PROXY), patch(
        "analyzer.analyze_telemetry", return_value=analysis
    ) as analyze:
        res = client.get("/api/telemetry/recommendations")
    assert res.get_json()["status"] == "ok"
    assert analyze.call_args.args[0] == "m1"


def test_recommendations_are_insufficient_data_with_nothing_loaded(client):
    with _patched(_upstream({"active_model": None})), patch("web.app._find_proxy_url", return_value=PROXY), patch(
        "analyzer.analyze_telemetry"
    ) as analyze:
        res = client.get("/api/telemetry/recommendations")
    assert res.get_json()["status"] == "insufficient_data"
    assert analyze.called is False


def test_recommendations_survive_an_unreachable_proxy(client):
    with _no_proxy():
        res = client.get("/api/telemetry/recommendations")
    assert res.get_json()["status"] == "insufficient_data"


@pytest.mark.parametrize("strategy,expected", [("performance", True), ("safe", False), ("anything-else", False)])
def test_recommendations_select_the_strategy(client, strategy, expected):
    with patch("analyzer.analyze_telemetry", return_value={"status": "ok"}) as analyze, _no_proxy():
        client.get(f"/api/telemetry/recommendations?model=m1&strategy={strategy}")
    assert analyze.call_args.kwargs["performance_first"] is expected


def test_recommendations_sanitize_the_model_name(client):
    """The telemetry files are named with / : . replaced by _."""
    with patch("analyzer.analyze_telemetry", return_value={"status": "ok"}) as analyze, _no_proxy():
        client.get("/api/telemetry/recommendations?model=qwen3.6-35b-a3b:q4_k_m")
    assert analyze.call_args.args[0] == "qwen3.6-35b-a3b_q4_k_m"


def test_recommendations_500_shape_when_the_analyzer_itself_raises(client):
    with patch("analyzer.analyze_telemetry", side_effect=RuntimeError("no telemetry module")):
        res = client.get("/api/telemetry/recommendations?model=m1")
    assert res.get_json()["status"] == "error"
    assert "no telemetry module" in res.get_json()["detected_issues"][0]


def test_recommendations_fold_in_vram_stepdown_records(client):
    http = Mock()
    http.get.side_effect = lambda url, *a, **k: _resp_only(
        {"recommendations": {"m1": {"count": 3, "last_budget": {"n-gpu-layers": 20}, "recommend": True}}}
        if url.endswith("/admin/vram-recommendations")
        else _resp_only({"active_model": "m1"})
    )
    analysis = {"status": "ok", "detected_issues": []}
    with _patched(http), patch("web.app._find_proxy_url", return_value=PROXY), patch(
        "analyzer.analyze_telemetry", return_value=analysis
    ):
        res = client.get("/api/telemetry/recommendations?model=m1")
    body = res.get_json()
    assert body["vram_budgeting"]["count"] == 3
    assert any("stepped this model down 3x" in i for i in body["detected_issues"])
    assert any("lower ctx-size" in i for i in body["detected_issues"])


def test_recommendations_ignore_a_recommendation_below_the_threshold(client):
    http = Mock()
    http.get.side_effect = lambda url, *a, **k: (
        _resp_only({"recommendations": {"m1": {"count": 1, "recommend": False}}})
        if url.endswith("/admin/vram-recommendations")
        else _resp_only({"active_model": "m1"})
    )
    with _patched(http), patch("web.app._find_proxy_url", return_value=PROXY), patch(
        "analyzer.analyze_telemetry", return_value={"status": "ok", "detected_issues": []}
    ):
        body = client.get("/api/telemetry/recommendations?model=m1").get_json()
    assert len(body["detected_issues"]) == 1
    assert "lower ctx-size" not in body["detected_issues"][0]


def test_apply_recommendations_requires_model_and_payload(client):
    assert client.post("/api/telemetry/recommendations/apply", json={"recommendations": {"a": 1}}).status_code == 400
    assert client.post("/api/telemetry/recommendations/apply", json={"model": "m1"}).status_code == 400
    assert client.post("/api/telemetry/recommendations/apply", json={"model": "m1", "recommendations": {}}).status_code == 400


def test_apply_recommendations_merges_into_the_existing_profile(client, tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_MODELS_DIR", str(tmp_path))
    (tmp_path / "m1.profile.json").write_text(json.dumps({"temperature": 0.6, "ctx-size": 4096}))
    with patch("alpaca_puller.update_models_ini") as ini:
        res = client.post(
            "/api/telemetry/recommendations/apply",
            json={"model": "m1", "recommendations": {"ctx-size": 32768, "n-gpu-layers": 30}},
        )
    body = res.get_json()
    assert body["status"] == "success"
    assert ini.called is True
    saved = json.loads((tmp_path / "m1.profile.json").read_text())
    # An existing unrelated key must survive the apply.
    assert saved["temperature"] == 0.6
    assert saved["ctx-size"] == 32768
    assert saved["n-gpu-layers"] == 30


def test_apply_recommendations_creates_a_missing_profile(client, tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_MODELS_DIR", str(tmp_path))
    with patch("alpaca_puller.update_models_ini"):
        res = client.post("/api/telemetry/recommendations/apply", json={"model": "brand-new", "recommendations": {"ctx-size": 8192}})
    assert res.status_code == 200
    assert json.loads((tmp_path / "brand-new.profile.json").read_text()) == {"ctx-size": 8192}


def test_apply_recommendations_resolves_the_gguf_stem(client, tmp_path, monkeypatch):
    """The dashboard sends the public name; the profile is keyed by the
    GGUF file stem, which uses the -- separator."""
    monkeypatch.setenv("ROUTER_MODELS_DIR", str(tmp_path))
    (tmp_path / "qwen3.6-35b-a3b--q4_k_m.gguf").touch()
    with patch("alpaca_puller.update_models_ini"):
        client.post(
            "/api/telemetry/recommendations/apply",
            json={"model": "qwen3.6-35b-a3b:q4_k_m", "recommendations": {"ctx-size": 8192}},
        )
    assert (tmp_path / "qwen3.6-35b-a3b--q4_k_m.profile.json").exists()


def test_apply_recommendations_says_so_when_the_ini_cannot_be_regenerated(client, tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_MODELS_DIR", str(tmp_path))
    with patch("alpaca_puller.update_models_ini", side_effect=OSError("no puller")):
        res = client.post("/api/telemetry/recommendations/apply", json={"model": "m1", "recommendations": {"ctx-size": 8192}})
    assert "could not regenerate models.ini" in res.get_json()["message"]


def test_apply_recommendations_500_when_the_profile_cannot_be_written(client, tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_MODELS_DIR", str(tmp_path))
    with patch("builtins.open", side_effect=OSError("disk full")):
        res = client.post("/api/telemetry/recommendations/apply", json={"model": "m1", "recommendations": {"ctx-size": 8192}})
    assert res.status_code == 500
    assert "disk full" in res.get_json()["error"]


# --------------------------------------------------------------------------
# /api/tests/shared_llm, /api/tests/<id>/{download,responses,attachment}
# --------------------------------------------------------------------------


def test_shared_llm_tasks_are_listed_with_their_own_type(client):
    tasks = [
        {"id": "fast_path_light", "category": "FastPath Intent", "label": "HA Intent", "prompt": "x", "max_tokens": 1, "task_type": "intent"},
        {"id": "code_raven_redis_lock", "category": "Raven Code Gen", "label": "Lock", "prompt": "x", "max_tokens": 1, "task_type": "ast_code"},
    ]
    with patch("web.shared_llm_benchmark.SharedLLMModelBenchmark.get_all_tasks", return_value=tasks):
        res = client.get("/api/tests/shared_llm")
    assert res.get_json()["tests"] == [
        {"id": "fast_path_light", "category": "FastPath Intent", "label": "HA Intent", "type": "shared_llm"},
        {"id": "code_raven_redis_lock", "category": "Raven Code Gen", "label": "Lock", "type": "shared_llm"},
    ]


def test_shared_llm_500_when_the_task_registry_raises(client):
    with patch("web.shared_llm_benchmark.SharedLLMModelBenchmark.get_all_tasks", side_effect=RuntimeError("bad registry")):
        res = client.get("/api/tests/shared_llm")
    assert res.status_code == 500
    assert "bad registry" in res.get_json()["error"]


def test_test_download_returns_the_definition_as_an_attachment(client):
    test = {"id": "t1", "label": "T One", "prompt": "do the thing", "expected": "yes", "num_predict": 500}
    with patch("web.app._find_test", return_value=("cat", test)):
        res = client.get("/api/tests/t1/download")
    assert res.mimetype == "application/json"
    assert "attachment" in res.headers["Content-Disposition"]
    body = json.loads(res.data)
    assert body["id"] == "t1"
    assert body["kind"] == "text"
    assert body["prompt"] == "do the thing"
    assert body["expected"] == "yes"
    # The full attachment payload must not be inlined into the download.
    assert body["attachments"] == []


def test_test_download_404_for_an_unknown_id(client):
    with patch("web.app._find_test", return_value=(None, None)):
        res = client.get("/api/tests/nope/download")
    assert res.status_code == 404
    assert res.get_json()["error"] == "test not found"


def test_test_download_500_on_an_unexpected_error(client):
    with patch("web.app._find_test", side_effect=RuntimeError("registry gone")):
        res = client.get("/api/tests/t1/download")
    assert res.status_code == 500
    assert "registry gone" in res.get_json()["error"]


def test_test_responses_404_for_an_unknown_id(client):
    with patch("web.app._find_test", return_value=(None, None)):
        assert client.get("/api/tests/nope/responses").status_code == 404


def test_test_responses_are_empty_when_nothing_ran(client, tmp_path):
    test = {"id": "t1", "label": "T One", "prompt": "p", "type": "code"}
    with (
        patch("web.app._find_test", return_value=("cat", test)),
        patch.object(benchmark, "MODELS_DIR", tmp_path),
        patch.object(multistep_benchmark, "MODELS_DIR", tmp_path),
    ):
        res = client.get("/api/tests/t1/responses")
    assert res.status_code == 200
    assert res.get_json()["responses"] == []


def test_test_responses_keep_the_longest_response_per_model(client, tmp_path):
    test = {"id": "t1", "label": "T One", "prompt": "p", "type": "code"}
    (tmp_path / "general_m.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "category_code": {
                            "tests": [
                                {"test_id": "t1", "response": "short", "passed": False},
                                {"test_id": "other", "response": "not this one"},
                            ]
                        }
                    }
                ]
            }
        )
    )
    with (
        patch("web.app._find_test", return_value=("cat", test)),
        patch.object(benchmark, "MODELS_DIR", tmp_path),
        patch.object(multistep_benchmark, "MODELS_DIR", tmp_path),
    ):
        body = client.get("/api/tests/t1/responses").get_json()
    assert body["test_id"] == "t1"
    assert len(body["responses"]) == 1


def test_attachment_404_for_an_unknown_test(client):
    with patch("web.app._find_test", return_value=(None, None)):
        res = client.get("/api/tests/nope/attachment/a.png")
    assert res.status_code == 404
    assert res.get_json()["error"] == "test not found"


def test_attachment_404_for_a_name_the_test_does_not_have(client):
    test = {"id": "t1", "attachments": [{"name": "a.png", "data_base64": "aGk="}]}
    with patch("web.app._find_test", return_value=("cat", test)):
        res = client.get("/api/tests/t1/attachment/b.png")
    assert res.status_code == 404
    assert res.get_json()["error"] == "attachment not found"


def test_attachment_serves_inline_base64_with_its_declared_mime(client):
    test = {"id": "t1", "attachments": [{"name": "a.png", "data_base64": "aGk=", "mime": "image/png"}]}
    with patch("web.app._find_test", return_value=("cat", test)):
        res = client.get("/api/tests/t1/attachment/a.png")
    assert res.mimetype == "image/png"
    assert res.data == b"hi"
    assert "attachment" not in res.headers.get("Content-Disposition", "")


def test_attachment_honours_the_download_flag(client):
    test = {"id": "t1", "attachments": [{"name": "a.png", "data_base64": "aGk=", "mime": "image/png"}]}
    with patch("web.app._find_test", return_value=("cat", test)):
        res = client.get("/api/tests/t1/attachment/a.png?download=1")
    assert "attachment" in res.headers["Content-Disposition"]


def test_attachment_refuses_a_path_escaping_the_repo(client):
    test = {"id": "t1", "attachments": [{"name": "secret", "path": "../../etc/passwd"}]}
    with patch("web.app._find_test", return_value=("cat", test)):
        res = client.get("/api/tests/t1/attachment/secret")
    assert res.status_code == 404


def test_attachment_serves_a_text_attachment(client):
    test = {"id": "t1", "attachments": [{"name": "notes.md", "text": "# hi", "mime": "text/markdown"}]}
    with patch("web.app._find_test", return_value=("cat", test)):
        res = client.get("/api/tests/t1/attachment/notes.md")
    assert res.data == b"# hi"
    assert res.mimetype == "text/markdown"


# --------------------------------------------------------------------------
# /api/ratings
# --------------------------------------------------------------------------


@pytest.fixture
def _ratings_file(tmp_path, monkeypatch):
    import web.app as web_app

    path = tmp_path / "human_ratings.json"
    monkeypatch.setattr(web_app, "_RATINGS_FILE", path)
    return path


def test_ratings_get_starts_empty(client, _ratings_file):
    assert client.get("/api/ratings").get_json() == {"ratings": {}}


def test_ratings_round_trip(client, _ratings_file):
    client.post("/api/ratings", json={"testId": "t1", "model": "m1", "rating": 4})
    assert client.get("/api/ratings").get_json()["ratings"] == {"t1": {"m1": 4.0}}


def test_ratings_accept_every_id_spelling(client, _ratings_file):
    for key in ("testId", "test_id", "testid", "id"):
        client.post("/api/ratings", json={key: "t1", "model": "m1", "rating": 3})
    assert client.get("/api/ratings").get_json()["ratings"] == {"t1": {"m1": 3.0}}


@pytest.mark.parametrize("rating,stored", [(0.1, 0.0), (3.3, 3.5), (4.8, 5.0), (5, 5.0), (99, 5.0)])
def test_ratings_are_clamped_and_snapped(client, _ratings_file, rating, stored):
    client.post("/api/ratings", json={"testId": "t1", "model": "m1", "rating": rating})
    ratings = client.get("/api/ratings").get_json()["ratings"]
    # 0.1 snaps down to 0.0, which is the delete signal - nothing is stored.
    assert (ratings.get("t1") or {}).get("m1") == (stored or None)


@pytest.mark.parametrize("rating", [0, -3])
def test_a_rating_that_clamps_to_zero_deletes(client, _ratings_file, rating):
    client.post("/api/ratings", json={"testId": "t1", "model": "m1", "rating": 3})
    client.post("/api/ratings", json={"testId": "t1", "model": "m1", "rating": rating})
    assert client.get("/api/ratings").get_json()["ratings"] == {}


def test_rating_of_zero_deletes_the_whole_test_when_empty(client, _ratings_file):
    client.post("/api/ratings", json={"testId": "t1", "model": "m1", "rating": 3})
    client.post("/api/ratings", json={"testId": "t1", "model": "m1", "rating": 0})
    assert client.get("/api/ratings").get_json()["ratings"] == {}


def test_rating_of_zero_leaves_sibling_models_alone(client, _ratings_file):
    client.post("/api/ratings", json={"testId": "t1", "model": "m1", "rating": 3})
    client.post("/api/ratings", json={"testId": "t1", "model": "m2", "rating": 5})
    client.post("/api/ratings", json={"testId": "t1", "model": "m1", "rating": 0})
    assert client.get("/api/ratings").get_json()["ratings"] == {"t1": {"m2": 5.0}}


@pytest.mark.parametrize("body", [{}, {"model": "m1", "rating": 3}, {"testId": "t1", "rating": 3}, {"testId": "t1", "model": "m1"}])
def test_ratings_reject_an_incomplete_body(client, _ratings_file, body):
    res = client.post("/api/ratings", json=body)
    assert res.status_code == 400
    if not body.get("testId") or not body.get("model"):
        assert res.get_json()["error"] == "testId and model are required"
    else:
        assert res.get_json()["error"] == "invalid rating"


def test_ratings_reject_a_non_numeric_rating(client, _ratings_file):
    res = client.post("/api/ratings", json={"testId": "t1", "model": "m1", "rating": "great"})
    assert res.status_code == 400
    assert res.get_json()["error"] == "invalid rating"


def test_ratings_tolerate_a_missing_body(client, _ratings_file):
    assert client.post("/api/ratings").status_code == 400


def test_ratings_500_when_the_store_cannot_be_read(client, _ratings_file):
    # _save_ratings_file swallows its own write errors, so the only way the
    # caller learns the store is broken is a failure to load it.
    with patch("web.app._load_ratings_file", side_effect=OSError("corrupt json")):
        res = client.post("/api/ratings", json={"testId": "t1", "model": "m1", "rating": 3})
    assert res.status_code == 500
    assert "corrupt json" in res.get_json()["error"]


def test_ratings_get_500_when_the_store_cannot_be_read(client, _ratings_file):
    with patch("web.app._load_ratings_file", side_effect=OSError("corrupt json")):
        res = client.get("/api/ratings")
    assert res.status_code == 500
    # A broken store must not look like "nobody has rated anything".
    assert res.get_json()["ratings"] == {}


# --------------------------------------------------------------------------
# /api/thermal/watchdog
# --------------------------------------------------------------------------


def test_thermal_watchdog_get_returns_config_and_temps(client):
    cfg = {"enabled": True, "abort_c": 93, "throttle_c": 85, "resume_c": 78, "poll_s": 5, "pretest_max_wait_s": 600}
    with patch("web.thermal.load_config", return_value=cfg), patch("web.thermal._read_live_temps", return_value={"cpu": 50, "gpu": 60}):
        res = client.get("/api/thermal/watchdog")
    assert res.get_json() == {"config": cfg, "temps": {"cpu": 50, "gpu": 60}}


def test_thermal_watchdog_get_degrades_when_the_probe_fails(client):
    """A machine with no nvidia-smi must still show the config."""
    cfg = {"enabled": False}
    with patch("web.thermal.load_config", return_value=cfg), patch(
        "web.thermal._read_live_temps", side_effect=OSError("no sensors")
    ):
        res = client.get("/api/thermal/watchdog")
    assert res.status_code == 200
    assert res.get_json()["config"] == cfg
    assert "no sensors" in res.get_json()["temps"]["error"]


def test_thermal_watchdog_get_500_when_the_config_cannot_load(client):
    with patch("web.thermal.load_config", side_effect=OSError("corrupt json")):
        assert client.get("/api/thermal/watchdog").status_code == 500


def test_thermal_watchdog_post_saves_and_echoes_the_clamped_config(client):
    saved = {"enabled": True, "throttle_c": 80, "resume_c": 70}
    with patch("web.thermal.save_config", return_value=saved) as save:
        res = client.post("/api/thermal/watchdog", json={"throttle_c": 80, "resume_c": 70})
    assert res.get_json() == {"success": True, "config": saved}
    assert save.call_args.args[0] == {"throttle_c": 80, "resume_c": 70}


def test_thermal_watchdog_post_tolerates_a_missing_body(client):
    with patch("web.thermal.save_config", return_value={"enabled": True}):
        assert client.post("/api/thermal/watchdog").status_code == 200


def test_thermal_watchdog_post_500_when_saving_fails(client):
    with patch("web.thermal.save_config", side_effect=OSError("read-only fs")):
        res = client.post("/api/thermal/watchdog", json={"throttle_c": 80})
    assert res.status_code == 500
    assert "read-only fs" in res.get_json()["error"]


# --------------------------------------------------------------------------
# /api/models/pulls/<id>/stop and /cancel
# --------------------------------------------------------------------------


@pytest.fixture
def _pull():
    import web.app as web_app

    # Seed and tear down under the lock, but release it across the yield:
    # the routes take active_pulls_lock themselves, so holding it here
    # would deadlock the very request under test.
    with web_app.active_pulls_lock:
        web_app.active_pulls.clear()
        web_app.active_pulls["m1"] = {"model": "m1:latest", "status": "running", "logs": []}
    try:
        yield web_app.active_pulls
    finally:
        with web_app.active_pulls_lock:
            web_app.active_pulls.clear()


def test_stop_unknown_pull_is_404(client, _pull, tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_MODELS_DIR", str(tmp_path))
    assert client.post("/api/models/pulls/ghost/stop", json={}).status_code == 404


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled", "error"])
def test_stop_refuses_a_pull_that_is_not_running(client, _pull, tmp_path, monkeypatch, status):
    monkeypatch.setenv("ROUTER_MODELS_DIR", str(tmp_path))
    _pull["m1"]["status"] = status
    res = client.post("/api/models/pulls/m1/stop", json={})
    assert res.status_code == 400
    assert status in res.get_json()["error"]


def test_stop_writes_the_marker_and_flips_the_status(client, _pull, tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_MODELS_DIR", str(tmp_path))
    res = client.post("/api/models/pulls/m1/stop", json={})
    assert res.get_json()["status"] == "stopping"
    assert _pull["m1"]["status"] == "stopping"
    marker = tmp_path / ".alpaca-stop" / "m1"
    assert marker.exists()
    assert float(marker.read_text()) > 0


def test_stop_marker_name_is_sanitized_like_the_puller_expects(client, _pull, tmp_path, monkeypatch):
    """alpaca-puller replaces / and : with _ in the marker filename, so a
    model id carrying a tag must land on the same marker the puller polls."""
    monkeypatch.setenv("ROUTER_MODELS_DIR", str(tmp_path))
    _pull.clear()
    _pull["org_model:q4"] = {"model": "org_model:q4", "status": "running"}
    res = client.post("/api/models/pulls/org_model:q4/stop", json={})
    assert res.status_code == 200
    assert (tmp_path / ".alpaca-stop" / "org_model_q4").exists()


def test_cancel_unknown_pull_is_404(client, _pull, tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_MODELS_DIR", str(tmp_path))
    assert client.post("/api/models/pulls/ghost/cancel", json={}).status_code == 404


def test_cancel_twice_is_400(client, _pull, tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_MODELS_DIR", str(tmp_path))
    assert client.post("/api/models/pulls/m1/cancel", json={}).status_code == 200
    assert client.post("/api/models/pulls/m1/cancel", json={}).status_code == 400


def test_cancel_removes_its_own_stop_marker(client, _pull, tmp_path, monkeypatch):
    """A leftover marker is the documented footgun: it silently blocks
    every future pull of that model."""
    monkeypatch.setenv("ROUTER_MODELS_DIR", str(tmp_path))
    marker = tmp_path / ".alpaca-stop" / "m1"
    marker.parent.mkdir(parents=True)
    marker.write_text("1")
    client.post("/api/models/pulls/m1/cancel", json={})
    assert marker.exists() is False
    assert _pull["m1"]["status"] == "cancelled"


def test_cancel_leaves_other_markers_alone(client, _pull, tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_MODELS_DIR", str(tmp_path))
    other = tmp_path / ".alpaca-stop" / "m2"
    other.parent.mkdir(parents=True)
    other.write_text("1")
    client.post("/api/models/pulls/m1/cancel", json={})
    assert other.exists() is True


def test_cancel_succeeds_when_there_was_no_marker(client, _pull, tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_MODELS_DIR", str(tmp_path))
    res = client.post("/api/models/pulls/m1/cancel", json={})
    assert res.status_code == 200
    assert res.get_json()["status"] == "cancelled"


# --------------------------------------------------------------------------
# /api/vision/ocr
# --------------------------------------------------------------------------


def _png_bytes(color=(255, 0, 0), size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _ocr_upload(data, name="page.png", extra=None):
    payload = {"file": (io.BytesIO(data), name)}
    if extra:
        payload.update(extra)
    return payload


def test_ocr_requires_a_file(client):
    res = client.post("/api/vision/ocr", data={})
    assert res.status_code == 400
    assert res.get_json()["error"] == "No file uploaded"


def test_ocr_requires_a_model(client):
    with patch("web.app.httpx.Client"):
        res = client.post("/api/vision/ocr", data=_ocr_upload(_png_bytes()))
    assert res.status_code == 400
    assert "model" in res.get_json()["error"]


def test_ocr_posts_the_image_as_a_jpeg_data_url(client):
    body = {"choices": [{"message": {"content": '{"full_text": "FLYER", "headline": "F", "subtext": "", "badge": ""}'}}]}
    http = _upstream(body)
    with _patched(http) as rec, patch.dict("os.environ", {"ALPACA_API_KEY": "k"}):
        res = client.post(
            "/api/vision/ocr",
            data=_ocr_upload(_png_bytes(), extra={"model": "qwen2.5vl:7b"}),
            content_type="multipart/form-data",
        )
    assert res.get_json()["ocr_result"]["full_text"] == "FLYER"
    assert http.post.call_args.args[0] == f"{PROXY_URL}/v1/chat/completions"
    assert http.post.call_args.kwargs["json"]["max_tokens"] == 1000
    assert http.post.call_args.kwargs["json"]["temperature"] == 0.1
    image_url = http.post.call_args.kwargs["json"]["messages"][0]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/jpeg;base64,")
    assert http.post.call_args.kwargs["headers"]["X-API-Key"] == "k"
    assert rec.timeout == 300.0


def test_ocr_translates_a_router_model_id(client):
    body = {"choices": [{"message": {"content": "{}"}}]}
    http = _upstream(body)
    with _patched(http):
        client.post(
            "/api/vision/ocr",
            data=_ocr_upload(_png_bytes(), extra={"model": "qwen2.5-vl--q4_k_m"}),
            content_type="multipart/form-data",
        )
    assert http.post.call_args.kwargs["json"]["model"] == "qwen2.5-vl:q4_k_m"


def test_ocr_strips_a_fenced_json_block(client):
    raw = '```json\n{"headline": "NIGHT", "full_text": "NIGHT"}\n```'
    http = _upstream({"choices": [{"message": {"content": raw}}]})
    with _patched(http):
        res = client.post(
            "/api/vision/ocr",
            data=_ocr_upload(_png_bytes(), extra={"model": "m"}),
            content_type="multipart/form-data",
        )
    assert res.get_json()["ocr_result"]["headline"] == "NIGHT"
    assert res.get_json()["raw_response"] == raw


def test_ocr_falls_back_to_raw_text_when_the_json_is_bad(client):
    http = _upstream({"choices": [{"message": {"content": "just some prose"}}]})
    with _patched(http):
        res = client.post(
            "/api/vision/ocr",
            data=_ocr_upload(_png_bytes(), extra={"model": "m"}),
            content_type="multipart/form-data",
        )
    assert res.get_json()["ocr_result"]["full_text"] == "just some prose"


def test_ocr_502_on_an_empty_model_response(client):
    http = _upstream({"choices": [{"message": {"content": ""}}]})
    with _patched(http):
        res = client.post(
            "/api/vision/ocr",
            data=_ocr_upload(_png_bytes(), extra={"model": "m"}),
            content_type="multipart/form-data",
        )
    assert res.status_code == 502
    assert res.get_json()["error"] == "Vision OCR extraction failed"


def test_ocr_502_when_the_model_says_it_is_unsupported(client):
    """A text-only model answers prose instead of the JSON contract."""
    http = _upstream({"choices": [{"message": {"content": "image input is not supported"}}]})
    with _patched(http):
        res = client.post(
            "/api/vision/ocr",
            data=_ocr_upload(_png_bytes(), extra={"model": "m"}),
            content_type="multipart/form-data",
        )
    assert res.status_code == 502
    body = res.get_json()
    assert body["error"] == "Vision OCR extraction failed"
    # The upstream answered 200, so there is no HTTP status to quote - the
    # route says why in prose instead.
    assert "empty or unsupported" in body["details"]


def test_ocr_502_with_the_upstream_status_on_a_proxy_error(client):
    http = _upstream({}, status_code=500)
    http.post.return_value.text = "backend down"
    with _patched(http):
        res = client.post(
            "/api/vision/ocr",
            data=_ocr_upload(_png_bytes(), extra={"model": "m"}),
            content_type="multipart/form-data",
        )
    assert res.status_code == 502
    assert "HTTP 500" in res.get_json()["details"]


def test_ocr_400_on_a_pdf_it_cannot_open(client):
    with _patched(_upstream({})):
        res = client.post(
            "/api/vision/ocr",
            data=_ocr_upload(b"%PDF-1.4 not really a pdf", name="doc.pdf", extra={"model": "m"}),
            content_type="multipart/form-data",
        )
    assert res.status_code == 400
    assert "PDF processing error" in res.get_json()["error"]


def test_ocr_500_on_an_undecodable_image(client):
    with _patched(_upstream({})):
        res = client.post(
            "/api/vision/ocr",
            data=_ocr_upload(b"not an image", name="x.png", extra={"model": "m"}),
            content_type="multipart/form-data",
        )
    assert res.status_code == 500


# --------------------------------------------------------------------------
# /api/auth/status
# --------------------------------------------------------------------------


def test_auth_status_reports_local_and_authenticated(client):
    with patch("web.app.is_flask_client_local", return_value=True), patch(
        "web.app.is_flask_request_authenticated", return_value=True
    ), patch.dict("os.environ", {}, clear=False):
        os_env = {k: v for k, v in __import__("os").environ.items() if k not in ("ALPACA_API_KEY", "ADMIN_PASSWORD")}
        with patch.dict("os.environ", os_env, clear=True):
            body = client.get("/api/auth/status").get_json()
    assert body == {"authenticated": True, "is_local": True, "auth_required": False, "client_ip": body["client_ip"]}


def test_auth_status_says_auth_is_required_when_a_key_is_set(client):
    with patch.dict("os.environ", {"ALPACA_API_KEY": "secret"}, clear=True):
        body = client.get("/api/auth/status").get_json()
    assert body["auth_required"] is True
    # The key itself must never be echoed back.
    assert "secret" not in json.dumps(body)


def test_auth_status_falls_back_to_the_admin_password(client):
    with patch.dict("os.environ", {"ADMIN_PASSWORD": "pw"}, clear=True):
        assert client.get("/api/auth/status").get_json()["auth_required"] is True


def test_auth_status_is_reachable_without_a_session(client):
    """The login page calls it to show a network/origin badge."""
    with patch.dict("os.environ", {"ALPACA_API_KEY": "secret"}, clear=True):
        res = client.get("/api/auth/status")
    assert res.status_code == 200
    assert res.get_json()["authenticated"] is True  # the test client is local
