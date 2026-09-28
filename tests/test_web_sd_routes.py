"""Tests for the web layer's /api/sd/* bridge to the proxy's image routes.

The Image Studio is 6 panels wide and routes everything through four of these
endpoints, yet none of the seven `/api/sd/*` routes had any coverage.

Two behaviours are easy to break and worth pinning hard:

  * **`qr_*` fields are web-only.** `qr_text` / `qr_url` / `qr_position` / `qr_label`
    are burned into the image here, in PIL, and must be stripped before the payload
    reaches the proxy — otherwise every flyer request carries four fields sd-server
    has no idea what to do with.
  * **`capabilities` failing is a 503, not a 500.** The Image Studio's advanced
    drawer is populated from it, and a 500 would be indistinguishable from "the
    engine has no samplers", which is a different message with a different fix.
"""

import base64
import io
import os
from contextlib import contextmanager
from unittest.mock import Mock, patch

import pytest
from werkzeug.datastructures import MultiDict

from web.app import PROXY_URL, app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


class _Rec:
    """What the route handed to httpx.Client: the timeout it asked for."""

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
    """A mock client whose every verb returns `payload` with `status_code`."""
    resp = Mock()
    resp.status_code = status_code
    resp.json.return_value = payload
    http = Mock()
    for verb in ("get", "post", "patch", "delete", "put"):
        getattr(http, verb).return_value = resp
    return http


def _dead(side_effect):
    """A mock client whose every verb raises, simulating an unreachable service."""
    http = Mock()
    for verb in ("get", "post", "patch", "delete", "put"):
        getattr(http, verb).side_effect = side_effect
    return http


@contextmanager
def _patched(http):
    """Patch httpx.Client so `with httpx.Client(timeout=...) as c` yields `http`."""
    rec = _Rec()

    def _ctor(*args, **kwargs):
        rec.timeout = kwargs.get("timeout", args[0] if args else None)
        return _Ctx(http)

    with patch("web.app.httpx.Client", side_effect=_ctor):
        yield rec


# --- GET /api/sd/models -------------------------------------------------------


def test_sd_models_lists_local_image_models(client):
    payload = {"data": [{"name": "Qwen-Image-2.1-GGUF/qwen_image_2.1-Q4_K", "family": "qwen-image"}]}
    http = _upstream(payload)
    with _patched(http):
        res = client.get("/api/sd/models")

    assert res.status_code == 200
    assert res.get_json()["data"][0]["family"] == "qwen-image"
    assert http.get.call_args.args[0] == f"{PROXY_URL}/v1/images/models"


def test_sd_models_sends_the_proxy_auth_headers(client):
    """The proxy sits behind API-key auth; omitting the header would 401 the drawer."""
    http = _upstream({"data": []})
    with _patched(http), patch.dict(os.environ, {"ALPACA_API_KEY": "k"}):
        client.get("/api/sd/models")

    headers = http.get.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer k"
    assert headers["X-API-Key"] == "k"


def test_sd_models_reports_500_with_an_empty_list_when_unreachable(client):
    """The Studio renders `data: []` as "no models"; a missing key would throw in JS."""
    with _patched(_dead(OSError("proxy down"))):
        res = client.get("/api/sd/models")

    assert res.status_code == 500
    body = res.get_json()
    assert body["data"] == []
    assert "proxy down" in body["error"]


# --- GET /api/sd/capabilities -------------------------------------------------


def test_sd_capabilities_returns_samplers_and_schedulers(client):
    payload = {"samplers": ["euler", "dpmpp_2m"], "schedulers": ["karras"], "upscalers": ["esrgan"]}
    http = _upstream(payload)
    with _patched(http):
        res = client.get("/api/sd/capabilities")

    assert res.status_code == 200
    assert res.get_json()["samplers"] == ["euler", "dpmpp_2m"]
    assert http.get.call_args.args[0] == f"{PROXY_URL}/v1/images/capabilities"


def test_sd_capabilities_uses_a_short_timeout(client):
    """It is fetched while the drawer opens; a slow call would leave the drawer empty."""
    http = _upstream({})
    with _patched(http) as rec:
        client.get("/api/sd/capabilities")

    assert rec.timeout == 5.0


def test_sd_capabilities_returns_503_with_native_false_when_unreachable(client):
    """503 + `native: False` is the signal the drawer uses to show the legacy
    (flattened) parameter path instead of pretending the engine has no samplers."""
    with _patched(_dead(OSError("no route"))):
        res = client.get("/api/sd/capabilities")

    assert res.status_code == 503
    body = res.get_json()
    assert body["native"] is False
    assert "no route" in body["error"]


def test_sd_capabilities_propagates_the_proxies_own_503(client):
    """A proxy too old to parse <sd_cpp_extra_args> answers 503; that must not
    be re-wrapped, or the client cannot tell an old proxy from an offline one."""
    http = _upstream({"error": "native sd_cpp_extra_args unsupported"}, status_code=503)
    with _patched(http):
        res = client.get("/api/sd/capabilities")

    assert res.status_code == 503
    assert "unsupported" in res.get_json()["error"]


# --- GET /api/sd/presets ------------------------------------------------------


def test_sd_presets_returns_the_flyer_presets(client):
    payload = {"flyer_design": {"music_event": {"size": "832x1216"}}}
    http = _upstream(payload)
    with _patched(http):
        res = client.get("/api/sd/presets")

    assert res.status_code == 200
    assert "music_event" in res.get_json()["flyer_design"]
    assert http.get.call_args.args[0] == f"{PROXY_URL}/v1/images/presets"


def test_sd_presets_reports_500_when_unreachable(client):
    with _patched(_dead(OSError("no route"))):
        res = client.get("/api/sd/presets")

    assert res.status_code == 500


# --- POST /api/sd/load --------------------------------------------------------


def test_sd_load_preloads_a_model(client):
    http = _upstream({"status": "loading", "model": "qwen-image"})
    with _patched(http):
        res = client.post("/api/sd/load", json={"model": "Qwen-Image-2.1-GGUF/qwen_image_2.1-Q4_K"})

    assert res.status_code == 200
    assert res.get_json()["model"] == "qwen-image"
    assert http.post.call_args.args[0] == f"{PROXY_URL}/v1/images/models/load"
    assert http.post.call_args.kwargs["json"]["model"].startswith("Qwen-Image-2.1")


def test_sd_load_allows_time_for_the_container_to_swap(client):
    """Loading evicts the LLM and waits for sd-server health, so 120s is not arbitrary."""
    http = _upstream({})
    with _patched(http) as rec:
        client.post("/api/sd/load", json={"model": "m"})

    assert rec.timeout == 120.0


def test_sd_load_propagates_the_llm_manifest_guardrail(client):
    """The proxy rejects an LLM manifest with 400; the Studio shows that verbatim."""
    http = _upstream({"error": "not an image model manifest"}, status_code=400)
    with _patched(http):
        res = client.post("/api/sd/load", json={"model": "qwen3-35b"})

    assert res.status_code == 400
    assert "image model manifest" in res.get_json()["error"]


def test_sd_load_reports_500_when_unreachable(client):
    with _patched(_dead(OSError("no route"))):
        res = client.post("/api/sd/load", json={"model": "m"})

    assert res.status_code == 500


# --- POST /api/sd/generate ----------------------------------------------------


def test_sd_generate_forwards_the_prompt_and_options(client):
    body = {"model": "m", "prompt": "a lit street", "size": "832x1216", "n": 1, "steps": 25}
    http = _upstream({"data": [{"b64_json": _png_b64()}]})
    with _patched(http):
        res = client.post("/api/sd/generate", json=body)

    assert res.status_code == 200
    assert http.post.call_args.args[0] == f"{PROXY_URL}/v1/images/generations"
    assert http.post.call_args.kwargs["json"] == body


def test_sd_generate_strips_the_web_only_qr_fields(client):
    """qr_* are burned in here; the proxy must never see them."""
    body = {"model": "m", "prompt": "flyer", "qr_text": "https://x.test", "qr_position": "top_right", "qr_label": "HI"}
    http = _upstream({"data": [{"b64_json": _png_b64()}]})
    with _patched(http), patch("web.app.embed_qr_code_onto_image", return_value=_png_b64()):
        client.post("/api/sd/generate", json=body)

    forwarded = http.post.call_args.kwargs["json"]
    assert forwarded == {"model": "m", "prompt": "flyer"}


def test_sd_generate_accepts_qr_url_as_an_alias_for_qr_text(client):
    http = _upstream({"data": [{"b64_json": _png_b64()}]})
    with _patched(http), patch("web.app.embed_qr_code_onto_image", return_value=_png_b64()) as embed:
        client.post("/api/sd/generate", json={"prompt": "p", "qr_url": "https://x.test"})

    assert embed.call_args.args[1] == "https://x.test"


def test_sd_generate_burns_the_qr_into_every_returned_image(client):
    """A 2-image flyer needs the badge on both, or the second one ships unbranded."""
    http = _upstream({"data": [{"b64_json": _png_b64()}, {"b64_json": _png_b64(w=32, h=32)}]})
    with _patched(http), patch("web.app.embed_qr_code_onto_image", return_value=_png_b64(color=(1, 2, 3))) as embed:
        res = client.post("/api/sd/generate", json={"prompt": "p", "qr_text": "https://x.test"})

    assert embed.call_count == 2
    assert all(item["b64_json"] == _png_b64(color=(1, 2, 3)) for item in res.get_json()["data"])


def test_sd_generate_does_not_embed_when_the_upstream_failed(client):
    """A 500 from the proxy must not be papered over with a 200 and a QR."""
    http = _upstream({"error": "VRAM exhausted"}, status_code=500)
    with _patched(http), patch("web.app.embed_qr_code_onto_image") as embed:
        res = client.post("/api/sd/generate", json={"prompt": "p", "qr_text": "x"})

    assert res.status_code == 500
    embed.assert_not_called()


def test_sd_generate_tolerates_a_url_only_response(client):
    """`response_format=url` yields no b64_json; the QR loop must not KeyError."""
    http = _upstream({"data": [{"url": "http://x/y.png"}]})
    with _patched(http), patch("web.app.embed_qr_code_onto_image") as embed:
        res = client.post("/api/sd/generate", json={"prompt": "p", "qr_text": "x"})

    assert res.status_code == 200
    embed.assert_not_called()
    assert res.get_json()["data"][0]["url"] == "http://x/y.png"


def test_sd_generate_uses_a_very_long_read_timeout_with_a_short_connect(client):
    """Qwen image gen on the 4060 routinely exceeds 10 minutes; the connect budget
    stays short so a dead container fails fast instead of holding the socket 30 min."""
    http = _upstream({"data": []})
    with _patched(http) as rec:
        client.post("/api/sd/generate", json={"prompt": "p"})

    timeout = rec.timeout
    assert timeout.read == 1800.0
    assert timeout.connect == 30.0


def test_sd_generate_propagates_validation_errors(client):
    http = _upstream({"error": "size must be <= 2048x2048"}, status_code=400)
    with _patched(http):
        res = client.post("/api/sd/generate", json={"prompt": "p", "size": "4096x4096"})

    assert res.status_code == 400
    assert "2048x2048" in res.get_json()["error"]


def test_sd_generate_reports_500_when_unreachable(client):
    with _patched(_dead(OSError("no route"))):
        res = client.post("/api/sd/generate", json={"prompt": "p"})

    assert res.status_code == 500
    assert "no route" in res.get_json()["error"]


# --- POST /api/sd/edit --------------------------------------------------------


def _png_b64(w=64, h=48, color=(200, 40, 40)):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _png_upload(name="photo.png"):
    """A Werkzeug multipart file part: (file-object, filename)."""
    return (io.BytesIO(base64.b64decode(_png_b64())), name)


def test_sd_edit_forwards_multipart_files_and_form_fields(client):
    http = _upstream({"data": [{"b64_json": _png_b64()}]})
    with _patched(http):
        res = client.post(
            "/api/sd/edit",
            data=MultiDict([("prompt", "make it night"), ("strength", "0.6"), ("image", _png_upload())]),
            content_type="multipart/form-data",
        )

    assert res.status_code == 200
    assert http.post.call_args.args[0] == f"{PROXY_URL}/v1/images/edits"
    assert http.post.call_args.kwargs["data"]["prompt"] == "make it night"
    key, (fname, content, mimetype) = http.post.call_args.kwargs["files"][0]
    assert key == "image"
    assert fname == "photo.png"
    assert mimetype == "image/png"
    assert content


def test_sd_edit_collects_repeated_image_keys(client):
    """Multi-image Qwen edits send `image[]` repeatedly; only the first must survive
    or the proxy would silently edit one of the references."""
    http = _upstream({"data": []})
    with _patched(http):
        client.post(
            "/api/sd/edit",
            data=MultiDict([("prompt", "merge"), ("image[]", _png_upload("a.png")), ("image[]", _png_upload("b.png"))]),
            content_type="multipart/form-data",
        )

    files = http.post.call_args.kwargs["files"]
    assert len(files) == 2
    assert [f[1][0] for f in files] == ["a.png", "b.png"]


def test_sd_edit_keeps_repeated_form_fields_as_a_list(client):
    http = _upstream({"data": []})
    with _patched(http):
        client.post(
            "/api/sd/edit",
            data=MultiDict(
                [
                    ("prompt", "p"),
                    ("image_role", "subject"),
                    ("image_role", "scene"),
                    ("image", _png_upload()),
                ]
            ),
            content_type="multipart/form-data",
        )

    assert http.post.call_args.kwargs["data"]["image_role"] == ["subject", "scene"]


def test_sd_edit_strips_the_web_only_qr_fields(client):
    http = _upstream({"data": [{"b64_json": _png_b64()}]})
    with _patched(http), patch("web.app.embed_qr_code_onto_image", return_value=_png_b64()):
        client.post(
            "/api/sd/edit",
            data=MultiDict([("prompt", "p"), ("qr_text", "https://x.test"), ("image", _png_upload())]),
            content_type="multipart/form-data",
        )

    assert "qr_text" not in http.post.call_args.kwargs["data"]


def test_sd_edit_falls_back_to_an_empty_files_mapping(client):
    """A form-only edit is legal; sending `files={}` beats sending a falsy list."""
    http = _upstream({"data": []})
    with _patched(http):
        client.post("/api/sd/edit", data=MultiDict([("prompt", "p")]), content_type="multipart/form-data")

    assert http.post.call_args.kwargs["files"] == {}


def test_sd_edit_uses_the_long_generation_timeout(client):
    """Multi-image Qwen edits can run 15+ minutes on the 4060."""
    http = _upstream({"data": []})
    with _patched(http) as rec:
        client.post(
            "/api/sd/edit",
            data=MultiDict([("prompt", "p"), ("image", _png_upload())]),
            content_type="multipart/form-data",
        )

    assert rec.timeout.read == 1800.0


def test_sd_edit_propagates_the_four_image_cap(client):
    http = _upstream({"error": "at most 4 images"}, status_code=413)
    with _patched(http):
        res = client.post(
            "/api/sd/edit",
            data=MultiDict([("prompt", "p")] + [("image[]", _png_upload(f"{i}.png")) for i in range(5)]),
            content_type="multipart/form-data",
        )

    assert res.status_code == 413


def test_sd_edit_reports_500_when_unreachable(client):
    with _patched(_dead(OSError("no route"))):
        res = client.post(
            "/api/sd/edit",
            data=MultiDict([("prompt", "p"), ("image", _png_upload())]),
            content_type="multipart/form-data",
        )

    assert res.status_code == 500


# --- embed_qr_code_onto_image -------------------------------------------------


@pytest.mark.parametrize("position", ["bottom_right", "bottom_left", "bottom_center", "top_right", "nonsense"])
def test_qr_embedding_supports_every_advertised_position(position):
    from web.app import embed_qr_code_onto_image

    out = embed_qr_code_onto_image(_png_b64(200, 300), "https://x.test", position=position, label="SCAN")
    assert out != _png_b64(200, 300)

    from PIL import Image

    img = Image.open(io.BytesIO(base64.b64decode(out)))
    # The badge is pasted onto a card, so the result is a JPEG re-encode of the flyer.
    assert img.format == "JPEG"
    assert img.size == (200, 300)


def test_qr_embedding_returns_the_input_untouched_on_undecodable_data():
    """A bad base64 body must not blank the whole flyer."""
    from web.app import embed_qr_code_onto_image

    assert embed_qr_code_onto_image("not-base64!!", "https://x.test") == "not-base64!!"


def test_qr_embedding_is_a_noop_when_qrcode_is_missing():
    """The web image is slimmed down; without the qrcode package the flyer must
    still come back, just without a badge."""
    from web.app import embed_qr_code_onto_image

    original = _png_b64()
    with patch.dict("sys.modules", {"qrcode": None}):
        assert embed_qr_code_onto_image(original, "https://x.test") == original


# --- GET /api/companions ------------------------------------------------------


def test_companions_lists_only_model_files_from_every_directory(client, tmp_path):
    router = tmp_path / "router"
    models = tmp_path / "models"
    for d in (router / "companions", models / "companions"):
        d.mkdir(parents=True)
    (router / "companions" / "qwen_image_2.1_vae_bf16.safetensors").write_bytes(b"x")
    (models / "companions" / "Qwen3VL-8B-Instruct-Q4_K_M.gguf").write_bytes(b"x")
    # Not a companion asset, and not a model file - both must be ignored.
    (models / "companions" / "README.md").write_text("nope")
    (models / "companions" / "mmproj-Qwen3VL-8B-Instruct-F16.gguf").write_bytes(b"x")

    with patch.dict(os.environ, {"ROUTER_MODELS_DIR": str(router), "MODELS_DIR": str(models)}):
        res = client.get("/api/companions")

    companions = res.get_json()["companions"]
    assert "qwen_image_2.1_vae_bf16.safetensors" in companions
    assert "Qwen3VL-8B-Instruct-Q4_K_M.gguf" in companions
    assert "README.md" not in companions


def test_companions_deduplicates_across_directories(client, tmp_path):
    """The same file is visible under both ROUTER_MODELS_DIR and MODELS_DIR, and
    several hard-coded fallbacks are probed too; it must be listed once."""
    for name in ("router", "models"):
        d = tmp_path / name / "companions"
        d.mkdir(parents=True)
        (d / "t5xxl.safetensors").write_bytes(b"x")

    with patch.dict(os.environ, {"ROUTER_MODELS_DIR": str(tmp_path / "router"), "MODELS_DIR": str(tmp_path / "models")}):
        companions = client.get("/api/companions").get_json()["companions"]

    assert companions.count("t5xxl.safetensors") == 1


def test_companions_is_empty_not_broken_when_no_directory_exists(client, tmp_path):
    with patch.dict(os.environ, {"ROUTER_MODELS_DIR": str(tmp_path / "nope"), "MODELS_DIR": str(tmp_path / "nope2")}):
        res = client.get("/api/companions")

    assert res.status_code == 200
    assert res.get_json()["companions"] == []


# --- cross-route invariants ---------------------------------------------------


def test_sd_routes_never_address_the_sd_server_directly(client):
    """sd-server is a sibling of the proxy; only the proxy knows about VRAM
    arbitration, so the Studio must not bypass it."""
    for route in ("/api/sd/models", "/api/sd/capabilities", "/api/sd/presets"):
        http = _upstream({})
        with _patched(http):
            client.get(route)

        assert ":8081" not in http.get.call_args.args[0]
        assert "/admin/sd/health" in http.get.call_args.args[0] or http.get.call_args.args[0].startswith(
            f"{PROXY_URL}/v1/images"
        )


def test_sd_unload_goes_through_the_proxy_sd_admin_route(client):
    http = _upstream({"status": "unloaded"})
    with _patched(http) as rec:
        res = client.post("/api/sd/unload", json={})

    assert res.status_code == 200
    assert http.post.call_args.args[0] == f"{PROXY_URL}/admin/sd/unload"
    # Eviction takes the backend swap lock, but it must not hold the socket 5s.
    assert rec.timeout == 5.0


def test_sd_health_passes_the_proxy_payload_through(client):
    http = _upstream({"active_model": "qwen-image", "sd_server_healthy": True})
    with _patched(http):
        res = client.get("/api/sd/health")

    assert res.status_code == 200
    assert res.get_json()["active_model"] == "qwen-image"
    assert http.get.call_args.args[0] == f"{PROXY_URL}/admin/sd/health"


def test_sd_status_reports_offline_rather_than_erroring(client):
    """/api/sd/status is what the System Monitor card polls; it must always answer."""
    with _patched(_dead(OSError("proxy down"))):
        res = client.get("/api/sd/status")

    assert res.status_code == 200
    body = res.get_json()
    assert body["online"] is False
    assert "proxy down" in body["error"]


def test_sd_get_routes_reject_post(client):
    for route in ("/api/sd/models", "/api/sd/capabilities", "/api/sd/presets", "/api/sd/status"):
        assert client.post(route, json={}).status_code == 405
