"""The edit-recipes catalogue, from the proxy recipe layer to the dashboard.

The point of this pair of routes is that the photo editor asks "what can I
change?" instead of hardcoding a model name. That is the whole fix for the
intuitiveness complaint, so these tests pin the two things that would quietly
undo it:

* the catalogue must describe *changes* and never name a model, and
* a proxy that is unreachable must say so -- returning an empty catalogue would
  render as "this build cannot edit photos", which is a different and wrong
  answer.
"""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PROXY_PY = REPO_ROOT / "alpaca-proxy.py"


def _load_proxy():
    """Load alpaca-proxy.py by path, stubbing the module-level AsyncMock seams."""
    name = "proxy_edit_recipes_test"
    spec = importlib.util.spec_from_file_location(name, PROXY_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    for attr in (
        "restart_llama_server",
        "wait_for_llama_server",
        "restore_slot_cache",
        "save_slot_cache",
        "find_slot_for_request",
    ):
        if hasattr(module, attr):
            setattr(module, attr, Mock())
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def proxy():
    return _load_proxy()


def _resp(payload, status_code=200):
    resp = Mock()
    resp.status_code = status_code
    resp.json.return_value = payload
    return resp


def test_the_proxy_route_serves_the_catalogue_and_the_model_capabilities(proxy):
    from fastapi.testclient import TestClient

    client = TestClient(proxy.app)
    resp = client.get("/v1/images/edit-recipes", params={"model": "Qwen-Image-2.1-GGUF/qwen_image_2.1"})
    assert resp.status_code == 200
    body = resp.json()

    # The catalogue is keyed on changes.
    assert body["recipes"], "no recipes served"
    assert {r["change"] for r in body["recipes"]} >= {"face", "outfit", "hair", "scene"}
    # And it must not name a model anywhere: that is the coupling being removed.
    blob = json.dumps(body["recipes"]).lower()
    for vendor in ("qwen", "sdxl", "flux", "stable-diffusion", "2.1"):
        assert vendor not in blob, f"the recipe catalogue names {vendor!r}; the panel must offer changes"

    assert body["model"] == "Qwen-Image-2.1-GGUF/qwen_image_2.1"
    assert body["capabilities"]["reference_images"] is True
    assert body["capabilities"]["max_reference_images"] == 4


def test_the_proxy_route_works_with_no_model_chosen(proxy):
    """The panel must be able to draw itself before anything is loaded."""
    from fastapi.testclient import TestClient

    body = TestClient(proxy.app).get("/v1/images/edit-recipes").json()
    assert body["recipes"], "the catalogue must not depend on a model being chosen"
    assert body["model"] == ""
    assert body["capabilities"] is None


def test_the_proxy_route_reports_what_a_single_image_model_cannot_do(proxy):
    """References unsupported must be visible before the user spends the upload."""
    from fastapi.testclient import TestClient

    body = TestClient(proxy.app).get("/v1/images/edit-recipes", params={"model": "stable-diffusion-xl-base-1.0"}).json()
    assert body["capabilities"]["reference_images"] is False
    assert body["capabilities"]["max_reference_images"] == 0
    # ...but the text-only recipes are still there. Refusing everything would be
    # the unintuitive behaviour this feature exists to remove.
    assert all(r["min_images"] == 1 for r in body["recipes"])


def test_the_legacy_preset_is_listed_separately_and_not_offered_as_a_recipe(proxy):
    from fastapi.testclient import TestClient

    body = TestClient(proxy.app).get("/v1/images/edit-recipes").json()
    assert "qwen_image_21.identity" in body["legacy_presets"]
    assert "qwen_image_21.identity" not in {r["id"] for r in body["recipes"]}


def test_each_recipe_carries_the_instruction_the_panel_shows_the_user(proxy):
    """Transparency is the point: the panel shows what it is going to ask for."""
    from fastapi.testclient import TestClient

    for recipe in TestClient(proxy.app).get("/v1/images/edit-recipes").json()["recipes"]:
        assert recipe["instruction"].strip(), f"{recipe['id']} has no instruction to preview"
        assert "<image1>" in recipe["instruction"], f"{recipe['id']} does not refer to the photo it edits"


def test_the_web_route_forwards_the_model_and_passes_the_body_through():
    from web.app import PROXY_URL, app

    app.config["TESTING"] = True
    payload = {"recipes": [{"id": "edit.face"}], "capabilities": {"reference_images": True}}
    with app.test_client() as client, patch("web.app.httpx.Client") as http_cls:
        ctx = Mock()
        ctx.__enter__ = Mock(return_value=ctx)
        ctx.__exit__ = Mock(return_value=False)
        http_cls.return_value = ctx
        ctx.get.return_value = _resp(payload)

        resp = client.get("/api/sd/edit-recipes?model=Qwen-Image-2.1-GGUF/qwen_image_2.1")

    assert resp.status_code == 200
    assert resp.get_json() == payload, "the dashboard must render the proxy's body, not a reshaped one"
    url, kwargs = ctx.get.call_args
    assert url[0] == f"{PROXY_URL}/v1/images/edit-recipes"
    assert kwargs["params"] == {"model": "Qwen-Image-2.1-GGUF/qwen_image_2.1"}


def test_the_web_route_omits_the_param_entirely_when_no_model_is_chosen():
    """Sending model="" would look like a request for a model literally named ""."""
    from web.app import app

    app.config["TESTING"] = True
    with app.test_client() as client, patch("web.app.httpx.Client") as http_cls:
        ctx = Mock()
        ctx.__enter__ = Mock(return_value=ctx)
        ctx.__exit__ = Mock(return_value=False)
        http_cls.return_value = ctx
        ctx.get.return_value = _resp({"recipes": []})

        resp = client.get("/api/sd/edit-recipes")

    assert resp.status_code == 200
    _, kwargs = ctx.get.call_args
    assert kwargs["params"] is None


def test_a_dead_proxy_does_not_look_like_a_build_with_no_edits():
    """An empty catalogue would render as "this cannot edit photos". Say what broke."""
    from web.app import app

    app.config["TESTING"] = False
    with app.test_client() as client, patch("web.app.httpx.Client", side_effect=OSError("connection refused")):
        resp = client.get("/api/sd/edit-recipes")

    assert resp.status_code == 500
    body = resp.get_json()
    assert "error" in body and body["recipes"] == []


def test_a_non_json_proxy_body_is_an_error_not_a_silent_empty_catalogue():
    from web.app import app

    app.config["TESTING"] = True
    with app.test_client() as client, patch("web.app.httpx.Client") as http_cls:
        ctx = Mock()
        ctx.__enter__ = Mock(return_value=ctx)
        ctx.__exit__ = Mock(return_value=False)
        http_cls.return_value = ctx
        ctx.get.return_value = Mock(status_code=502, json=Mock(side_effect=ValueError("not json")))

        resp = client.get("/api/sd/edit-recipes")

    assert resp.status_code == 500
    assert resp.get_json()["recipes"] == []
