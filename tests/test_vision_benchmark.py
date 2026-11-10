"""The vision category: image attachments reach the model, and a model that
cannot see is skipped rather than failed."""

import base64
import json
import sys
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from llm_benchmark_suite import LLMModelBenchmark, _skipped_result, _test_images

REPO = Path(__file__).resolve().parents[1]
VISION = json.loads((REPO / "benchmark_tests.json").read_text())["vision"]


def _chat_ndjson(text: str) -> bytes:
    lines = [
        json.dumps({"message": {"role": "assistant", "content": text}, "done": False}),
        json.dumps(
            {
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "eval_count": 3,
                "eval_duration": 100,
                "prompt_eval_duration": 50,
            }
        ),
    ]
    return ("\n".join(lines) + "\n").encode()


# ---------- the fixtures ----------


def test_every_vision_test_carries_one_real_png_and_a_key():
    assert len(VISION) == 8
    for t in VISION:
        (att,) = t["attachments"]
        raw = base64.b64decode(att["data_base64"])
        assert raw[:8] == b"\x89PNG\r\n\x1a\n", t["id"]
        assert t["expected"] and t["id"].startswith("vision_")
        assert len(raw) < 32_000  # flat drawings; keeps benchmark_tests.json small


def test_the_shipped_vision_category_is_what_the_generator_draws():
    """The images and their answer keys come from one place; editing either by
    hand would leave a key that no longer describes its image."""
    sys.path.insert(0, str(REPO / "scripts"))
    import gen_vision_fixtures

    assert gen_vision_fixtures.build() == VISION


def test_test_images_reads_only_image_attachments():
    t = {
        "attachments": [
            {"name": "a.png", "mime": "image/png", "data_base64": "AAA"},
            {"name": "notes.txt", "text": "hello"},
            {"name": "b.JPG", "data_base64": "BBB"},
        ]
    }
    assert _test_images(t) == ["AAA", "BBB"]
    assert _test_images({"prompt": "no attachments"}) == []


# ---------- the request ----------


def _benchmark(tmp_path, monkeypatch) -> LLMModelBenchmark:
    ini = tmp_path / "models.ini"
    ini.write_text("[*]\ntemperature = 0.1\n")
    monkeypatch.setenv("MODELS_INI_PATH", str(ini))
    return LLMModelBenchmark()


async def _run(benchmark, handler, coro_factory):
    """Await INSIDE the patch: returning the coroutine would run it after the
    patch ended, against the real client (and the no-network guard)."""
    real_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        kwargs.pop("timeout", None)
        return real_client(transport=httpx.MockTransport(handler))

    with patch("httpx.AsyncClient", side_effect=client_factory):
        return await coro_factory()


def _runtime_or(handler):
    def wrapped(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/admin/runtime":
            # A loaded model with a known context, so the context guard resolves
            # n_ctx instead of waiting for a load.
            model = {"name": "vlm--latest", "backend_model": "vlm--latest", "running_settings": {"ctx-size": "65536"}}
            return httpx.Response(200, json={"loaded_models": [model]})
        return handler(request)

    return wrapped


@pytest.mark.asyncio
async def test_the_image_rides_in_the_user_message(tmp_path, monkeypatch):
    benchmark = _benchmark(tmp_path, monkeypatch)
    test = next(t for t in VISION if t["id"] == "vision_grid_cell")
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, content=_chat_ndjson("8"))

    res = await _run(benchmark, _runtime_or(handler), lambda: benchmark.test_model_proxy("vlm", test))
    assert res["success"] is True
    (msg,) = seen[0]["messages"]
    assert msg["images"] == [test["attachments"][0]["data_base64"]]
    assert msg["content"] == test["prompt"]


@pytest.mark.asyncio
async def test_a_model_without_a_projector_is_skipped_not_failed(tmp_path, monkeypatch):
    benchmark = _benchmark(tmp_path, monkeypatch)
    test = VISION[0]

    def handler(request):
        return httpx.Response(
            500,
            text='{"error":"image input is not supported - hint: if this is unexpected, provide the mmproj"}',
        )

    res = await _run(benchmark, _runtime_or(handler), lambda: benchmark.test_model_proxy("vlm", test))
    assert res["skipped"] is True and res["success"] is False
    assert "no vision support" in res["error"]


@pytest.mark.asyncio
async def test_a_server_error_unrelated_to_images_is_still_a_failure(tmp_path, monkeypatch):
    benchmark = _benchmark(tmp_path, monkeypatch)

    def handler(request):
        return httpx.Response(503, text='{"error":"queue_timeout"}')

    res = await _run(benchmark, _runtime_or(handler), lambda: benchmark.test_model_proxy("vlm", VISION[0]))
    assert not res.get("skipped")
    assert "HTTP 503" in res["error"]


@pytest.mark.asyncio
async def test_the_direct_path_and_online_providers_skip_images(tmp_path, monkeypatch):
    benchmark = _benchmark(tmp_path, monkeypatch)
    res = await benchmark.test_model_direct("any-model", VISION[0])
    assert res["skipped"] is True
    online = await benchmark.test_model_proxy("openrouter:some/model", VISION[0])
    assert online["skipped"] is True


# ---------- scoring ----------


def test_skipped_results_do_not_count_toward_a_category():
    benchmark = LLMModelBenchmark()
    passed = {"success": True, "score": 100, "tokens_generated": 1, "latency": 1}
    stats = benchmark._calculate_category_stats([passed, _skipped_result(VISION[0], "no vision")])
    assert stats["tests_run"] == 1 and stats["tests_passed"] == 1 and stats["score"] == 100
    assert len(stats["tests"]) == 2  # the skip is still recorded, just not scored


def test_per_model_file_reports_the_latest_run_date(tmp_path):
    """Its first date was kept forever, so a run tonight showed up as August."""
    benchmark = LLMModelBenchmark()
    benchmark.MODELS_DIR = tmp_path
    block = {"category_logic": {"tests": [{"test_id": "logic_knights", "success": True, "score": 100}]}}
    first = benchmark.save_per_model_result({"model": "m", **block}, "all", True, "2026-08-20T07:42:40")
    benchmark.save_per_model_result({"model": "m", **block}, "all", True, "2026-11-09T17:23:17")
    data = json.loads(first.read_text())
    assert data["generated_at"] == "2026-11-09T17:23:17"
    assert data["first_generated_at"] == "2026-08-20T07:42:40"
