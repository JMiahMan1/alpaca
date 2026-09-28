"""Unit tests for telemetry_monitor.py - the async daemon that has zero tests.

AGENTS.md documents a very specific contract for this module (skip /slots when
backend_model is None, sanitise the public name to match the web app, poll on a
5 s interval). Those rules are all asserted here because a silent violation
produces telemetry that *looks* fine and is quietly wrong.
"""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import telemetry_monitor as tm

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


class FakeClient:
    """Records every (url, kwargs) so the routing rules can be asserted."""

    def __init__(self, routes: dict):
        self.routes = routes
        self.calls: list[tuple[str, dict]] = []

    async def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        for frag, resp in self.routes.items():
            if frag in url:
                return resp(url) if callable(resp) else resp
        raise AssertionError(f"unexpected url: {url}")


def _proc(stdout: str = "", returncode: int = 0):
    proc = Mock()
    proc.communicate = AsyncMock(return_value=(stdout.encode(), b""))
    proc.returncode = returncode
    return proc


def _subprocess_seq(*procs_or_excs):
    """Drive asyncio.create_subprocess_exec with an ordered script of outcomes."""
    calls = []

    async def _exec(*cmd, **kwargs):
        calls.append(cmd)
        nxt = procs_or_excs[len(calls) - 1]
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    return _exec, calls


@pytest.fixture
def telemetry_dir(tmp_path, monkeypatch):
    d = tmp_path / "telemetry"
    monkeypatch.setattr(tm, "TELEMETRY_DIR", d)
    return d


# --------------------------------------------------------------------------- #
# get_system_metrics                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_system_metrics_report_gibibytes_and_percentages():
    mem = SimpleNamespace(total=32 * 1024**3, used=12 * 1024**3, available=20 * 1024**3, percent=37.5)
    with patch("psutil.virtual_memory", return_value=mem), patch("psutil.cpu_percent", return_value=12.5):
        m = await tm.get_system_metrics()
    assert m == {
        "ram_total_gb": 32.0,
        "ram_used_gb": 12.0,
        "ram_free_gb": 20.0,
        "ram_used_pct": 37.5,
        "cpu_util_pct": 12.5,
    }


@pytest.mark.asyncio
async def test_system_metrics_degrade_to_zeros_instead_of_killing_the_daemon():
    with patch("psutil.virtual_memory", side_effect=OSError("no /proc")):
        assert await tm.get_system_metrics() == {
            "ram_total_gb": 0.0,
            "ram_used_gb": 0.0,
            "ram_free_gb": 0.0,
            "ram_used_pct": 0.0,
            "cpu_util_pct": 0.0,
        }


# --------------------------------------------------------------------------- #
# get_gpu_metrics                                                              #
# --------------------------------------------------------------------------- #

NVIDIA_CSV = "0, NVIDIA GeForce RTX 4060, 8192, 4096, 4096, 37\n"


@pytest.mark.asyncio
async def test_gpu_metrics_prefer_docker_exec_and_do_not_fall_back_when_it_succeeds():
    _exec, calls = _subprocess_seq(_proc(NVIDIA_CSV))
    with patch("asyncio.create_subprocess_exec", _exec):
        gpus = await tm.get_gpu_metrics()
    assert len(calls) == 1 and calls[0][:3] == ("docker", "exec", tm.DOCKER_CONTAINER)
    assert gpus[0] == {
        "index": 0,
        "name": "NVIDIA GeForce RTX 4060",
        "vram_total_mb": 8192,
        "vram_used_mb": 4096,
        "vram_free_mb": 4096,
        "vram_used_pct": 50.0,
        "gpu_util_pct": 37.0,
    }


@pytest.mark.asyncio
async def test_gpu_metrics_fall_back_to_the_host_when_docker_exec_raises():
    _exec, calls = _subprocess_seq(FileNotFoundError("no docker"), _proc(NVIDIA_CSV))
    with patch("asyncio.create_subprocess_exec", _exec):
        gpus = await tm.get_gpu_metrics()
    assert len(calls) == 2 and calls[1][0] == "nvidia-smi"
    assert gpus[0]["name"] == "NVIDIA GeForce RTX 4060"


@pytest.mark.asyncio
async def test_gpu_metrics_fall_back_when_docker_exec_returns_nonzero():
    _exec, _ = _subprocess_seq(_proc("", returncode=126), _proc(NVIDIA_CSV))
    with patch("asyncio.create_subprocess_exec", _exec):
        gpus = await tm.get_gpu_metrics()
    assert gpus[0]["vram_total_mb"] == 8192


@pytest.mark.asyncio
async def test_gpu_metrics_emit_a_synthetic_row_when_nvidia_smi_is_absent():
    _exec, _ = _subprocess_seq(FileNotFoundError("a"), FileNotFoundError("b"))
    with patch("asyncio.create_subprocess_exec", _exec):
        gpus = await tm.get_gpu_metrics()
    # A missing GPU is reported as data, not as an exception, so one broken probe
    # cannot stop the daemon writing the other two metric families.
    assert gpus == [
        {
            "index": 0,
            "name": "Unknown GPU (Unavailable)",
            "vram_total_mb": 0,
            "vram_used_mb": 0,
            "vram_free_mb": 0,
            "vram_used_pct": 0.0,
            "gpu_util_pct": 0.0,
        }
    ]


@pytest.mark.asyncio
async def test_gpu_metrics_parse_every_gpu_in_a_multi_gpu_csv():
    csv = "0, A, 8192, 4096, 4096, 10\n1, B, 16384, 0, 16384, 0\n"
    _exec, _ = _subprocess_seq(_proc(csv))
    with patch("asyncio.create_subprocess_exec", _exec):
        gpus = await tm.get_gpu_metrics()
    assert [g["name"] for g in gpus] == ["A", "B"]
    assert gpus[1]["vram_used_pct"] == 0.0 and gpus[1]["index"] == 1


@pytest.mark.asyncio
async def test_gpu_metrics_treat_n_a_as_zero_rather_than_crashing():
    _exec, _ = _subprocess_seq(_proc("0, A, [N/A], [N/A], [N/A], [N/A]\n"))
    with patch("asyncio.create_subprocess_exec", _exec):
        gpus = await tm.get_gpu_metrics()
    assert gpus[0]["vram_total_mb"] == 0 and gpus[0]["gpu_util_pct"] == 0.0
    assert gpus[0]["vram_used_pct"] == 0.0  # guarded against a zero denominator


@pytest.mark.asyncio
async def test_gpu_metrics_skip_unparseable_lines_and_keep_the_good_ones():
    _exec, _ = _subprocess_seq(_proc("garbage line\n0, A, 8192, 1024, 7168, 5\nshort,row\n"))
    with patch("asyncio.create_subprocess_exec", _exec):
        gpus = await tm.get_gpu_metrics()
    assert len(gpus) == 1 and gpus[0]["name"] == "A"


# --------------------------------------------------------------------------- #
# get_llama_server_metrics                                                     #
# --------------------------------------------------------------------------- #


def _runtime(name="qwen3.6-35b-a3b:q4_k_m", backend="qwen3.6-35b-a3b--q4_k_m"):
    return FakeResponse(200, {"loaded_models": [{"name": name, "backend_model": backend}]})


PROPS = FakeResponse(200, {"model_path": "/router-models/x.gguf", "n_ctx": 32768, "n_gpu_layers": 33, "flash_attn": True})


@pytest.mark.asyncio
async def test_runtime_name_is_sanitised_to_the_same_form_the_web_app_globs(monkeypatch):
    monkeypatch.setenv("PROXY_URL", "http://proxy:11434")
    client = FakeClient({"/admin/runtime": _runtime(), "/props": PROPS})
    m = await tm.get_llama_server_metrics(client)
    # web/app.py sanitises the public name with the identical expression.
    import re

    assert m["model_alias"] == re.sub(r"[/:.]", "_", "qwen3.6-35b-a3b:q4_k_m") == "qwen3_6-35b-a3b_q4_k_m"


@pytest.mark.asyncio
async def test_proxy_auth_header_is_sent_only_when_a_key_is_configured(monkeypatch):
    monkeypatch.setenv("PROXY_URL", "http://proxy:11434")
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    client = FakeClient({"/admin/runtime": _runtime(), "/props": PROPS})
    await tm.get_llama_server_metrics(client)
    assert client.calls[0][1]["headers"] == {}

    monkeypatch.setenv("ALPACA_API_KEY", " secret ")
    client = FakeClient({"/admin/runtime": _runtime(), "/props": PROPS})
    await tm.get_llama_server_metrics(client)
    # Stripped, so a stray newline in .env cannot smuggle a second header.
    assert client.calls[0][1]["headers"] == {"Authorization": "Bearer secret"}


@pytest.mark.asyncio
async def test_slots_are_skipped_entirely_when_backend_model_is_none(monkeypatch):
    """llama-server's /slots REQUIRES ?model=; calling it bare returns 400."""
    monkeypatch.setenv("PROXY_URL", "http://proxy:11434")
    loading = FakeResponse(200, {"loaded_models": [{"name": "m", "backend_model": None}]})
    client = FakeClient({"/admin/runtime": loading, "/props": PROPS})
    m = await tm.get_llama_server_metrics(client)
    assert not any("/slots" in url for url, _ in client.calls)
    assert m["total_slots"] == 0 and m["model_alias"] == "m"


@pytest.mark.asyncio
async def test_slots_are_requested_with_the_backend_model_id_underscores_intact(monkeypatch):
    monkeypatch.setenv("PROXY_URL", "http://proxy:11434")
    client = FakeClient(
        {
            "/admin/runtime": _runtime(backend="qwen3.6-35b-a3b--q4_k_m"),
            "/props": PROPS,
            "/slots": FakeResponse(200, [{"state": 0, "n_past": 0}, {"state": 0, "n_past": 0}]),
        }
    )
    await tm.get_llama_server_metrics(client)
    slots_call = next(kw for url, kw in client.calls if "/slots" in url)
    # The double dash is the router separator, not a typo to be stripped.
    assert slots_call["params"] == {"model": "qwen3.6-35b-a3b--q4_k_m"}


@pytest.mark.asyncio
async def test_slot_activity_and_token_accounting(monkeypatch):
    monkeypatch.setenv("PROXY_URL", "http://proxy:11434")
    slots = [
        {"is_processing": True, "state": 0, "n_past": 1000},  # explicit flag wins
        {"state": 1, "n_past": 2000},  # no flag, non-idle state
        {"state": 0, "n_past": 500},  # idle
    ]
    client = FakeClient({"/admin/runtime": _runtime(), "/props": PROPS, "/slots": FakeResponse(200, slots)})
    m = await tm.get_llama_server_metrics(client)
    assert m["total_slots"] == 3
    assert m["active_slots"] == 2
    assert m["total_tokens_cached"] == 3500
    # 3500 of (32768 ctx * 3 slots)
    assert m["kv_cache_used_pct"] == 3.6


@pytest.mark.asyncio
async def test_kv_percent_is_zero_without_a_context_or_without_slots(monkeypatch):
    monkeypatch.setenv("PROXY_URL", "http://proxy:11434")
    client = FakeClient({"/admin/runtime": _runtime(), "/props": FakeResponse(200, {"n_ctx": 0}), "/slots": FakeResponse(200, [{}])})
    m = await tm.get_llama_server_metrics(client)
    assert m["kv_cache_used_pct"] == 0.0  # no divide by zero


@pytest.mark.asyncio
async def test_alias_falls_back_to_the_gguf_stem_when_the_proxy_is_unreachable(monkeypatch):
    monkeypatch.setenv("PROXY_URL", "http://proxy:11434")
    client = FakeClient({"/props": FakeResponse(200, {"model_path": "/router-models/qwen3.6-35b-a3b--q4_k_m.gguf", "n_ctx": 8192})})
    m = await tm.get_llama_server_metrics(client)
    assert m["model_alias"] == "qwen3_6-35b-a3b--q4_k_m"


@pytest.mark.asyncio
async def test_alias_is_system_idle_when_nothing_is_loaded(monkeypatch):
    monkeypatch.setenv("PROXY_URL", "http://proxy:11434")
    client = FakeClient({"/admin/runtime": FakeResponse(200, {"loaded_models": []}), "/props": FakeResponse(200, {"model_path": "none", "n_ctx": 8192})})
    m = await tm.get_llama_server_metrics(client)
    assert m["model_alias"] == "system_idle"


@pytest.mark.asyncio
async def test_a_dead_proxy_does_not_stop_the_llama_server_half_of_the_sample(monkeypatch):
    monkeypatch.setenv("PROXY_URL", "http://proxy:11434")
    client = FakeClient({"/props": PROPS})
    m = await tm.get_llama_server_metrics(client)
    assert m["n_ctx"] == 32768 and m["n_gpu_layers"] == 33 and m["flash_attn"] is True


@pytest.mark.asyncio
async def test_a_non_200_from_the_proxy_leaves_the_alias_for_the_props_fallback(monkeypatch):
    monkeypatch.setenv("PROXY_URL", "http://proxy:11434")
    client = FakeClient({"/admin/runtime": FakeResponse(503), "/props": PROPS})
    m = await tm.get_llama_server_metrics(client)
    assert m["model_alias"] == "x"  # stem of /router-models/x.gguf


@pytest.mark.asyncio
async def test_an_unexpected_client_exception_is_swallowed(monkeypatch):
    monkeypatch.setenv("PROXY_URL", "http://proxy:11434")
    client = Mock()
    client.get = AsyncMock(side_effect=OSError("dns failure"))
    m = await tm.get_llama_server_metrics(client)
    assert m["model_alias"] == "system_idle" and m["total_slots"] == 0


# --------------------------------------------------------------------------- #
# write_telemetry_log                                                          #
# --------------------------------------------------------------------------- #


def test_write_telemetry_log_creates_the_directory_and_appends_jsonl(telemetry_dir):
    tm.write_telemetry_log("model_a", {"epoch_time": 1})
    tm.write_telemetry_log("model_a", {"epoch_time": 2})
    tm.write_telemetry_log("model_b", {"epoch_time": 3})
    assert (telemetry_dir / "model_a.jsonl").read_text().count("\n") == 2
    assert (telemetry_dir / "model_b.jsonl").read_text().count("\n") == 1
    assert json.loads((telemetry_dir / "model_a.jsonl").read_text().splitlines()[1])["epoch_time"] == 2


def test_write_telemetry_log_keeps_going_when_the_disk_refuses(telemetry_dir):
    with patch("builtins.open", side_effect=OSError("read-only filesystem")):
        tm.write_telemetry_log("model_a", {"epoch_time": 1})  # must not raise
    assert not (telemetry_dir / "model_a.jsonl").exists()


# --------------------------------------------------------------------------- #
# handle_signals + main loop                                                   #
# --------------------------------------------------------------------------- #


def test_signal_handler_marshals_the_stop_event_back_onto_the_loop():
    """The handler runs outside the loop thread, so it must use the loop captured
    at startup. asyncio.get_event_loop() raises RuntimeError on 3.12+ under
    asyncio.run(), which would swallow the shutdown request entirely."""
    loop = Mock()
    event = Mock()
    tm._stop_event, tm._stop_loop = event, loop
    try:
        tm.handle_signals(15, None)
        loop.call_soon_threadsafe.assert_called_once_with(event.set)
    finally:
        tm._stop_event = tm._stop_loop = None


def test_signal_handler_before_the_loop_exists_is_harmless():
    tm._stop_event, tm._stop_loop = None, None
    tm.handle_signals(2, None)  # must not raise


def test_signal_handler_with_an_event_but_no_loop_does_not_fall_back_to_get_event_loop():
    """A half-initialised daemon (signal between import and main()) must not
    reach for a loop that does not exist."""
    tm._stop_event, tm._stop_loop = Mock(), None
    try:
        tm.handle_signals(2, None)  # must not raise
    finally:
        tm._stop_event = tm._stop_loop = None


@pytest.mark.asyncio
async def test_main_writes_one_row_per_poll_and_exits_when_stopped(telemetry_dir):
    """The loop races the gather against the stop event so SIGTERM interrupts a
    hung gather. One poll, then stop, must produce exactly one well-formed row."""
    polls = {"n": 0}

    async def fake_system():
        return {"ram_used_pct": 40.0, "ram_total_gb": 32.0}

    async def fake_gpu():
        return [{"name": "A", "vram_used_pct": 12.0, "vram_used_mb": 1, "vram_free_mb": 2, "vram_total_mb": 3, "index": 0, "gpu_util_pct": 5.0}]

    async def fake_llama(_client):
        polls["n"] += 1
        return {
            "model_alias": "model_x",
            "model_path": "/router-models/x.gguf",
            "n_ctx": 32768,
            "n_gpu_layers": 33,
            "flash_attn": True,
            "total_slots": 2,
            "active_slots": 1,
            "total_tokens_cached": 900,
            "kv_cache_used_pct": 1.4,
        }

    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    real_write = tm.write_telemetry_log

    def write_then_stop(alias, payload):
        real_write(alias, payload)
        polls["n"] = 99
        tm._stop_event.set()  # type: ignore[union-attr]

    with (
        patch.object(tm, "get_system_metrics", fake_system),
        patch.object(tm, "get_gpu_metrics", fake_gpu),
        patch.object(tm, "get_llama_server_metrics", fake_llama),
        patch.object(tm, "write_telemetry_log", write_then_stop),
        patch.object(tm.httpx, "AsyncClient", _Client),
    ):
        await tm.main()

    row = json.loads((telemetry_dir / "model_x.jsonl").read_text().splitlines()[0])
    assert len((telemetry_dir / "model_x.jsonl").read_text().splitlines()) == 1
    assert row["model_alias"] == "model_x"
    assert row["system"] == {"ram_used_pct": 40.0, "ram_total_gb": 32.0}
    assert row["gpus"][0]["vram_used_pct"] == 12.0
    # The nested shape is what web/app.py's /api/telemetry/history and
    # analyzer.py's creep regression both read.
    assert row["llama_server"]["slots"] == {"total": 2, "active": 1, "tokens_cached": 900, "kv_cache_used_pct": 1.4}
    assert row["timestamp"].endswith("+00:00") and row["epoch_time"] > 0


@pytest.mark.asyncio
async def test_a_hung_poll_is_abandoned_when_the_stop_event_fires(telemetry_dir):
    """A wedged gather must not block shutdown - that is the whole reason the
    gather is raced against the stop event instead of being awaited directly."""
    started = asyncio.Event()

    async def never(*_a, **_kw):
        started.set()
        await asyncio.sleep(3600)

    async def quick():
        return {}

    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    async def fire_stop():
        await started.wait()
        tm._stop_event.set()  # type: ignore[union-attr]

    with patch.object(tm, "get_system_metrics", quick), patch.object(tm, "get_gpu_metrics", quick), patch.object(tm, "get_llama_server_metrics", never), patch.object(tm.httpx, "AsyncClient", _Client):
        await asyncio.wait_for(asyncio.gather(tm.main(), fire_stop()), timeout=5)
    assert not list(telemetry_dir.glob("*.jsonl"))  # stopped before the first write
    assert started.is_set()
