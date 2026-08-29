"""Tests for the UI-configurable thermal watchdog (web/thermal.py) and its
integration points in the web benchmark path."""

import asyncio

import pytest

import web.thermal as thermal
from web.thermal import ThermalAbortError, ThermalWatchdog, load_config, save_config


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("THERMAL_WATCHDOG_CONFIG", str(tmp_path / "thermal_watchdog.json"))
    # Clear env overrides so tests see the data-file behavior
    for name in (
        "THERMAL_WATCHDOG",
        "THERMAL_ABORT_C",
        "THERMAL_THROTTLE_C",
        "THERMAL_RESUME_C",
        "THERMAL_POLL_S",
        "THERMAL_PRETEST_MAX_WAIT_S",
    ):
        monkeypatch.delenv(name, raising=False)


def test_load_config_defaults():
    cfg = load_config()
    assert cfg["enabled"] is True
    assert cfg["resume_c"] < cfg["throttle_c"] < cfg["abort_c"]


def test_env_overrides_win(monkeypatch):
    monkeypatch.setenv("THERMAL_WATCHDOG", "off")
    monkeypatch.setenv("THERMAL_ABORT_C", "88")
    cfg = load_config()
    assert cfg["enabled"] is False
    assert cfg["abort_c"] == 88.0


def test_save_config_persists_and_clamps():
    cfg = save_config({"enabled": True, "abort_c": 99.0, "throttle_c": 99.0, "resume_c": 99.0, "poll_s": 0})
    assert cfg["abort_c"] == 99.0
    assert cfg["throttle_c"] < cfg["abort_c"]  # clamped to keep ordering
    assert cfg["resume_c"] < cfg["throttle_c"]
    assert cfg["poll_s"] >= 1.0
    stored = thermal.load_config()
    assert stored["enabled"] is True


def test_save_config_disabled_is_honored():
    cfg = save_config({"enabled": False})
    assert cfg["enabled"] is False
    assert load_config()["enabled"] is False


def test_heartbeat_aborts_at_threshold(monkeypatch):
    w = ThermalWatchdog({"poll_s": 0.0})
    monkeypatch.setattr(w, "_read_temps", lambda: {"cpu": 94.0, "gpu": 60.0})
    with pytest.raises(ThermalAbortError):
        asyncio.run(w.heartbeat())
    assert w.aborted
    assert w.peak_cpu_c == 94.0
    stats = w.stats()
    assert stats["aborted"] is True and "94.0" in stats["abort_reason"]


def test_heartbeat_throttles_then_resumes(monkeypatch):
    w = ThermalWatchdog({"poll_s": 0.0, "resume_c": 78.0, "throttle_c": 85.0, "abort_c": 93.0})
    readings = iter(
        [
            {"cpu": 86.0, "gpu": 60.0},  # throttle trigger
            {"cpu": 80.0, "gpu": 60.0},  # still hot in pause loop
            {"cpu": 70.0, "gpu": 60.0},  # cooled -> resume
        ]
    )
    monkeypatch.setattr(w, "_read_temps", lambda: next(readings))

    async def fast_sleep(_):
        return None

    monkeypatch.setattr(thermal.asyncio, "sleep", fast_sleep)
    assert asyncio.run(w.heartbeat()) == "ok"
    assert w.throttled_s >= 0.0
    assert w.peak_cpu_c == 86.0


def test_heartbeat_disabled_never_reads(monkeypatch):
    w = ThermalWatchdog({"enabled": False})

    def _boom():
        raise AssertionError("should not read temps when disabled")

    monkeypatch.setattr(w, "_read_temps", _boom)
    assert asyncio.run(w.heartbeat()) == "ok"
    assert w.peak_cpu_c is None


def test_heartbeat_rate_limited(monkeypatch):
    w = ThermalWatchdog({"poll_s": 999.0})
    calls = {"n": 0}

    def _read():
        calls["n"] += 1
        return {"cpu": 50.0, "gpu": 50.0}

    monkeypatch.setattr(w, "_read_temps", _read)
    assert asyncio.run(w.heartbeat()) == "ok"
    assert asyncio.run(w.heartbeat()) == "ok"
    assert calls["n"] == 1  # second call within poll window is a no-op


def test_pre_test_wait_returns_when_cool(monkeypatch):
    w = ThermalWatchdog({"poll_s": 0.0})
    monkeypatch.setattr(w, "_read_temps", lambda: {"cpu": 60.0, "gpu": 50.0})
    asyncio.run(w.pre_test_wait())  # returns immediately
    assert w.peak_cpu_c == 60.0


def test_pre_test_wait_bounds_when_hot(monkeypatch):
    w = ThermalWatchdog({"pretest_max_wait_s": 0.0})
    monkeypatch.setattr(w, "_read_temps", lambda: {"cpu": 91.0, "gpu": 55.0})
    asyncio.run(w.pre_test_wait())  # deadline already passed -> proceeds
    assert w.peak_cpu_c == 91.0


def test_stats_shape():
    w = ThermalWatchdog({"enabled": False})
    stats = w.stats()
    assert set(stats) == {"enabled", "peak_cpu_c", "peak_gpu_c", "throttled_s", "aborted", "abort_reason"}


# ---------- stream reader integration ----------


class _FakeResponse:
    def __init__(self, lines):
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line


def test_chat_stream_stops_on_abort(monkeypatch):
    from web.shared_llm_benchmark import _read_chat_stream

    lines = [
        '{"message":{"content":"partial"}}',
        '{"message":{"content":" more"}}',
        '{"done":true,"eval_count":42}',
    ]
    resp = _FakeResponse(lines)

    class _HotWatchdog:
        aborted = False

        def __init__(self):
            self.calls = 0

        async def heartbeat(self):
            self.calls += 1
            if self.calls >= 2:
                self.aborted = True  # real watchdog sets this before raising
                raise ThermalAbortError("cpu=94.0C >= abort 93C")

    fake = _HotWatchdog()
    data = asyncio.run(_read_chat_stream(resp, watchdog=fake))
    assert data["content"] == "partial"  # stopped before the second chunk
    assert data["thermal_abort"] is True
    assert data["eval_count"] == 0  # final done line never processed
    assert fake.calls == 2  # heartbeat ran on the first two lines


def test_chat_stream_without_watchdog_unchanged():
    from web.shared_llm_benchmark import _read_chat_stream

    resp = _FakeResponse(['{"message":{"content":"hi"}}', '{"done":true,"eval_count":7}'])
    data = asyncio.run(_read_chat_stream(resp))
    assert data["content"] == "hi"
    assert data["eval_count"] == 7
    assert data["thermal_abort"] is False


def test_thermal_abort_result_shape():
    from web.shared_llm_benchmark import SharedLLMModelBenchmark

    w = ThermalWatchdog({"enabled": True})
    w.aborted = True
    w.abort_reason = "cpu=94.0C >= abort 93C"
    res = SharedLLMModelBenchmark._thermal_abort_result(w, 12.0, "partial", 100)
    assert res["success"] is False
    assert res["thermal_aborted"] is True
    assert "thermal watchdog abort" in res["error"]
    assert res["temps"]["aborted"] is True


def test_watchdog_stats_absent_when_no_watchdog():
    from web.shared_llm_benchmark import SharedLLMModelBenchmark

    assert SharedLLMModelBenchmark._watchdog_stats(None) == {}
    w = ThermalWatchdog({"enabled": False})
    assert SharedLLMModelBenchmark._watchdog_stats(w)["temps"]["enabled"] is False
