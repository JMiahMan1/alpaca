"""Unit tests for settings_scan.py (resource-guided smart-guess engine)."""

import configparser

import settings_scan

# ---------- response gate ----------


def test_hello_verdict_passes_on_real_response():
    ok, reason = settings_scan.hello_verdict({"success": True, "content": "hi there", "tps": 2.5})
    assert ok is True
    assert reason == ""


def test_hello_verdict_fails_on_empty_content():
    ok, reason = settings_scan.hello_verdict({"success": False, "content": "", "tokens": 32, "error": "EMPTY CONTENT"})
    assert ok is False
    assert "no response" in reason or "EMPTY" in reason


def test_hello_verdict_fails_on_error():
    ok, reason = settings_scan.hello_verdict({"success": False, "content": "", "error": "HTTP 503: queue_timeout"})
    assert ok is False
    assert "HTTP 503" in reason


# ---------- long response validity ----------


def test_long_response_valid_accepts_complete_fenced_code():
    content = "```python\n" + "x = 1\n" * 100 + "```"
    ok, reason = settings_scan.long_response_valid({"success": True, "content": content})
    assert ok is True
    assert reason == ""


def test_long_response_valid_rejects_short_response():
    ok, reason = settings_scan.long_response_valid({"success": True, "content": "```python\nx=1\n```"})
    assert ok is False
    assert "too short" in reason


def test_long_response_valid_rejects_unclosed_fence():
    content = "```python\n" + "x = 1\n" * 100  # truncated, never closed
    ok, reason = settings_scan.long_response_valid({"success": True, "content": content})
    assert ok is False
    assert "fence" in reason


def test_long_response_valid_rejects_failed_probe():
    ok, _reason = settings_scan.long_response_valid({"success": False, "error": "timed out", "content": ""})
    assert ok is False


# ---------- constants / budgets ----------


def test_hello_is_cheaper_and_faster_than_probe():
    assert settings_scan.HELLO_TIMEOUT_S < settings_scan.PROBE_TIMEOUT_S
    assert settings_scan.HELLO_NUM_PREDICT < settings_scan.DENSE_MAX_TOKENS


def test_effective_num_predict_dense_is_capped():
    assert settings_scan.effective_num_predict(True) == settings_scan.MAX_TOKENS
    assert settings_scan.effective_num_predict(False) == settings_scan.DENSE_MAX_TOKENS


# ---------- resource estimation ----------


def test_kv_bytes_per_token_quant_ordering():
    shape = {"n_layers": 48, "n_head_kv": 8, "head_dim": 128}
    f16 = settings_scan.kv_bytes_per_token(shape, "f16")
    q8 = settings_scan.kv_bytes_per_token(shape, "q8_0")
    q4 = settings_scan.kv_bytes_per_token(shape, "q4_0")
    assert q4 < q8 < f16
    assert q8 == 2 * 48 * 8 * 128 * 1.0625


def test_smart_baseline_roomy_model_maxes_out_gpu_layers():
    shape = {"n_layers": 48, "n_head_kv": 8, "head_dim": 128, "file_size_mb": 2000, "expert_count": 0, "arch": "x"}
    s, est = settings_scan.compute_smart_baseline(shape, 65536, ["q8_0", "q4_0"], 16000)
    assert s["n-gpu-layers"] == "48"
    assert s["cache-type-k"] == "q8_0"  # roomy: highest-quality kv wins the tie
    assert est > 0


def test_smart_baseline_tight_model_downgrades_kv_and_layers():
    shape = {"n_layers": 48, "n_head_kv": 8, "head_dim": 128, "file_size_mb": 14000, "expert_count": 0, "arch": "x"}
    s, _est = settings_scan.compute_smart_baseline(shape, 65536, ["q8_0", "q4_0"], 8188)
    # q8_0 KV alone (~6.8GB at 64K) leaves nothing; q4_0 must win with partial offload
    assert s["cache-type-k"] == "q4_0"
    assert 0 < int(s["n-gpu-layers"]) < 48


def test_smart_baseline_unknown_shape_falls_back_conservative():
    s, _est = settings_scan.compute_smart_baseline({}, 65536, ["q8_0", "q4_0"], 8188)
    assert s["n-gpu-layers"] == "26"
    assert s["cache-type-k"] == "q4_0"


def test_smart_baseline_kv_in_ram_only_when_strictly_better(monkeypatch):
    """Dense model, KV-in-RAM lever fires only when it buys more GPU layers and RAM allows."""
    shape = {"n_layers": 48, "n_head_kv": 8, "head_dim": 128, "file_size_mb": 14000, "expert_count": 0, "arch": "x"}
    monkeypatch.setattr(settings_scan, "ram_available_mb", lambda: 30000)
    # Roomy VRAM: full offload already fits (48 layers incl. KV) -> no-kv-offload must NOT fire
    s, _ = settings_scan.compute_smart_baseline(shape, 65536, ["q8_0", "q4_0"], 24000)
    assert "no-kv-offload" not in s
    # Tight VRAM: KV-in-RAM frees the KV budget for weights -> 21 GPU layers vs 9
    s2, _ = settings_scan.compute_smart_baseline(shape, 65536, ["q8_0", "q4_0"], 8188)
    assert s2.get("no-kv-offload") == "true"
    assert s2["n-gpu-layers"] == "21"


def test_smart_baseline_kv_in_ram_never_for_moe(monkeypatch):
    """MoE experts already live in system RAM - stacking KV there risks exhaustion."""
    shape = {"n_layers": 48, "n_head_kv": 8, "head_dim": 128, "file_size_mb": 14000, "expert_count": 64, "arch": "x"}
    monkeypatch.setattr(settings_scan, "ram_available_mb", lambda: 30000)
    s, _ = settings_scan.compute_smart_baseline(shape, 65536, ["q8_0", "q4_0"], 8188, is_moe=True)
    assert "no-kv-offload" not in s
    # Same shape as dense would take the lever; MoE must not
    s_dense, _ = settings_scan.compute_smart_baseline(shape, 65536, ["q8_0", "q4_0"], 8188, is_moe=False)
    assert s_dense.get("no-kv-offload") == "true"


# ---------- escalation (no response => free VRAM and retry) ----------


def test_escalate_step_dense_lowers_ngl_then_exhausts():
    shape = {"n_layers": 48, "n_head_kv": 8, "head_dim": 128, "file_size_mb": 14000, "expert_count": 0}
    s1 = {"n-gpu-layers": "15", "ctx-size": "65536", "cache-type-k": "q4_0", "cache-type-v": "q4_0", "flash-attn": "on"}
    s2 = settings_scan.escalate_step(s1, shape, is_moe=False)
    assert s2 is not None and s2["n-gpu-layers"] == "11"  # step = max(4, 48//16)
    s3 = settings_scan.escalate_step(s2, shape, is_moe=False)
    assert s3 is None  # 11 - 4 = 7 < 8 floor


def test_escalate_step_moe_bumps_ncpu_moe_first():
    shape = {"n_layers": 48, "n_head_kv": 8, "head_dim": 128, "file_size_mb": 14000, "expert_count": 64}
    s1 = {"n-gpu-layers": "99", "ctx-size": "65536", "cache-type-k": "q8_0", "cache-type-v": "q8_0", "flash-attn": "on"}
    s2 = settings_scan.escalate_step(s1, shape, is_moe=True)
    assert s2 is not None and s2["n-cpu-moe"] == "28"
    assert s2["n-gpu-layers"] == "99"  # ngl untouched while the moe knob has room
    s2["n-cpu-moe"] = "48"
    s3 = settings_scan.escalate_step(s2, shape, is_moe=True)
    assert s3 is not None and s3["n-gpu-layers"] == "95"  # hierarchy exhausted -> step ngl


# ---------- hill climb ----------


def test_hill_neighbors_upgrade_ngl_and_kv_when_they_fit():
    shape = {"n_layers": 48, "n_head_kv": 8, "head_dim": 128, "file_size_mb": 2000, "expert_count": 0}
    base = {
        "ctx-size": "65536",
        "cache-type-k": "q4_0",
        "cache-type-v": "q4_0",
        "n-gpu-layers": "40",
        "flash-attn": "on",
    }
    neighbors = settings_scan.hill_neighbors(base, shape, is_moe=False, ctx=65536, vram_total=16000)
    labels = [settings_scan.label_for(n) for n in neighbors]
    assert any("ngl=42" in lab for lab in labels)  # ngl upgrade fits comfortably
    assert any("kv=q8_0" in lab for lab in labels)  # kv upgrade fits comfortably
    assert settings_scan.label_for(base) not in labels


def test_hill_neighbors_excludes_upgrades_that_bust_vram():
    shape = {"n_layers": 48, "n_head_kv": 8, "head_dim": 128, "file_size_mb": 14000, "expert_count": 0}
    base = {
        "ctx-size": "65536",
        "cache-type-k": "q4_0",
        "cache-type-v": "q4_0",
        "n-gpu-layers": "15",
        "flash-attn": "on",
    }
    neighbors = settings_scan.hill_neighbors(base, shape, is_moe=False, ctx=65536, vram_total=8188)
    assert neighbors == []  # 14GB weights on 8GB VRAM: no upgrade can fit


def test_label_for_includes_moe_knob():
    s = {"ctx-size": "98304", "cache-type-k": "q8_0", "cache-type-v": "q8_0", "n-gpu-layers": "99", "flash-attn": "on"}
    assert settings_scan.label_for(s) == "ctx=98304 kv=q8_0 ngl=99 fa=on"
    s["n-cpu-moe"] = "36"
    assert "moe=36" in settings_scan.label_for(s)


# ---------- profile mirror ----------


def test_write_profile_mirror_merges_ini_section(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_scan, "REPO", tmp_path)
    (tmp_path / ".alpaca-router").mkdir()
    prof = tmp_path / ".alpaca-router" / "test-model--latest.profile.json"
    prof.write_text('{"model": "x.gguf", "parallel": "1"}')

    c = configparser.ConfigParser(delimiters=("=",))
    c.add_section("test-model--latest")
    c["test-model--latest"]["ctx-size"] = "65536"
    c["test-model--latest"]["n-gpu-layers"] = "15"

    out = settings_scan.write_profile_mirror("test-model--latest", c)
    data = __import__("json").loads(out.read_text())
    assert data["ctx-size"] == "65536"
    assert data["n-gpu-layers"] == "15"
    assert data["parallel"] == "1"  # pre-existing keys preserved
    assert data["model"] == "x.gguf"


def test_probe_guard_blocks_hot_gpu(monkeypatch):
    monkeypatch.setattr(settings_scan, "gpu_temp_c", lambda: 90)
    ok, reason = settings_scan.probe_guard({}, {"n-gpu-layers": "10"}, 65536, 8188)
    assert not ok
    assert "temp" in reason


def test_probe_guard_blocks_tight_vram(monkeypatch):
    monkeypatch.setattr(settings_scan, "gpu_temp_c", lambda: 50)
    monkeypatch.setattr(settings_scan, "vram_mb", lambda: 8000)
    monkeypatch.setattr(settings_scan, "llama_server_vram_mb", lambda: 0)
    shape = {"n_layers": 48, "file_size_mb": 15000, "n_head_kv": 8, "head_dim": 128}
    ok, reason = settings_scan.probe_guard(shape, {"n-gpu-layers": "99"}, 65536, 8188)
    assert not ok
    assert "vram guard" in reason


def test_probe_guard_adds_back_llama_server_vram(monkeypatch):
    """Resident llama-server memory is released by the restart, so it must not
    count against the guard's free-VRAM budget."""
    monkeypatch.setattr(settings_scan, "gpu_temp_c", lambda: 50)
    monkeypatch.setattr(settings_scan, "vram_mb", lambda: 6080)
    monkeypatch.setattr(settings_scan, "llama_server_vram_mb", lambda: 6032)
    monkeypatch.setattr(settings_scan, "ram_available_mb", lambda: 30000)
    shape = {"n_layers": 48, "file_size_mb": 15000, "n_head_kv": 8, "head_dim": 128}
    # est for ngl=10: 15000*10/48 + q4_0 kv + 1500 ≈ 3125 + 5530 + 1500 = 10155 > 8188 → still blocks
    ok, reason = settings_scan.probe_guard(shape, {"n-gpu-layers": "10", "cache-type-k": "q4_0"}, 65536, 8188)
    assert not ok
    assert "free 8140" in reason  # 8188 - 6080 + 6032


def test_llama_server_vram_mb_parses_compute_apps(monkeypatch):
    class FakeOut:
        stdout = (
            "2661, /usr/bin/kwin_wayland, 7\n5666, /app/llama-server, 6032\n  ,  ,  \n777, /opt/other-server, 512\n"
        )

    monkeypatch.setattr(
        settings_scan.subprocess,
        "run",
        lambda *a, **k: FakeOut(),
    )
    assert settings_scan.llama_server_vram_mb() == 6032


def test_llama_server_vram_mb_tolerant_on_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no nvidia-smi")

    monkeypatch.setattr(settings_scan.subprocess, "run", boom)
    assert settings_scan.llama_server_vram_mb() == 0


def test_probe_guard_blocks_low_ram(monkeypatch):
    monkeypatch.setattr(settings_scan, "gpu_temp_c", lambda: 50)
    monkeypatch.setattr(settings_scan, "vram_mb", lambda: 500)
    monkeypatch.setattr(settings_scan, "llama_server_vram_mb", lambda: 0)
    monkeypatch.setattr(settings_scan, "ram_available_mb", lambda: 1000)
    shape = {"n_layers": 48, "file_size_mb": 8000, "n_head_kv": 8, "head_dim": 128}
    ok, reason = settings_scan.probe_guard(shape, {"n-gpu-layers": "10"}, 65536, 8188)
    assert not ok
    assert "ram guard" in reason


def test_probe_guard_allows_headroom(monkeypatch):
    monkeypatch.setattr(settings_scan, "gpu_temp_c", lambda: 50)
    monkeypatch.setattr(settings_scan, "vram_mb", lambda: 100)
    monkeypatch.setattr(settings_scan, "llama_server_vram_mb", lambda: 0)
    monkeypatch.setattr(settings_scan, "ram_available_mb", lambda: 30000)
    shape = {"n_layers": 48, "file_size_mb": 8000, "n_head_kv": 8, "head_dim": 128}
    ok, reason = settings_scan.probe_guard(
        shape, {"n-gpu-layers": "10", "cache-type-k": "q4_0", "cache-type-v": "q4_0"}, 65536, 8188
    )
    assert ok
    assert reason == ""


def test_load_section_unpins_large_ctx():
    import configparser

    ini = configparser.ConfigParser()
    ini.add_section("m")
    settings_scan.load_section(ini, "m", {"ctx-size": "65536"})
    assert ini["m"]["mlock"] == "false"
    assert ini["m"]["no-mmap"] == "false"


def test_load_section_keeps_pin_small_ctx():
    import configparser

    ini = configparser.ConfigParser()
    ini.add_section("m")
    ini["m"]["mlock"] = "true"
    settings_scan.load_section(ini, "m", {"ctx-size": "8192"})
    assert ini["m"]["mlock"] == "true"


# ---------- in-run thermal watchdog ----------


def test_cpu_tctl_parse():
    import subprocess

    class FakeOut:
        stdout = "Tctl:         +54.6°C  \nComposite:    +38.9°C\n"

        def __init__(self):
            self.returncode = 0

    orig = settings_scan.subprocess.run
    settings_scan.subprocess.run = lambda *a, **k: FakeOut()
    try:
        assert settings_scan._cpu_tctl_c() == 54.6
    finally:
        settings_scan.subprocess.run = orig
    assert orig is subprocess.run


def test_cpu_tctl_parse_missing_sensor():
    class FakeOut:
        stdout = "Composite:    +38.9°C\n"

    orig = settings_scan.subprocess.run
    settings_scan.subprocess.run = lambda *a, **k: FakeOut()
    try:
        assert settings_scan._cpu_tctl_c() is None
    finally:
        settings_scan.subprocess.run = orig


def test_cpu_tctl_parse_error_tolerant():
    orig = settings_scan.subprocess.run

    def boom(*a, **k):
        raise OSError("no sensors")

    settings_scan.subprocess.run = boom
    try:
        assert settings_scan._cpu_tctl_c() is None
    finally:
        settings_scan.subprocess.run = orig


def test_thermal_exceeded_gpu_boundary():
    assert settings_scan.thermal_exceeded({"gpu": settings_scan.GUARD_MAX_TEMP_C - 0.5, "cpu": 60.0}) == (False, "")
    ok, reason = settings_scan.thermal_exceeded({"gpu": settings_scan.GUARD_MAX_TEMP_C, "cpu": 60.0})
    assert ok is True
    assert "gpu" in reason


def test_thermal_exceeded_cpu_boundary():
    assert settings_scan.thermal_exceeded({"gpu": 45, "cpu": settings_scan.THERMAL_CPU_ABORT_C - 0.5}) == (False, "")
    ok, reason = settings_scan.thermal_exceeded({"gpu": 45, "cpu": settings_scan.THERMAL_CPU_ABORT_C})
    assert ok is True
    assert "cpu" in reason


def test_thermal_exceeded_none_tolerant():
    assert settings_scan.thermal_exceeded({"gpu": None, "cpu": None}) == (False, "")


def test_thermal_stop_skips_models():
    """THERMAL_STOP is a module flag checked before each model's phases."""
    assert isinstance(settings_scan.THERMAL_STOP, bool)


def test_thermal_thresholds_are_ordered():
    """Pre-cool target < throttle start < abort, or the watchdogs fight each other."""
    assert settings_scan.GUARD_PRECOOL_C < settings_scan.THERMAL_THROTTLE_C
    assert settings_scan.THERMAL_THROTTLE_C < settings_scan.THERMAL_CPU_ABORT_C
