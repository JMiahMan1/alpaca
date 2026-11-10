"""Unit tests for settings_scan.py (resource-guided smart-guess engine)."""

import configparser

import pytest

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


@pytest.mark.parametrize(
    ("kv_choices", "expected_kv"),
    [(["q8_0", "q4_0"], "q8_0"), (["q8_0"], "q8_0"), (["q4_0"], "q4_0"), ([], "q8_0")],
)
def test_smart_baseline_unknown_shape_falls_back_conservative(kv_choices, expected_kv):
    s, est = settings_scan.compute_smart_baseline({}, 65536, kv_choices, 8188)
    assert s["n-gpu-layers"] == "26"
    assert s["cache-type-k"] == s["cache-type-v"] == expected_kv
    assert s["ctx-size"] == "65536"
    assert est == 0


def test_smart_baseline_never_offloads_kv_to_ram(monkeypatch):
    """KV-in-RAM lever retired: KV-over-PCIe decode collapses long-form generation."""
    shape = {"n_layers": 48, "n_head_kv": 8, "head_dim": 128, "file_size_mb": 14000, "expert_count": 0, "arch": "x"}
    monkeypatch.setattr(settings_scan, "ram_available_mb", lambda: 30000)
    s, _ = settings_scan.compute_smart_baseline(shape, 65536, ["q8_0", "q4_0"], 24000)
    assert "no-kv-offload" not in s
    # Tight VRAM: stays with fewer GPU layers and KV in VRAM - never the lever
    s2, _ = settings_scan.compute_smart_baseline(shape, 65536, ["q8_0", "q4_0"], 8188)
    assert "no-kv-offload" not in s2
    assert s2["cache-type-k"] == s2["cache-type-v"]


def test_smart_baseline_kv_in_ram_never_for_moe(monkeypatch):
    """MoE experts already live in system RAM - stacking KV there risks exhaustion."""
    shape = {"n_layers": 48, "n_head_kv": 8, "head_dim": 128, "file_size_mb": 14000, "expert_count": 64, "arch": "x"}
    monkeypatch.setattr(settings_scan, "ram_available_mb", lambda: 30000)
    s, _ = settings_scan.compute_smart_baseline(shape, 65536, ["q8_0", "q4_0"], 8188, is_moe=True)
    assert "no-kv-offload" not in s
    # Same shape as dense: retired lever means dense never sets it either
    s_dense, _ = settings_scan.compute_smart_baseline(shape, 65536, ["q8_0", "q4_0"], 8188, is_moe=False)
    assert "no-kv-offload" not in s_dense


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
    assert neighbors == [dict(base, **{"ubatch-size": "512"}), dict(base, **{"cache-reuse": "0"})]


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


@pytest.mark.parametrize("ctx", [settings_scan.GUARD_UNPIN_CTX, 65536])
@pytest.mark.parametrize("existing", [{}, {"mlock": "true", "no-mmap": "true", "load-mode": "mlock"}])
def test_load_section_unpins_large_ctx(ctx, existing):
    ini = configparser.ConfigParser()
    ini["m"] = existing
    settings_scan.load_section(ini, "m", {"ctx-size": str(ctx)})
    assert "mlock" not in ini["m"]
    assert "no-mmap" not in ini["m"]
    assert ini["m"]["load-mode"] == "mmap"


@pytest.mark.parametrize("ctx", [8192, settings_scan.GUARD_UNPIN_CTX - 1])
def test_load_section_keeps_pin_small_ctx(ctx):
    ini = configparser.ConfigParser()
    ini["m"] = {"load-mode": "mlock"}
    settings_scan.load_section(ini, "m", {"ctx-size": str(ctx)})
    assert ini["m"]["load-mode"] == "mlock"


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


# ---------- MoE expert cache, incumbent, n-cpu-moe-aware VRAM estimate ----------

MOE_SHAPE = {"n_layers": 40, "n_head_kv": 2, "head_dim": 256, "file_size_mb": 18536, "expert_count": 256}
MOE_WORKING = {
    "ctx-size": "65536",
    "cache-type-k": "q8_0",
    "cache-type-v": "q8_0",
    "n-gpu-layers": "26",
    "flash-attn": "on",
    "n-cpu-moe": "34",
}


def test_moe_cache_is_sized_from_measured_headroom_and_rounded():
    # 8188 * 0.85 = 6959.8 budget; 5000 used -> 1959.8 spare -> 80% = 1567 -> 1536
    assert settings_scan.moe_cache_size_mib(5000, 8188) == 1536


@pytest.mark.parametrize("measured", [None, 0, 6500, 9000])
def test_moe_cache_is_skipped_without_a_measurement_or_headroom(measured):
    assert settings_scan.moe_cache_size_mib(measured, 8188) == 0


def test_hill_offers_the_moe_cache_first_when_experts_are_offloaded():
    neighbors = settings_scan.hill_neighbors(
        MOE_WORKING, MOE_SHAPE, is_moe=True, ctx=65536, vram_total=8188, measured_vram_mb=5000
    )
    assert neighbors[0] == {**MOE_WORKING, "moe-cache-mib": "1536"}
    assert neighbors[1] == {**MOE_WORKING, "moe-cache-mib": "1536", "moe-cache-cpu-share": "0.5"}
    assert "mcache=1536/0.5" in settings_scan.label_for(neighbors[1])


def test_hill_offers_no_moe_cache_without_offloaded_experts_or_measurement():
    no_offload = {k: v for k, v in MOE_WORKING.items() if k != "n-cpu-moe"}
    for settings, measured in ((no_offload, 5000), (MOE_WORKING, None)):
        neighbors = settings_scan.hill_neighbors(
            settings, MOE_SHAPE, is_moe=True, ctx=65536, vram_total=8188, measured_vram_mb=measured
        )
        assert not any(n.get("moe-cache-mib") for n in neighbors)


def test_gpu_weights_subtract_only_experts_of_offloaded_gpu_layers():
    wpl = 18536 / 40
    # ngl=26 puts layers 14..39 on the GPU; n-cpu-moe=34 covers layers 0..33,
    # so 20 GPU layers have their experts in RAM.
    assert settings_scan.gpu_weights_mb(MOE_WORKING, MOE_SHAPE) == pytest.approx(wpl * 26 - wpl * 0.9 * 20)
    # ngl=99 clamps to the layer count instead of counting 99 layers of VRAM.
    full = {**MOE_WORKING, "n-gpu-layers": "99", "n-cpu-moe": "0"}
    assert settings_scan.gpu_weights_mb(full, MOE_SHAPE) == pytest.approx(18536)
    # n-cpu-moe means nothing for a dense model.
    dense = {**MOE_SHAPE, "expert_count": 0}
    assert settings_scan.gpu_weights_mb(MOE_WORKING, dense) == pytest.approx(wpl * 26)


def test_guard_no_longer_blocks_a_high_ngl_moe_config_with_offloaded_experts(monkeypatch):
    monkeypatch.setattr(settings_scan, "gpu_temp_c", lambda: 50)
    monkeypatch.setattr(settings_scan, "vram_mb", lambda: 200)
    monkeypatch.setattr(settings_scan, "llama_server_vram_mb", lambda: 0)
    monkeypatch.setattr(settings_scan, "ram_available_mb", lambda: 20000)
    ornith = {**MOE_WORKING, "n-gpu-layers": "99", "n-cpu-moe": "38"}
    ok, reason = settings_scan.probe_guard(MOE_SHAPE, ornith, 65536, 8188)
    assert ok, reason


def test_incumbent_is_offered_on_the_same_rung_with_every_managed_key():
    original = {**MOE_WORKING, "spec-type": "draft-mtp", "spec-draft-n-max": "3", "model": "/m.gguf"}
    winner = {**MOE_WORKING, "n-gpu-layers": "12", "n-cpu-moe": ""}
    cand = settings_scan.incumbent_candidate(original, winner)
    assert cand is not None
    assert set(cand) == set(settings_scan.MANAGED_KEYS)  # blanks remove keys a probe left behind
    assert cand["n-cpu-moe"] == "34" and cand["ubatch-size"] == ""
    assert "model" not in cand


def test_incumbent_is_not_offered_off_rung_or_when_identical():
    winner = dict(MOE_WORKING)
    assert settings_scan.incumbent_candidate({**MOE_WORKING, "ctx-size": "32768"}, winner) is None
    assert settings_scan.incumbent_candidate({**MOE_WORKING, "cache-type-k": "q4_0"}, winner) is None
    assert settings_scan.incumbent_candidate(dict(MOE_WORKING), winner) is None
    assert settings_scan.incumbent_candidate({}, winner) is None


def test_measured_configs_skip_only_the_vram_estimate(monkeypatch):
    monkeypatch.setattr(settings_scan, "vram_mb", lambda: 200)
    monkeypatch.setattr(settings_scan, "llama_server_vram_mb", lambda: 0)
    monkeypatch.setattr(settings_scan, "ram_available_mb", lambda: 20000)
    monkeypatch.setattr(settings_scan, "gpu_temp_c", lambda: 50)
    huge = {**MOE_WORKING, "n-gpu-layers": "41", "n-cpu-moe": ""}
    assert not settings_scan.probe_guard(MOE_SHAPE, huge, 65536, 8188)[0]
    assert settings_scan.probe_guard(MOE_SHAPE, huge, 65536, 8188, check_vram=False)[0]
    monkeypatch.setattr(settings_scan, "gpu_temp_c", lambda: 90)  # heat still blocks
    assert not settings_scan.probe_guard(MOE_SHAPE, huge, 65536, 8188, check_vram=False)[0]


def test_vram_is_measured_for_the_incumbent_and_cache_probes_only():
    inc = dict(MOE_WORKING)
    assert settings_scan.vram_is_measured(dict(MOE_WORKING), inc)
    assert settings_scan.vram_is_measured({**MOE_WORKING, "n-gpu-layers": "12", "moe-cache-mib": "1024"}, None)
    assert not settings_scan.vram_is_measured({**MOE_WORKING, "n-gpu-layers": "30"}, inc)
    assert not settings_scan.vram_is_measured(dict(MOE_WORKING), None)


def test_a_winner_from_this_run_ends_the_rung_ladder_even_with_fresh():
    won = {"m": {"done": True}}
    assert settings_scan.rung_already_won({}, "m", fresh=True, won_this_run={"m"})
    assert settings_scan.rung_already_won({"m::winner": won["m"]}, "m", fresh=False, won_this_run=set())
    # --fresh ignores an earlier run's winner, but only that.
    assert not settings_scan.rung_already_won({"m::winner": won["m"]}, "m", fresh=True, won_this_run=set())
    assert not settings_scan.rung_already_won({}, "m", fresh=False, won_this_run=set())


def test_moe_without_offloaded_experts_gets_the_all_layers_on_gpu_layout():
    partial = {k: v for k, v in MOE_WORKING.items() if k != "n-cpu-moe"} | {"n-gpu-layers": "7"}
    neighbors = settings_scan.hill_neighbors(partial, MOE_SHAPE, is_moe=True, ctx=65536, vram_total=8188)
    full = [n for n in neighbors if n.get("n-cpu-moe")]
    assert len(full) == 1
    cand = full[0]
    assert cand["n-gpu-layers"] == "40"
    k = int(cand["n-cpu-moe"])
    # The smallest expert offload the estimate admits: one fewer would not fit.
    assert settings_scan._est_vram_mb(cand, MOE_SHAPE, 65536) <= 8188 * settings_scan.VRAM_TARGET_FRAC
    tighter = {**cand, "n-cpu-moe": str(k - 1)}
    assert settings_scan._est_vram_mb(tighter, MOE_SHAPE, 65536) > 8188 * settings_scan.VRAM_TARGET_FRAC


def test_dense_models_never_get_an_expert_offload_layout():
    shape = {**MOE_SHAPE, "expert_count": 0}
    base = {k: v for k, v in MOE_WORKING.items() if k != "n-cpu-moe"}
    neighbors = settings_scan.hill_neighbors(base, shape, is_moe=False, ctx=65536, vram_total=8188)
    assert not any(n.get("n-cpu-moe") for n in neighbors)


def test_dense_models_get_the_tighter_hill_cap():
    assert settings_scan.hill_limit(10, 4, is_moe=False) == 4
    assert settings_scan.hill_limit(10, 4, is_moe=True) == 10
    assert settings_scan.hill_limit(10, None, is_moe=False) == 10
    assert settings_scan.hill_limit(3, 4, is_moe=False) == 3
    # A dense model entirely in VRAM decodes fast: its probes are cheap, no cap.
    assert settings_scan.hill_limit(10, 4, is_moe=False, fully_on_gpu=True) == 10


def test_a_dense_models_first_hill_neighbors_are_the_ones_that_matter():
    """With the dense cap of 4 (incumbent first), the next three must be more GPU
    layers, ubatch and MTP - cache-reuse on/off is the one a cap may drop."""
    shape = {"n_layers": 64, "n_head_kv": 8, "head_dim": 128, "file_size_mb": 13000, "expert_count": 0, "has_mtp": True}
    base = {
        "ctx-size": "32768",
        "cache-type-k": "q8_0",
        "cache-type-v": "q8_0",
        "n-gpu-layers": "20",
        "flash-attn": "on",
    }
    neighbors = settings_scan.hill_neighbors(base, shape, is_moe=False, ctx=32768, vram_total=16000)
    first3 = neighbors[:3]
    assert first3[0]["n-gpu-layers"] != "20"
    assert any(n.get("ubatch-size") for n in first3)
    assert any(n.get("spec-type") == "draft-mtp" for n in first3)
    assert neighbors[-1].get("cache-reuse") is not None


def test_the_scan_holds_the_lock_while_it_writes_models_ini(monkeypatch, tmp_path):
    monkeypatch.setattr(settings_scan, "INI", tmp_path / "models.ini")
    monkeypatch.setattr(settings_scan, "SCAN_LOCK", tmp_path / ".settings-scan.lock")
    c = configparser.ConfigParser(delimiters=("=",))
    c["m"] = {"ctx-size": "65536"}
    settings_scan.write_ini(c)
    assert (tmp_path / ".settings-scan.lock").exists()
    settings_scan.release_scan_lock()
    assert not (tmp_path / ".settings-scan.lock").exists()
    settings_scan.release_scan_lock()  # idempotent


def test_kv_cache_counts_only_the_layers_that_keep_one():
    dense = {"n_layers": 64, "n_head_kv": 4, "head_dim": 256}
    hybrid = {**dense, "full_attention_interval": 4}  # qwen3.8-27b: 1 KV layer in 4
    full = settings_scan.kv_cache_mb(dense, "q8_0", 65536)
    assert settings_scan.kv_cache_mb(hybrid, "q8_0", 65536) == pytest.approx(full / 4)
    # Sliding window: 3 of 4 layers hold at most 2048 tokens.
    swa = {"n_layers": 52, "n_head_kv": 2, "head_dim": 128, "sliding_window": 2048, "sliding_window_pattern": 4}
    per = settings_scan.kv_bytes_per_token(swa, "q8_0") / 52
    assert settings_scan.kv_cache_mb(swa, "q8_0", 65536) == pytest.approx(per * (13 * 65536 + 39 * 2048) / 1e6)
    # At a context inside the window it is just every layer.
    assert settings_scan.kv_cache_mb(swa, "q8_0", 1024) == pytest.approx(per * 52 * 1024 / 1e6)


def test_hybrid_baseline_no_longer_starves_qwen38_27b_of_gpu_layers():
    shape = {"n_layers": 65, "n_head_kv": 4, "head_dim": 256, "file_size_mb": 13100, "full_attention_interval": 4}
    baseline, _ = settings_scan.compute_smart_baseline(shape, 65536, ["q8_0"], 8188)
    # Counting every layer's KV put this at 0 GPU layers (then the guard blocked
    # even that); its old hand profile ran 24 layers at 32K.
    assert int(baseline["n-gpu-layers"]) >= 15


def test_a_partial_ngl_moe_config_with_offloaded_experts_still_gets_the_full_layout():
    """qwen3.6's incumbent (ngl=26, moe=34) never tried every layer on the GPU."""
    neighbors = settings_scan.hill_neighbors(MOE_WORKING, MOE_SHAPE, is_moe=True, ctx=65536, vram_total=8188)
    full = [n for n in neighbors if n.get("n-gpu-layers") == "40" and n.get("n-cpu-moe")]
    assert len(full) == 1  # every layer on the GPU, the smallest expert offload that fits
    # Already all on the GPU with experts offloaded: nothing to add.
    done = {**MOE_WORKING, "n-gpu-layers": "40"}
    again = settings_scan.hill_neighbors(done, MOE_SHAPE, is_moe=True, ctx=65536, vram_total=8188)
    assert not [n for n in again if n.get("n-gpu-layers") == "40" and n.get("n-cpu-moe") != done["n-cpu-moe"]]


def _write_gguf(path, kv: dict):
    """Minimal GGUF v3 header with the given metadata (no tensors)."""
    import struct

    def s(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kv))
    for key, val in kv.items():
        out += s(key)
        if isinstance(val, str):
            out += struct.pack("<I", 8) + s(val)
        elif isinstance(val, list) and val and isinstance(val[0], bool):
            out += struct.pack("<I", 9) + struct.pack("<I", 7) + struct.pack("<Q", len(val))
            out += bytes(int(v) for v in val)
        elif isinstance(val, list):
            out += struct.pack("<I", 9) + struct.pack("<I", 5) + struct.pack("<Q", len(val))
            out += struct.pack(f"<{len(val)}i", *val)
        else:
            out += struct.pack("<I", 4) + struct.pack("<I", val)
    path.write_bytes(out)


def test_per_layer_arrays_are_read_not_replaced_by_their_byte_size(tmp_path):
    """gemma4 / lfm2moe store head_count_kv and the sliding pattern per layer.
    Reading only the byte size gave '120 KV heads' and a 115 GB KV estimate."""
    gemma = tmp_path / "gemma.gguf"
    heads = [8, 8, 8, 8, 8, 2] * 5
    swa = [True, True, True, True, True, False] * 5
    _write_gguf(
        gemma,
        {
            "general.architecture": "gemma4",
            "gemma4.block_count": 30,
            "gemma4.attention.head_count_kv": heads,
            "gemma4.attention.sliding_window_pattern": swa,
            "gemma4.attention.key_length": 512,
            "gemma4.attention.key_length_swa": 256,
            "gemma4.attention.sliding_window": 1024,
            "gemma4.expert_count": 128,
        },
    )
    sh = settings_scan.read_gguf_shape(str(gemma))
    assert sh["kv_heads_per_layer"] == heads and sh["swa_per_layer"] == swa and sh["n_head_kv"] == 8
    # 25 sliding layers at 1024 tokens (8 heads x 256) + 5 full layers at ctx (2 heads x 512).
    expect = (25 * 2 * 8 * 256 * 1.0625 * 1024 + 5 * 2 * 2 * 512 * 1.0625 * 65536) / 1e6
    assert settings_scan.kv_cache_mb(sh, "q8_0", 65536) == pytest.approx(expect)

    lfm = tmp_path / "lfm.gguf"
    lfm_heads = [0, 0, 0, 8] * 10  # conv layers carry no KV cache
    _write_gguf(
        lfm,
        {
            "general.architecture": "lfm2moe",
            "lfm2moe.block_count": 40,
            "lfm2moe.attention.head_count_kv": lfm_heads,
            "lfm2moe.attention.head_count": 32,
            "lfm2moe.embedding_length": 2048,
            "lfm2moe.expert_count": 64,
        },
    )
    sh = settings_scan.read_gguf_shape(str(lfm))
    assert sh["head_dim"] == 64  # no key_length: embedding / heads
    assert settings_scan.kv_cache_mb(sh, "q8_0", 65536) == pytest.approx(10 * 2 * 8 * 64 * 1.0625 * 65536 / 1e6)
