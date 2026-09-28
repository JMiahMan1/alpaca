"""Unit tests for analyzer.py — the telemetry-driven tuning recommender.

The dashboard's "Resource Analysis" and "Apply Auto-Tuning Configs" cards are
this module's output, so the things pinned here are the ones a bad
recommendation would damage: the age window, the two threshold sets, the
creep slope, and above all the blacklist (a recommendation that walks a model
back into a config that already OOMed it is worse than no recommendation).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import analyzer


def _point(
    ram: float,
    vram: float = 40.0,
    vram_used: int = 4000,
    vram_total: int = 8192,
    cpu: float = 20.0,
    gpu: float = 30.0,
    cached: int = 1000,
    age_s: float = 0.0,
) -> dict:
    return {
        "epoch_time": time.time() - age_s,
        "system": {"ram_used_pct": ram, "cpu_util_pct": cpu},
        "gpus": [
            {
                "gpu_util_pct": gpu,
                "vram_used_pct": vram,
                "vram_used_mb": vram_used,
                "vram_total_mb": vram_total,
            }
        ],
        "llama_server": {"slots": {"tokens_cached": cached}},
    }


@pytest.fixture
def telemetry(tmp_path, monkeypatch):
    """A TELEMETRY_DIR the module will actually read (patched attribute wins)."""

    def _write(alias: str, points: list[dict]) -> Path:
        d = tmp_path / "telemetry"
        d.mkdir(exist_ok=True)
        (d / f"{alias}.jsonl").write_text("\n".join(json.dumps(p) for p in points) + "\n")
        monkeypatch.setattr(analyzer, "TELEMETRY_DIR", d)
        return d

    return _write


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """`data/failed_configs.json` is read from a hard-coded cwd-relative path, so
    every test runs in its own empty directory: the repo's real blacklist can
    never leak in, and a test that writes one cannot affect another."""
    d = tmp_path / "cwd"
    d.mkdir()
    monkeypatch.chdir(d)
    return d


# --------------------------------------------------------------------------
# _dir(): the import-time vs call-time directory fix
# --------------------------------------------------------------------------


def test_dir_prefers_an_explicitly_patched_attribute(monkeypatch, tmp_path):
    monkeypatch.setattr(analyzer, "TELEMETRY_DIR", tmp_path / "patched")
    monkeypatch.setenv("TELEMETRY_DIR", str(tmp_path / "from_env"))
    assert analyzer._dir("TELEMETRY_DIR") == tmp_path / "patched"


def test_dir_falls_back_to_the_environment_variable(monkeypatch, tmp_path):
    monkeypatch.setattr(analyzer, "TELEMETRY_DIR", analyzer._IMPORT_TIME["TELEMETRY_DIR"])
    monkeypatch.setenv("TELEMETRY_DIR", str(tmp_path / "from_env"))
    assert analyzer._dir("TELEMETRY_DIR") == tmp_path / "from_env"


def test_dir_falls_back_to_the_import_time_default(monkeypatch):
    monkeypatch.delenv("TELEMETRY_DIR", raising=False)
    monkeypatch.setattr(analyzer, "TELEMETRY_DIR", analyzer._IMPORT_TIME["TELEMETRY_DIR"])
    assert str(analyzer._dir("TELEMETRY_DIR")) == analyzer._IMPORT_TIME["TELEMETRY_DIR"]


def test_load_telemetry_reads_the_directory_the_env_var_points_at(monkeypatch, tmp_path):
    """The regression for the bug where the route globbed TELEMETRY_DIR per
    request while the analyzer captured it at import: every model came back
    `insufficient_data` with no error anywhere."""
    d = tmp_path / "telemetry"
    d.mkdir()
    (d / "m.jsonl").write_text(json.dumps(_point(50.0)) + "\n")

    monkeypatch.setattr(analyzer, "TELEMETRY_DIR", analyzer._IMPORT_TIME["TELEMETRY_DIR"])
    monkeypatch.setenv("TELEMETRY_DIR", str(d))
    assert len(analyzer.load_telemetry("m")) == 1


# --------------------------------------------------------------------------
# load_telemetry
# --------------------------------------------------------------------------


def test_load_telemetry_returns_nothing_for_an_unknown_model(telemetry):
    telemetry("other", [_point(50.0)])
    assert analyzer.load_telemetry("missing") == []


def test_load_telemetry_ignores_stale_points(telemetry):
    """A session from yesterday must not skew today's creep slope."""
    telemetry(
        "m",
        [
            _point(10.0, age_s=7200),
            _point(90.0, age_s=3601),
            _point(50.0, age_s=0),
        ],
    )
    points = analyzer.load_telemetry("m", max_age_seconds=3600)
    assert [p["system"]["ram_used_pct"] for p in points] == [50.0]


def test_load_telemetry_keeps_only_the_last_limit_points(telemetry):
    telemetry("m", [_point(float(i)) for i in range(20)])
    points = analyzer.load_telemetry("m", limit=5)
    assert [p["system"]["ram_used_pct"] for p in points] == [15.0, 16.0, 17.0, 18.0, 19.0]


def test_load_telemetry_skips_unparseable_lines(telemetry):
    d = telemetry("m", [_point(50.0)])
    with (d / "m.jsonl").open("a") as fh:
        fh.write("{not json\n\n")
    assert len(analyzer.load_telemetry("m")) == 1


def test_a_point_with_no_epoch_time_is_treated_as_stale(telemetry):
    p = _point(50.0)
    p.pop("epoch_time")
    telemetry("m", [p])
    assert analyzer.load_telemetry("m") == []


# --------------------------------------------------------------------------
# read_current_config
# --------------------------------------------------------------------------


def test_read_current_config_merges_defaults_then_the_model_section(tmp_path, monkeypatch):
    ini = tmp_path / "models.ini"
    ini.write_text("[*]\nn-gpu-layers = 99\nctx-size = 8192\n\n[m]\nctx-size = 32768\n")
    monkeypatch.setattr(analyzer, "ROUTER_MODELS_DIR", tmp_path)
    cfg = analyzer.read_current_config("m")
    assert cfg["n-gpu-layers"] == "99"  # from [*]
    assert cfg["ctx-size"] == "32768"  # section wins over the default


def test_read_current_config_matches_a_section_loosely(tmp_path, monkeypatch):
    ini = tmp_path / "models.ini"
    ini.write_text("[qwen3-35b--q4]\nctx-size = 16384\n")
    monkeypatch.setattr(analyzer, "ROUTER_MODELS_DIR", tmp_path)
    # clean_str() strips / _ - and lowercases, so an underscored alias still hits.
    assert analyzer.read_current_config("qwen3_35b_q4")["ctx-size"] == "16384"


def test_read_current_config_falls_back_to_the_profile_json(tmp_path, monkeypatch):
    monkeypatch.setattr(analyzer, "ROUTER_MODELS_DIR", tmp_path)
    (tmp_path / "m.profile.json").write_text(json.dumps({"quality": {"verdict": "ok"}}))
    cfg = analyzer.read_current_config("m")
    # nested values are stringified rather than dropped
    assert cfg["quality"] == "{'verdict': 'ok'}"


def test_a_profile_overlay_wins_over_models_ini(tmp_path, monkeypatch):
    ini = tmp_path / "models.ini"
    ini.write_text("[m]\nctx-size = 4096\n")
    (tmp_path / "m.profile.json").write_text(json.dumps({"ctx-size": 16384}))
    monkeypatch.setattr(analyzer, "ROUTER_MODELS_DIR", tmp_path)
    assert analyzer.read_current_config("m")["ctx-size"] == "16384"


def test_a_corrupt_ini_is_survivable(tmp_path, monkeypatch):
    (tmp_path / "models.ini").write_text("this is not = an ini [[[\n")
    monkeypatch.setattr(analyzer, "ROUTER_MODELS_DIR", tmp_path)
    assert isinstance(analyzer.read_current_config("m"), dict)


# --------------------------------------------------------------------------
# analyze_telemetry: the empty and healthy paths
# --------------------------------------------------------------------------


def test_no_telemetry_is_reported_as_insufficient_data_not_as_an_error(telemetry):
    telemetry("other", [_point(50.0)])
    out = analyzer.analyze_telemetry("missing")
    assert out["status"] == "insufficient_data"
    assert out["recommendations"] == {}
    assert out["detected_issues"] == ["No telemetry data found for this model."]


def test_a_quiet_model_gets_the_no_adjustment_explanation(telemetry):
    telemetry("m", [_point(20.0, vram=10.0, vram_used=1000, vram_total=8192) for _ in range(20)])
    out = analyzer.analyze_telemetry("m", current_config={"n-gpu-layers": "99", "ctx-size": "32768"})
    assert out["status"] == "ok"
    assert out["recommendations"] == {}
    assert out["tuning_strategy"] == "performance_first"
    assert "No tuning adjustments required" in out["explanation"]


def test_the_strategy_is_echoed_in_the_result(telemetry):
    telemetry("m", [_point(20.0)])
    assert analyzer.analyze_telemetry("m", performance_first=False)["tuning_strategy"] == "safe_first"


def test_the_metrics_summary_reports_the_window_it_saw(telemetry):
    telemetry("m", [_point(30.0, vram=40.0, vram_used=4000, vram_total=8192, cached=1234) for _ in range(3)])
    summary = analyzer.analyze_telemetry("m")["metrics_summary"]
    assert summary["system_ram"]["max_pct"] == 30.0
    assert summary["vram"]["total_mb"] == 8192
    assert summary["vram"]["headroom_mb"] == 4192
    assert summary["context_slots"]["max_tokens_cached"] == 1234


def test_a_point_with_no_gpus_is_treated_as_zero_not_skipped(telemetry):
    """One degraded telemetry sample must not remove the point from the series
    and silently shorten the window the creep slope is fitted over."""
    bare = _point(40.0)
    bare["gpus"] = []
    telemetry("m", [bare, _point(41.0), _point(42.0)])
    out = analyzer.analyze_telemetry("m")
    assert out["metrics_summary"]["vram"]["max_pct"] == 40.0


def test_a_point_with_no_slot_data_does_not_crash_the_aggregation(telemetry):
    """A sample taken before llama-server reported /slots has no tokens_cached.
    The comprehension filters it out and the [0] fallback keeps max() valid."""
    points = []
    for i in range(3):
        pt = _point(40.0 + i)
        pt["llama_server"] = {}
        points.append(pt)
    telemetry("m", points)
    out = analyzer.analyze_telemetry("m")
    assert out["metrics_summary"]["context_slots"]["max_tokens_cached"] == 0
    assert out["metrics_summary"]["system_ram"]["final_pct"] == 42.0


# --------------------------------------------------------------------------
# thresholds: performance_first vs safe_first
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("strategy", "perf", "expected"),
    [
        # performance_first tolerates 94% RAM / 95% VRAM as merely a warning.
        ("performance", 95.0, "warning"),
        ("performance", 99.0, "critical"),
        # safe_first warns at 85% and goes critical at 92%.
        ("safe", 86.0, "warning"),
        ("safe", 93.0, "critical"),
    ],
)
def test_the_two_threshold_sets_differ(telemetry, strategy, perf, expected):
    telemetry("m", [_point(perf, vram=10.0, vram_used=500, vram_total=8192) for _ in range(3)])
    out = analyzer.analyze_telemetry("m", performance_first=strategy == "performance")
    assert out["status"] == expected


def test_a_brief_ram_spike_is_critical_in_both_strategies(telemetry):
    """`max_ram > ram_critical + 1.0` deliberately ignores the strategy: a single
    99% sample means the host came close to swapping, and neither strategy wants
    to learn that only from the final sample."""
    telemetry("m", [_point(50.0), _point(99.0), _point(50.0)])
    perf = analyzer.analyze_telemetry("m", performance_first=True)
    safe = analyzer.analyze_telemetry("m", performance_first=False)
    assert perf["status"] == "critical"
    assert safe["status"] == "critical"
    assert "Peak: 99.0%" in perf["detected_issues"][0]


def test_vram_pressure_alone_is_reported(telemetry):
    telemetry(
        "m",
        [_point(10.0, vram=97.0, vram_used=7900, vram_total=8192) for _ in range(3)],
    )
    out = analyzer.analyze_telemetry("m", current_config={"n-gpu-layers": "40", "ctx-size": "32768"})
    assert out["status"] in ("warning", "critical")
    assert any("VRAM" in i for i in out["detected_issues"])


# --------------------------------------------------------------------------
# creep detection
# --------------------------------------------------------------------------


def test_sustained_ram_growth_is_flagged_as_creep_in_the_safe_strategy(telemetry):
    telemetry("m", [_point(40.0 + i * 2.0, vram=10.0, vram_used=500, vram_total=8192) for i in range(20)])
    out = analyzer.analyze_telemetry("m", performance_first=False)
    assert out["metrics_summary"]["system_ram"]["creep_slope"] > 0.05


def test_a_flat_series_has_a_zero_slope(telemetry):
    telemetry("m", [_point(50.0) for _ in range(20)])
    assert analyzer.analyze_telemetry("m")["metrics_summary"]["system_ram"]["creep_slope"] == 0.0


def test_creep_needs_ten_points_before_it_is_fitted(telemetry):
    telemetry("m", [_point(40.0 + i * 5.0) for i in range(9)])
    assert analyzer.analyze_telemetry("m")["metrics_summary"]["system_ram"]["creep_slope"] == 0.0


def test_a_decreasing_series_never_trips_creep(telemetry):
    telemetry("m", [_point(90.0 - i * 2.0) for i in range(20)])
    assert analyzer.analyze_telemetry("m")["metrics_summary"]["system_ram"]["creep_slope"] < 0


# --------------------------------------------------------------------------
# recommendation rules
# --------------------------------------------------------------------------


def test_high_ram_with_vram_headroom_pushes_layers_onto_the_gpu(telemetry):
    telemetry("m", [_point(96.0, vram=30.0, vram_used=3000, vram_total=8192) for _ in range(3)])
    out = analyzer.analyze_telemetry("m", current_config={"n-gpu-layers": "40", "ctx-size": "32768"})
    assert int(out["recommendations"]["n-gpu-layers"]) > 40


def test_high_ram_with_no_headroom_quantises_the_kv_cache_instead(telemetry):
    telemetry("m", [_point(96.0, vram=96.0, vram_used=7900, vram_total=8192) for _ in range(3)])
    out = analyzer.analyze_telemetry("m", current_config={"n-gpu-layers": "40", "ctx-size": "32768"})
    assert out["recommendations"]["cache-type-k"] == "q8_0"
    assert out["recommendations"]["cache-type-v"] == "q8_0"


def test_the_safe_strategy_jumps_straight_to_q4_0_cache(telemetry):
    telemetry("m", [_point(96.0, vram=96.0, vram_used=7900, vram_total=8192) for _ in range(3)])
    out = analyzer.analyze_telemetry(
        "m", current_config={"n-gpu-layers": "40", "ctx-size": "32768"}, performance_first=False
    )
    assert out["recommendations"]["cache-type-k"] == "q4_0"


def test_underused_vram_offers_more_gpu_layers(telemetry):
    """The headline optimisation, and it must apply to a HEALTHY model: idle
    VRAM and a partial CPU offload is the common case, not a distress state."""
    telemetry("m", [_point(10.0, vram=20.0, vram_used=1600, vram_total=8192) for _ in range(3)])
    out = analyzer.analyze_telemetry("m", current_config={"n-gpu-layers": "20", "ctx-size": "32768"})
    assert out["status"] == "ok"  # the model is not in distress
    assert int(out["recommendations"]["n-gpu-layers"]) > 20


def test_underused_vram_with_plenty_of_headroom_upgrades_the_cache(telemetry):
    telemetry("m", [_point(10.0, vram=20.0, vram_used=1600, vram_total=8192) for _ in range(3)])
    out = analyzer.analyze_telemetry(
        "m", current_config={"n-gpu-layers": "99", "ctx-size": "32768", "cache-type-k": "q4_0", "cache-type-v": "q4_0"}
    )
    assert out["recommendations"]["cache-type-k"] == "q8_0"  # performance stops at q8_0


def test_the_safe_strategy_upgrades_all_the_way_to_f16(telemetry):
    """safe_first explicitly wants full quality when there is >2000MB spare;
    performance_first stops at q8_0."""
    telemetry("m", [_point(10.0, vram=20.0, vram_used=1600, vram_total=8192) for _ in range(3)])
    out = analyzer.analyze_telemetry(
        "m",
        current_config={"n-gpu-layers": "99", "ctx-size": "32768", "cache-type-k": "q4_0", "cache-type-v": "q4_0"},
        performance_first=False,
    )
    assert out["recommendations"]["cache-type-k"] == "f16"
    assert out["recommendations"]["cache-type-v"] == "f16"


def test_vram_oom_with_ram_to_spare_pulls_layers_back(telemetry):
    telemetry("m", [_point(10.0, vram=99.0, vram_used=8150, vram_total=8192) for _ in range(3)])
    out = analyzer.analyze_telemetry("m", current_config={"n-gpu-layers": "40", "ctx-size": "32768"})
    assert int(out["recommendations"]["n-gpu-layers"]) < 40


def test_idle_vram_does_not_downgrade_an_f16_kv_cache(telemetry):
    """Rule 5 only moves up the quality ladder. Walking the candidates blindly
    had a model already on f16 being told to move to q8_0, described as an
    "upgrade ... to improve coherence"."""
    telemetry("m", [_point(10.0, vram=20.0, vram_used=1600, vram_total=8192) for _ in range(3)])
    out = analyzer.analyze_telemetry(
        "m",
        current_config={
            "n-gpu-layers": "99",
            "ctx-size": "32768",
            "cache-type-k": "f16",
            "cache-type-v": "f16",
        },
    )
    assert "cache-type-k" not in out["recommendations"]


def test_large_vram_headroom_offers_a_bigger_context_window(telemetry):
    telemetry("m", [_point(10.0, vram=20.0, vram_used=1000, vram_total=16384) for _ in range(3)])
    out = analyzer.analyze_telemetry("m", current_config={"n-gpu-layers": "99", "ctx-size": "16384"})
    assert int(out["recommendations"]["ctx-size"]) > 16384


def test_a_large_batch_size_is_reduced_when_memory_is_tight(telemetry):
    telemetry("m", [_point(96.0, vram=96.0, vram_used=7900, vram_total=8192) for _ in range(3)])
    out = analyzer.analyze_telemetry(
        "m",
        current_config={
            "n-gpu-layers": "40",
            "ctx-size": "4096",
            "cache-type-k": "q4_0",
            "cache-type-v": "q4_0",
            "batch-size": "2048",
        },
    )
    assert out["recommendations"].get("batch-size") == "512"


def test_a_float_formatted_ini_value_still_parses(telemetry):
    """llama.cpp writes some values as floats; int("99.0") would raise and take
    the whole recommendation path down with it."""
    telemetry("m", [_point(96.0, vram=30.0, vram_used=3000, vram_total=8192) for _ in range(3)])
    out = analyzer.analyze_telemetry("m", current_config={"n-gpu-layers": "40.0", "ctx-size": "32768"})
    # rule 1 would give 40 + 15 = 55; rules 4-6 may raise it further. The point
    # is that "40.0" parsed at all rather than raising ValueError.
    assert int(out["recommendations"]["n-gpu-layers"]) > 40


def test_a_nonsense_ini_value_falls_back_to_the_default(telemetry):
    telemetry("m", [_point(96.0, vram=30.0, vram_used=3000, vram_total=8192) for _ in range(3)])
    out = analyzer.analyze_telemetry("m", current_config={"n-gpu-layers": "banana"})
    assert int(out["recommendations"]["n-gpu-layers"]) == 99  # -1 default -> force 99


# --------------------------------------------------------------------------
# the failed-config blacklist
# --------------------------------------------------------------------------


def _blacklist(workdir, entries):
    (workdir / "data").mkdir(exist_ok=True)
    (workdir / "data" / "failed_configs.json").write_text(json.dumps(entries))


def test_a_known_failed_config_is_flagged_critical(telemetry, workdir):
    telemetry("qwen3-35b", [_point(20.0) for _ in range(3)])
    _blacklist(
        workdir,
        [
            {
                "model": "qwen3-35b--q4",
                "cache-type-k": "f16",
                "cache-type-v": "f16",
                "n-gpu-layers": "40",
                "ctx-size": "32768",
            }
        ],
    )
    out = analyzer.analyze_telemetry(
        "qwen3-35b",
        current_config={"n-gpu-layers": "40", "ctx-size": "32768", "cache-type-k": "f16", "cache-type-v": "f16"},
    )
    assert out["status"] == "critical"
    assert any("known to have failed" in i for i in out["detected_issues"])


def test_a_failed_config_is_never_recommended_back(telemetry, workdir):
    """The whole point of the blacklist: do not walk a model into a config that
    already failed to load."""
    telemetry("qwen3-35b", [_point(96.0, vram=96.0, vram_used=7900, vram_total=8192) for _ in range(3)])
    cfg = {"n-gpu-layers": "40", "ctx-size": "32768", "cache-type-k": "f16", "cache-type-v": "f16"}
    _blacklist(
        workdir,
        [
            {
                "model": "qwen3-35b--q4",
                "cache-type-k": "q8_0",
                "cache-type-v": "q8_0",
                "n-gpu-layers": "40",
                "ctx-size": "32768",
            }
        ],
    )
    out = analyzer.analyze_telemetry("qwen3-35b", current_config=dict(cfg))
    assert out["recommendations"]["cache-type-k"] != "q8_0"


def test_the_blacklist_only_matches_the_same_model(telemetry, workdir):
    # is_blacklisted matches on substrings in both directions, so the aliases
    # have to be realistic: a one-character alias is a substring of everything.
    telemetry("qwen3-35b", [_point(20.0) for _ in range(3)])
    _blacklist(
        workdir,
        [
            {
                "model": "qwen3-14b",
                "cache-type-k": "f16",
                "cache-type-v": "f16",
                "n-gpu-layers": "40",
                "ctx-size": "32768",
            }
        ],
    )
    out = analyzer.analyze_telemetry(
        "qwen3-35b",
        current_config={"n-gpu-layers": "40", "ctx-size": "32768", "cache-type-k": "f16", "cache-type-v": "f16"},
    )
    assert out["status"] != "critical"


def test_a_corrupt_blacklist_is_ignored_rather_than_fatal(telemetry, workdir):
    telemetry("m", [_point(20.0) for _ in range(3)])
    (workdir / "data").mkdir(exist_ok=True)
    (workdir / "data" / "failed_configs.json").write_text("{not json")
    out = analyzer.analyze_telemetry("m", current_config={"n-gpu-layers": "40", "ctx-size": "32768"})
    assert out["status"] in ("ok", "warning", "critical")
    assert "recommendations" in out


def test_the_failed_config_ladder_walks_f16_q8_q5_q4_in_order(telemetry, workdir):
    """Each rung is only recommended once the one above it is blacklisted."""
    telemetry("qwen3-35b", [_point(20.0) for _ in range(3)])
    entries = []
    for ck, cv in (("f16", "f16"), ("q8_0", "q8_0"), ("q5_0", "q5_0")):
        entries.append(
            {
                "model": "qwen3-35b",
                "cache-type-k": ck,
                "cache-type-v": cv,
                "n-gpu-layers": "-1",
                "ctx-size": "4096",
            }
        )
    _blacklist(workdir, entries)
    out = analyzer.analyze_telemetry(
        "qwen3-35b",
        current_config={"n-gpu-layers": "-1", "ctx-size": "4096", "cache-type-k": "q5_0", "cache-type-v": "q5_0"},
    )
    assert out["recommendations"]["cache-type-k"] == "q4_0"


# --------------------------------------------------------------------------
# load_latest_benchmark / baseline_comparison
# --------------------------------------------------------------------------


def test_no_benchmark_directory_means_no_baseline(telemetry, tmp_path, monkeypatch):
    telemetry("m", [_point(20.0) for _ in range(3)])
    monkeypatch.setattr(analyzer, "BENCHMARK_DIR", tmp_path / "nope")
    assert analyzer.load_latest_benchmark("m") is None
    assert analyzer.analyze_telemetry("m")["baseline_comparison"] == {}


def test_the_baseline_is_read_from_the_newest_result_file(telemetry, tmp_path, monkeypatch):
    telemetry("m", [_point(20.0) for _ in range(3)])
    bdir = tmp_path / "bench"
    bdir.mkdir()
    (bdir / "old.json").write_text(json.dumps({"results": [{"model": "m", "avg_ttft_ms": 900}]}))
    (bdir / "new.json").write_text(
        json.dumps({"results": [{"model": "m", "avg_ttft_ms": 120, "avg_tokens_per_sec": 44.0}]})
    )
    # mtimes must differ; the older file is written first
    import os
    import time as _t

    now = _t.time()
    os.utime(bdir / "old.json", (now - 500, now - 500))
    os.utime(bdir / "new.json", (now, now))
    monkeypatch.setattr(analyzer, "BENCHMARK_DIR", bdir)
    out = analyzer.analyze_telemetry("m")
    assert out["baseline_comparison"] == {"baseline_ttft_ms": 120, "baseline_tps": 44.0}


def test_a_corrupt_benchmark_file_degrades_to_an_empty_baseline(telemetry, tmp_path, monkeypatch):
    telemetry("m", [_point(20.0) for _ in range(3)])
    bdir = tmp_path / "bench"
    bdir.mkdir()
    (bdir / "x.json").write_text("{not json")
    monkeypatch.setattr(analyzer, "BENCHMARK_DIR", bdir)
    assert analyzer.analyze_telemetry("m")["baseline_comparison"] == {}


# --------------------------------------------------------------------------
# result shape — the dashboard reads these keys directly
# --------------------------------------------------------------------------


def test_the_result_carries_every_key_the_dashboard_renders(telemetry):
    telemetry("m", [_point(20.0) for _ in range(3)])
    out = analyzer.analyze_telemetry("m")
    for key in (
        "status",
        "tuning_strategy",
        "model_alias",
        "metrics_summary",
        "baseline_comparison",
        "detected_issues",
        "recommendations",
        "explanation",
    ):
        assert key in out, key
    assert out["model_alias"] == "m"
    assert isinstance(out["recommendations"], dict)


def test_recommendations_are_all_strings(telemetry):
    """They are written straight into a models.ini section, so an int would
    render as `1234` only by luck of str() and a float would break the ini."""
    telemetry("m", [_point(96.0, vram=96.0, vram_used=7900, vram_total=8192) for _ in range(3)])
    out = analyzer.analyze_telemetry("m", current_config={"n-gpu-layers": "40", "ctx-size": "32768"})
    for k, v in out["recommendations"].items():
        assert isinstance(v, str), k
        assert v.strip() != "", k


def test_a_healthy_model_still_lists_an_issue_entry(telemetry):
    """The dashboard renders `detected_issues` as a list; an empty list would
    collapse the card."""
    telemetry("m", [_point(20.0) for _ in range(3)])
    assert analyzer.analyze_telemetry("m")["detected_issues"] == ["No resource utilization issues detected."]
