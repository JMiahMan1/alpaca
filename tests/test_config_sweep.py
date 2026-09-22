"""Unit tests for the prefill (batch size) stage of benchmark-configs.py.

The module name has a hyphen, so it is loaded by path rather than imported.
"""

import configparser
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sweeper():
    spec = importlib.util.spec_from_file_location("benchmark_configs", REPO / "benchmark-configs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config(tmp_path, sweeper, monkeypatch):
    ini = tmp_path / "models.ini"
    ini.write_text("[*]\ntemperature = 0.6\n\n[m]\nctx-size = 8192\n")
    monkeypatch.setattr(sweeper, "INI_PATH", ini)
    monkeypatch.setattr(sweeper, "restart_backend", lambda: True)
    parser = configparser.ConfigParser()
    parser.read(ini)
    return parser


def test_prefill_sweep_ranks_fastest_pair_first(sweeper, config, monkeypatch):
    # Time-to-first-token per (batch, ubatch); lower is faster.
    ttft_by_pair = {(2048, 512): 2.0, (2048, 1024): 1.0, (1024, 1024): 4.0}

    def fake_run_test(public, prompt=None, n_predict=64, timeout=45.0):
        pair = (int(config["m"]["batch-size"]), int(config["m"]["ubatch-size"]))
        return True, ttft_by_pair[pair], 10.0, ""

    monkeypatch.setattr(sweeper, "run_test", fake_run_test)
    rows = sweeper.sweep_batch_sizes("m", "m:q4", config, list(ttft_by_pair))

    assert [(r["batch"], r["ubatch"]) for r in rows] == [(2048, 1024), (2048, 512), (1024, 1024)]
    assert rows[0]["pp"] == pytest.approx(sweeper.LONG_PROMPT_TOKENS / 1.0)


def test_prefill_sweep_skips_physical_batch_larger_than_logical(sweeper, config, monkeypatch):
    monkeypatch.setattr(sweeper, "run_test", lambda *a, **k: (True, 1.0, 10.0, ""))
    rows = sweeper.sweep_batch_sizes("m", "m:q4", config, [(512, 2048), (2048, 512)])
    assert [(r["batch"], r["ubatch"]) for r in rows] == [(2048, 512)]


def test_prefill_sweep_records_failures_last(sweeper, config, monkeypatch):
    def fake_run_test(public, prompt=None, n_predict=64, timeout=45.0):
        if config["m"]["ubatch-size"] == "1024":
            return False, None, None, "out of memory"
        return True, 2.0, 10.0, ""

    monkeypatch.setattr(sweeper, "run_test", fake_run_test)
    rows = sweeper.sweep_batch_sizes("m", "m:q4", config, [(2048, 1024), (2048, 512)])

    assert rows[0]["status"] == "PASS"
    assert rows[-1]["status"] == "FAIL"
    assert rows[-1]["detail"] == "out of memory"


def test_long_prompt_is_long_enough_to_measure_prefill(sweeper):
    """A one-line prompt makes TTFT scheduling noise, which is why the stage
    exists at all - guard the filler against being trimmed away."""
    assert len(sweeper.LONG_PROMPT.split()) > 1000
