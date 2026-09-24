import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import voice_clone as vc  # noqa: E402


def _speechlike(seconds: float, noise: float = 0.001, seed: int = 0):
    """Amplitude-modulated tone bursts with silent gaps, over a noise floor."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * vc.SR)) / vc.SR
    bursts = (np.sin(2 * np.pi * 0.25 * t) > 0).astype(np.float32)  # 2 s on / 2 s off
    voice = 0.3 * np.sin(2 * np.pi * 180 * t) * bursts
    return (voice + noise * rng.standard_normal(len(t))).astype(np.float32)


def test_analyze_keeps_speech_and_reports_quality():
    a = vc.analyze(_speechlike(10.0))
    rep = a["report"]
    assert rep["duration_s"] == 10.0
    assert 4.0 < rep["speech_s"] < 8.0          # gaps removed, hangover kept
    assert rep["snr_db"] > 30 and rep["warnings"] == []
    assert rep["clipping_pct"] == 0.0


def test_analyze_flags_noise_and_clipping():
    noisy = vc.analyze(_speechlike(6.0, noise=0.08))["report"]
    assert any("noise" in w for w in noisy["warnings"])
    loud = np.clip(_speechlike(6.0) * 5, -1, 1)
    assert any("clips" in w for w in vc.analyze(loud)["report"]["warnings"])


def test_windows_cover_all_speech():
    speech = np.zeros(int(25.5 * vc.SR), dtype=np.float32)
    ws = vc._windows(speech)
    assert len(ws) == 3 and sum(len(w) for w in ws) == len(speech)


@pytest.mark.parametrize("pid", ["../etc", "a/b", "", "X", "UPPER-case", "x" * 80])
def test_profile_ids_cannot_escape_storage(pid):
    with pytest.raises(KeyError):
        vc._pdir(pid)


def test_prompts_are_complete():
    ids = [p["id"] for p in vc.PROMPTS]
    assert len(ids) == len(set(ids)) >= 3
    assert all(p["text"] and p["min_s"] > 0 and p["title"] and p["tip"] for p in vc.PROMPTS)
