"""Piper voices on POST /api/tts ("piper:<name>"), with a fake PiperVoice.

No real model is needed: a fake ``piper`` module is put in sys.modules and the
voices directory is a tmp_path holding empty <name>.onnx + <name>.onnx.json.
"""

import base64
import sys
import types
import wave
from io import BytesIO
from pathlib import Path
from typing import ClassVar
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import audio_server as audio

TestClient = pytest.importorskip("fastapi.testclient").TestClient

PIPER_SR = 22050


class _FakePiperVoice:
    loads: ClassVar[list] = []

    def __init__(self, model_path):
        self.model_path = model_path
        self.texts: list = []
        self.syn_configs: list = []

    @classmethod
    def load(cls, model_path, config_path=None, **_kw):
        voice = cls(model_path)
        cls.loads.append(voice)
        return voice

    def synthesize_wav(self, text, wav_file, syn_config=None):
        self.texts.append(text)
        self.syn_configs.append(syn_config)
        n = int(0.3 * PIPER_SR)
        t = np.arange(n) / PIPER_SR
        pcm = (0.4 * np.sin(2 * np.pi * 200 * t) * 32767).astype("<i2")
        wav_file.setframerate(PIPER_SR)
        wav_file.setsampwidth(2)
        wav_file.setnchannels(1)
        wav_file.writeframes(pcm.tobytes())


class _FakeSynthesisConfig:
    def __init__(self, length_scale=None):
        self.length_scale = length_scale


@pytest.fixture
def fake_piper(monkeypatch):
    mod = types.ModuleType("piper")
    mod.PiperVoice = _FakePiperVoice  # type: ignore[attr-defined]
    mod.SynthesisConfig = _FakeSynthesisConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "piper", mod)
    _FakePiperVoice.loads = []
    return mod


@pytest.fixture
def voices_dir(tmp_path, monkeypatch):
    for name in ("en_US-hfc_male-medium", "en_GB-alan-low"):
        (tmp_path / f"{name}.onnx").write_bytes(b"")
        (tmp_path / f"{name}.onnx.json").write_text("{}")
    (tmp_path / "half-a-voice.onnx").write_bytes(b"")  # no config: not a voice
    monkeypatch.setattr(audio, "PIPER_VOICES_DIR", str(tmp_path))
    monkeypatch.setattr(audio, "_piper_voices", {})
    return tmp_path


@pytest.fixture
def client():
    return TestClient(audio.app)


def _kokoro_pipe():
    class R:
        def __init__(self, a):
            self.audio = a

    def call(_text, voice=None, speed=1.0, **_kw):
        t = np.arange(int(0.3 * 24000)) / 24000
        return [R((0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))]

    return call


def test_a_piper_voice_is_synthesized_at_its_own_sample_rate(client, fake_piper, voices_dir):
    with patch.object(audio, "_ensure_model", AsyncMock()) as ensure:
        resp = client.post("/api/tts", json={"text": "One. Two.", "voice": "piper:en_US-hfc_male-medium"})
    assert resp.status_code == 200, resp.text
    assert ensure.call_count == 0  # Kokoro is never loaded for a Piper voice
    body = resp.json()
    assert body["mime"] == "audio/wav"
    with wave.open(BytesIO(base64.b64decode(body["audio_b64"])), "rb") as wf:
        assert wf.getframerate() == PIPER_SR
        assert 0.8 < wf.getnframes() / PIPER_SR < 2.0
    meta = body["meta"]
    assert meta["engine"] == "piper" and meta["voice"] == "piper:en_US-hfc_male-medium"
    assert meta["sample_rate"] == PIPER_SR and meta["chunks"] == 2 and meta["clone"] is None


def test_a_piper_voice_is_loaded_once_and_cached(client, fake_piper, voices_dir):
    for _ in range(2):
        assert client.post("/api/tts", json={"text": "Hi.", "voice": "piper:en_GB-alan-low"}).status_code == 200
    assert len(_FakePiperVoice.loads) == 1
    assert _FakePiperVoice.loads[0].model_path.endswith("en_GB-alan-low.onnx")


def test_piper_reads_the_normalized_text_without_kokoro_markup(client, fake_piper, voices_dir):
    resp = client.post(
        "/api/tts",
        json={"text": "Read John 3:16 with [Bresee](/bɹəzi/).", "voice": "piper:en_GB-alan-low"},
    )
    assert resp.status_code == 200
    spoken = " ".join(_FakePiperVoice.loads[0].texts)
    assert "chapter 3" in spoken and "Bresee" in spoken
    assert "](/" not in spoken and "[" not in spoken


def test_speed_becomes_piper_length_scale(client, fake_piper, voices_dir):
    client.post("/api/tts", json={"text": "Hi.", "voice": "piper:en_GB-alan-low", "speed": 2.0})
    assert _FakePiperVoice.loads[0].syn_configs[0].length_scale == 0.5


def test_health_lists_the_piper_voices(client, fake_piper, voices_dir):
    body = client.get("/health").json()
    assert body["piper"]["voices"] == ["piper:en_GB-alan-low", "piper:en_US-hfc_male-medium"]
    assert body["piper"]["available"] is True and body["piper"]["error"] is None
    assert body["tts"]["voices"] == audio.KOKORO_VOICES  # Kokoro list unchanged


def test_an_unknown_piper_voice_is_a_400_naming_the_ones_that_exist(client, fake_piper, voices_dir):
    resp = client.post("/api/tts", json={"text": "Hi.", "voice": "piper:nope"})
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert "piper:nope" in err and "piper:en_GB-alan-low" in err and "piper:en_US-hfc_male-medium" in err
    assert "half-a-voice" not in err


@pytest.mark.parametrize("voice", ["piper:en_GB-alan-low,af_heart", "af_heart,piper:en_GB-alan-low"])
def test_a_blend_with_a_piper_voice_is_a_400(client, fake_piper, voices_dir, voice):
    resp = client.post("/api/tts", json={"text": "Hi.", "voice": voice})
    assert resp.status_code == 400 and "cannot be blended" in resp.json()["error"]
    assert _FakePiperVoice.loads == []


@pytest.mark.parametrize("extra", [{"clone": "abc"}, {"clone_tau": 0.5}, {"clone_seed": 3}])
def test_clone_parameters_with_a_piper_voice_are_a_400(client, fake_piper, voices_dir, extra):
    resp = client.post("/api/tts", json={"text": "Hi.", "voice": "piper:en_GB-alan-low", **extra})
    assert resp.status_code == 400 and "Piper voice" in resp.json()["error"]


def test_a_missing_piper_package_is_a_503_and_kokoro_still_works(client, voices_dir, monkeypatch):
    monkeypatch.setitem(sys.modules, "piper", None)  # import piper -> ImportError
    resp = client.post("/api/tts", json={"text": "Hi.", "voice": "piper:en_GB-alan-low"})
    assert resp.status_code == 503 and "piper-tts is not installed" in resp.json()["error"]
    assert client.get("/health").json()["piper"]["available"] is False
    with patch.object(audio, "_ensure_model", AsyncMock(return_value=_kokoro_pipe())):
        ok = client.post("/api/tts", json={"text": "Hi.", "voice": "af_heart"})
    assert ok.status_code == 200 and ok.json()["meta"]["engine"] == "kokoro"


def test_a_missing_voices_directory_is_a_503_and_kokoro_still_works(client, fake_piper, tmp_path, monkeypatch):
    missing = tmp_path / "no-such-dir"
    monkeypatch.setattr(audio, "PIPER_VOICES_DIR", str(missing))
    monkeypatch.setattr(audio, "_piper_voices", {})
    resp = client.post("/api/tts", json={"text": "Hi.", "voice": "piper:en_GB-alan-low"})
    assert resp.status_code == 503 and str(missing) in resp.json()["error"]
    status = client.get("/health").json()["piper"]
    assert status["voices"] == [] and status["available"] is False
    with patch.object(audio, "_ensure_model", AsyncMock(return_value=_kokoro_pipe())):
        ok = client.post("/api/tts", json={"text": "Hi."})
    assert ok.status_code == 200 and ok.json()["meta"]["engine"] == "kokoro"
