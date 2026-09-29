"""Unit tests for audio_server.py - the 606-line TTS/music service that shipped with ZERO tests.

Only torch, kokoro, transformers and soundfile are unavailable on a test host, and
every one of them is imported *inside* a function, so the whole module is importable
and most of it is exercisable with numpy + a fake pipeline. What is left needs a
fake ``torch`` / ``kokoro`` injected into sys.modules.
"""

import sys
import types
import wave
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import audio_server as audio

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient

SR = 24000


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _clean_state():
    """_state is module-global and mutated by every route; snapshot and restore it."""
    before = {
        "tts": audio._state["tts"],
        "music": audio._state["music"],
        "loading": set(audio._state["loading"]),  # type: ignore[arg-type]
        "last_used": dict(audio._state["last_used"]),  # type: ignore[arg-type]
    }
    yield
    audio._state["tts"] = before["tts"]
    audio._state["music"] = before["music"]
    audio._state["loading"].clear()  # type: ignore[union-attr]
    audio._state["loading"].update(before["loading"])  # type: ignore[union-attr]
    audio._state["last_used"].clear()  # type: ignore[union-attr]
    audio._state["last_used"].update(before["last_used"])  # type: ignore[union-attr]


@pytest.fixture
def client():
    # Deliberately NOT a context manager: entering would fire the startup hook and
    # spawn the real idle-unloader task.
    return TestClient(audio.app)


class _Result:
    """Stands in for a Kokoro pipeline result (only ``.audio`` is read)."""

    def __init__(self, audio_arr):
        self.audio = audio_arr


def _pipe(seconds: float = 0.3, sr: int = SR, drop: bool = False):
    """A callable standing in for KPipeline: yields one _Result per call."""

    def call(_text, voice=None, speed=1.0, **kwargs):
        if drop:
            return []
        n = max(int(seconds * sr / max(speed, 0.01)), 1)
        t = np.arange(n) / sr
        return [_Result((0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))]

    call.calls = []  # type: ignore[attr-defined]

    def recorder(_text, voice=None, speed=1.0, **kwargs):
        call.calls.append({"voice": voice, "speed": speed})  # type: ignore[attr-defined]
        return call(_text, voice=voice, speed=speed, **kwargs)

    recorder.calls = call.calls  # type: ignore[attr-defined]
    return recorder


def _fake_torch(cuda: bool = True):
    mod = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    mod.no_grad = _NoGrad  # type: ignore[attr-defined]
    mod.manual_seed = Mock()  # type: ignore[attr-defined]
    mod.cuda = types.SimpleNamespace(is_available=lambda: cuda)  # type: ignore[attr-defined]
    return mod


class _Arr:
    """Wrapper giving a numpy array the torch tensor method chain the route calls."""

    def __init__(self, a):
        self._a = a

    def __getitem__(self, i):
        return _Arr(self._a[i])

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self._a)

    def astype(self, dtype):
        return np.asarray(self._a).astype(dtype)


def _music_bundle(samples: int = 32000, sampling_rate: int = 32000):
    model = Mock()
    model.config.audio_encoder.sampling_rate = sampling_rate
    captured = {}

    def generate(**kwargs):
        captured.update(kwargs)
        return _Arr(np.zeros((1, 1, samples), dtype=np.float32))

    model.generate.side_effect = generate
    proc = Mock()
    proc.return_value.to.return_value = {}
    return {"model": model, "processor": proc}, captured


# --------------------------------------------------------------------------- #
# _wav_bytes                                                                  #
# --------------------------------------------------------------------------- #


def test_wav_bytes_writes_a_mono_16bit_header_at_the_requested_rate():
    raw = audio._wav_bytes(np.full(SR, 0.5, dtype=np.float32), SR)
    with wave.open(BytesIO(raw), "rb") as wf:
        assert (wf.getnchannels(), wf.getsampwidth(), wf.getframerate()) == (1, 2, SR)
        assert wf.getnframes() == SR


def test_wav_bytes_peak_normalises_so_a_quiet_signal_still_fills_the_range():
    quiet = audio._wav_bytes(np.full(SR, 0.01, dtype=np.float32), SR)
    loud = audio._wav_bytes(np.full(SR, 0.9, dtype=np.float32), SR)
    # Identical shapes; a peak-normalising encoder erases the level difference.
    assert quiet == loud


def test_wav_bytes_preserves_the_sign_of_negative_samples():
    import array

    raw = audio._wav_bytes(np.array([-1.0, 1.0, 0.0, 0.5] * 100, dtype=np.float32), SR)
    with wave.open(BytesIO(raw), "rb") as wf:
        pcm = array.array("h")
        pcm.frombytes(wf.readframes(wf.getnframes()))
    assert pcm[0] < 0 and pcm[1] > 0 and pcm[2] == 0  # signed 16-bit, not unsigned


def test_wav_bytes_on_an_empty_array_still_produces_a_playable_file():
    raw = audio._wav_bytes(np.array([], dtype=np.float32), SR)
    with wave.open(BytesIO(raw), "rb") as wf:
        assert wf.getnframes() == 0


# --------------------------------------------------------------------------- #
# _trim_and_fade                                                              #
# --------------------------------------------------------------------------- #


def test_trim_and_fade_removes_leading_and_trailing_silence():
    silent = int(0.25 * SR)
    body = np.concatenate([np.zeros(silent, np.float32), np.full(SR, 0.5, np.float32), np.zeros(silent, np.float32)])
    out = audio._trim_and_fade(body, SR)
    assert len(out) < len(body)
    assert abs(np.flatnonzero(np.abs(out) > 0.004)[0]) < 0.05 * SR


def test_trim_and_fade_returns_empty_for_digital_silence():
    assert audio._trim_and_fade(np.zeros(SR, dtype=np.float32), SR).size == 0


def test_trim_and_fade_ramps_both_edges_so_the_join_cannot_click():
    out = audio._trim_and_fade(np.full(SR, 0.5, dtype=np.float32), SR, margin_s=0.0, fade_s=0.01)
    head = out[: int(0.01 * SR)]
    assert head[0] == pytest.approx(0.0, abs=1e-6)
    assert head.max() == pytest.approx(0.5, rel=1e-3)


# --------------------------------------------------------------------------- #
# Device / VRAM helpers (no torch on a test host)                              #
# --------------------------------------------------------------------------- #


def test_device_falls_back_to_cpu_when_cuda_is_requested_but_absent():
    with patch.object(audio, "DEVICE", "cuda"), patch.object(audio, "_torch", lambda: types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))):
        assert audio._device() == "cpu"


def test_device_honours_an_explicit_non_cuda_request():
    with patch.object(audio, "DEVICE", "cpu"), patch.object(audio, "_torch", lambda: types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))):
        assert audio._device() == "cpu"


def test_free_vram_reports_none_none_without_cuda():
    assert audio._free_vram_mb() == (None, None)


def test_empty_cache_swallows_a_broken_torch_instead_of_raising():
    def boom():
        raise RuntimeError("no torch here")

    with patch.object(audio, "_torch", boom):
        audio._empty_cache()  # must not raise


# --------------------------------------------------------------------------- #
# _unload_model / _ensure_model                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_unload_reports_false_when_nothing_was_loaded():
    assert await audio._unload_model("tts") is False
    assert await audio._unload_model("music") is False


@pytest.mark.asyncio
async def test_unloading_tts_also_drops_the_openvoice_converter():
    audio._state["tts"] = object()
    with patch.object(audio.voice_clone, "unload", Mock()) as unload:
        assert await audio._unload_model("tts") is True
    assert unload.call_count == 1
    assert audio._state["tts"] is None


@pytest.mark.asyncio
async def test_unloading_tts_when_only_the_converter_is_loaded_still_reports_true():
    audio._state["tts"] = None
    with patch.object(audio.voice_clone, "loaded", lambda: True), patch.object(audio.voice_clone, "unload", Mock()) as unload:
        assert await audio._unload_model("tts") is True
    assert unload.call_count == 1


@pytest.mark.asyncio
async def test_unloading_music_leaves_tts_alone():
    tts, music = object(), object()
    audio._state["tts"], audio._state["music"] = tts, music
    assert await audio._unload_model("music") is True
    assert audio._state["music"] is None
    assert audio._state["tts"] is tts


@pytest.mark.asyncio
async def test_ensure_model_returns_the_cached_pipeline_and_refreshes_last_used():
    pipe = object()
    audio._state["tts"] = pipe
    audio._state["last_used"]["tts"] = 0.0
    assert await audio._ensure_model("tts") is pipe
    assert audio._state["last_used"]["tts"] > 0.0


@pytest.mark.asyncio
async def test_ensure_model_evicts_the_music_model_before_loading_tts():
    audio._state["music"] = object()
    fake = types.ModuleType("kokoro")
    fake.KPipeline = Mock(return_value="tts-pipe")  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"kokoro": fake, "torch": _fake_torch()}):
        assert await audio._ensure_model("tts") == "tts-pipe"
    assert audio._state["music"] is None
    assert audio._state["tts"] == "tts-pipe"
    assert audio._state["loading"] == set()  # cleaned up even though nothing raised


@pytest.mark.asyncio
async def test_ensure_model_refuses_a_second_concurrent_load():
    audio._state["loading"].add("music")  # type: ignore[union-attr]
    with pytest.raises(RuntimeError, match="already loading"):
        await audio._ensure_model("music")


@pytest.mark.asyncio
async def test_ensure_model_clears_the_loading_marker_when_the_load_raises():
    fake = types.ModuleType("kokoro")
    fake.KPipeline = Mock(side_effect=OSError("weights missing"))  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"kokoro": fake, "torch": _fake_torch()}), pytest.raises(OSError):
        await audio._ensure_model("tts")
    assert audio._state["loading"] == set()


# --------------------------------------------------------------------------- #
# /health and /api/status                                                      #
# --------------------------------------------------------------------------- #


def test_health_reports_both_engines_and_the_voice_list(client):
    with patch.object(audio.voice_clone, "list_profiles", return_value=[{"id": "a"}]):
        body = client.get("/health").json()
    assert body["status"] == "ok" and body["service"] == "audio-server"
    assert body["tts"] == {"model": audio.TTS_MODEL_ID, "loaded": False, "voices": audio.KOKORO_VOICES}
    assert body["clone"] == {"engine": "openvoice-v2", "loaded": False, "profiles": 1}
    assert body["music"]["max_duration_s"] == audio.MAX_MUSIC_SECONDS
    assert body["music"]["presets"] == audio.MUSIC_PRESETS
    assert body["vram_free_mb"] is None  # no CUDA on a test host


def test_api_status_is_the_same_document_as_health(client):
    """The two paths are stacked on one handler, so they must be the same
    document. ``uptime_s`` is recomputed per call, so it is compared
    separately - asserting whole-document equality made this test fail about
    one run in ten, on nothing but the clock."""
    health = client.get("/health").json()
    status = client.get("/api/status").json()
    assert set(health) == set(status)
    assert {k: v for k, v in health.items() if k != "uptime_s"} == {k: v for k, v in status.items() if k != "uptime_s"}
    assert 0 <= status["uptime_s"] <= health["uptime_s"] + 1.0


def test_health_reports_a_loaded_model_and_the_real_uptime(client):
    audio._state["tts"] = object()
    body = client.get("/health").json()
    assert body["tts"]["loaded"] is True
    assert body["uptime_s"] >= 0.0


# --------------------------------------------------------------------------- #
# /api/tts - validation happens before any model is touched                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("payload", "needle"),
    [
        ({"text": ""}, "text is required"),
        ({"text": "   "}, "text is required"),
        ({"text": "x" * (audio.MAX_TTS_CHARS + 1)}, "exceeds"),
        ({"text": "hi", "voice": "af_nope"}, "unknown voice"),
        ({"text": "hi", "voice": ""}, "unknown voice"),
        ({"text": "hi", "voice": "af_heart, af_nope"}, "af_nope"),
        ({"text": "hi", "speed": 0.49}, "speed must be"),
        ({"text": "hi", "speed": 2.01}, "speed must be"),
        ({"text": "hi", "sentence_pause_s": 3.5}, "pauses must be"),
        ({"text": "hi", "paragraph_pause_s": 6.0}, "pauses must be"),
        ({"text": "hi", "sentence_pause_s": -0.1}, "pauses must be"),
    ],
)
def test_tts_rejects_bad_requests_before_loading_a_model(client, payload, needle):
    with patch.object(audio, "_ensure_model", AsyncMock()) as ensure:
        resp = client.post("/api/tts", json=payload)
    assert resp.status_code == 400 and needle in resp.json()["error"]
    assert ensure.call_count == 0  # a 400 must never cost a 2 GB model load


def test_tts_accepts_a_blend_of_known_voices(client):
    pipe = _pipe()
    with patch.object(audio, "_ensure_model", AsyncMock(return_value=pipe)):
        resp = client.post("/api/tts", json={"text": "Hello there.", "voice": "af_heart, am_adam"})
    assert resp.status_code == 200
    # Blending is ONE voice: the whole blend string is handed to Kokoro, which
    # averages the style vectors. Two hosts must be two requests.
    assert pipe.calls[0]["voice"] == "af_heart,am_adam"


def test_tts_reports_a_load_failure_as_503(client):
    with patch.object(audio, "_ensure_model", AsyncMock(side_effect=OSError("no CUDA"))):
        resp = client.post("/api/tts", json={"text": "hello"})
    assert resp.status_code == 503 and "failed to load" in resp.json()["error"]


def test_tts_returns_502_when_the_pipeline_emits_nothing(client):
    with patch.object(audio, "_ensure_model", AsyncMock(return_value=_pipe(drop=True))):
        resp = client.post("/api/tts", json={"text": "hello"})
    assert resp.status_code == 502 and resp.json()["error"] == "TTS produced no audio"


def test_tts_synthesises_a_wav_and_reports_honest_meta(client):
    pipe = _pipe(seconds=0.4)
    with patch.object(audio, "_ensure_model", AsyncMock(return_value=pipe)):
        resp = client.post("/api/tts", json={"text": "One. Two.", "speed": 1.0, "sentence_pause_s": 0.0})
    body = resp.json()
    assert resp.status_code == 200 and body["mime"] == "audio/wav"
    raw = __import__("base64").b64decode(body["audio_b64"])
    with wave.open(BytesIO(raw), "rb") as wf:
        assert wf.getframerate() == SR
        # 2 sentences + the 0.15 s lead-in/out pads, with a zero sentence pause.
        assert 0.9 < wf.getnframes() / SR < 2.2
    meta = body["meta"]
    assert meta["engine"] == "kokoro" and meta["voice"] == "af_heart"
    assert meta["chunks"] == 2 and meta["sample_rate"] == SR
    assert meta["rtf"] > 0 and meta["normalized"] is True and meta["clone"] is None


def test_tts_normalization_is_on_by_default_and_can_be_switched_off(client):
    """meta.chars counts the text actually spoken, so a scripture citation that
    normalized from "1 Cor. 13:4-7" to "First Corinthians chapter 13, verses 4
    through 7" reports the expanded length."""
    import tts_text

    spoken = "Read 1 Cor. 13:4-7."
    pipe = _pipe()
    with patch.object(audio, "_ensure_model", AsyncMock(return_value=pipe)):
        on = client.post("/api/tts", json={"text": spoken}).json()
        off = client.post("/api/tts", json={"text": spoken, "normalize": False}).json()
    assert on["meta"]["chars"] == len(tts_text.normalize(spoken)) > len(spoken)
    assert off["meta"]["chars"] == len(spoken)
    assert on["meta"]["normalized"] is True and off["meta"]["normalized"] is False


def test_tts_clamps_speed_into_the_kokoro_range(client):
    pipe = _pipe()
    with patch.object(audio, "_ensure_model", AsyncMock(return_value=pipe)):
        fast = client.post("/api/tts", json={"text": "hi", "speed": 2.0}).json()
        slow = client.post("/api/tts", json={"text": "hi", "speed": 0.5}).json()
    assert fast["meta"]["duration_s"] < slow["meta"]["duration_s"]


# --------------------------------------------------------------------------- #
# /api/tts + OpenVoice clone                                                   #
# --------------------------------------------------------------------------- #


def test_tts_rejects_an_unknown_clone_id(client):
    with patch.object(audio, "_ensure_model", AsyncMock()) as ensure, patch.object(audio.voice_clone, "get_profile", side_effect=KeyError("nope")):
        resp = client.post("/api/tts", json={"text": "hi", "clone": "ghost-abc123"})
    assert resp.status_code == 404 and "ghost-abc123" in resp.json()["error"]
    assert ensure.call_count == 0


@pytest.mark.parametrize("tau", [0.09, 1.01])
def test_tts_rejects_a_clone_tau_outside_the_openvoice_range(client, tau):
    with patch.object(audio.voice_clone, "get_profile", return_value={"id": "v", "name": "V"}):
        resp = client.post("/api/tts", json={"text": "hi", "clone": "v", "clone_tau": tau})
    assert resp.status_code == 400 and "clone_tau" in resp.json()["error"]


def test_tts_re_timbres_every_chunk_through_openvoice_and_records_it(client):
    pipe = _pipe(seconds=0.3)
    src_se, tgt_se = object(), object()
    with (
        patch.object(audio, "_ensure_model", AsyncMock(return_value=pipe)),
        patch.object(audio.voice_clone, "get_profile", return_value={"id": "v", "name": "Voicey"}),
        patch.object(audio.voice_clone, "source_se", Mock(return_value=src_se)) as src,
        patch.object(audio.voice_clone, "target_se", Mock(return_value=tgt_se)),
        patch.object(audio.voice_clone, "convert", Mock(side_effect=lambda a, *r: a)) as conv,
        patch.object(audio.voice_clone, "correct_pitch", Mock(side_effect=lambda a, s, t: (a, {"corrected": False}))) as cp,
        patch.object(audio.voice_clone, "clone_similarity", Mock(return_value=None)),
    ):
        resp = client.post("/api/tts", json={"text": "One. Two.", "clone": "v", "sentence_pause_s": 0.0})
    assert resp.status_code == 200
    meta = resp.json()["meta"]["clone"]
    assert (meta["id"], meta["name"], meta["tau"]) == ("v", "Voicey", audio.voice_clone.DEFAULT_TAU)
    # The source embedding is derived from the *source* Kokoro voice, by
    # synthesising the read-aloud script once (then caching it per voice).
    assert src.call_args.args[0] == "af_heart"
    assert callable(src.call_args.args[1])  # the _synth_source callback
    assert conv.call_count == 2  # one convert per sentence
    # Pitch is corrected once, on the merged render, not per sentence: it is the
    # same shift for all of them and one vocoder pass is cheaper and steadier.
    assert cp.call_count == 1


def test_the_clone_report_says_what_was_measured(client):
    """A listener who thinks the clone is wrong needs to see WHICH side of the
    comparison is at fault, so the numbers are reported rather than asserted."""
    profile = {"id": "v", "name": "V", "median_f0_hz": 101.7, "intraspeaker_spread": 0.09}
    pitch = {"target_f0_hz": 101.7, "measured_f0_hz": 112.2,
             "applied_semitones": -1.7, "corrected": True}
    fixed = np.zeros(2400, dtype=np.float32)
    with (
        patch.object(audio, "_ensure_model", AsyncMock(return_value=_pipe(seconds=0.3))),
        patch.object(audio.voice_clone, "get_profile", return_value=profile),
        patch.object(audio.voice_clone, "source_se", Mock(return_value=object())),
        patch.object(audio.voice_clone, "target_se", Mock(return_value=object())),
        patch.object(audio.voice_clone, "convert", Mock(side_effect=lambda a, *r: a)),
        patch.object(audio.voice_clone, "correct_pitch", Mock(return_value=(fixed, pitch))),
        patch.object(audio.voice_clone, "clone_similarity", Mock(return_value=0.85)),
    ):
        resp = client.post("/api/tts", json={"text": "One. Two.", "clone": "v", "sentence_pause_s": 0.0})
    meta = resp.json()["meta"]["clone"]
    assert meta["pitch"] == pitch
    assert meta["similarity"] == 0.85
    assert meta["intraspeaker_spread"] == 0.09
    assert meta["as_close_as_their_own_re_recordings"] is True
    # 0.85 is below the speaker's own 0.09-based bar only if the numbers say so;
    # here the clone is *closer* than the speaker's re-records, which is the
    # point of reporting the spread next to the score.
    assert "not the speaker's" in meta["note"]


def test_a_clone_report_never_claims_a_verdict_it_cannot_support(client):
    """No measurement means no verdict - `as_close_as...` is None, not True."""
    with (
        patch.object(audio, "_ensure_model", AsyncMock(return_value=_pipe(seconds=0.3))),
        patch.object(audio.voice_clone, "get_profile", return_value={"id": "v", "name": "V"}),
        patch.object(audio.voice_clone, "source_se", Mock(return_value=object())),
        patch.object(audio.voice_clone, "target_se", Mock(return_value=object())),
        patch.object(audio.voice_clone, "convert", Mock(side_effect=lambda a, *r: a)),
        patch.object(audio.voice_clone, "correct_pitch", Mock(side_effect=lambda a, s, t: (a, {"corrected": False, "reason": "no target"}))),
        patch.object(audio.voice_clone, "clone_similarity", Mock(return_value=None)),
    ):
        resp = client.post("/api/tts", json={"text": "One.", "clone": "v", "sentence_pause_s": 0.0})
    meta = resp.json()["meta"]["clone"]
    assert meta["similarity"] is None
    assert meta["intraspeaker_spread"] is None
    assert meta["as_close_as_their_own_re_recordings"] is None


def test_an_unmeasurable_clone_does_not_fail_the_render(client):
    """A failed pitch correction or similarity check must cost the report, not
    the audio the user asked for."""
    with (
        patch.object(audio, "_ensure_model", AsyncMock(return_value=_pipe(seconds=0.3))),
        patch.object(audio.voice_clone, "get_profile", return_value={"id": "v", "name": "V", "median_f0_hz": 100.0}),
        patch.object(audio.voice_clone, "source_se", Mock(return_value=object())),
        patch.object(audio.voice_clone, "target_se", Mock(return_value=object())),
        patch.object(audio.voice_clone, "convert", Mock(side_effect=lambda a, *r: a)),
        patch.object(audio.voice_clone, "correct_pitch", Mock(side_effect=RuntimeError("vocoder died"))),
        patch.object(audio.voice_clone, "clone_similarity", Mock(side_effect=RuntimeError("encoder died"))),
    ):
        resp = client.post("/api/tts", json={"text": "One.", "clone": "v", "sentence_pause_s": 0.0})
    assert resp.status_code == 200
    assert resp.json()["audio_b64"]


def test_tts_source_embedding_is_synthesised_at_speed_one(client):
    """Prosody belongs to the clone target, so the source pass must not use the
    caller's speed - otherwise a fast read bakes a fast cadence into the timbre."""
    pipe = _pipe()
    with (
        patch.object(audio, "_ensure_model", AsyncMock(return_value=pipe)),
        patch.object(audio.voice_clone, "get_profile", return_value={"id": "v", "name": "V"}),
        patch.object(audio.voice_clone, "source_se", Mock(return_value=object())) as src,
        patch.object(audio.voice_clone, "target_se", Mock(return_value=object())),
        patch.object(audio.voice_clone, "convert", Mock(side_effect=lambda a, *r: a)),
    ):
        client.post("/api/tts", json={"text": "hi", "clone": "v", "speed": 2.0})
    assert src.call_args.args[0] == "af_heart"


# --------------------------------------------------------------------------- #
# /api/tts/normalize                                                           #
# --------------------------------------------------------------------------- #


def test_normalize_preview_returns_the_spoken_text_and_its_sentences(client):
    body = client.post("/api/tts/normalize", json={"text": "John 1:1-18. Then he said go."}).json()
    assert "John chapter 1, verses 1 through 18" in body["text"]
    assert len(body["sentences"]) >= 2


def test_normalize_preview_of_empty_text_is_empty_not_an_error(client):
    body = client.post("/api/tts/normalize", json={"text": ""}).json()
    assert body == {"text": "", "sentences": []}


# --------------------------------------------------------------------------- #
# /api/voices* - consent gate, prompt coverage, error mapping                 #
# --------------------------------------------------------------------------- #


def test_voice_prompts_expose_the_read_aloud_script(client):
    with patch.object(audio.voice_clone, "PROMPTS", [{"id": "1", "title": "t", "text": "x", "min_s": 20}]):
        body = client.get("/api/voices/prompts").json()
    assert body == {"prompts": [{"id": "1", "title": "t", "text": "x", "min_s": 20}], "min_total_speech_s": audio.voice_clone.MIN_TOTAL_SPEECH_S}


def test_creating_a_voice_requires_explicit_consent(client):
    resp = client.post("/api/voices", json={"name": "Me", "consent": False, "recordings": []})
    assert resp.status_code == 400 and "agreed to be cloned" in resp.json()["error"]


def test_consent_must_be_the_boolean_true_not_a_truthy_string(client):
    resp = client.post("/api/voices", json={"name": "Me", "consent": "yes", "recordings": []})
    assert resp.status_code == 400


def test_creating_a_voice_lists_the_prompts_that_are_still_missing(client):
    with patch.object(audio.voice_clone, "PROMPTS", [{"id": "1"}, {"id": "2"}, {"id": "3"}]), patch.object(audio.voice_clone, "create_profile") as create:
        resp = client.post(
            "/api/voices",
            json={"name": "Me", "consent": True, "recordings": [{"prompt_id": "1", "audio_b64": "AAAA"}]},
        )
    assert resp.status_code == 400
    assert resp.json()["error"] == "missing recordings for: 2, 3"
    assert create.call_count == 0


def test_malformed_recordings_are_a_400_naming_the_shape(client):
    with patch.object(audio.voice_clone, "PROMPTS", []):
        resp = client.post(
            "/api/voices",
            json={"name": "Me", "consent": True, "recordings": [{"prompt_id": "1"}]},
        )
    assert resp.status_code == 400 and "recordings must be" in resp.json()["error"]


def test_voice_creation_passes_decoded_audio_through_to_the_cloner(client):
    import base64 as b64

    prompts = [{"id": "1"}]
    with patch.object(audio.voice_clone, "PROMPTS", prompts), patch.object(audio.voice_clone, "create_profile", return_value={"id": "x", "name": "Me"}) as create:
        resp = client.post(
            "/api/voices",
            json={"name": "Me", "consent": True, "recordings": [{"prompt_id": "1", "audio_b64": b64.b64encode(b"RIFFfake").decode()}]},
        )
    assert resp.status_code == 200 and resp.json()["voice"]["id"] == "x"
    assert create.call_args.args == ("Me", [("1", b"RIFFfake")])


def test_voice_creation_maps_a_quality_rejection_to_422(client):
    prompts = [{"id": "1"}]
    with patch.object(audio.voice_clone, "PROMPTS", prompts), patch.object(audio.voice_clone, "create_profile", side_effect=ValueError("total speech 4.0s is under 30s")):
        resp = client.post("/api/voices", json={"name": "Me", "consent": True, "recordings": [{"prompt_id": "1", "audio_b64": "AAAA"}]})
    assert resp.status_code == 422 and "under 30s" in resp.json()["error"]


def test_voice_creation_maps_an_unexpected_failure_to_500(client):
    prompts = [{"id": "1"}]
    with patch.object(audio.voice_clone, "PROMPTS", prompts), patch.object(audio.voice_clone, "create_profile", side_effect=RuntimeError("cuda oom")):
        resp = client.post("/api/voices", json={"name": "Me", "consent": True, "recordings": [{"prompt_id": "1", "audio_b64": "AAAA"}]})
    assert resp.status_code == 500 and "custom voice creation failed" in resp.json()["error"]


def test_listing_and_renaming_and_deleting_a_voice(client):
    with patch.object(audio.voice_clone, "list_profiles", return_value=[{"id": "v"}]):
        assert client.get("/api/voices").json() == {"voices": [{"id": "v"}]}

    with patch.object(audio.voice_clone, "rename_profile", return_value={"id": "v", "name": "New"}) as rename:
        assert client.patch("/api/voices/v", json={"name": "New"}).json()["voice"]["name"] == "New"
    assert rename.call_args.args == ("v", "New")

    with patch.object(audio.voice_clone, "delete_profile", Mock(return_value=True)) as delete:
        assert client.delete("/api/voices/v").json() == {"deleted": "v"}
    assert delete.call_args.args == ("v",)


def test_renaming_to_a_taken_name_is_409_and_to_an_unknown_voice_is_404(client):
    with patch.object(audio.voice_clone, "rename_profile", side_effect=ValueError("name already used")):
        assert client.patch("/api/voices/v", json={"name": "Taken"}).status_code == 409
    with patch.object(audio.voice_clone, "rename_profile", side_effect=KeyError("v")):
        resp = client.patch("/api/voices/v", json={"name": "New"})
    assert resp.status_code == 404 and "unknown custom voice" in resp.json()["error"]


def test_deleting_an_unknown_voice_is_404(client):
    with patch.object(audio.voice_clone, "delete_profile", side_effect=KeyError("ghost")):
        resp = client.delete("/api/voices/ghost")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# /api/music                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("payload", "needle"),
    [
        ({"prompt": ""}, "prompt is required"),
        ({"tags": "   "}, "prompt is required"),
        ({"prompt": "x", "duration_s": 1.5}, "duration_s must be within"),
        ({"prompt": "x", "duration_s": 31}, "duration_s must be within"),
    ],
)
def test_music_rejects_bad_requests_before_loading_a_model(client, payload, needle):
    with patch.object(audio, "_ensure_model", AsyncMock()) as ensure:
        resp = client.post("/api/music", json=payload)
    assert resp.status_code == 400 and needle in resp.json()["error"]
    assert ensure.call_count == 0


def test_music_accepts_tags_as_an_alias_for_prompt(client):
    bundle, _ = _music_bundle()
    with patch.object(audio, "_ensure_model", AsyncMock(return_value=bundle)), patch.dict(sys.modules, {"torch": _fake_torch()}):
        body = client.post("/api/music", json={"tags": "gentle pad"}).json()
    assert body["meta"]["prompt"] == "gentle pad"


def test_music_returns_503_when_the_model_cannot_load(client):
    with patch.object(audio, "_ensure_model", AsyncMock(side_effect=RuntimeError("weights missing"))):
        resp = client.post("/api/music", json={"prompt": "pad"})
    assert resp.status_code == 503 and "failed to load" in resp.json()["error"]


def test_music_reports_the_real_sample_rate_and_duration(client):
    bundle, captured = _music_bundle(samples=32000, sampling_rate=32000)
    with patch.object(audio, "_ensure_model", AsyncMock(return_value=bundle)), patch.dict(sys.modules, {"torch": _fake_torch()}):
        body = client.post("/api/music", json={"prompt": "pad", "duration_s": 10}).json()
    assert body["meta"]["sample_rate"] == 32000
    assert body["meta"]["duration_s"] == 1.0  # 32000 samples at 32 kHz
    assert captured["max_new_tokens"] == 500  # 10 s * 50 frames/s


def test_music_frame_budget_is_clamped_at_1500_whatever_the_requested_duration(client):
    """musicgen-small was trained on 30 s windows, so its positional table tops out
    at 1500 frames. Raising AUDIO_MAX_MUSIC_S cannot buy longer audio: min() wins,
    and the only honest signal is that requested_duration_s != duration_s."""
    bundle, captured = _music_bundle(samples=1500, sampling_rate=32000)
    with (
        patch.object(audio, "MAX_MUSIC_SECONDS", 300.0),
        patch.object(audio, "_ensure_model", AsyncMock(return_value=bundle)),
        patch.dict(sys.modules, {"torch": _fake_torch()}),
    ):
        resp = client.post("/api/music", json={"prompt": "pad", "duration_s": 300})
    body = resp.json()
    assert resp.status_code == 200  # the 300 s request is accepted...
    assert captured["max_new_tokens"] == 1500  # ...but only 1500 frames are generated
    assert body["meta"]["requested_duration_s"] == 300.0
    assert body["meta"]["tokens"] == 1500
    # A caller can therefore always tell the clamp happened instead of silently
    # receiving 30 s of audio for a 300 s request.
    assert body["meta"]["duration_s"] != body["meta"]["requested_duration_s"]


def test_music_seeds_the_generator_only_when_a_seed_is_given(client):
    bundle, _ = _music_bundle()
    torch_mod = _fake_torch([])
    with patch.object(audio, "_ensure_model", AsyncMock(return_value=bundle)), patch.dict(sys.modules, {"torch": torch_mod}):
        client.post("/api/music", json={"prompt": "pad", "duration_s": 2})
        assert torch_mod.manual_seed.call_count == 0
        client.post("/api/music", json={"prompt": "pad", "duration_s": 2, "seed": 7})
        torch_mod.manual_seed.assert_called_once_with(7)


def test_music_passes_generation_knobs_through(client):
    bundle, captured = _music_bundle()
    with patch.object(audio, "_ensure_model", AsyncMock(return_value=bundle)), patch.dict(sys.modules, {"torch": _fake_torch()}):
        client.post("/api/music", json={"prompt": "pad", "duration_s": 3, "temperature": 0.7, "guidance_scale": 2.0, "top_k": 100})
    assert captured["temperature"] == 0.7
    assert captured["guidance_scale"] == 2.0
    assert captured["top_k"] == 100 and captured["do_sample"] is True


# --------------------------------------------------------------------------- #
# /api/unload                                                                  #
# --------------------------------------------------------------------------- #


def test_unload_only_reports_what_was_actually_loaded(client):
    audio._state["music"] = object()
    with patch.object(audio, "_ensure_model", AsyncMock()):
        body = client.post("/api/unload").json()
    assert body == {"unloaded": ["music"], "freed": True}


def test_unload_with_nothing_loaded_is_still_200(client):
    assert client.post("/api/unload").json() == {"unloaded": [], "freed": True}


def test_unload_drops_both_engines_when_both_are_resident(client):
    audio._state["tts"], audio._state["music"] = object(), object()
    with patch.object(audio.voice_clone, "unload", Mock()):
        assert client.post("/api/unload").json()["unloaded"] == ["tts", "music"]


# --------------------------------------------------------------------------- #
# Module-level configuration contracts the rest of the stack depends on       #
# --------------------------------------------------------------------------- #


def test_every_kokoro_voice_follows_the_kokoro_naming_scheme():
    assert all(v.startswith(("af_", "am_", "bf_", "bm_")) for v in audio.KOKORO_VOICES)
    assert len(set(audio.KOKORO_VOICES)) == len(audio.KOKORO_VOICES)


def test_tts_default_voice_exists_so_a_bare_request_cannot_404():
    assert "af_heart" in audio.KOKORO_VOICES


def test_capitals_come_from_the_environment_not_from_the_gitignore_default():
    assert audio.MAX_TTS_CHARS == 4000
    assert audio.MAX_MUSIC_SECONDS == 30.0
    assert audio.IDLE_UNLOAD_S == 180


def test_the_tts_cap_is_checked_before_normalization_which_lengthens_text():
    """Text just under the 4000-char cap normalizes to MORE than the cap, so the
    check must stay where it is (pre-normalize) or a valid request gets a 400."""
    import tts_text

    raw = "1 Cor. 13:4-7. " * 250
    assert len(raw) < audio.MAX_TTS_CHARS
    assert len(tts_text.normalize(raw)) > audio.MAX_TTS_CHARS
