"""The music queue: songs as jobs, one at a time, falling back to the CPU."""
import asyncio
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import audio_server as audio
from fastapi.testclient import TestClient

SR = 1000


class _Arr:
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


def _torch(cuda=True):
    mod = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    mod.no_grad = _NoGrad
    mod.manual_seed = Mock()
    mod.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    return mod


def _bundle(device="cuda", oom=False, calls=None):
    model = Mock()
    model.config.audio_encoder.sampling_rate = SR

    prompt_audio = {"n": 0}

    def generate(**kwargs):
        if calls is not None:
            calls.append((device, kwargs["max_new_tokens"]))
        if oom:
            raise RuntimeError("CUDA out of memory. Tried to allocate 94.00 MiB")
        # Like MusicGen: a continuation's output starts with the prompt audio.
        n = prompt_audio["n"] + kwargs["max_new_tokens"] * SR // 50
        return _Arr(np.full((1, 1, n), 0.1, dtype=np.float32))

    def processor(**kwargs):
        prompt_audio["n"] = len(kwargs["audio"][0]) if "audio" in kwargs else 0
        out = Mock()
        out.to.return_value = {}
        return out

    model.generate.side_effect = generate
    proc = Mock(side_effect=processor)
    return {"model": model, "processor": proc, "device": device}


@pytest.fixture(autouse=True)
def _clean():
    audio._music_jobs.clear()
    audio._music_order.clear()
    audio._state["music_busy"] = False
    yield
    audio._music_jobs.clear()
    audio._music_order.clear()


def _run(coro):
    return asyncio.run(coro)


async def _wait_done(job_id, timeout=5.0):
    for _ in range(int(timeout / 0.01)):
        if audio._music_jobs[job_id]["status"] in ("done", "error"):
            return audio._music_jobs[job_id]
        await asyncio.sleep(0.01)
    raise AssertionError("job did not finish")


def test_a_song_job_runs_through_the_queue_and_reports_its_sections():
    calls = []
    bundle = _bundle(calls=calls)

    async def go():
        with patch.object(audio, "_music_bundle", AsyncMock(return_value=(bundle, "cuda"))), patch.dict(sys.modules, {"torch": _torch()}):
            req = audio._parse_music_request({"prompt": "folk", "duration_s": 60, "structure": "short"}, song=True)
            job = audio._enqueue_music(req)
            return await _wait_done(job["id"])

    job = _run(go())
    assert job["status"] == "done", job.get("error")
    assert job["sections"] == ["intro", "verse", "chorus", "outro"]
    assert job["meta"]["kind"] == "song" and job["meta"]["device"] == "cuda"
    assert abs(job["meta"]["duration_s"] - 60) < 6
    assert len(calls) == 4
    view = audio._music_job_view(job)
    assert view["audio_b64"] and "request" not in view


def test_running_out_of_gpu_memory_mid_song_finishes_on_the_cpu():
    gpu = _bundle(device="cuda", oom=True)
    cpu_calls = []
    cpu = _bundle(device="cpu", calls=cpu_calls)

    async def fake_bundle(force_cpu=False):
        return (cpu, "cpu") if force_cpu else (gpu, "cuda")

    async def go():
        with patch.object(audio, "_music_bundle", side_effect=fake_bundle), patch.dict(sys.modules, {"torch": _torch()}):
            job = audio._enqueue_music(audio._parse_music_request({"prompt": "folk", "duration_s": 30, "structure": "short"}, song=True))
            return await _wait_done(job["id"])

    job = _run(go())
    assert job["status"] == "done", job.get("error")
    assert job["meta"]["fell_back_to_cpu"] is True and job["meta"]["device"] == "cpu+cuda"
    assert len(cpu_calls) == 4


def test_jobs_run_one_at_a_time_in_order():
    order = []

    async def slow_job(job):
        order.append(("start", job["id"]))
        await asyncio.sleep(0.05)
        order.append(("end", job["id"]))
        job["audio_b64"], job["meta"] = "AA", {}

    async def go():
        with patch.object(audio, "_run_music_job", side_effect=slow_job):
            a = audio._enqueue_music({"kind": "clip"})
            b = audio._enqueue_music({"kind": "clip"})
            assert audio._music_job_view(b)["position"] == 2
            await _wait_done(b["id"])
            return a["id"], b["id"]

    a, b = _run(go())
    assert order == [("start", a), ("end", a), ("start", b), ("end", b)]


def test_tts_does_not_evict_the_music_model_while_a_song_is_written():
    audio._state["music"] = object()
    audio._state["music_busy"] = True
    try:
        assert _run(audio._unload_model("music")) is False
        assert audio._state["music"] is not None
    finally:
        audio._state["music"] = None
        audio._state["music_busy"] = False


def test_no_room_on_the_gpu_means_the_cpu_model():
    cpu = _bundle(device="cpu")
    with (
        patch.dict(sys.modules, {"torch": _torch(cuda=True)}),
        patch.object(audio, "DEVICE", "cuda"),
        patch.object(audio, "_free_vram_mb", return_value=(87, 8188)),
        patch.object(audio, "_ensure_model", AsyncMock()) as gpu_load,
        patch.object(audio, "_ensure_music_cpu", AsyncMock(return_value=cpu)),
    ):
        audio._state["music"] = None
        bundle, device = _run(audio._music_bundle())
    assert device == "cpu" and bundle is cpu and gpu_load.call_count == 0


def test_song_requests_are_validated_before_queueing():
    client = TestClient(audio.app)
    assert client.post("/api/music/jobs", json={"prompt": ""}).status_code == 400
    too_long = client.post("/api/music/jobs", json={"prompt": "x", "duration_s": audio.MAX_SONG_SECONDS + 1})
    assert too_long.status_code == 400 and "duration_s" in too_long.json()["error"]
    bad = client.post("/api/music/jobs", json={"prompt": "x", "structure": "polka-rap"})
    assert bad.status_code == 400 and "unknown structure" in bad.json()["error"]
    assert audio._music_order == []


def test_an_unknown_job_is_404_and_a_queued_one_can_be_cancelled():
    client = TestClient(audio.app)
    assert client.get("/api/music/jobs/nope").status_code == 404
    audio._music_jobs["j1"] = {"id": "j1", "status": "queued", "request": {}, "progress": {}}
    audio._music_order.append("j1")
    assert client.delete("/api/music/jobs/j1").json()["status"] == "cancelled"
    assert audio._music_order == []


def test_a_full_queue_is_a_clear_503():
    client = TestClient(audio.app)
    audio._music_order.extend(f"x{i}" for i in range(audio.MAX_QUEUED_MUSIC_JOBS))
    resp = client.post("/api/music/jobs", json={"prompt": "x", "duration_s": 30})
    assert resp.status_code == 503 and "already waiting" in resp.json()["error"]
