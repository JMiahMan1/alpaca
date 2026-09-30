"""The converter's latent draw is pinned, so a narration is one voice.

OpenVoice's `PosteriorEncoder.forward` builds its latent as

    z = m + torch.randn_like(m) * tau * exp(logs)

and `voice_conversion` is called once per sentence. With torch's global
generator unseeded, every sentence of one narration is therefore re-timbred
from a *different* sample of the same distribution. A listener does not
describe that as "slightly noisy". They describe it as a voice that keeps
changing, breaking up, or as an echo, because consecutive sentences no longer
share a timbre to be heard against.

`convert` forks the generator and seeds it, so every sentence draws the same
latent and the narrator holds one voice for a whole piece. The tests below pin
that, and pin the two things that make it safe to do: the caller's RNG state
is untouched, and `seed=None` still reaches the old unseeded behaviour.

torch only exists inside the audio image, so it is faked here. The fake is
deliberately literal: `randn_like` really does draw from the generator, so a
test cannot pass by the noise never being requested.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_voice_clone():
    """Load voice_clone.py standalone. torch arrives via the `torch_mod` fixture."""
    spec = importlib.util.spec_from_file_location("vc_seed", ROOT / "voice_clone.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["vc_seed"] = module
    spec.loader.exec_module(module)
    return module


class FakeTensor:
    """Just enough tensor for the converter's noise term and the embedding move.

    `.data` is a property returning self, mirroring the torch idiom
    `y = y.data.cpu().float().numpy()` that `convert` ends with, so the fake has
    to be a chain of no-ops ending in a real array.
    """

    def __init__(self, arr, device="cpu"):
        self._arr = np.asarray(arr, dtype=np.float32)
        self.device = device

    def to(self, _device):
        return self

    @property
    def data(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self._arr

    def __getitem__(self, key):
        data = self._arr
        for part in key if isinstance(key, tuple) else (key,):
            data = data[part]
        return FakeTensor(data, self.device)

    def size(self, dim=None):
        return self._arr.shape if dim is None else self._arr.shape[dim]


class FakeTorch(types.ModuleType):
    """torch, with a real linear-congruential generator behind `manual_seed`."""

    class _Generator:
        """A stream, not a function: drawing advances it, as a real one does.

        Without this an unseeded double would hand back the same noise forever,
        and `seed=None` would look deterministic - which is the one thing it
        is not, and the reason it has to be testable.
        """

        def __init__(self):
            self.state = 0

        def seed(self, value):
            self.state = (int(value) & 0x7FFFFFFF) or 1

        def _next(self) -> float:
            # Numerical Recipes' LCG. Reproducible and stdlib-only.
            self.state = (1664525 * self.state + 1013904223) & 0x7FFFFFFF
            return self.state / 0x7FFFFFFF * 2.0 - 1.0

        def randn(self, count):
            return np.array([self._next() for _ in range(count)], dtype=np.float64)

    def __init__(self):
        super().__init__("fake_torch")
        self.gen = self._Generator()
        self.manual_seed_calls: list[int] = []
        self.fork_depth = 0
        self.fork_devices: list = []

    def no_grad(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def manual_seed(self, value):
        self.manual_seed_calls.append(int(value))
        self.gen.seed(value)

    @property
    def random(self):
        return self

    def fork_rng(self, devices=None):
        torch = self

        class _Fork:
            def __enter__(self):
                torch.fork_depth += 1
                torch.fork_devices.append(devices)
                # torch's fork_rng saves and restores the generators it names.
                # Saving here is what makes the leak test below real rather than
                # a test of this fake.
                self.saved = torch.gen.state
                return self

            def __exit__(self, *exc):
                torch.fork_depth -= 1
                torch.gen.state = self.saved
                return False

        return _Fork()

    def randn_like(self, tensor):
        return FakeTensor(self.gen.randn(tensor._arr.size))

    def LongTensor(self, values):
        return FakeTensor(values)

    def zeros_like(self, tensor):
        return FakeTensor(np.zeros_like(tensor._arr))

    def device(self, value):
        # torch.device accepts a name or a device; `convert` hands it conv.device.
        return value if isinstance(value, _FakeDevice) else _FakeDevice(value)

    class cuda:
        @staticmethod
        def current_device():
            return 0


class _FakeDevice:
    def __init__(self, value):
        self.value = value

    @property
    def type(self):
        return str(self.value).split(":")[0]

    @property
    def index(self):
        parts = str(self.value).split(":")
        return int(parts[1]) if len(parts) > 1 and parts[1] else None


class FakeModel:
    """`voice_conversion`, reduced to the one line that is stochastic.

    OpenVoice's real method noises the posterior and then decodes it. All the
    noise reaches the output here, which is the property under test: equal draws
    in, equal audio out, and a different draw in means a different voice.
    """

    def __init__(self, torch):
        self._torch = torch
        self.tau_seen: list[float] = []

    def voice_conversion(self, spec, lengths, sid_src=None, sid_tgt=None, tau=1.0):
        self.tau_seen.append(tau)
        n = spec._arr.size
        noise = self._torch.randn_like(FakeTensor(np.zeros(n)))._arr * tau
        out = spec._arr + noise
        return FakeTensor(out.reshape(1, 1, n)), None, None


class FakeConverter:
    def __init__(self, torch, device="cpu"):
        self.model = FakeModel(torch)
        self.device = _FakeDevice(device)
        self.watermarks: list[str] = []

    def add_watermark(self, audio, message):
        self.watermarks.append(message)
        return audio


@pytest.fixture
def torch_mod():
    module = FakeTorch()
    sys.modules["torch"] = module
    yield module
    sys.modules.pop("torch", None)


@pytest.fixture
def vc(torch_mod, monkeypatch):
    module = _load_voice_clone()
    conv = FakeConverter(torch_mod)
    monkeypatch.setattr(module, "_converter", lambda: conv)
    # Same-rate resample is a no-op in the real code path, and the spectrogram
    # is not what is under test, so both are identity.
    monkeypatch.setattr(module, "_resample", lambda a, *_: np.asarray(a, dtype=np.float32))
    monkeypatch.setattr(
        module,
        "_spec",
        lambda conv_, audio: FakeTensor(np.asarray(audio, dtype=np.float32)),
    )
    module._test_converter = conv  # type: ignore[attr-defined]
    return module


# --------------------------------------------------------------------------- #
# The same sentence, twice
# --------------------------------------------------------------------------- #


def test_the_same_sentence_converted_twice_comes_back_identical(vc, torch_mod):
    """Seeded, `convert` is a function. Unseeded, it is not."""
    audio = np.linspace(0.0, 1.0, 4096, dtype=np.float32)
    se_src, se_tgt = FakeTensor([0.1]), FakeTensor([0.2])

    first = vc.convert(audio, 22050, se_src, se_tgt, 0.3, 4242)
    second = vc.convert(audio, 22050, se_src, se_tgt, 0.3, 4242)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, audio)  # the seed must not disable the noise
    assert torch_mod.manual_seed_calls == [4242, 4242]


def test_different_seeds_give_different_audio(vc):
    """A seed is a choice, not a constant masquerading as one."""
    audio = np.linspace(0.0, 1.0, 4096, dtype=np.float32)
    se_src, se_tgt = FakeTensor([0.1]), FakeTensor([0.2])

    a = vc.convert(audio, 22050, se_src, se_tgt, 0.3, 1)
    b = vc.convert(audio, 22050, se_src, se_tgt, 0.3, 2)

    assert not np.array_equal(a, b)


def test_a_narration_of_several_sentences_is_one_voice(vc):
    """The symptom, stated as a test: no sentence sounds like a different person.

    Seven different sentences is a typical slide. The noise draw is the same
    every time, so the offset the converter adds to each is the same offset, and
    the *relative* timbre between sentences is whatever Kokoro produced. This is
    what the unseeded path loses.
    """
    rng = np.random.default_rng(0)
    se_src, se_tgt = FakeTensor([0.1]), FakeTensor([0.2])
    offsets = []
    for _ in range(7):
        speech = rng.standard_normal(2048).astype(np.float32) * 0.1
        converted = vc.convert(speech, 22050, se_src, se_tgt, 0.3)  # default seed
        offsets.append(converted - speech)

    for other in offsets[1:]:
        assert np.allclose(other, offsets[0], atol=1e-6)


def test_seed_none_restores_the_unseeded_behaviour(vc, torch_mod):
    audio = np.linspace(0.0, 1.0, 4096, dtype=np.float32)
    se_src, se_tgt = FakeTensor([0.1]), FakeTensor([0.2])

    a = vc.convert(audio, 22050, se_src, se_tgt, 0.3, None)
    b = vc.convert(audio, 22050, se_src, se_tgt, 0.3, None)

    assert not np.array_equal(a, b)
    assert torch_mod.manual_seed_calls == []
    # Nothing is forked either. A fork would restore the generator on the way
    # out, and an unseeded draw from a restored generator is the same draw
    # twice: the opt-out would be quieter than it looks.
    assert torch_mod.fork_devices == []


def test_the_default_is_a_seed_not_none(vc, torch_mod):
    audio = np.zeros(2048, dtype=np.float32)
    se_src, se_tgt = FakeTensor([0.1]), FakeTensor([0.2])

    a = vc.convert(audio, 22050, se_src, se_tgt, 0.3)
    b = vc.convert(audio, 22050, se_src, se_tgt, 0.3)

    assert np.array_equal(a, b)
    assert torch_mod.manual_seed_calls == [vc.CONVERT_SEED] * 2


# --------------------------------------------------------------------------- #
# Not leaking
# --------------------------------------------------------------------------- #


def test_the_callers_generator_state_survives(vc, torch_mod):
    """`convert` must not decide what the rest of the process does.

    MusicGen draws from the global generator, and a second request may be
    running on another thread. Seeding globally without forking would make this
    endpoint's internals other people's output.
    """
    torch_mod.gen.seed(999)
    vc.convert(np.zeros(2048, dtype=np.float32), 22050, FakeTensor([0.1]), FakeTensor([0.2]))
    assert torch_mod.gen.state == 999


def test_the_generator_is_forked(vc, torch_mod):
    vc.convert(np.zeros(2048, dtype=np.float32), 22050, FakeTensor([0.1]), FakeTensor([0.2]))
    assert torch_mod.fork_depth == 0  # closed again
    assert len(torch_mod.fork_devices) == 1


def test_a_cpu_device_forks_no_cuda_generators(vc, torch_mod):
    vc._test_converter.device = _FakeDevice("cpu")  # type: ignore[attr-defined]
    vc.convert(np.zeros(2048, dtype=np.float32), 22050, FakeTensor([0.1]), FakeTensor([0.2]))
    assert torch_mod.fork_devices == [[]]


@pytest.mark.parametrize(
    "device,expected",
    [("cuda", [0]), ("cuda:0", [0]), ("cuda:1", [1]), ("cpu", [])],
)
def test_only_the_cuda_generators_belonging_to_the_device_are_forked(
    vc, torch_mod, device, expected
):
    assert vc._seed_devices(torch_mod, _FakeDevice(device)) == expected


def test_tau_still_reaches_the_converter(vc):
    """Seeding pins the draw; it must not quietly change its size."""
    vc.convert(np.zeros(2048, dtype=np.float32), 22050, FakeTensor([0.1]), FakeTensor([0.2]), 0.7)
    assert vc._test_converter.model.tau_seen == [0.7]  # type: ignore[attr-defined]
