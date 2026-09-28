"""Unit tests for `tests/test-alpaca.py`, the single-model CLI verifier.

The file is named `test-alpaca.py` with a hyphen, so pytest's default
`python_files` patterns never match it and it is not part of the collected
suite. That is correct - it is an argparse CLI that shells out to `ollama` and
hits a running proxy - but it also means none of its logic had any regression
protection. The name-resolution and disk checks are pure, so they are covered
here by loading the file by path.

The two checks that need a live stack (`ollama list/show`, the proxy `/api/tags`
call) are covered with a fake `ollama` script on PATH and a stubbed
`requests.get`, so the matching rules are verified without either.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

VERIFIER_PATH = Path(__file__).resolve().parent / "test-alpaca.py"


def load_verifier():
    spec = importlib.util.spec_from_file_location("alpaca_cli_verifier", VERIFIER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


@pytest.fixture(scope="module")
def verifier():
    return load_verifier()


# --- name normalisation ----------------------------------------------------


def test_normalize_adds_the_latest_tag(verifier):
    assert verifier.normalize_model_name("qwen3") == "qwen3:latest"
    assert verifier.normalize_model_name("qwen3:8b") == "qwen3:8b"
    assert verifier.normalize_model_name("acme/model") == "acme/model:latest"


def test_public_name_strips_only_the_latest_tag(verifier):
    assert verifier.public_model_name("qwen3:8b") == "qwen3:8b"
    assert verifier.public_model_name("qwen3:latest") == "qwen3"
    assert verifier.public_model_name("qwen3") == "qwen3"


def test_validate_rejects_a_router_filename(verifier):
    # The router filename is the single most likely thing to paste in here, and
    # it otherwise produces a confusing "manifest missing" instead of a clear
    # error naming the right form to use.
    with pytest.raises(ValueError, match="not the internal router filename"):
        verifier.validate_user_model_name("qwen3-8b--q4_k_m.gguf")
    verifier.validate_user_model_name("qwen3:8b")


def test_manifest_path_inserts_the_library_namespace(verifier, tmp_path, monkeypatch):
    monkeypatch.setattr(verifier, "MODELS_DIR", str(tmp_path))
    assert verifier.manifest_path_for_model("tinyllama") == str(
        tmp_path / "manifests" / "registry.ollama.ai" / "library" / "tinyllama" / "latest"
    )
    assert verifier.manifest_path_for_model("acme/model:Q4") == str(
        tmp_path / "manifests" / "registry.ollama.ai" / "acme" / "model" / "Q4"
    )


# --- the disk check --------------------------------------------------------


def _manifest_path_in(models_dir: Path, model: str) -> Path:
    """Where the ollama store would put this model's manifest."""
    repo, tag = (model if ":" in model else f"{model}:latest").rsplit(":", 1)
    parts = repo.split("/")
    if len(parts) == 1:
        parts = ["library", parts[0]]
    return models_dir / "manifests" / "registry.ollama.ai" / Path(*parts) / tag


def _write_manifest(models_dir: Path, model: str, digests: list[str], *, config: str | None = "sha256:cfg") -> None:
    path = _manifest_path_in(models_dir, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"layers": [{"digest": d, "size": 1} for d in digests]}
    if config:
        manifest["config"] = {"digest": config, "size": 1}
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_check_files_exist_passes_when_every_blob_is_present(verifier, tmp_path, monkeypatch):
    monkeypatch.setattr(verifier, "MODELS_DIR", str(tmp_path))
    _write_manifest(tmp_path, "tinyllama", ["sha256:aaa", "sha256:bbb"])
    blobs = tmp_path / "blobs"
    blobs.mkdir(parents=True)
    for digest in ("sha256-aaa", "sha256-bbb", "sha256-cfg"):
        (blobs / digest).write_bytes(b"x")

    ok, detail = verifier.check_files_exist("tinyllama")
    assert ok, detail


def test_check_files_exist_names_the_missing_blob(verifier, tmp_path, monkeypatch):
    monkeypatch.setattr(verifier, "MODELS_DIR", str(tmp_path))
    _write_manifest(tmp_path, "tinyllama", ["sha256:aaa", "sha256:bbb"])
    blobs = tmp_path / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "sha256-aaa").write_bytes(b"x")  # bbb and cfg deliberately absent

    ok, detail = verifier.check_files_exist("tinyllama")
    assert not ok
    assert "blob missing" in detail


def test_check_files_exist_reports_a_missing_manifest_first(verifier, tmp_path, monkeypatch):
    monkeypatch.setattr(verifier, "MODELS_DIR", str(tmp_path))
    ok, detail = verifier.check_files_exist("nope")
    assert not ok
    assert "manifest missing" in detail


def test_check_files_exist_tolerates_a_manifest_with_no_digests(verifier, tmp_path, monkeypatch):
    # A layerless manifest is not corrupt, it just has nothing to verify; it
    # must not raise out of the verifier.
    monkeypatch.setattr(verifier, "MODELS_DIR", str(tmp_path))
    path = _manifest_path_in(tmp_path, "tinyllama")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({}), encoding="utf-8")

    ok, detail = verifier.check_files_exist("tinyllama")
    assert ok, detail


def test_check_files_exist_handles_a_namespaced_model(verifier, tmp_path, monkeypatch):
    monkeypatch.setattr(verifier, "MODELS_DIR", str(tmp_path))
    _write_manifest(tmp_path, "acme/model:Q4", ["sha256:aaa"], config=None)
    (tmp_path / "blobs").mkdir(parents=True)
    (tmp_path / "blobs" / "sha256-aaa").write_bytes(b"x")

    assert verifier.check_files_exist("acme/model:Q4")[0]


# --- the ollama + proxy checks (subprocess / requests stubbed) -------------


def _fake_ollama(tmp_path: Path, body: str) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "ollama"
    script.write_text(f"#!/bin/sh\ncat <<'EOF'\n{body}\nEOF\n", encoding="utf-8")
    script.chmod(0o755)


def test_check_ollama_list_matches_both_spellings(verifier, tmp_path, monkeypatch):
    _fake_ollama(tmp_path, "NAME  ID  SIZE\nqwen3:latest  abc  1 GB\nother:q4  def  2 GB")
    monkeypatch.setenv("PATH", str(tmp_path / "bin"), prepend=os.pathsep)

    for spelling in ("qwen3:latest", "qwen3"):
        ok, detail = verifier.check_ollama_list(spelling)
        assert ok, f"{spelling}: {detail}"
    # A bare repo means :latest, never "whichever tag happens to exist".
    assert not verifier.check_ollama_list("qwen3:8b")[0]
    assert not verifier.check_ollama_list("absent")[0]


def test_check_ollama_list_matches_an_explicit_tag(verifier, tmp_path, monkeypatch):
    _fake_ollama(tmp_path, "NAME  ID  SIZE\nother:q4  def  2 GB")
    monkeypatch.setenv("PATH", str(tmp_path / "bin"), prepend=os.pathsep)
    ok, detail = verifier.check_ollama_list("other:q4")
    assert ok, detail
    assert not verifier.check_ollama_list("other")[0]


def test_check_ollama_list_finds_nothing_in_an_empty_store(verifier, tmp_path, monkeypatch):
    # `ollama list` on a store with no models prints the header and nothing
    # else. The verifier skips line 0, so the header can never be mistaken for
    # a data row.
    _fake_ollama(tmp_path, "NAME  ID  SIZE")
    monkeypatch.setenv("PATH", str(tmp_path / "bin"), prepend=os.pathsep)
    ok, _ = verifier.check_ollama_list("qwen3:8b")
    assert not ok


def test_check_ollama_list_reports_a_failing_command(verifier, tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "ollama"
    script.write_text("#!/bin/sh\necho 'ollama not running' >&2\nexit 1\n", encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir), prepend=os.pathsep)

    ok, detail = verifier.check_ollama_list("qwen3:8b")
    assert not ok
    assert "ollama not running" in detail


def test_check_proxy_tags_matches_the_proxied_name(verifier, monkeypatch):
    class _Resp:
        @staticmethod
        def json():
            return {"models": [{"name": "qwen3:8b"}, {"name": "acme/model:q4"}]}

    monkeypatch.setattr(verifier.requests, "get", lambda url, timeout=None: _Resp())
    for spelling in ("qwen3:8b", "acme/model:q4"):
        assert verifier.check_proxy_tags(spelling)[0], spelling
    # A bare repo means :latest, never "whichever tag happens to exist", so it
    # does not match a store that only holds q4.
    assert not verifier.check_proxy_tags("acme/model")[0]
    assert not verifier.check_proxy_tags("nope:1b")[0]


def test_check_proxy_tags_survives_an_unreachable_proxy(verifier, monkeypatch):
    def _boom(url, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(verifier.requests, "get", _boom)
    ok, detail = verifier.check_proxy_tags("qwen3:8b")
    assert not ok
    assert "proxy request failed" in detail


# --- the aggregate ---------------------------------------------------------


@pytest.mark.parametrize("failing", ["check_files_exist", "check_ollama_list", "check_ollama_show", "check_proxy_tags"])
def test_verify_model_counts_every_failing_check(verifier, monkeypatch, capsys, failing):
    for name in ("check_files_exist", "check_ollama_list", "check_ollama_show", "check_proxy_tags"):
        monkeypatch.setattr(verifier, name, (lambda _m: (False, "boom")) if name == failing else (lambda _m: (True, "ok")))

    assert verifier.verify_model("qwen3:8b") == 1
    out = capsys.readouterr().out
    assert "[FAIL]" in out
    assert "1 failing check" in out


def test_verify_model_returns_zero_when_everything_passes(verifier, monkeypatch, capsys):
    for name in ("check_files_exist", "check_ollama_list", "check_ollama_show", "check_proxy_tags"):
        monkeypatch.setattr(verifier, name, lambda _m: (True, "ok"))
    assert verifier.verify_model("qwen3:8b") == 0
    assert "All checks passed." in capsys.readouterr().out


def test_build_parser_takes_exactly_one_model(verifier):
    assert verifier.build_parser().parse_args(["qwen3:8b"]).model == "qwen3:8b"
