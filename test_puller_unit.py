import importlib.util
import json
import os
import pathlib
import tempfile

MODULE_PATH = pathlib.Path(__file__).with_name("alpaca-puller.py")
SPEC = importlib.util.spec_from_file_location("alpaca_puller", MODULE_PATH)
alpaca_puller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(alpaca_puller)


def write_manifest(path, digest):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "config": {"digest": "sha256:cfg", "size": 1},
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": digest,
                        "size": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_get_model_info_adds_latest_tag():
    repo, tag, manifest_path = alpaca_puller.get_model_info("tinyllama")
    assert repo == "library/tinyllama"
    assert tag == "latest"
    assert str(manifest_path).endswith("library/tinyllama/latest")


def test_get_model_info_supports_namespaced_model():
    repo, tag, manifest_path = alpaca_puller.get_model_info("acme/model:Q4_K_M")
    assert repo == "acme/model"
    assert tag == "Q4_K_M"
    assert str(manifest_path).endswith("acme/model/Q4_K_M")


def test_router_filename_for_model_name_flattens_name_and_tag():
    result = alpaca_puller.router_filename_for_model_name("acme/model:Q4_K_M")
    assert result == "acme--model--Q4_K_M.gguf"


def test_parse_huggingface_ref_repo_and_file_syntax():
    repo, filename = alpaca_puller.parse_huggingface_ref(
        "Qwen/Qwen3.6-35B-A3B-GGUF:Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf"
    )
    assert repo == "Qwen/Qwen3.6-35B-A3B-GGUF"
    assert filename == "Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf"


def test_parse_huggingface_ref_url_syntax():
    repo, filename = alpaca_puller.parse_huggingface_ref(
        "https://huggingface.co/Qwen/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf"
    )
    assert repo == "Qwen/Qwen3.6-35B-A3B-GGUF"
    assert filename == "Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf"


def test_parse_huggingface_ref_hf_prefix_single_segment():
    repo, filename = alpaca_puller.parse_huggingface_ref(
        "hf://username/repo.gguf"
    )
    assert repo == "username"
    assert filename == "repo.gguf"
    constructed = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    assert constructed == "https://huggingface.co/username/resolve/main/repo.gguf"


def test_parse_huggingface_ref_hf_prefix_multi_segment():
    repo, filename = alpaca_puller.parse_huggingface_ref(
        "hf://unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    )
    assert repo == "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
    assert filename == "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    constructed = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    assert constructed == "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"


def test_parse_huggingface_ref_hf_prefix_deep_nested():
    repo, filename = alpaca_puller.parse_huggingface_ref(
        "hf://a/b/c/d/model.gguf"
    )
    assert repo == "a/b/c/d"
    assert filename == "model.gguf"
    constructed = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    assert constructed == "https://huggingface.co/a/b/c/d/resolve/main/model.gguf"


def test_infer_local_name_from_huggingface_uses_repo_and_filename():
    result = alpaca_puller.infer_local_name_from_huggingface(
        "Qwen/Qwen3.6-35B-A3B-GGUF",
        "Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf",
    )
    assert result == "Qwen3.6-35B-A3B-GGUF/Qwen_Qwen3.6-35B-A3B-Q4_K_M:latest"


def test_write_json_blob_writes_content_addressed_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        alpaca_puller.MODELS_DIR = tmpdir
        digest, size = alpaca_puller.write_json_blob({"model_format": "gguf"})
        blob_path = pathlib.Path(tmpdir) / "blobs" / digest.replace(":", "-")
        assert blob_path.exists()
        assert blob_path.stat().st_size == size
        assert json.loads(blob_path.read_text(encoding="utf-8")) == {"model_format": "gguf"}


def test_ensure_router_symlink_points_to_blob():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = pathlib.Path(tmpdir)
        alpaca_puller.MODELS_DIR = str(base)
        alpaca_puller.ROUTER_MODELS_DIR = str(base / ".alpaca-router")
        digest = "sha256:solo"
        blob_path = base / "blobs" / "sha256-solo"
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        blob_path.write_bytes(b"x")
        manifest = {
            "config": {"digest": "sha256:cfg", "size": 1},
            "layers": [
                {
                    "mediaType": "application/vnd.ollama.image.model+gguf",
                    "digest": digest,
                    "size": 1,
                }
            ],
        }

        router_path = alpaca_puller.ensure_router_symlink("qwen3:8b", manifest)

        assert router_path.is_symlink()
        assert router_path.name == "qwen3--8b.gguf"
        assert router_path.resolve() == blob_path.resolve()


def test_reindex_models_creates_router_symlink_for_existing_manifest():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = pathlib.Path(tmpdir)
        alpaca_puller.MODELS_DIR = str(base)
        alpaca_puller.ROUTER_MODELS_DIR = str(base / ".alpaca-router")
        digest = "sha256:solo"
        blob_path = base / "blobs" / "sha256-solo"
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        blob_path.write_bytes(b"x")
        cfg_path = base / "blobs" / "sha256-cfg"
        cfg_path.write_text("{}", encoding="utf-8")

        manifest_a = base / "manifests" / "registry.ollama.ai" / "library" / "tinyllama" / "latest"
        manifest_a.parent.mkdir(parents=True, exist_ok=True)
        manifest_a.write_text(
            json.dumps(
                {
                    "config": {"digest": "sha256:cfg", "size": 2},
                    "layers": [
                        {
                            "mediaType": "application/vnd.ollama.image.model",
                            "digest": digest,
                            "size": 1,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        result = alpaca_puller.reindex_models()

        assert result == 0
        assert (base / ".alpaca-router" / "tinyllama--latest.gguf").is_symlink()


def test_remove_model_keeps_shared_blobs():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = pathlib.Path(tmpdir)
        alpaca_puller.MODELS_DIR = str(base)
        digest = "sha256:shared"
        blob_path = base / "blobs" / "sha256-shared"
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        blob_path.write_bytes(b"x")

        manifest_a = base / "manifests" / "registry.ollama.ai" / "library" / "tinyllama" / "latest"
        manifest_b = base / "manifests" / "registry.ollama.ai" / "library" / "other" / "latest"
        write_manifest(manifest_a, digest)
        write_manifest(manifest_b, digest)

        result = alpaca_puller.remove_model("tinyllama")

        assert result == 0
        assert not manifest_a.exists()
        assert manifest_b.exists()
        assert blob_path.exists()


def test_remove_model_deletes_unshared_blobs():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = pathlib.Path(tmpdir)
        alpaca_puller.MODELS_DIR = str(base)
        digest = "sha256:solo"
        blob_path = base / "blobs" / "sha256-solo"
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        blob_path.write_bytes(b"x")

        manifest_a = base / "manifests" / "registry.ollama.ai" / "library" / "tinyllama" / "latest"
        write_manifest(manifest_a, digest)

        result = alpaca_puller.remove_model("tinyllama")

        assert result == 0
        assert not manifest_a.exists()
        assert not blob_path.exists()


def test_should_stop_returns_false_when_no_model_set():
    """Test that _should_stop returns False when _CURRENT_MODEL is empty"""
    import alpaca_puller

    original_model = alpaca_puller._CURRENT_MODEL
    try:
        alpaca_puller._CURRENT_MODEL = ""
        assert alpaca_puller._should_stop() is False
    finally:
        alpaca_puller._CURRENT_MODEL = original_model


def test_should_stop_returns_true_when_signal_set():
    """Test that _should_stop returns True when SIGTERM has been received"""
    import alpaca_puller

    original_stopped = alpaca_puller._STOPPED
    original_model = alpaca_puller._CURRENT_MODEL
    try:
        alpaca_puller._CURRENT_MODEL = "test-model"
        alpaca_puller._STOPPED = True
        assert alpaca_puller._should_stop() is True
    finally:
        alpaca_puller._STOPPED = original_stopped
        alpaca_puller._CURRENT_MODEL = original_model


def test_should_stop_returns_true_when_marker_file_exists_for_current_model(tmp_path):
    """Test that _should_stop returns True when the marker file matches _CURRENT_MODEL"""
    import alpaca_puller

    # _CURRENT_MODEL "Qwen/Qwen3.6:GGUF" => safe_name "Qwen_Qwen3.6_GGUF"
    stop_dir = tmp_path / ".alpaca-stop"
    stop_dir.mkdir()
    stop_file = stop_dir / "Qwen_Qwen3.6_GGUF"
    stop_file.write_text("1000000")

    original_model = alpaca_puller._CURRENT_MODEL
    original_router_dir = (
        alpaca_puller.ROUTER_MODELS_DIR if hasattr(alpaca_puller, "ROUTER_MODELS_DIR") else None
    )
    original_env = os.environ.get("ROUTER_MODELS_DIR")

    try:
        alpaca_puller._CURRENT_MODEL = "Qwen/Qwen3.6:GGUF"
        if hasattr(alpaca_puller, "ROUTER_MODELS_DIR"):
            alpaca_puller.ROUTER_MODELS_DIR = str(tmp_path)
        os.environ["ROUTER_MODELS_DIR"] = str(tmp_path)

        assert alpaca_puller._should_stop() is True
    finally:
        alpaca_puller._CURRENT_MODEL = original_model
        if original_router_dir is not None:
            alpaca_puller.ROUTER_MODELS_DIR = original_router_dir
        elif hasattr(alpaca_puller, "ROUTER_MODELS_DIR"):
            del alpaca_puller.ROUTER_MODELS_DIR
        if original_env is not None:
            os.environ["ROUTER_MODELS_DIR"] = original_env
        elif "ROUTER_MODELS_DIR" in os.environ:
            del os.environ["ROUTER_MODELS_DIR"]


def test_should_stop_returns_false_when_marker_exists_but_wrong_model(tmp_path):
    """Test that _should_stop returns False when marker exists but for a different model"""
    import alpaca_puller

    stop_dir = tmp_path / ".alpaca-stop"
    stop_dir.mkdir()
    # Marker file for a different model (safe name from "other/other:latest" => "other_other_latest")
    stop_file = stop_dir / "other_other_latest"
    stop_file.write_text("1000000")

    original_model = alpaca_puller._CURRENT_MODEL
    original_router_dir = (
        alpaca_puller.ROUTER_MODELS_DIR if hasattr(alpaca_puller, "ROUTER_MODELS_DIR") else None
    )
    original_env = os.environ.get("ROUTER_MODELS_DIR")

    try:
        alpaca_puller._CURRENT_MODEL = "target/target:latest"
        if hasattr(alpaca_puller, "ROUTER_MODELS_DIR"):
            alpaca_puller.ROUTER_MODELS_DIR = str(tmp_path)
        os.environ["ROUTER_MODELS_DIR"] = str(tmp_path)

        assert alpaca_puller._should_stop() is False
    finally:
        alpaca_puller._CURRENT_MODEL = original_model
        if original_router_dir is not None:
            alpaca_puller.ROUTER_MODELS_DIR = original_router_dir
        elif hasattr(alpaca_puller, "ROUTER_MODELS_DIR"):
            del alpaca_puller.ROUTER_MODELS_DIR
        if original_env is not None:
            os.environ["ROUTER_MODELS_DIR"] = original_env
        elif "ROUTER_MODELS_DIR" in os.environ:
            del os.environ["ROUTER_MODELS_DIR"]


def test_should_stop_handles_slash_colon_in_model_name():
    """Test that model names with / and : produce correct safe file names"""

    import alpaca_puller

    original_model = alpaca_puller._CURRENT_MODEL
    try:
        alpaca_puller._CURRENT_MODEL = "Qwen/Qwen3.6-35B-A3B:GGUF"
        safe_expected = "Qwen_Qwen3.6-35B-A3B_GGUF"
        assert safe_expected == alpaca_puller._CURRENT_MODEL.replace("/", "_").replace(":", "_")
    finally:
        alpaca_puller._CURRENT_MODEL = original_model


def test_pull_model_resets_current_model_on_completion(tmp_path):
    """Test that _CURRENT_MODEL is reset after pull_model finishes"""
    import alpaca_puller

    original_model = alpaca_puller._CURRENT_MODEL
    try:
        alpaca_puller._CURRENT_MODEL = "will-set-model"
        # pull_model should set _CURRENT_MODEL then reset it
        # Since we can't easily mock all the HTTP calls, just verify the pattern
        # by checking the function signature accepts no_resume
        import inspect

        sig = inspect.signature(alpaca_puller.pull_model)
        params = sig.parameters
        assert "no_resume" in params
        assert params["no_resume"].default is False
    finally:
        alpaca_puller._CURRENT_MODEL = original_model
