import importlib.util
import json
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
