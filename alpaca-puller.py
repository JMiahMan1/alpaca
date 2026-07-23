import argparse
import contextlib
import hashlib
import json
import os
import re
import signal
import ssl
import struct
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


OLLAMA_REGISTRY = os.getenv("OLLAMA_REGISTRY", "https://registry.ollama.ai/v2")
OLLAMA_AUTH_URL = os.getenv(
    "OLLAMA_AUTH_URL",
    "https://ollama.com/v2/auth/token?scope=repository:{repo}:pull&service=registry.ollama.ai",
)
MODELS_DIR = os.getenv("MODELS_DIR", "/usr/share/ollama/.ollama/models")
ROUTER_MODELS_DIR = os.getenv(
    "ROUTER_MODELS_DIR",
    str(Path(__file__).resolve().parent / ".alpaca-router"),
)
HUGGING_FACE_BASE = os.getenv("HUGGING_FACE_BASE", "https://huggingface.co")


def _load_dotenv(env_path):
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("\"'")
                os.environ.setdefault(key, value)


def _find_dotenv():
    for candidate in [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
    ]:
        if candidate.exists():
            return candidate
    parent = Path(__file__).resolve().parent
    for _ in range(5):
        parent = parent.parent
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
    return None


dotenv_path = _find_dotenv()
if dotenv_path:
    _load_dotenv(dotenv_path)

HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN") or os.getenv("HF_TOKEN")
_STOPPED = False
_CURRENT_MODEL = ""


def _signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT for graceful download interruption."""
    global _STOPPED
    _STOPPED = True


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


def _should_stop():
    """Check if download should be stopped (signal or marker file)."""
    if _STOPPED:
        return True
    if not _CURRENT_MODEL:
        return False
    # Check for stop marker file: .alpaca-stop/{model_name}
    safe_name = _CURRENT_MODEL.replace("/", "_").replace(":", "_")
    router_dir = os.getenv("ROUTER_MODELS_DIR")
    stop_dirs = [
        Path.cwd() / ".alpaca-stop",
        Path(__file__).resolve().parent / ".alpaca-stop",
    ]
    if router_dir:
        stop_dirs.append(Path(router_dir) / ".alpaca-stop")
    for stop_dir in stop_dirs:
        if stop_dir.exists():
            stop_file = stop_dir / safe_name
            if stop_file.is_file():
                return True
    return False


MODEL_LAYER_MEDIA_TYPE = "application/vnd.ollama.image.model"
MODEL_GGUF_MEDIA_TYPE = "application/vnd.ollama.image.model+gguf"
MANIFEST_MEDIA_TYPE = "application/vnd.docker.distribution.manifest.v2+json"
CONFIG_MEDIA_TYPE = "application/vnd.docker.container.image.v1+json"


def normalize_model_name(model_name):
    return model_name if ":" in model_name else f"{model_name}:latest"


def sanitize_model_component(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    value = value.strip("-.")
    return value or "model"


def _read_gguf_metadata(path: str) -> dict:
    """Parse only the metadata header from a GGUF file (fast, no full load)."""
    meta: dict = {}
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            return meta
        struct.unpack("<I", f.read(4))
        struct.unpack("<Q", f.read(8))
        kv_count = struct.unpack("<Q", f.read(8))[0]

        for _ in range(kv_count):
            key_len = struct.unpack("<Q", f.read(8))[0]  # uint64, not uint32
            key = f.read(key_len).decode("utf-8", errors="replace")
            val_type = struct.unpack("<I", f.read(4))[0]

            # GGUF value types: 0=uint8,1=int8,2=uint16,3=int16,4=uint32,
            # 5=int32,6=float32,7=bool,8=str,9=array,10=uint64,11=int64,12=float64
            if val_type == 8:  # string
                str_len = struct.unpack("<Q", f.read(8))[0]
                val = f.read(str_len).decode("utf-8", errors="replace")
                meta[key] = val
            elif val_type in (0, 1, 4, 5, 10, 11):  # integer types
                fmt = {0: "<B", 1: "<b", 4: "<I", 5: "<i", 10: "<Q", 11: "<q"}
                val = struct.unpack(fmt[val_type], f.read(struct.calcsize(fmt[val_type])))[0]
                meta[key] = val
            elif val_type == 7:  # bool
                meta[key] = struct.unpack("<?", f.read(1))[0]
            elif val_type == 6:  # float32
                meta[key] = struct.unpack("<f", f.read(4))[0]
            elif val_type == 12:  # float64
                meta[key] = struct.unpack("<d", f.read(8))[0]
            elif val_type == 9:  # array
                arr_type = struct.unpack("<I", f.read(4))[0]
                arr_len = struct.unpack("<Q", f.read(8))[0]
                # Skip array contents — we only care about scalar metadata
                if arr_type == 8:  # array of strings
                    for _ in range(arr_len):
                        sl = struct.unpack("<Q", f.read(8))[0]
                        f.read(sl)
                else:
                    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
                    skip = arr_len * sizes.get(arr_type, 0)
                    f.read(skip)
            else:
                # Unknown type — skip conservatively
                break

    return meta


def _model_size_params(meta: dict) -> int:
    """Return total parameter count from GGUF metadata, or 0 if unknown."""
    pc = meta.get("general.parameter_count", 0)
    if isinstance(pc, int) and pc > 0:
        return pc
    label = meta.get("general.size_label", "")
    if isinstance(label, str):
        m = re.match(r"(\d+(?:\.\d+)?)\s*[Bb]", label)
        if m:
            return int(float(m.group(1)) * 1_000_000_000)
    return 0


_FA_UNSUPPORTED_ARCHS = {
    "mamba",
    "rwkv",
    "rwkv6",
    "wavtokenizer",
}


def _supports_flash_attn(meta: dict) -> bool:
    arch = meta.get("general.architecture", "").lower()
    if not arch:
        return False
    return arch not in _FA_UNSUPPORTED_ARCHS


def _is_moe(meta: dict) -> bool:
    for key, val in meta.items():
        if key.endswith(".expert_count") and isinstance(val, int) and val > 0:
            return True
        if key.endswith(".expert_used_count") and isinstance(val, int) and val > 0:
            return True
    arch = meta.get("general.architecture", "")
    return "moe" in arch.lower()


def _image_model_family_for_router_entry(entry):
    """Return the model_family for a router .gguf entry if it is an image model, else None.

    Matches the router filename against local Ollama manifests and inspects the
    config blob's ``model_family``/``families`` — the authoritative signal written by
    the puller for Stable Diffusion imports (e.g. ``stable-diffusion``). This reliably
    catches models like ``qwen-image`` whose GGUF architecture is not literally "sd".
    """
    entry_name = entry.name
    try:
        for model_name, _path, manifest in iter_local_models():
            if not manifest:
                continue
            if router_filename_for_model_name(model_name) != entry_name:
                continue
            config = manifest.get("config", {})
            digest = config.get("digest")
            cfg = {}
            if digest:
                cfg_path = blob_path_for_digest(digest)
                if cfg_path.exists():
                    try:
                        cfg = json.loads(cfg_path.read_text())
                    except Exception:
                        cfg = {}
            family = cfg.get("model_family")
            families = cfg.get("families") or []
            known_image_families = {"stable-diffusion", "qwen-image", "flux", "sdxl", "sd3", "wan"}
            if family in known_image_families or any(f in known_image_families for f in families) or (family and ("image" in family.lower() or "diffusion" in family.lower())):
                return family or "stable-diffusion"
    except Exception as e:
        if isinstance(e, PermissionError):
            print(f"Warning: Manifest access permission denied ({e}). Ensure container/process has read access.", file=sys.stderr)
        return None
    return None


def _detect_sd_family(repo, filename):
    """Best-effort detection of an image/SD model family from a Hugging Face
    repo/file reference. Returns one of 'qwen-image', 'flux', 'stable-diffusion'
    or None for plain LLM GGUFs."""
    name_lower = f"{repo}/{filename}".lower()
    if "qwen-image" in name_lower or "qwen_image" in name_lower:
        return "qwen-image"
    if "flux" in name_lower:
        return "flux"
    if any(
        kw in name_lower
        for kw in [
            "stable-diffusion",
            "sdxl",
            "sd1.",
            "sd2.",
            "sd3",
            "sd-",
            "illustrious",
            "pony",
            "photoreal",
        ]
    ):
        return "stable-diffusion"
    return None


def create_model_profile(router_path, manifest=None, repo=None, filename=None):
    """Create a companion ``.profile.json`` for a freshly pulled model.

    This makes the model appear in the dashboard profiles UI, and (for
    image/SD models) lets the sd-server load it. Image models get a specific
    ``model_family`` plus a default ``gpu_layers``; LLM models get an (initially
    empty) overlay profile that mirrors the behaviour of benchmark-generated
    profiles. No-op if a profile already exists, so re-pulls and manual edits
    are preserved.
    """
    if not router_path or not (router_path.exists() or router_path.is_symlink()):
        return

    # Prefer a specific family from the source reference, then fall back to the
    # manifest's recorded model_family (reliable for Ollama pulls).
    family = None
    if repo and filename:
        family = _detect_sd_family(repo, filename)
    if family is None and manifest is not None:
        cfg = {}
        config = manifest.get("config", {})
        digest = config.get("digest")
        if digest:
            cfg_path = blob_path_for_digest(digest)
            if cfg_path.exists():
                try:
                    cfg = json.loads(cfg_path.read_text())
                except Exception:
                    cfg = {}
        fam = cfg.get("model_family")
        if fam and fam != "gguf":
            family = fam

    if family:
        # Image/SD models: name the profile after the full symlink so the proxy's
        # iter_router_sd_models() (entry.path + ".profile.json") discovers it.
        profile_path = router_path.with_name(router_path.name + ".profile.json")
        content = {"model_family": family}
        if family in ("qwen-image", "flux", "stable-diffusion"):
            content["gpu_layers"] = os.environ.get("SD_GPU_LAYERS", "40")
    else:
        # LLM models: name the profile after the symlink stem (alias.profile.json).
        profile_path = router_path.with_name(router_path.stem + ".profile.json")
        content = {}

    if profile_path.exists():
        return
    try:
        profile_path.write_text(json.dumps(content, indent=2) + "\n")
        with contextlib.suppress(Exception):
            os.chmod(profile_path, 0o666)
        print(f"Created model profile: {profile_path.name}")
    except Exception as e:
        print(f"Warning: could not create model profile {profile_path}: {e}", file=sys.stderr)


def update_models_ini():
    """Scan all GGUF models in the router directory and regenerate models.ini."""
    router_dir = Path(ROUTER_MODELS_DIR)
    ini_path = router_dir / "models.ini"

    # Load runtime exclusions
    mtp_incompatible = set()
    mtp_inc_file = router_dir / ".mtp_incompatible_models.json"
    if mtp_inc_file.exists():
        try:
            with open(mtp_inc_file) as f:
                mtp_incompatible = set(json.load(f))
        except Exception:
            pass

    safe_settings = set()
    safe_file = router_dir / ".safe_settings_models.json"
    if safe_file.exists():
        try:
            with open(safe_file) as f:
                safe_settings = set(json.load(f))
        except Exception:
            pass

    content = [
        "# models.ini - Per-model presets for llama-server router mode",
        "",
        "[*]",
        "mlock = true",
        "no-mmap = true",
        "slot-save-path = /slots-cache",
        "batch-size = 1024",
        "ubatch-size = 1024",
        "parallel = 2",
        "kv-unified = true",
        "n-gpu-layers = 99",
        "",
    ]

    if router_dir.exists():
        for entry in sorted(router_dir.iterdir()):
            if entry.suffix == ".gguf":
                resolved = entry.resolve()
                if not resolved.exists():
                    continue

                alias = entry.stem
                if "projector" in alias.lower() or "mmproj" in alias.lower():
                    continue

                # Skip stable-diffusion / image-generation models in the llama-server
                # models.ini. These must never be registered as llama.cpp LLMs — they
                # are handled exclusively by the sd-server. The authoritative signal is
                # the model's Ollama manifest `model_family` (set to "stable-diffusion"
                # by the puller for SD imports); we fall back to a GGUF arch/keyword
                # heuristic only when no manifest is available.
                is_sd = False
                try:
                    manifest_family = _image_model_family_for_router_entry(entry)
                    if manifest_family:
                        is_sd = True
                    else:
                        meta = _read_gguf_metadata(str(resolved))
                        arch = meta.get("general.architecture", "").lower()
                        if arch in ("sd", "stable-diffusion", "flux", "qwen-image", "qwen_image", "wan", "diffusion") or "image" in arch or "diffusion" in arch or any(
                            kw in entry.name.lower() or kw in alias.lower()
                            for kw in [
                                "stable-diffusion",
                                "sdxl",
                                "sd1.",
                                "sd2.",
                                "sd3",
                                "flux",
                                "pony",
                                "photoreal",
                                "sd-",
                                "illustrious",
                                "qwen-image",
                                "qwen_image",
                                "wan-2",
                                "wan_2",
                            ]
                        ):
                            is_sd = True
                except Exception:
                    pass
                if is_sd:
                    continue

                is_moe = False
                param_count = 0
                flash_attn = False
                projector_path = None

                # Look for a projector file for this model
                for p_entry in router_dir.iterdir():
                    if p_entry.suffix == ".gguf" and p_entry != entry:
                        p_name = p_entry.name.lower()
                        if "projector" in p_name or "mmproj" in p_name:
                            stem_clean = (
                                entry.stem.lower()
                                .replace("--latest", "")
                                .replace("--latest-gguf", "")
                            )
                            p_clean = p_entry.stem.lower()
                            if (
                                stem_clean in p_clean
                                or p_clean.startswith(stem_clean)
                                or len(list(router_dir.glob("*projector*.gguf"))) == 1
                                or len(list(router_dir.glob("*mmproj*.gguf"))) == 1
                            ):
                                projector_path = f"/router-models/{p_entry.name}"
                                break

                try:
                    meta = _read_gguf_metadata(str(resolved))
                    is_moe = _is_moe(meta)
                    param_count = _model_size_params(meta)
                    flash_attn = _supports_flash_attn(meta)
                except Exception as e:
                    print(
                        f"Warning: could not read model metadata for {entry.name}: {e}",
                        file=sys.stderr,
                    )

                small_model = param_count > 0 and param_count < 9_000_000_000
                is_mtp_capable = ("mtp" in entry.name.lower() or "mtp" in alias.lower()) and (
                    entry.name not in mtp_incompatible and alias not in mtp_incompatible
                )
                is_safe = alias in safe_settings or entry.name in safe_settings

                profile_file = router_dir / f"{alias}.profile.json"
                profile = {}
                if profile_file.exists():
                    try:
                        with open(profile_file) as pf:
                            profile = json.load(pf)
                    except Exception as e:
                        print(
                            f"Warning: could not read model profile for {alias}: {e}",
                            file=sys.stderr,
                        )

                model_settings = {}
                if projector_path:
                    model_settings["mmproj"] = projector_path
                if is_safe or small_model:
                    model_settings["ctx-size"] = "8192"
                    model_settings["cache-type-k"] = "f16"
                    model_settings["cache-type-v"] = "f16"
                else:
                    if not is_moe:
                        model_settings["ctx-size"] = "32768"
                    else:
                        model_settings["ctx-size"] = "98304"
                    model_settings["cache-type-k"] = "q4_0"
                    model_settings["cache-type-v"] = "q4_0"

                if is_safe:
                    model_settings["flash-attn"] = "off"
                elif flash_attn:
                    model_settings["flash-attn"] = "on"
                else:
                    model_settings["flash-attn"] = "off"

                if is_moe:
                    model_settings["n-cpu-moe"] = "40"
                    if is_mtp_capable:
                        model_settings["spec-type"] = "draft-mtp"
                        model_settings["spec-draft-n-max"] = "3"
                    else:
                        model_settings["spec-type"] = "none"
                        model_settings["spec-draft-n-max"] = "0"
                else:
                    model_settings["spec-type"] = "none"
                    model_settings["spec-draft-n-max"] = "0"

                if profile:
                    for k, v in profile.items():
                        if k == "model":
                            continue
                        model_settings[k] = str(v)

                content.append(f"[{alias}]")
                content.append(f"model = /router-models/{entry.name}")
                for k, v in model_settings.items():
                    content.append(f"{k} = {v}")

                content.append("")

    temp_ini = ini_path.with_suffix(".ini.tmp")
    with open(temp_ini, "w", encoding="utf-8") as f:
        f.write("\n".join(content))
    with contextlib.suppress(Exception):
        os.chmod(temp_ini, 0o666)
    os.replace(temp_ini, ini_path)
    with contextlib.suppress(Exception):
        os.chmod(ini_path, 0o666)
    print(f"Updated models preset configuration at {ini_path}")


def get_model_info(model_name):
    normalized = normalize_model_name(model_name)
    name, tag = normalized.rsplit(":", 1)
    repo = f"library/{name}" if "/" not in name else name
    manifest_path = Path(MODELS_DIR) / "manifests" / "registry.ollama.ai" / repo / tag
    return repo, tag, manifest_path


def manifest_path_for_local_name(local_name):
    return get_model_info(local_name)[2]


def model_layer(manifest):
    for layer in manifest.get("layers", []):
        media_type = layer.get("mediaType", "")
        if media_type.startswith(MODEL_LAYER_MEDIA_TYPE):
            return layer
    return {}


def normalized_model_parts(model_name):
    normalized = normalize_model_name(model_name)
    name, tag = normalized.rsplit(":", 1)
    return name, tag


def router_filename_for_model_name(model_name):
    name, tag = normalized_model_parts(model_name)
    flattened = f"{name}--{tag}".replace("/", "--")
    ext = (
        ".safetensors"
        if "safetensors" in model_name.lower() or "safetensors" in tag.lower()
        else ".gguf"
    )
    return f"{sanitize_model_component(flattened)}{ext}"


def router_models_dir():
    return Path(ROUTER_MODELS_DIR)


def router_model_path_for_name(model_name):
    return router_models_dir() / router_filename_for_model_name(model_name)


def model_blobs(manifest):
    layers = manifest.get("layers", [])
    config = manifest.get("config", {})
    for layer in [config, *layers]:
        digest = layer.get("digest")
        if digest:
            yield digest


def blob_path_for_digest(digest):
    return Path(MODELS_DIR) / "blobs" / digest.replace(":", "-")


def load_manifest(manifest_path):
    with open(manifest_path, encoding="utf-8") as handle:
        return json.load(handle)


def write_manifest_atomic(manifest_path, manifest):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = manifest_path.with_suffix(f"{manifest_path.suffix}.tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    os.replace(temp_path, manifest_path)


def is_model_complete(manifest):
    for layer in [manifest.get("config", {}), *manifest.get("layers", [])]:
        digest = layer.get("digest")
        if not digest:
            continue
        blob_path = blob_path_for_digest(digest)
        if not blob_path.exists():
            return False
        if blob_path.stat().st_size != layer.get("size", 0):
            return False
    return True


def ensure_router_symlink(model_name, manifest):
    layer = model_layer(manifest)
    digest = layer.get("digest")
    if not digest:
        raise RuntimeError(f"Model {normalize_model_name(model_name)} has no GGUF model layer.")

    blob_path = blob_path_for_digest(digest)
    if not blob_path.exists():
        raise RuntimeError(f"Missing blob for {normalize_model_name(model_name)}: {blob_path}")

    router_path = router_model_path_for_name(model_name)
    router_path.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(blob_path, start=router_path.parent)

    if router_path.exists() or router_path.is_symlink():
        if router_path.is_symlink() and os.readlink(router_path) == relative_target:
            return router_path
        router_path.unlink()

    os.symlink(relative_target, router_path)
    return router_path


def iter_local_models():
    manifest_root = Path(MODELS_DIR) / "manifests" / "registry.ollama.ai"
    if not manifest_root.exists():
        return

    for path in manifest_root.rglob("*"):
        if not path.is_file() or "sha256" in path.name:
            continue
        try:
            manifest = load_manifest(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not is_model_complete(manifest):
            continue

        rel = path.relative_to(manifest_root)
        parts = rel.parts
        if len(parts) < 2:
            continue
        tag = parts[-1]
        name = "/".join(parts[:-1])
        if name.startswith("library/"):
            name = name[8:]
        yield normalize_model_name(f"{name}:{tag}"), path, manifest


def manifest_response(repo, tag, client, insecure=False):
    headers = {}
    url = f"{OLLAMA_REGISTRY}/{repo}/manifests/{tag}"
    if insecure and url.startswith("https://"):
        url = "http://" + url[len("https://") :]

    resp = client.get(url)
    if resp.status_code == 401:
        auth_url = OLLAMA_AUTH_URL.format(repo=repo)
        token_resp = client.get(auth_url)
        token_resp.raise_for_status()
        token = token_resp.json().get("token")
        if not token:
            raise RuntimeError("Registry auth succeeded but no token was returned.")
        headers["Authorization"] = f"Bearer {token}"
        resp = client.get(url, headers=headers)
    resp.raise_for_status()
    return resp, headers


def download_blob(client, repo, digest, expected_size, headers):
    blob_path = blob_path_for_digest(digest)
    blob_path.parent.mkdir(parents=True, exist_ok=True)

    current_size = 0
    if blob_path.exists():
        current_size = blob_path.stat().st_size
        if current_size == expected_size:
            print(f"Layer {digest[:12]} already complete.")
            return
        if current_size > expected_size:
            print(f"Layer {digest[:12]} is larger than expected. Restarting...")
            blob_path.unlink()
            current_size = 0
        else:
            print(f"Resuming layer {digest[:12]} from {current_size // 1024 // 1024} MB...")

    blob_url = f"{OLLAMA_REGISTRY}/{repo}/blobs/{digest}"
    request_headers = dict(headers)
    if current_size > 0:
        request_headers["Range"] = f"bytes={current_size}-"

    with client.stream("GET", blob_url, headers=request_headers) as response:
        if response.status_code == 416:
            print(f"Layer {digest[:12]} verified.")
            return

        mode = "ab" if response.status_code == 206 else "wb"
        if response.status_code == 200 and current_size > 0:
            print("Server does not support resume. Restarting download...")
            current_size = 0
        response.raise_for_status()

        total = int(response.headers.get("Content-Length", 0)) + current_size
        if tqdm:
            with (
                open(blob_path, mode) as handle,
                tqdm(
                    total=total,
                    initial=current_size,
                    unit_divisor=1024,
                    unit="B",
                    unit_scale=True,
                    desc=f"Pulling {digest[:12]}",
                ) as bar,
            ):
                for chunk in response.iter_bytes():
                    if _should_stop():
                        print(
                            f"\nDownload interrupted: {digest[:12]} ({blob_path.stat().st_size} bytes)"
                        )
                        return
                    handle.write(chunk)
                    bar.update(len(chunk))
        else:
            downloaded = current_size
            with open(blob_path, mode) as handle:
                for chunk in response.iter_bytes():
                    if _should_stop():
                        print(
                            f"\nDownload interrupted: {digest[:12]} ({blob_path.stat().st_size} bytes)"
                        )
                        return
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        sys.stdout.write(
                            f"\rProgress {digest[:12]}: {downloaded / total * 100:.1f}%"
                        )
                        sys.stdout.flush()
            print()

    final_size = blob_path.stat().st_size
    if final_size != expected_size:
        raise RuntimeError(
            f"Layer {digest[:12]} downloaded with size {final_size}, expected {expected_size}."
        )


def choose_source(model_name, source):
    if source != "auto":
        return source
    if model_name.startswith(("hf://", "https://huggingface.co/", "https://hf.co/")):
        return "huggingface"
    name = model_name.split(":")[0] if ":" in model_name else model_name
    if "/" in name and "gguf" in name.lower():
        return "huggingface"
    return "ollama"


def parse_huggingface_ref(model_name):
    if model_name.startswith("hf://"):
        parsed = urlparse(model_name)
        full_path = parsed.netloc + "/" + parsed.path.lstrip("/")
        path_parts = [part for part in full_path.split("/") if part]
        if len(path_parts) < 2:
            raise ValueError("Expected hf://<repo>/<filename> for Hugging Face pulls.")
        filename = path_parts[-1]
        repo = "/".join(path_parts[:-1])
        if not repo:
            raise ValueError("Expected hf://<repo>/<filename> for Hugging Face pulls.")
        return repo, filename

    if model_name.startswith(("https://huggingface.co/", "https://hf.co/")):
        parsed = urlparse(model_name)
        parts = [part for part in parsed.path.split("/") if part]
        if "resolve" not in parts:
            raise ValueError("Expected a Hugging Face resolve URL ending in the filename.")
        resolve_index = parts.index("resolve")
        if resolve_index < 2 or resolve_index + 2 >= len(parts):
            raise ValueError("Could not parse repository and filename from Hugging Face URL.")
        repo = "/".join(parts[:resolve_index])
        filename = "/".join(parts[resolve_index + 2 :])
        return repo, unquote(filename)

    if ":" in model_name and "/" in model_name:
        repo, filename = model_name.split(":", 1)
        if filename:
            return repo, filename

    raise ValueError(
        "For Hugging Face pulls, use one of: hf://repo/filename, https://huggingface.co/repo/resolve/main/filename, or repo:filename."
    )


def huggingface_headers():
    headers = {"User-Agent": "alpaca-puller/1.0"}
    if HUGGING_FACE_TOKEN:
        headers["Authorization"] = f"Bearer {HUGGING_FACE_TOKEN}"
    return headers


def download_with_progress(client, url, headers, label, output_path, expected_total=None):
    current_size = output_path.stat().st_size if output_path.exists() else 0
    if current_size > 0:
        headers["Range"] = f"bytes={current_size}-"
        print(f"Resuming {label} from {current_size // 1024 // 1024} MB...")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            with client.stream("GET", url, headers=headers) as response:
                if response.status_code == 416:
                    print(f"{label} already complete according to server.")
                    return output_path

                mode = "ab" if response.status_code == 206 else "wb"
                if response.status_code == 200 and current_size > 0:
                    print("Server does not support resume. Restarting download...")
                    current_size = 0

                if response.status_code >= 400:
                    body = response.read()
                    raise httpx.HTTPStatusError(
                        f"{response.status_code}: {body.decode(errors='ignore')}",
                        request=response.request,
                        response=response,
                    )

                total = int(response.headers.get("Content-Length", 0)) + current_size
                if tqdm:
                    with (
                        open(output_path, mode) as handle,
                        tqdm(
                            total=total if total > 0 else None,
                            initial=current_size,
                            unit_divisor=1024,
                            unit="B",
                            unit_scale=True,
                            desc=label,
                        ) as bar,
                    ):
                        for chunk in response.iter_bytes():
                            if _should_stop():
                                print(
                                    f"\nDownload interrupted: {label} ({output_path.stat().st_size} bytes)"
                                )
                                return None
                            handle.write(chunk)
                            bar.update(len(chunk))
                else:
                    downloaded = current_size
                    with open(output_path, mode) as handle:
                        for chunk in response.iter_bytes():
                            if _should_stop():
                                print(
                                    f"\nDownload interrupted: {label} ({output_path.stat().st_size} bytes)"
                                )
                                return None
                            handle.write(chunk)
                            downloaded += len(chunk)
                            if total > 0:
                                sys.stdout.write(
                                    f"\rProgress {label}: {downloaded / total * 100:.1f}%"
                                )
                                sys.stdout.flush()
                    if total > 0:
                        print()

                return output_path
        except (httpx.ReadError, httpx.RemoteProtocolError, ssl.SSLError) as exc:
            if attempt == max_retries - 1:
                raise
            print(f"\nDownload interrupted ({exc}). Retrying {attempt + 1}/{max_retries}...")
            time.sleep(2 * (attempt + 1))


def list_huggingface_files(repo):
    url = f"{HUGGING_FACE_BASE}/api/models/{repo}"
    try:
        resp = httpx.get(url, headers=huggingface_headers(), timeout=30.0)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def resolve_huggingface_filename(repo, requested_filename):
    supported_exts = (".gguf", ".safetensors", ".ckpt", ".bin")
    if requested_filename.endswith(supported_exts):
        return requested_filename
    model_info = list_huggingface_files(repo)
    if not model_info or not isinstance(model_info, dict):
        return requested_filename
    siblings = model_info.get("siblings", [])
    files = [
        s.get("rfilename", "") for s in siblings if s.get("rfilename", "").endswith(supported_exts)
    ]
    if not files:
        return requested_filename
    target_stem = requested_filename.replace(" ", "").replace("_", "").lower()
    exact_match = None
    best_match = None
    for fname in files:
        stem = Path(fname).stem.replace(" ", "").replace("_", "").lower()
        if stem == target_stem:
            exact_match = fname
            break
        if target_stem in stem and best_match is None:
            best_match = fname
    if exact_match:
        return exact_match
    if best_match:
        return best_match
    for ext in supported_exts:
        fallback = f"{repo.rsplit('/', 1)[-1]}-{requested_filename}{ext}"
        for fname in files:
            if fname == fallback:
                return fname
    return files[0]


def huggingface_blob_url(repo, filename):
    encoded_parts = [part for part in filename.split("/") if part]
    return f"{HUGGING_FACE_BASE.rstrip('/')}/{repo}/resolve/main/" + "/".join(encoded_parts)


def hash_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def write_json_blob(payload):
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    digest = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
    blob_path = blob_path_for_digest(digest)
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    if not blob_path.exists():
        blob_path.write_bytes(encoded)
    return digest, len(encoded)


def write_text_blob(text):
    encoded = text.encode("utf-8")
    digest = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
    blob_path = blob_path_for_digest(digest)
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    if not blob_path.exists():
        blob_path.write_bytes(encoded)
    return digest, len(encoded)


def infer_local_name_from_huggingface(repo, filename):
    repo_name = sanitize_model_component(repo.rsplit("/", 1)[-1])
    file_name = sanitize_model_component(Path(filename).stem)
    return f"{repo_name}/{file_name}:latest"


def partial_download_path(kind, key):
    key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    partial_dir = Path(MODELS_DIR) / "partials" / kind
    partial_dir.mkdir(parents=True, exist_ok=True)
    return partial_dir / f"{key_hash}.part"


def download_to_partial_file(client, url, headers, label, partial_path):
    current_size = partial_path.stat().st_size if partial_path.exists() else 0
    request_headers = dict(headers)
    if current_size > 0:
        request_headers["Range"] = f"bytes={current_size}-"
        print(f"Resuming {label} from {current_size // 1024 // 1024} MB...")

    with client.stream("GET", url, headers=request_headers) as response:
        if response.status_code == 416:
            print(f"{label} already complete according to server.")
            return partial_path

        mode = "ab" if response.status_code == 206 else "wb"
        if response.status_code == 200 and current_size > 0:
            print("Server does not support resume. Restarting download...")
            current_size = 0
        response.raise_for_status()

        total = int(response.headers.get("Content-Length", 0)) + current_size
        if tqdm:
            with (
                open(partial_path, mode) as handle,
                tqdm(
                    total=total if total > 0 else None,
                    initial=current_size,
                    unit_divisor=1024,
                    unit="B",
                    unit_scale=True,
                    desc=label,
                ) as bar,
            ):
                for chunk in response.iter_bytes():
                    if _should_stop():
                        print(
                            f"\nDownload interrupted: {label} ({partial_path.stat().st_size} bytes)"
                        )
                        return None
                    handle.write(chunk)
                    bar.update(len(chunk))
        else:
            downloaded = current_size
            with open(partial_path, mode) as handle:
                for chunk in response.iter_bytes():
                    if _should_stop():
                        print(
                            f"\nDownload interrupted: {label} ({partial_path.stat().st_size} bytes)"
                        )
                        return None
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        sys.stdout.write(f"\rProgress {label}: {downloaded / total * 100:.1f}%")
                        sys.stdout.flush()
            if total > 0:
                print()

    return partial_path


def import_huggingface_gguf(model_name, local_name=None, insecure=False, no_resume=False):
    if insecure:
        raise RuntimeError("--insecure is only supported for Ollama registry pulls.")

    repo, filename = parse_huggingface_ref(model_name)
    filename = resolve_huggingface_filename(repo, filename)
    local_name = local_name or infer_local_name_from_huggingface(repo, filename)
    manifest_path = manifest_path_for_local_name(local_name)
    url = huggingface_blob_url(repo, filename)
    headers = huggingface_headers()
    partial_path = partial_download_path("huggingface", f"{repo}:{filename}")

    print(f"Importing Hugging Face GGUF: {repo}/{filename}")
    print(f"Local Ollama model name: {normalize_model_name(local_name)}")
    if no_resume and partial_path.exists():
        print("Ignoring partial download and starting fresh.")
        partial_path.unlink()
    client = httpx.Client(
        timeout=httpx.Timeout(300.0, connect=30.0, read=300.0),
        follow_redirects=True,
    )

    try:
        download_with_progress(
            client, url, headers, f"Importing {Path(filename).name}", partial_path
        )
        if _should_stop():
            print(f"\nInterrupted: {local_name} ({partial_path.stat().st_size} bytes downloaded).")
            return 1
        digest = hash_file(partial_path)
        blob_path = blob_path_for_digest(digest)
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        if blob_path.exists():
            existing_size = blob_path.stat().st_size
            incoming_size = partial_path.stat().st_size
            if existing_size != incoming_size:
                raise RuntimeError(
                    f"Blob digest collision with mismatched size at {blob_path} ({existing_size} != {incoming_size})."
                )
            partial_path.unlink(missing_ok=True)
        else:
            os.replace(partial_path, blob_path)

        def is_stable_diffusion_model(repo_name: str, file_name: str) -> bool:
            name_lower = f"{repo_name}/{file_name}".lower()
            sd_keywords = [
                "stable-diffusion",
                "sdxl",
                "sd1.",
                "sd2.",
                "sd3",
                "flux",
                "pony",
                "photoreal",
                "sd-",
                "illustrious",
                "safetensors",
                "qwen-image",
                "qwen_image",
            ]
            return any(kw in name_lower for kw in sd_keywords)

        is_sd = is_stable_diffusion_model(repo, filename)
        is_safetensors = filename.endswith(".safetensors")
        config = {
            "model_format": "safetensors" if is_safetensors else "gguf",
            "model_family": "stable-diffusion" if is_sd else "gguf",
            "families": ["stable-diffusion" if is_sd else "gguf"],
            "general.source": "huggingface",
            "general.source_repo": repo,
            "general.source_file": filename,
        }
        config_digest, config_size = write_json_blob(config)
        blob_size = blob_path.stat().st_size
        template_digest, template_size = write_text_blob("{{ .Prompt }}")
        params_digest, params_size = write_text_blob("{}")
        manifest = {
            "schemaVersion": 2,
            "mediaType": MANIFEST_MEDIA_TYPE,
            "config": {
                "mediaType": CONFIG_MEDIA_TYPE,
                "digest": config_digest,
                "size": config_size,
            },
            "layers": [
                {
                    "mediaType": MODEL_LAYER_MEDIA_TYPE,
                    "digest": digest,
                    "size": blob_size,
                },
                {
                    "mediaType": "application/vnd.ollama.image.template",
                    "digest": template_digest,
                    "size": template_size,
                },
                {
                    "mediaType": "application/vnd.ollama.image.params",
                    "digest": params_digest,
                    "size": params_size,
                },
            ],
        }
        write_manifest_atomic(manifest_path, manifest)
        router_path = ensure_router_symlink(local_name, manifest)
        update_models_ini()
        create_model_profile(router_path, manifest=manifest, repo=repo, filename=filename)
        print(f"Registered router model: {router_path.name}")
        print(f"\nSuccessfully imported {normalize_model_name(local_name)}")
        return 0
    finally:
        client.close()


def pull_ollama_model(model_name, insecure=False, no_resume=False):
    repo, tag, manifest_path = get_model_info(model_name)
    print(f"Resolving model: {repo}:{tag}...")
    client = httpx.Client(timeout=httpx.Timeout(60.0, read=None), follow_redirects=True)

    try:
        response, headers = manifest_response(repo, tag, client, insecure=insecure)
        manifest = response.json()
        # Clear partial blobs when no_resume is requested
        if no_resume:
            for layer in [manifest.get("config", {}), *manifest.get("layers", [])]:
                digest = layer.get("digest")
                if digest:
                    blob_path = blob_path_for_digest(digest)
                    if blob_path.exists():
                        blob_size = blob_path.stat().st_size
                        expected = layer.get("size", 0)
                        if blob_size != expected:
                            blob_path.unlink()
                            print(f"Removed incomplete blob {digest[:12]} ({blob_size} bytes)")
        for layer in [manifest.get("config", {}), *manifest.get("layers", [])]:
            digest = layer.get("digest")
            if not digest:
                continue
            if _should_stop():
                print(
                    f"\nInterrupted: {model_name} (downloaded {sum(l.get('size', 0) for l in manifest.get('layers', []))} bytes)."
                )
                return 1
            download_blob(client, repo, digest, layer.get("size", 0), headers)
            if _should_stop():
                print(
                    f"\nInterrupted: {model_name} (downloaded {sum(l.get('size', 0) for l in manifest.get('layers', []))} bytes)."
                )
                return 1
        write_manifest_atomic(manifest_path, manifest)
        router_path = ensure_router_symlink(model_name, manifest)
        update_models_ini()
        create_model_profile(router_path, manifest=manifest)
        print(f"Registered router model: {router_path.name}")
        print(f"\nSuccessfully pulled {normalize_model_name(model_name)}")
        return 0
    finally:
        client.close()


def pull_companion(model_name, no_resume=False):
    try:
        repo, filename = parse_huggingface_ref(model_name)
    except ValueError as exc:
        print(f"Error parsing Hugging Face ref: {exc}")
        return 1
    filename = resolve_huggingface_filename(repo, filename)
    url = huggingface_blob_url(repo, filename)
    headers = huggingface_headers()

    # Companions always go into .alpaca-router/companions/ so the reindexer
    # can discover them relative to the router directory, and so the container
    # can access them at /router-models/companions/ without needing /models/.
    router_dir = Path(ROUTER_MODELS_DIR)
    companions_dir = router_dir / "companions"
    try:
        companions_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating companions directory: {e}")
        return 1

    output_path = companions_dir / filename

    print(f"Downloading companion file: {repo}/{filename}")
    print(f"Destination: {output_path}")

    partial_path = partial_download_path("companion", f"{repo}:{filename}")
    if no_resume and partial_path.exists():
        with contextlib.suppress(Exception):
            partial_path.unlink()

    client = httpx.Client(
        timeout=httpx.Timeout(300.0, connect=30.0, read=300.0),
        follow_redirects=True,
    )
    try:
        download_with_progress(client, url, headers, f"Companion {filename}", partial_path)
        if _should_stop():
            print(
                f"\nInterrupted: companion download ({partial_path.stat().st_size} bytes downloaded)."
            )
            return 1

        # Move partial file to final companion location
        if output_path.exists():
            with contextlib.suppress(Exception):
                output_path.unlink()
        try:
            os.replace(partial_path, output_path)
        except Exception as e:
            print(f"Error moving companion file: {e}")
            return 1

        with contextlib.suppress(Exception):
            os.chmod(output_path, 0o666)
        print(f"Successfully downloaded companion file: {output_path.name}")

        # If this looks like a VL/LLM GGUF (not a projector/VAE), create a router
        # symlink and profile.json so the reindexer wires up mmproj automatically.
        fn_lower = filename.lower()
        is_vl_model = (
            fn_lower.endswith(".gguf")
            and "mmproj" not in fn_lower
            and "projector" not in fn_lower
            and "vae" not in fn_lower
            and "clip" not in fn_lower
        )
        if is_vl_model:
            # Derive a clean router alias: keep only alphanumeric, dots, dashes
            stem = Path(filename).stem
            alias = re.sub(r"[^A-Za-z0-9._-]", "-", stem).lower()
            # Collapse duplicate dashes and convert single dashes to double (router convention)
            alias = re.sub(r"-+", "--", alias).strip("-")
            symlink_path = router_dir / f"{alias}.gguf"
            if not symlink_path.exists():
                try:
                    symlink_path.symlink_to(f"companions/{filename}")
                    with contextlib.suppress(Exception):
                        os.chmod(symlink_path, 0o666)
                    print(f"Created router symlink: {symlink_path.name} -> companions/{filename}")
                except Exception as e:
                    print(f"Warning: could not create router symlink: {e}")

            # Find the mmproj file in companions if one exists
            mmproj_candidates = sorted(companions_dir.glob("*mmproj*.gguf")) + \
                                sorted(companions_dir.glob("*projector*.gguf"))
            mmproj_rel = None
            if mmproj_candidates:
                mmproj_rel = f"/router-models/companions/{mmproj_candidates[0].name}"

            profile_path = router_dir / f"{alias}.profile.json"
            if not profile_path.exists():
                profile: dict = {
                    "ctx-size": "8192",
                    "n-gpu-layers": "99",
                    "kv-unified": "false",
                    "flash-attn": "on",
                    "cache-type-k": "f16",
                    "cache-type-v": "f16",
                    "spec-type": "none",
                    "spec-draft-n-max": "0",
                }
                if mmproj_rel:
                    profile["mmproj"] = mmproj_rel
                try:
                    profile_path.write_text(json.dumps(profile, indent=2) + "\n")
                    with contextlib.suppress(Exception):
                        os.chmod(profile_path, 0o666)
                    print(f"Created VL model profile: {profile_path.name}")
                except Exception as e:
                    print(f"Warning: could not write profile: {e}")

        return 0
    except Exception as e:
        print(f"Error downloading companion file: {e}")
        return 1
    finally:
        client.close()



def pull_model(
    model_name, source="auto", local_name=None, insecure=False, no_resume=False, companion=False
):
    global _CURRENT_MODEL
    _CURRENT_MODEL = model_name
    if companion:
        return pull_companion(model_name, no_resume=no_resume)

    resolved_source = choose_source(model_name, source)
    try:
        if resolved_source == "huggingface":
            result = import_huggingface_gguf(
                model_name, local_name=local_name, insecure=insecure, no_resume=no_resume
            )
        else:
            result = pull_ollama_model(model_name, insecure=insecure, no_resume=no_resume)
        if result == 0:
            reindex_models()
        return result
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        print(f"Error: request failed ({status})")
        return 1
    except httpx.ReadTimeout:
        print("\nTimeout reached while pulling. Please run the command again to resume.")
        return 1
    except Exception as exc:
        print(f"\nError pulling model: {exc}")
        return 1
    finally:
        _CURRENT_MODEL = ""


def blob_referenced_elsewhere(digest, current_manifest_path):
    manifest_root = Path(MODELS_DIR) / "manifests" / "registry.ollama.ai"
    if not manifest_root.exists():
        return False

    for path in manifest_root.rglob("*"):
        if not path.is_file() or path == current_manifest_path or "sha256" in path.name:
            continue
        try:
            manifest = load_manifest(path)
        except (OSError, json.JSONDecodeError):
            continue
        if digest in set(model_blobs(manifest)):
            return True
    return False


def reindex_models():
    count = 0
    for model_name, _manifest_path, manifest in iter_local_models():
        router_path = ensure_router_symlink(model_name, manifest)
        print(f"Indexed {normalize_model_name(model_name)} -> {router_path.name}")
        count += 1
    if count == 0:
        print("No complete local models found to index.")
        return 0
    update_models_ini()
    print(f"Indexed {count} model(s) for llama-server router discovery.")
    return 0


def remove_model(model_name):
    _repo, _tag, manifest_path = get_model_info(model_name)
    if not manifest_path.exists():
        print(f"Error: Model {normalize_model_name(model_name)} not found locally.")
        return 1

    try:
        manifest = load_manifest(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Error: Could not read manifest {manifest_path}: {exc}")
        return 1

    print(f"Removing model {normalize_model_name(model_name)}...")
    manifest_path.unlink()
    print(f"Deleted manifest: {manifest_path}")

    router_path = router_model_path_for_name(model_name)
    if router_path.exists() or router_path.is_symlink():
        router_path.unlink()
        print(f"Deleted router index: {router_path}")
        # Remove any companion profile overlays for this model so they don't
        # linger as stale profiles in the dashboard after the model is gone.
        for prof_ext in (".profile.json", ".gguf.profile.json", ".safetensors.profile.json"):
            prof_path = router_path.with_name(router_path.stem + prof_ext)
            if prof_path.exists():
                prof_path.unlink()
                print(f"Deleted router profile: {prof_path}")
        update_models_ini()

    for digest in model_blobs(manifest):
        blob_path = blob_path_for_digest(digest)
        if not blob_path.exists():
            continue
        if blob_referenced_elsewhere(digest, manifest_path):
            print(f"Keeping shared blob: {digest[:12]}")
            continue
        try:
            blob_path.unlink()
            print(f"Deleted blob: {digest[:12]}")
        except OSError as exc:
            print(f"Could not delete blob {digest[:12]}: {exc}")

    print(f"Successfully removed {normalize_model_name(model_name)}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="Pull, reindex, or remove models in Ollama storage format."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pull_parser = subparsers.add_parser(
        "pull", help="Download a model into the local Ollama store."
    )
    pull_parser.add_argument(
        "model",
        help=(
            "Ollama model name, or a Hugging Face GGUF reference such as "
            "hf://repo/file.gguf, repo:file.gguf, or a huggingface.co resolve URL."
        ),
    )
    pull_parser.add_argument(
        "--source",
        choices=("auto", "ollama", "huggingface"),
        default="auto",
        help="Choose the upstream source. Defaults to auto-detect.",
    )
    pull_parser.add_argument(
        "--name",
        help="Local Ollama model name to create for Hugging Face imports, for example qwen3:35b-q4.",
    )
    pull_parser.add_argument(
        "--insecure",
        action="store_true",
        help="Allow HTTP access to the configured Ollama registry endpoint.",
    )
    pull_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any partial downloads and start from scratch.",
    )
    pull_parser.add_argument(
        "--companion",
        action="store_true",
        help="Download as a companion file (VAE, CLIP, etc.) and save to companion directory.",
    )

    remove_parser = subparsers.add_parser(
        "remove", help="Remove a local model manifest and unshared blobs."
    )
    remove_parser.add_argument(
        "model", help="Local model name, for example tinyllama or qwen3:35b-q4."
    )

    subparsers.add_parser(
        "reindex",
        help="Create or refresh llama-server router symlinks for all complete local models.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "pull":
        return pull_model(
            args.model,
            source=args.source,
            local_name=args.name,
            insecure=args.insecure,
            no_resume=args.no_resume,
            companion=args.companion,
        )
    if args.command == "reindex":
        return reindex_models()
    if args.command == "remove":
        return remove_model(args.model)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
