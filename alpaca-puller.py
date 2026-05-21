import argparse
import hashlib
import json
import os
import re
import sys
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
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN") or os.getenv("HF_TOKEN")
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
    return f"{sanitize_model_component(flattened)}.gguf"


def router_models_dir():
    return Path(ROUTER_MODELS_DIR)


def router_model_path_for_name(model_name):
    return router_models_dir() / router_filename_for_model_name(model_name)


def model_blobs(manifest):
    layers = manifest.get("layers", [])
    config = manifest.get("config", {})
    for layer in [config] + layers:
        digest = layer.get("digest")
        if digest:
            yield digest


def blob_path_for_digest(digest):
    return Path(MODELS_DIR) / "blobs" / digest.replace(":", "-")


def load_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_manifest_atomic(manifest_path, manifest):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = manifest_path.with_suffix(f"{manifest_path.suffix}.tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    os.replace(temp_path, manifest_path)


def is_model_complete(manifest):
    for layer in [manifest.get("config", {})] + manifest.get("layers", []):
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
            with open(blob_path, mode) as handle, tqdm(
                total=total,
                initial=current_size,
                unit_divisor=1024,
                unit="B",
                unit_scale=True,
                desc=f"Pulling {digest[:12]}",
            ) as bar:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
                    bar.update(len(chunk))
        else:
            downloaded = current_size
            with open(blob_path, mode) as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        sys.stdout.write(f"\rProgress {digest[:12]}: {downloaded / total * 100:.1f}%")
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
    return "ollama"


def parse_huggingface_ref(model_name):
    if model_name.startswith("hf://"):
        parsed = urlparse(model_name)
        repo = parsed.netloc
        filename = parsed.path.lstrip("/")
        if repo and filename:
            return repo, filename
        raise ValueError("Expected hf://<repo>/<file.gguf> for Hugging Face pulls.")

    if model_name.startswith(("https://huggingface.co/", "https://hf.co/")):
        parsed = urlparse(model_name)
        parts = [part for part in parsed.path.split("/") if part]
        if "resolve" not in parts:
            raise ValueError("Expected a Hugging Face resolve URL ending in the GGUF filename.")
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
        "For Hugging Face pulls, use one of: hf://repo/file.gguf, https://huggingface.co/repo/resolve/main/file.gguf, or repo:file.gguf."
    )


def huggingface_headers():
    if not HUGGING_FACE_TOKEN:
        return {}
    return {"Authorization": f"Bearer {HUGGING_FACE_TOKEN}"}


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
            with open(partial_path, mode) as handle, tqdm(
                total=total if total > 0 else None,
                initial=current_size,
                unit_divisor=1024,
                unit="B",
                unit_scale=True,
                desc=label,
            ) as bar:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
                    bar.update(len(chunk))
        else:
            downloaded = current_size
            with open(partial_path, mode) as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        sys.stdout.write(f"\rProgress {label}: {downloaded / total * 100:.1f}%")
                        sys.stdout.flush()
            if total > 0:
                print()

    return partial_path


def import_huggingface_gguf(model_name, local_name=None, insecure=False):
    if insecure:
        raise RuntimeError("--insecure is only supported for Ollama registry pulls.")

    repo, filename = parse_huggingface_ref(model_name)
    local_name = local_name or infer_local_name_from_huggingface(repo, filename)
    manifest_path = manifest_path_for_local_name(local_name)
    url = huggingface_blob_url(repo, filename)
    headers = huggingface_headers()
    partial_path = partial_download_path("huggingface", f"{repo}:{filename}")

    print(f"Importing Hugging Face GGUF: {repo}/{filename}")
    print(f"Local Ollama model name: {normalize_model_name(local_name)}")
    client = httpx.Client(timeout=httpx.Timeout(60.0, read=None), follow_redirects=True)

    try:
        download_to_partial_file(client, url, headers, f"Importing {Path(filename).name}", partial_path)
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

        config = {
            "model_format": "gguf",
            "model_family": "gguf",
            "families": ["gguf"],
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
        print(f"Registered router model: {router_path.name}")
        print(f"\nSuccessfully imported {normalize_model_name(local_name)}")
        return 0
    finally:
        client.close()


def pull_ollama_model(model_name, insecure=False):
    repo, tag, manifest_path = get_model_info(model_name)
    print(f"Resolving model: {repo}:{tag}...")
    client = httpx.Client(timeout=httpx.Timeout(60.0, read=None), follow_redirects=True)

    try:
        response, headers = manifest_response(repo, tag, client, insecure=insecure)
        manifest = response.json()
        for layer in [manifest.get("config", {})] + manifest.get("layers", []):
            digest = layer.get("digest")
            if not digest:
                continue
            download_blob(client, repo, digest, layer.get("size", 0), headers)
        write_manifest_atomic(manifest_path, manifest)
        router_path = ensure_router_symlink(model_name, manifest)
        print(f"Registered router model: {router_path.name}")
        print(f"\nSuccessfully pulled {normalize_model_name(model_name)}")
        return 0
    finally:
        client.close()


def pull_model(model_name, source="auto", local_name=None, insecure=False):
    resolved_source = choose_source(model_name, source)
    try:
        if resolved_source == "huggingface":
            return import_huggingface_gguf(model_name, local_name=local_name, insecure=insecure)
        return pull_ollama_model(model_name, insecure=insecure)
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
    print(f"Indexed {count} model(s) for llama-server router discovery.")
    return 0


def remove_model(model_name):
    repo, tag, manifest_path = get_model_info(model_name)
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
    parser = argparse.ArgumentParser(description="Pull, reindex, or remove models in Ollama storage format.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pull_parser = subparsers.add_parser("pull", help="Download a model into the local Ollama store.")
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

    remove_parser = subparsers.add_parser("remove", help="Remove a local model manifest and unshared blobs.")
    remove_parser.add_argument("model", help="Local model name, for example tinyllama or qwen3:35b-q4.")

    subparsers.add_parser("reindex", help="Create or refresh llama-server router symlinks for all complete local models.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "pull":
        return pull_model(args.model, source=args.source, local_name=args.name, insecure=args.insecure)
    if args.command == "reindex":
        return reindex_models()
    if args.command == "remove":
        return remove_model(args.model)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
