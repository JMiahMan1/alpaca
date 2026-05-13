import argparse
import json
import os
import subprocess
import sys

import requests

MODELS_DIR = os.getenv("MODELS_DIR", "/usr/share/ollama/.ollama/models")
PROXY_URL = os.getenv("PROXY_URL", "http://localhost:11434/api/tags")


def normalize_model_name(model_name):
    return model_name if ":" in model_name else f"{model_name}:latest"


def validate_user_model_name(model_name):
    if model_name.endswith(".gguf"):
        raise ValueError(
            f"Use the local model name, not the internal router filename: {model_name}. "
            "For example, use qwen3:8b instead of qwen3--8b.gguf."
        )


def manifest_path_for_model(model_name):
    normalized = normalize_model_name(model_name)
    name, tag = normalized.rsplit(":", 1)
    parts = name.split("/")
    if len(parts) == 1:
        parts = ["library", parts[0]]
    return os.path.join(MODELS_DIR, "manifests", "registry.ollama.ai", *parts, tag)


def public_model_name(model_name):
    normalized = normalize_model_name(model_name)
    if normalized.endswith(":latest"):
        return normalized[:-7]
    return normalized


def run_cmd(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def check_files_exist(model_name):
    manifest_path = manifest_path_for_model(model_name)
    found = os.path.exists(manifest_path)
    if not found:
        return False, f"manifest missing: {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    for layer in manifest.get("layers", []) + [manifest.get("config", {})]:
        digest = layer.get("digest")
        if not digest:
            continue
        blob_path = os.path.join(MODELS_DIR, "blobs", digest.replace(":", "-"))
        if not os.path.exists(blob_path):
            return False, f"blob missing: {digest}"
    return True, manifest_path


def check_ollama_list(model_name):
    code, stdout, stderr = run_cmd(["ollama", "list"])
    if code != 0:
        return False, stderr.strip() or "ollama list failed"
    target = public_model_name(model_name)
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    found = any(line.split()[0] in {target, normalize_model_name(model_name)} for line in lines[1:])
    return found, stdout.strip()


def check_ollama_show(model_name):
    code, stdout, stderr = run_cmd(["ollama", "show", normalize_model_name(model_name)])
    if code != 0:
        return False, stderr.strip() or stdout.strip() or "ollama show failed"
    return True, stdout.strip()


def check_proxy_tags(model_name):
    try:
        payload = requests.get(PROXY_URL, timeout=5).json()
    except Exception as exc:
        return False, f"proxy request failed: {exc}"

    target = public_model_name(model_name)
    models = [item.get("name", "") for item in payload.get("models", [])]
    found = any(name in {target, normalize_model_name(model_name)} for name in models)
    return found, json.dumps(models, indent=2)


def verify_model(model_name):
    checks = [
        ("disk files", check_files_exist),
        ("ollama list", check_ollama_list),
        ("ollama show", check_ollama_show),
        ("alpaca tags", check_proxy_tags),
    ]

    failures = 0
    print(f"=== Verifying {normalize_model_name(model_name)} ===")
    for label, func in checks:
        ok, detail = func(model_name)
        status = "PASS" if ok else "FAIL"
        print(f"\n[{status}] {label}")
        print(detail)
        if not ok:
            failures += 1

    if failures:
        print(f"\nVerification failed with {failures} failing check(s).")
        return 1

    print("\nAll checks passed.")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="Verify that a model is visible in the local Ollama store, native Ollama, and Alpaca."
    )
    parser.add_argument("model", help="Local model name, for example qwen3:8b or tinyllama.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    try:
        validate_user_model_name(args.model)
    except ValueError as exc:
        print(str(exc))
        sys.exit(2)
    sys.exit(verify_model(args.model))
