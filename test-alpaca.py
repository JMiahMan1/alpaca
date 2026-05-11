import json
import os
import subprocess

import requests

TEST_MODEL = "tinyllama:latest"
MODELS_DIR = "/usr/share/ollama/.ollama/models"
PROXY_URL = "http://localhost:11434/api/tags"

def run_cmd(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.returncode == 0

def check_files_exist(exists=True):
    # Resolve the manifest path
    manifest_path = os.path.join(MODELS_DIR, "manifests", "registry.ollama.ai", "library", "tinyllama", "latest")
    found = os.path.exists(manifest_path)
    
    if found != exists:
        print(f"FAIL: Manifest existence is {found}, expected {exists}")
        return False
    
    if found:
        # Check blobs
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            for layer in manifest.get("layers", []) + [manifest.get("config", {})]:
                digest = layer.get("digest")
                if digest:
                    blob_path = os.path.join(MODELS_DIR, "blobs", digest.replace(":", "-"))
                    if not os.path.exists(blob_path):
                        print(f"FAIL: Blob {digest} missing!")
                        return False
    return True

def check_proxy_tags(contains=True):
    try:
        resp = requests.get(PROXY_URL, timeout=5).json()
        models = [m['name'] for m in resp.get('models', [])]
        found = any(m.startswith("tinyllama") for m in models)
        if found != contains:
            print(f"FAIL: Proxy tags contains tinyllama: {found}, expected {contains}")
            return False
        return True
    except Exception as e:
        print(f"Proxy check failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Standalone Puller Integration Test ===\n")
    
    # 1. Pull
    print("Step 1: Pulling model...")
    if not run_cmd(["python3", "alpaca-puller.py", "pull", TEST_MODEL]):
        exit(1)
    
    # 2. Verify files
    print("\nStep 2: Verifying disk files...")
    if not check_files_exist(exists=True):
        exit(1)
    print("Files verified on disk.")
    
    # 3. Verify Proxy
    print("\nStep 3: Verifying proxy tags...")
    if not check_proxy_tags(contains=True):
        exit(1)
    print("Proxy correctly detected the new model.")
    
    # 4. Remove
    print("\nStep 4: Removing model...")
    if not run_cmd(["python3", "alpaca-puller.py", "remove", TEST_MODEL]):
        exit(1)
    
    # 5. Verify Cleanup
    print("\nStep 5: Verifying cleanup...")
    if not check_files_exist(exists=False):
        exit(1)
    if not check_proxy_tags(contains=False):
        exit(1)
    
    print("\nALL TESTS PASSED: Standalone pull and cleanup is functional.")
