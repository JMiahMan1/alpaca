import json
import os
import sys

import httpx

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

OLLAMA_REGISTRY = "https://registry.ollama.ai/v2"
MODELS_DIR = "/usr/share/ollama/.ollama/models"

def get_model_info(model_name):
    if ":" not in model_name:
        model_name += ":latest"
    name, tag = model_name.split(":")
    repo = f"library/{name}" if "/" not in name else name
    manifest_path = os.path.join(MODELS_DIR, "manifests", "registry.ollama.ai", repo, tag)
    return repo, tag, manifest_path

def pull_model(model_name):
    repo, tag, manifest_path = get_model_info(model_name)
    print(f"Resolving model: {repo}:{tag}...")
    
    # Use a long timeout for the manifest resolution
    client = httpx.Client(timeout=httpx.Timeout(60.0, read=None), follow_redirects=True)
    
    resp = client.get(f"{OLLAMA_REGISTRY}/{repo}/manifests/{tag}")
    
    if resp.status_code == 401:
        auth_url = f"https://ollama.com/v2/auth/token?scope=repository:{repo}:pull&service=registry.ollama.ai"
        token_resp = client.get(auth_url)
        if token_resp.status_code == 200:
            token = token_resp.json().get("token")
            resp = client.get(f"{OLLAMA_REGISTRY}/{repo}/manifests/{tag}", headers={"Authorization": f"Bearer {token}"})

    if resp.status_code != 200:
        print(f"Error: Model not found ({resp.status_code})")
        return
    
    manifest = resp.json()
    layers = manifest.get("layers", [])
    config = manifest.get("config", {})
    
    blobs_dir = os.path.join(MODELS_DIR, "blobs")
    manifest_dir = os.path.dirname(manifest_path)
    
    os.makedirs(blobs_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)
    
    all_layers = [config] + layers
    auth_header = resp.request.headers.get("Authorization")
    headers = {"Authorization": auth_header} if auth_header else {}

    for layer in all_layers:
        digest = layer.get("digest")
        if not digest: continue
        
        blob_path = os.path.join(blobs_dir, digest.replace(":", "-"))
        expected_size = layer.get("size", 0)
        
        current_size = 0
        if os.path.exists(blob_path):
            current_size = os.path.getsize(blob_path)
            if current_size == expected_size:
                print(f"Layer {digest[:12]} already complete.")
                continue
            elif current_size > expected_size:
                print(f"Layer {digest[:12]} is larger than expected. Restarting...")
                os.remove(blob_path)
                current_size = 0
            else:
                print(f"Resuming layer {digest[:12]} from {current_size // 1024 // 1024} MB...")

        blob_url = f"{OLLAMA_REGISTRY}/{repo}/blobs/{digest}"
        
        # Add Range header for resume if partial file exists
        request_headers = headers.copy()
        if current_size > 0:
            request_headers["Range"] = f"bytes={current_size}-"

        try:
            with client.stream("GET", blob_url, headers=request_headers) as r:
                if r.status_code == 416: # Range not satisfiable (file already complete?)
                    print(f"Layer {digest[:12]} verified.")
                    continue
                
                # If we asked for a range but got 200 instead of 206, the server doesn't support resume
                mode = "ab" if r.status_code == 206 else "wb"
                if r.status_code == 200 and current_size > 0:
                    print("Server does not support resume. Restarting download...")
                    current_size = 0
                
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0)) + current_size
                
                if tqdm:
                    with open(blob_path, mode) as f, tqdm(total=total, initial=current_size, unit_divisor=1024, unit="B", unit_scale=True, desc=f"Pulling {digest[:12]}") as bar:
                        for chunk in r.iter_bytes():
                            f.write(chunk)
                            bar.update(len(chunk))
                else:
                    downloaded = current_size
                    with open(blob_path, mode) as f:
                        for chunk in r.iter_bytes():
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total > 0:
                                sys.stdout.write(f"\rProgress {digest[:12]}: {downloaded/total*100:.1f}%")
                                sys.stdout.flush()
                    print()
        except httpx.ReadTimeout:
            print(f"\nTimeout reached while pulling {digest[:12]}. Please run the command again to resume.")
            return
        except Exception as e:
            print(f"\nError pulling layer: {e}")
            return

    # ONLY write manifest if we finished all layers
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSuccessfully pulled {model_name}")

def remove_model(model_name):
    repo, tag, manifest_path = get_model_info(model_name)
    if not os.path.exists(manifest_path):
        print(f"Error: Model {model_name} not found locally.")
        return
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    layers = manifest.get("layers", [])
    config = manifest.get("config", {})
    all_layers = [config] + layers
    
    print(f"Removing model {model_name}...")
    if os.path.exists(manifest_path):
        os.remove(manifest_path)
        print(f"Deleted manifest: {manifest_path}")
    
    blobs_dir = os.path.join(MODELS_DIR, "blobs")
    for layer in all_layers:
        digest = layer.get("digest")
        if not digest: continue
        blob_path = os.path.join(blobs_dir, digest.replace(":", "-"))
        if os.path.exists(blob_path):
            try:
                os.remove(blob_path)
                print(f"Deleted blob: {digest[:12]}")
            except Exception as e:
                print(f"Could not delete blob {digest[:12]}: {e}")
    print(f"Successfully removed {model_name}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python3 alpaca-puller.py pull tinyllama")
        print("  python3 alpaca-puller.py remove tinyllama")
    else:
        cmd = sys.argv[1].lower()
        model = sys.argv[2]
        if cmd == "pull":
            pull_model(model)
        elif cmd == "remove":
            remove_model(model)
