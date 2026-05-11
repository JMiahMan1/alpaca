import os
import json
import httpx
import sys

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
    
    manifest_url = f"{OLLAMA_REGISTRY}/{repo}/manifests/{tag}"
    resp = httpx.get(manifest_url, follow_redirects=True)
    
    if resp.status_code == 401:
        auth_url = f"https://ollama.com/v2/auth/token?scope=repository:{repo}:pull&service=registry.ollama.ai"
        token_resp = httpx.get(auth_url)
        if token_resp.status_code == 200:
            token = token_resp.json().get("token")
            resp = httpx.get(manifest_url, headers={"Authorization": f"Bearer {token}"}, follow_redirects=True)

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
    for layer in all_layers:
        digest = layer.get("digest")
        if not digest: continue
        
        blob_path = os.path.join(blobs_dir, digest.replace(":", "-"))
        if os.path.exists(blob_path):
            print(f"Layer {digest[:12]} already exists.")
            continue
            
        print(f"Downloading {digest[:12]} ({layer.get('size', 0) // 1024 // 1024} MB)...")
        blob_url = f"{OLLAMA_REGISTRY}/{repo}/blobs/{digest}"
        
        with httpx.stream("GET", blob_url, follow_redirects=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            if tqdm:
                with open(blob_path, "wb") as f, tqdm(total=total, unit_divisor=1024, unit="B", unit_scale=True) as bar:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
                        bar.update(len(chunk))
            else:
                downloaded = 0
                with open(blob_path, "wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            print(f"\rProgress: {downloaded/total*100:.1f}%", end="")
                print()

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
        print("  python3 standalone-puller.py pull tinyllama")
        print("  python3 standalone-puller.py remove tinyllama")
    else:
        cmd = sys.argv[1].lower()
        model = sys.argv[2]
        if cmd == "pull":
            pull_model(model)
        elif cmd == "remove":
            remove_model(model)
