import os
import json
import httpx
import asyncio
import re
import logging
import docker
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager

# Precise logging setup
logger = logging.getLogger("alpaca-proxy")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

client_httpx = None
current_model = None

# Configuration
LLAMA_SERVER_URL = "http://llama-server:8080"
OLLAMA_BASE = "/models"
COMPOSE_FILE = "/config/docker-compose.yml"

def get_model_info(model_name):
    if ":" not in model_name: model_name += ":latest"
    parts = model_name.split(":")
    name, tag = parts[0], parts[1]
    manifest_path = f"{OLLAMA_BASE}/manifests/registry.ollama.ai/library/{name}/{tag}"
    if not os.path.exists(manifest_path):
        manifest_path = f"{OLLAMA_BASE}/manifests/registry.ollama.ai/{name}/{tag}"
    
    if not os.path.exists(manifest_path): return None
    
    try:
        with open(manifest_path, 'r') as f: manifest = json.load(f)
        size = 0
        digest = ""
        for layer in manifest.get('layers', []):
            if layer.get('mediaType', '').startswith('application/vnd.ollama.image.model'):
                size = layer.get('size', 0)
                digest = layer.get('digest', "")
                break
        return {
            "size": size,
            "digest": digest,
            "details": {"format": "gguf", "family": "llama"}
        }
    except: return None

def detect_current_model():
    try:
        if not os.path.exists(COMPOSE_FILE): return None
        with open(COMPOSE_FILE, 'r') as f: content = f.read()
        match = re.search(r'sha256-([a-f0-9]+)', content)
        if not match: return None
        target_hash = "sha256:" + match.group(1)
        
        manifest_base = f"{OLLAMA_BASE}/manifests/registry.ollama.ai"
        for root, dirs, files in os.walk(manifest_base):
            for file in files:
                if "sha256" not in file:
                    path = os.path.join(root, file)
                    with open(path, 'r') as f:
                        manifest = json.load(f)
                        for layer in manifest.get('layers', []):
                            if layer.get('digest') == target_hash:
                                rel_path = os.path.relpath(path, manifest_base)
                                parts = rel_path.split("/")
                                tag = parts[-1]
                                name = "/".join(parts[:-1])
                                if name.startswith("library/"): name = name[8:]
                                return f"{name}:{tag}"
    except Exception as e:
        logger.warning(f"Could not detect current model: {e}")
    return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client_httpx, current_model
    client_httpx = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=60.0))
    current_model = detect_current_model()
    if current_model:
        logger.info(f"Detected running model: {current_model}")
    yield
    await client_httpx.aclose()

app = FastAPI(lifespan=lifespan)
client_docker = docker.from_env()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Hit: {request.method} {request.url.path}")
    response = await call_next(request)
    return response

async def ensure_model(model_name: str):
    global current_model
    if current_model != model_name:
        # Resolve to the actual tag name if it was passed as just the model name
        if ":" not in model_name: model_name += ":latest"
        
        # Get hash from manifest
        blob_hash = None
        manifest_path = f"{OLLAMA_BASE}/manifests/registry.ollama.ai/library/{model_name.replace(':', '/')}"
        if not os.path.exists(manifest_path):
            manifest_path = f"{OLLAMA_BASE}/manifests/registry.ollama.ai/{model_name.replace(':', '/')}"
        
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f: manifest = json.load(f)
            for layer in manifest.get('layers', []):
                if layer.get('mediaType', '').startswith('application/vnd.ollama.image.model'):
                    blob_hash = layer.get('digest')
                    break
        
        if not blob_hash: raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        logger.info(f"Switching model to {model_name} (blob: {blob_hash})")
        with open(COMPOSE_FILE, 'r') as f: content = f.read()
        new_content = re.sub(r'sha256-[a-f0-9]+', blob_hash.replace(':', '-'), content)
        with open(COMPOSE_FILE, 'w') as f: f.write(new_content)

        logger.info("Restarting llama-server container...")
        client_docker.containers.get("llama-server").restart()
        
        for _ in range(120):
            try:
                resp = await client_httpx.get(f"{LLAMA_SERVER_URL}/health")
                if resp.status_code == 200:
                    current_model = model_name
                    logger.info(f"Model {model_name} ready.")
                    return
            except: pass
            await asyncio.sleep(1)
        raise HTTPException(status_code=504, detail="Engine startup timeout")

@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    model_name = body.get("model")

    async def stream_proxy():
        try:
            task = asyncio.create_task(ensure_model(model_name))
            while not task.done():
                yield " " 
                await asyncio.sleep(2)
            await task
            
            async with client_httpx.stream("POST", f"{LLAMA_SERVER_URL}/v1/chat/completions", json={"messages": body.get("messages"), "stream": True}) as resp:
                async for line in resp.aiter_lines():
                    if not line or "[DONE]" in line: continue
                    if line.startswith("data: "): line = line[6:]
                    try:
                        data = json.loads(line)
                        choice = data["choices"][0]
                        content = choice["delta"].get("content") or choice["message"].get("content") or ""
                        done = choice.get("finish_reason") is not None
                        yield json.dumps({
                            "model": model_name,
                            "created_at": datetime.utcnow().isoformat() + "Z",
                            "message": {"role": "assistant", "content": content},
                            "done": done
                        }) + "\n"
                    except: continue
        except Exception as e:
            logger.error(f"Chat error: {e}")
            yield json.dumps({"error": str(e)}) + "\n"
    return StreamingResponse(stream_proxy(), media_type="application/x-ndjson")

@app.post("/api/generate")
async def generate(request: Request):
    body = await request.json()
    model_name = body.get("model")

    async def stream_proxy():
        try:
            task = asyncio.create_task(ensure_model(model_name))
            while not task.done():
                yield " "
                await asyncio.sleep(2)
            await task
            
            async with client_httpx.stream("POST", f"{LLAMA_SERVER_URL}/completion", json={"prompt": body.get("prompt"), "stream": True}) as resp:
                async for line in resp.aiter_lines():
                    if not line: continue
                    try:
                        data = json.loads(line)
                        yield json.dumps({
                            "model": model_name,
                            "created_at": datetime.utcnow().isoformat() + "Z",
                            "response": data.get("content") or "",
                            "done": data.get("stop", False)
                        }) + "\n"
                    except: continue
        except Exception as e:
            logger.error(f"Generate error: {e}")
            yield json.dumps({"error": str(e)}) + "\n"
    return StreamingResponse(stream_proxy(), media_type="application/x-ndjson")

@app.get("/api/tags")
async def tags():
    models = []
    manifest_base = f"{OLLAMA_BASE}/manifests/registry.ollama.ai"
    if os.path.exists(manifest_base):
        for root, dirs, files in os.walk(manifest_base):
            for file in files:
                if "sha256" not in file:
                    rel_path = os.path.relpath(os.path.join(root, file), manifest_base)
                    parts = rel_path.split("/")
                    if len(parts) >= 2:
                        tag = parts[-1]
                        name = "/".join(parts[:-1])
                        if name.startswith("library/"): name = name[8:]
                        models.append({
                            "name": f"{name}:{tag}",
                            "model": f"{name}:{tag}",
                            "details": {"format": "gguf", "family": "llama"}
                        })
    return {"models": models}

@app.get("/api/ps")
async def ps():
    global current_model
    if not current_model: return {"models": []}
    info = get_model_info(current_model)
    if not info:
        return {"models": [{
            "name": current_model,
            "model": current_model,
            "size": 0,
            "digest": "",
            "details": {"format": "gguf", "family": "llama"},
            "expires_at": "0001-01-01T00:00:00Z"
        }]}
    
    return {"models": [{
        "name": current_model,
        "model": current_model,
        "size": info["size"],
        "digest": info["digest"],
        "details": info["details"],
        "expires_at": "0001-01-01T00:00:00Z",
        "size_vram": info["size"]
    }]}

@app.get("/api/version")
async def version():
    return {"version": "0.3.1"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11434)
