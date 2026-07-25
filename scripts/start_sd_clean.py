#!/usr/bin/env python3
import json
import subprocess
import time

import httpx


def main():
    cfg = {
        "model_path": "/router-models/qwen-image-edit-rapid-aio--q4_k.gguf",
        "vae_path": "/router-models/companions/qwen_image_vae.safetensors",
        "clip_l_path": "",
        "t5xxl_path": "",
        "llm_path": "/router-models/companions/Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf",
        "model_family": "qwen-image",
        "extra_args": "",
        "gpu_layers": 20,
        "threads": 8,
        "cache_mode": "easycache"
    }

    active_json_path = ".alpaca-router/sd_active_model.json"
    with open(active_json_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print("✅ Wrote .alpaca-router/sd_active_model.json")

    cmd = [
        "docker", "exec", "-d", "sd-server",
        "/sd.cpp/bin/sd-server",
        "--diffusion-model", "/router-models/qwen-image-edit-rapid-aio--q4_k.gguf",
        "--vae", "/router-models/companions/qwen_image_vae.safetensors",
        "--llm", "/router-models/companions/Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf",
        "--lora-model-dir", "/router-models/companions/lora",
        "--listen-ip", "0.0.0.0",
        "--listen-port", "8081",
        "--qwen-image-layers", "20",
        "--threads", "8"
    ]
    subprocess.run(cmd, check=True)
    print("🚀 Launched sd-server inside container!")

    for i in range(30):
        time.sleep(1)
        try:
            r = httpx.get("http://localhost:8081/health", timeout=2.0)
            if r.status_code == 200:
                print(f"✅ sd-server is healthy on port 8081! (took {i+1}s)")
                return
        except Exception:
            pass
    print("⚠️ Timed out waiting for health check")

if __name__ == "__main__":
    main()
