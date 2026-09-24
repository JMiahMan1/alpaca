#!/usr/bin/env python3
import json
import subprocess
import time

import httpx


def main():
    cfg = {
        "model_path": "/router-models/Qwen-Image-2.1-GGUF--qwen_image_2.1-Q4_K--latest.gguf",
        "vae_path": "/router-models/companions/qwen_image_2.1_vae_bf16.safetensors",
        "clip_l_path": "",
        "t5xxl_path": "",
        "llm_path": "/router-models/companions/Qwen3VL-8B-Instruct-Q4_K_M.gguf",
        "model_family": "qwen-image",
        "extra_args": "--offload-to-cpu --max-vram 7 --backend te=cpu,vae=cpu --vae-tiling --vae-tile-size 128x128 --llm_vision /router-models/companions/mmproj-Qwen3VL-8B-Instruct-F16.gguf",
        "gpu_layers": 40,
        "threads": 6,
        "cache_mode": "easycache",
    }

    active_json_path = ".alpaca-router/sd_active_model.json"
    with open(active_json_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print("✅ Wrote .alpaca-router/sd_active_model.json")

    cmd = [
        "docker",
        "exec",
        "-d",
        "sd-server",
        "/sd.cpp/bin/sd-server",
        "--diffusion-model",
        "/router-models/Qwen-Image-2.1-GGUF--qwen_image_2.1-Q4_K--latest.gguf",
        "--vae",
        "/router-models/companions/qwen_image_2.1_vae_bf16.safetensors",
        "--llm",
        "/router-models/companions/Qwen3VL-8B-Instruct-Q4_K_M.gguf",
        "--lora-model-dir",
        "/router-models/companions/lora",
        "--listen-ip",
        "0.0.0.0",
        "--listen-port",
        "8081",
        "--qwen-image-layers",
        "40",
        "--threads",
        "6",
        "--offload-to-cpu",
        "--max-vram",
        "7",
        "--backend",
        "te=cpu,vae=cpu",
        "--vae-tiling",
        "--vae-tile-size",
        "128x128",
        "--llm_vision",
        "/router-models/companions/mmproj-Qwen3VL-8B-Instruct-F16.gguf",
    ]
    subprocess.run(cmd, check=True)
    print("🚀 Launched sd-server inside container!")

    for i in range(30):
        time.sleep(1)
        try:
            r = httpx.get("http://localhost:8081/health", timeout=2.0)
            if r.status_code == 200:
                print(f"✅ sd-server is healthy on port 8081! (took {i + 1}s)")
                return
        except Exception:
            pass
    print("⚠️ Timed out waiting for health check")


if __name__ == "__main__":
    main()
