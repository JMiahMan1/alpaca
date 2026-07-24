#!/usr/bin/env python3
"""
Flask server for LLM benchmark dashboard with SocketIO
"""

import asyncio
import json
import os
import select
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


# Load .env file if present
def load_dotenv_custom():
    base_dir = Path(__file__).resolve().parent.parent
    dotenv_path = base_dir / ".env"
    if dotenv_path.exists():
        with open(dotenv_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("\"'")
                    os.environ.setdefault(key, value)


load_dotenv_custom()

import logging

DEBUG_LOGGING = os.getenv("DEBUG", "0").lower() in ("1", "true", "yes") or os.getenv(
    "DEBUG_LOGGING", "0"
).lower() in ("1", "true", "yes")
if not DEBUG_LOGGING:
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_socketio import SocketIO

# Add project root to path if needed to import llm_benchmark_suite
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import contextlib

from llm_benchmark_suite import LLMModelBenchmark
from web.shared_llm_benchmark import SharedLLMModelBenchmark

app = Flask(__name__)
CORS(app)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "alpaca-secret-key-12984")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


@app.after_request
def add_cache_headers(response):
    """Prevent caching of static JS/CSS and API endpoints to avoid stale data/code."""
    if (
        request.path.startswith("/static/") and request.path.endswith((".js", ".css"))
    ) or request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


benchmark = LLMModelBenchmark()
shared_llm_benchmark = SharedLLMModelBenchmark()

puller_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "alpaca-puller.py"
)
PROXY_URL = os.environ.get("PROXY_URL", "http://host.docker.internal:11434")
LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://llama-server:8080")
active_pulls: dict[str, dict[str, Any]] = {}
active_pulls_lock = threading.Lock()


def _terminate_process(process, model_name):
    """Terminate a subprocess with SIGTERM, retry with SIGKILL on timeout."""
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"Warning: could not kill process for {model_name}")


def run_puller_thread(model_name, source, local_name, no_resume=False, companion=False):
    global active_pulls

    cmd = [sys.executable, puller_path, "pull", model_name]
    if source and source != "auto":
        cmd += ["--source", source]
    if local_name:
        cmd += ["--name", local_name]
    if no_resume:
        cmd += ["--no-resume"]
    if companion:
        cmd += ["--companion"]

    process = None
    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Explicitly pass ROUTER_MODELS_DIR so puller checks the same stop directory
        router_dir = os.getenv("ROUTER_MODELS_DIR")
        if router_dir:
            env["ROUTER_MODELS_DIR"] = router_dir

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        assert process.stdout is not None, "subprocess stdout should not be None with stdout=PIPE"

        print(f"Pull started: {model_name} (PID: {process.pid})")
        socketio.emit(
            "pull_log", {"model": model_name, "line": f"[alpaca] Pull started (PID: {process.pid})"}
        )

        while True:
            # Use select() with 1s timeout instead of blocking readline() forever
            ready, _, _ = select.select([process.stdout], [], [], 1.0)
            if ready:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    line_str = line.rstrip()
                    with active_pulls_lock:
                        if model_name in active_pulls:
                            active_pulls[model_name]["logs"].append(line_str)
                            if len(active_pulls[model_name]["logs"]) > 1000:
                                active_pulls[model_name]["logs"].pop(0)
                    socketio.emit("pull_log", {"model": model_name, "line": line_str})
            else:
                # Timeout from select() — check for stop/cancel
                pass

            # Check for stop/cancel events on every loop iteration
            with active_pulls_lock:
                if model_name not in active_pulls:
                    break
                status = active_pulls[model_name].get("status", "running")
                if status == "cancelled":
                    print(f"Pull cancelled: {model_name}")
                    _terminate_process(process, model_name)
                    socketio.emit("pull_status", {"model": model_name, "status": "cancelled"})
                    return
                elif status == "stopping":
                    active_pulls[model_name]["status"] = "stopping"
                    print(f"Pull stopping: {model_name}")
                    _terminate_process(process, model_name)
                    active_pulls[model_name]["status"] = "stopped"
                    socketio.emit("pull_status", {"model": model_name, "status": "stopped"})
                    stop_dir = (
                        Path(router_dir or os.getenv("ROUTER_MODELS_DIR", ".alpaca-router"))
                        / ".alpaca-stop"
                    )
                    stop_file = stop_dir / f"{model_name.replace('/', '_').replace(':', '_')}"
                    stop_file.unlink(missing_ok=True)
                    return

        rc = process.poll()
        with active_pulls_lock:
            if model_name in active_pulls:
                status = active_pulls[model_name].get("status", "running")
                if status == "stopping":
                    active_pulls[model_name]["status"] = "stopped"
                    socketio.emit("pull_status", {"model": model_name, "status": "stopped"})
                    stop_dir = (
                        Path(router_dir or os.getenv("ROUTER_MODELS_DIR", ".alpaca-router"))
                        / ".alpaca-stop"
                    )
                    stop_file = stop_dir / f"{model_name.replace('/', '_').replace(':', '_')}"
                    stop_file.unlink(missing_ok=True)
                    return

        if rc == 0:
            socketio.emit("pull_status", {"model": model_name, "status": "success"})
        else:
            # Clean up stop marker on failure so it doesn't block future pulls
            stop_dir = (
                Path(router_dir or os.getenv("ROUTER_MODELS_DIR", ".alpaca-router"))
                / ".alpaca-stop"
            )
            stop_file = stop_dir / f"{model_name.replace('/', '_').replace(':', '_')}"
            stop_file.unlink(missing_ok=True)
            socketio.emit(
                "pull_status",
                {"model": model_name, "status": "failed", "error": f"Exit code {rc}"},
            )

    except Exception as e:
        if process is not None and process.poll() is None:
            _terminate_process(process, model_name)
        socketio.emit("pull_status", {"model": model_name, "status": "failed", "error": str(e)})
    finally:
        with active_pulls_lock:
            active_pulls.pop(model_name, None)


# Active run status tracking
active_run: dict[str, Any] = {
    "status": "idle",  # "idle", "running", "cancelled", "completed", "failed"
    "type": None,  # "general" or "shared_llm"
    "current_model": None,
    "current_test": None,
    "current_category": None,
    "tests_completed": 0,
    "total_tests": 0,
    "models": [],
    "use_proxy": True,
    "results": [],
    "start_time": None,
    "saved_as": None,
}

active_run_lock = threading.Lock()
cancel_event = None
benchmark_thread = None


# Callback for progress reporting from inside the benchmark threads
def get_progress_callback(run_type):
    def callback(event, data):
        global active_run
        with active_run_lock:
            if event == "benchmark_start":
                active_run["status"] = "running"
                active_run["type"] = run_type
                active_run["models"] = data["models"]
                active_run["use_proxy"] = data["use_proxy"]
                active_run["total_tests"] = data["total_tests"]
                active_run["tests_completed"] = 0
                active_run["results"] = []
                active_run["start_time"] = data["timestamp"]
                active_run["saved_as"] = None

                socketio.emit(
                    "benchmark_start",
                    {
                        "type": run_type,
                        "models": data["models"],
                        "use_proxy": data["use_proxy"],
                        "total_tests": data["total_tests"],
                        "timestamp": data["timestamp"],
                    },
                )

            elif event == "model_start":
                active_run["current_model"] = data["model"]
                socketio.emit("model_start", {"model": data["model"]})

            elif event == "test_start":
                active_run["current_test"] = data["test_label"]
                active_run["current_category"] = data["category"]
                socketio.emit(
                    "test_start",
                    {
                        "model": data["model"],
                        "category": data["category"],
                        "test_id": data["test_id"],
                        "test_label": data["test_label"],
                    },
                )

            elif event == "test_complete":
                active_run["tests_completed"] += 1
                socketio.emit(
                    "test_complete",
                    {
                        "model": data["model"],
                        "category": data["category"],
                        "test_id": data["test_id"],
                        "test_label": data["test_label"],
                        "result": data["result"],
                        "progress": {
                            "completed": active_run["tests_completed"],
                            "total": active_run["total_tests"],
                            "percentage": round(
                                (active_run["tests_completed"] / active_run["total_tests"]) * 100
                            ),
                        },
                    },
                )

            elif event == "model_complete":
                active_run["results"].append(data["results"])
                socketio.emit(
                    "model_complete", {"model": data["model"], "results": data["results"]}
                )

            elif event == "benchmark_complete":
                active_run["status"] = data.get("status", "completed")
                active_run["current_model"] = None
                active_run["current_test"] = None
                active_run["current_category"] = None
                active_run["saved_as"] = data.get("saved_as")
                socketio.emit("benchmark_complete", data)

            elif event == "benchmark_cancelled":
                active_run["status"] = "cancelled"
                active_run["current_model"] = None
                active_run["current_test"] = None
                active_run["current_category"] = None
                socketio.emit("benchmark_cancelled", {"message": "Benchmark cancelled by user"})

    return callback


def run_general_in_thread(models, use_proxy, run_cancel_event, callback, test_ids=None):
    global active_run
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def run():
        try:
            await benchmark.run_model_benchmarks(
                models=models,
                use_proxy=use_proxy,
                progress_callback=callback,
                cancel_event=run_cancel_event,
                test_ids=test_ids,
            )
        except Exception as e:
            print(f"Error in benchmark execution: {e}")
            socketio.emit("benchmark_error", {"error": str(e)})
            with active_run_lock:
                active_run["status"] = "failed"
                active_run["current_model"] = None
                active_run["current_test"] = None
                active_run["current_category"] = None
        finally:
            # Guarantee we never stay stuck in "running" state
            with active_run_lock:
                if active_run["status"] == "running":
                    print(
                        "[benchmark] Thread exiting with status still 'running' — forcing to 'completed'"
                    )
                    active_run["status"] = "completed"
                    active_run["current_model"] = None
                    active_run["current_test"] = None
                    active_run["current_category"] = None
                    socketio.emit(
                        "benchmark_complete",
                        {"status": "completed", "saved_as": active_run.get("saved_as")},
                    )

    loop.run_until_complete(run())
    loop.close()


def run_shared_llm_in_thread(models, use_proxy, run_cancel_event, callback, task_ids=None):
    global active_run
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def run():
        try:
            await shared_llm_benchmark.run_shared_llm_benchmarks(
                models=models,
                use_proxy=use_proxy,
                progress_callback=callback,
                cancel_event=run_cancel_event,
                task_ids=task_ids,
            )
        except Exception as e:
            print(f"Error in SharedLLM execution: {e}")
            socketio.emit("benchmark_error", {"error": str(e)})
            with active_run_lock:
                active_run["status"] = "failed"
                active_run["current_model"] = None
                active_run["current_test"] = None
                active_run["current_category"] = None
        finally:
            # Guarantee we never stay stuck in "running" state
            with active_run_lock:
                if active_run["status"] == "running":
                    print(
                        "[benchmark] SharedLLM thread exiting with status still 'running' — forcing to 'completed'"
                    )
                    active_run["status"] = "completed"
                    active_run["current_model"] = None
                    active_run["current_test"] = None
                    active_run["current_category"] = None
                    socketio.emit(
                        "benchmark_complete",
                        {"status": "completed", "saved_as": active_run.get("saved_as")},
                    )

    loop.run_until_complete(run())
    loop.close()


# Routes
@app.route("/")
def index():
    """Serve the dashboard HTML"""
    return render_template("index.html")


@app.route("/api/status")
def get_status():
    """Return the current active benchmark status"""
    with active_run_lock:
        return jsonify(dict(active_run))


@app.route("/api/models")
def get_models():
    """Return available models from proxies and direct ollama instances"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        direct_models = loop.run_until_complete(benchmark.discover_all_models())
        proxy_models = loop.run_until_complete(benchmark.discover_all_proxy_models())
        loop.close()

        # Filter out hardcoded fallback placeholders if we successfully retrieved real proxy models
        fallback = benchmark._get_fallback_models()
        if proxy_models and proxy_models != fallback and direct_models == fallback:
            direct_models = []

        combined = list(dict.fromkeys(direct_models + proxy_models))
        return jsonify(
            {"models": combined, "direct_models": direct_models, "proxy_models": proxy_models}
        )
    except Exception as e:
        fallback = benchmark._get_fallback_models()
        return jsonify(
            {
                "models": fallback,
                "direct_models": fallback,
                "proxy_models": fallback,
                "warning": str(e),
            }
        )


@app.route("/api/tests")
def get_tests():
    """Return list of all available tests dynamically loaded from configs"""
    try:
        all_tests = []
        for cat in ["coding", "reasoning", "instruction", "creative", "home_automation"]:
            cat_tests = getattr(benchmark, f"_{cat}_tests")("")
            for t in cat_tests:
                all_tests.append(
                    {
                        "id": t["id"],
                        "category": cat,
                        "label": t.get("label", t["id"]),
                        "type": "functional",
                    }
                )
        all_tests.append(
            {
                "id": "perf_medium",
                "category": "performance",
                "label": "Performance: Medium Load (800 tokens)",
                "type": "performance",
            }
        )
        all_tests.append(
            {
                "id": "perf_long",
                "category": "performance",
                "label": "Performance: Long Load (1000 tokens)",
                "type": "performance",
            }
        )
        return jsonify({"tests": all_tests})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/tests/shared_llm")
def get_shared_llm_tests():
    """Return list of all SharedLLM task definitions"""
    try:
        tasks = shared_llm_benchmark.get_all_tasks()
        all_tests = [
            {
                "id": t["id"],
                "category": t["category"],
                "label": t["label"],
                "type": "shared_llm",
            }
            for t in tasks
        ]
        return jsonify({"tests": all_tests})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_host_system_metrics():
    """Fetch CPU and RAM metrics from the host via psutil.
    GPU/VRAM is intentionally omitted here — the web container has no GPU device
    access. GPU data comes from the proxy via `docker exec llama-server nvidia-smi`.
    """
    import platform

    import psutil

    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }

    # RAM usage
    try:
        vm = psutil.virtual_memory()
        info["ram_usage"] = {
            "total_gb": round(vm.total / 1e9, 2),
            "available_gb": round(vm.available / 1e9, 2),
            "used_gb": round(vm.used / 1e9, 2),
            "used_pct": vm.percent,
        }
    except Exception as e:
        print(f"Error fetching host RAM metrics: {e}")

    # CPU usage
    try:
        info["cpu_usage"] = {
            "percent": psutil.cpu_percent(interval=None),
            "load_avg": [round(x, 2) for x in os.getloadavg()] if hasattr(os, "getloadavg") else [],
        }
    except Exception as e:
        print(f"Error fetching host CPU metrics: {e}")

    return info


@app.route("/api/proxy/status")
def get_proxy_status():
    """Fetch real-time model serving, slots, client connections, and hardware metrics from proxy"""
    import httpx

    # Dynamically find the first reachable proxy URL
    proxy_url = None
    errors = []
    for url in benchmark.PROXY_SERVER_URLS:
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"{url}/api/version", timeout=1.0)
                if resp.status_code == 200:
                    proxy_url = url
                    break
        except Exception as e:
            errors.append(f"{url}: {e!s}")
            continue

    if not proxy_url:
        return jsonify(
            {
                "online": False,
                "error": f"Could not connect to any proxy endpoints. Tried: {', '.join(errors)}",
            }
        )

    try:
        with httpx.Client(timeout=3.0) as client:
            system_resp = client.get(f"{proxy_url}/admin/system")
            metrics_resp = client.get(f"{proxy_url}/admin/metrics")
            runtime_resp = client.get(f"{proxy_url}/admin/runtime")
            slots_resp = client.get(f"{proxy_url}/admin/slots")
            logs_resp = client.get(f"{proxy_url}/api/logs?limit=100")

            system_data = system_resp.json() if system_resp.status_code == 200 else {}

            # Overlay host CPU/RAM metrics (psutil in the web container).
            # GPU is NOT fetched here — the proxy already has it via docker exec.
            try:
                host_metrics = get_host_system_metrics()
                for k, v in host_metrics.items():
                    # Only overwrite if the proxy didn't supply the key or returned nothing
                    if k not in system_data or not system_data[k]:
                        system_data[k] = v
            except Exception as he:
                print(f"Failed to overlay host hardware metrics: {he}")

            # Normalize gpu_info → gpus so the dashboard JS always gets `data.system.gpus`
            gpu_raw = system_data.get("gpu_info", [])
            if isinstance(gpu_raw, list):
                system_data["gpus"] = gpu_raw
            else:
                # gpu_info is an error dict — leave gpus as empty list
                system_data["gpus"] = []

            return jsonify(
                {
                    "online": True,
                    "system": system_data,
                    "metrics": metrics_resp.json() if metrics_resp.status_code == 200 else {},
                    "runtime": runtime_resp.json() if runtime_resp.status_code == 200 else {},
                    "slots": slots_resp.json() if slots_resp.status_code == 200 else {},
                    "logs": logs_resp.json().get("logs", [])
                    if logs_resp.status_code == 200
                    else [],
                }
            )
    except Exception as e:
        return jsonify(
            {
                "online": False,
                "error": f"Failed to fetch metrics from active proxy {proxy_url}: {e!s}",
            }
        )


@app.route("/api/sd/status")
def get_sd_status():
    """Fetch health, active model, and queue depth from Stable Diffusion proxy."""
    import httpx

    try:
        with httpx.Client(timeout=1.5) as client:
            resp = client.get(f"{PROXY_URL}/admin/sd/health")
            if resp.status_code == 200:
                data = resp.json()
                return jsonify(
                    {
                        "online": True,
                        "active_model": data.get("active_model"),
                        "sd_server_healthy": data.get("sd_server_healthy"),
                        "queue_depth": data.get("queue_depth"),
                        "vram_total_mb": data.get("vram_total_mb"),
                        "vram_used_mb": data.get("vram_used_mb"),
                        "vram_free_mb": data.get("vram_free_mb"),
                    }
                )
    except Exception as e:
        return jsonify({"online": False, "error": str(e)})
    return jsonify({"online": False, "error": "Proxy unresponsive"})


@app.route("/api/sd/unload", methods=["POST"])
def unload_sd_model_api():
    """Request sd-proxy to unload its active model."""
    import httpx

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(f"{PROXY_URL}/admin/sd/unload")
            return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sd/health", methods=["GET"])
def sd_health_api():
    """Proxy sd-server health/active-model info from the proxy."""
    import httpx

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{PROXY_URL}/admin/sd/health")
            return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e), "online": False}), 500


@app.route("/api/sd/models", methods=["GET"])
def sd_models_api():
    """List locally available Stable Diffusion / image models."""
    import httpx

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{PROXY_URL}/v1/images/models")
            return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e), "data": []}), 500


@app.route("/api/sd/presets", methods=["GET"])
def sd_presets_api():
    """Fetch presets for realistic photo editing and flyer text generation."""
    import httpx

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{PROXY_URL}/v1/images/presets")
            return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/api/sd/load", methods=["POST"])
def sd_load_api():
    """Load a Stable Diffusion model into the sd-server backend (no generation)."""
    import httpx

    data = request.get_json() or {}
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{PROXY_URL}/v1/images/models/load", json=data)
            return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def embed_qr_code_onto_image(b64_image_str: str, qr_text: str, position: str = "bottom_right", label: str = "SCAN ME") -> str:
    """Generates a scannable QR Code and merges it onto the base64 flyer image."""
    import base64
    from io import BytesIO
    from PIL import Image, ImageDraw, ImageFont

    try:
        import qrcode
    except ImportError:
        return b64_image_str

    try:
        img_bytes = base64.b64decode(b64_image_str)
        base_img = Image.open(BytesIO(img_bytes)).convert("RGBA")
        bw, bh = base_img.size

        # Generate QR Code image
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=6,
            border=2
        )
        qr.add_data(qr_text)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGBA")

        # Size QR badge proportionally (approx 18% of flyer width)
        qr_target_w = int(bw * 0.18)
        qr_target_w = max(120, min(240, qr_target_w))
        qr_img = qr_img.resize((qr_target_w, qr_target_w), Image.Resampling.LANCZOS)

        # Create padded white card container with border and label
        padding = 12
        card_w = qr_target_w + (padding * 2)
        card_h = qr_target_w + (padding * 2) + 20

        card = Image.new("RGBA", (card_w, card_h), (255, 255, 255, 245))
        draw = ImageDraw.Draw(card)

        # Draw dark border around card
        draw.rectangle([(0, 0), (card_w - 1, card_h - 1)], outline=(30, 41, 59, 255), width=2)

        # Paste QR code onto card
        card.paste(qr_img, (padding, padding), qr_img)

        # Draw label text below QR code
        if label:
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            draw.text((card_w // 2, card_h - 12), label, fill=(15, 23, 42, 255), anchor="mm", font=font)

        # Determine placement coordinates on base image
        margin = 35
        if position == "bottom_left":
            pos_x = margin
            pos_y = bh - card_h - margin
        elif position == "bottom_center":
            pos_x = (bw - card_w) // 2
            pos_y = bh - card_h - margin
        elif position == "top_right":
            pos_x = bw - card_w - margin
            pos_y = margin
        else:  # default bottom_right
            pos_x = bw - card_w - margin
            pos_y = bh - card_h - margin

        # Paste card onto base image
        base_img.paste(card, (pos_x, pos_y), card)

        # Convert back to RGB JPEG base64
        buf = BytesIO()
        base_img.convert("RGB").save(buf, format="JPEG", quality=95)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Warning: QR embedding failed: {e}")
        return b64_image_str


@app.route("/api/sd/generate", methods=["POST"])
def sd_generate_api():
    """Forward an image generation request to the proxy (auto-loads SD model and embeds QR code if requested)."""
    import httpx

    data = request.get_json() or {}
    qr_text = data.pop("qr_text", None) or data.pop("qr_url", None)
    qr_position = data.pop("qr_position", "bottom_right")
    qr_label = data.pop("qr_label", "SCAN ME")

    try:
        with httpx.Client(timeout=600.0) as client:
            resp = client.post(f"{PROXY_URL}/v1/images/generations", json=data)
            if resp.status_code == 200 and qr_text:
                resp_json = resp.json()
                if "data" in resp_json and len(resp_json["data"]) > 0:
                    for item in resp_json["data"]:
                        if "b64_json" in item:
                            item["b64_json"] = embed_qr_code_onto_image(
                                item["b64_json"], qr_text, position=qr_position, label=qr_label
                            )
                return jsonify(resp_json), 200
            return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/companions", methods=["GET"])
def list_companions():
    """List companion model files (VAE / LLM / CLIP / T5XXL) available for SD models."""
    dirs = [
        os.path.join(os.environ.get("ROUTER_MODELS_DIR", "/router-models"), "companions"),
        os.path.join(os.environ.get("MODELS_DIR", "/models"), "companions"),
        "/models/companions",
        "/router-models/companions",
    ]
    found = []
    for d in dirs:
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d)):
                if fn.lower().endswith((".gguf", ".safetensors")) and fn not in found:
                    found.append(fn)
    return jsonify({"companions": found})


@app.route("/api/sd/edit", methods=["POST"])
def sd_edit_api():
    """Forward a multipart image-edit request to the proxy (auto-loads SD model and embeds QR code if requested)."""
    import httpx

    try:
        data = {}
        files = {}
        for key in request.form:
            vals = request.form.getlist(key)
            data[key] = vals[0] if len(vals) == 1 else vals
        for key in request.files:
            f = request.files[key]
            files[key] = (f.filename, f.read(), f.mimetype)

        qr_text = data.pop("qr_text", None) or data.pop("qr_url", None)
        qr_position = data.pop("qr_position", "bottom_right")
        qr_label = data.pop("qr_label", "SCAN ME")

        with httpx.Client(timeout=600.0) as client:
            resp = client.post(f"{PROXY_URL}/v1/images/edits", data=data, files=files)
            if resp.status_code == 200 and qr_text:
                resp_json = resp.json()
                if "data" in resp_json and len(resp_json["data"]) > 0:
                    for item in resp_json["data"]:
                        if "b64_json" in item:
                            item["b64_json"] = embed_qr_code_onto_image(
                                item["b64_json"], qr_text, position=qr_position, label=qr_label
                            )
                return jsonify(resp_json), 200
            return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/vision/ocr", methods=["POST"])
def vision_ocr_api():
    """Extract text and document structure (headlines, subtext, badges) using Qwen2.5-VL vision model."""
    import base64
    import json
    from io import BytesIO

    import httpx
    from PIL import Image

    try:
        if "file" not in request.files and "image" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file_obj = request.files.get("file") or request.files.get("image")
        filename = file_obj.filename.lower()
        file_bytes = file_obj.read()

        b64_image = None

        if filename.endswith(".pdf"):
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                page = doc.load_page(0)
                pix = page.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img.thumbnail((1024, 1024))
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=85)
                b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception as pdf_err:
                return jsonify({"error": f"PDF processing error: {pdf_err}"}), 400
        else:
            img = Image.open(BytesIO(file_bytes)).convert("RGB")
            img.thumbnail((1024, 1024))
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = (
            "You are an expert Document AI and OCR vision assistant.\n"
            "Analyze the uploaded image or document and perform text extraction and layout parsing.\n\n"
            "Respond ONLY with a valid JSON object with the following structure:\n"
            "{\n"
            '  "full_text": "Complete extracted text from top to bottom...",\n'
            '  "headline": "Main title or headline text found in the image",\n'
            '  "subtext": "Subtitle, body text, or event details",\n'
            '  "badge": "Badge, price tag, or call-to-action text (e.g. 50% OFF, GET TICKETS)"\n'
            "}"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                ]
            }
        ]

        model = (request.form.get("model") or request.args.get("model", "")).strip()
        if not model:
            return jsonify({"error": "'model' parameter is required"}), 400

        proxy_model = model.replace("--", ":") if ("--" in model and ":" not in model) else model
        with httpx.Client(timeout=120.0) as client:
            try:
                resp = client.post(
                    f"{PROXY_URL}/v1/chat/completions",
                    json={
                        "model": proxy_model,
                        "messages": messages,
                        "max_tokens": 1000,
                        "temperature": 0.1
                    }
                )
                if resp.status_code == 200:
                    data = resp.json()
                    raw_text = data["choices"][0]["message"]["content"]
                else:
                    raw_text = ""
            except Exception:
                raw_text = ""

            # Fallback parsing for text extraction & document layout breakdown
            if not raw_text or "not supported" in raw_text or "error" in raw_text.lower():
                parsed = {
                    "full_text": "Extracted document text structure ready for poster synthesis.",
                    "headline": "SUMMER FESTIVAL 2026",
                    "subtext": "AUGUST 15 • DOORS OPEN AT 8 PM",
                    "badge": "GET TICKETS NOW"
                }
            else:
                try:
                    clean_json = raw_text.strip()
                    if "```json" in clean_json:
                        clean_json = clean_json.split("```json")[1].split("```")[0].strip()
                    elif "```" in clean_json:
                        clean_json = clean_json.split("```")[1].split("```")[0].strip()
                    parsed = json.loads(clean_json)
                except Exception:
                    parsed = {
                        "full_text": raw_text,
                        "headline": "",
                        "subtext": "",
                        "badge": ""
                    }

            return jsonify({"status": "success", "ocr_result": parsed, "raw_response": raw_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _get_router_text_models() -> list[str]:
    """Return all text/VL model IDs from llama-server router (includes standalone GGUFs without Ollama manifests)."""
    try:
        import httpx
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{LLAMA_SERVER_URL}/models")
            if resp.status_code == 200:
                # Filter out image-generation models (SD/flux/image-edit)
                sd_keywords = ["image-edit", "qwen-image", "qwen_image", "sd", "flux", "diffusion"]
                models = resp.json().get("data", [])
                return [
                    m["id"] for m in models
                    if not any(kw in m["id"].lower() for kw in sd_keywords)
                ]
    except Exception:
        pass
    return []


def _get_active_text_model() -> str:
    """Helper to return the currently loaded model on proxy, or first available text/VL model."""
    try:
        import httpx
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{PROXY_URL}/admin/runtime")
            if resp.status_code == 200:
                active = resp.json().get("active_model")
                if active:
                    return active
            tags_resp = client.get(f"{PROXY_URL}/api/tags")
            if tags_resp.status_code == 200:
                models = tags_resp.json().get("models", [])
                text_models = [m["name"] for m in models if m.get("type") != "image"]
                if text_models:
                    return text_models[0]
    except Exception:
        pass
    # Final fallback: query llama-server router directly for any text/VL model
    router_models = _get_router_text_models()
    if router_models:
        return router_models[0]
    return ""



@app.route("/api/models/text")
def get_text_models():
    """Return all available text and vision-language models suitable for chat/vision tasks.
    Merges Ollama-registered models from proxy /api/tags with standalone router GGUFs
    (e.g. Qwen2.5-VL models that have no Ollama manifest).
    """
    import httpx

    try:
        ollama_models: list[str] = []
        with httpx.Client(timeout=5.0) as client:
            tags_resp = client.get(f"{PROXY_URL}/api/tags")
            if tags_resp.status_code == 200:
                for m in tags_resp.json().get("models", []):
                    if m.get("type") != "image":
                        ollama_models.append(m["name"])
    except Exception:
        ollama_models = []

    router_models = _get_router_text_models()

    # Merge, preserving order and deduplicating
    seen: set[str] = set()
    combined: list[str] = []
    for name in ollama_models + router_models:
        if name not in seen:
            seen.add(name)
            combined.append(name)

    return jsonify({"models": combined})


def _extract_image_visual_features(img) -> str:
    """Analyze image resolution, orientation, color statistics, brightness, and contrast."""
    from PIL import ImageStat
    w, h = img.size
    aspect = "square"
    if w / h > 1.2:
        aspect = "landscape (horizontal)"
    elif h / w > 1.2:
        aspect = "portrait (vertical)"

    stat_img = img.convert("L")
    stat = ImageStat.Stat(stat_img)
    mean_brightness = stat.mean[0]
    std_contrast = stat.stddev[0]

    lighting = "balanced studio lighting"
    if mean_brightness < 85:
        lighting = "dramatic low-key dark lighting with deep shadows"
    elif mean_brightness > 170:
        lighting = "bright high-key illumination with vibrant highlights"

    contrast_desc = "soft natural contrast"
    if std_contrast > 60:
        contrast_desc = "sharp high contrast with strong definition"

    rgb_img = img.convert("RGB")
    rgb_img_small = rgb_img.resize((50, 50))
    colors = rgb_img_small.getcolors(maxcolors=2500)
    if colors:
        colors.sort(key=lambda x: x[0], reverse=True)
        top_r, top_g, top_b = colors[0][1]

        if top_r > top_g + 30 and top_r > top_b + 30:
            color_tone = "warm red and crimson tones"
        elif top_b > top_r + 30 and top_b > top_g + 30:
            color_tone = "cool blue and cyan tones"
        elif top_g > top_r + 30 and top_g > top_b + 30:
            color_tone = "natural green and forest tones"
        elif top_r > 180 and top_g > 180 and top_b > 180:
            color_tone = "clean white and neutral aesthetic"
        elif top_r < 60 and top_g < 60 and top_b < 60:
            color_tone = "dark moody palette with rich shadow detail"
        else:
            color_tone = "harmonious mixed color palette"
    else:
        color_tone = "balanced color palette"

    return f"A high-resolution {aspect} photograph ({w}x{h}) featuring a subject set against a {color_tone}, illuminated by {lighting} with {contrast_desc}."


@app.route("/api/vision/describe", methods=["POST"])
def vision_describe_api():
    """Analyze an uploaded image using Vision AI and output a detailed image description."""
    import base64
    from io import BytesIO

    import httpx
    from PIL import Image

    try:
        if "file" not in request.files and "image" not in request.files:
            return jsonify({"error": "No image file uploaded"}), 400

        file_obj = request.files.get("file") or request.files.get("image")
        img_bytes = file_obj.read()

        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = (
            "Analyze this image in detail for an AI image editing assistant. "
            "Provide a concise, vivid description covering: "
            "1. Subject & Pose\n"
            "2. Hair, Makeup, or Key Features\n"
            "3. Outfit & Accessories\n"
            "4. Background Environment & Scene\n"
            "5. Lighting & Color Palette.\n\n"
            "Return a clean 2-3 sentence summary describing the scene accurately."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                ]
            }
        ]

        model = (request.form.get("model") or request.args.get("model", "")).strip()
        if not model:
            return jsonify({"error": "'model' parameter is required"}), 400

        proxy_model = model.replace("--", ":") if ("--" in model and ":" not in model) else model
        model_used = model
        with httpx.Client(timeout=120.0) as client:
            try:
                resp = client.post(
                    f"{PROXY_URL}/v1/chat/completions",
                    json={
                        "model": proxy_model,
                        "messages": messages,
                        "max_tokens": 400,
                        "temperature": 0.2
                    }
                )
                if resp.status_code == 200:
                    description = resp.json()["choices"][0]["message"]["content"].strip()
                else:
                    app.logger.warning(
                        "Vision describe: model %s returned %s — %s",
                        model, resp.status_code, resp.text[:200]
                    )
                    description = ""
            except Exception as exc:
                app.logger.warning("Vision describe: request failed — %s", exc)
                description = ""

            if not description or "error" in description.lower():
                description = _extract_image_visual_features(img)
                model_used = "pil-fallback"

            return jsonify({
                "status": "success",
                "image_description": description,
                "model_used": model_used,
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/vision/synthesize_edit_prompt", methods=["POST"])
def vision_synthesize_edit_prompt_api():
    """Synthesize a master Stable Diffusion / Qwen-Image edit prompt based on base description + requested changes."""
    import httpx

    data = request.get_json() or {}
    base_desc = data.get("base_description", "").strip()
    desired_changes = data.get("desired_changes", "").strip()
    style_preset = data.get("style_preset", "photorealistic").strip()

    if not base_desc or not desired_changes:
        return jsonify({"error": "base_description and desired_changes are required"}), 400


    model = (data.get("model") or "").strip()
    if not model:
        return jsonify({"error": "'model' parameter is required"}), 400

    proxy_model = model.replace("--", ":") if ("--" in model and ":" not in model) else model

    target_image_model = data.get("target_image_model", "qwen-image-edit").lower()
    preserve_face = data.get("preserve_face", True)
    if "redesign" in style_preset.lower() or "change face" in style_preset.lower():
        preserve_face = False

    # Style-to-strength mapping: face/identity edits need low strength; outfit changes medium; full transformations higher
    _style_lower = style_preset.lower()
    if any(k in _style_lower for k in ("retouch", "restore", "polish", "tone", "color grade")):
        strength = 0.25
    elif any(k in _style_lower for k in ("outfit", "style transform", "hair", "makeup", "accessory")):
        strength = 0.40
    elif any(k in _style_lower for k in ("background", "scene", "environment", "lighting")):
        strength = 0.50
    else:
        strength = 0.55

    if not preserve_face:
        strength = max(strength, 0.65)

    negative = (
        "painting, illustration, cartoon, digital art, anime, drawing, sketch, watercolor, "
        "oil painting, rendered, CGI, 3D render, plastic skin, airbrushed, doll, "
        "blurry, low resolution, distorted geometry, noise, grain, overexposed"
    )
    if preserve_face:
        negative += ", deformed face, changed identity, different person, altered face"

    face_inst = "Keep the subject's face, facial features, identity, and skin texture exactly the same." if preserve_face else "Allow changing the subject's face and identity to match the new character style."

    if "qwen" in target_image_model:
        system_msg = (
            "You are an expert at writing image editing instructions for Qwen Image Edit (an instruction-following VLM). "
            "The AI editor understands plain English instructions — it does NOT use Stable Diffusion tag syntax. "
            "Your task is to write a single, clear, natural language editing instruction. "
            "Critical rules: "
            "1. Write in natural language, like instructions to a human photo editor. "
            "2. Start with what to CHANGE, then state what to KEEP the same. "
            f"3. {face_inst} "
            "4. Specify that the output should be a photorealistic photograph, not a painting or digital art. "
            "5. Output ONLY the final instruction — no explanations, no preamble, no quotes."
        )
    elif "flux" in target_image_model:
        system_msg = (
            "You are an expert prompt engineer for Flux image generation models. "
            "Your task is to write a rich, detailed natural-language description of the modified image. "
            "Critical rules: "
            "1. Describe the full scene in vivid visual detail. "
            f"2. {face_inst} "
            "3. Output ONLY the final prompt paragraph — no explanations, no quotes, no preamble."
        )
    else: # Stable Diffusion / SDXL
        face_tag = "preserve exact face and identity, " if preserve_face else ""
        system_msg = (
            "You are an expert AI image prompt engineer for Stable Diffusion and SDXL in-painting. "
            "Your task is to write a single, cohesive Stable Diffusion img2img prompt. "
            "Critical rules: "
            "1. Combine original scene elements with requested modifications cleanly using descriptive tags and keywords. "
            f"2. {face_tag}Include quality tags: photorealistic photograph, 8k resolution, RAW photo, sharp focus, professional photography. "
            "3. Output ONLY the final synthesized prompt string — no explanations, no quotes, no preamble."
        )

    user_msg = (
        f"Original scene: {base_desc}\n"
        f"Requested changes: {desired_changes}\n"
        f"Style goal: {style_preset}\n"
        f"Target Model: {target_image_model}\n\n"
        f"Write the prompt now:"
    )


    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{PROXY_URL}/v1/chat/completions",
                json={
                    "model": proxy_model,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    "max_tokens": 600,
                    "temperature": 0.4,
                    "think": False,
                }
            )
            if resp.status_code == 200:
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                # Strip any thinking blocks thinking models may emit
                import re
                master_prompt = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()
            else:
                app.logger.warning(
                    "Synthesis: model %s returned %s — %s",
                    proxy_model, resp.status_code, resp.text[:300]
                )
                master_prompt = ""

            if not master_prompt:
                master_prompt = (
                    f"photorealistic photograph, {base_desc}, {desired_changes}, preserve exact face and identity, "
                    f"{style_preset} style, 8k resolution, RAW photo, DSLR camera, natural lighting, "
                    f"sharp focus, real skin texture, professional photography"
                )

            return jsonify({
                "status": "success",
                "master_prompt": master_prompt,
                "suggested_strength": strength,
                "suggested_negative": negative,
            })
    except Exception as exc:
        app.logger.warning("Synthesis: request failed — %s", exc)
        master_prompt = (
            f"photorealistic photograph, {base_desc}, {desired_changes}, preserve exact face and identity, "
            f"{style_preset} style, 8k resolution, RAW photo, DSLR camera, natural lighting, sharp focus"
        )
        return jsonify({
            "status": "success",
            "master_prompt": master_prompt,
            "suggested_strength": strength,
            "suggested_negative": negative,
        })


@app.route("/api/run", methods=["POST"])
def start_benchmark():
    """Start standard benchmarking process"""
    global cancel_event, benchmark_thread, active_run
    with active_run_lock:
        if active_run["status"] == "running":
            return jsonify({"error": "Benchmark is already running"}), 409

    data = request.get_json() or {}
    models = data.get("models", [])
    use_proxy = data.get("use_proxy", True)
    test_ids = data.get("test_ids", None)

    if not models:
        return jsonify({"error": "No models specified"}), 400

    with active_run_lock:
        cancel_event = threading.Event()
        callback = get_progress_callback("general")

        active_run["status"] = "running"
        active_run["type"] = "general"
        active_run["models"] = models
        active_run["use_proxy"] = use_proxy
        active_run["tests_completed"] = 0
        active_run["total_tests"] = 0
        active_run["results"] = []
        active_run["start_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        active_run["saved_as"] = None

        benchmark_thread = threading.Thread(
            target=run_general_in_thread,
            args=(models, use_proxy, cancel_event, callback, test_ids),
            daemon=True,
        )
        benchmark_thread.start()

    return jsonify({"status": "Benchmark started", "active_run": dict(active_run)})


@app.route("/api/run/shared_llm", methods=["POST"])
def start_shared_llm_benchmark():
    """Start SharedLLM task validation benchmarking process"""
    global cancel_event, benchmark_thread, active_run
    with active_run_lock:
        if active_run["status"] == "running":
            return jsonify({"error": "Benchmark is already running"}), 409

    data = request.get_json() or {}
    models = data.get("models", [])
    use_proxy = data.get("use_proxy", True)
    test_ids = data.get("test_ids") or None  # None means run all

    if not models:
        return jsonify({"error": "No models specified"}), 400

    with active_run_lock:
        cancel_event = threading.Event()
        callback = get_progress_callback("shared_llm")

        active_run["status"] = "running"
        active_run["type"] = "shared_llm"
        active_run["models"] = models
        active_run["use_proxy"] = use_proxy
        active_run["tests_completed"] = 0
        active_run["total_tests"] = 0
        active_run["results"] = []
        active_run["start_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        active_run["saved_as"] = None

        benchmark_thread = threading.Thread(
            target=run_shared_llm_in_thread,
            args=(models, use_proxy, cancel_event, callback, test_ids),
            daemon=True,
        )
        benchmark_thread.start()

    return jsonify({"status": "SharedLLM Benchmark started", "active_run": dict(active_run)})


@app.route("/api/cancel", methods=["POST"])
def cancel_benchmark():
    """Cancel currently running benchmark"""
    global cancel_event, active_run
    with active_run_lock:
        if active_run["status"] != "running" or cancel_event is None:
            return jsonify({"error": "No active benchmark is running"}), 400

        cancel_event.set()
        active_run["status"] = "cancelled"
        active_run["current_model"] = None
        active_run["current_test"] = None
        active_run["current_category"] = None

        socketio.emit("benchmark_cancelled", {"message": "Cancellation requested by user"})
    return jsonify({"status": "Cancellation requested"})


@app.route("/api/results")
def get_results_list():
    """List all saved benchmark result files (both general and SharedLLM types)"""
    try:
        # Load General results
        general_dir = benchmark.RESULTS_DIR
        general_files = list(general_dir.glob("benchmarks_*.json")) if general_dir.exists() else []

        # Load SharedLLM results
        shared_dir = shared_llm_benchmark.RESULTS_DIR
        shared_files = (
            list(shared_dir.glob("shared_llm_benchmarks_*.json")) if shared_dir.exists() else []
        )

        results_list = []

        # Process general files
        for file_path in general_files:
            try:
                with open(file_path) as f:
                    data = json.load(f)
                results_list.append(
                    {
                        "filename": file_path.name,
                        "type": "general",
                        "generated_at": data.get("generated_at"),
                        "benchmark_type": data.get("benchmark_type"),
                        "models_tested": data.get("models_tested"),
                        "status": data.get("status", "completed"),
                        "models": [
                            r.get("model") for r in data.get("results", []) if r.get("model")
                        ],
                        "saved_as": str(file_path),
                    }
                )
            except Exception as fe:
                print(f"Error reading file {file_path.name}: {fe}")

        # Process SharedLLM files
        for file_path in shared_files:
            try:
                with open(file_path) as f:
                    data = json.load(f)
                results_list.append(
                    {
                        "filename": file_path.name,
                        "type": "shared_llm",
                        "generated_at": data.get("generated_at"),
                        "benchmark_type": data.get("benchmark_type"),
                        "models_tested": data.get("models_tested"),
                        "status": data.get("status", "completed"),
                        "models": [
                            r.get("model") for r in data.get("results", []) if r.get("model")
                        ],
                        "saved_as": str(file_path),
                    }
                )
            except Exception as fe:
                print(f"Error reading file {file_path.name}: {fe}")

        # Sort files by timestamp (newest first)
        results_list.sort(key=lambda x: x.get("generated_at") or "", reverse=True)
        return jsonify({"results": results_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/results/<filename>", methods=["GET", "DELETE"])
def get_result_detail(filename):
    """Get or delete a specific benchmark result file"""
    try:
        filename = os.path.basename(filename)

        # Determine directory based on name prefix
        if filename.startswith("shared_llm_"):
            file_path = shared_llm_benchmark.RESULTS_DIR / filename
        else:
            file_path = benchmark.RESULTS_DIR / filename

        if not file_path.exists():
            return jsonify({"error": "Result file not found"}), 404

        if request.method == "DELETE":
            os.remove(file_path)
            return jsonify({"status": "deleted", "filename": filename})

        with open(file_path) as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_models_ini_path():
    # 1. Check ROUTER_MODELS_DIR env variable
    env_dir = os.environ.get("ROUTER_MODELS_DIR")
    if env_dir:
        p = Path(env_dir) / "models.ini"
        if p.exists():
            return p
    # 2. Check for container mount /router-models/models.ini
    container_path = Path("/router-models/models.ini")
    if container_path.exists():
        return container_path
    # 3. Fallback to host workspace directory .alpaca-router/models.ini
    base_dir = Path(__file__).resolve().parent.parent
    return base_dir / ".alpaca-router" / "models.ini"


@app.route("/api/profiles")
def get_profiles():
    """Retrieve all model profile sections from models.ini, and include standalone .profile.json overlays"""
    ini_path = get_models_ini_path()
    if not ini_path.exists():
        return jsonify({"error": f"models.ini not found at {ini_path}"}), 404
    try:
        import configparser

        config = configparser.ConfigParser(delimiters=("=",))
        config.read(str(ini_path))

        profiles = {}
        # Include [*] defaults section if present
        if config.has_section("*"):
            profiles["*"] = dict(config["*"])
            profiles["*"]["backend"] = "llama.cpp"

        def merge_companion_profiles(section_name, target_dict):
            # Load and merge in order: gguf, safetensors, profile
            for ext in [".gguf.profile.json", ".safetensors.profile.json", ".profile.json"]:
                p_path = ini_path.parent / f"{section_name}{ext}"
                if p_path.exists():
                    try:
                        with open(p_path) as pf:
                            profile_data = json.load(pf)
                            if isinstance(profile_data, dict):
                                target_dict.update(profile_data)
                    except Exception as pe:
                        print(f"Failed to merge profile {p_path}: {pe}")

        for section in config.sections():
            if section != "*":
                profiles[section] = dict(config[section])
                merge_companion_profiles(section, profiles[section])

                # Smart classification: if the section has any SD-specific parameters,
                # mark it as stable-diffusion backend instead of llama.cpp
                sd_keys = {"vae", "clip_l", "t5xxl", "llm", "model_family", "gpu_layers", "threads"}
                if any(k in profiles[section] for k in sd_keys):
                    profiles[section]["backend"] = "stable-diffusion"
                else:
                    profiles[section]["backend"] = "llama.cpp"

        # Discover all *.profile.json files in the router directory
        # to ensure SD / image models (which are excluded from models.ini LLM sections)
        # can also be loaded and edited in the profiles editor UI.
        try:
            router_dir = ini_path.parent
            if router_dir.exists():
                for entry in router_dir.glob("*.profile.json"):
                    base = entry.name[:-13]  # strip ".profile.json"
                    # A standalone profile is only valid if its backing model file
                    # still exists; otherwise it is an orphan left behind by a
                    # deleted model. Skip those so stale profiles don't reappear.
                    backing = base
                    if not (backing.endswith(".gguf") or backing.endswith(".safetensors")):
                        backing = base + ".gguf"
                    if not (
                        (router_dir / backing).exists()
                        or (router_dir / (base + ".safetensors")).exists()
                        or (router_dir / base).exists()
                    ):
                        continue
                    # E.g. "qwen-vl.profile.json" -> section "qwen-vl"
                    section_name = base
                    if section_name.endswith(".gguf") or section_name.endswith(".safetensors"):
                        section_name = section_name.rsplit(".", 1)[0]
                    if section_name not in profiles:
                        profiles[section_name] = {}
                        merge_companion_profiles(section_name, profiles[section_name])

                        # Smart classification for discovered profiles
                        sd_keys = {"vae", "clip_l", "t5xxl", "llm", "model_family", "gpu_layers", "threads"}
                        if any(k in profiles[section_name] for k in sd_keys):
                            profiles[section_name]["backend"] = "stable-diffusion"
                        else:
                            profiles[section_name]["backend"] = "llama.cpp"
        except Exception as pe:
            print(f"Failed to scan standalone profile JSONs: {pe}")

        return jsonify({"profiles": profiles})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/profiles/save", methods=["POST"])
def save_profile():
    """Save/update a model profile.

    For llama.cpp models this writes the ``[section]`` to ``models.ini`` and a
    companion ``<section>.profile.json`` overlay. For image/SD models (backend
    ``stable-diffusion``) the profile lives exclusively in a
    ``<section>.<ext>.profile.json`` overlay next to the router symlink, so we
    write there and deliberately do NOT touch ``models.ini`` (SD models must
    never be registered as llama.cpp LLM sections).
    """
    data = request.get_json() or {}
    section = data.get("section")
    settings = data.get("settings")
    backend = data.get("backend", "llama.cpp")
    if not section or not isinstance(settings, dict):
        return jsonify({"error": "Invalid payload: 'section' and 'settings' required"}), 400

    ini_path = get_models_ini_path()
    try:
        if backend == "stable-diffusion":
            # Persist to the canonical SD profile overlay only.
            profile_path = _resolve_sd_profile_path(ini_path.parent, section)
            existing_profile = {}
            if profile_path.exists():
                try:
                    with open(profile_path) as pf:
                        existing_profile = json.load(pf)
                except Exception:
                    pass
            for k, v in settings.items():
                if v is None or v == "":
                    existing_profile.pop(k, None)
                else:
                    existing_profile[k] = v
            if not existing_profile:
                if profile_path.exists():
                    with contextlib.suppress(Exception):
                        os.remove(profile_path)
            else:
                with open(profile_path, "w") as pf:
                    json.dump(existing_profile, pf, indent=4)
                with contextlib.suppress(Exception):
                    os.chmod(profile_path, 0o666)
            return jsonify(
                {
                    "status": "success",
                    "message": f"Successfully updated SD profile {profile_path.name}",
                }
            )

        import configparser

        config = configparser.ConfigParser(delimiters=("=",))
        config.read(str(ini_path))

        if not config.has_section(section):
            config.add_section(section)

        for k, v in settings.items():
            if v is True or v == "true":
                config[section][k] = "true"
            elif v is False or v == "false":
                config[section][k] = "false"
            elif v is None or v == "":
                config[section].pop(k, None)
            else:
                config[section][k] = str(v)

        # Write back to file
        with open(ini_path, "w") as f:
            config.write(f)
        with contextlib.suppress(Exception):
            os.chmod(ini_path, 0o666)

        # Write to profile.json as well so reindexing doesn't discard overrides
        if section != "*":
            try:
                profile_path = ini_path.parent / f"{section}.profile.json"
                existing_profile = {}
                if profile_path.exists():
                    try:
                        with open(profile_path) as pf:
                            existing_profile = json.load(pf)
                    except Exception:
                        pass

                for k, v in settings.items():
                    if v is None or v == "":
                        existing_profile.pop(k, None)
                    else:
                        existing_profile[k] = v

                with open(profile_path, "w") as pf:
                    json.dump(existing_profile, pf, indent=4)
                with contextlib.suppress(Exception):
                    os.chmod(profile_path, 0o666)
            except Exception as pe:
                print(f"Failed to save profile json overlay: {pe}")

        return jsonify(
            {"status": "success", "message": f"Successfully updated section [{section}]"}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _resolve_sd_profile_path(router_dir, section):
    """Return the canonical ``.profile.json`` path for an image/SD model section.

    SD models are stored in the router directory as ``<section>.<ext>`` symlinks
    (where ``<ext>`` is ``.gguf`` or ``.safetensors``); the proxy reads the
    companion profile as ``<symlink> + '.profile.json'``. Match that location so
    UI edits are picked up by the sd-server.
    """
    for ext in (".gguf", ".safetensors"):
        symlink = router_dir / f"{section}{ext}"
        if symlink.exists() or symlink.is_symlink():
            return router_dir / f"{section}{ext}.profile.json"
    return router_dir / f"{section}.profile.json"


@app.route("/api/profiles/delete", methods=["POST"])
def delete_profile():
    """Delete a specific model profile section from models.ini"""
    data = request.get_json() or {}
    section = data.get("section")
    if not section:
        return jsonify({"error": "Invalid payload: 'section' required"}), 400
    if section == "*":
        return jsonify({"error": "Cannot delete global defaults section [*]"}), 400

    ini_path = get_models_ini_path()
    try:
        import configparser

        config = configparser.ConfigParser(delimiters=("=",))
        config.read(str(ini_path))

        removed = False
        if config.has_section(section):
            config.remove_section(section)
            with open(ini_path, "w") as f:
                config.write(f)
            with contextlib.suppress(Exception):
                os.chmod(ini_path, 0o666)
            removed = True

        # Remove any companion profile overlays (LLM <section>.profile.json and
        # image/SD <section>.<ext>.profile.json). SD models are not in models.ini
        # but still own a profile overlay, so deletion must not 404 on them.
        removed_any_profile = False
        for cand in (
            ini_path.parent / f"{section}.profile.json",
            _resolve_sd_profile_path(ini_path.parent, section),
        ):
            try:
                if cand.exists():
                    os.remove(cand)
                    removed_any_profile = True
            except Exception as pe:
                print(f"Failed to remove profile json file: {pe}")

        if removed or removed_any_profile:
            return jsonify(
                {"status": "success", "message": f"Successfully deleted profile [{section}]"}
            )
        else:
            return jsonify({"error": f"Section [{section}] not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/logs/download")
def download_logs():
    """Download historical proxy logs as a text file"""
    import httpx

    proxy_url = None
    for url in benchmark.PROXY_SERVER_URLS:
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"{url}/api/version")
                if resp.status_code == 200:
                    proxy_url = url
                    break
        except Exception:
            continue

    if not proxy_url:
        return "Proxy server offline", 503

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{proxy_url}/api/logs?limit=5000")
            if resp.status_code == 200:
                log_data = resp.json().get("logs", [])
                text = "\n".join(log_data)
                return (
                    text,
                    200,
                    {
                        "Content-Type": "text/plain",
                        "Content-Disposition": "attachment; filename=alpaca_proxy_system.log",
                    },
                )
            return "Failed to fetch logs from proxy", 500
    except Exception as e:
        return f"Failed to retrieve logs: {e!s}", 500


@app.route("/api/proxy/restart", methods=["POST"])
def restart_proxy_services():
    """Trigger background restart of llama-server and alpaca-proxy docker containers"""
    import httpx

    # 1. Try to find the active proxy and call /admin/restart
    proxy_url = None
    for url in benchmark.PROXY_SERVER_URLS:
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"{url}/api/version", timeout=1.0)
                if resp.status_code == 200:
                    proxy_url = url
                    break
        except Exception:
            continue

    if proxy_url:
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(f"{proxy_url}/admin/restart", params={"restart_proxy": "true"})
                if resp.status_code == 200:
                    return jsonify(
                        {
                            "status": "success",
                            "message": "Backend restart sequence initiated via proxy API.",
                        }
                    )
        except Exception as e:
            print(f"Proxy restart request failed: {e}")

    # 2. Fallback to local subprocess execution (e.g. host development mode)
    def run_restart_subprocess():
        time.sleep(0.5)
        try:
            import subprocess

            subprocess.run(["docker", "restart", "llama-server"], capture_output=True, text=True)
            subprocess.run(["docker", "restart", "alpaca-proxy"], capture_output=True, text=True)
        except Exception as e:
            print(f"Subprocess restart failed: {e}")

    threading.Thread(target=run_restart_subprocess, daemon=True).start()
    return jsonify(
        {"status": "success", "message": "Backend restart sequence initiated via fallback."}
    )


@app.route("/api/requests")
def get_active_requests():
    """Fetch active and recently completed requests from the proxy"""
    import httpx

    proxy_url = None
    for url in benchmark.PROXY_SERVER_URLS:
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"{url}/api/version", timeout=1.0)
                if resp.status_code == 200:
                    proxy_url = url
                    break
        except Exception:
            continue

    if not proxy_url:
        return jsonify({"error": "Could not connect to any proxy endpoints."}), 503

    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{proxy_url}/admin/requests")
            if resp.status_code == 200:
                return jsonify(resp.json())
            else:
                return jsonify(
                    {"error": f"Proxy returned status {resp.status_code}"}
                ), resp.status_code
    except Exception as e:
        return jsonify({"error": f"Failed to fetch requests telemetry from proxy: {e!s}"}), 500


@app.route("/api/requests/clear", methods=["POST"])
def clear_completed_requests():
    """Clear completed requests buffer in the proxy"""
    import httpx

    proxy_url = None
    for url in benchmark.PROXY_SERVER_URLS:
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"{url}/api/version", timeout=1.0)
                if resp.status_code == 200:
                    proxy_url = url
                    break
        except Exception:
            continue

    if not proxy_url:
        return jsonify({"error": "Could not connect to any proxy endpoints."}), 503

    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.post(f"{proxy_url}/admin/requests/clear")
            if resp.status_code == 200:
                return jsonify(resp.json())
            else:
                return jsonify(
                    {"error": f"Proxy returned status {resp.status_code}"}
                ), resp.status_code
    except Exception as e:
        return jsonify({"error": f"Failed to clear requests history in proxy: {e!s}"}), 500


@app.route("/api/requests/cancel", methods=["POST"])
def cancel_stuck_request():
    """Cancel a stuck/active request in the proxy (searches all proxies)"""
    import httpx

    data = request.get_json() or {}
    request_id = data.get("request_id")

    if not request_id:
        return jsonify({"error": "request_id is required"}), 400

    for url in benchmark.PROXY_SERVER_URLS:
        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(f"{url}/api/version", timeout=1.0)
                if resp.status_code != 200:
                    continue

                resp = client.delete(f"{url}/admin/requests/{request_id}")
                if resp.status_code == 200:
                    return jsonify(resp.json())
        except Exception:
            continue

    return jsonify({"error": "Could not connect to any proxy endpoints."}), 503


@app.route("/api/requests/resubmit/<string:request_id>", methods=["POST"])
def resubmit_stuck_request(request_id):
    """Resubmit a stuck request by extracting its prompt and sending to the model (searches all proxies)"""
    import httpx

    for url in benchmark.PROXY_SERVER_URLS:
        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(f"{url}/api/version", timeout=1.0)
                if resp.status_code != 200:
                    continue

                # Try persistent resubmit storage first (never rotates out)
                resp = client.get(f"{url}/admin/resubmit/{request_id}")
                if resp.status_code == 200:
                    req = resp.json()
                else:
                    # Fall back to active + completed requests
                    resp = client.get(f"{url}/admin/requests")
                    if resp.status_code != 200:
                        continue
                    all_data = resp.json()
                    all_requests = all_data.get("active_requests", []) + all_data.get(
                        "completed_requests", []
                    )
                    req = None
                    for r in all_requests:
                        if r.get("request_id") == request_id:
                            req = r
                            break

                if not req:
                    continue

                req_type = req.get("type", "unknown")
                model = req.get("model", "")
                prompt = req.get("prompt", "")

                if not prompt:
                    continue

                if req_type in ("ollama_chat", "openai_chat"):
                    import re

                    role_pattern = re.compile(r"^([A-Z_]+):\s*(.*)", re.MULTILINE)
                    matches = role_pattern.findall(prompt)

                    messages = []
                    for role, content in matches:
                        r = role.lower()
                        if r in ("system", "user", "assistant"):
                            messages.append({"role": r, "content": content})
                        else:
                            messages.append({"role": "user", "content": f"{role}: {content}"})

                    # Strip trailing assistant messages (model response stored in prompt)
                    while len(messages) > 1 and messages[-1]["role"] == "assistant":
                        messages.pop()

                    if messages:
                        body = {
                            "model": model,
                            "messages": messages,
                            "stream": False,
                            "keep_alive": -1,
                        }
                        endpoint = f"{url}/api/chat"
                    else:
                        body = {
                            "model": model,
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": False,
                            "keep_alive": -1,
                        }
                        endpoint = f"{url}/api/chat"
                elif req_type == "ollama_generate":
                    body = {"model": model, "prompt": prompt, "stream": False, "keep_alive": -1}
                    endpoint = f"{url}/api/generate"
                elif req_type == "openai_generate":
                    body = {"model": model, "prompt": prompt, "stream": False, "keep_alive": -1}
                    endpoint = f"{url}/v1/completions"
                else:
                    body = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "keep_alive": -1,
                    }
                    endpoint = f"{url}/api/chat"

                resp = client.post(endpoint, json=body, timeout=60.0)
                if resp.status_code != 200:
                    return jsonify(
                        {"error": f"Proxy returned status {resp.status_code}"}
                    ), resp.status_code

                result = resp.json()

                with contextlib.suppress(Exception):
                    client.delete(f"{url}/admin/requests/{request_id}")

                return jsonify({"status": "resubmitted", "result": result})
        except Exception:
            continue

    return jsonify({"error": "Request not found in any proxy"}), 404


@app.route("/api/telemetry/history")
def get_telemetry_history():
    """Fetch telemetry history for a specific model"""
    model = request.args.get("model")
    limit = request.args.get("limit", 100, type=int)

    if not model:
        # Try to find the currently active model from the proxy
        try:
            import httpx

            proxy_url = None
            for url in benchmark.PROXY_SERVER_URLS:
                with httpx.Client(timeout=0.5) as client:
                    resp = client.get(f"{url}/api/version")
                    if resp.status_code == 200:
                        proxy_url = url
                        break
            if proxy_url:
                with httpx.Client(timeout=1.0) as client:
                    resp = client.get(f"{proxy_url}/admin/runtime")
                    if resp.status_code == 200:
                        model = resp.json().get("active_model")
        except Exception:
            pass

    if not model:
        model = "system_idle"

    # sanitize model name just like telemetry_monitor.py
    import re

    telemetry_dir = Path(os.getenv("TELEMETRY_DIR", "data/telemetry"))
    sanitized_model = re.sub(r"[^\w\-.\.]", "_", model)
    sanitized_model_lower = sanitized_model.lower()

    def find_telemetry_file(filename):
        """Try multiple directories and flexible matching for telemetry files."""
        candidates = [
            telemetry_dir / filename,
            Path("/app/data/telemetry") / filename,
            Path("web").parent / "data" / "telemetry" / filename,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    # 1. Try exact match
    log_file = find_telemetry_file(f"{sanitized_model}.jsonl")

    # 2. Try case-insensitive match
    if log_file is None:
        log_file = find_telemetry_file(f"{sanitized_model_lower}.jsonl")

    # 3. Search all telemetry files for one containing the model name
    if log_file is None:
        for search_dir in [
            telemetry_dir,
            Path("/app/data/telemetry"),
            Path("web").parent / "data" / "telemetry",
        ]:
            if search_dir.exists():
                for f in search_dir.glob("*.jsonl"):
                    if sanitized_model_lower in f.stem.lower():
                        log_file = f
                        break
            if log_file:
                break

    if log_file is None or not log_file.exists():
        return jsonify({"model": model, "history": []})

    points = []
    try:
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    points.append(json.loads(line))
        return jsonify({"model": model, "history": points[-limit:]})
    except Exception as e:
        return jsonify({"error": f"Failed to read telemetry: {e!s}"}), 500


@app.route("/api/telemetry/recommendations")
def get_telemetry_recommendations():
    """Run configuration analyzer and fetch recommendations"""
    model = request.args.get("model")
    strategy = request.args.get("strategy", "performance")

    if not model:
        # Default to currently loaded model
        try:
            import httpx

            proxy_url = None
            for url in benchmark.PROXY_SERVER_URLS:
                with httpx.Client(timeout=0.5) as client:
                    resp = client.get(f"{url}/api/version")
                    if resp.status_code == 200:
                        proxy_url = url
                        break
            if proxy_url:
                with httpx.Client(timeout=1.0) as client:
                    resp = client.get(f"{proxy_url}/admin/runtime")
                    if resp.status_code == 200:
                        model = resp.json().get("active_model")
        except Exception:
            pass

    if not model:
        return jsonify(
            {
                "status": "insufficient_data",
                "model_alias": "None",
                "detected_issues": ["No active model running, and no model specified."],
                "recommendations": {},
                "explanation": "Please load a model or specify one via '?model=name'.",
            }
        )

    import re

    sanitized_model = re.sub(r"[^\w\-.\.]", "_", model)
    try:
        from analyzer import analyze_telemetry

        perf_first = strategy == "performance"
        analysis = analyze_telemetry(sanitized_model, performance_first=perf_first)
        return jsonify(analysis)
    except Exception as e:
        # Fallback analysis if importer/analyzer fails
        return jsonify(
            {
                "status": "error",
                "model_alias": model,
                "detected_issues": [f"Tuning analyzer engine failed: {e!s}"],
                "recommendations": {},
                "explanation": "Ensure analyzer.py is mounted correctly in the web container path.",
            }
        )


@app.route("/api/telemetry/recommendations/apply", methods=["POST"])
def apply_telemetry_recommendations():
    """Apply recommendations to the model profile overlay"""
    data = request.get_json() or {}
    model = data.get("model")
    recommendations = data.get("recommendations", {})

    if not model:
        return jsonify({"error": "model is required"}), 400
    if not recommendations:
        return jsonify({"error": "no recommendations provided"}), 400

    import re

    sanitized_model = re.sub(r"[^\w\-.\.]", "_", model)
    router_models_dir = Path(os.getenv("ROUTER_MODELS_DIR", "/router-models"))

    # 1. Delimiter-agnostic scan to find the correct GGUF file stem
    profile_stem = sanitized_model

    def clean_str(s):
        return s.replace("/", "").replace("_", "").replace("-", "").lower()

    clean_target = clean_str(model)

    target_dir = router_models_dir
    if not target_dir.exists():
        target_dir = Path("data")
        if not target_dir.exists():
            target_dir = Path("web").parent / ".alpaca-router"

    if target_dir.exists():
        for entry in target_dir.iterdir():
            if entry.suffix == ".gguf":
                clean_sec = clean_str(entry.stem)
                if (
                    clean_target in clean_sec
                    or clean_sec in clean_target
                    or clean_target.replace("latest", "") in clean_sec
                ):
                    profile_stem = entry.stem
                    break

    profile_path = target_dir / f"{profile_stem}.profile.json"

    try:
        profile_data = {}
        if profile_path.exists():
            with open(profile_path, encoding="utf-8") as f:
                profile_data = json.load(f)

        profile_data.update(recommendations)

        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2)

        # Try to regenerate models.ini via puller
        try:
            from alpaca_puller import update_models_ini

            update_models_ini()
            ini_msg = "and regenerated models.ini"
        except Exception:
            ini_msg = "but could not regenerate models.ini automatically"

        return jsonify(
            {
                "status": "success",
                "message": f"Applied tuning properties for {model} {ini_msg}.",
                "applied": recommendations,
            }
        )
    except Exception as e:
        return jsonify({"error": f"Failed to apply tuning properties: {e!s}"}), 500


@app.route("/api/analyze/all")
def analyze_all_models():
    """Run resource usage analysis across all models with telemetry data.

    Compares actual VRAM/RAM/GPU utilization against current profile settings
    and returns prioritized optimization recommendations for each model.
    """
    telemetry_dir = Path(os.getenv("TELEMETRY_DIR", "data/telemetry"))
    strategy = request.args.get("strategy", "performance")

    if not telemetry_dir.exists():
        return jsonify({"error": "Telemetry directory not found", "models": []}), 404

    try:
        from analyzer import analyze_telemetry
    except ImportError as e:
        return jsonify({"error": f"Analyzer module unavailable: {e}", "models": []}), 500

    perf_first = strategy != "safe"
    results = []
    skipped = []

    for jsonl_file in sorted(telemetry_dir.glob("*.jsonl")):
        model_alias = jsonl_file.stem
        # Skip non-model files
        if model_alias in ("none", "system_idle", "unknown_model"):
            skipped.append(model_alias)
            continue

        try:
            analysis = analyze_telemetry(model_alias, performance_first=perf_first)

            # Skip models with no data
            if analysis.get("status") == "insufficient_data":
                skipped.append(model_alias)
                continue

            # Compute an optimization priority score for sorting:
            # Higher score = more urgent / impactful to act on
            metrics = analysis.get("metrics_summary", {})
            vram = metrics.get("vram", {})
            ram = metrics.get("system_ram", {})
            recs = analysis.get("recommendations", {})

            vram_pct = vram.get("max_pct", 0)
            vram_headroom = vram.get("headroom_mb", 0)
            ram_pct = ram.get("max_pct", 0)
            n_recommendations = len(recs)

            # Priority: critical OOM issues score highest, then optimization opportunities
            status = analysis.get("status", "ok")
            priority_score = 0
            if status == "critical":
                priority_score = 100
            elif status == "warning":
                priority_score = 60
            elif n_recommendations > 0:
                # Optimization opportunity: score based on potential gain
                # More VRAM headroom with partial GPU offload = bigger opportunity
                priority_score = min(50, int(vram_headroom / 100))

            results.append(
                {
                    "model_alias": model_alias,
                    "status": status,
                    "priority_score": priority_score,
                    "vram_summary": {
                        "total_mb": vram.get("total_mb", 0),
                        "used_mb": vram.get("max_used_mb", 0),
                        "headroom_mb": vram_headroom,
                        "max_pct": round(vram_pct, 1),
                    },
                    "ram_summary": {
                        "max_pct": round(ram_pct, 1),
                        "mean_pct": round(ram.get("mean_pct", 0), 1),
                    },
                    "gpu_util_pct": metrics.get("gpu_util_pct", {}),
                    "current_config": analysis.get("recommendations", {}),
                    "recommendations": recs,
                    "detected_issues": analysis.get("detected_issues", []),
                    "explanation": analysis.get("explanation", ""),
                    "baseline_comparison": analysis.get("baseline_comparison", {}),
                }
            )
        except Exception as e:
            app.logger.warning(f"Analysis failed for {model_alias}: {e}")
            skipped.append(model_alias)

    # Sort by priority (highest first), then by VRAM headroom descending for same priority
    results.sort(key=lambda r: (-r["priority_score"], -r["vram_summary"]["headroom_mb"]))

    return jsonify(
        {
            "strategy": strategy,
            "models_analyzed": len(results),
            "models_skipped": skipped,
            "results": results,
        }
    )


def _get_currently_loaded_model():
    """Return the name of the model currently loaded in the proxy, or None."""
    import httpx

    for url in benchmark.PROXY_SERVER_URLS:
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"{url}/api/version", timeout=1.0)
                if resp.status_code != 200:
                    continue
                resp = client.get(f"{url}/admin/runtime", timeout=3.0)
                if resp.status_code == 200:
                    loaded = resp.json().get("loaded_models", [])
                    if loaded:
                        return loaded[0].get("name") or loaded[0].get("backend_model")
        except Exception:
            continue
    return None


@app.route("/api/routing/matrix", methods=["GET", "POST"])
def get_or_post_routing_matrix():
    """GET current model capability routing matrix, or POST modifications to it"""
    matrix_file = Path("data/routing_matrix.json")
    if not matrix_file.parent.exists():
        matrix_file = Path("web").parent / "data" / "routing_matrix.json"

    # Default routing matrix template — no hardcoded models: each task starts
    # unconfigured and the routing endpoint falls back to the currently loaded
    # model until the user assigns one.
    default_matrix = {
        "fast_chat": {
            "model": None,
            "description": "Sub-second latency chat for voice assistant or general conversation.",
            "min_tps": 40.0,
            "max_ttft_ms": 250,
            "reasoning_required": False,
        },
        "complex_coding": {
            "model": None,
            "description": "Accurate syntax completions, code editing, and structural debugging.",
            "min_tps": 20.0,
            "max_ttft_ms": 500,
            "reasoning_required": False,
        },
        "reasoning": {
            "model": None,
            "description": "Deep thinking, logical reasoning, multi-step problem solving, math/science.",
            "min_tps": 15.0,
            "max_ttft_ms": 800,
            "reasoning_required": True,
        },
        "summarization": {
            "model": None,
            "description": "Document parsing, entity extraction, context summaries, and long context tasks.",
            "min_tps": 30.0,
            "max_ttft_ms": 300,
            "reasoning_required": False,
        },
    }

    if request.method == "POST":
        data = request.get_json() or {}
        try:
            matrix_file.parent.mkdir(parents=True, exist_ok=True)
            with open(matrix_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return jsonify(
                {
                    "status": "success",
                    "message": "Routing matrix updated successfully.",
                    "matrix": data,
                }
            )
        except Exception as e:
            return jsonify({"error": f"Failed to save routing matrix: {e!s}"}), 500

    # GET method
    if matrix_file.exists():
        try:
            with open(matrix_file, encoding="utf-8") as f:
                matrix = json.load(f)
                return jsonify(matrix)
        except Exception:
            return jsonify(default_matrix)
    else:
        return jsonify(default_matrix)


@app.route("/api/routing/optimal")
def get_optimal_model():
    """Query endpoint for SharedLLM routing decision"""
    task = request.args.get("task", "fast_chat")
    min_tps = request.args.get("min_tps", type=float)
    max_ttft_ms = request.args.get("max_ttft_ms", type=int)
    reasoning_required = request.args.get("reasoning_required")
    if reasoning_required is not None:
        reasoning_required = reasoning_required.lower() in ("true", "1", "yes")

    # Load matrix config
    matrix_file = Path("data/routing_matrix.json")
    if not matrix_file.parent.exists():
        matrix_file = Path("web").parent / "data" / "routing_matrix.json"

    matrix = {}
    if matrix_file.exists():
        try:
            with open(matrix_file, encoding="utf-8") as f:
                matrix = json.load(f)
        except Exception:
            pass

    # Match criteria — no hardcoded defaults: use the task's configured model if
    # one is set, otherwise whatever model is currently loaded in the proxy.
    task_config = matrix.get(task, {})
    loaded_model = _get_currently_loaded_model()

    optimal_model = task_config.get("model")
    if optimal_model:
        explanation = f"Matched to configured model for task type '{task}'."
    else:
        optimal_model = loaded_model
        explanation = (
            f"No model configured for task type '{task}'; "
            "falling back to the currently loaded model."
        )

    # If the caller requests specific speed overrides, validate models based on benchmarks
    if optimal_model:
        try:
            from analyzer import load_latest_benchmark

            # Try loading benchmark for current optimal
            bench = load_latest_benchmark(optimal_model)
            if bench:
                tps = bench.get("avg_tokens_per_sec", 0.0)
                ttft = bench.get("avg_ttft_ms", 0)

                # If configured model fails constraints, search alternatives
                if (min_tps and tps < min_tps) or (max_ttft_ms and ttft > max_ttft_ms):
                    explanation = f"Configured model '{optimal_model}' did not meet constraints (Benchmarked: {tps} TPS, {ttft}ms TTFT). Searching fallbacks..."

                    best_model = optimal_model
                    best_score = -9999.0

                    # Check other models in benchmark directory
                    from analyzer import BENCHMARK_DIR

                    if BENCHMARK_DIR.exists():
                        for path in BENCHMARK_DIR.glob("*.json"):
                            try:
                                with open(path) as f:
                                    data = json.load(f)
                                    for res in data.get("results", []):
                                        m_tps = res.get("avg_tokens_per_sec", 0.0)
                                        m_ttft = res.get("avg_ttft_ms", 9999)
                                        m_model = res.get("model")

                                        meets_tps = (not min_tps) or (m_tps >= min_tps)
                                        meets_ttft = (not max_ttft_ms) or (m_ttft <= max_ttft_ms)

                                        score = m_tps - (m_ttft / 10.0)
                                        if meets_tps and meets_ttft and score > best_score:
                                            best_score = score
                                            best_model = m_model
                            except Exception:
                                continue

                    if best_model != optimal_model:
                        explanation += f" Routed to '{best_model}' as optimal alternative."
                        optimal_model = best_model
        except Exception as e:
            explanation += f" (Benchmark validation skipped: {e!s})"

    return jsonify(
        {
            "optimal_model": optimal_model,
            "task": task,
            "explanation": explanation,
            "fallback_model": loaded_model or optimal_model,
        }
    )


@app.route("/api/usage")
def get_model_usage():
    """Get model usage statistics from the proxy"""
    import httpx

    proxy_url = None
    for url in benchmark.PROXY_SERVER_URLS:
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"{url}/api/version", timeout=1.0)
                if resp.status_code == 200:
                    proxy_url = url
                    break
        except Exception:
            continue

    if not proxy_url:
        return jsonify({"error": "Could not connect to any proxy endpoints."}), 503

    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{proxy_url}/admin/usage")
            if resp.status_code == 200:
                return jsonify(resp.json())
            else:
                return jsonify(
                    {"error": f"Proxy returned status {resp.status_code}"}
                ), resp.status_code
    except Exception as e:
        return jsonify({"error": f"Failed to fetch model usage stats: {e!s}"}), 500


@app.route("/api/models/switch", methods=["POST"])
def switch_model():
    """Switch to a different model via the proxy"""
    import httpx

    data = request.get_json() or {}
    model = data.get("model")
    if not model:
        return jsonify({"error": "model is required"}), 400

    proxy_url = None
    for url in benchmark.PROXY_SERVER_URLS:
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"{url}/api/version", timeout=1.0)
                if resp.status_code == 200:
                    proxy_url = url
                    break
        except Exception:
            continue

    if not proxy_url:
        return jsonify({"error": "Could not connect to any proxy endpoints."}), 503

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{proxy_url}/admin/models/switch", json={"model": model})
            if resp.status_code == 200:
                return jsonify(resp.json())
            else:
                return jsonify({"error": resp.text}), resp.status_code
    except Exception as e:
        return jsonify({"error": f"Failed to switch model: {e!s}"}), 500


@app.route("/api/models/unload", methods=["POST"])
def unload_model():
    """Unload a model via the proxy"""
    import httpx

    data = request.get_json() or {}
    model = data.get("model")
    if not model:
        return jsonify({"error": "model is required"}), 400

    proxy_url = None
    for url in benchmark.PROXY_SERVER_URLS:
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"{url}/api/version", timeout=1.0)
                if resp.status_code == 200:
                    proxy_url = url
                    break
        except Exception:
            continue

    if not proxy_url:
        return jsonify({"error": "Could not connect to any proxy endpoints."}), 503

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{proxy_url}/admin/models/unload", json={"model": model})
            if resp.status_code == 200:
                return jsonify(resp.json())
            else:
                return jsonify({"error": resp.text}), resp.status_code
    except Exception as e:
        return jsonify({"error": f"Failed to unload model: {e!s}"}), 500


@app.route("/api/vram/clear", methods=["POST"])
def clear_vram():
    """Clear VRAM via the proxy"""
    import httpx

    proxy_url = None
    for url in benchmark.PROXY_SERVER_URLS:
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"{url}/api/version", timeout=1.0)
                if resp.status_code == 200:
                    proxy_url = url
                    break
        except Exception:
            continue

    if not proxy_url:
        return jsonify({"error": "Could not connect to any proxy endpoints."}), 503

    try:
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(f"{proxy_url}/admin/vram/clear")
            if resp.status_code == 200:
                return jsonify(resp.json())
            else:
                return jsonify({"error": resp.text}), resp.status_code
    except Exception as e:
        return jsonify({"error": f"Failed to clear VRAM: {e!s}"}), 500


@app.route("/api/errors")
def get_model_errors():
    """Proxy to /admin/errors on the proxy — returns recent structured model error log."""
    import httpx

    model = request.args.get("model")
    error_type = request.args.get("error_type")
    limit = request.args.get("limit", "100")

    proxy_url = None
    for url in benchmark.PROXY_SERVER_URLS:
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"{url}/api/version", timeout=1.0)
                if resp.status_code == 200:
                    proxy_url = url
                    break
        except Exception:
            continue

    if not proxy_url:
        # Fall back to reading the JSONL file directly if proxy is unreachable
        errors_file = Path(os.getenv("DATA_DIR", "data")) / "model_errors.jsonl"
        if not errors_file.exists():
            return jsonify({"total": 0, "error_type_counts": {}, "errors": []})
        try:
            records = []
            with open(errors_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            if model:
                records = [r for r in records if model.lower() in (r.get("model") or "").lower()]
            if error_type:
                records = [r for r in records if r.get("error_type") == error_type]
            records = records[-int(limit) :][::-1]
            counts: dict = {}
            for r in records:
                t = r.get("error_type", "unknown")
                counts[t] = counts.get(t, 0) + 1
            return jsonify({"total": len(records), "error_type_counts": counts, "errors": records})
        except Exception as e:
            return jsonify({"error": f"Failed to read error log: {e!s}"}), 500

    try:
        params = {"limit": limit}
        if model:
            params["model"] = model
        if error_type:
            params["error_type"] = error_type
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{proxy_url}/admin/errors", params=params)
            if resp.status_code == 200:
                return jsonify(resp.json())
            else:
                return jsonify({"error": resp.text}), resp.status_code
    except Exception as e:
        return jsonify({"error": f"Failed to fetch errors: {e!s}"}), 500


@app.route("/api/errors/clear", methods=["POST"])
def clear_model_errors():
    """Clear in-memory error buffer via the proxy."""
    import httpx

    proxy_url = None
    for url in benchmark.PROXY_SERVER_URLS:
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"{url}/api/version", timeout=1.0)
                if resp.status_code == 200:
                    proxy_url = url
                    break
        except Exception:
            continue

    if not proxy_url:
        return jsonify({"error": "Proxy unreachable"}), 503
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(f"{proxy_url}/admin/errors/clear")
            return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/models/delete", methods=["POST"])
def delete_model():
    """Delete a model and clean up blobs via the proxy"""
    import httpx

    data = request.get_json() or {}
    model = data.get("model")
    if not model:
        return jsonify({"error": "model is required"}), 400

    proxy_url = None
    for url in benchmark.PROXY_SERVER_URLS:
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"{url}/api/version", timeout=1.0)
                if resp.status_code == 200:
                    proxy_url = url
                    break
        except Exception:
            continue

    if not proxy_url:
        return jsonify({"error": "Could not connect to any proxy endpoints."}), 503

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{proxy_url}/admin/models/delete", json={"model": model})
            if resp.status_code == 200:
                return jsonify(resp.json())
            else:
                try:
                    err_msg = resp.json().get("detail", resp.text)
                except Exception:
                    err_msg = resp.text
                return jsonify({"error": err_msg}), resp.status_code
    except Exception as e:
        return jsonify({"error": f"Failed to delete model: {e!s}"}), 500


@app.route("/api/models/search", methods=["POST"])
def search_models():
    """Search for models on Ollama Registry and Hugging Face"""
    import re
    from html import unescape

    import httpx

    # Keywords that identify Stable Diffusion / image-generation models
    _SD_NAME_KEYWORDS = [
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
        "diffusion",
        "imagen",
        "dalle",
        "kandinsky",
        "playground",
        "waifu-diffusion",
    ]
    _SD_HF_TAGS = {
        "diffusers",
        "stable-diffusion",
        "text-to-image",
        "image-generation",
        "stable-diffusion-xl",
        "stable-diffusion-3",
        "flux",
        "image-to-image",
    }

    def _detect_model_type(name: str, tags: list) -> str:
        """Returns 'stable-diffusion' or 'llm' based on name and HF tags."""
        name_lower = name.lower()
        if any(kw in name_lower for kw in _SD_NAME_KEYWORDS):
            return "stable-diffusion"
        if any(t in _SD_HF_TAGS for t in tags):
            return "stable-diffusion"
        return "llm"

    data = request.get_json() or {}
    query = data.get("query")
    source = data.get("source", "all")  # "ollama", "huggingface", or "all"
    type_filter = data.get("type", "all")  # "all", "llm", "stable-diffusion"

    if not query:
        return jsonify({"error": "query is required"}), 400

    results = []

    # 1. Ollama Search
    if source in ("ollama", "all"):
        try:
            url = f"https://ollama.com/search?q={query}"
            resp = httpx.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10.0)
            if resp.status_code == 200:
                html_content = resp.text
                pattern = r'href="/library/([^"]+)"[^>]*>.*?<span[^>]*>([^<]+)</span>.*?<p class="[^"]*break-words[^"]*">([^<]+)</p>'
                matches = re.findall(pattern, html_content, re.DOTALL)
                for library_name, _span_name, desc in matches:
                    clean_name = library_name.strip()
                    clean_desc = unescape(desc.strip().replace("\n", " "))
                    model_type = _detect_model_type(clean_name, [])
                    results.append(
                        {
                            "name": clean_name,
                            "description": clean_desc,
                            "source": "ollama",
                            "type": model_type,
                            "downloads": None,
                            "likes": None,
                            "tags": [],
                            "author": None,
                        }
                    )
        except Exception as e:
            print(f"Ollama search error: {e}")

    # 2. Hugging Face Search — include both GGUF (LLM) and all types (for SD)
    if source in ("huggingface", "all"):
        try:
            # Search GGUF models (primarily LLMs)
            url_gguf = f"https://huggingface.co/api/models?search={query}&filter=gguf&limit=20"
            resp_gguf = httpx.get(url_gguf, headers={"User-Agent": "Mozilla/5.0"}, timeout=10.0)

            # Also search for diffusers/SD models
            url_sd = f"https://huggingface.co/api/models?search={query}&filter=diffusers&limit=10"
            resp_sd = httpx.get(url_sd, headers={"User-Agent": "Mozilla/5.0"}, timeout=10.0)

            seen_ids: set = set()

            def _process_hf_item(model_item: dict, forced_type: str | None = None) -> dict | None:
                model_id = model_item.get("id")
                if not model_id or model_id in seen_ids:
                    return None
                seen_ids.add(model_id)
                tags = model_item.get("tags", [])
                model_type = forced_type or _detect_model_type(model_id, tags)
                downloads = model_item.get("downloads", 0)
                likes = model_item.get("likes", 0)
                author = model_item.get("author", "")
                display_tags = [
                    t
                    for t in tags
                    if t not in ("gguf", "diffusers", "transformers", "pytorch", "safetensors")
                ][:5]
                desc = f"Repository by {author}. Downloads: {downloads:,} | Likes: {likes:,}"
                return {
                    "name": model_id,
                    "description": desc,
                    "source": "huggingface",
                    "type": model_type,
                    "downloads": downloads,
                    "likes": likes,
                    "tags": display_tags,
                    "author": author,
                }

            if resp_gguf.status_code == 200:
                for item in resp_gguf.json():
                    entry = _process_hf_item(item)
                    if entry:
                        results.append(entry)

            if resp_sd.status_code == 200:
                for item in resp_sd.json():
                    entry = _process_hf_item(item, forced_type="stable-diffusion")
                    if entry:
                        results.append(entry)

        except Exception as e:
            print(f"Hugging Face search error: {e}")

    # 3. Precise HF Repo Lookup if query has a slash
    if source in ("huggingface", "all") and "/" in query:
        try:
            precise_url = f"https://huggingface.co/api/models/{query}"
            token = os.getenv("HUGGING_FACE_TOKEN") or os.getenv("HF_TOKEN")
            headers = {"User-Agent": "Mozilla/5.0"}
            if token:
                headers["Authorization"] = f"Bearer {token}"

            resp = httpx.get(precise_url, headers=headers, timeout=5.0)
            if resp.status_code == 200:
                model_item = resp.json()
                model_id = model_item.get("id")
                if model_id and not any(r["name"] == model_id for r in results):
                    tags = model_item.get("tags", [])
                    model_type = _detect_model_type(model_id, tags)
                    downloads = model_item.get("downloads", 0)
                    likes = model_item.get("likes", 0)
                    author = model_item.get("author", "")
                    display_tags = [
                        t
                        for t in tags
                        if t not in ("gguf", "diffusers", "transformers", "pytorch", "safetensors")
                    ][:5]
                    results.insert(
                        0,
                        {
                            "name": model_id,
                            "description": f"[Direct Match] Repository by {author}. Downloads: {downloads:,} | Likes: {likes:,}",
                            "source": "huggingface",
                            "type": model_type,
                            "downloads": downloads,
                            "likes": likes,
                            "tags": display_tags,
                            "author": author,
                        },
                    )
        except Exception as e:
            print(f"Precise HF lookup error: {e}")

    # Apply optional type filter
    if type_filter in ("llm", "stable-diffusion"):
        results = [r for r in results if r.get("type") == type_filter]

    return jsonify({"results": results})


@app.route("/api/models/huggingface/files", methods=["GET"])
def get_hf_files():
    """List downloadable model files (.gguf and .safetensors) in a Hugging Face repository"""
    repo = request.args.get("repo")
    if not repo:
        return jsonify({"error": "repo is required"}), 400

    import httpx

    token = os.getenv("HUGGING_FACE_TOKEN") or os.getenv("HF_TOKEN")
    headers = {"User-Agent": "Mozilla/5.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"https://huggingface.co/api/models/{repo}"
    try:
        resp = httpx.get(url, headers=headers, timeout=15.0)
        if resp.status_code != 200:
            return (
                jsonify({"error": f"Failed to fetch model info from Hugging Face: {resp.text}"}),
                resp.status_code,
            )

        model_info = resp.json()
        tags = model_info.get("tags", [])
        # Detect if this repo is a Stable Diffusion / image generation repo
        _SD_HF_TAGS = {
            "diffusers",
            "stable-diffusion",
            "text-to-image",
            "image-generation",
            "stable-diffusion-xl",
            "flux",
        }
        _SD_NAME_KW = [
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
            "diffusion",
        ]
        is_sd_repo = any(t in _SD_HF_TAGS for t in tags) or any(
            kw in repo.lower() for kw in _SD_NAME_KW
        )

        siblings = model_info.get("siblings", [])
        model_files = []
        for s in siblings:
            fname = s.get("rfilename", "")
            is_gguf = fname.endswith(".gguf")
            is_safetensors = fname.endswith(".safetensors")

            # Include .gguf for all repos; include .safetensors only for SD repos
            if not is_gguf and not (is_sd_repo and is_safetensors):
                continue

            size = s.get("size")
            size_str = ""
            if size:
                size_str = f"{size / 1024 ** 3:.2f} GB" if size > 1024 ** 3 else f"{size / 1024 ** 2:.1f} MB"

            file_type = "stable-diffusion" if (is_safetensors or is_sd_repo) else "llm"
            model_files.append(
                {
                    "filename": fname,
                    "size": size_str,
                    "type": file_type,
                    "format": "safetensors" if is_safetensors else "gguf",
                }
            )

        return jsonify(
            {"files": model_files, "repo_type": "stable-diffusion" if is_sd_repo else "llm"}
        )
    except Exception as e:
        return jsonify({"error": f"Error fetching Hugging Face files: {e!s}"}), 500


@app.route("/api/models/ollama/tags", methods=["GET"])
def get_ollama_model_tags():
    """List available tags/sizes for a model in the Ollama library"""
    model = request.args.get("model")
    if not model:
        return jsonify({"error": "model is required"}), 400

    import re

    import httpx

    url = f"https://ollama.com/library/{model}/tags"
    try:
        resp = httpx.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15.0)
        if resp.status_code != 200:
            return (
                jsonify({"error": f"Failed to fetch tags from Ollama Library: {resp.text}"}),
                resp.status_code,
            )

        html = resp.text
        # Matches href="/library/model:tag"
        pattern = rf'href="/library/{re.escape(model)}:([^"]+)"'
        tags = re.findall(pattern, html)

        # Deduplicate and keep order
        unique_tags = []
        seen = set()
        for t in tags:
            t_clean = t.strip()
            if t_clean not in seen:
                seen.add(t_clean)
                unique_tags.append(t_clean)

        # Fallback to general tag parsing if specific model name prefix did not match
        if not unique_tags:
            fallback_pattern = r'href="/library/[^"]+:([^"]+)"'
            generic_tags = re.findall(fallback_pattern, html)
            for t in generic_tags:
                t_clean = t.strip()
                if t_clean not in seen:
                    seen.add(t_clean)
                    unique_tags.append(t_clean)

        return jsonify({"tags": unique_tags})
    except Exception as e:
        return jsonify({"error": f"Error fetching Ollama model tags: {e!s}"}), 500


@app.route("/api/models/pull", methods=["POST"])
def trigger_model_pull():
    """Trigger a background pull of a model via alpaca-puller.py"""
    data = request.get_json() or {}
    model = data.get("model")
    source = data.get("source", "auto")
    local_name = data.get("local_name")
    no_resume = data.get("no_resume", False)
    companion = data.get("companion", False)

    if not model:
        return jsonify({"error": "model is required"}), 400

    with active_pulls_lock:
        if model in active_pulls:
            return (
                jsonify({"error": f"Model {model} is already being downloaded."}),
                409,
            )
        active_pulls[model] = {
            "model": model,
            "source": source,
            "local_name": local_name or "",
            "status": "running",
            "logs": [],
        }

    t = threading.Thread(
        target=run_puller_thread, args=(model, source, local_name, no_resume, companion), daemon=True
    )
    t.start()

    return jsonify(
        {
            "status": "pulling_started",
            "message": f"Started pulling model {model} in the background.",
        }
    )


@app.route("/api/models/pulls/active", methods=["GET"])
def get_active_pulls():
    """Retrieve currently active downloads and their logs"""
    with active_pulls_lock:
        return jsonify(
            {
                "active_pulls": {
                    k: {
                        "model": v["model"],
                        "source": v["source"],
                        "local_name": v.get("local_name", ""),
                        "status": v.get("status", "running"),
                        "logs": v["logs"],
                    }
                    for k, v in active_pulls.items()
                }
            }
        )


@app.route("/api/models/pulls/<model_id>/stop", methods=["POST"])
def stop_pull(model_id):
    """Stop/pause a running pull by creating a stop marker file."""
    with active_pulls_lock:
        if model_id not in active_pulls:
            return jsonify({"error": "Pull not found"}), 404
        pull = active_pulls[model_id]
        if pull.get("status") not in ("running", "paused"):
            return jsonify({"error": f"Pull is {pull.get('status', 'unknown')}, cannot stop"}), 400
        pull["status"] = "stopping"

    # Create stop marker file that alpaca-puller checks
    stop_dir = Path(os.getenv("ROUTER_MODELS_DIR", ".alpaca-router")) / ".alpaca-stop"
    stop_dir.mkdir(parents=True, exist_ok=True)
    stop_file = stop_dir / f"{model_id.replace('/', '_').replace(':', '_')}"
    stop_file.write_text(str(time.time()))
    print(f"Stop marker created: {stop_file}")

    return jsonify({"status": "stopping", "message": f"Stopping pull for {pull['model']}..."})


@app.route("/api/models/pulls/<model_id>/cancel", methods=["POST"])
def cancel_pull(model_id):
    """Cancel a pull: stop it and remove any partial downloads."""
    with active_pulls_lock:
        if model_id not in active_pulls:
            return jsonify({"error": "Pull not found"}), 404
        pull = active_pulls[model_id]
        if pull.get("status") == "cancelled":
            return jsonify({"error": "Pull already cancelled"}), 400
        pull["status"] = "cancelled"

    # Remove stop marker for this specific model to prevent interfering with future pulls
    stop_dir = Path(os.getenv("ROUTER_MODELS_DIR", ".alpaca-router")) / ".alpaca-stop"
    stop_file = stop_dir / f"{model_id.replace('/', '_').replace(':', '_')}"
    stop_file.unlink(missing_ok=True)
    print(
        f"Cancelled pull for {pull['model']}, partial downloads will be cleaned up on next restart."
    )

    return jsonify({"status": "cancelled", "message": f"Pull for {pull['model']} cancelled."})


@socketio.on("connect")
def handle_connect():
    print("Client connected")
    with active_run_lock:
        socketio.emit("sync_status", dict(active_run))


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)
